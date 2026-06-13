"""
Elasticsearch archive client.

Provides a thin wrapper around the Elasticsearch Python client for
writing raw log events (archive-first pattern) and retrieving them
by document ID (forensic chain-of-custody lookup).

Index naming: raw-logs-YYYY.MM.DD (daily rolling index)
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from elasticsearch import Elasticsearch, helpers

from logfilter.security.redaction import RedactionConfig, redact

logger = structlog.get_logger(__name__)

_ALREADY_EXISTS_STATUS = 409


class BulkArchiveError(RuntimeError):
    """Raised when a bulk archive write fails for reasons other than an
    idempotent create-conflict, so callers never treat unpersisted events as
    archived (chain-of-custody integrity)."""


def _bulk_error_status(error: Any) -> int | None:
    if isinstance(error, dict):
        for op_result in error.values():
            if isinstance(op_result, dict) and "status" in op_result:
                return int(op_result["status"])
    return None


class LogArchive:
    """
    Write raw log events to Elasticsearch.

    Parameters
    ----------
    hosts : list[str]
        ES host URLs, e.g. ["http://localhost:9200"]
    index_prefix : str
        Daily index prefix, e.g. "raw-logs"
    username / password : str
        Basic auth credentials
    """

    def __init__(
        self,
        hosts: list[str] | None = None,
        index_prefix: str = "raw-logs",
        username: str = "elastic",
        password: str | None = None,
        shards: int = 1,
        replicas: int = 0,
        redaction_config: RedactionConfig | None = None,
    ) -> None:
        if not password:
            raise ValueError(
                "LogArchive: Elasticsearch password is required. "
                "Set ES_PASSWORD env var or config.elasticsearch.password "
                "(no built-in default)."
            )
        self.index_prefix = index_prefix
        self.shards = shards
        self.replicas = replicas
        self.redaction_config = redaction_config or RedactionConfig()
        client_kwargs: dict[str, Any] = {
            "hosts": hosts or ["http://localhost:9200"],
            "retry_on_timeout": True,
            "max_retries": 3,
        }
        if password:
            client_kwargs["basic_auth"] = (username, password)
        self._es = Elasticsearch(**client_kwargs)
        self._ensure_index_template()

    @property
    def client(self) -> Elasticsearch:
        return self._es

    def close(self) -> None:
        self._es.transport.close()

    def _today_index(self) -> str:
        return f"{self.index_prefix}-{time.strftime('%Y.%m.%d')}"

    def _index_for_ts(self, ingest_ts: float) -> str:
        """
        Resolve the daily index name for a given ingest timestamp (epoch seconds).

        This is the canonical way to derive the index from an envelope timestamp
        (see B8 acceptance condition): we MUST use the event's ingest time, not
        ``time.time()`` at write time, so the index name reflects when the event
        actually happened (matters for late-arriving events crossing midnight UTC).
        """
        return f"{self.index_prefix}-{time.strftime('%Y.%m.%d', time.gmtime(ingest_ts))}"

    def _ensure_index_template(self) -> None:
        """Create an index template so daily indices inherit correct mappings."""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": self.shards,
                    "number_of_replicas": self.replicas,
                    "index.mapping.total_fields.limit": 5000,
                },
                "mappings": {
                    "properties": {
                        "raw": {"type": "text", "store": True},
                        "source_type": {"type": "keyword"},
                        "host": {"type": "keyword"},
                        "ingest_ts": {"type": "date", "format": "epoch_second"},
                        "kafka_offset": {"type": "long"},
                        "kafka_partition": {"type": "integer"},
                    }
                },
            },
        }
        try:
            self._es.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create index template", error=str(exc))

    def write(
        self,
        raw: str,
        source_type: str = "generic",
        host: str = "unknown",
        extra: dict[str, Any] | None = None,
        ingest_ts: float | None = None,
    ) -> str:
        """
        Write a single raw log event and return its Elasticsearch document ID.

        Parameters
        ----------
        raw : str    Original raw log string.
        source_type : str
        host : str
        extra : dict  Additional metadata fields.
        ingest_ts : float | None
            Epoch seconds. When provided, the index name is derived from this
            timestamp (B8); defaults to ``time.time()`` at write time.

        Returns
        -------
        str — The assigned document ID (used as raw_log_ref in LEEF).
        """
        if ingest_ts is None:
            ingest_ts = time.time()
        doc = {
            "raw": self._redact_raw(raw),
            "source_type": source_type,
            "host": host,
            "ingest_ts": ingest_ts,
            **(extra or {}),
        }
        result = self._es.index(index=self._index_for_ts(ingest_ts), body=doc)
        return result["_id"]

    def write_with_id(
        self,
        raw_log_ref: str,
        raw: str,
        source_type: str = "generic",
        host: str = "unknown",
        ingest_ts: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """
        Write a raw log to ES with a caller-supplied document ID (B8).

        This is the chain-of-custody primitive: the caller computes
        ``raw_log_ref`` (sha256 of raw+source+host+ingest_ts) and we attach the
        event with ``op_type=create`` so re-using a ref fails fast. The
        enrichment stage then embeds the SAME ref as ``raw_log_ref`` in LEEF,
        giving SIEM operators a single click to retrieve the original log.

        Raises ``elasticsearch.ConflictError`` if the ref already exists.
        """
        if ingest_ts is None:
            ingest_ts = time.time()
        doc = {
            "raw": self._redact_raw(raw),
            "source_type": source_type,
            "host": host,
            "ingest_ts": ingest_ts,
            **(extra or {}),
        }
        result = self._es.index(
            index=self._index_for_ts(ingest_ts),
            id=raw_log_ref,
            op_type="create",
            body=doc,
        )
        return result["_id"]

    def write_bulk(self, events: list[dict[str, Any]]) -> list[str]:
        """
        Bulk-write events. Returns list of document IDs (same order as input).

        Each event dict: {raw, source_type, host, [optional extra fields]}
        """
        actions = []
        doc_ids: list[str] = []
        for event in events:
            raw = str(event.get("raw", ""))
            source_type = str(event.get("source_type", "generic"))
            host = str(event.get("host", "unknown"))
            ingest_ts = float(event.get("ingest_ts", time.time()))
            doc_id = str(
                event.get("raw_log_ref")
                or compute_raw_log_ref(raw, source_type, host, ingest_ts)
            )
            doc = {
                "_index": self._index_for_ts(ingest_ts),
                "_id": doc_id,
                "_op_type": "create",
                "_source": {
                    "raw": self._redact_raw(raw),
                    "source_type": source_type,
                    "host": host,
                    "ingest_ts": ingest_ts,
                },
            }
            actions.append(doc)
            doc_ids.append(doc_id)

        successes, errors = helpers.bulk(
            self._es,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )
        if errors:
            if isinstance(errors, list):
                failure_count = sum(
                    1 for e in errors if _bulk_error_status(e) != _ALREADY_EXISTS_STATUS
                )
                conflict_count = len(errors) - failure_count
            else:
                failure_count = int(errors)
                conflict_count = 0
            if failure_count:
                logger.error(
                    "ES bulk write had non-conflict failures",
                    failure_count=failure_count,
                )
                raise BulkArchiveError(
                    f"{failure_count} bulk archive operation(s) failed; "
                    "raw logs were not persisted"
                )
            logger.debug(
                "ES bulk create conflicts ignored (already archived)", conflicts=conflict_count
            )

        logger.debug("Bulk archived", count=successes)
        return doc_ids

    def _redact_raw(self, raw: str) -> str:
        return redact(raw, config=self.redaction_config)

    def get_by_id(self, doc_id: str, index: str | None = None) -> dict[str, Any] | None:
        """Retrieve a raw log document by its Elasticsearch ID."""
        idx = index or self._today_index()
        try:
            result = self._es.get(index=idx, id=doc_id)
            return result["_source"]
        except Exception as exc:  # noqa: BLE001
            if index is not None:
                logger.warning("Could not retrieve doc", doc_id=doc_id, error=str(exc))
                return None
            logger.warning(
                "Could not retrieve doc from today's index", doc_id=doc_id, error=str(exc)
            )
        try:
            result = self._es.search(
                index=f"{self.index_prefix}-*",
                body={"query": {"ids": {"values": [doc_id]}}, "size": 1},
            )
            hits = result.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not retrieve doc across rolled indices", doc_id=doc_id, error=str(exc)
            )
            return None
        return None

    def search_recent(
        self,
        host: str | None = None,
        source_type: str | None = None,
        minutes: int = 60,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        """Simple time-windowed search for recent events."""
        must: list[dict] = [{"range": {"ingest_ts": {"gte": f"now-{minutes}m", "lte": "now"}}}]
        if host:
            must.append({"term": {"host": host}})
        if source_type:
            must.append({"term": {"source_type": source_type}})

        result = self._es.search(
            index=f"{self.index_prefix}-*",
            body={"query": {"bool": {"must": must}}, "size": size},
        )
        return [hit["_source"] for hit in result["hits"]["hits"]]

    def health(self) -> dict[str, Any]:
        """Return ES cluster health summary."""
        try:
            health: Any = self._es.cluster.health()
            return dict(health)
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": str(exc)}


def compute_raw_log_ref(
    raw: str,
    source_type: str,
    host: str,
    ingest_ts: float,
) -> str:
    """
    Deterministic chain-of-custody reference for a raw log event.

    Hash inputs include the raw payload, source type, host, and ingest
    timestamp so an identical re-ingest of the same event produces the same
    ref (idempotent) but two different hosts or timestamps produce different
    refs (collision-resistant for the same payload arriving twice).

    Returns 64-char lowercase hex (sha256).
    """
    import hashlib
    import json

    payload = json.dumps(
        {
            "raw": raw,
            "source_type": source_type,
            "host": host,
            "ingest_ts": round(float(ingest_ts), 6),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_kafka_raw_log_ref(topic: str, partition: int, offset: int) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {"topic": topic, "partition": int(partition), "offset": int(offset)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
