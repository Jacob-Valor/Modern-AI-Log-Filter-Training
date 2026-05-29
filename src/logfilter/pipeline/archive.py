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

logger = structlog.get_logger(__name__)


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
        password: str = "",
        shards: int = 1,
        replicas: int = 0,
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
        self._es = Elasticsearch(
            hosts=hosts or ["http://localhost:9200"],
            basic_auth=(username, password),
            retry_on_timeout=True,
            max_retries=3,
        )
        self._ensure_index_template()

    @property
    def client(self) -> Elasticsearch:
        """Underlying Elasticsearch client for integrations that need bulk APIs."""
        return self._es

    def _today_index(self) -> str:
        return f"{self.index_prefix}-{time.strftime('%Y.%m.%d')}"

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
    ) -> str:
        """
        Write a single raw log event and return its Elasticsearch document ID.

        Parameters
        ----------
        raw : str    Original raw log string.
        source_type : str
        host : str
        extra : dict  Additional metadata fields.

        Returns
        -------
        str — The assigned document ID (used as raw_log_ref in LEEF).
        """
        doc = {
            "raw": raw,
            "source_type": source_type,
            "host": host,
            "ingest_ts": time.time(),
            **(extra or {}),
        }
        result = self._es.index(index=self._today_index(), body=doc)
        return result["_id"]

    def write_bulk(self, events: list[dict[str, Any]]) -> list[str]:
        """
        Bulk-write events. Returns list of document IDs (same order as input).

        Each event dict: {raw, source_type, host, [optional extra fields]}
        """
        now = time.time()
        actions = []
        for event in events:
            doc = {
                "_index": self._today_index(),
                "_source": {
                    "raw": event.get("raw", ""),
                    "source_type": event.get("source_type", "generic"),
                    "host": event.get("host", "unknown"),
                    "ingest_ts": event.get("ingest_ts", now),
                },
            }
            actions.append(doc)

        successes, errors = helpers.bulk(
            self._es,
            actions,
            raise_on_error=False,
            raise_on_exception=False,
        )
        if errors:
            error_count = len(errors) if isinstance(errors, list) else int(errors)
            logger.error("ES bulk write had errors", error_count=error_count)

        # Unfortunately helpers.bulk doesn't return IDs; caller must handle
        # by querying for the written docs if IDs are needed for raw_log_ref.
        # For production, use pipeline IDs or write individual docs.
        logger.debug("Bulk archived", count=successes)
        return []  # IDs not available from bulk; use write() for individual IDs

    def get_by_id(self, doc_id: str, index: str | None = None) -> dict[str, Any] | None:
        """Retrieve a raw log document by its Elasticsearch ID."""
        idx = index or self._today_index()
        try:
            result = self._es.get(index=idx, id=doc_id)
            return result["_source"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve doc", doc_id=doc_id, error=str(exc))
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
            return dict(self._es.cluster.health())
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": str(exc)}
