"""
Kafka consumers for the AI log filter pipeline.

Two consumers share the same raw-logs topic (different consumer groups):

  ArchiveConsumer   — writes raw events to Elasticsearch immediately.
                      Consumer group: logfilter-archiver
                      Must commit BEFORE the scorer processes the event
                      (archive-first pattern ensures zero raw log loss).

  ScorerConsumer    — reads raw events, normalises, scores, enriches,
                      and forwards enriched LEEF events to QRadar + scored-logs topic.
                      Consumer group: logfilter-scorer
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from logfilter import telemetry
from logfilter.kafka.config import (
    decode_kafka_value,
    is_transient_failure,
    kafka_consumer_health_kwargs,
    kafka_security_kwargs,
)
from logfilter.kafka.producer import DlqNotConfiguredError, LogProducer
from logfilter.pipeline.archive import compute_kafka_raw_log_ref
from logfilter.security.redaction import RedactionConfig, redact
from logfilter.utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)


def _new_es_breaker() -> CircuitBreaker:
    """A fresh ES bulk breaker. Per-instance so trip state never leaks sideways."""
    return CircuitBreaker(
        name="elasticsearch_bulk",
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3,
        expected_exception=Exception,
    )

_TRANSIENT_BACKOFF_SECONDS = 1.0


class UndecodableRecordError(ValueError):
    """A message could not be decoded into an event payload."""


def _backoff(stop_event: Any) -> None:
    """Briefly pause so a transient outage does not turn into a hot loop."""
    stop_event.wait(_TRANSIENT_BACKOFF_SECONDS)


def _bulk_item_failures(result: Any, expected: int) -> list[int] | None:
    """Indices of documents the ES bulk response reports as failed.

    Returns ``None`` when the response is missing, short, or otherwise cannot
    be mapped back onto the submitted documents — the caller must then treat
    the batch as unattributable rather than guess which records to discard.
    """
    if not isinstance(result, Mapping):
        return None
    if not result.get("errors"):
        return []
    items = result.get("items")
    if not isinstance(items, list) or len(items) != expected:
        return None
    failures: list[int] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or not item:
            return None
        action = next(iter(item.values()))
        if not isinstance(action, Mapping):
            return None
        status = action.get("status")
        if not isinstance(status, int):
            return None
        if not 200 <= status < 300:
            failures.append(index)
    return failures


def _document_id(payload: Mapping[str, Any], topic: str, partition: int, offset: int) -> str:
    """Reuse a replayed ``raw_log_ref``; mint one only when it is absent."""
    existing = payload.get("raw_log_ref")
    if isinstance(existing, str) and existing:
        return existing
    return compute_kafka_raw_log_ref(topic, partition, offset)


class ArchiveConsumer:
    """
    Reads raw-logs topic and writes each event to Elasticsearch.

    This consumer runs INDEPENDENTLY from the scoring consumer.
    It should be deployed as a separate process/container.
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        raw_topic: str,
        es_client: Any,  # Elasticsearch client instance
        index_prefix: str = "raw-logs",
        batch_size: int = 100,
        poll_timeout_ms: int = 1000,
        es_breaker: CircuitBreaker | None = None,
        dlq_producer: LogProducer | None = None,
        kafka_config: dict[str, Any] | None = None,
        redaction_config: RedactionConfig | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.es = es_client
        self.index_prefix = index_prefix
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._running = False
        self._stop_event = threading.Event()
        self._es_breaker = es_breaker or _new_es_breaker()
        self._dlq_producer = dlq_producer
        self._kafka_config = kafka_config or {}
        self._redaction_config = redaction_config or RedactionConfig()

    def stop(self) -> None:
        """Signal the run loop to exit after the current poll cycle."""
        self._stop_event.set()

    def _index_name(self, ingest_ts: float) -> str:
        date_str = time.strftime("%Y.%m.%d", time.gmtime(ingest_ts))
        return f"{self.index_prefix}-{date_str}"

    def _build_bulk_entry(self, msg: Any) -> dict[str, Any]:
        """Map one Kafka message onto its ES action header and source document."""
        payload = msg.value
        raw = str(payload.get("raw", ""))
        ingest_ts = float(payload.get("ingest_ts", time.time()))
        return {
            "action": {
                "index": {
                    "_index": self._index_name(ingest_ts),
                    "_id": _document_id(payload, self.raw_topic, msg.partition, msg.offset),
                }
            },
            "source": {
                "raw": redact(raw, config=self._redaction_config),
                "source_type": payload.get("source_type", "generic"),
                "host": payload.get("host", "unknown"),
                "ingest_ts": ingest_ts,
                "kafka_offset": msg.offset,
                "kafka_partition": msg.partition,
            },
        }

    def _dlq_record(self, msg: Any, error: str) -> None:
        """Send one record to the DLQ; raises so the caller can skip the commit."""
        if not self._dlq_producer:
            raise DlqNotConfiguredError(
                "DLQ producer is not configured; cannot record a permanent failure"
            )
        original = msg.value if isinstance(msg.value, dict) else {}
        # ``_dlq_replay_count`` is the key DLQReplay stamps onto the message body,
        # so the counter keeps climbing across replays.
        retry_count = int(original.get("_dlq_replay_count", 0)) + 1
        self._dlq_producer.send_dlq(
            original_message=original,
            error=error,
            original_topic=self.raw_topic,
            retry_count=retry_count,
        )

    def _poll(self, consumer: Any) -> dict[Any, Any]:
        """Poll, converting a broker error into an empty, uncommitted round."""
        try:
            return consumer.poll(timeout_ms=self.poll_timeout_ms)
        except KafkaError as exc:
            logger.error("Kafka poll failed; no offsets committed", error=str(exc))
            _backoff(self._stop_event)
            return {}

    def run(self) -> None:
        """Blocking run loop. Call in a dedicated thread or process."""
        consumer = KafkaConsumer(
            self.raw_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="logfilter-archiver",
            auto_offset_reset=self._kafka_config.get("auto_offset_reset", "earliest"),
            enable_auto_commit=False,
            value_deserializer=decode_kafka_value,
            max_poll_records=self.batch_size,
            **kafka_consumer_health_kwargs(self._kafka_config),
            **kafka_security_kwargs(self._kafka_config),
        )

        self._stop_event.clear()
        archived_total = 0
        logger.info("Archive consumer started", topic=self.raw_topic)

        try:
            while not self._stop_event.is_set():
                records = self._poll(consumer)
                entries: list[dict[str, Any]] = []
                messages: list[Any] = []
                poison_handled = True

                for tp, polled in records.items():
                    del tp
                    for msg in polled:
                        extracted = telemetry.extract_kafka_context(getattr(msg, "headers", None))
                        with telemetry.start_as_current_span(
                            "kafka.consumer.archive",
                            {
                                "messaging.system": "kafka",
                                "messaging.destination.name": self.raw_topic,
                                "messaging.kafka.partition": msg.partition,
                                "messaging.kafka.offset": msg.offset,
                            },
                            trace_context=extracted,
                        ) as span:
                            token = telemetry.attach_span_context(span, extracted)
                            try:
                                if not isinstance(msg.value, Mapping):
                                    raise UndecodableRecordError(
                                        f"payload is {type(msg.value).__name__}, not an object"
                                    )
                                entries.append(self._build_bulk_entry(msg))
                                messages.append(msg)
                            except UndecodableRecordError as exc:
                                telemetry.record_exception(span, exc)
                                logger.error(
                                    "Undecodable message; routing to DLQ",
                                    error=str(exc),
                                    partition=msg.partition,
                                    offset=msg.offset,
                                )
                                try:
                                    self._dlq_record(msg, str(exc))
                                except Exception as dlq_exc:  # noqa: BLE001
                                    poison_handled = False
                                    logger.error(
                                        "Undecodable message could not be DLQ'd; "
                                        "offsets not committed",
                                        error=str(dlq_exc),
                                    )
                            finally:
                                telemetry.detach_context(token)

                if not records:
                    continue
                if not poison_handled:
                    _backoff(self._stop_event)
                    continue
                if entries:
                    archived_total += self._archive_batch(consumer, entries, messages)
                else:
                    consumer.commit()
        finally:
            consumer.close()
            logger.info("Archive consumer stopped", archived_total=archived_total)

    def _archive_batch(
        self,
        consumer: Any,
        entries: list[dict[str, Any]],
        messages: list[Any],
    ) -> int:
        """Bulk-index one poll, archiving successes and DLQ'ing only the failures.

        Returns the number of documents archived. Offsets are committed only
        when every message in the poll is either archived or safely DLQ'd.
        """
        bulk_body: list[dict] = []
        for entry in entries:
            bulk_body.append(entry["action"])
            bulk_body.append(entry["source"])

        with telemetry.start_as_current_span(
            "elasticsearch.archive.bulk",
            {
                "db.system": "elasticsearch",
                "logfilter.batch_size": len(entries),
                "logfilter.index_prefix": self.index_prefix,
            },
        ) as span:

            def _counts_as_failure(response: Any) -> bool:
                """A rejected document or an unattributable response is a failure."""
                detail = _bulk_item_failures(response, len(entries))
                return detail is None or bool(detail)

            try:
                result = self._es_breaker.call_checked(
                    self.es.bulk,
                    _counts_as_failure,
                    body=bulk_body,
                )
            except Exception as exc:  # noqa: BLE001
                telemetry.record_exception(span, exc)
                if is_transient_failure(exc):
                    logger.error(
                        "ES bulk write failed transiently; offsets not committed",
                        error=str(exc),
                        batch_size=len(entries),
                    )
                    _backoff(self._stop_event)
                    return 0
                logger.error(
                    "ES bulk write failed permanently; sending whole batch to DLQ",
                    error=str(exc),
                    batch_size=len(entries),
                )
                if self._dlq_whole_batch(messages, str(exc)):
                    consumer.commit()
                return 0

            failures = _bulk_item_failures(result, len(entries))
            if failures is None:
                # ``errors: true`` with an unusable item list: the failing records
                # cannot be identified, so the whole batch is retried, not discarded.
                logger.error(
                    "ES bulk response reported errors without per-item detail; "
                    "offsets not committed",
                    batch_size=len(entries),
                    response=result,
                )
                _backoff(self._stop_event)
                return 0

            if failures:
                error = f"Elasticsearch rejected {len(failures)} of {len(entries)} documents"
                for index in failures:
                    try:
                        self._dlq_record(messages[index], error)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "DLQ publish failed; offsets not committed",
                            error=str(exc),
                            partition=messages[index].partition,
                            offset=messages[index].offset,
                        )
                        return 0

            archived = len(entries) - len(failures)
            consumer.commit()
            span.set_attribute("logfilter.archived_total", archived)
            span.set_attribute("logfilter.dlq_count", len(failures))
            logger.debug("Archived to ES", n=archived, dlq=len(failures))
            return archived

    def _dlq_whole_batch(self, messages: list[Any], error: str) -> bool:
        """DLQ every message in the poll; False if any publish failed."""
        for msg in messages:
            try:
                self._dlq_record(msg, error)
            except Exception as exc:  # noqa: BLE001
                logger.error("DLQ publish failed; offsets not committed", error=str(exc))
                return False
        return True



class ScorerConsumer:
    """
    Reads raw-logs topic, scores each event, and routes the enriched LEEF
    to QRadar and publishes the scored result to the scored-logs topic.

    Parameters
    ----------
    score_fn : Callable
        Function (list[dict]) → list[dict] — the full scoring pipeline.
        Typically wraps LogNormalizer + LogScorer + LEEFEnricher + LogRouter.
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        raw_topic: str,
        scored_topic: str,
        score_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        batch_size: int = 100,
        poll_timeout_ms: int = 500,
        producer: Any = None,  # KafkaProducer for scored-logs output
        dlq_producer: LogProducer | None = None,
        kafka_config: dict[str, Any] | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.scored_topic = scored_topic
        self.score_fn = score_fn
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._producer = producer
        self._dlq_producer = dlq_producer
        self._running = False
        self._stop_event = threading.Event()
        self._kafka_config = kafka_config or {}

    def stop(self) -> None:
        """Signal the run loop to exit after the current poll cycle."""
        self._stop_event.set()

    def _dlq_record(self, event: Mapping[str, Any], error: str) -> None:
        """Send one record to the DLQ; raises so the caller can skip the commit."""
        if not self._dlq_producer:
            raise DlqNotConfiguredError(
                "DLQ producer is not configured; cannot record a permanent failure"
            )
        # Same key DLQReplay stamps, so retries accumulate across replays.
        retry_count = int(event.get("_dlq_replay_count", 0)) + 1
        self._dlq_producer.send_dlq(
            original_message=dict(event),
            error=error,
            original_topic=self.raw_topic,
            retry_count=retry_count,
        )

    def _poll(self, consumer: Any) -> dict[Any, Any]:
        """Poll, converting a broker error into an empty, uncommitted round."""
        try:
            return consumer.poll(timeout_ms=self.poll_timeout_ms)
        except KafkaError as exc:
            logger.error("Kafka poll failed; no offsets committed", error=str(exc))
            _backoff(self._stop_event)
            return {}

    def run(self) -> None:
        """Blocking run loop."""
        consumer = KafkaConsumer(
            self.raw_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="logfilter-scorer",
            auto_offset_reset=self._kafka_config.get("auto_offset_reset", "earliest"),
            enable_auto_commit=False,
            value_deserializer=decode_kafka_value,
            max_poll_records=self.batch_size,
            **kafka_consumer_health_kwargs(self._kafka_config),
            **kafka_security_kwargs(self._kafka_config),
        )

        self._stop_event.clear()
        scored_total = 0
        logger.info("Scorer consumer started", topic=self.raw_topic)

        try:
            while not self._stop_event.is_set():
                records = self._poll(consumer)
                batch: list[dict[str, Any]] = []

                batch_context = None
                poison_handled = True
                for tp, messages in records.items():
                    del tp
                    for msg in messages:
                        if batch_context is None:
                            batch_context = telemetry.extract_kafka_context(
                                getattr(msg, "headers", None)
                            )
                        if isinstance(msg.value, Mapping):
                            batch.append(dict(msg.value))
                            continue
                        exc = UndecodableRecordError(
                            f"payload is {type(msg.value).__name__}, not an object"
                        )
                        logger.error(
                            "Undecodable message; routing to DLQ",
                            error=str(exc),
                            partition=msg.partition,
                            offset=msg.offset,
                        )
                        try:
                            self._dlq_record(
                                {
                                    "raw": repr(msg.value),
                                    "raw_log_ref": _document_id(
                                        {}, self.raw_topic, msg.partition, msg.offset
                                    ),
                                },
                                str(exc),
                            )
                        except Exception as dlq_exc:  # noqa: BLE001
                            poison_handled = False
                            logger.error(
                                "Undecodable message could not be DLQ'd; "
                                "offsets not committed",
                                error=str(dlq_exc),
                            )

                if not batch:
                    if records and poison_handled:
                        consumer.commit()
                    elif records:
                        _backoff(self._stop_event)
                    continue

                with telemetry.start_as_current_span(
                    "kafka.consumer.score_batch",
                    {
                        "messaging.system": "kafka",
                        "messaging.destination.name": self.raw_topic,
                        "logfilter.batch_size": len(batch),
                    },
                    trace_context=batch_context,
                ) as span:
                    token = telemetry.attach_span_context(span, batch_context)
                    try:
                        scored_batch = self.score_fn(batch)
                        scored_total += len(scored_batch)

                        if self._producer and scored_batch:
                            for scored_event in scored_batch:
                                self._producer.send(
                                    self.scored_topic,
                                    value=scored_event,
                                    key=scored_event.get("host", "unknown"),
                                    headers=telemetry.inject_kafka_headers(span=span),
                                )
                            if hasattr(self._producer, "flush"):
                                self._producer.flush(timeout=30)

                        consumer.commit()
                        span.set_attribute("logfilter.scored_total", scored_total)
                        logger.debug(
                            "Batch scored and routed",
                            n=len(scored_batch),
                            total=scored_total,
                        )
                    except Exception as exc:  # noqa: BLE001
                        telemetry.record_exception(span, exc)
                        if not poison_handled:
                            logger.error(
                                "Scoring batch failed and poison records are undeliverable; "
                                "offsets not committed",
                                error=str(exc),
                                batch_size=len(batch),
                            )
                        elif is_transient_failure(exc):
                            logger.error(
                                "Scoring batch failed transiently; offsets not committed",
                                error=str(exc),
                                batch_size=len(batch),
                            )
                            _backoff(self._stop_event)
                        else:
                            logger.error(
                                "Scoring batch failed permanently; sending to DLQ",
                                error=str(exc),
                                batch_size=len(batch),
                            )
                            if self._dlq_whole_batch(batch, str(exc)):
                                consumer.commit()
                    finally:
                        telemetry.detach_context(token)

        finally:
            consumer.close()
            logger.info("Scorer consumer stopped", scored_total=scored_total)

    def _dlq_whole_batch(self, batch: list[dict[str, Any]], error: str) -> bool:
        """DLQ every event in the poll; False if any publish failed."""
        for event in batch:
            try:
                self._dlq_record(event, error)
            except Exception as exc:  # noqa: BLE001
                logger.error("DLQ publish failed; offsets not committed", error=str(exc))
                return False
        return True

