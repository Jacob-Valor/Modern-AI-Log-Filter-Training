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

import json
import signal
import time
from collections.abc import Callable
from typing import Any

import structlog
from kafka import KafkaConsumer

from logfilter import telemetry
from logfilter.utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

_DEFAULT_ES_BREAKER = CircuitBreaker(
    name="elasticsearch_bulk",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
    expected_exception=Exception,
)


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
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.es = es_client
        self.index_prefix = index_prefix
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._running = False
        self._es_breaker = es_breaker or _DEFAULT_ES_BREAKER

    def _index_name(self) -> str:
        date_str = time.strftime("%Y.%m.%d")
        return f"{self.index_prefix}-{date_str}"

    def run(self) -> None:
        """Blocking run loop. Call in a dedicated thread or process."""
        consumer = KafkaConsumer(
            self.raw_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="logfilter-archiver",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=self.batch_size,
        )

        self._running = True
        archived_total = 0
        logger.info("Archive consumer started", topic=self.raw_topic)

        def _shutdown(signum, frame):  # noqa: ANN001
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        try:
            while self._running:
                records = consumer.poll(timeout_ms=self.poll_timeout_ms)
                bulk_body: list[dict] = []

                for tp, messages in records.items():
                    for msg in messages:
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
                                payload = msg.value
                                bulk_body.append({"index": {"_index": self._index_name()}})
                                bulk_body.append(
                                    {
                                        "raw": payload.get("raw", ""),
                                        "source_type": payload.get("source_type", "generic"),
                                        "host": payload.get("host", "unknown"),
                                        "ingest_ts": payload.get("ingest_ts", time.time()),
                                        "kafka_offset": msg.offset,
                                        "kafka_partition": msg.partition,
                                    }
                                )
                            finally:
                                telemetry.detach_context(token)

                if bulk_body:
                    with telemetry.start_as_current_span(
                        "elasticsearch.archive.bulk",
                        {
                            "db.system": "elasticsearch",
                            "logfilter.batch_size": len(bulk_body) // 2,
                            "logfilter.index_prefix": self.index_prefix,
                        },
                    ) as span:
                        try:
                            result = self._es_breaker.call(self.es.bulk, body=bulk_body)
                            if result.get("errors"):
                                raise RuntimeError("Elasticsearch bulk response contained errors")
                            n = len(bulk_body) // 2
                            archived_total += n
                            consumer.commit()
                            span.set_attribute("logfilter.archived_total", archived_total)
                            logger.debug("Archived to ES", n=n, total=archived_total)
                        except Exception as exc:  # noqa: BLE001
                            telemetry.record_exception(span, exc)
                            logger.error(
                                "ES bulk write failed; offsets not committed", error=str(exc)
                            )

        finally:
            consumer.close()
            logger.info("Archive consumer stopped", archived_total=archived_total)


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
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.scored_topic = scored_topic
        self.score_fn = score_fn
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._producer = producer
        self._running = False

    def run(self) -> None:
        """Blocking run loop."""
        consumer = KafkaConsumer(
            self.raw_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="logfilter-scorer",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=self.batch_size,
        )

        self._running = True
        scored_total = 0
        logger.info("Scorer consumer started", topic=self.raw_topic)

        def _shutdown(signum, frame):  # noqa: ANN001
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        try:
            while self._running:
                records = consumer.poll(timeout_ms=self.poll_timeout_ms)
                batch: list[dict[str, Any]] = []

                batch_context = None
                for tp, messages in records.items():
                    for msg in messages:
                        if batch_context is None:
                            batch_context = telemetry.extract_kafka_context(
                                getattr(msg, "headers", None)
                            )
                        batch.append(msg.value)

                if not batch:
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
                        logger.error(
                            "Scoring batch failed; offsets not committed",
                            error=str(exc),
                            batch_size=len(batch),
                        )
                    finally:
                        telemetry.detach_context(token)

        finally:
            consumer.close()
            logger.info("Scorer consumer stopped", scored_total=scored_total)
