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
from typing import Any, Callable

import structlog
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = structlog.get_logger(__name__)


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
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.es = es_client
        self.index_prefix = index_prefix
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self._running = False

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
            enable_auto_commit=True,
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

                if bulk_body:
                    try:
                        self.es.bulk(body=bulk_body)
                        n = len(bulk_body) // 2
                        archived_total += n
                        logger.debug("Archived to ES", n=n, total=archived_total)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("ES bulk write failed", error=str(exc))

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
            enable_auto_commit=True,
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

                for tp, messages in records.items():
                    for msg in messages:
                        batch.append(msg.value)

                if not batch:
                    continue

                try:
                    scored_batch = self.score_fn(batch)
                    scored_total += len(scored_batch)

                    if self._producer and scored_batch:
                        for scored_event in scored_batch:
                            self._producer.send(
                                self.scored_topic,
                                value=scored_event,
                                key=scored_event.get("host", "unknown"),
                            )

                    logger.debug(
                        "Batch scored and routed",
                        n=len(scored_batch),
                        total=scored_total,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Scoring batch failed", error=str(exc), batch_size=len(batch))

        finally:
            consumer.close()
            logger.info("Scorer consumer stopped", scored_total=scored_total)
