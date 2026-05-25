"""
Kafka producer — publishes raw log events to the raw-logs topic.

Designed for the log collector service that receives syslog/WinEvent/
cloud logs and pushes them onto the message bus before any processing.

The archive-first pattern is implemented by having the archive consumer
(Elasticsearch writer) also subscribe to raw-logs BEFORE the AI scorer.
Both consumers are independent; Kafka handles fan-out durably.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from kafka import KafkaProducer
from kafka.errors import KafkaError
from tenacity import retry, stop_after_attempt, wait_exponential

from logfilter import telemetry

logger = structlog.get_logger(__name__)


class LogProducer:
    """
    Wraps KafkaProducer with retry logic and structured logging.

    Parameters
    ----------
    bootstrap_servers : str | list[str]
        Kafka broker(s) e.g. "localhost:9092" or ["broker1:9092", "broker2:9092"]
    topic : str
        Target Kafka topic (typically 'raw-logs')
    batch_size_bytes : int
        Kafka producer batch size in bytes (tune for throughput vs latency)
    linger_ms : int
        How long to wait to fill a batch before sending (0 = low latency)
    compression : str | None
        Compression type: 'gzip', 'snappy', 'lz4', or None
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] = "localhost:9092",
        topic: str = "raw-logs",
        batch_size_bytes: int = 65536,  # 64 KB
        linger_ms: int = 10,
        compression: str | None = "lz4",
    ) -> None:
        self.topic = topic
        self._producer: KafkaProducer | None = None
        self._config = {
            "bootstrap_servers": bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "batch_size": batch_size_bytes,
            "linger_ms": linger_ms,
            "compression_type": compression,
            "acks": "all",  # wait for full ISR acknowledgement
            "retries": 5,
            "max_in_flight_requests_per_connection": 1,
        }

    def _get_producer(self) -> KafkaProducer:
        if self._producer is None:
            logger.info("Connecting to Kafka", servers=self._config["bootstrap_servers"])
            self._producer = KafkaProducer(**self._config)
            logger.info("Kafka producer connected")
        return self._producer

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=5),
        reraise=True,
    )
    def send(
        self,
        raw_log: str,
        source_type: str = "generic",
        host: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Send a single raw log event to Kafka.

        The message value is a JSON envelope preserving the raw payload.
        Partition key = host (for ordered processing per source host).
        """
        with telemetry.start_as_current_span(
            "kafka.producer.send",
            {
                "messaging.system": "kafka",
                "messaging.destination.name": self.topic,
                "logfilter.host": host,
                "logfilter.source_type": source_type,
            },
        ) as span:
            payload = {
                "raw": raw_log,
                "source_type": source_type,
                "host": host,
                "ingest_ts": time.time(),
                **(metadata or {}),
            }
            producer = self._get_producer()
            headers = telemetry.inject_kafka_headers(span=span)
            future = producer.send(
                self.topic,
                value=payload,
                key=host,
                headers=headers,
            )
            try:
                record_metadata = future.get(timeout=10)
                telemetry.set_span_attributes(
                    span,
                    {
                        "messaging.kafka.partition": record_metadata.partition,
                        "messaging.kafka.offset": record_metadata.offset,
                    },
                )
                logger.debug(
                    "Message sent",
                    topic=record_metadata.topic,
                    partition=record_metadata.partition,
                    offset=record_metadata.offset,
                )
            except KafkaError as exc:
                telemetry.record_exception(span, exc)
                logger.error("Failed to send message to Kafka", error=str(exc))
                raise

    def send_batch(
        self,
        events: list[dict[str, Any]],
    ) -> int:
        """
        Fire-and-forget batch send. Returns number of messages sent.

        Each event dict should have keys: raw, source_type, host.
        """
        producer = self._get_producer()
        sent = 0
        with telemetry.start_as_current_span(
            "kafka.producer.send_batch",
            {
                "messaging.system": "kafka",
                "messaging.destination.name": self.topic,
                "logfilter.batch_size": len(events),
            },
        ) as span:
            for event in events:
                host = event.get("host", "unknown")
                source_type = event.get("source_type", "generic")
                payload = {
                    "raw": event.get("raw", ""),
                    "source_type": source_type,
                    "host": host,
                    "ingest_ts": time.time(),
                }
                try:
                    producer.send(
                        self.topic,
                        value=payload,
                        key=host,
                        headers=telemetry.inject_kafka_headers(span=span),
                    )
                    sent += 1
                except KafkaError as exc:
                    telemetry.record_exception(span, exc)
                    logger.error("Batch send error", error=str(exc))

            # Flush buffered messages
            producer.flush(timeout=30)
            span.set_attribute("logfilter.kafka.messages_sent", sent)
            logger.debug("Batch flushed", sent=sent, total=len(events))
            return sent

    def close(self) -> None:
        if self._producer:
            self._producer.flush(timeout=30)
            self._producer.close()
            self._producer = None
            logger.info("Kafka producer closed")
