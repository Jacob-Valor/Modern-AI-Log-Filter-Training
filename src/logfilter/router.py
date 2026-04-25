"""Kafka-to-QRadar router entrypoint.

Consumes raw events, calls the scoring API, forwards LEEF to QRadar, and
publishes scored JSON to Kafka.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from kafka import KafkaConsumer, KafkaProducer

from logfilter.config import load_config
from logfilter.pipeline.router import SyslogSender

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RouterSettings:
    bootstrap_servers: str | list[str]
    raw_topic: str
    scored_topic: str
    consumer_group: str
    max_poll_records: int
    poll_timeout_ms: int
    api_url: str
    qradar_host: str
    qradar_port: int
    qradar_protocol: str


def _settings() -> RouterSettings:
    config = load_config()
    kafka_cfg = config.get("kafka", {})
    topics = kafka_cfg.get("topics", {})
    qradar_cfg = config.get("qradar", {})
    return RouterSettings(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        raw_topic=topics.get("raw_logs", "raw-logs"),
        scored_topic=topics.get("scored_logs", "scored-logs"),
        consumer_group=os.environ.get(
            "ROUTER_CONSUMER_GROUP",
            kafka_cfg.get("consumer_group", "logfilter-scorer"),
        ),
        max_poll_records=int(kafka_cfg.get("max_poll_records", 100)),
        poll_timeout_ms=int(os.environ.get("ROUTER_POLL_TIMEOUT_MS", "1000")),
        api_url=os.environ.get("LOGFILTER_API_URL", "http://logfilter-api:8080").rstrip("/"),
        qradar_host=qradar_cfg.get("syslog_host", "localhost"),
        qradar_port=int(qradar_cfg.get("syslog_port", 514)),
        qradar_protocol=qradar_cfg.get("syslog_protocol", "tcp"),
    )


class KafkaQRadarRouter:
    """At-least-once raw-log consumer and LEEF forwarder."""

    def __init__(self, settings: RouterSettings) -> None:
        self.settings = settings
        self.running = True
        self.sender = SyslogSender(
            host=settings.qradar_host,
            port=settings.qradar_port,
            protocol=settings.qradar_protocol,
        )
        self.http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self.consumer = KafkaConsumer(
            settings.raw_topic,
            bootstrap_servers=settings.bootstrap_servers,
            group_id=settings.consumer_group,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=settings.max_poll_records,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=settings.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=5,
            max_in_flight_requests_per_connection=1,
        )

    def _score_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        request = {
            "events": [
                {
                    "raw": str(event.get("raw", "")),
                    "source_type": event.get("source_type"),
                }
                for event in events
            ]
        }
        response = self.http.post(f"{self.settings.api_url}/score/batch", json=request)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if len(results) != len(events):
            raise RuntimeError(
                f"Scoring API returned {len(results)} results for {len(events)} events"
            )
        return results

    def _route_scored(self, scored_events: list[dict[str, Any]]) -> None:
        for scored in scored_events:
            leef = scored.get("leef_payload", "")
            priority = scored.get("ai_priority", "INFO")
            if not leef:
                raise RuntimeError("Scoring API response missing leef_payload")
            self.sender.send(leef, priority)
            self.producer.send(
                self.settings.scored_topic,
                value=scored,
                key=scored.get("host", "unknown"),
            )
        self.producer.flush(timeout=30)

    def run(self) -> None:
        logger.info(
            "Kafka QRadar router started",
            raw_topic=self.settings.raw_topic,
            scored_topic=self.settings.scored_topic,
            api_url=self.settings.api_url,
        )
        try:
            while self.running:
                records = self.consumer.poll(timeout_ms=self.settings.poll_timeout_ms)
                batch: list[dict[str, Any]] = []
                for messages in records.values():
                    for msg in messages:
                        if isinstance(msg.value, dict):
                            batch.append(msg.value)
                        else:
                            batch.append({"raw": str(msg.value), "source_type": "generic"})

                if not batch:
                    continue

                try:
                    scored = self._score_batch(batch)
                    self._route_scored(scored)
                    self.consumer.commit()
                    logger.info("Batch routed", count=len(scored))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Batch routing failed; offsets not committed", error=str(exc))
        finally:
            self.consumer.close()
            self.producer.close()
            self.sender.close()
            self.http.close()
            logger.info("Kafka QRadar router stopped")


def main() -> None:
    router = KafkaQRadarRouter(_settings())

    def _shutdown(signum, frame):  # noqa: ANN001
        router.running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    router.run()


if __name__ == "__main__":
    main()
