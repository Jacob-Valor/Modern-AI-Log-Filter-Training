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

from logfilter import telemetry
from logfilter.config import load_config
from logfilter.kafka.config import kafka_security_kwargs
from logfilter.kafka.producer import LogProducer
from logfilter.pipeline.router import SyslogSender
from logfilter.utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

_API_BREAKER = CircuitBreaker(
    name="scoring_api",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
    expected_exception=(httpx.HTTPError, RuntimeError),
)


@dataclass(frozen=True)
class RouterSettings:
    bootstrap_servers: str | list[str]
    raw_topic: str
    scored_topic: str
    dlq_topic: str | None
    consumer_group: str
    auto_offset_reset: str
    max_poll_records: int
    poll_timeout_ms: int
    api_url: str
    api_token: str
    qradar_host: str
    qradar_port: int
    qradar_protocol: str
    kafka_config: dict[str, Any]


def _settings() -> RouterSettings:
    config = load_config()
    kafka_cfg = config.get("kafka", {})
    topics = kafka_cfg.get("topics", {})
    qradar_cfg = config.get("qradar", {})
    api_cfg = config.get("api", {})
    api_token = os.environ.get("LOGFILTER_API_TOKEN") or api_cfg.get("scoring_token", "")
    if not api_token:
        raise SystemExit("LOGFILTER_API_TOKEN must be set for router-to-API scoring calls")
    return RouterSettings(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        raw_topic=topics.get("raw_logs", "raw-logs"),
        scored_topic=topics.get("scored_logs", "scored-logs"),
        dlq_topic=topics.get("dlq"),
        consumer_group=os.environ.get(
            "ROUTER_CONSUMER_GROUP",
            kafka_cfg.get("router_consumer_group", "logfilter-router"),
        ),
        auto_offset_reset=kafka_cfg.get("auto_offset_reset", "earliest"),
        max_poll_records=int(kafka_cfg.get("max_poll_records", 100)),
        poll_timeout_ms=int(os.environ.get("ROUTER_POLL_TIMEOUT_MS", "1000")),
        api_url=os.environ.get("LOGFILTER_API_URL", "http://logfilter-api:8080").rstrip("/"),
        api_token=api_token,
        qradar_host=qradar_cfg.get("syslog_host", "localhost"),
        qradar_port=int(qradar_cfg.get("syslog_port", 514)),
        qradar_protocol=qradar_cfg.get("syslog_protocol", "tcp"),
        kafka_config=kafka_cfg,
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
            auto_offset_reset=settings.auto_offset_reset,
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=settings.max_poll_records,
            **kafka_security_kwargs(settings.kafka_config),
        )
        self.producer = KafkaProducer(
            bootstrap_servers=settings.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=5,
            max_in_flight_requests_per_connection=1,
            **kafka_security_kwargs(settings.kafka_config),
        )
        self.dlq_producer: LogProducer | None = None
        if settings.dlq_topic:
            self.dlq_producer = LogProducer(
                bootstrap_servers=settings.bootstrap_servers,
                topic=settings.dlq_topic,
                dlq_topic=settings.dlq_topic,
                kafka_config=settings.kafka_config,
            )
        self._current_batch_context = None

    def _score_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with telemetry.start_as_current_span(
            "router.api.score_batch",
            {
                "http.method": "POST",
                "http.url": f"{self.settings.api_url}/score/batch",
                "logfilter.batch_size": len(events),
            },
            trace_context=self._current_batch_context,
        ) as span:
            token = telemetry.attach_span_context(span, self._current_batch_context)
            try:
                request = {
                    "events": [
                        {
                            "raw": str(event.get("raw", "")),
                            "source_type": event.get("source_type"),
                        }
                        for event in events
                    ]
                }

                def _call_api():
                    headers = telemetry.inject_http_headers(
                        {"X-API-Token": self.settings.api_token}
                    )
                    response = self.http.post(
                        f"{self.settings.api_url}/score/batch",
                        json=request,
                        headers=headers,
                    )
                    response.raise_for_status()
                    status_code = getattr(response, "status_code", None)
                    if status_code is not None:
                        span.set_attribute("http.status_code", status_code)
                    return response.json()

                payload = _API_BREAKER.call(_call_api)
                results = payload.get("results", [])
                if len(results) != len(events):
                    raise RuntimeError(
                        f"Scoring API returned {len(results)} results for {len(events)} events"
                    )
                return results
            except Exception as exc:
                telemetry.record_exception(span, exc)
                raise
            finally:
                telemetry.detach_context(token)

    def _route_scored(self, scored_events: list[dict[str, Any]]) -> None:
        with telemetry.start_as_current_span(
            "router.route_scored",
            {
                "messaging.destination.name": self.settings.scored_topic,
                "logfilter.batch_size": len(scored_events),
            },
            trace_context=self._current_batch_context,
        ) as span:
            token = telemetry.attach_span_context(span, self._current_batch_context)
            try:
                for scored in scored_events:
                    leef = scored.get("leef_payload", "")
                    priority = scored.get("ai_priority", "INFO")
                    if not leef:
                        raise RuntimeError("Scoring API response missing leef_payload")
                    with telemetry.start_as_current_span(
                        "router.qradar.send",
                        {
                            "logfilter.priority": priority,
                            "server.address": self.settings.qradar_host,
                            "server.port": self.settings.qradar_port,
                            "network.transport": self.settings.qradar_protocol,
                        },
                    ):
                        self.sender.send(leef, priority)
                    self.producer.send(
                        self.settings.scored_topic,
                        value=scored,
                        key=scored.get("host", "unknown"),
                        headers=telemetry.inject_kafka_headers(span=span),
                    )
                self.producer.flush(timeout=30)
            except Exception as exc:
                telemetry.record_exception(span, exc)
                raise
            finally:
                telemetry.detach_context(token)

    def run(self) -> None:  # pragma: no cover
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
                self._current_batch_context = None
                for messages in records.values():
                    for msg in messages:
                        if self._current_batch_context is None:
                            self._current_batch_context = telemetry.extract_kafka_context(
                                getattr(msg, "headers", None)
                            )
                        if isinstance(msg.value, dict):
                            batch.append(msg.value)
                        else:
                            batch.append({"raw": str(msg.value), "source_type": "generic"})

                if not batch:
                    continue

                with telemetry.start_as_current_span(
                    "router.kafka.batch",
                    {
                        "messaging.system": "kafka",
                        "messaging.destination.name": self.settings.raw_topic,
                        "logfilter.batch_size": len(batch),
                    },
                    trace_context=self._current_batch_context,
                ) as span:
                    token = telemetry.attach_span_context(span, self._current_batch_context)
                    try:
                        scored = self._score_batch(batch)
                        self._route_scored(scored)
                        self.consumer.commit()
                        span.set_attribute("logfilter.routed_count", len(scored))
                        logger.info("Batch routed", count=len(scored))
                    except Exception as exc:  # noqa: BLE001
                        telemetry.record_exception(span, exc)
                        logger.error("Batch routing failed; sending to DLQ", error=str(exc))
                        if self.dlq_producer:
                            for event in batch:
                                retry_count = event.get("_dlq_retry_count", 0) + 1
                                self.dlq_producer.send_dlq(
                                    original_message=event,
                                    error=str(exc),
                                    original_topic=self.settings.raw_topic,
                                    retry_count=retry_count,
                                )
                            self.consumer.commit()
                        else:
                            logger.error(
                                "Batch routing failed; offsets not committed", error=str(exc)
                            )
                    finally:
                        telemetry.detach_context(token)
        finally:
            self.consumer.close()
            self.producer.close()
            if self.dlq_producer:
                self.dlq_producer.close()
            self.sender.close()
            self.http.close()
            logger.info("Kafka QRadar router stopped")


class DLQReplay:
    """Replay dead-lettered messages back to their original topic.

    Consumes from the DLQ topic, extracts the original payload, and
    re-publishes it to ``original_topic`` with a replay counter so the
    router can apply a replay-specific consumer group.
    """

    def __init__(self, settings: RouterSettings) -> None:
        self.settings = settings
        self.producer = KafkaProducer(
            bootstrap_servers=settings.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=5,
            max_in_flight_requests_per_connection=1,
            **kafka_security_kwargs(settings.kafka_config),
        )
        self.consumer = KafkaConsumer(
            settings.dlq_topic,
            bootstrap_servers=settings.bootstrap_servers,
            group_id=f"{settings.consumer_group}-dlq-replay",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            max_poll_records=settings.max_poll_records,
            **kafka_security_kwargs(settings.kafka_config),
        )
        self.running = True

    def _replay(self, dlq_record: dict[str, Any]) -> dict[str, Any] | None:
        original = dlq_record.get("original")
        if not isinstance(original, dict):
            return None
        original_topic = dlq_record.get("original_topic", self.settings.raw_topic)
        replay_count = original.get("_dlq_replay_count", 0) + 1
        if replay_count > 3:
            return None
        original["_dlq_replay_count"] = replay_count
        return {"original": original, "original_topic": original_topic}

    def _publish(self, replay: dict[str, Any]) -> None:
        self.producer.send(
            replay["original_topic"],
            value=replay["original"],
            key=replay["original"].get("host", "unknown"),
        )
        self.producer.flush(timeout=10)

    def run(
        self, max_messages: int | None = None, max_empty_polls: int = 3
    ) -> None:  # pragma: no cover
        logger.info("DLQ replay started", dlq_topic=self.settings.dlq_topic)
        processed = 0
        empty_polls = 0
        try:
            while self.running:
                records = self.consumer.poll(timeout_ms=self.settings.poll_timeout_ms)
                if not records:
                    empty_polls += 1
                    if empty_polls >= max_empty_polls:
                        break
                    continue
                empty_polls = 0
                for messages in records.values():
                    for msg in messages:
                        if max_messages is not None and processed >= max_messages:
                            self.running = False
                            break
                        replay = self._replay(msg.value)
                        if replay is None:
                            continue
                        self._publish(replay)
                        processed += 1
                    if not self.running:
                        break
                if max_messages is not None and processed >= max_messages:
                    self.running = False
                self.consumer.commit()
                if not self.running:
                    break
        finally:
            self.consumer.close()
            self.producer.close()
            logger.info("DLQ replay stopped")


def main() -> None:  # pragma: no cover
    router = KafkaQRadarRouter(_settings())

    def _shutdown(signum, frame):  # noqa: ANN001
        router.running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    router.run()


if __name__ == "__main__":  # pragma: no cover
    main()
