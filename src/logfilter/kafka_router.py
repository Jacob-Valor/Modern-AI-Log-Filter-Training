"""Kafka-to-QRadar router entrypoint.

Consumes raw events, calls the scoring API, forwards LEEF to QRadar, and
publishes scored JSON to Kafka.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from kafka import KafkaConsumer, KafkaProducer

from logfilter import telemetry
from logfilter.config import load_config
from logfilter.kafka.config import (
    decode_kafka_value,
    exhausted_dlq_topic,
    is_transient_failure,
    kafka_consumer_health_kwargs,
    kafka_security_kwargs,
)
from logfilter.kafka.producer import DlqNotConfiguredError, LogProducer
from logfilter.pipeline.archive import compute_kafka_raw_log_ref
from logfilter.pipeline.router import RoutingMode, build_qradar_sender, decide_routing
from logfilter.utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

_API_BREAKER = CircuitBreaker(
    name="scoring_api",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
    expected_exception=(httpx.HTTPError, RuntimeError),
)

_MAX_DLQ_REPLAYS = 3
_TRANSIENT_BACKOFF_SECONDS = 1.0


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
    qradar_mode: str
    qradar_gateway: dict[str, Any]
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
        qradar_gateway=dict(qradar_cfg.get("gateway", {})),
        qradar_mode=qradar_cfg.get("mode", "enrich_only"),
        kafka_config=kafka_cfg,
    )


class RouteBatchError(RuntimeError):
    """Routing failed after every event before ``failed_index`` completed."""

    def __init__(self, failed_index: int, message: str) -> None:
        super().__init__(message)
        self.failed_index = failed_index


def _event_with_archive_ref(message: Any) -> dict[str, Any]:
    """Copy one Kafka payload and attach its replay-stable archive reference.

    A message replayed from the DLQ already carries the reference minted when
    it was first consumed; reusing it keeps the Elasticsearch document id
    stable across replays instead of minting a duplicate for the same event.
    """
    event = (
        dict(message.value)
        if isinstance(message.value, dict)
        else {"raw": str(message.value), "source_type": "generic"}
    )
    event["raw_log_ref"] = event.get("raw_log_ref") or compute_kafka_raw_log_ref(
        str(message.topic),
        int(message.partition),
        int(message.offset),
    )
    return event


def _backoff() -> None:
    """Briefly pause so a transient outage does not turn into a hot poll loop."""
    time.sleep(_TRANSIENT_BACKOFF_SECONDS)


class KafkaQRadarRouter:
    """At-least-once raw-log consumer and LEEF forwarder."""

    def __init__(self, settings: RouterSettings) -> None:
        self.settings = settings
        self.running = True
        self.sender = build_qradar_sender(
            {
                "syslog_host": settings.qradar_host,
                "syslog_port": settings.qradar_port,
                "syslog_protocol": settings.qradar_protocol,
                "gateway": settings.qradar_gateway,
            }
        )
        self.http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self.consumer = KafkaConsumer(
            settings.raw_topic,
            bootstrap_servers=settings.bootstrap_servers,
            group_id=settings.consumer_group,
            auto_offset_reset=settings.auto_offset_reset,
            enable_auto_commit=False,
            value_deserializer=decode_kafka_value,
            max_poll_records=settings.max_poll_records,
            **kafka_consumer_health_kwargs(settings.kafka_config),
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
                            "raw_log_ref": event.get("raw_log_ref"),
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
                mode = RoutingMode(self.settings.qradar_mode)
                for index, scored in enumerate(scored_events):
                    try:
                        leef = scored.get("leef_payload", "")
                        priority = scored.get("ai_priority", "INFO")
                        if not leef:
                            raise RuntimeError("Scoring API response missing leef_payload")
                        decision = decide_routing(
                            mode,
                            priority,
                            float(scored.get("ai_threat_score", 0.0)),
                        )
                        scored["qradar_forwarded"] = decision.forward_to_qradar
                        scored["qradar_routing_reason"] = decision.reason
                        if decision.forward_to_qradar:
                            with telemetry.start_as_current_span(
                                "router.qradar.send",
                                {
                                    "logfilter.priority": priority,
                                    "server.address": self.settings.qradar_host,
                                    "server.port": self.settings.qradar_port,
                                    "network.transport": self.settings.qradar_protocol,
                                },
                            ):
                                self.sender.send(
                                    leef,
                                    priority,
                                    event_id=scored.get("raw_log_ref"),
                                )
                        # Await the broker acknowledgement. This is a raw
                        # KafkaProducer, so send() only queues the record and
                        # returns a future; a failure recorded on that future is
                        # otherwise dropped on the floor by flush(), silently
                        # losing the scored event. Raising here routes the batch
                        # into the DLQ path and leaves offsets uncommitted.
                        future = self.producer.send(
                            self.settings.scored_topic,
                            value=scored,
                            key=scored.get("raw_log_ref") or scored.get("host", "unknown"),
                            headers=telemetry.inject_kafka_headers(span=span),
                        )
                        future.get(timeout=10)
                    except Exception as exc:
                        raise RouteBatchError(index, str(exc)) from exc
                self.producer.flush(timeout=30)
            except Exception as exc:
                telemetry.record_exception(span, exc)
                raise
            finally:
                telemetry.detach_context(token)

    def _dlq_events(self, events: list[dict[str, Any]], error: str) -> bool:
        """Publish records to the DLQ; False if any publish failed.

        Callers must not commit when this returns False: the failed records
        would then have no copy anywhere.
        """
        if not self.dlq_producer:
            logger.error(
                "Batch routing failed and no DLQ producer is configured; "
                "offsets NOT committed — records stay on the topic for reprocessing",
                error=error,
                records=len(events),
            )
            return False
        for event in events:
            # Same key DLQReplay stamps, so retries accumulate across replays.
            retry_count = int(event.get("_dlq_replay_count", 0)) + 1
            try:
                self.dlq_producer.send_dlq(
                    original_message=event,
                    error=error,
                    original_topic=self.settings.raw_topic,
                    retry_count=retry_count,
                )
            except DlqNotConfiguredError as exc:
                logger.error(
                    "DLQ topic is not configured; offsets NOT committed",
                    error=str(exc),
                    records=len(events),
                )
                return False
            except Exception as exc:  # noqa: BLE001
                logger.error("DLQ publish failed; offsets NOT committed", error=str(exc))
                return False
        return True

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
                        batch.append(_event_with_archive_ref(msg))

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
                        if is_transient_failure(exc):
                            # The batch is intact: retrying it costs nothing, while
                            # DLQ'ing it would permanently discard up to
                            # max_poll_records raw logs during an API outage.
                            logger.error(
                                "Batch routing failed transiently; offsets not committed",
                                error=str(exc),
                                batch_size=len(batch),
                            )
                            _backoff()
                            continue
                        failed_index = (
                            exc.failed_index if isinstance(exc, RouteBatchError) else 0
                        )
                        failed_events = batch[failed_index:]
                        logger.error(
                            "Batch routing failed permanently; "
                            "sending unfinished suffix to DLQ",
                            error=str(exc),
                            completed=failed_index,
                            failed=len(failed_events),
                        )
                        if self._dlq_events(failed_events, str(exc)):
                            self.consumer.commit()
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
        if not settings.dlq_topic:
            raise SystemExit("kafka.topics.dlq must be configured to run the DLQ replay")
        self.settings = settings
        self.exhausted_topic = exhausted_dlq_topic(settings.dlq_topic)
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
            value_deserializer=decode_kafka_value,
            max_poll_records=settings.max_poll_records,
            **kafka_consumer_health_kwargs(settings.kafka_config),
            **kafka_security_kwargs(settings.kafka_config),
        )
        self.running = True

    def _replay(self, dlq_record: Any) -> dict[str, Any] | None:
        """Decide what to do with one DLQ record.

        Returns ``None`` for an unusable record, otherwise a dict carrying the
        destination ``topic``, the ``original`` payload, and a ``reason``.
        """
        if not isinstance(dlq_record, Mapping):
            return None
        original = dlq_record.get("original")
        if not isinstance(original, dict):
            return None
        original = dict(original)
        # The archive reference must survive the replay: it is the idempotency
        # key for the Elasticsearch document, and it was minted from the offset
        # the event first landed on.
        raw_log_ref = original.get("raw_log_ref") or dlq_record.get("raw_log_ref")
        if raw_log_ref:
            original["raw_log_ref"] = raw_log_ref
        original_topic = dlq_record.get("original_topic", self.settings.raw_topic)
        replay_count = int(original.get("_dlq_replay_count", 0)) + 1
        if replay_count > _MAX_DLQ_REPLAYS:
            return {
                "topic": self.exhausted_topic,
                "original": original,
                "original_topic": original_topic,
                "reason": (
                    f"replay budget exhausted after {replay_count - 1} attempts: "
                    f"{dlq_record.get('error', 'unknown error')}"
                ),
            }
        original["_dlq_replay_count"] = replay_count
        return {"topic": original_topic, "original": original, "reason": None}

    def _publish(self, replay: dict[str, Any]) -> None:
        if replay["reason"]:
            future = self.producer.send(
                replay["topic"],
                value={
                    "original": replay["original"],
                    "original_topic": replay["original_topic"],
                    "error": replay["reason"],
                    "exhausted_at": time.time(),
                },
                key=replay["original"].get("host", "unknown"),
            )
        else:
            future = self.producer.send(
                replay["topic"],
                value=replay["original"],
                key=replay["original"].get("host", "unknown"),
            )
        # Await the broker acknowledgement so a failed replay is not counted as
        # processed. Raising here aborts the loop before consumer.commit(), so the
        # DLQ record stays uncommitted and is retried on the next run. flush()
        # alone waits without surfacing the per-record error.
        future.get(timeout=10)
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
                            logger.error(
                                "Discarding unusable DLQ record",
                                topic=self.settings.dlq_topic,
                                partition=getattr(msg, "partition", None),
                                offset=getattr(msg, "offset", None),
                            )
                            continue
                        if replay["reason"]:
                            logger.error(
                                "DLQ record exhausted its replay budget; "
                                "publishing to the exhausted topic",
                                topic=replay["topic"],
                                original_topic=replay["original_topic"],
                                reason=replay["reason"],
                            )
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
