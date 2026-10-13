"""Tests for Kafka-to-QRadar router entrypoint logic."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from logfilter import kafka_router


def _settings() -> kafka_router.RouterSettings:
    return kafka_router.RouterSettings(
        bootstrap_servers="kafka:29092",
        raw_topic="raw-logs",
        scored_topic="scored-logs",
        dlq_topic=None,
        consumer_group="router",
        auto_offset_reset="earliest",
        max_poll_records=10,
        poll_timeout_ms=100,
        api_url="http://api",
        api_token="token",
        qradar_host="qradar",
        qradar_port=514,
        qradar_protocol="tcp",
        qradar_gateway={},
        qradar_mode="enrich_only",
        kafka_config={},
    )


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


class FakeHTTPClient:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {"results": [{"leef_payload": "leef", "ai_priority": "HIGH"}]}
        self.posts = []
        self.closed = False

    def post(self, *args, **kwargs):
        self.posts.append((args, kwargs))
        return FakeHTTPResponse(self.payload)

    def close(self) -> None:
        self.closed = True


class FakeSender:
    def __init__(self, *args, **kwargs) -> None:
        self.sent = []
        self.closed = False

    def send(self, leef: str, priority: str, *, event_id: str | None = None) -> None:
        self.sent.append((leef, priority, event_id))

    def close(self) -> None:
        self.closed = True


class FakeFuture:
    """Mimics kafka's FutureRecordMetadata, which is what send() must return."""

    def __init__(self, producer: FakeKafkaProducer, args: tuple, kwargs: dict) -> None:
        self._producer = producer
        self._args = args
        self._kwargs = kwargs

    def get(self, timeout: int | None = None):
        if self._producer.fail_on_get:
            raise RuntimeError("broker unavailable")
        self._producer.acked.append((self._args, self._kwargs))
        return object()


class FakeKafkaProducer:
    instances: list[FakeKafkaProducer] = []
    fail_on_get: bool = False

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.sent = []
        self.acked = []
        self.flushed = False
        self.closed = False
        FakeKafkaProducer.instances.append(self)

    def send(self, *args, **kwargs):
        self.sent.append((args, kwargs))
        return FakeFuture(self, args, kwargs)

    def flush(self, timeout: int) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class FakeKafkaConsumer:
    instances: list[FakeKafkaConsumer] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.closed = False
        FakeKafkaConsumer.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeDLQConsumer:
    instances: list[FakeDLQConsumer] = []
    messages: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.commits = 0
        self.poll_calls = 0
        FakeDLQConsumer.instances.append(self)

    def poll(self, timeout_ms: int = 0):
        self.poll_calls += 1
        if self.poll_calls == 1 and self.messages:
            return {"tp": [SimpleNamespace(value=m) for m in self.messages]}
        return {}

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_router(monkeypatch):
    FakeKafkaProducer.instances = []
    FakeKafkaConsumer.instances = []
    monkeypatch.setattr(kafka_router, "build_qradar_sender", lambda config: FakeSender())
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(kafka_router.httpx, "Client", lambda *args, **kwargs: FakeHTTPClient())
    return kafka_router.KafkaQRadarRouter(_settings())


def test_settings_requires_api_token(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_API_TOKEN", raising=False)
    monkeypatch.setattr(kafka_router, "load_config", lambda: {"api": {}})

    with pytest.raises(SystemExit, match="LOGFILTER_API_TOKEN"):
        kafka_router._settings()


def test_settings_reads_config_and_environment(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_API_TOKEN", "env-token")
    monkeypatch.setenv("LOGFILTER_API_URL", "http://api/")
    monkeypatch.setattr(
        kafka_router,
        "load_config",
        lambda: {
            "kafka": {
                "bootstrap_servers": "kafka:29092",
                "topics": {"raw_logs": "raw", "scored_logs": "scored"},
                "security": {"protocol": "SSL", "ssl": {"cafile": "/etc/kafka/ca.pem"}},
            },
            "qradar": {
                "syslog_host": "qradar",
                "syslog_port": 1514,
                "syslog_protocol": "udp",
                "mode": "suppress_low",
            },
        },
    )

    settings = kafka_router._settings()

    assert settings.api_token == "env-token"
    assert settings.api_url == "http://api"
    assert settings.qradar_port == 1514
    assert settings.qradar_mode == "suppress_low"
    assert settings.consumer_group == "logfilter-router"
    assert settings.auto_offset_reset == "earliest"
    assert settings.kafka_config["security"]["protocol"] == "SSL"


def test_settings_accepts_router_group_and_offset_override(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_API_TOKEN", "env-token")
    monkeypatch.setattr(
        kafka_router,
        "load_config",
        lambda: {
            "kafka": {
                "router_consumer_group": "custom-router",
                "auto_offset_reset": "none",
            }
        },
    )

    settings = kafka_router._settings()

    assert settings.consumer_group == "custom-router"
    assert settings.auto_offset_reset == "none"


def test_router_consumer_uses_settings_offset_reset(fake_router) -> None:
    assert FakeKafkaConsumer.instances[0].kwargs["auto_offset_reset"] == "earliest"


def test_router_passes_kafka_security_config(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeKafkaConsumer.instances = []
    monkeypatch.setattr(kafka_router, "build_qradar_sender", lambda config: FakeSender())
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(kafka_router.httpx, "Client", lambda *args, **kwargs: FakeHTTPClient())

    settings = replace(
        _settings(),
        kafka_config={
            "security": {
                "protocol": "SASL_SSL",
                "sasl": {
                    "mechanism": "PLAIN",
                    "username": "logfilter",
                    "password": "secret",
                },
                "ssl": {"cafile": "/etc/kafka/ca.pem"},
            }
        },
    )

    kafka_router.KafkaQRadarRouter(settings)

    consumer = FakeKafkaConsumer.instances[0]
    producer = FakeKafkaProducer.instances[0]
    assert consumer.kwargs["security_protocol"] == "SASL_SSL"
    assert consumer.kwargs["sasl_plain_username"] == "logfilter"
    assert consumer.kwargs["ssl_cafile"] == "/etc/kafka/ca.pem"
    assert producer.kwargs["security_protocol"] == "SASL_SSL"
    assert producer.kwargs["sasl_plain_password"] == "secret"
    assert producer.kwargs["ssl_cafile"] == "/etc/kafka/ca.pem"


def test_score_batch_posts_api_token(fake_router) -> None:
    results = fake_router._score_batch([{"raw": "raw", "source_type": "syslog"}])

    assert results == [{"leef_payload": "leef", "ai_priority": "HIGH"}]
    assert fake_router.http.posts[0][1]["headers"]["X-API-Token"] == "token"
    assert fake_router.http.posts[0][1]["json"]["events"][0]["raw_log_ref"] is None


def test_score_batch_rejects_result_count_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(kafka_router, "build_qradar_sender", lambda config: FakeSender())
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(
        kafka_router.httpx,
        "Client",
        lambda *args, **kwargs: FakeHTTPClient(payload={"results": []}),
    )
    router = kafka_router.KafkaQRadarRouter(_settings())

    with pytest.raises(RuntimeError, match="returned 0 results"):
        router._score_batch([{"raw": "raw"}])


def test_route_scored_sends_leef_and_publishes(fake_router) -> None:
    fake_router._route_scored(
        [
            {
                "leef_payload": "leef",
                "ai_priority": "HIGH",
                "host": "host",
                "raw_log_ref": "a" * 64,
            }
        ]
    )

    assert fake_router.sender.sent == [("leef", "HIGH", "a" * 64)]
    assert fake_router.producer.sent[0][0][0] == "scored-logs"
    assert fake_router.producer.flushed


def test_route_scored_raises_when_broker_does_not_acknowledge(fake_router) -> None:
    """A scored event must not be counted as published unless the broker acked it.

    Regression: this is a raw KafkaProducer, so send() only queues the record.
    flush() waits but does not raise, so an unacknowledged publish used to be
    silently dropped — the event reached neither the scored-logs topic nor the DLQ.
    """
    fake_router.producer.fail_on_get = True
    try:
        with pytest.raises(kafka_router.RouteBatchError, match="broker unavailable"):
            fake_router._route_scored(
                [
                    {
                        "leef_payload": "leef",
                        "ai_priority": "HIGH",
                        "host": "host",
                        "raw_log_ref": "b" * 64,
                    }
                ]
            )
    finally:
        fake_router.producer.fail_on_get = False

    # It was queued, but never acknowledged — so it must not count as published.
    assert fake_router.producer.acked == []


def test_route_scored_records_acknowledgement(fake_router) -> None:
    fake_router._route_scored(
        [
            {
                "leef_payload": "leef",
                "ai_priority": "HIGH",
                "host": "host",
                "raw_log_ref": "c" * 64,
            }
        ]
    )

    assert len(fake_router.producer.acked) == 1


def test_suppress_low_publishes_without_forwarding_to_qradar(fake_router) -> None:
    fake_router.settings = replace(fake_router.settings, qradar_mode="suppress_low")

    fake_router._route_scored(
        [
            {
                "leef_payload": "leef",
                "ai_priority": "INFO",
                "ai_threat_score": 0.1,
                "host": "host",
            }
        ]
    )

    assert fake_router.sender.sent == []
    assert fake_router.producer.sent[0][1]["value"]["qradar_forwarded"] is False


def test_route_error_identifies_only_unfinished_suffix(fake_router) -> None:
    calls = 0

    def fail_second(leef: str, priority: str, *, event_id: str | None = None) -> None:
        nonlocal calls
        del leef, priority, event_id
        calls += 1
        if calls == 2:
            raise OSError("qradar unavailable")

    fake_router.sender.send = fail_second
    scored = [
        {"leef_payload": "first", "ai_priority": "HIGH", "host": "one"},
        {"leef_payload": "second", "ai_priority": "HIGH", "host": "two"},
        {"leef_payload": "third", "ai_priority": "HIGH", "host": "three"},
    ]

    with pytest.raises(kafka_router.RouteBatchError) as exc_info:
        fake_router._route_scored(scored)

    assert exc_info.value.failed_index == 1
    assert len(fake_router.producer.sent) == 1


def test_kafka_event_receives_replay_stable_archive_reference() -> None:
    message = SimpleNamespace(
        value={"raw": "raw", "source_type": "syslog"},
        topic="raw-logs",
        partition=2,
        offset=7,
    )

    event = kafka_router._event_with_archive_ref(message)

    assert event["raw_log_ref"] == kafka_router.compute_kafka_raw_log_ref("raw-logs", 2, 7)


def test_route_scored_requires_leef_payload(fake_router) -> None:
    with pytest.raises(RuntimeError, match="missing leef_payload"):
        fake_router._route_scored([{"ai_priority": "HIGH"}])


def test_dlq_replay_replays_to_original_topic(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeDLQConsumer.instances = []
    FakeDLQConsumer.messages = [
        {
            "original": {"raw": "raw", "source_type": "syslog", "host": "host"},
            "original_topic": "raw-logs",
            "error": "test",
        }
    ]
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeDLQConsumer)
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    settings = replace(_settings(), dlq_topic="dlq")
    replay = kafka_router.DLQReplay(settings)
    replay.run(max_messages=1, max_empty_polls=1)

    assert FakeKafkaProducer.instances[0].sent[0][0][0] == "raw-logs"
    assert FakeDLQConsumer.instances[0].commits == 1
    assert FakeDLQConsumer.instances[0].closed is True


def test_dlq_replay_skips_invalid_records(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeDLQConsumer.instances = []
    FakeDLQConsumer.messages = [{"original": "not-a-dict"}]
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeDLQConsumer)
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    settings = replace(_settings(), dlq_topic="dlq")
    replay = kafka_router.DLQReplay(settings)
    replay.run(max_messages=1, max_empty_polls=1)

    assert FakeKafkaProducer.instances[0].sent == []


def test_dlq_replay_publishes_exhausted_record_to_terminal_topic(monkeypatch) -> None:
    """Exhausted records must be published, not dropped before the commit."""
    FakeKafkaProducer.instances = []
    FakeDLQConsumer.instances = []
    FakeDLQConsumer.messages = [
        {
            "original": {"raw": "raw", "_dlq_replay_count": 3},
            "original_topic": "raw-logs",
            "error": "permanent failure",
        }
    ]
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeDLQConsumer)
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    settings = replace(_settings(), dlq_topic="logfilter-dlq")
    replay = kafka_router.DLQReplay(settings)
    replay.run(max_messages=1, max_empty_polls=1)

    sent = FakeKafkaProducer.instances[0].sent
    assert len(sent) == 1
    assert sent[0][0][0] == "logfilter-dlq-exhausted"
    assert sent[0][1]["value"]["original"]["raw"] == "raw"
    assert "permanent failure" in sent[0][1]["value"]["error"]
    assert sent[0][1]["value"]["original_topic"] == "raw-logs"


def test_dlq_replay_exhausted_topic_is_derived_from_dlq_topic() -> None:
    from logfilter.kafka.config import exhausted_dlq_topic

    assert exhausted_dlq_topic("logfilter-dlq") == "logfilter-dlq-exhausted"
    assert exhausted_dlq_topic(None) is None


def test_dlq_replay_requires_a_dlq_topic() -> None:
    with pytest.raises(SystemExit, match="dlq must be configured"):
        kafka_router.DLQReplay(_settings())


class ScriptedRunConsumer:
    """Serves scripted poll rounds, then interrupts to end ``run()``."""

    def __init__(self, rounds: list[dict]) -> None:
        self.rounds = rounds
        self.polls = 0
        self.commits = 0
        self.closed = False

    def poll(self, timeout_ms: int = 0):
        self.polls += 1
        if self.polls > len(self.rounds):
            raise KeyboardInterrupt
        return self.rounds[self.polls - 1]

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def _run_message(offset: int, partition: int, raw: str) -> SimpleNamespace:
    return SimpleNamespace(
        value={"raw": raw},
        topic="raw-logs",
        partition=partition,
        offset=offset,
    )


def _build_router_for_run(monkeypatch, rounds, *, score, route) -> tuple[Any, Any]:
    """A router wired to fakes, with ``score``/``route`` injected."""
    FakeKafkaProducer.instances = []
    FakeKafkaConsumer.instances = []
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(kafka_router, "build_qradar_sender", lambda config: FakeSender())
    monkeypatch.setattr(kafka_router.httpx, "Client", lambda *a, **k: FakeHTTPClient())
    monkeypatch.setattr(kafka_router, "LogProducer", _FakeDLQ)

    settings = replace(_settings(), dlq_topic="logfilter-dlq")
    router = kafka_router.KafkaQRadarRouter(settings)
    router.consumer = ScriptedRunConsumer(rounds)
    router._score_batch = score
    router._route_scored = route
    return router, router.dlq_producer


class _FakeDLQ:
    """Stands in for LogProducer so the router never opens a real connection."""

    def __init__(self, fail: Exception | None = None, **kwargs) -> None:
        del kwargs
        self.sent: list[dict] = []
        self.fail = fail

    def send_dlq(self, original_message, error, original_topic, retry_count=0):
        if self.fail:
            raise self.fail
        self.sent.append(
            {
                "original": dict(original_message),
                "error": error,
                "original_topic": original_topic,
                "retry_count": retry_count,
            }
        )

    def close(self) -> None:
        pass


def test_router_keeps_batch_when_api_circuit_breaker_is_open(monkeypatch) -> None:
    """CircuitBreakerOpen must never send a whole poll to the DLQ."""
    from logfilter.utils.circuit_breaker import CircuitBreakerOpen

    rounds = [
        {"tp": [_run_message(1, 0, "one")], "other": [_run_message(2, 0, "two")]},
        {"tp": [_run_message(3, 0, "three")]},
    ]

    def score(batch):
        raise CircuitBreakerOpen("scoring_api")

    router, dlq = _build_router_for_run(monkeypatch, rounds, score=score, route=lambda s: None)

    slept: list[float] = []
    monkeypatch.setattr(kafka_router, "_TRANSIENT_BACKOFF_SECONDS", 2.0)
    monkeypatch.setattr(kafka_router.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(KeyboardInterrupt):
        router.run()

    assert dlq.sent == [], "an API outage must not DLQ the polled batch"
    assert router.consumer.commits == 0, "offsets must stay uncommitted for a retry"
    assert slept, "the loop must back off instead of spinning on a dead API"


@pytest.mark.parametrize("error", [httpx.ConnectError("down"), httpx.ReadTimeout("slow")])
def test_router_keeps_batch_on_transient_http_error(monkeypatch, error) -> None:
    rounds = [{"tp": [_run_message(1, 0, "one"), _run_message(2, 0, "two")]}]

    def score(batch):
        raise error

    router, dlq = _build_router_for_run(monkeypatch, rounds, score=score, route=lambda s: None)
    monkeypatch.setattr(kafka_router, "_TRANSIENT_BACKOFF_SECONDS", 0.0)

    with pytest.raises(KeyboardInterrupt):
        router.run()

    assert dlq.sent == []
    assert router.consumer.commits == 0


def test_router_keeps_batch_on_429_and_5xx(monkeypatch) -> None:
    for status in (429, 503):
        rounds = [{"tp": [_run_message(1, 0, "one")]}]

        def score(batch, status=status):
            response = httpx.Response(status, request=httpx.Request("POST", "http://api"))
            raise httpx.HTTPStatusError("nope", request=response.request, response=response)

        router, dlq = _build_router_for_run(
            monkeypatch, rounds, score=score, route=lambda s: None
        )
        monkeypatch.setattr(kafka_router, "_TRANSIENT_BACKOFF_SECONDS", 0.0)

        with pytest.raises(KeyboardInterrupt):
            router.run()

        assert dlq.sent == [], f"status {status} must not discard the batch"
        assert router.consumer.commits == 0


def test_router_dlqs_only_the_unfinished_suffix_and_commits(monkeypatch) -> None:
    """A per-record failure discards that record, not the records already routed."""
    rounds = [
        {"tp": [_run_message(1, 0, "one"), _run_message(2, 0, "two"), _run_message(3, 0, "three")]}
    ]

    def score(batch):
        return [{"leef_payload": "leef", "ai_priority": "HIGH", **e} for e in batch]

    def route(scored):
        scored[1].pop("leef_payload", None)
        raise kafka_router.RouteBatchError(1, "missing leef_payload")

    router, dlq = _build_router_for_run(monkeypatch, rounds, score=score, route=route)

    with pytest.raises(KeyboardInterrupt):
        router.run()

    assert len(dlq.sent) == 2, "the failed record and everything after it"
    assert [d["original"]["raw"] for d in dlq.sent] == ["two", "three"]
    assert router.consumer.commits == 1


def test_router_does_not_commit_when_dlq_publish_fails(monkeypatch) -> None:
    from logfilter.kafka.producer import DlqNotConfiguredError

    rounds = [{"tp": [_run_message(1, 0, "one")]}]

    def score(batch):
        raise ValueError("permanent per-record failure")

    router, dlq = _build_router_for_run(monkeypatch, rounds, score=score, route=lambda s: None)
    router.dlq_producer = _FakeDLQ(fail=DlqNotConfiguredError("no dlq topic"))

    with pytest.raises(KeyboardInterrupt):
        router.run()

    assert router.consumer.commits == 0, "an undelivered record must stay on the topic"


def test_router_does_not_commit_when_no_dlq_producer(monkeypatch) -> None:
    rounds = [{"tp": [_run_message(1, 0, "one")]}]

    def score(batch):
        raise ValueError("permanent per-record failure")

    router, _ = _build_router_for_run(monkeypatch, rounds, score=score, route=lambda s: None)
    router.dlq_producer = None

    with pytest.raises(KeyboardInterrupt):
        router.run()

    assert router.consumer.commits == 0


def test_router_consumer_sets_max_poll_interval(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeKafkaConsumer.instances = []
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(kafka_router, "build_qradar_sender", lambda config: FakeSender())
    monkeypatch.setattr(kafka_router.httpx, "Client", lambda *a, **k: FakeHTTPClient())

    settings = replace(_settings(), kafka_config={"max_poll_interval_ms": 900000})
    kafka_router.KafkaQRadarRouter(settings)

    assert FakeKafkaConsumer.instances[0].kwargs["max_poll_interval_ms"] == 900000


def test_router_replay_reuses_the_original_raw_log_ref() -> None:
    """A replayed message must not mint a second archive ref for the same event."""
    original_ref = kafka_router.compute_kafka_raw_log_ref("raw-logs", 0, 5)
    replayed = SimpleNamespace(
        value={"raw": "raw", "raw_log_ref": original_ref},
        topic="raw-logs",
        partition=4,
        offset=900,
    )

    event = kafka_router._event_with_archive_ref(replayed)

    assert event["raw_log_ref"] == original_ref
    assert event["raw_log_ref"] != kafka_router.compute_kafka_raw_log_ref("raw-logs", 4, 900)


def test_dlq_replay_carries_raw_log_ref_from_the_dlq_record(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeDLQConsumer.instances = []
    FakeDLQConsumer.messages = [
        {
            "original": {"raw": "raw", "host": "h"},
            "original_topic": "raw-logs",
            "raw_log_ref": "ref-7",
        }
    ]
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeDLQConsumer)
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    replay = kafka_router.DLQReplay(replace(_settings(), dlq_topic="dlq"))
    replay.run(max_messages=1, max_empty_polls=1)

    published = FakeKafkaProducer.instances[0].sent[0][1]["value"]
    assert published["raw_log_ref"] == "ref-7"
