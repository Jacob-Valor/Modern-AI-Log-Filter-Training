"""Tests for Kafka-to-QRadar router entrypoint logic."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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

    def send(self, leef: str, priority: str) -> None:
        self.sent.append((leef, priority))

    def close(self) -> None:
        self.closed = True


class FakeKafkaProducer:
    instances: list[FakeKafkaProducer] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.sent = []
        self.flushed = False
        self.closed = False
        FakeKafkaProducer.instances.append(self)

    def send(self, *args, **kwargs):
        self.sent.append((args, kwargs))

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
    monkeypatch.setattr(kafka_router, "SyslogSender", FakeSender)
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
            "qradar": {"syslog_host": "qradar", "syslog_port": 1514, "syslog_protocol": "udp"},
        },
    )

    settings = kafka_router._settings()

    assert settings.api_token == "env-token"
    assert settings.api_url == "http://api"
    assert settings.qradar_port == 1514
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
    monkeypatch.setattr(kafka_router, "SyslogSender", FakeSender)
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
        }
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


def test_score_batch_rejects_result_count_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(kafka_router, "SyslogSender", FakeSender)
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
    fake_router._route_scored([{"leef_payload": "leef", "ai_priority": "HIGH", "host": "host"}])

    assert fake_router.sender.sent == [("leef", "HIGH")]
    assert fake_router.producer.sent[0][0][0] == "scored-logs"
    assert fake_router.producer.flushed


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


def test_dlq_replay_caps_replay_count(monkeypatch) -> None:
    FakeKafkaProducer.instances = []
    FakeDLQConsumer.instances = []
    FakeDLQConsumer.messages = [
        {
            "original": {"raw": "raw", "_dlq_replay_count": 3},
            "original_topic": "raw-logs",
        }
    ]
    monkeypatch.setattr(kafka_router, "KafkaConsumer", FakeDLQConsumer)
    monkeypatch.setattr(kafka_router, "KafkaProducer", FakeKafkaProducer)
    settings = replace(_settings(), dlq_topic="dlq")
    replay = kafka_router.DLQReplay(settings)
    replay.run(max_messages=1, max_empty_polls=1)

    assert FakeKafkaProducer.instances[0].sent == []
