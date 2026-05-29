"""Tests for Kafka-to-QRadar router entrypoint logic."""

from __future__ import annotations

from dataclasses import replace

import pytest

from logfilter import kafka_router


def _settings() -> kafka_router.RouterSettings:
    return kafka_router.RouterSettings(
        bootstrap_servers="kafka:29092",
        raw_topic="raw-logs",
        scored_topic="scored-logs",
        consumer_group="router",
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
    assert settings.kafka_config["security"]["protocol"] == "SSL"


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
