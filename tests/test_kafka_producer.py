"""Tests for Kafka producer wrapper behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest

from logfilter.kafka import producer as producer_module
from logfilter.kafka.producer import LogProducer


class FakeRecordMetadata:
    topic = "raw-logs"
    partition = 0
    offset = 42


class FakeFuture:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def get(self, timeout: int) -> FakeRecordMetadata:
        if self.fail:
            raise producer_module.KafkaError("send failed")
        return FakeRecordMetadata()


class FakeKafkaProducer:
    instances: list[FakeKafkaProducer] = []

    def __init__(self, **config) -> None:
        self.config = config
        self.sent: list[dict] = []
        self.flushed = False
        self.closed = False
        self.fail_next = False
        FakeKafkaProducer.instances.append(self)

    def send(self, topic, value, key=None, headers=None):
        self.sent.append({"topic": topic, "value": value, "key": key, "headers": headers})
        return FakeFuture(fail=self.fail_next)

    def flush(self, timeout: int) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_kafka(monkeypatch):
    FakeKafkaProducer.instances = []
    monkeypatch.setattr(producer_module, "KafkaProducer", FakeKafkaProducer)


def test_log_producer_send_builds_payload() -> None:
    producer = LogProducer(bootstrap_servers="kafka:29092", topic="raw-logs")

    producer.send("raw log", source_type="syslog", host="host", metadata={"peer": "10.0.0.1"})

    fake = FakeKafkaProducer.instances[0]
    assert fake.sent[0]["topic"] == "raw-logs"
    assert fake.sent[0]["key"] == "host"
    assert fake.sent[0]["value"]["raw"] == "raw log"
    assert fake.sent[0]["value"]["peer"] == "10.0.0.1"
    assert isinstance(fake.sent[0]["headers"], list)


def test_log_producer_passes_kafka_security_config() -> None:
    producer = LogProducer(
        bootstrap_servers="kafka:29092",
        topic="raw-logs",
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

    producer.send("raw log", host="host")

    fake = FakeKafkaProducer.instances[0]
    assert fake.config["security_protocol"] == "SASL_SSL"
    assert fake.config["sasl_mechanism"] == "PLAIN"
    assert fake.config["sasl_plain_username"] == "logfilter"
    assert fake.config["sasl_plain_password"] == "secret"
    assert fake.config["ssl_cafile"] == "/etc/kafka/ca.pem"


def test_log_producer_send_raises_kafka_error() -> None:
    producer = LogProducer(topic="raw-logs")
    fake = cast(Any, producer._get_producer())
    fake.fail_next = True

    with pytest.raises(producer_module.KafkaError):
        cast(Any, producer.send).retry.statistics.clear()
        producer.send("raw")


def test_log_producer_send_batch_flushes() -> None:
    producer = LogProducer(topic="raw-logs")

    count = producer.send_batch(
        [
            {"raw": "a", "source_type": "syslog", "host": "h1"},
            {"raw": "b", "host": "h2"},
        ]
    )

    fake = FakeKafkaProducer.instances[0]
    assert count == 2
    assert fake.flushed
    assert fake.sent[1]["value"]["source_type"] == "generic"


def test_log_producer_close_flushes_and_closes() -> None:
    producer = LogProducer(topic="raw-logs")
    fake = cast(Any, producer._get_producer())

    producer.close()

    assert fake.flushed
    assert fake.closed
    assert producer._producer is None


def test_log_producer_send_dlq_builds_payload() -> None:
    producer = LogProducer(
        bootstrap_servers="kafka:29092",
        topic="raw-logs",
        dlq_topic="logfilter-dlq",
    )

    producer.send_dlq(
        original_message={"raw": "failed log", "host": "host1", "raw_log_ref": "ref-1"},
        error="ES bulk failed",
        original_topic="raw-logs",
        retry_count=2,
    )

    fake = FakeKafkaProducer.instances[0]
    assert fake.sent[0]["topic"] == "logfilter-dlq"
    assert fake.sent[0]["value"]["original"]["raw"] == "failed log"
    assert fake.sent[0]["value"]["error"] == "ES bulk failed"
    assert fake.sent[0]["value"]["original_topic"] == "raw-logs"
    assert fake.sent[0]["value"]["retry_count"] == 2
    assert "timestamp" in fake.sent[0]["value"]
    assert fake.sent[0]["key"] == "host1"


def test_log_producer_send_dlq_carries_raw_log_ref_for_replay() -> None:
    """The replayed event must keep the archive ref minted on first consumption."""
    producer = LogProducer(topic="raw-logs", dlq_topic="logfilter-dlq")

    producer.send_dlq(
        original_message={"raw": "failed log", "host": "h", "raw_log_ref": "ref-42"},
        error="boom",
        original_topic="raw-logs",
    )

    payload = FakeKafkaProducer.instances[0].sent[0]["value"]
    assert payload["raw_log_ref"] == "ref-42"
    assert payload["original"]["raw_log_ref"] == "ref-42"


def test_log_producer_send_dlq_without_topic_raises_instead_of_dropping() -> None:
    """A silently dropped DLQ message loses the raw log while callers commit."""
    producer = LogProducer(topic="raw-logs")

    with pytest.raises(producer_module.DlqNotConfiguredError):
        producer.send_dlq(
            original_message={"raw": "failed log"},
            error="ES bulk failed",
            original_topic="raw-logs",
        )

    assert FakeKafkaProducer.instances == []


def test_log_producer_send_dlq_raises_on_kafka_error() -> None:
    producer = LogProducer(topic="raw-logs", dlq_topic="logfilter-dlq")
    fake = cast(Any, producer._get_producer())
    fake.fail_next = True

    with pytest.raises(producer_module.KafkaError):
        cast(Any, producer._send_dlq_payload).retry.statistics.clear()
        producer.send_dlq(
            original_message={"raw": "failed log"},
            error="ES bulk failed",
            original_topic="raw-logs",
        )


def test_log_producer_send_retry_exit_branch() -> None:
    producer = LogProducer(topic="raw-logs")
    fake = cast(Any, producer._get_producer())
    fake.fail_next = True

    with pytest.raises(producer_module.KafkaError):
        cast(Any, producer.send).retry.statistics.clear()
        producer.send("raw log", host="host")


def test_log_producer_send_batch_exception_handling() -> None:
    class ExceptionProducer:
        def __init__(self) -> None:
            self.sent = []
            self.flushed = False

        def send(self, *args, **kwargs):
            from kafka.errors import KafkaError

            raise KafkaError("batch fail")

        def flush(self, timeout: int) -> None:
            self.flushed = True

    producer = LogProducer(topic="raw-logs")
    producer._producer = ExceptionProducer()

    count = producer.send_batch([{"raw": "a", "host": "h1"}])
    assert count == 0


def test_log_producer_send_batch_excludes_unacknowledged_sends() -> None:
    """A future carrying a KafkaError means the broker never took the message."""

    class PerRecordFuture:
        def __init__(self, fail: bool) -> None:
            self._fail = fail

        def get(self, timeout: int):
            if self._fail:
                raise producer_module.KafkaError("not acknowledged")
            return FakeRecordMetadata()

    class PerRecordProducer:
        def send(self, topic, value, key=None, headers=None):
            return PerRecordFuture(fail=key == "bad-host")

        def flush(self, timeout: int) -> None:
            pass

    producer = LogProducer(topic="raw-logs")
    producer._producer = cast(Any, PerRecordProducer())

    count = producer.send_batch(
        [
            {"raw": "a", "host": "good-host"},
            {"raw": "b", "host": "bad-host"},
            {"raw": "c", "host": "good-host"},
        ]
    )

    assert count == 2


