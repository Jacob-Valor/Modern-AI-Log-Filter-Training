"""Tests for Kafka producer wrapper behavior."""

from __future__ import annotations

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

    def send(self, topic, value, key=None):
        self.sent.append({"topic": topic, "value": value, "key": key})
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


def test_log_producer_send_raises_kafka_error() -> None:
    producer = LogProducer(topic="raw-logs")
    fake = producer._get_producer()
    fake.fail_next = True

    with pytest.raises(producer_module.KafkaError):
        producer.send.retry.statistics.clear()
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
    fake = producer._get_producer()

    producer.close()

    assert fake.flushed
    assert fake.closed
    assert producer._producer is None
