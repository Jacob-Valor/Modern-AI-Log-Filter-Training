"""Tests for Kafka consumer orchestration with fake clients."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from logfilter.kafka import consumer as consumer_module
from logfilter.kafka.consumer import ArchiveConsumer, ScorerConsumer


class FakeKafkaConsumer:
    instances: list[FakeKafkaConsumer] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.poll_calls = 0
        self.commits = 0
        self.closed = False
        FakeKafkaConsumer.instances.append(self)

    def poll(self, timeout_ms: int):
        self.poll_calls += 1
        if self.poll_calls == 1:
            msg = SimpleNamespace(
                value={"raw": "raw", "source_type": "syslog", "host": "host", "ingest_ts": 1.0},
                offset=7,
                partition=2,
            )
            return {"topic-partition": [msg]}
        raise KeyboardInterrupt

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_consumer(monkeypatch):
    FakeKafkaConsumer.instances = []
    monkeypatch.setattr(consumer_module, "KafkaConsumer", FakeKafkaConsumer)


def test_archive_consumer_writes_bulk_and_commits() -> None:
    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return {"errors": False}

    es = FakeES()
    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=es)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.commits == 1
    assert fake.closed
    assert es.bulk_calls[0][1]["raw"] == "raw"


def test_archive_consumer_passes_kafka_security_config() -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": False}

    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        kafka_config={
            "security": {
                "protocol": "SSL",
                "ssl": {"cafile": "/etc/kafka/ca.pem", "check_hostname": "false"},
            }
        },
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.kwargs["security_protocol"] == "SSL"
    assert fake.kwargs["ssl_cafile"] == "/etc/kafka/ca.pem"
    assert fake.kwargs["ssl_check_hostname"] is False


def test_archive_consumer_does_not_commit_on_bulk_errors() -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": True}

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert FakeKafkaConsumer.instances[0].commits == 0


def test_scorer_consumer_scores_publishes_and_commits() -> None:
    class FakeProducer:
        def __init__(self) -> None:
            self.sent = []
            self.flushed = False

        def send(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def flush(self, timeout: int) -> None:
            self.flushed = True

    producer = FakeProducer()
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host", "score": 1.0}],
        producer=producer,
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.commits == 1
    assert fake.closed
    assert producer.sent[0][0][0] == "scored-logs"
    assert producer.flushed


def test_scorer_consumer_passes_kafka_security_config() -> None:
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host", "score": 1.0}],
        kafka_config={
            "security": {
                "protocol": "SASL_PLAINTEXT",
                "sasl": {
                    "mechanism": "SCRAM-SHA-256",
                    "username": "logfilter",
                    "password": "secret",
                },
            }
        },
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.kwargs["security_protocol"] == "SASL_PLAINTEXT"
    assert fake.kwargs["sasl_mechanism"] == "SCRAM-SHA-256"
    assert fake.kwargs["sasl_plain_username"] == "logfilter"
    assert fake.kwargs["sasl_plain_password"] == "secret"


def test_scorer_consumer_does_not_commit_when_score_fails() -> None:
    def fail(batch):
        raise RuntimeError("score failed")

    consumer = ScorerConsumer("kafka:29092", "raw-logs", "scored-logs", score_fn=fail)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert FakeKafkaConsumer.instances[0].commits == 0
