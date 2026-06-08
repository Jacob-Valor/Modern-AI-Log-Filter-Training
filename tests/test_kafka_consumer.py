"""Tests for Kafka consumer orchestration with fake clients."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from logfilter.kafka import consumer as consumer_module
from logfilter.kafka.consumer import ArchiveConsumer, ScorerConsumer
from logfilter.kafka.producer import LogProducer
from logfilter.pipeline.archive import compute_kafka_raw_log_ref


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


def test_archive_consumer_redacts_raw_and_uses_kafka_offset_document_id(monkeypatch) -> None:
    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return {"errors": False}

    class SensitiveKafkaConsumer(FakeKafkaConsumer):
        def poll(self, timeout_ms: int):
            self.poll_calls += 1
            if self.poll_calls == 1:
                msg = SimpleNamespace(
                    value={
                        "raw": "email=alice@example.com password=hunter2",
                        "source_type": "syslog",
                        "host": "host",
                        "ingest_ts": 1.0,
                    },
                    offset=7,
                    partition=2,
                )
                return {"topic-partition": [msg]}
            raise KeyboardInterrupt

    monkeypatch.setattr(consumer_module, "KafkaConsumer", SensitiveKafkaConsumer)
    es = FakeES()
    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=es)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert es.bulk_calls[0][0]["index"]["_id"] == compute_kafka_raw_log_ref(
        "raw-logs", 2, 7
    )
    assert es.bulk_calls[0][1]["raw"] == "email=<EMAIL> password=<REDACTED>"


def test_archive_consumer_defaults_to_earliest_offsets() -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": False}

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert FakeKafkaConsumer.instances[0].kwargs["auto_offset_reset"] == "earliest"


def test_scorer_consumer_uses_configured_offset_reset() -> None:
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host", "score": 1.0}],
        kafka_config={"auto_offset_reset": "none"},
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert FakeKafkaConsumer.instances[0].kwargs["auto_offset_reset"] == "none"


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


def test_archive_consumer_sends_to_dlq_on_bulk_failure() -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": True}

    class FakeDLQProducer:
        def __init__(self) -> None:
            self.dlq_sent: list[dict] = []

        def send_dlq(self, original_message, error, original_topic, retry_count=0):
            self.dlq_sent.append({
                "original_message": original_message,
                "error": error,
                "original_topic": original_topic,
                "retry_count": retry_count,
            })

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092", "raw-logs", es_client=FakeES(), dlq_producer=cast(LogProducer, dlq)
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.commits == 1
    assert len(dlq.dlq_sent) == 1
    assert dlq.dlq_sent[0]["original_topic"] == "raw-logs"
    assert dlq.dlq_sent[0]["error"] == "Elasticsearch bulk response contained errors"
    assert dlq.dlq_sent[0]["retry_count"] == 1


def test_archive_consumer_increments_retry_count() -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": True}

    class FakeDLQProducer:
        def __init__(self) -> None:
            self.dlq_sent: list[dict] = []

        def send_dlq(self, original_message, error, original_topic, retry_count=0):
            self.dlq_sent.append({"retry_count": retry_count})

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092", "raw-logs", es_client=FakeES(), dlq_producer=cast(LogProducer, dlq)
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert dlq.dlq_sent[0]["retry_count"] == 1


def test_scorer_consumer_sends_to_dlq_on_score_failure() -> None:
    class FakeDLQProducer:
        def __init__(self) -> None:
            self.dlq_sent: list[dict] = []

        def send_dlq(self, original_message, error, original_topic, retry_count=0):
            self.dlq_sent.append({
                "original_message": original_message,
                "error": error,
                "original_topic": original_topic,
                "retry_count": retry_count,
            })

    dlq = FakeDLQProducer()
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: (_ for _ in ()).throw(RuntimeError("score failed")),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = FakeKafkaConsumer.instances[0]
    assert fake.commits == 1
    assert len(dlq.dlq_sent) == 1
    assert dlq.dlq_sent[0]["original_topic"] == "raw-logs"
    assert "score failed" in dlq.dlq_sent[0]["error"]
    assert dlq.dlq_sent[0]["retry_count"] == 1


def test_archive_consumer_empty_poll(monkeypatch) -> None:
    class EmptyPollConsumer:
        instances: list = []

        def __init__(self, *args, **kwargs) -> None:
            self.commits = 0
            self.closed = False
            self.poll_calls = 0
            EmptyPollConsumer.instances.append(self)

        def poll(self, timeout_ms: int):
            self.poll_calls += 1
            if self.poll_calls > 1:
                raise KeyboardInterrupt
            return {}

        def commit(self) -> None:
            self.commits += 1

        def close(self) -> None:
            self.closed = True

    EmptyPollConsumer.instances = []
    monkeypatch.setattr(consumer_module, "KafkaConsumer", EmptyPollConsumer)

    class FakeES:
        def bulk(self, body):
            return {"errors": False}

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert EmptyPollConsumer.instances[0].commits == 0


def test_scorer_consumer_empty_batch_continue(monkeypatch) -> None:
    class EmptyPollConsumer:
        instances: list = []

        def __init__(self, *args, **kwargs) -> None:
            self.commits = 0
            self.closed = False
            self.poll_calls = 0
            EmptyPollConsumer.instances.append(self)

        def poll(self, timeout_ms: int):
            self.poll_calls += 1
            if self.poll_calls > 1:
                raise KeyboardInterrupt
            return {}

        def commit(self) -> None:
            self.commits += 1

        def close(self) -> None:
            self.closed = True

    EmptyPollConsumer.instances = []
    monkeypatch.setattr(consumer_module, "KafkaConsumer", EmptyPollConsumer)

    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: batch,
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert EmptyPollConsumer.instances[0].commits == 0


def test_scorer_consumer_no_flush_without_producer(monkeypatch) -> None:
    class NoFlushConsumer:
        instances: list = []

        def __init__(self, *args, **kwargs) -> None:
            self.commits = 0
            self.closed = False
            self.poll_calls = 0
            NoFlushConsumer.instances.append(self)

        def poll(self, timeout_ms: int):
            self.poll_calls += 1
            if self.poll_calls > 1:
                raise KeyboardInterrupt
            msg = SimpleNamespace(
                value={"raw": "raw", "source_type": "syslog", "host": "host"},
                offset=1,
                partition=0,
            )
            return {"tp": [msg]}

        def commit(self) -> None:
            self.commits += 1

        def close(self) -> None:
            self.closed = True

    NoFlushConsumer.instances = []
    monkeypatch.setattr(consumer_module, "KafkaConsumer", NoFlushConsumer)

    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host"}],
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert NoFlushConsumer.instances[0].commits == 1


def test_archive_consumer_bulk_error_without_dlq(monkeypatch) -> None:
    class FakeES:
        def bulk(self, body):
            return {"errors": True}

    monkeypatch.setattr(consumer_module, "KafkaConsumer", FakeKafkaConsumer)
    FakeKafkaConsumer.instances = []

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert FakeKafkaConsumer.instances[0].commits == 0
