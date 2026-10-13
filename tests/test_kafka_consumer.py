"""Tests for Kafka consumer orchestration with fake clients."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from elastic_transport import ApiResponseMeta
from elasticsearch import exceptions as es_exceptions

from logfilter.kafka import consumer as consumer_module
from logfilter.kafka.consumer import ArchiveConsumer, ScorerConsumer
from logfilter.kafka.producer import LogProducer
from logfilter.pipeline.archive import compute_kafka_raw_log_ref
from logfilter.utils.circuit_breaker import CircuitBreaker


def es_status_error(status: int) -> es_exceptions.ApiError:
    """An Elasticsearch ApiError carrying a specific HTTP status code."""
    meta = ApiResponseMeta(
        status=status, http_version="1.1", headers={}, duration=0.0, node=None
    )
    return es_exceptions.ApiError("elasticsearch rejected the request", meta, {})


class FakeKafkaConsumer:
    """Replays scripted polls, then interrupts to end ``run()``."""

    instances: list[FakeKafkaConsumer] = []
    rounds: list[dict[Any, Any]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.poll_calls = 0
        self.commits = 0
        self.closed = False
        self.rounds = FakeKafkaConsumer.rounds
        FakeKafkaConsumer.instances.append(self)

    def poll(self, timeout_ms: int):
        self.poll_calls += 1
        if self.poll_calls > len(self.rounds):
            raise KeyboardInterrupt
        return self.rounds[self.poll_calls - 1]

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def message(offset: int, value: Any, partition: int = 2) -> SimpleNamespace:
    return SimpleNamespace(value=value, offset=offset, partition=partition)


def scripted(rounds: list[dict[Any, Any]]) -> None:
    FakeKafkaConsumer.instances = []
    FakeKafkaConsumer.rounds = rounds


def built() -> FakeKafkaConsumer:
    return FakeKafkaConsumer.instances[-1]


class FakeDLQProducer:
    def __init__(self, fail: bool = False) -> None:
        self.dlq_sent: list[dict] = []
        self.fail = fail

    def send_dlq(self, original_message, error, original_topic, retry_count=0):
        if self.fail:
            raise RuntimeError("dlq broker unreachable")
        self.dlq_sent.append(
            {
                "original_message": dict(original_message),
                "error": error,
                "original_topic": original_topic,
                "retry_count": retry_count,
            }
        )

    def close(self) -> None:
        pass


def ok_bulk(status: int = 201) -> dict:
    return {"errors": False, "items": [{"index": {"status": status}}]}


class _RecordingStopEvent:
    """Stands in for the run-loop stop event, recording backoff waits."""

    def __init__(self, waits: list[float]) -> None:
        self.waits = waits

    def clear(self) -> None:
        pass

    def set(self) -> None:
        pass

    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout if timeout is not None else 0.0)
        return False


@pytest.fixture(autouse=True)
def fake_consumer(monkeypatch):
    FakeKafkaConsumer.instances = []
    FakeKafkaConsumer.rounds = []
    monkeypatch.setattr(consumer_module, "KafkaConsumer", FakeKafkaConsumer)
    monkeypatch.setattr(consumer_module, "_TRANSIENT_BACKOFF_SECONDS", 0.0)


# --------------------------------------------------------------------- wiring


def test_archive_consumer_writes_bulk_and_commits() -> None:
    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return ok_bulk()

    scripted([{"topic-partition": [message(7, {"raw": "raw", "ingest_ts": 1.0})]}])
    es = FakeES()
    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=es)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().commits == 1
    assert built().closed
    assert es.bulk_calls[0][1]["raw"] == "raw"
    assert es.bulk_calls[0][0]["index"]["_index"] == "raw-logs-1970.01.01"


def test_archive_consumer_redacts_raw_and_uses_kafka_offset_document_id() -> None:
    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return ok_bulk()

    scripted(
        [
            {
                "topic-partition": [
                    message(
                        7,
                        {
                            "raw": "email=alice@example.com password=hunter2",
                            "ingest_ts": 1.0,
                        },
                    )
                ]
            }
        ]
    )
    es = FakeES()
    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=es)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert es.bulk_calls[0][0]["index"]["_id"] == compute_kafka_raw_log_ref("raw-logs", 2, 7)
    assert es.bulk_calls[0][1]["raw"] == "email=<EMAIL> password=<REDACTED>"


def test_archive_consumer_defaults_to_earliest_offsets() -> None:
    class FakeES:
        def bulk(self, body):
            return ok_bulk()

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().kwargs["auto_offset_reset"] == "earliest"


def test_scorer_consumer_uses_configured_offset_reset() -> None:
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host"}],
        kafka_config={"auto_offset_reset": "none"},
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().kwargs["auto_offset_reset"] == "none"


def test_archive_consumer_passes_kafka_security_config() -> None:
    class FakeES:
        def bulk(self, body):
            return ok_bulk()

    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        kafka_config={
            "security": {
                "protocol": "SSL",
                "ssl": {"cafile": "/etc/kafka/ca.pem", "check_hostname": False},
            }
        },
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    fake = built()
    assert fake.kwargs["security_protocol"] == "SSL"
    assert fake.kwargs["ssl_cafile"] == "/etc/kafka/ca.pem"
    assert fake.kwargs["ssl_check_hostname"] is False


def test_consumers_set_max_poll_interval_from_config() -> None:
    """A slow DLQ retry path can exceed kafka-python's 300s default poll interval."""

    class FakeES:
        def bulk(self, body):
            return ok_bulk()

    kafka_config = {"max_poll_interval_ms": "1200000", "session_timeout_ms": 30000}

    with pytest.raises(KeyboardInterrupt):
        ArchiveConsumer(
            "kafka:29092", "raw-logs", es_client=FakeES(), kafka_config=kafka_config
        ).run()
    archive_kwargs = built().kwargs

    with pytest.raises(KeyboardInterrupt):
        ScorerConsumer(
            "kafka:29092",
            "raw-logs",
            "scored-logs",
            score_fn=lambda batch: [],
            kafka_config=kafka_config,
        ).run()
    scorer_kwargs = built().kwargs

    assert archive_kwargs["max_poll_interval_ms"] == 1200000
    assert archive_kwargs["session_timeout_ms"] == 30000
    assert scorer_kwargs["max_poll_interval_ms"] == 1200000
    assert scorer_kwargs["session_timeout_ms"] == 30000


def test_archive_consumer_defaults_max_poll_interval_when_unset() -> None:
    class FakeES:
        def bulk(self, body):
            return ok_bulk()

    with pytest.raises(KeyboardInterrupt):
        ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES()).run()

    from logfilter.kafka.config import DEFAULT_MAX_POLL_INTERVAL_MS

    assert built().kwargs["max_poll_interval_ms"] == DEFAULT_MAX_POLL_INTERVAL_MS


# ------------------------------------------------- transient: never DLQ, never commit


@pytest.mark.parametrize("status", [429, 500, 503])
def test_archive_consumer_keeps_batch_on_transient_es_status(status) -> None:
    """A down/throttled ES must not cost us the whole polled batch."""
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            raise es_status_error(status)

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert dlq.dlq_sent == []
    assert built().commits == 0


def test_archive_consumer_keeps_batch_on_es_transport_error() -> None:
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            raise es_exceptions.ConnectionError("no route to host")

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert dlq.dlq_sent == []
    assert built().commits == 0


def test_archive_consumer_keeps_batch_when_breaker_is_open() -> None:
    """CircuitBreakerOpen must not be mistaken for a per-record failure."""
    from logfilter.utils.circuit_breaker import CircuitBreakerOpen

    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])
    breaker = CircuitBreaker("es", failure_threshold=1, recovery_timeout=600.0)
    with pytest.raises(Exception):
        breaker.call(lambda: (_ for _ in ()).throw(es_status_error(500)))

    class FakeES:
        def bulk(self, body):
            raise AssertionError("ES must not be called while the breaker is open")

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        es_breaker=breaker,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert breaker.state.name == "OPEN"
    assert dlq.dlq_sent == []
    assert built().commits == 0
    assert CircuitBreakerOpen is not None


def test_archive_consumer_backs_off_instead_of_spinning_on_transient_failure() -> None:
    """A paused loop is the difference between retrying and a hot spin."""
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}] * 3)

    class FakeES:
        def bulk(self, body):
            raise httpx.ConnectTimeout("timed out")

    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, FakeDLQProducer()),
    )
    waits: list[float] = []
    consumer._stop_event = _RecordingStopEvent(waits)

    # The autouse fixture zeroes the backoff; this test needs a real pause.
    consumer_module._TRANSIENT_BACKOFF_SECONDS = 5.0
    try:
        with pytest.raises(KeyboardInterrupt):
            consumer.run()
    finally:
        consumer_module._TRANSIENT_BACKOFF_SECONDS = 0.0

    assert len(waits) == 3, "every transient round must pause before the next poll"
    assert all(w == 5.0 for w in waits)



def test_scorer_consumer_keeps_batch_on_transient_api_failure() -> None:
    scripted([{"tp": [message(1, {"raw": "a"})]}])

    def score(batch):
        raise httpx.ConnectError("scoring api down")

    dlq = FakeDLQProducer()
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=score,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert dlq.dlq_sent == []
    assert built().commits == 0


def test_archive_consumer_keeps_batch_when_bulk_reports_errors_without_items() -> None:
    """``errors: true`` with no per-item detail cannot be attributed to records."""
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            return {"errors": True}

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert dlq.dlq_sent == []
    assert built().commits == 0


# ----------------------------------------- partial bulk failure: per-record handling


def test_archive_consumer_archives_499_and_dlqs_1_of_500() -> None:
    """One bad document must not discard the 499 good ones in the same batch."""
    records = [message(i, {"raw": f"log-{i}", "ingest_ts": 1.0}) for i in range(500)]
    scripted([{"tp": records}])

    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            items = [{"index": {"status": 201}} for _ in range(500)]
            items[317] = {"index": {"status": 400, "error": {"type": "mapper_parsing"}}}
            return {"errors": True, "items": items}

    es = FakeES()
    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=es,
        batch_size=500,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert len(es.bulk_calls) == 1
    assert len(es.bulk_calls[0]) == 1000, "all 500 documents were submitted to ES"
    assert len(dlq.dlq_sent) == 1
    assert dlq.dlq_sent[0]["original_message"]["raw"] == "log-317"
    assert built().commits == 1, "the other 499 are safely archived, so commit"


def test_archive_consumer_dlqs_each_rejected_document_with_its_retry_count() -> None:
    scripted(
        [
            {
                "tp": [
                    message(1, {"raw": "a", "ingest_ts": 1.0, "_dlq_replay_count": 2}),
                    message(2, {"raw": "b", "ingest_ts": 1.0}),
                ]
            }
        ]
    )

    class FakeES:
        def bulk(self, body):
            return {
                "errors": True,
                "items": [
                    {"index": {"status": 429}},
                    {"index": {"status": 201}},
                ],
            }

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert [d["original_message"]["raw"] for d in dlq.dlq_sent] == ["a"]
    assert dlq.dlq_sent[0]["retry_count"] == 3
    assert built().commits == 1


def test_archive_consumer_trips_breaker_on_sustained_partial_failures() -> None:
    """The partial failure must reach the breaker, which already saw a success."""
    rounds = [{"tp": [message(i, {"raw": f"log-{i}", "ingest_ts": 1.0})]} for i in range(6)]
    scripted(rounds)

    class FakeES:
        def bulk(self, body):
            return {
                "errors": True,
                "items": [{"index": {"status": 400}} for _ in range(len(body) // 2)],
            }

    breaker = CircuitBreaker("es", failure_threshold=3)
    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        batch_size=1,
        es_breaker=breaker,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert breaker.state.name == "OPEN", "a stream of partial failures must open the breaker"


def test_archive_consumer_does_not_commit_when_dlq_publish_fails() -> None:
    """A failed DLQ publish leaves the record only on the topic — never commit."""
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            return {"errors": True, "items": [{"index": {"status": 400}}]}

    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, FakeDLQProducer(fail=True)),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().commits == 0


def test_archive_consumer_dlqs_whole_batch_on_permanent_bulk_rejection() -> None:
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            raise es_exceptions.BadRequestError("malformed bulk request")

    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=FakeES(),
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert len(dlq.dlq_sent) == 1
    assert dlq.dlq_sent[0]["original_message"]["raw"] == "a"
    assert built().commits == 1


def test_archive_consumer_keeps_batch_when_no_dlq_producer_configured() -> None:
    scripted([{"tp": [message(1, {"raw": "a", "ingest_ts": 1.0})]}])

    class FakeES:
        def bulk(self, body):
            return {"errors": True, "items": [{"index": {"status": 400}}]}

    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=FakeES())

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().commits == 0


def test_archive_consumer_reuses_replayed_raw_log_ref_as_document_id() -> None:
    """A DLQ replay lands at a new offset; the ES document id must not change."""
    replayed_ref = compute_kafka_raw_log_ref("raw-logs", 0, 5)
    scripted([{"tp": [message(99, {"raw": "a", "ingest_ts": 1.0, "raw_log_ref": replayed_ref})]}])

    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return ok_bulk()

    es = FakeES()
    consumer = ArchiveConsumer("kafka:29092", "raw-logs", es_client=es)

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert es.bulk_calls[0][0]["index"]["_id"] == replayed_ref


# ------------------------------------------------------------------ poison messages


def test_archive_consumer_dlqs_undecodable_message_and_keeps_polling() -> None:
    """A poison message must not crash the loop into an endless restart cycle."""
    scripted(
        [
            {"tp": [message(1, None), message(2, {"raw": "ok", "ingest_ts": 1.0})]},
            {"tp": [message(3, {"raw": "second round", "ingest_ts": 1.0})]},
        ]
    )

    class FakeES:
        def __init__(self) -> None:
            self.bulk_calls = []

        def bulk(self, body):
            self.bulk_calls.append(body)
            return {"errors": False, "items": [{"index": {"status": 201}}] * (len(body) // 2)}

    es = FakeES()
    dlq = FakeDLQProducer()
    consumer = ArchiveConsumer(
        "kafka:29092",
        "raw-logs",
        es_client=es,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert len(es.bulk_calls) == 2, "the loop kept processing after the poison message"
    assert len(dlq.dlq_sent) == 1
    assert built().commits == 2, "both rounds committed, so the poison is not re-read"


def test_scorer_consumer_dlqs_undecodable_message_and_scores_the_rest() -> None:
    scripted([{"tp": [message(1, None), message(2, {"raw": "ok"})]}])
    seen: list[list[dict]] = []

    def score(batch):
        seen.append(batch)
        return [{"host": "h"} for _ in batch]

    dlq = FakeDLQProducer()
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=score,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert [event["raw"] for event in seen[0]] == ["ok"]
    assert len(dlq.dlq_sent) == 1
    assert built().commits == 1


def test_decode_kafka_value_tolerates_bad_json() -> None:
    from logfilter.kafka.config import decode_kafka_value

    assert decode_kafka_value(b"not json") is None
    assert decode_kafka_value(b"\xff\xfe") is None
    assert decode_kafka_value(None) is None
    assert decode_kafka_value(b'{"raw": "x"}') == {"raw": "x"}


# ------------------------------------------------------------------- scorer success


def test_scorer_consumer_scores_publishes_and_commits() -> None:
    class FakeProducer:
        def __init__(self) -> None:
            self.sent = []
            self.flushed = False

        def send(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def flush(self, timeout: int) -> None:
            self.flushed = True

    scripted([{"topic-partition": [message(1, {"raw": "raw", "host": "host"})]}])
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

    assert built().commits == 1
    assert built().closed
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

    fake = built()
    assert fake.kwargs["security_protocol"] == "SASL_PLAINTEXT"
    assert fake.kwargs["sasl_mechanism"] == "SCRAM-SHA-256"
    assert fake.kwargs["sasl_plain_username"] == "logfilter"
    assert fake.kwargs["sasl_plain_password"] == "secret"


def test_scorer_consumer_dlqs_batch_on_permanent_score_failure() -> None:
    scripted([{"tp": [message(1, {"raw": "a"}), message(2, {"raw": "b"})]}])

    def fail(batch):
        raise ValueError("schema violation in event")

    dlq = FakeDLQProducer()
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=fail,
        dlq_producer=cast(LogProducer, dlq),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert [d["original_message"]["raw"] for d in dlq.dlq_sent] == ["a", "b"]
    assert "schema violation" in dlq.dlq_sent[0]["error"]
    assert dlq.dlq_sent[0]["original_topic"] == "raw-logs"
    assert dlq.dlq_sent[0]["retry_count"] == 1
    assert built().commits == 1


def test_scorer_consumer_does_not_commit_when_dlq_publish_fails() -> None:
    scripted([{"tp": [message(1, {"raw": "a"})]}])

    def fail(batch):
        raise ValueError("permanent")

    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=fail,
        dlq_producer=cast(LogProducer, FakeDLQProducer(fail=True)),
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().commits == 0


def test_scorer_consumer_no_flush_without_producer() -> None:
    scripted([{"tp": [message(1, {"raw": "raw", "host": "host"})]}])
    consumer = ScorerConsumer(
        "kafka:29092",
        "raw-logs",
        "scored-logs",
        score_fn=lambda batch: [{"host": "host"}],
    )

    with pytest.raises(KeyboardInterrupt):
        consumer.run()

    assert built().commits == 1
