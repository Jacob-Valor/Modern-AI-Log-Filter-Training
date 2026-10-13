"""Tests for archive consumer entrypoint wiring."""

from __future__ import annotations

import pytest

from logfilter import archive_consumer


def test_archive_consumer_main_requires_password(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_consumer,
        "load_config",
        lambda: {"elasticsearch": {"password": ""}},
    )

    with pytest.raises(SystemExit, match="ES password is unset"):
        archive_consumer.main()


def test_archive_consumer_main_requires_a_dlq_topic(monkeypatch) -> None:
    """Without a DLQ the archiver has nowhere to put a permanently bad record."""

    class FakeArchive:
        def __init__(self, **kwargs) -> None:
            self.client = object()

    monkeypatch.setattr(
        archive_consumer,
        "load_config",
        lambda: {
            "kafka": {"topics": {"raw_logs": "raw"}},
            "elasticsearch": {"password": "secret"},
        },
    )
    monkeypatch.setattr(archive_consumer, "LogArchive", FakeArchive)

    with pytest.raises(SystemExit, match="DLQ topic is unset"):
        archive_consumer.main()


def test_archive_consumer_main_wires_archive_consumer(monkeypatch) -> None:
    calls: dict = {}

    class FakeArchive:
        def __init__(self, **kwargs) -> None:
            calls["archive_kwargs"] = kwargs
            self.client = object()

    class FakeConsumer:
        def __init__(self, **kwargs) -> None:
            calls["consumer_kwargs"] = kwargs

        def run(self) -> None:
            calls["run"] = True

    class FakeProducer:
        instances: list[FakeProducer] = []

        def __init__(self, **kwargs) -> None:
            calls["producer_kwargs"] = kwargs
            FakeProducer.instances.append(self)

    FakeProducer.instances = []
    monkeypatch.setattr(
        archive_consumer,
        "load_config",
        lambda: {
            "kafka": {
                "bootstrap_servers": "kafka:29092",
                "topics": {"raw_logs": "raw", "dlq": "logfilter-dlq"},
                "max_poll_records": 33,
                "security": {"protocol": "PLAINTEXT"},
            },
            "elasticsearch": {
                "hosts": ["http://es:9200"],
                "index_prefix": "logs",
                "username": "elastic",
                "password": "secret",
                "index_shards": 2,
                "index_replicas": 1,
            },
        },
    )
    monkeypatch.setattr(archive_consumer, "LogArchive", FakeArchive)
    monkeypatch.setattr(archive_consumer, "ArchiveConsumer", FakeConsumer)
    monkeypatch.setattr(archive_consumer, "LogProducer", FakeProducer)

    archive_consumer.main()

    assert calls["archive_kwargs"]["password"] == "secret"
    assert calls["consumer_kwargs"]["raw_topic"] == "raw"
    assert calls["consumer_kwargs"]["batch_size"] == 33
    assert calls["consumer_kwargs"]["kafka_config"]["security"]["protocol"] == "PLAINTEXT"
    assert calls["producer_kwargs"]["dlq_topic"] == "logfilter-dlq"
    assert calls["consumer_kwargs"]["dlq_producer"] is FakeProducer.instances[0]
    assert calls["run"] is True
