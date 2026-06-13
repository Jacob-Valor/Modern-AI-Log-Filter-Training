"""Tests for Elasticsearch archive wrapper behavior."""

from __future__ import annotations

import pytest

from logfilter.pipeline import archive as archive_module
from logfilter.pipeline.archive import LogArchive, compute_kafka_raw_log_ref, compute_raw_log_ref
from logfilter.security.redaction import RedactionConfig


class FakeIndices:
    def __init__(self) -> None:
        self.templates: list[dict] = []

    def put_index_template(self, **kwargs) -> None:
        self.templates.append(kwargs)


class FakeElasticsearch:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.indices = FakeIndices()
        self.indexed: list[dict] = []
        self.get_calls: list[dict] = []
        self.raise_get = False
        self.raise_health = False

    def index(self, **kwargs) -> dict:
        self.indexed.append(kwargs)
        return {"_id": "doc-1"}

    def get(self, **kwargs) -> dict:
        self.get_calls.append(kwargs)
        if self.raise_get:
            raise RuntimeError("missing")
        return {"_source": {"raw": "event"}}

    def search(self, **kwargs) -> dict:
        body = kwargs.get("body", {})
        if body.get("query", {}).get("ids"):
            return {"hits": {"hits": []}}
        return {"hits": {"hits": [{"_source": {"raw": "a"}}, {"_source": {"raw": "b"}}]}}

    @property
    def cluster(self):
        parent = self

        class Cluster:
            def health(self) -> dict:
                if parent.raise_health:
                    raise RuntimeError("down")
                return {"status": "green"}

        return Cluster()


@pytest.fixture
def fake_archive(monkeypatch) -> LogArchive:
    monkeypatch.setattr(archive_module, "Elasticsearch", FakeElasticsearch)
    return LogArchive(hosts=["http://es:9200"], username="elastic", password="secret")


def test_archive_requires_password() -> None:
    with pytest.raises(ValueError, match="password is required"):
        LogArchive(password="")


def test_archive_creates_template(fake_archive) -> None:
    assert fake_archive.client.indices.templates[0]["name"] == "raw-logs-template"


def test_archive_write_returns_document_id(fake_archive) -> None:
    doc_id = fake_archive.write("raw", source_type="syslog", host="host", extra={"k": "v"})

    assert doc_id == "doc-1"
    body = fake_archive.client.indexed[0]["body"]
    assert body["raw"] == "raw"
    assert body["k"] == "v"


def test_archive_write_redacts_sensitive_raw_payload(monkeypatch) -> None:
    monkeypatch.setattr(archive_module, "Elasticsearch", FakeElasticsearch)
    archive = LogArchive(hosts=["http://es:9200"], username="elastic", password="secret")

    archive.write(
        "user=alice email=alice@example.com password=hunter2 src=10.0.0.5",
        source_type="syslog",
        host="edge-router-1",
    )

    assert isinstance(archive.client, FakeElasticsearch)
    body = archive.client.indexed[0]["body"]
    assert body["raw"] == "user=alice email=<EMAIL> password=<REDACTED> src=10.0.0.5"


def test_archive_write_can_disable_redaction(monkeypatch) -> None:
    monkeypatch.setattr(archive_module, "Elasticsearch", FakeElasticsearch)
    archive = LogArchive(
        hosts=["http://es:9200"],
        username="elastic",
        password="secret",
        redaction_config=RedactionConfig(enabled=False),
    )

    raw = "email=alice@example.com password=hunter2"
    archive.write(raw, source_type="syslog", host="edge-router-1")

    assert isinstance(archive.client, FakeElasticsearch)
    assert archive.client.indexed[0]["body"]["raw"] == raw


def test_archive_write_bulk_uses_helpers(monkeypatch, fake_archive) -> None:
    calls = []

    def fake_bulk(es_client, actions, **kwargs):
        calls.append((es_client, list(actions), kwargs))
        return 2, []

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk([{"raw": "a"}, {"raw": "b", "host": "h"}])

    assert len(ids) == 2
    assert len(calls[0][1]) == 2
    assert calls[0][1][0]["_id"] == ids[0]
    assert calls[0][1][1]["_source"]["host"] == "h"


def test_archive_write_bulk_returns_deterministic_document_ids(monkeypatch, fake_archive) -> None:
    calls = []

    def fake_bulk(es_client, actions, **kwargs):
        calls.append((es_client, list(actions), kwargs))
        return 1, []

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk(
        [{"raw": "a", "source_type": "syslog", "host": "h", "ingest_ts": 0.0}]
    )

    expected = compute_raw_log_ref("a", "syslog", "h", 0.0)
    assert ids == [expected]
    assert calls[0][1][0]["_id"] == expected


def test_archive_write_bulk_redacts_sensitive_raw_payloads(monkeypatch, fake_archive) -> None:
    calls = []

    def fake_bulk(es_client, actions, **kwargs):
        calls.append((es_client, list(actions), kwargs))
        return 1, []

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    fake_archive.write_bulk([{"raw": "token=AKIAABCDEFGHIJKLMNOP email=bob@example.com"}])

    assert calls[0][1][0]["_source"]["raw"] == "token=<REDACTED> email=<EMAIL>"


def test_archive_get_by_id_returns_source_or_none(fake_archive) -> None:
    assert fake_archive.get_by_id("doc-1") == {"raw": "event"}

    fake_archive.client.raise_get = True
    assert fake_archive.get_by_id("missing") is None


def test_archive_get_by_id_searches_all_rolled_indices_when_index_omitted(fake_archive) -> None:
    search_calls = []
    fake_archive.client.raise_get = True

    def search(**kwargs):
        search_calls.append(kwargs)
        return {"hits": {"hits": [{"_source": {"raw": "rolled"}}]}}

    fake_archive.client.search = search

    assert fake_archive.get_by_id("doc-rolled") == {"raw": "rolled"}
    assert search_calls[0]["index"] == "raw-logs-*"
    assert search_calls[0]["body"]["query"]["ids"]["values"] == ["doc-rolled"]


def test_archive_search_recent_builds_filters(fake_archive) -> None:
    results = fake_archive.search_recent(host="host", source_type="syslog")

    assert results == [{"raw": "a"}, {"raw": "b"}]


def test_archive_health_handles_success_and_failure(fake_archive) -> None:
    assert fake_archive.health() == {"status": "green"}

    fake_archive.client.raise_health = True
    assert fake_archive.health()["status"] == "unavailable"


def test_compute_kafka_raw_log_ref_is_replay_stable_and_offset_specific() -> None:
    same_ref = compute_kafka_raw_log_ref("raw-logs", partition=2, offset=7)

    assert compute_kafka_raw_log_ref("raw-logs", partition=2, offset=7) == same_ref
    assert compute_kafka_raw_log_ref("raw-logs", partition=2, offset=8) != same_ref
    assert compute_kafka_raw_log_ref("raw-logs", partition=3, offset=7) != same_ref


def test_archive_password_none_raises() -> None:
    with pytest.raises(ValueError, match="password is required"):
        LogArchive(password=None)


def test_archive_index_template_exception(monkeypatch) -> None:
    class FailingES:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        class indices:
            @staticmethod
            def put_index_template(**kwargs):
                del kwargs
                raise RuntimeError("template fail")

    monkeypatch.setattr(archive_module, "Elasticsearch", FailingES)
    archive = LogArchive(hosts=["http://es:9200"], password="secret")
    assert archive.client is not None


def test_archive_write_bulk_raises_on_non_conflict_failures(monkeypatch, fake_archive) -> None:
    def fake_bulk(es_client, actions, **kwargs):
        del es_client, actions, kwargs
        return 0, [{"create": {"_id": "x", "status": 503, "error": "unavailable"}}]

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    with pytest.raises(archive_module.BulkArchiveError):
        fake_archive.write_bulk([{"raw": "a"}])


def test_archive_write_bulk_ignores_idempotent_create_conflicts(monkeypatch, fake_archive) -> None:
    def fake_bulk(es_client, actions, **kwargs):
        del es_client, actions, kwargs
        return 0, [{"create": {"_id": "dup", "status": 409, "error": "already exists"}}]

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk([{"raw": "a"}])
    assert len(ids) == 1


def test_archive_write_bulk_raises_when_error_count_is_int(monkeypatch, fake_archive) -> None:
    def fake_bulk(es_client, actions, **kwargs):
        del es_client, actions, kwargs
        return 1, 5

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    with pytest.raises(archive_module.BulkArchiveError):
        fake_archive.write_bulk([{"raw": "a"}])


def test_archive_search_recent_with_filters(fake_archive) -> None:
    results = fake_archive.search_recent(host="host1", source_type="syslog", minutes=30, size=50)
    assert len(results) == 2


def test_archive_search_recent_without_filters(fake_archive) -> None:
    results = fake_archive.search_recent()
    assert len(results) == 2
