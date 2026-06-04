"""Tests for Elasticsearch archive wrapper behavior."""

from __future__ import annotations

import pytest

from logfilter.pipeline import archive as archive_module
from logfilter.pipeline.archive import LogArchive


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
        self.raise_get = False
        self.raise_health = False

    def index(self, **kwargs) -> dict:
        self.indexed.append(kwargs)
        return {"_id": "doc-1"}

    def get(self, **kwargs) -> dict:
        if self.raise_get:
            raise RuntimeError("missing")
        return {"_source": {"raw": "event"}}

    def search(self, **kwargs) -> dict:
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


def test_archive_write_bulk_uses_helpers(monkeypatch, fake_archive) -> None:
    calls = []

    def fake_bulk(es_client, actions, **kwargs):
        calls.append((es_client, list(actions), kwargs))
        return 2, [{"error": "bad"}]

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk([{"raw": "a"}, {"raw": "b", "host": "h"}])

    assert ids == []
    assert len(calls[0][1]) == 2
    assert calls[0][1][1]["_source"]["host"] == "h"


def test_archive_get_by_id_returns_source_or_none(fake_archive) -> None:
    assert fake_archive.get_by_id("doc-1") == {"raw": "event"}

    fake_archive.client.raise_get = True
    assert fake_archive.get_by_id("missing") is None


def test_archive_search_recent_builds_filters(fake_archive) -> None:
    results = fake_archive.search_recent(host="host", source_type="syslog")

    assert results == [{"raw": "a"}, {"raw": "b"}]


def test_archive_health_handles_success_and_failure(fake_archive) -> None:
    assert fake_archive.health() == {"status": "green"}

    fake_archive.client.raise_health = True
    assert fake_archive.health()["status"] == "unavailable"


def test_archive_password_none_raises() -> None:
    with pytest.raises(ValueError, match="password is required"):
        LogArchive(password=None)


def test_archive_index_template_exception(monkeypatch) -> None:
    class FailingES:
        def __init__(self, *args, **kwargs) -> None:
            pass

        class indices:
            @staticmethod
            def put_index_template(**kwargs):
                raise RuntimeError("template fail")

    monkeypatch.setattr(archive_module, "Elasticsearch", FailingES)
    archive = LogArchive(hosts=["http://es:9200"], password="secret")
    assert archive.client is not None


def test_archive_write_bulk_with_errors(monkeypatch, fake_archive) -> None:
    def fake_bulk(es_client, actions, **kwargs):
        return 1, [{"error": "bad"}]

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk([{"raw": "a"}])
    assert ids == []


def test_archive_write_bulk_error_count_as_int(monkeypatch, fake_archive) -> None:
    def fake_bulk(es_client, actions, **kwargs):
        return 1, 5

    monkeypatch.setattr(archive_module.helpers, "bulk", fake_bulk)

    ids = fake_archive.write_bulk([{"raw": "a"}])
    assert ids == []


def test_archive_search_recent_with_filters(fake_archive) -> None:
    results = fake_archive.search_recent(host="host1", source_type="syslog", minutes=30, size=50)
    assert len(results) == 2


def test_archive_search_recent_without_filters(fake_archive) -> None:
    results = fake_archive.search_recent()
    assert len(results) == 2
