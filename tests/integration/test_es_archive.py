"""Integration test: Elasticsearch archive indexing and retrieval."""

from __future__ import annotations

import time
import uuid

import pytest


@pytest.mark.integration
def test_es_index_and_search(es_client, es_host: str):
    """Index a document and retrieve it via search."""
    index = f"test-logs-{uuid.uuid4().hex[:8]}".lower()
    doc = {
        "raw": "<14>integration test event",
        "host": "test-host",
        "timestamp": time.time(),
    }

    # Index
    resp = es_client.index(index=index, document=doc, refresh=True)
    assert resp["result"] in ("created", "updated")

    # Search
    time.sleep(1)
    es_client.indices.refresh(index=index)
    search = es_client.search(index=index, query={"match_all": {}})
    assert search["hits"]["total"]["value"] >= 1
    assert search["hits"]["hits"][0]["_source"]["host"] == "test-host"

    # Cleanup
    es_client.indices.delete(index=index, ignore=[404])


@pytest.mark.integration
def test_log_archive_integration(es_client, es_host: str):
    """Smoke-test LogArchive against the real Elasticsearch instance."""
    from logfilter.pipeline.archive import LogArchive

    archive = LogArchive(
        hosts=[es_host],
        index_prefix="test-raw",
    )

    # Write and retrieve (index template is created automatically on first write)
    doc_id = archive.write(
        raw="integration test log line",
        host="int-test",
    )
    assert doc_id is not None

    # Refresh index so the doc is searchable immediately
    es_client.indices.refresh(index=f"{archive.index_prefix}-*")

    # Search recent
    results = archive.search_recent(host="int-test", minutes=1)
    assert len(results) >= 1
    assert results[0]["host"] == "int-test"

    # Get by ID
    source = archive.get_by_id(doc_id)
    assert source is not None
    assert source["host"] == "int-test"

    # Cleanup
    for idx in es_client.indices.get(index=f"{archive.index_prefix}-*").keys():
        es_client.indices.delete(index=idx, ignore=[404])
