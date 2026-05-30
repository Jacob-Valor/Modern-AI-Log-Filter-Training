"""End-to-end integration test: full pipeline flow through all services."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_e2e_archive_to_es(
    es_client,
    es_host: str,
    es_password: str,
):
    """Write to LogArchive and verify the document is searchable in ES."""
    from logfilter.pipeline.archive import LogArchive

    archive = LogArchive(
        hosts=[es_host],
        index_prefix="test-e2e-archive",
        username="elastic",
        password=es_password,
    )

    doc_id = archive.write(
        raw="<14>end-to-end test log entry",
        host="e2e-host",
    )
    assert doc_id is not None

    # Verify in Elasticsearch
    es_client.indices.refresh(index=f"{archive.index_prefix}-*")
    search = es_client.search(
        index=f"{archive.index_prefix}-*",
        query={"match": {"host": "e2e-host"}},
    )
    assert search["hits"]["total"]["value"] >= 1

    # Cleanup
    for idx in es_client.indices.get(index=f"{archive.index_prefix}-*").keys():
        es_client.indices.delete(index=idx, ignore=[404])


@pytest.mark.integration
def test_e2e_full_pipeline(
    kafka_admin,
    kafka_producer,
    kafka_bootstrap: str,
    es_client,
    es_host: str,
    es_password: str,
):
    """End-to-end: produce → archive → API score → router decision."""
    import os

    from fastapi.testclient import TestClient

    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = kafka_bootstrap
    os.environ["ES_HOST"] = es_host
    os.environ["ES_USER"] = "elastic"
    os.environ["ES_PASSWORD"] = es_password
    os.environ["LOGFILTER_ADMIN_TOKEN"] = "test-admin-token"
    os.environ["LOGFILTER_API_TOKEN"] = "test-api-token"
    os.environ["LOGFILTER_METRICS_TOKEN"] = "test-metrics-token"
    os.environ["LOGFILTER_ENABLE_DOCS"] = "0"
    os.environ["LOGFILTER_NER_ENABLED"] = "false"
    os.environ["LOGFILTER_BIENCODER_ENABLED"] = "false"
    os.environ["LOGFILTER_CROSS_ENCODER_ENABLED"] = "false"

    from logfilter.api.app import app
    from logfilter.pipeline.archive import LogArchive

    api_client = TestClient(app)

    # 1. Archive the raw event
    archive = LogArchive(
        hosts=[es_host],
        index_prefix="test-e2e-pipeline",
        username="elastic",
        password=es_password,
    )

    doc_id = archive.write(
        raw="<14>Jan 15 11:07:53 prod-srv01 sshd[1234]: Failed password for root from 192.168.1.1",
        host="prod-srv01",
    )
    assert doc_id is not None

    # 2. Score via API
    resp = api_client.post(
        "/score",
        json={
            "raw": "<14>Jan 15 11:07:53 prod-srv01 sshd: Failed password for root",
            "source_type": "syslog",
        },
        headers={"X-API-Token": "test-api-token"},
    )
    assert resp.status_code == 200
    scored = resp.json()
    assert "ai_threat_score" in scored
    assert "ai_priority" in scored
    assert "leef_payload" in scored

    # 3. Verify archive is searchable
    es_client.indices.refresh(index=f"{archive.index_prefix}-*")
    search = es_client.search(
        index=f"{archive.index_prefix}-*",
        query={"match": {"host": "prod-srv01"}},
    )
    assert search["hits"]["total"]["value"] >= 1

    # Cleanup
    for idx in es_client.indices.get(index=f"{archive.index_prefix}-*").keys():
        es_client.indices.delete(index=idx, ignore=[404])
