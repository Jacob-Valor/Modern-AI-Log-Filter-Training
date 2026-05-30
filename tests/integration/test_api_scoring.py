"""Integration test: full HTTP scoring round-trip against the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(kafka_bootstrap: str, es_host: str, es_password: str):
    """Build a TestClient against the real API with test-service env."""
    import os

    # Point the API at our integration test services
    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = kafka_bootstrap
    os.environ["ES_HOST"] = es_host
    os.environ["ES_USER"] = "elastic"
    os.environ["ES_PASSWORD"] = es_password
    os.environ["LOGFILTER_ADMIN_TOKEN"] = "test-admin-token"
    os.environ["LOGFILTER_API_TOKEN"] = "test-api-token"
    os.environ["LOGFILTER_METRICS_TOKEN"] = "test-metrics-token"
    os.environ["LOGFILTER_ENABLE_DOCS"] = "0"
    # Disable heavy transformer models for faster integration test startup
    os.environ["LOGFILTER_NER_ENABLED"] = "false"
    os.environ["LOGFILTER_BIENCODER_ENABLED"] = "false"
    os.environ["LOGFILTER_CROSS_ENCODER_ENABLED"] = "false"

    from logfilter.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_api_health_check(api_client):
    """The health endpoint returns 200 and a status payload."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in ("healthy", "degraded")


@pytest.mark.integration
def test_api_score_event_requires_token(api_client):
    """/score rejects requests without the API token."""
    resp = api_client.post("/score", json={"raw": "test event"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_api_score_event_with_token(api_client):
    """/score accepts valid events and returns a scored result."""
    resp = api_client.post(
        "/score",
        json={
            "raw": "<14>Jan 15 11:07:53 prod-srv01 sshd: Accepted publickey for root",
            "source_type": "syslog",
        },
        headers={"X-API-Token": "test-api-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ai_threat_score" in data
    assert "ai_priority" in data
    assert 0.0 <= data["ai_threat_score"] <= 1.0


@pytest.mark.integration
def test_api_score_batch_with_token(api_client):
    """/score/batch accepts multiple events."""
    payload = {
        "events": [
            {"raw": f"test event {i}", "source_type": "syslog"}
            for i in range(3)
        ]
    }
    resp = api_client.post(
        "/score/batch",
        json=payload,
        headers={"X-API-Token": "test-api-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 3
    for item in data["results"]:
        assert "ai_threat_score" in item
        assert "ai_priority" in item


@pytest.mark.integration
def test_api_metrics_prometheus_format(api_client):
    """/metrics returns a Prometheus text payload."""
    resp = api_client.get("/metrics", headers={"X-Metrics-Token": "test-metrics-token"})
    assert resp.status_code == 200
    assert "logfilter_" in resp.text or "# TYPE" in resp.text
