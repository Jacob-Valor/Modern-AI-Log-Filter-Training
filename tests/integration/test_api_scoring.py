"""Integration test: full HTTP scoring round-trip against the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch, kafka_bootstrap: str, es_host: str):
    """Build a TestClient against the real API with test-service env.

    ``monkeypatch.setenv`` is restored after each test: the app resolves
    ``${VAR:default}`` from ``os.environ`` on every request, so plain
    ``os.environ[...] = ...`` here would leak the tokens into every later test
    in the session (including any test asserting the unconfigured-403 path).
    """
    # Point the API at our integration test services
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", kafka_bootstrap)
    monkeypatch.setenv("ES_HOST", es_host)
    monkeypatch.setenv("LOGFILTER_API_TOKEN", "test-api-token")
    monkeypatch.setenv("LOGFILTER_METRICS_TOKEN", "test-metrics-token")
    monkeypatch.setenv("LOGFILTER_ENABLE_DOCS", "0")
    # Disable heavy transformer models for faster integration test startup
    monkeypatch.setenv("LOGFILTER_NER_ENABLED", "false")
    monkeypatch.setenv("LOGFILTER_BIENCODER_ENABLED", "false")
    monkeypatch.setenv("LOGFILTER_CROSS_ENCODER_ENABLED", "false")

    from logfilter.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def unconfigured_api_client(monkeypatch, kafka_bootstrap: str, es_host: str):
    """TestClient for an API deployed with no scoring token configured.

    Deleting LOGFILTER_API_TOKEN makes ``${LOGFILTER_API_TOKEN:}`` in
    config/config.yaml resolve to the empty string, which is what a real
    unconfigured deployment looks like — the 403 branch, as opposed to the
    401 branch covered by :func:`test_api_score_event_requires_token`.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", kafka_bootstrap)
    monkeypatch.setenv("ES_HOST", es_host)
    monkeypatch.setenv("LOGFILTER_ENABLE_DOCS", "0")
    monkeypatch.delenv("LOGFILTER_API_TOKEN", raising=False)

    from logfilter.api.app import app

    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_api_health_check(api_client):
    """The health endpoint returns 200 when healthy or 503 when degraded (models missing)."""
    resp = api_client.get("/health")
    # 200 = all models loaded and deps healthy; 503 = degraded (e.g. no model artifacts on CI)
    assert resp.status_code in (200, 503)
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
    payload = {"events": [{"raw": f"test event {i}", "source_type": "syslog"} for i in range(3)]}
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


@pytest.mark.integration
def test_api_score_event_rejects_when_token_unconfigured(unconfigured_api_client):
    """An unconfigured scoring token must yield 403, not 401 and not 200.

    401 means "wrong token", 403 means "no token configured at all" — the two
    are different operational failures and must not collapse into one.
    """
    resp = unconfigured_api_client.post("/score", json={"raw": "test event"})
    assert resp.status_code == 403, resp.text
    assert "not configured" in resp.json()["detail"]


@pytest.mark.integration
def test_api_score_event_rejects_wrong_token(api_client):
    """A configured token still rejects a wrong one with 401."""
    resp = api_client.post(
        "/score",
        json={"raw": "test event"},
        headers={"X-API-Token": "wrong-token"},
    )
    assert resp.status_code == 401, resp.text
