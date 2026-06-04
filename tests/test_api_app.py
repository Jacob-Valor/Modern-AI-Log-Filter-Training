"""Tests for FastAPI app helpers without loading external ML models."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from logfilter.api import app as api_app
from logfilter.api.schemas import BatchScoreRequest, ScoreRequest
from logfilter.pipeline.normalizer import LogSourceType
from logfilter.pipeline.scorer import ScoredEvent


def _request(host: str = "198.51.100.10") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": [],
            "client": (host, 54123),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


@pytest.fixture(autouse=True)
def reset_app_state(monkeypatch):
    original_config = api_app._state.config
    original_scorer = api_app._state.scorer
    original_enricher = api_app._state.enricher
    original_windows = api_app._state.rate_limit_windows
    original_redis_client = api_app._state.redis_client
    original_rate_limiter = api_app._state.rate_limiter

    monkeypatch.delenv("LOGFILTER_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("LOGFILTER_API_TOKEN", raising=False)
    monkeypatch.delenv("LOGFILTER_METRICS_TOKEN", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    api_app._state.config = {"api": {}}
    api_app._state.scorer = None
    api_app._state.enricher = None
    api_app._state.rate_limit_windows = {}
    api_app._state.redis_client = None
    api_app._state.rate_limiter = None

    yield

    api_app._state.config = original_config
    api_app._state.scorer = original_scorer
    api_app._state.enricher = original_enricher
    api_app._state.rate_limit_windows = original_windows
    api_app._state.redis_client = original_redis_client
    api_app._state.rate_limiter = original_rate_limiter


def test_source_hint_handles_valid_invalid_and_missing_values() -> None:
    assert api_app._source_hint("SYSLOG") == LogSourceType.SYSLOG
    assert api_app._source_hint("not-real") is None
    assert api_app._source_hint(None) is None


def test_env_enabled_parses_true_values(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_ENABLE_DOCS", "yes")

    assert api_app._env_enabled("LOGFILTER_ENABLE_DOCS")
    assert not api_app._env_enabled("MISSING_FLAG")


def test_extract_bearer_token() -> None:
    assert api_app._extract_bearer_token("Bearer abc") == "abc"
    assert api_app._extract_bearer_token("Basic abc") is None
    assert api_app._extract_bearer_token(None) is None


def test_require_admin_fails_closed_and_accepts_valid_token() -> None:
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_app._require_admin(x_admin_token="token"))
    assert missing.value.status_code == 403

    api_app._state.config = {"api": {"admin_token": "secret"}}
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(api_app._require_admin(x_admin_token="wrong"))
    assert invalid.value.status_code == 401

    asyncio.run(api_app._require_admin(x_admin_token="secret"))


def test_require_metrics_access_supports_optional_token_and_bearer() -> None:
    asyncio.run(api_app._require_metrics_access())

    api_app._state.config = {"api": {"metrics_token": "metrics"}}
    asyncio.run(
        api_app._require_metrics_access(x_metrics_token=None, authorization="Bearer metrics")
    )

    with pytest.raises(HTTPException) as invalid:
        asyncio.run(api_app._require_metrics_access(x_metrics_token="wrong"))
    assert invalid.value.status_code == 401


def test_scoring_access_rate_limit() -> None:
    api_app._state.config = {
        "api": {
            "scoring_token": "secret",
            "rate_limit_per_minute": 1,
        }
    }

    asyncio.run(api_app._require_scoring_access(_request(), x_api_token="secret"))
    with pytest.raises(HTTPException) as limited:
        asyncio.run(api_app._require_scoring_access(_request(), x_api_token="secret"))

    assert limited.value.status_code == 429


def test_configure_rate_limiter_enables_redis_backend(monkeypatch) -> None:
    client = Mock()
    client.ping.return_value = True
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_module = Mock()
    fake_module.Redis.from_url.return_value = client
    monkeypatch.setattr(api_app.importlib, "import_module", Mock(return_value=fake_module))

    api_app._configure_rate_limiter()

    assert api_app._state.redis_client is client
    assert api_app._state.rate_limiter is not None


def test_configure_rate_limiter_falls_back_when_redis_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    fake_module = Mock()
    fake_module.Redis.from_url.side_effect = ConnectionError("boom")
    monkeypatch.setattr(api_app.importlib, "import_module", Mock(return_value=fake_module))

    api_app._configure_rate_limiter()

    assert api_app._state.redis_client is None
    assert api_app._state.rate_limiter is None


def test_health_reports_degraded_without_scorer() -> None:
    response = asyncio.run(api_app.health())

    assert response.status == "degraded"
    assert response.models_loaded["scorer"] is False


def test_health_reports_classifier_state_when_ready() -> None:
    class FakeClassifier:
        def is_ready(self) -> bool:
            return True

    class FakeScorer:
        classifier = FakeClassifier()

    api_app._state.scorer = cast(Any, FakeScorer())
    api_app._state.enricher = cast(Any, object())

    response = asyncio.run(api_app.health())

    assert response.status == "healthy"
    assert response.models_loaded["classifier"] is True


def test_health_drift_returns_503_when_detector_not_configured() -> None:
    api_app._state.scorer = None

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.health_drift())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Drift detector not configured"


def test_health_drift_returns_503_when_scorer_has_no_detector() -> None:
    class FakeScorer:
        drift_detector = None

    api_app._state.scorer = cast(Any, FakeScorer())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.health_drift())

    assert exc_info.value.status_code == 503


def test_health_drift_reports_status_when_configured() -> None:
    class FakeStatus:
        drift_detected = True
        psi = 0.123456
        reference_count = 1000
        current_count = 250
        fallback_active = True

    class FakeDetector:
        def check_drift(self) -> FakeStatus:
            return FakeStatus()

    class FakeScorer:
        drift_detector = FakeDetector()

    api_app._state.scorer = cast(Any, FakeScorer())

    response = asyncio.run(api_app.health_drift())

    assert response.drift_detected is True
    assert response.psi == 0.1235
    assert response.reference_count == 1000
    assert response.current_count == 250
    assert response.fallback_active is True


def test_metrics_snapshot_uses_counters() -> None:
    api_app._state.events_scored = 2
    api_app._state.events_high = 1
    api_app._state.events_duplicate = 1
    api_app._state.events_sigma = 1
    api_app._state.latency_sum = 10.0
    api_app._state.score_sum = 1.5

    snapshot = asyncio.run(api_app.metrics_snapshot())

    assert snapshot.events_scored_total == 2
    assert snapshot.avg_latency_ms == 5.0
    assert snapshot.avg_threat_score == 0.75


def test_security_headers_are_added_to_responses() -> None:
    async def call_next(request: Request) -> Response:
        return Response("ok")

    response = asyncio.run(api_app.add_security_headers(_request(), call_next))

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def _scored(priority: str = "HIGH", score: float = 0.9) -> ScoredEvent:
    return ScoredEvent(
        source_type="syslog",
        timestamp="2026-01-15T11:07:53Z",
        host="prod",
        raw="raw",
        normalized_text="normalized",
        ai_threat_score=score,
        ai_priority=priority,
        ai_mitre_technique="T1110",
        ai_entities="10.0.0.5",
        ai_confidence=0.8,
        sigma_matched=True,
        is_duplicate=False,
        dedup_similarity=0.0,
        entities={
            "indicators": ["10.0.0.5"],
            "malware": [],
            "vulnerabilities": [],
            "organizations": [],
            "systems": [],
            "confidence": 0.9,
            "has_high_value_entities": True,
        },
        cross_encoder_scores=[{"id": "T1110", "name": "Brute Force", "score": 0.7}],
        classifier_score=0.6,
        entity_boost=0.2,
        cross_encoder_max=0.7,
        scoring_latency_ms=12.0,
    )


class FakeScorer:
    class Classifier:
        def is_ready(self) -> bool:
            return True

    classifier = Classifier()

    def score(self, normalized):
        if normalized.raw == "bad":
            raise ValueError("bad event")
        return _scored()

    def score_batch(self, normalized_events):
        return [_scored("HIGH", 0.9), _scored("MEDIUM", 0.6)][: len(normalized_events)]


class FakeEnricher:
    def enrich(self, scored):
        return "leef"

    def enrich_batch(self, scored_events):
        return [f"leef-{i}" for i, _ in enumerate(scored_events)]


def test_score_event_success_and_validation_error() -> None:
    api_app._state.scorer = cast(Any, FakeScorer())
    api_app._state.enricher = cast(Any, FakeEnricher())

    response = asyncio.run(
        api_app.score_event(ScoreRequest(raw="Jan 15 host sshd: Failed password"))
    )

    assert response.ai_priority == "HIGH"
    assert response.attack_matches[0].technique_id == "T1110"
    assert response.leef_payload == "leef"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_event(ScoreRequest(raw="bad")))
    assert exc_info.value.status_code == 400


def test_score_event_requires_initialized_service() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_event(ScoreRequest(raw="raw")))

    assert exc_info.value.status_code == 503


def test_score_batch_success_and_configured_limit() -> None:
    api_app._state.scorer = cast(Any, FakeScorer())
    api_app._state.enricher = cast(Any, FakeEnricher())
    api_app._state.config = {"api": {"max_batch_size": 1}}

    too_large = BatchScoreRequest(
        events=[
            ScoreRequest(raw="Jan 15 host sshd: Failed password"),
            ScoreRequest(raw="Jan 15 host sshd: Accepted password"),
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_batch(too_large))
    assert exc_info.value.status_code == 413

    api_app._state.config = {"api": {"max_batch_size": 2}}
    response = asyncio.run(api_app.score_batch(too_large))

    assert response.total == 2
    assert response.high_priority_count == 1
    assert response.medium_priority_count == 1


def test_score_batch_requires_initialized_service() -> None:
    payload = BatchScoreRequest(events=[ScoreRequest(raw="raw")])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_batch(payload))

    assert exc_info.value.status_code == 503


def test_reload_models_requires_scorer_and_replaces_it(monkeypatch) -> None:
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_app.reload_models())
    assert missing.value.status_code == 503

    api_app._state.scorer = cast(Any, FakeScorer())
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda path: {"scoring": {"routing": {"high": "0.90"}}},
    )
    monkeypatch.setattr(api_app, "LogScorer", lambda config: "new-scorer")

    response = asyncio.run(api_app.reload_models())

    assert response == {"status": "reloaded"}
    assert api_app._state.config == {"scoring": {"routing": {"high": "0.90"}}}
    assert api_app._state.scorer == "new-scorer"


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    response = asyncio.run(api_app.metrics())

    assert response.media_type == api_app.CONTENT_TYPE_LATEST
    assert b"logfilter_events_total" in response.body


def test_lifespan_initializes_scorer_and_enricher(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "load_config", lambda path: {"qradar": {"leef_vendor": "Vendor"}})
    monkeypatch.setattr(api_app, "LogScorer", lambda config, model_version="": "scorer")
    monkeypatch.setattr(api_app, "LEEFEnricher", lambda **kwargs: ("enricher", kwargs))

    async def run_lifespan() -> None:
        async with api_app.lifespan(api_app.app):
            assert api_app._state.config == {"qradar": {"leef_vendor": "Vendor"}}
            assert api_app._state.scorer == "scorer"
            assert cast(tuple[Any, dict[str, str]], api_app._state.enricher)[0] == "enricher"

    asyncio.run(run_lifespan())


def test_env_enabled_with_default() -> None:
    assert api_app._env_enabled("MISSING_VAR", default="0") is False
    assert api_app._env_enabled("MISSING_VAR", default="1") is True


def test_score_event_with_none_scorer() -> None:
    api_app._state.scorer = None
    api_app._state.enricher = cast(Any, FakeEnricher())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_event(ScoreRequest(raw="test")))
    assert exc_info.value.status_code == 503


def test_score_event_exception_handling() -> None:
    class BrokenScorer:
        class Classifier:
            def is_ready(self) -> bool:
                return True

        classifier = Classifier()

        def score(self, normalized):
            raise ValueError("bad input")

    api_app._state.scorer = cast(Any, BrokenScorer())
    api_app._state.enricher = cast(Any, FakeEnricher())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_event(ScoreRequest(raw="test")))
    assert exc_info.value.status_code == 400


def test_score_batch_empty_events() -> None:
    api_app._state.scorer = cast(Any, FakeScorer())
    api_app._state.enricher = cast(Any, FakeEnricher())

    class FakeEmptyBatch:
        events = []

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app.score_batch(FakeEmptyBatch()))  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400


def test_score_batch_exception_handling() -> None:
    class BrokenScorer:
        class Classifier:
            def is_ready(self) -> bool:
                return True

        classifier = Classifier()

        def score_batch(self, normalized_events):
            raise ValueError("bad batch")

    api_app._state.scorer = cast(Any, BrokenScorer())
    api_app._state.enricher = cast(Any, FakeEnricher())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api_app.score_batch(
                BatchScoreRequest(events=[ScoreRequest(raw="test")])
            )
        )
    assert exc_info.value.status_code == 400


def test_health_with_tier2_classifier(monkeypatch) -> None:
    class FakeTier2:
        def is_ready(self) -> bool:
            return True

    class FakeScorerWithTier2:
        class Classifier:
            def is_ready(self) -> bool:
                return True

        classifier = Classifier()
        tier2_classifier = FakeTier2()

    api_app._state.scorer = cast(Any, FakeScorerWithTier2())
    api_app._state.enricher = cast(Any, object())

    response = asyncio.run(api_app.health())
    assert response.models_loaded["tier2_classifier"] is True


def test_reload_models_with_empty_config(monkeypatch) -> None:
    api_app._state.scorer = cast(Any, FakeScorer())
    monkeypatch.setattr(api_app, "load_config", lambda path: {})
    monkeypatch.setattr(api_app, "LogScorer", lambda config, model_version="": "new-scorer")

    response = asyncio.run(api_app.reload_models())
    assert response == {"status": "reloaded"}


def test_metrics_snapshot_with_zero_events() -> None:
    api_app._state.events_scored = 0
    api_app._state.events_high = 0
    api_app._state.events_duplicate = 0
    api_app._state.events_sigma = 0
    api_app._state.latency_sum = 0.0
    api_app._state.score_sum = 0.0

    snapshot = asyncio.run(api_app.metrics_snapshot())
    assert snapshot.events_scored_total == 0
    assert snapshot.avg_latency_ms == 0.0
    assert snapshot.avg_threat_score == 0.0


def test_lifespan_with_empty_config(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "load_config", lambda path: {})
    monkeypatch.setattr(api_app, "LogScorer", lambda config, model_version="": "scorer")
    monkeypatch.setattr(api_app, "LEEFEnricher", lambda **kwargs: ("enricher", kwargs))

    async def run_lifespan() -> None:
        async with api_app.lifespan(api_app.app):
            assert api_app._state.config == {}

    asyncio.run(run_lifespan())


def test_require_metrics_access_with_env_token(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_METRICS_TOKEN", "metrics-secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_app._require_metrics_access(x_metrics_token="wrong"))
    assert exc_info.value.status_code == 401

    asyncio.run(api_app._require_metrics_access(x_metrics_token="metrics-secret"))


def test_require_scoring_access_with_env_token(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_API_TOKEN", "api-secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api_app._require_scoring_access(_request(), x_api_token="wrong")
        )
    assert exc_info.value.status_code == 401

    asyncio.run(api_app._require_scoring_access(_request(), x_api_token="api-secret"))
