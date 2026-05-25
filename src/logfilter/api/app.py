"""
FastAPI application — AI Log Filter scoring service.

Endpoints:
  POST /score              — Score a single log event
  POST /score/batch        — Score up to 200 events in one call
  GET  /health             — Liveness + readiness check
  GET  /metrics            — Prometheus metrics (text/plain)
  GET  /metrics/snapshot   — JSON metrics snapshot
  POST /admin/reload       — Trigger model reload (admin only)
"""

from __future__ import annotations

import importlib
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from logfilter import telemetry
from logfilter.api.schemas import (
    ATTACKMatch,
    BatchScoreRequest,
    BatchScoreResponse,
    EntitySummary,
    HealthResponse,
    MetricsSnapshot,
    ScoreRequest,
    ScoreResponse,
)
from logfilter.api.security import AccessDenied, enforce_rate_limit, require_configured_token
from logfilter.config import load_config
from logfilter.pipeline.enricher import LEEFEnricher
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType
from logfilter.pipeline.scorer import LogScorer

logger = structlog.get_logger(__name__)

# ── Config path ────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path("config/config.yaml")
_TRUE_VALUES = {"1", "true", "yes", "on"}

# ── Prometheus metrics ─────────────────────────────────────────────────────────
_events_total = Counter("logfilter_events_total", "Total log events scored", ["priority"])
_events_duplicate = Counter("logfilter_events_duplicate_total", "Duplicate events detected")
_events_sigma = Counter("logfilter_events_sigma_total", "Sigma rule matches")
_score_histogram = Histogram(
    "logfilter_threat_score",
    "Threat score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)
_latency_histogram = Histogram(
    "logfilter_scoring_latency_ms",
    "Scoring latency per event in ms",
    buckets=[5, 10, 20, 50, 100, 200, 500, 1000],
)
_batch_size_histogram = Histogram(
    "logfilter_batch_size",
    "Batch size distribution",
    buckets=[1, 5, 10, 20, 50, 100, 200],
)
_model_loaded = Gauge("logfilter_model_loaded", "Whether model is loaded", ["model"])


# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.normalizer: LogNormalizer = LogNormalizer()
        self.scorer: LogScorer | None = None
        self.enricher: LEEFEnricher | None = None
        self.start_time: float = time.monotonic()

        # In-process counters (redundant with Prometheus but useful for /metrics/snapshot)
        self.events_scored: int = 0
        self.events_high: int = 0
        self.events_duplicate: int = 0
        self.events_sigma: int = 0
        self.score_sum: float = 0.0
        self.latency_sum: float = 0.0
        self.rate_limit_windows: dict[str, deque[float]] = {}


_state = AppState()


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_VALUES


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and initialise models at startup."""
    telemetry.setup_tracing()
    telemetry.instrument_fastapi_app(app)
    with telemetry.start_as_current_span("api.lifespan.startup"):
        logger.info("LogFilter API starting …")

        _state.config = load_config(_CONFIG_PATH)
        if not _state.config:
            logger.warning("config.yaml not found or empty — using defaults")

        # Build scorer and enricher
        model_version = os.environ.get("LOGFILTER_MODEL_VERSION", "")
        _state.scorer = LogScorer(config=_state.config, model_version=model_version)
        qradar_cfg = _state.config.get("qradar", {})
        _state.enricher = LEEFEnricher(
            vendor=qradar_cfg.get("leef_vendor", "YourCo"),
            product=qradar_cfg.get("leef_product", "AIPreprocessor"),
            version=qradar_cfg.get("leef_version", "1.0"),
        )

        _model_loaded.labels(model="scorer").set(1)
        logger.info("LogFilter API ready")

    yield

    with telemetry.start_as_current_span("api.lifespan.shutdown"):
        logger.info("LogFilter API shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LogFilter AI Scoring API",
    description=(
        "AI-powered log preprocessing service for IBM QRadar SIEM. "
        "Scores log events using SecureBERT 2.0 models and MITRE ATT&CK matching."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _env_enabled("LOGFILTER_ENABLE_DOCS") else None,
    redoc_url="/redoc" if _env_enabled("LOGFILTER_ENABLE_DOCS") else None,
    openapi_url="/openapi.json" if _env_enabled("LOGFILTER_ENABLE_DOCS") else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CORS_ALLOW_ORIGINS",
            "http://localhost,http://localhost:3000,http://localhost:8080",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-API-Token", "Authorization"],
    allow_credentials=False,
    max_age=3600,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):  # noqa: ANN001
    extracted = telemetry.extract_http_context(request.headers)
    token = telemetry.attach_context(extracted)
    try:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'",
        )
        return response
    finally:
        telemetry.detach_context(token)


# ── Helper ─────────────────────────────────────────────────────────────────────
def _source_hint(source_type_str: str | None) -> LogSourceType | None:
    if source_type_str is None:
        return None
    try:
        return LogSourceType(source_type_str.lower())
    except ValueError:
        return None


def _build_response(scored_event, leef_payload: str) -> ScoreResponse:
    entities_dict = scored_event.entities or {}
    entity_summary = EntitySummary(
        indicators=entities_dict.get("indicators", []),
        malware=entities_dict.get("malware", []),
        vulnerabilities=entities_dict.get("vulnerabilities", []),
        organizations=entities_dict.get("organizations", []),
        systems=entities_dict.get("systems", []),
        confidence=entities_dict.get("confidence", 0.0),
        has_high_value_entities=entities_dict.get("has_high_value_entities", False),
    )
    attack_matches = [
        ATTACKMatch(
            technique_id=ce["id"],
            name=ce["name"],
            score=ce["score"],
        )
        for ce in scored_event.cross_encoder_scores
    ]
    return ScoreResponse(
        ai_threat_score=scored_event.ai_threat_score,
        ai_priority=scored_event.ai_priority,
        ai_mitre_technique=scored_event.ai_mitre_technique,
        ai_entities=scored_event.ai_entities,
        ai_confidence=scored_event.ai_confidence,
        sigma_matched=scored_event.sigma_matched,
        is_duplicate=scored_event.is_duplicate,
        dedup_similarity=scored_event.dedup_similarity,
        entities=entity_summary,
        attack_matches=attack_matches,
        classifier_score=scored_event.classifier_score,
        tier2_score=scored_event.tier2_score,
        tier2_used=scored_event.tier2_used,
        entity_boost=scored_event.entity_boost,
        cross_encoder_max=scored_event.cross_encoder_max,
        source_type=scored_event.source_type,
        host=scored_event.host,
        timestamp=scored_event.timestamp,
        normalized_text=scored_event.normalized_text,
        scoring_latency_ms=scored_event.scoring_latency_ms,
        leef_payload=leef_payload,
    )


def _update_metrics(scored_event) -> None:
    _events_total.labels(priority=scored_event.ai_priority).inc()
    _score_histogram.observe(scored_event.ai_threat_score)
    _latency_histogram.observe(scored_event.scoring_latency_ms)
    if scored_event.is_duplicate:
        _events_duplicate.inc()
        _state.events_duplicate += 1
    if scored_event.sigma_matched:
        _events_sigma.inc()
        _state.events_sigma += 1
    _state.events_scored += 1
    if scored_event.ai_priority == "HIGH":
        _state.events_high += 1
    _state.score_sum += scored_event.ai_threat_score
    _state.latency_sum += scored_event.scoring_latency_ms


async def _require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = os.environ.get("LOGFILTER_ADMIN_TOKEN") or _state.config.get("api", {}).get(
        "admin_token",
        "",
    )
    try:
        require_configured_token(
            x_admin_token,
            token,
            not_configured_detail="Admin token is not configured",
            invalid_detail="Invalid admin token",
        )
    except AccessDenied as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _configured_api_token() -> str:
    return os.environ.get("LOGFILTER_API_TOKEN") or _state.config.get("api", {}).get(
        "scoring_token",
        "",
    )


def _configured_metrics_token() -> str:
    return os.environ.get("LOGFILTER_METRICS_TOKEN") or _state.config.get("api", {}).get(
        "metrics_token",
        "",
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _enforce_rate_limit(request: Request) -> None:
    limit = int(_state.config.get("api", {}).get("rate_limit_per_minute", 60))
    client_host = request.client.host if request.client else "unknown"
    try:
        enforce_rate_limit(_state.rate_limit_windows, client_host, limit)
    except AccessDenied as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


async def _require_scoring_access(
    request: Request,
    x_api_token: str | None = Header(default=None),
) -> None:
    token = _configured_api_token()
    try:
        require_configured_token(
            x_api_token,
            token,
            not_configured_detail="Scoring API token is not configured",
            invalid_detail="Invalid scoring API token",
        )
    except AccessDenied as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    _enforce_rate_limit(request)


async def _require_metrics_access(
    x_metrics_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    token = _configured_metrics_token()
    if not token:
        return
    provided = x_metrics_token or _extract_bearer_token(authorization)
    try:
        require_configured_token(
            provided,
            token,
            not_configured_detail="Metrics token is not configured",
            invalid_detail="Invalid metrics token",
        )
    except AccessDenied as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_event(
    payload: ScoreRequest,
    _: None = Depends(_require_scoring_access),
) -> ScoreResponse:
    """
    Score a single log event.

    Returns threat score, ATT&CK technique match, extracted entities,
    and a LEEF-formatted enriched payload ready for QRadar forwarding.
    """
    with telemetry.start_as_current_span(
        "api.score_event",
        {"logfilter.event_count": 1, "logfilter.source_type": payload.source_type or "generic"},
    ) as span:
        if _state.scorer is None or _state.enricher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scoring service not initialised",
            )

        hint = _source_hint(payload.source_type)
        try:
            normalized = _state.normalizer.normalize(payload.raw, source_type_hint=hint)
            scored = _state.scorer.score(normalized)
            leef = _state.enricher.enrich(scored)
        except (ValueError, TypeError, KeyError) as exc:
            telemetry.record_exception(span, exc)
            logger.warning("Score request rejected", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Invalid log event: {exc}") from exc

        telemetry.set_span_attributes(
            span,
            {
                "logfilter.host": scored.host,
                "logfilter.priority": scored.ai_priority,
                "logfilter.threat_score": scored.ai_threat_score,
                "logfilter.sigma_matched": scored.sigma_matched,
            },
        )
        _update_metrics(scored)
        return _build_response(scored, leef)


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(
    payload: BatchScoreRequest,
    _: None = Depends(_require_scoring_access),
) -> BatchScoreResponse:
    """
    Score a batch of up to 200 log events in a single call.

    Recommended for high-throughput scenarios — batching amortises model
    loading overhead and enables efficient GPU/CPU utilisation.
    """
    with telemetry.start_as_current_span(
        "api.score_batch",
        {"logfilter.batch_size": len(payload.events), "logfilter.event_count": len(payload.events)},
    ) as span:
        if _state.scorer is None or _state.enricher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scoring service not initialised",
            )

        # Enforce configured batch size cap (config: api.max_batch_size, default 200).
        max_batch = int(_state.config.get("api", {}).get("max_batch_size", 200))
        if len(payload.events) > max_batch:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: {len(payload.events)} events (max {max_batch})",
            )
        if not payload.events:
            raise HTTPException(status_code=400, detail="Batch must contain at least one event")

        t0 = time.perf_counter()
        _batch_size_histogram.observe(len(payload.events))

        try:
            normalized_events = [
                _state.normalizer.normalize(ev.raw, source_type_hint=_source_hint(ev.source_type))
                for ev in payload.events
            ]
            scored_events = _state.scorer.score_batch(normalized_events)
            leef_payloads = _state.enricher.enrich_batch(scored_events)
        except (ValueError, TypeError, KeyError) as exc:
            telemetry.record_exception(span, exc)
            logger.warning("Batch score request rejected", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Invalid batch: {exc}") from exc

        responses = []
        high_count = 0
        medium_count = 0
        for scored, leef in zip(scored_events, leef_payloads):
            _update_metrics(scored)
            if scored.ai_priority == "HIGH":
                high_count += 1
            elif scored.ai_priority == "MEDIUM":
                medium_count += 1
            responses.append(_build_response(scored, leef))

        elapsed = (time.perf_counter() - t0) * 1000.0
        telemetry.set_span_attributes(
            span,
            {
                "logfilter.high_priority_count": high_count,
                "logfilter.medium_priority_count": medium_count,
                "logfilter.elapsed_ms": elapsed,
            },
        )
        return BatchScoreResponse(
            results=responses,
            total=len(responses),
            high_priority_count=high_count,
            medium_priority_count=medium_count,
            elapsed_ms=round(elapsed, 2),
        )


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Liveness and readiness check."""
    with telemetry.start_as_current_span("api.health") as span:
        scorer_ready = _state.scorer is not None

        models_loaded = {
            "scorer": scorer_ready,
            "enricher": _state.enricher is not None,
        }
        if scorer_ready and _state.scorer is not None:
            models_loaded["classifier"] = _state.scorer.classifier.is_ready()
            tier2 = getattr(_state.scorer, "tier2_classifier", None)
            if tier2 is not None:
                models_loaded["tier2_classifier"] = tier2.is_ready()

        overall_status = "healthy" if scorer_ready else "degraded"
        span.set_attribute("logfilter.health.status", overall_status)

        return HealthResponse(
            status=overall_status,
            version="0.1.0",
            models_loaded=models_loaded,
            uptime_seconds=round(time.monotonic() - _state.start_time, 1),
        )


@app.get("/metrics", tags=["Operations"])
async def metrics(_: None = Depends(_require_metrics_access)) -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/snapshot", response_model=MetricsSnapshot, tags=["Operations"])
async def metrics_snapshot(_: None = Depends(_require_admin)) -> MetricsSnapshot:
    """JSON snapshot of in-process metrics counters."""
    n = max(_state.events_scored, 1)
    return MetricsSnapshot(
        events_scored_total=_state.events_scored,
        events_high_priority_total=_state.events_high,
        events_duplicate_total=_state.events_duplicate,
        events_sigma_matched_total=_state.events_sigma,
        avg_latency_ms=round(_state.latency_sum / n, 2),
        avg_threat_score=round(_state.score_sum / n, 4),
    )


@app.post("/admin/reload", tags=["Operations"])
async def reload_models(_: None = Depends(_require_admin)) -> dict[str, str]:
    """
    Trigger a hot reload of the scoring pipeline.
    Useful after retraining the classifier without restarting the service.
    """
    if _state.scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialised")

    _state.config = load_config(_CONFIG_PATH)
    if not _state.config:
        logger.warning("config.yaml not found or empty — using defaults")
    _state.scorer = LogScorer(config=_state.config)
    logger.info("Models reloaded via /admin/reload")
    return {"status": "reloaded"}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    uvicorn = importlib.import_module("uvicorn")

    uvicorn.run("logfilter.api.app:app", host="0.0.0.0", port=8080, reload=False)
