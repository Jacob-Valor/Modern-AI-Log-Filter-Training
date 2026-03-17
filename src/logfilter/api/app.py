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

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import yaml
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

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
from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.ner import NERModel
from logfilter.pipeline.enricher import LEEFEnricher
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType
from logfilter.pipeline.scorer import LogScorer

logger = structlog.get_logger(__name__)

# ── Config path ────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path("config/config.yaml")

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


_state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and initialise models at startup."""
    logger.info("LogFilter API starting …")

    # Load config
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            _state.config = yaml.safe_load(f) or {}
    else:
        logger.warning("config.yaml not found — using defaults")

    # Resolve env vars in config (simple pattern: "${VAR:default}")
    _state.config = _resolve_env_vars(_state.config)

    # Build scorer and enricher
    _state.scorer = LogScorer(config=_state.config)
    qradar_cfg = _state.config.get("qradar", {})
    _state.enricher = LEEFEnricher(
        vendor=qradar_cfg.get("leef_vendor", "YourCo"),
        product=qradar_cfg.get("leef_product", "AIPreprocessor"),
        version=qradar_cfg.get("leef_version", "1.0"),
    )

    _model_loaded.labels(model="scorer").set(1)
    logger.info("LogFilter API ready")

    yield

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
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_event(request: ScoreRequest) -> ScoreResponse:
    """
    Score a single log event.

    Returns threat score, ATT&CK technique match, extracted entities,
    and a LEEF-formatted enriched payload ready for QRadar forwarding.
    """
    if _state.scorer is None or _state.enricher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring service not initialised",
        )

    hint = _source_hint(request.source_type)
    normalized = _state.normalizer.normalize(request.raw, source_type_hint=hint)
    scored = _state.scorer.score(normalized)
    leef = _state.enricher.enrich(scored)

    _update_metrics(scored)
    return _build_response(scored, leef)


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(request: BatchScoreRequest) -> BatchScoreResponse:
    """
    Score a batch of up to 200 log events in a single call.

    Recommended for high-throughput scenarios — batching amortises model
    loading overhead and enables efficient GPU/CPU utilisation.
    """
    if _state.scorer is None or _state.enricher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring service not initialised",
        )

    t0 = time.perf_counter()
    _batch_size_histogram.observe(len(request.events))

    normalized_events = [
        _state.normalizer.normalize(ev.raw, source_type_hint=_source_hint(ev.source_type))
        for ev in request.events
    ]
    scored_events = _state.scorer.score_batch(normalized_events)
    leef_payloads = _state.enricher.enrich_batch(scored_events)

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
    scorer_ready = _state.scorer is not None

    models_loaded = {
        "scorer": scorer_ready,
        "enricher": _state.enricher is not None,
    }
    if scorer_ready and _state.scorer is not None:
        models_loaded["classifier"] = _state.scorer.classifier.is_ready()

    overall_status = "healthy" if scorer_ready else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        models_loaded=models_loaded,
        uptime_seconds=round(time.monotonic() - _state.start_time, 1),
    )


@app.get("/metrics", tags=["Operations"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/snapshot", response_model=MetricsSnapshot, tags=["Operations"])
async def metrics_snapshot() -> MetricsSnapshot:
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
async def reload_models() -> dict[str, str]:
    """
    Trigger a hot reload of the scoring pipeline.
    Useful after retraining the classifier without restarting the service.
    """
    if _state.scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialised")

    _state.scorer = LogScorer(config=_state.config)
    logger.info("Models reloaded via /admin/reload")
    return {"status": "reloaded"}


# ── Env var resolution ─────────────────────────────────────────────────────────
import os
import re as _re


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR:default} placeholders in a config dict."""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    if isinstance(obj, str):

        def _replace(m: _re.Match) -> str:
            var, _, default = m.group(1).partition(":")
            return os.environ.get(var, default)

        return _re.sub(r"\$\{([^}]+)\}", _replace, obj)
    return obj


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("logfilter.api.app:app", host="0.0.0.0", port=8080, reload=False)
