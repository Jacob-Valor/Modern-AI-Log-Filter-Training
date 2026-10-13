"""
FastAPI application — AI Log Filter scoring service.

Endpoints:
  POST /score              — Score a single log event
  POST /score/batch        — Score up to 200 events in one call
  GET  /health             — Liveness + readiness check
  GET  /metrics            — Prometheus metrics (text/plain)
  GET  /metrics/snapshot   — JSON metrics snapshot
"""

from __future__ import annotations

import importlib
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog  # type: ignore[import-not-found]
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool  # type: ignore[import-not-found]
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
    DriftHealthResponse,
    EntitySummary,
    HealthResponse,
    MetricsSnapshot,
    ResponseActionSummary,
    ScoreRequest,
    ScoreResponse,
)
from logfilter.api.security import (
    AccessDenied,
    RedisRateLimiter,
    client_ip_from_request,
    enforce_rate_limit,
    require_configured_token,
)
from logfilter.config import load_config
from logfilter.http_gateway import SignedHttpGatewayClient
from logfilter.monitoring.drift_detector import DriftStatus
from logfilter.pipeline.archive import LogArchive, compute_raw_log_ref
from logfilter.pipeline.enricher import LEEFEnricher
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType
from logfilter.pipeline.scorer import LogScorer
from logfilter.response.auto_response import AutoResponseEngine, SignedResponseGateway
from logfilter.security.redaction import RedactionConfig

logger = structlog.get_logger(__name__)

# ── Config path ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
_TRUE_VALUES = {"1", "true", "yes", "on"}

#: Delay before a previously-failed Redis rate limiter is re-probed (seconds).
_REDIS_RETRY_SECONDS = 30.0

#: How long a dependency-health fan-out result is reused before re-probing ES/Redis.
_HEALTH_CACHE_SECONDS = 5.0


def _health_cache_seconds() -> float:
    """Return the dependency-health cache TTL, overridable via the environment."""
    raw = os.environ.get("LOGFILTER_HEALTH_CACHE_SECONDS", "").strip()
    if not raw:
        return _HEALTH_CACHE_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("LOGFILTER_HEALTH_CACHE_SECONDS must be a number") from exc
    if value < 0:
        raise ValueError("LOGFILTER_HEALTH_CACHE_SECONDS must not be negative")
    return value


def _redis_retry_seconds() -> float:
    """Return the Redis re-probe delay, overridable via the environment."""
    raw = os.environ.get("LOGFILTER_REDIS_RETRY_SECONDS", "").strip()
    if not raw:
        return _REDIS_RETRY_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("LOGFILTER_REDIS_RETRY_SECONDS must be a number") from exc
    if value <= 0:
        raise ValueError("LOGFILTER_REDIS_RETRY_SECONDS must be positive")
    return value


# ── Prometheus metrics ─────────────────────────────────────────────────────────
_events_total = Counter("logfilter_events_total", "Total log events scored", ["priority"])
_events_duplicate = Counter("logfilter_events_duplicate_total", "Duplicate events detected")
_events_sigma = Counter("logfilter_events_sigma_total", "Sigma rule matches")
_events_by_technique = Counter(
    "logfilter_events_technique_total", "Events by MITRE technique", ["technique"]
)
_events_by_source = Counter(
    "logfilter_events_source_total", "Events by source type", ["source_type"]
)
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
_drift_psi = Gauge("logfilter_drift_psi", "Population Stability Index for classifier scores")
_drift_detected = Gauge("logfilter_drift_detected", "1 if model drift is currently detected")


def _config_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.normalizer: LogNormalizer = LogNormalizer()
        self.scorer: LogScorer | None = None
        self.enricher: LEEFEnricher | None = None
        self.archiver: LogArchive | None = None
        self.response_engine = AutoResponseEngine(dry_run=True)
        self.response_gateway: SignedResponseGateway | None = None
        self.start_time: float = time.monotonic()
        self.redis_client: Any | None = None
        self.rate_limiter: RedisRateLimiter | None = None

        # In-process counters (redundant with Prometheus but useful for /metrics/snapshot)
        self.events_scored: int = 0
        self.events_high: int = 0
        self.events_duplicate: int = 0
        self.events_sigma: int = 0
        self.score_sum: float = 0.0
        self.latency_sum: float = 0.0
        self.rate_limit_windows: dict[str, deque[float]] = {}
        self.redis_retry_at: float = 0.0
        self.health_cache_at: float = 0.0
        self.health_cache: dict[str, str] = {}
        self.redis_url: str = ""


_state = AppState()


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_VALUES


def _config_enabled(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in _TRUE_VALUES)


def _config_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _configure_response_engine() -> None:
    cfg = _state.config.get("response", {})
    mode = str(cfg.get("mode", "dry_run")).strip().lower()
    if mode not in {"dry_run", "armed"}:
        raise ValueError("response.mode must be dry_run or armed")
    dry_run = mode == "dry_run"
    gateway: SignedResponseGateway | None = None
    if not dry_run:
        gateway_cfg = cfg.get("gateway", {})
        client = SignedHttpGatewayClient(
            url=str(gateway_cfg.get("url", "")),
            hmac_secret=str(gateway_cfg.get("hmac_secret", "")),
            timeout_seconds=float(gateway_cfg.get("timeout_seconds", 10.0)),
            ca_file=str(gateway_cfg.get("ca_file", "")),
            client_cert_file=str(gateway_cfg.get("client_cert_file", "")),
            client_key_file=str(gateway_cfg.get("client_key_file", "")),
        )
        gateway = SignedResponseGateway(client)
    _state.response_gateway = gateway
    _state.response_engine = AutoResponseEngine(
        firewall=gateway,
        notifier=gateway,
        dry_run=dry_run,
        block_threshold=float(cfg.get("block_threshold", 0.85)),
        quarantine_priorities=frozenset(
            _config_values(cfg.get("quarantine_priorities", ("HIGH",)))
        ),
        allowed_networks=_config_values(cfg.get("allowed_networks", ())),
        allowed_hosts=frozenset(_config_values(cfg.get("allowed_hosts", ()))),
        kill_switch=_config_enabled(cfg.get("kill_switch", False)),
        max_actions_per_minute=_config_int(
            cfg.get("max_actions_per_minute", 60), "response max actions per minute"
        ),
    )


def _configure_rate_limiter() -> None:
    _state.redis_url = os.environ.get("REDIS_URL", "").strip()
    if not _state.redis_url:
        _state.redis_client = None
        _state.rate_limiter = None
        _state.redis_retry_at = 0.0
        return
    _connect_rate_limiter()


def _connect_rate_limiter() -> bool:
    """Try to adopt Redis as the rate-limit backend; arm a retry when it fails."""
    try:
        redis_module = importlib.import_module("redis")
        redis_client = redis_module.Redis.from_url(_state.redis_url, decode_responses=True)
        redis_client.ping()
    except Exception as exc:  # noqa: BLE001
        _state.redis_client = None
        _state.rate_limiter = None
        _state.redis_retry_at = time.monotonic() + _redis_retry_seconds()
        logger.warning(
            "Redis rate limiter unavailable; limiting is per-process only until a retry succeeds",
            error=str(exc),
            error_type=type(exc).__name__,
            redis_url=_state.redis_url,
            retry_in_seconds=_redis_retry_seconds(),
        )
        return False

    _state.redis_client = redis_client
    _state.rate_limiter = RedisRateLimiter(redis_client)
    _state.redis_retry_at = 0.0
    logger.info("Redis rate limiter enabled")
    return True


def _maybe_retry_rate_limiter() -> None:
    """Re-probe Redis after the retry delay so a recovery is not permanent."""
    if _state.rate_limiter is not None or not _state.redis_url:
        return
    if time.monotonic() < _state.redis_retry_at:
        return
    if _connect_rate_limiter():
        logger.info("Redis rate limiter recovered; distributed limiting re-enabled")



def _configure_archiver() -> None:
    """
    Initialise the Elasticsearch archiver so the API writes raw logs to ES
    with the same ID embedded in LEEF as ``raw_log_ref`` (B8 chain-of-custody).

    Reads ``ES_HOST`` / ``ES_USER`` / ``ES_PASSWORD`` from the loaded config
    (which already performs ``${ENV_VAR:default}`` resolution). If the password
    is missing, the archiver stays ``None`` and ``_archive_then_score`` falls
    back to a local-only ``raw_log_ref``. A clear warning is logged so the
    gap is visible in startup logs (the API still scores, but raw logs are
    not retained for forensic lookup).
    """
    es_cfg = _state.config.get("elasticsearch") or {}
    password = (es_cfg.get("password") or "").strip()
    if not password:
        logger.warning(
            "ES archiver NOT configured — raw logs will not be retained. "
            "Set ES_PASSWORD (and optionally ES_HOST/ES_USER) to enable "
            "chain-of-custody archive. LEEF raw_log_ref will be a local-only "
            "deterministic hash that does not resolve to a retrievable document."
        )
        _state.archiver = None
        return
    try:
        _state.archiver = LogArchive(
            hosts=es_cfg.get("hosts") or ["http://localhost:9200"],
            index_prefix=es_cfg.get("index_prefix", "raw-logs"),
            username=es_cfg.get("username", "elastic"),
            password=password,
            shards=int(es_cfg.get("index_shards", 1)),
            replicas=int(es_cfg.get("index_replicas", 0)),
            redaction_config=RedactionConfig.from_mapping(es_cfg.get("redaction")),
        )
        logger.info(
            "ES archiver configured",
            hosts=es_cfg.get("hosts"),
            index_prefix=es_cfg.get("index_prefix", "raw-logs"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ES archiver init failed; raw logs will not be retained",
            error=str(exc),
        )
        _state.archiver = None


def _is_disabled_stage(model: object) -> bool:
    return type(model).__name__.startswith("Disabled")


def _is_loaded_model(model: object, attr_name: str) -> bool:
    if _is_disabled_stage(model):
        return True
    return getattr(model, attr_name, None) is not None


def _collect_model_health() -> dict[str, bool]:
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
        biencoder = getattr(_state.scorer, "biencoder", None)
        if biencoder is not None:
            models_loaded["biencoder"] = _is_loaded_model(biencoder, "_model")
        ner = getattr(_state.scorer, "ner_model", None)
        if ner is not None:
            models_loaded["ner"] = _is_loaded_model(ner, "_pipeline")
        cross_encoder = getattr(_state.scorer, "cross_encoder", None)
        if cross_encoder is not None:
            models_loaded["cross_encoder"] = _is_loaded_model(cross_encoder, "_model")
        if getattr(_state.scorer, "flow_enabled", False):
            models_loaded["flow_classifier"] = _state.scorer.flow_classifier.is_loaded()
        if getattr(_state.scorer, "bot_flow_enabled", False):
            models_loaded["bot_flow_classifier"] = _state.scorer.bot_flow_classifier.is_loaded()
        if getattr(_state.scorer, "endpoint_novelty_enabled", False):
            models_loaded["endpoint_novelty"] = _state.scorer.endpoint_novelty_model.is_ready()
    return models_loaded


def _collect_dependency_health() -> dict[str, str]:
    dependencies = {"elasticsearch": "disabled", "redis": "disabled", "kafka": "not_checked"}
    if _state.archiver is not None:
        try:
            es_health = _state.archiver.health()
            dependencies["elasticsearch"] = str(es_health.get("status", "unknown"))
        except Exception as exc:  # noqa: BLE001
            dependencies["elasticsearch"] = f"down:{type(exc).__name__}"
    if _state.redis_client is not None:
        try:
            _state.redis_client.ping()
            dependencies["redis"] = "ok"
        except Exception as exc:  # noqa: BLE001
            dependencies["redis"] = f"down:{type(exc).__name__}"
    return dependencies


def _cached_dependency_health() -> dict[str, str]:
    """Return dependency health, re-fanning out to ES/Redis only after the TTL.

    ``/health`` is unauthenticated, so without this cache any anonymous caller
    could force one Elasticsearch ``cluster.health`` round-trip and one Redis
    ``PING`` per request.
    """
    ttl = _health_cache_seconds()
    now = time.monotonic()
    if _state.health_cache and now - _state.health_cache_at < ttl:
        return dict(_state.health_cache)
    _state.health_cache = _collect_dependency_health()
    _state.health_cache_at = now
    return dict(_state.health_cache)



def _dependencies_ready(dependencies: dict[str, str]) -> bool:
    ok_states = {"ok", "green", "yellow", "disabled", "not_checked"}
    return all(value in ok_states for value in dependencies.values())


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

        _configure_rate_limiter()
        _configure_response_engine()
        _configure_archiver()

        # Block production startup with localhost CORS origins
        if (
            not _env_enabled("LOGFILTER_ENABLE_DOCS")
            and set(_cors_origins) & _CORS_LOCALHOST_DEFAULTS
        ):
            localhost_origins = sorted(set(_cors_origins) & _CORS_LOCALHOST_DEFAULTS)
            raise RuntimeError(
                f"Refusing to start with localhost CORS origins in production mode: "
                f"{localhost_origins}. Set CORS_ALLOW_ORIGINS to your actual frontend "
                f"origin(s), or set LOGFILTER_ENABLE_DOCS=1 for local development."
            )
        if _state.rate_limiter is None:
            logger.warning(
                "Redis not configured — rate limiting is per-process only; "
                "ineffective behind multi-worker deployments. "
                "Set REDIS_URL for distributed rate limiting.",
            )

        # Build scorer and enricher
        model_version = os.environ.get("LOGFILTER_MODEL_VERSION", "")
        _state.scorer = LogScorer(config=_state.config, model_version=model_version)
        qradar_cfg = _state.config.get("qradar", {})
        _state.enricher = LEEFEnricher(
            vendor=qradar_cfg.get("leef_vendor", "YourCo"),
            product=qradar_cfg.get("leef_product", "AIPreprocessor"),
            version=qradar_cfg.get("leef_version", "1.0"),
        )

        t0 = time.perf_counter()
        logger.info("Pre-loading models …")
        _state.scorer.preload_models()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Models pre-loaded", elapsed_ms=round(elapsed, 1))

        _model_loaded.labels(model="scorer").set(1)
        logger.info("LogFilter API ready")

    yield

    with telemetry.start_as_current_span("api.lifespan.shutdown"):
        logger.info("LogFilter API shutting down")
        if _state.redis_client is not None:
            try:
                _state.redis_client.close()
            except Exception:  # noqa: BLE001
                pass
            _state.redis_client = None
        if _state.response_gateway is not None:
            try:
                _state.response_gateway.close()
            except Exception:  # noqa: BLE001
                pass
            _state.response_gateway = None
        if _state.archiver is not None:
            try:
                _state.archiver.close()
            except Exception:  # noqa: BLE001
                pass
            _state.archiver = None


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

_CORS_LOCALHOST_DEFAULTS = {"http://localhost", "http://localhost:3000", "http://localhost:8080"}
_DEV_CORS_ORIGINS = "http://localhost,http://localhost:3000,http://localhost:8080"
_cors_origins_str = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    _DEV_CORS_ORIGINS if _env_enabled("LOGFILTER_ENABLE_DOCS") else "",
)
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Token", "Authorization"],
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


def _ingest_ts(normalized_timestamp: str) -> float:
    """Parse a NormalizedEvent.timestamp into epoch seconds; fallback to now()."""
    if not normalized_timestamp:
        return time.time()
    from datetime import datetime, timezone

    try:
        normalised = normalized_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalised)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


def _archive_then_score(normalized, caller_ref: str | None) -> str:
    """
    Resolve one chain-of-custody ref without duplicating upstream archives.

    Kafka callers supply the ArchiveConsumer's stable topic/partition/offset
    reference. The API trusts that reference and does not write a second raw
    document. Direct API callers receive a locally computed reference and a
    best-effort Elasticsearch archive when the archiver is configured.
    """
    if caller_ref is not None:
        return caller_ref

    ingest_ts = _ingest_ts(normalized.timestamp)
    source_type = (
        normalized.source_type.value
        if hasattr(normalized.source_type, "value")
        else str(normalized.source_type)
    )
    raw_log_ref = compute_raw_log_ref(
        normalized.raw,
        source_type,
        normalized.host,
        ingest_ts,
    )
    if _state.archiver is not None:
        try:
            _state.archiver.write_with_id(
                raw_log_ref=raw_log_ref,
                raw=normalized.raw,
                source_type=source_type,
                host=normalized.host,
                ingest_ts=ingest_ts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ES archive failed; raw_log_ref is local-only",
                raw_log_ref=raw_log_ref,
                error=str(exc),
            )
    return raw_log_ref


def _do_score_sync(raw: str, source_type: str | None, raw_log_ref: str | None):
    """
    Synchronous helper: normalize → archive → score → enrich → respond.

    Returns ``(scored, leef, raw_log_ref, response_actions)`` and is invoked
    from async handlers via ``run_in_threadpool`` so model inference, archive
    I/O, and acknowledged gateway calls do not stall the FastAPI event loop.
    """
    if _state.scorer is None or _state.enricher is None:
        raise RuntimeError("Scoring service not initialised")

    normalized = _state.normalizer.normalize(raw, source_type_hint=_source_hint(source_type))
    es_doc_id = _archive_then_score(normalized, raw_log_ref)
    scored = _state.scorer.score(normalized)
    leef = _state.enricher.enrich(scored, es_doc_id=es_doc_id)
    response_actions = _state.response_engine.handle(scored, es_doc_id)
    return scored, leef, es_doc_id, response_actions


def _do_score_batch_sync(
    raws: list[str],
    source_types: list[str | None],
    raw_log_refs: list[str | None],
):
    """
    Synchronous batch helper. Mirrors ``_do_score_sync`` for the batch endpoint.
    """
    if _state.scorer is None or _state.enricher is None:
        raise RuntimeError("Scoring service not initialised")

    normalized_events = [
        _state.normalizer.normalize(r, source_type_hint=_source_hint(st))
        for r, st in zip(raws, source_types)
    ]
    es_doc_ids = [
        _archive_then_score(norm, ref) for norm, ref in zip(normalized_events, raw_log_refs)
    ]
    scored_events = _state.scorer.score_batch(normalized_events)
    leef_payloads = _state.enricher.enrich_batch(scored_events, es_doc_ids=es_doc_ids)
    response_actions = [
        _state.response_engine.handle(scored, event_ref)
        for scored, event_ref in zip(scored_events, es_doc_ids)
    ]
    return scored_events, leef_payloads, es_doc_ids, response_actions


def _build_response(
    scored_event,
    leef_payload: str,
    raw_log_ref: str,
    response_actions,
) -> ScoreResponse:
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
    recommended_actions = [
        ResponseActionSummary(
            action=action.action.value,
            target=action.target,
            reason=action.reason,
            action_id=action.action_id,
            status=action.status,
        )
        for action in response_actions
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
        flow_score=scored_event.flow_score,
        bot_score=scored_event.bot_score,
        ddos_window_score=scored_event.ddos_window_score,
        endpoint_score=scored_event.endpoint_score,
        http_score=scored_event.http_score,
        tier2_score=scored_event.tier2_score,
        tier2_used=scored_event.tier2_used,
        entity_boost=scored_event.entity_boost,
        cross_encoder_max=scored_event.cross_encoder_max,
        source_type=scored_event.source_type,
        host=scored_event.host,
        timestamp=scored_event.timestamp,
        normalized_text=scored_event.normalized_text,
        scoring_latency_ms=scored_event.scoring_latency_ms,
        score_degraded=scored_event.score_degraded,
        leef_payload=leef_payload,
        raw_log_ref=raw_log_ref,
        response_mode="dry_run" if _state.response_engine.dry_run else "armed",
        recommended_actions=recommended_actions,
    )


def _update_metrics(scored_event) -> None:
    _events_total.labels(priority=scored_event.ai_priority).inc()
    _score_histogram.observe(scored_event.ai_threat_score)
    if scored_event.is_duplicate:
        _events_duplicate.inc()
        _state.events_duplicate += 1
    if scored_event.sigma_matched:
        _events_sigma.inc()
        _state.events_sigma += 1
    # Attack-specific metrics
    technique = scored_event.ai_mitre_technique or "none"
    _events_by_technique.labels(technique=technique).inc()
    _events_by_source.labels(source_type=scored_event.source_type).inc()
    _state.events_scored += 1
    if scored_event.ai_priority == "HIGH":
        _state.events_high += 1
    _state.score_sum += scored_event.ai_threat_score
    _state.latency_sum += scored_event.scoring_latency_ms
    scorer = _state.scorer
    drift_detector = getattr(scorer, "drift_detector", None)
    if drift_detector is not None:
        status = drift_detector.check_drift()
        _drift_psi.set(status.psi)
        _drift_detected.set(1.0 if status.drift_detected else 0.0)


def _configured_api_token() -> str:
    env_token = os.environ.get("LOGFILTER_API_TOKEN", "")
    if env_token:
        return env_token
    cfg_token = _state.config.get("api", {}).get("scoring_token", "")
    if cfg_token:
        logger.warning("API token sourced from config.yaml — prefer LOGFILTER_API_TOKEN env var")
    return cfg_token


def _configured_metrics_token() -> str:
    env_token = os.environ.get("LOGFILTER_METRICS_TOKEN", "")
    if env_token:
        return env_token
    cfg_token = _state.config.get("api", {}).get("metrics_token", "")
    if cfg_token:
        logger.warning(
            "Metrics token sourced from config.yaml — prefer LOGFILTER_METRICS_TOKEN env var"
        )
    return cfg_token


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _enforce_rate_limit(request: Request) -> None:
    api_cfg = _state.config.get("api", {})
    limit = _config_int(api_cfg.get("rate_limit_per_minute", 60), "API rate limit")
    remote_addr = request.client.host if request.client else "unknown"
    client_host = client_ip_from_request(
        remote_addr=remote_addr,
        forwarded_for=request.headers.get("x-forwarded-for"),
        trusted_proxies=api_cfg.get("trusted_proxies", []),
    )
    _maybe_retry_rate_limiter()
    if _state.rate_limiter is not None:
        try:
            enforce_rate_limit(
                _state.rate_limit_windows,
                client_host,
                limit,
                backend=_state.rate_limiter,
            )
            return
        except AccessDenied as exc:
            # Convert to HTTPException like the in-process fallback below.
            # Re-raising the bare AccessDenied escaped FastAPI's handler and
            # surfaced as HTTP 500, so a throttled client got a server-error
            # instead of 429 whenever the Redis limiter was active.
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except Exception as exc:
            logger.warning(
                "Redis rate limiter request failed; limiting is per-process only "
                "until a retry succeeds",
                client_id=client_host,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            _state.redis_client = None
            _state.rate_limiter = None
            _state.redis_retry_at = time.monotonic() + _redis_retry_seconds()
    try:
        enforce_rate_limit(_state.rate_limit_windows, client_host, limit)
    except AccessDenied as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


async def _require_scoring_access(
    request: Request,
    x_api_token: str | None = Header(default=None),
) -> None:
    # Rate-limit before the token comparison: a wrong-token flood must be
    # throttled too, otherwise the token is brute-forceable without limit.
    _enforce_rate_limit(request)
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



async def _require_metrics_access(
    request: Request,
    x_metrics_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    # Rate-limit before the token comparison so brute-force attempts are throttled.
    _enforce_rate_limit(request)
    token = _configured_metrics_token()
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

        try:
            scored, leef, raw_log_ref, response_actions = await run_in_threadpool(
                _do_score_sync, payload.raw, payload.source_type, payload.raw_log_ref
            )
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
        return _build_response(scored, leef, raw_log_ref, response_actions)


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
        max_batch = _config_int(
            _state.config.get("api", {}).get("max_batch_size", 200), "maximum batch size"
        )
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
            (
                scored_events,
                leef_payloads,
                raw_log_refs,
                response_action_batches,
            ) = await run_in_threadpool(
                _do_score_batch_sync,
                [ev.raw for ev in payload.events],
                [ev.source_type for ev in payload.events],
                [ev.raw_log_ref for ev in payload.events],
            )
        except (ValueError, TypeError, KeyError) as exc:
            telemetry.record_exception(span, exc)
            logger.warning("Batch score request rejected", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Invalid batch: {exc}") from exc

        responses = []
        high_count = 0
        medium_count = 0
        for scored, leef, raw_log_ref, response_actions in zip(
            scored_events,
            leef_payloads,
            raw_log_refs,
            response_action_batches,
        ):
            _update_metrics(scored)
            if scored.ai_priority == "HIGH":
                high_count += 1
            elif scored.ai_priority == "MEDIUM":
                medium_count += 1
            responses.append(_build_response(scored, leef, raw_log_ref, response_actions))

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
async def health(response: Response) -> HealthResponse:
    """Liveness and readiness check.

    Returns HTTP 503 when the service is degraded (scorer not loaded) so that
    load-balancers and orchestration probes treat the pod as not-ready instead
    of routing traffic to a node that cannot score.
    """
    with telemetry.start_as_current_span("api.health") as span:
        models_loaded = _collect_model_health()
        dependencies = _cached_dependency_health()
        overall_status = (
            "healthy"
            if all(models_loaded.values()) and _dependencies_ready(dependencies)
            else "degraded"
        )
        span.set_attribute("logfilter.health.status", overall_status)
        if overall_status != "healthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return HealthResponse(
            status=overall_status,
            version="0.1.0",
            models_loaded=models_loaded,
            dependencies=dependencies,
            uptime_seconds=round(time.monotonic() - _state.start_time, 1),
        )


@app.get("/metrics", tags=["Operations"])
async def metrics(_: None = Depends(_require_metrics_access)) -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/snapshot", response_model=MetricsSnapshot, tags=["Operations"])
async def metrics_snapshot(_: None = Depends(_require_metrics_access)) -> MetricsSnapshot:
    """JSON snapshot of in-process metrics counters."""
    n = max(_state.events_scored, 1)
    drift_status = DriftStatus()
    scorer = _state.scorer
    drift_detector = getattr(scorer, "drift_detector", None)
    if drift_detector is not None:
        drift_status = drift_detector.check_drift()
    return MetricsSnapshot(
        events_scored_total=_state.events_scored,
        events_high_priority_total=_state.events_high,
        events_duplicate_total=_state.events_duplicate,
        events_sigma_matched_total=_state.events_sigma,
        avg_latency_ms=round(_state.latency_sum / n, 2),
        avg_threat_score=round(_state.score_sum / n, 4),
        drift_detected=drift_status.drift_detected,
        drift_psi=round(drift_status.psi, 4),
        drift_fallback_active=drift_status.fallback_active,
    )


@app.get("/health/drift", response_model=DriftHealthResponse, tags=["Operations"])
async def health_drift() -> DriftHealthResponse:
    """Return the current model drift status for load-balancers and operators."""
    scorer = _state.scorer
    if scorer is None or scorer.drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detector not configured",
        )
    drift = scorer.drift_detector.check_drift()
    return DriftHealthResponse(
        drift_detected=drift.drift_detected,
        psi=round(drift.psi, 4),
        reference_count=drift.reference_count,
        current_count=drift.current_count,
        fallback_active=drift.fallback_active,
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    uvicorn = importlib.import_module("uvicorn")

    # Pass the app object directly (not the "module:attr" string) — the string
    # form makes uvicorn re-import this file under the "logfilter.api.app"
    # module name in addition to the "__main__" name Python already used to
    # run it, executing all top-level code (including Prometheus Counter/Gauge
    # registration) twice and raising "Duplicated timeseries in CollectorRegistry".
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)
