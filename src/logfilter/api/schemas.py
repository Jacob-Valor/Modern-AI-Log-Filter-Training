"""Pydantic request/response schemas for the LogFilter API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request schemas ────────────────────────────────────────────────────────────


class ScoreRequest(BaseModel):
    """Score a single raw log event."""

    raw: str = Field(
        ...,
        description="Raw log string (syslog line, JSON event, CEF payload, etc.)",
        min_length=1,
        max_length=65536,
    )
    source_type: str | None = Field(
        None,
        description=("Optional hint: syslog | winevent | firewall | endpoint | cloudtrail | web"),
    )
    raw_log_ref: str | None = Field(
        None,
        description=(
            "Optional caller-supplied chain-of-custody ref (64-char sha256 hex). "
            "If provided, the API uses this ref in the LEEF ``raw_log_ref`` field "
            "instead of computing one locally — use this when the caller has "
            "already archived the raw event elsewhere."
        ),
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "raw": (
                    "Jan 15 11:07:53 prod-server01 sshd[22345]: Failed password "
                    "for root from 10.0.0.5 port 44382 ssh2"
                ),
                "source_type": "syslog",
            }
        }
    }


class BatchScoreRequest(BaseModel):
    """Score a batch of raw log events (up to 200 per request)."""

    events: list[ScoreRequest] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of log events to score",
    )


# ── Response schemas ───────────────────────────────────────────────────────────


class EntitySummary(BaseModel):
    indicators: list[str] = []
    malware: list[str] = []
    vulnerabilities: list[str] = []
    organizations: list[str] = []
    systems: list[str] = []
    confidence: float = 0.0
    has_high_value_entities: bool = False


class ATTACKMatch(BaseModel):
    technique_id: str
    name: str
    score: float


class ScoreResponse(BaseModel):
    """Scoring result for a single event."""

    # Core AI scores
    ai_threat_score: float = Field(description="Composite threat score 0.0–1.0")
    ai_priority: str = Field(description="HIGH / MEDIUM / LOW / INFO")
    ai_mitre_technique: str = Field(description="Top-matching ATT&CK technique ID")
    ai_entities: str = Field(description="Comma-separated extracted entities")
    ai_confidence: float = Field(description="Model confidence (0.0–1.0)")

    # Flags
    sigma_matched: bool
    is_duplicate: bool
    dedup_similarity: float

    # Detailed breakdowns
    entities: EntitySummary
    attack_matches: list[ATTACKMatch] = []

    # Score components
    classifier_score: float
    tier2_score: float
    tier2_used: bool
    entity_boost: float
    cross_encoder_max: float

    # Degraded scoring
    score_degraded: bool = Field(
        description=(
            "True when classifier failed or no model was loaded, "
            "so the score is a best-effort degraded placeholder"
        )
    )

    # Metadata
    source_type: str
    host: str
    timestamp: str
    normalized_text: str
    scoring_latency_ms: float

    # LEEF-formatted enriched event
    leef_payload: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "ai_threat_score": 0.87,
                "ai_priority": "HIGH",
                "ai_mitre_technique": "T1110.001",
                "ai_entities": "10.0.0.5",
                "ai_confidence": 0.83,
                "sigma_matched": False,
                "is_duplicate": False,
                "dedup_similarity": 0.12,
                "entities": {
                    "indicators": ["10.0.0.5"],
                    "malware": [],
                    "vulnerabilities": [],
                    "organizations": [],
                    "systems": [],
                    "confidence": 0.91,
                    "has_high_value_entities": True,
                },
                "attack_matches": [
                    {"technique_id": "T1110.001", "name": "Password Guessing", "score": 0.82}
                ],
                "classifier_score": 0.76,
                "tier2_score": 0.0,
                "tier2_used": False,
                "entity_boost": 0.20,
                "cross_encoder_max": 0.82,
                "source_type": "syslog",
                "host": "prod-server01",
                "timestamp": "Jan 15 11:07:53",
                "normalized_text": (
                    "Host prod-server01 Process sshd[22345]: Failed password "
                    "for root from 10.0.0.5 port 44382 ssh2"
                ),
                "scoring_latency_ms": 45.3,
                "leef_payload": (
                    "LEEF:2.0|YourCo|AIPreprocessor|1.0|LOG_EVENT|\t|"
                    "ai_threat_score=0.8700\t..."
                ),
            }
        }
    }


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    total: int
    high_priority_count: int
    medium_priority_count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    models_loaded: dict[str, bool]
    uptime_seconds: float


class DriftHealthResponse(BaseModel):
    drift_detected: bool
    psi: float
    reference_count: int
    current_count: int
    fallback_active: bool


class MetricsSnapshot(BaseModel):
    """Lightweight in-process metrics snapshot (complements /metrics Prometheus endpoint)."""

    events_scored_total: int
    events_high_priority_total: int
    events_duplicate_total: int
    events_sigma_matched_total: int
    avg_latency_ms: float
    avg_threat_score: float
    drift_detected: bool
    drift_psi: float
    drift_fallback_active: bool
