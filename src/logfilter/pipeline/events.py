"""Lightweight pipeline data structures shared across services.

This module deliberately imports only the standard library so that the slim
router/enricher container images (which do not ship numpy/torch) can import the
``ScoredEvent`` contract without pulling in the heavy scoring stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredEvent:
    """
    Complete scoring result for one log event, ready for LEEF enrichment.
    """

    # Provenance
    source_type: str
    timestamp: str
    host: str
    raw: str
    normalized_text: str
    fields: dict[str, Any] = field(default_factory=dict)

    # Tier 1 — Sigma
    sigma_matched: bool = False
    sigma_rule_ids: list[str] = field(default_factory=list)

    # Tier 2 — BiEncoder
    is_duplicate: bool = False
    dedup_similarity: float = 0.0
    attack_candidates: list[dict[str, Any]] = field(default_factory=list)

    # Tier 3 — NER + CrossEncoder
    entities: dict[str, Any] = field(default_factory=dict)
    cross_encoder_scores: list[dict[str, Any]] = field(default_factory=list)

    # Final score components
    classifier_score: float = 0.0
    tier2_score: float = 0.0
    tier2_used: bool = False
    entity_boost: float = 0.0
    cross_encoder_max: float = 0.0
    # Novelty detection is not yet implemented. Default 0.0 means it contributes
    # nothing to the fused score (was 0.5 placeholder, which silently added a
    # constant 0.5 * w_novelty bias to every event). See docs/ML_REMEDIATION.md.
    novelty_score: float = 0.0
    dedup_penalty: float = 0.0

    # Final composite score + routing label
    ai_threat_score: float = 0.0
    ai_priority: str = "LOW"  # HIGH / MEDIUM / LOW / INFO
    ai_mitre_technique: str = ""  # Top-matching ATT&CK technique ID
    ai_confidence: float = 0.0  # Average model confidence
    ai_entities: str = ""  # Comma-separated entity string

    # Timing
    scoring_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "host": self.host,
            "normalized_text": self.normalized_text,
            "fields": self.fields,
            "sigma_matched": self.sigma_matched,
            "sigma_rule_ids": self.sigma_rule_ids,
            "is_duplicate": self.is_duplicate,
            "dedup_similarity": round(self.dedup_similarity, 4),
            "attack_candidates": self.attack_candidates,
            "entities": self.entities,
            "cross_encoder_scores": self.cross_encoder_scores,
            "classifier_score": round(self.classifier_score, 4),
            "tier2_score": round(self.tier2_score, 4),
            "tier2_used": self.tier2_used,
            "entity_boost": round(self.entity_boost, 4),
            "cross_encoder_max": round(self.cross_encoder_max, 4),
            "novelty_score": round(self.novelty_score, 4),
            "dedup_penalty": round(self.dedup_penalty, 4),
            "ai_threat_score": round(self.ai_threat_score, 4),
            "ai_priority": self.ai_priority,
            "ai_mitre_technique": self.ai_mitre_technique,
            "ai_confidence": round(self.ai_confidence, 4),
            "ai_entities": self.ai_entities,
            "scoring_latency_ms": round(self.scoring_latency_ms, 2),
        }
