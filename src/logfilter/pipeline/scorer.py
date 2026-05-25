"""
Tiered AI scoring pipeline.

Architecture (per the session design doc):

  Tier 1 — Sigma rule engine (fast pattern matching, runs on every event)
  Tier 2 — BiEncoder embedding + FAISS dedup + ATT&CK candidate retrieval
  Tier 3 — NER extraction + CrossEncoder relevance scoring
            (only on non-duplicates, top-k candidates only)

Final score formula:
  score = w_cls  * classifier_score
        + w_ent  * entity_boost    (0.2 if high-value IOC/malware/CVE found)
        + w_ce   * cross_encoder_score  (max over top-k ATT&CK candidates)
        + w_nov  * novelty_score   (placeholder — 0.5 default)
        - dedup_penalty            (applied when is_duplicate=True)

All weights are read from config.yaml.
"""

from __future__ import annotations

import importlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from logfilter import telemetry
from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.ner import NERModel
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.pipeline.normalizer import NormalizedEvent

logger = structlog.get_logger(__name__)


def _enabled(value: Any, default: bool = True) -> bool:
    """Parse config values that may arrive as booleans or env-substituted strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _probability_config(value: Any, name: str) -> float:
    """Parse and validate a probability threshold from config or env."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric probability") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return parsed


def _routing_thresholds(routing: dict[str, Any]) -> tuple[float, float, float]:
    """Return validated low/medium/high routing thresholds."""
    high = _probability_config(routing.get("high", 0.85), "routing.high")
    medium = _probability_config(routing.get("medium", 0.50), "routing.medium")
    low = _probability_config(routing.get("low", 0.20), "routing.low")
    if not low < medium < high:
        raise ValueError("routing thresholds must satisfy low < medium < high")
    return low, medium, high


class DisabledNERModel(NERModel):
    """No-op NER stage for local validation or deployments without NER artifacts."""

    def extract_batch(self, texts: list[str]) -> list[Any]:
        from logfilter.models.ner import ExtractedEntities

        return [ExtractedEntities() for _ in texts]


class DisabledBiEncoderModel(BiEncoderModel):
    """No-op BiEncoder stage that marks events as non-duplicates with no candidates."""

    def check_dedup_and_retrieve_batch(self, texts: list[str]) -> list[tuple[Any, list[Any]]]:
        from logfilter.models.biencoder import DedupResult

        return [(DedupResult(is_duplicate=False, similarity=0.0), []) for _ in texts]


class DisabledCrossEncoderModel(CrossEncoderModel):
    """No-op CrossEncoder stage for local validation without downloading HF models."""

    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[Any]]:
        del candidates_per_log
        return [[] for _ in log_texts]


_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")
_FEATURE_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "return",
    "the",
    "to",
    "with",
}

# ── Score result ───────────────────────────────────────────────────────────────


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


# ── Scorer ─────────────────────────────────────────────────────────────────────


class LogScorer:
    """
    Orchestrates the full tiered scoring pipeline.

    Parameters
    ----------
    config : dict
        Loaded from config/config.yaml (the 'scoring' and 'models' sections).
    classifier : LogClassifier | None
        Pre-instantiated classifier (optional; created lazily if None).
    tier2_classifier : Tier2Classifier | None
        Pre-instantiated Tier-2 classifier for uncertain Tier-1 scores.
    ner_model : NERModel | None
    biencoder : BiEncoderModel | None
    cross_encoder : CrossEncoderModel | None
    """

    def __init__(
        self,
        config: dict[str, Any],
        classifier: LogClassifier | None = None,
        tier2_classifier: Tier2Classifier | None = None,
        ner_model: NERModel | None = None,
        biencoder: BiEncoderModel | None = None,
        cross_encoder: CrossEncoderModel | None = None,
        model_version: str = "",
    ) -> None:
        self._cfg = config
        self._model_version = model_version
        scoring = config.get("scoring", {})
        weights = scoring.get("weights", {})

        self.w_cls = float(weights.get("classifier", 0.30))
        self.w_ent = float(weights.get("entity_boost", 0.20))
        self.w_ce = float(weights.get("cross_encoder", 0.35))
        self.w_nov = float(weights.get("novelty", 0.15))
        self.entity_boost_value = float(scoring.get("entity_boost_value", 0.20))
        self.dedup_penalty_value = float(scoring.get("dedup_penalty", 0.30))

        routing = scoring.get("routing", {})
        self.threshold_low, self.threshold_medium, self.threshold_high = _routing_thresholds(
            routing
        )

        models_cfg = config.get("models", {})
        classifier_path = models_cfg.get("classifier", {}).get("path", "models/log_classifier.onnx")
        scaler_path = models_cfg.get("classifier", {}).get("scaler_path", "models/scaler.json")
        feature_names_path = models_cfg.get("classifier", {}).get(
            "feature_names_path", "models/feature_names.json"
        )
        if self._model_version:
            version_root = Path("models") / self._model_version
            classifier_path = str(version_root / "log_classifier.onnx")
            scaler_path = str(version_root / "scaler.json")
            feature_names_path = str(version_root / "feature_names.json")

        self.classifier = classifier or LogClassifier(
            model_path=classifier_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
        )
        tier2_cfg = scoring.get("tier2", {})
        tier2_model_dir = Path("models") / "tier2"
        if self._model_version:
            tier2_model_dir = Path("models") / self._model_version / "tier2"
        self.tier2_classifier = tier2_classifier or Tier2Classifier(
            model_dir=tier2_model_dir,
            uncertainty_low=tier2_cfg.get("uncertainty_low", 0.10),
            uncertainty_high=tier2_cfg.get("uncertainty_high", 0.90),
        )
        ner_cfg = models_cfg.get("ner", {})
        biencoder_cfg = models_cfg.get("biencoder", {})
        cross_encoder_cfg = models_cfg.get("cross_encoder", {})

        self.ner_model = ner_model or (
            NERModel(
                model_id=ner_cfg.get("model_id", NERModel.MODEL_ID),
                device=ner_cfg.get("device", "cpu"),
                batch_size=int(ner_cfg.get("batch_size", 32)),
                min_confidence=float(ner_cfg.get("min_confidence", 0.80)),
            )
            if _enabled(ner_cfg.get("enabled", True))
            else DisabledNERModel()
        )
        self.biencoder = biencoder or (
            BiEncoderModel(
                model_id=biencoder_cfg.get("model_id", BiEncoderModel.MODEL_ID),
                device=biencoder_cfg.get("device", "cpu"),
                batch_size=int(biencoder_cfg.get("batch_size", 64)),
                dedup_threshold=float(biencoder_cfg.get("dedup_threshold", 0.95)),
                dedup_window_minutes=float(biencoder_cfg.get("dedup_window_minutes", 5.0)),
                faiss_top_k=int(biencoder_cfg.get("faiss_top_k", 3)),
                mitre_techniques_path=models_cfg.get(
                    "mitre_techniques_path", "config/mitre_techniques.json"
                ),
            )
            if _enabled(biencoder_cfg.get("enabled", True))
            else DisabledBiEncoderModel()
        )
        self.cross_encoder = cross_encoder or (
            CrossEncoderModel(
                model_id=cross_encoder_cfg.get("model_id", CrossEncoderModel.MODEL_ID),
                device=cross_encoder_cfg.get("device", "cpu"),
                batch_size=int(cross_encoder_cfg.get("batch_size", 16)),
            )
            if _enabled(cross_encoder_cfg.get("enabled", True))
            else DisabledCrossEncoderModel()
        )
        self._feature_cache_names: tuple[str, ...] = ()
        self._feature_cache_tokens: list[tuple[str, tuple[str, ...]]] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def score(self, event: NormalizedEvent) -> ScoredEvent:
        """Score a single normalized event. Thread-safe."""
        results = self.score_batch([event])
        return results[0]

    def score_batch(self, events: list[NormalizedEvent]) -> list[ScoredEvent]:
        """
        Score a batch of normalized events through the full tiered pipeline.

        Returns a list of ScoredEvent, one per input event.
        """
        with telemetry.start_as_current_span(
            "scorer.score_batch",
            {"logfilter.batch_size": len(events), "logfilter.model_version": self._model_version},
        ) as span:
            t0 = time.perf_counter()

            scored = [self._init_scored(ev) for ev in events]
            texts = [ev.text for ev in events]

            # ── Tier 1: Sigma ──────────────────────────────────────────────────
            self._apply_sigma(scored, events)

            # ── Tier 1b: trained classifier ────────────────────────────────────
            self._apply_classifier(scored, events)

            # ── Tier 2: BiEncoder dedup + ATT&CK candidate retrieval ──────────
            with telemetry.start_as_current_span(
                "scorer.tier2.biencoder",
                {"logfilter.batch_size": len(texts)},
            ) as bi_span:
                bi_results = self.biencoder.check_dedup_and_retrieve_batch(texts)
                duplicate_count = 0
                candidate_count = 0
                for se, (dedup_res, candidates) in zip(scored, bi_results):
                    se.is_duplicate = dedup_res.is_duplicate
                    se.dedup_similarity = dedup_res.similarity
                    se.attack_candidates = [
                        {
                            "id": c.technique_id,
                            "name": c.name,
                            "description": c.description,
                            "bi_similarity": round(c.similarity, 4),
                        }
                        for c in candidates
                    ]
                    candidate_count += len(candidates)
                    if se.is_duplicate:
                        duplicate_count += 1
                        se.dedup_penalty = self.dedup_penalty_value
                telemetry.set_span_attributes(
                    bi_span,
                    {
                        "logfilter.duplicate_count": duplicate_count,
                        "logfilter.attack_candidate_count": candidate_count,
                    },
                )

            # ── Tier 3: NER + CrossEncoder (skip duplicates) ───────────────────
            non_dup_indices = [i for i, se in enumerate(scored) if not se.is_duplicate]
            if non_dup_indices:
                nd_texts = [texts[i] for i in non_dup_indices]
                nd_scored = [scored[i] for i in non_dup_indices]

                # NER
                with telemetry.start_as_current_span(
                    "scorer.ner.extract_batch",
                    {"logfilter.batch_size": len(nd_texts)},
                ) as ner_span:
                    ner_results = self.ner_model.extract_batch(nd_texts)
                    high_value_count = 0
                    for se, ner_result in zip(nd_scored, ner_results):
                        se.entities = ner_result.to_dict()
                        se.ai_entities = ner_result.flat_entity_string()
                        if ner_result.has_high_value_entities:
                            high_value_count += 1
                            se.entity_boost = self.entity_boost_value
                    ner_span.set_attribute("logfilter.high_value_entity_count", high_value_count)

                # CrossEncoder
                candidates_per_event = [
                    [
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "description": c["description"],
                        }
                        for c in se.attack_candidates
                    ]
                    for se in nd_scored
                ]
                with telemetry.start_as_current_span(
                    "scorer.cross_encoder.score_batch",
                    {
                        "logfilter.batch_size": len(nd_texts),
                        "logfilter.attack_candidate_count": sum(
                            len(candidates) for candidates in candidates_per_event
                        ),
                    },
                ) as ce_span:
                    ce_results = self.cross_encoder.score_batch(nd_texts, candidates_per_event)
                    matched_count = 0
                    for se, ce_scores in zip(nd_scored, ce_results):
                        se.cross_encoder_scores = [
                            {"id": s.technique_id, "name": s.name, "score": round(s.score, 4)}
                            for s in ce_scores
                        ]
                        if ce_scores:
                            matched_count += 1
                            se.cross_encoder_max = ce_scores[0].score
                            se.ai_mitre_technique = ce_scores[0].technique_id
                    ce_span.set_attribute("logfilter.cross_encoder_match_count", matched_count)

            # ── Final score + routing ──────────────────────────────────────────
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            high_count = 0
            medium_count = 0
            sigma_count = 0
            for se in scored:
                se.ai_threat_score = self._compute_score(se)
                se.ai_priority = self._routing_label(se)
                se.ai_confidence = self._confidence(se)
                if se.ai_priority == "HIGH":
                    high_count += 1
                elif se.ai_priority == "MEDIUM":
                    medium_count += 1
                if se.sigma_matched:
                    sigma_count += 1
                # Total wall-clock for the batch (per-event latency is meaningless
                # because models batch-encode). Each event records the full batch time.
                se.scoring_latency_ms = elapsed_ms

            telemetry.set_span_attributes(
                span,
                {
                    "logfilter.elapsed_ms": elapsed_ms,
                    "logfilter.high_priority_count": high_count,
                    "logfilter.medium_priority_count": medium_count,
                    "logfilter.sigma_match_count": sigma_count,
                },
            )
            logger.debug(
                "Batch scored",
                n=len(scored),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return scored

    # ── internal helpers ───────────────────────────────────────────────────────

    def _init_scored(self, event: NormalizedEvent) -> ScoredEvent:
        return ScoredEvent(
            source_type=event.source_type.value,
            timestamp=event.timestamp,
            host=event.host,
            raw=event.raw,
            normalized_text=event.text,
            fields=event.fields,
        )

    def _apply_sigma(self, scored: list[ScoredEvent], events: list[NormalizedEvent]) -> None:
        """
        Apply Sigma rules via sigma-cli / pySigma.

        Sigma rules live in config/sigma_rules/*.yml.
        If sigma-cli is not installed or no rules exist, this is a no-op.
        """
        with telemetry.start_as_current_span(
            "scorer.sigma.apply",
            {"logfilter.batch_size": len(events)},
        ) as span:
            try:
                sigma_collection = importlib.import_module("sigma.collection")
                sigma_collection_cls = sigma_collection.SigmaCollection
            except ImportError:
                # sigma is optional; skip gracefully
                span.set_attribute("logfilter.sigma.available", False)
                return

            sigma_rules_dir = Path("config/sigma_rules")
            if not sigma_rules_dir.exists() or not list(sigma_rules_dir.glob("*.yml")):
                span.set_attribute("logfilter.sigma.rule_count", 0)
                return

            try:
                rules = sigma_collection_cls.load_ruleset([str(sigma_rules_dir)])
                match_count = 0
                for se, ev in zip(scored, events):
                    # Simplified: we match raw text against rule titles/descriptions
                    # In production: pipe events through a full Sigma evaluation engine
                    for rule in rules:
                        detection = getattr(rule, "detection", {}) or {}
                        rule_text = (
                            (rule.title or "").lower()
                            + " "
                            + " ".join(str(d) for d in detection)
                        )
                        if any(kw in ev.text.lower() for kw in rule_text.split() if len(kw) > 4):
                            se.sigma_matched = True
                            se.sigma_rule_ids.append(str(rule.id or rule.title))
                            match_count += 1
                span.set_attribute("logfilter.sigma.match_count", match_count)
            except Exception as exc:  # noqa: BLE001
                telemetry.record_exception(span, exc)
                logger.warning("Sigma evaluation failed", error=str(exc))

    def _apply_classifier(self, scored: list[ScoredEvent], events: list[NormalizedEvent]) -> None:
        """Populate classifier_score using the trained event-count classifier."""
        with telemetry.start_as_current_span(
            "scorer.tier1.classifier",
            {"logfilter.batch_size": len(events)},
        ) as span:
            try:
                feature_vectors = self._classifier_feature_vectors(events)
                span.set_attribute("logfilter.feature_count", int(feature_vectors.shape[1]))
                probabilities = self.classifier.predict_proba(feature_vectors)
            except Exception as exc:  # noqa: BLE001
                telemetry.record_exception(span, exc)
                logger.warning("Classifier scoring failed; using neutral scores", error=str(exc))
                probabilities = np.full(len(events), 0.5, dtype=np.float32)

        for se, prob in zip(scored, probabilities):
            se.classifier_score = max(0.0, min(1.0, float(prob)))

        escalation_indices = [
            i
            for i, se in enumerate(scored)
            if self.tier2_classifier.should_escalate(se.classifier_score)
        ]
        with telemetry.start_as_current_span(
            "scorer.tier2.classifier",
            {"logfilter.escalation_count": len(escalation_indices)},
        ) as span:
            if not escalation_indices:
                return
            if not self.tier2_classifier.is_ready():
                span.set_attribute("logfilter.tier2.ready", False)
                logger.warning("Tier-2 classifier unavailable; keeping Tier-1 uncertain scores")
                return

            tier2_texts = [events[i].raw for i in escalation_indices]
            try:
                tier2_probs = self.tier2_classifier.predict_proba(tier2_texts)
            except Exception as exc:  # noqa: BLE001
                telemetry.record_exception(span, exc)
                logger.warning(
                    "Tier-2 classifier scoring failed; keeping Tier-1 scores", error=str(exc)
                )
                return

            for index, prob in zip(escalation_indices, tier2_probs):
                score = max(0.0, min(1.0, float(prob)))
                scored[index].classifier_score = score
                scored[index].tier2_score = score
                scored[index].tier2_used = True

    def _classifier_feature_vectors(self, events: list[NormalizedEvent]) -> np.ndarray:
        """
        Convert normalized log text into the bag-of-event vector expected by the
        HDFS TraceBench classifier artifacts.
        """
        feature_names = tuple(getattr(self.classifier, "feature_names", []) or ())
        n_features = len(feature_names) or int(
            getattr(self.classifier, "expected_feature_count", 0) or 1
        )
        vectors = np.zeros((len(events), n_features), dtype=np.float32)
        if not feature_names:
            return vectors

        prepared_features = self._prepared_classifier_features(feature_names)
        for row, event in enumerate(events):
            text = f"{event.text} {event.raw}".lower()
            text_tokens = set(_TOKEN_RE.findall(text))
            for col, (feature_text, feature_tokens) in enumerate(prepared_features):
                if feature_text and feature_text in text:
                    vectors[row, col] += 1.0
                    continue
                if feature_tokens and text_tokens:
                    hits = sum(1 for token in feature_tokens if token in text_tokens)
                    if hits / len(feature_tokens) >= 0.75:
                        vectors[row, col] += 1.0
        return vectors

    def _prepared_classifier_features(
        self, feature_names: tuple[str, ...]
    ) -> list[tuple[str, tuple[str, ...]]]:
        if feature_names == self._feature_cache_names:
            return self._feature_cache_tokens

        prepared: list[tuple[str, tuple[str, ...]]] = []
        for name in feature_names:
            lowered = name.lower()
            tokens = tuple(
                token
                for token in _TOKEN_RE.findall(lowered)
                if len(token) > 2 and token not in _FEATURE_STOPWORDS
            )
            prepared.append((lowered, tokens))

        self._feature_cache_names = feature_names
        self._feature_cache_tokens = prepared
        return prepared

    def _compute_score(self, se: ScoredEvent) -> float:
        """Compute final composite threat score."""
        # Sigma match → immediate HIGH signal
        if se.sigma_matched:
            se.classifier_score = max(se.classifier_score, 0.90)

        score = (
            self.w_cls * se.classifier_score
            + self.w_ent * se.entity_boost
            + self.w_ce * se.cross_encoder_max
            + self.w_nov * se.novelty_score
            - se.dedup_penalty
        )
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _routing_label(self, se: ScoredEvent) -> str:
        if se.sigma_matched or se.ai_threat_score >= self.threshold_high:
            return "HIGH"
        if se.ai_threat_score >= self.threshold_medium:
            return "MEDIUM"
        if se.ai_threat_score >= self.threshold_low:
            return "LOW"
        return "INFO"

    def _confidence(self, se: ScoredEvent) -> float:
        """Average non-zero model confidence signals."""
        signals = [
            se.classifier_score if se.classifier_score > 0 else None,
            se.entities.get("confidence", 0) if se.entities else None,
            se.cross_encoder_max if se.cross_encoder_max > 0 else None,
        ]
        valid = [s for s in signals if s is not None]
        return float(sum(valid) / len(valid)) if valid else 0.0
