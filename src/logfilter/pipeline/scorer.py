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
        + w_nov  * novelty_score   (disabled by default until implemented)
        - dedup_penalty            (applied when is_duplicate=True)

All weights are read from config.yaml.
"""

from __future__ import annotations

import importlib
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from logfilter import telemetry
from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.ner import NERModel
from logfilter.models.syslog_classifier import SyslogClassifier
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.monitoring.drift_detector import DriftDetector
from logfilter.pipeline.events import ScoredEvent
from logfilter.pipeline.normalizer import NormalizedEvent

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent


def _resolve_path(value: str | Path) -> Path:
    """Resolve a path against ROOT if it is relative, otherwise keep absolute."""
    p = Path(value)
    return ROOT / p if not p.is_absolute() else p


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
    high = _probability_config(routing.get("high", 0.80), "routing.high")
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

# Attack tool signatures that XGBoost misses due to sparse training signal.
_ATTACK_TOOL_RE = re.compile(
    "sqlmap|nikto|nmap|masscan|zgrab|gobuster|dirbuster|wpscan|metasploit",
    re.IGNORECASE,
)
_ATTACK_TOOL_FLOOR = 0.60

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
# ScoredEvent lives in the dependency-light ``events`` module so the slim
# router/enricher images can import it without pulling in numpy and the ML stack.
# Re-exported here for backwards compatibility with existing import sites.


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
        syslog_classifier: SyslogClassifier | None = None,
        model_version: str = "",
        drift_detector: DriftDetector | None = None,
    ) -> None:
        self._cfg = config
        self._model_version = model_version
        scoring = config.get("scoring", {})
        weights = scoring.get("weights", {})

        self.w_cls = float(weights.get("classifier", 0.35))
        self.w_ent = float(weights.get("entity_boost", 0.25))
        self.w_ce = float(weights.get("cross_encoder", 0.40))
        self.w_nov = float(weights.get("novelty", 0.0))
        self.entity_boost_value = float(scoring.get("entity_boost_value", 0.20))
        self.dedup_penalty_value = float(scoring.get("dedup_penalty", 0.30))

        routing = scoring.get("routing", {})
        self.threshold_low, self.threshold_medium, self.threshold_high = _routing_thresholds(
            routing
        )

        models_cfg = config.get("models", {})
        classifier_path = _resolve_path(
            models_cfg.get("classifier", {}).get("path", "models/log_classifier.onnx")
        )
        scaler_path = _resolve_path(
            models_cfg.get("classifier", {}).get("scaler_path", "models/scaler.json")
        )
        feature_names_path = _resolve_path(
            models_cfg.get("classifier", {}).get("feature_names_path", "models/feature_names.json")
        )
        if self._model_version:
            version_root = _resolve_path(Path("models") / self._model_version)
            classifier_path = version_root / "log_classifier.onnx"
            scaler_path = version_root / "scaler.json"
            feature_names_path = version_root / "feature_names.json"

        self.classifier = classifier or LogClassifier(
            model_path=classifier_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
        )
        tier2_cfg = scoring.get("tier2", {})
        tier2_model_dir = _resolve_path(Path("models") / "tier2")
        if self._model_version:
            tier2_model_dir = _resolve_path(Path("models") / self._model_version / "tier2")
        self.tier2_classifier = tier2_classifier or Tier2Classifier(
            model_dir=tier2_model_dir,
            uncertainty_low=tier2_cfg.get("uncertainty_low", 0.10),
            uncertainty_high=tier2_cfg.get("uncertainty_high", 0.90),
        )
        ner_cfg = models_cfg.get("ner", {})
        biencoder_cfg = models_cfg.get("biencoder", {})
        cross_encoder_cfg = models_cfg.get("cross_encoder", {})

        ner_cache = ner_cfg.get("cache_dir", "")
        ner_revision = ner_cfg.get("revision", "")
        self.ner_model = ner_model or (
            NERModel(
                model_id=ner_cfg.get("model_id", NERModel.MODEL_ID),
                device=ner_cfg.get("device", "cpu"),
                batch_size=int(ner_cfg.get("batch_size", 32)),
                min_confidence=float(ner_cfg.get("min_confidence", 0.80)),
                cache_dir=ner_cache if ner_cache else None,
                revision=ner_revision if ner_revision else None,
            )
            if _enabled(ner_cfg.get("enabled", True))
            else DisabledNERModel()
        )
        bi_cache = biencoder_cfg.get("cache_dir", "")
        bi_revision = biencoder_cfg.get("revision", "")
        self.biencoder = biencoder or (
            BiEncoderModel(
                model_id=biencoder_cfg.get("model_id", BiEncoderModel.MODEL_ID),
                device=biencoder_cfg.get("device", "cpu"),
                batch_size=int(biencoder_cfg.get("batch_size", 64)),
                dedup_threshold=float(biencoder_cfg.get("dedup_threshold", 0.95)),
                dedup_window_minutes=float(biencoder_cfg.get("dedup_window_minutes", 5.0)),
                faiss_top_k=int(biencoder_cfg.get("faiss_top_k", 3)),
                mitre_techniques_path=_resolve_path(
                    models_cfg.get("mitre_techniques_path", "config/mitre_techniques.json")
                ),
                cache_dir=bi_cache if bi_cache else None,
                revision=bi_revision if bi_revision else None,
            )
            if _enabled(biencoder_cfg.get("enabled", True))
            else DisabledBiEncoderModel()
        )
        ce_cache = cross_encoder_cfg.get("cache_dir", "")
        ce_revision = cross_encoder_cfg.get("revision", "")
        self.cross_encoder = cross_encoder or (
            CrossEncoderModel(
                model_id=cross_encoder_cfg.get("model_id", CrossEncoderModel.MODEL_ID),
                device=cross_encoder_cfg.get("device", "cpu"),
                batch_size=int(cross_encoder_cfg.get("batch_size", 16)),
                cache_dir=ce_cache if ce_cache else None,
                revision=ce_revision if ce_revision else None,
            )
            if _enabled(cross_encoder_cfg.get("enabled", True))
            else DisabledCrossEncoderModel()
        )
        drift_cfg = config.get("monitoring", {}).get("drift", {})
        self.drift_detector: DriftDetector | None
        if drift_detector is not None:
            self.drift_detector = drift_detector
        elif _enabled(drift_cfg.get("enabled", False)):
            self.drift_detector = DriftDetector(
                window_size=int(drift_cfg.get("window_size", 1000)),
                psi_threshold=float(drift_cfg.get("psi_threshold", 0.25)),
                check_interval=int(drift_cfg.get("check_interval", 100)),
                auto_fallback=_enabled(drift_cfg.get("auto_fallback", True)),
            )
        else:
            self.drift_detector = None

        self.syslog_classifier = syslog_classifier or SyslogClassifier()

        self._feature_cache_names: tuple[str, ...] = ()
        self._feature_cache_tokens: list[tuple[str, tuple[str, ...]]] = []
        self._syslog_feature_cache_names: tuple[str, ...] = ()
        self._syslog_feature_cache_tokens: list[tuple[str, tuple[str, ...]]] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def preload_models(self) -> None:
        """Eagerly load all models to eliminate cold-start latency."""
        logger.info("Pre-loading classifier")
        _ = self.classifier.is_ready()
        logger.info("Pre-loading syslog classifier")
        _ = self.syslog_classifier.is_ready()
        logger.info("Pre-loading tier2 classifier")
        _ = self.tier2_classifier.is_ready()
        logger.info("Pre-loading biencoder")
        _ = self.biencoder.check_dedup_and_retrieve_batch([])
        logger.info("Pre-loading NER model")
        _ = self.ner_model.extract_batch([])
        logger.info("Pre-loading cross encoder")
        _ = self.cross_encoder.score_batch([], [])

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
            if self.drift_detector is not None:
                for se in scored:
                    self.drift_detector.record_score(se.classifier_score)

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
            for se, ev in zip(scored, events):
                if _ATTACK_TOOL_RE.search(ev.raw):
                    se.classifier_score = max(se.classifier_score, _ATTACK_TOOL_FLOOR)
                    se.dedup_penalty = 0.0
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
        Lightweight Sigma rule matching for syslog events.

        Sigma rules are loaded with pySigma and evaluated against the raw
        normalized log text.  This is a simplified, text-only matcher: it
        honours ``contains`` modifiers by doing substring searches in the
        log text, but it does not perform structured field extraction (e.g.
        ``Image``, ``ParentImage``) because syslog events lack those fields.
        Rules that reference un-mappable fields are skipped with a debug log.

        For full fidelity Sigma evaluation, pipe the enriched event to a
        dedicated Sigma backend (Splunk, Elastic, or a field-extracted
        normaliser) rather than relying on this in-process matcher.
        """
        with telemetry.start_as_current_span(
            "scorer.sigma.apply",
            {"logfilter.batch_size": len(events)},
        ) as span:
            try:
                sigma_collection = importlib.import_module("sigma.collection")
                sigma_collection_cls = sigma_collection.SigmaCollection
            except ImportError:
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
                    for rule in rules:
                        if self._sigma_rule_matches(rule, ev.text):
                            se.sigma_matched = True
                            se.sigma_rule_ids.append(str(rule.id or rule.title))
                            match_count += 1
                span.set_attribute("logfilter.sigma.match_count", match_count)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                telemetry.record_exception(span, exc)
                logger.warning("Sigma evaluation failed", error=str(exc))

    def _sigma_rule_matches(self, rule: Any, event_text: str) -> bool:
        """
        Evaluate a single SigmaRule against a raw log text.

        Returns ``True`` if any mappable ``contains`` detection item
        matches the event text.
        """
        detection = getattr(rule, "detection", None)
        if detection is None:
            return False

        text_lower = event_text.lower()

        for name, det in detection.detections.items():
            for item in det.detection_items:
                field = getattr(item, "field", None)
                if field is None:
                    continue

                # Syslog text fields we can reasonably map to raw log text
                mappable_fields = {
                    "CommandLine",
                    "ServiceName",
                    "TargetObject",
                    "Details",
                    "QueryName",
                    "QueryResults",
                    "Path",
                    "LogonType",
                    "TargetUserName",
                    "SubjectUserName",
                    "ObjectName",
                    "ProcessName",
                    "ParentCommandLine",
                    "TargetFilename",
                }
                if field not in mappable_fields:
                    continue

                modifiers = getattr(item, "modifiers", [])
                modifier_names = {
                    m.__name__ if hasattr(m, "__name__") else str(m) for m in modifiers
                }

                for val in item.value:
                    # Extract plain string tokens from SigmaString
                    plain_parts = [
                        str(part)
                        for part in val.s
                        if not hasattr(part, "name")  # skip SpecialChars enum members
                    ]
                    search_text = " ".join(plain_parts).strip().lower()

                    if not search_text:
                        continue

                    if "SigmaContainsModifier" in modifier_names:
                        if search_text in text_lower:
                            return True
                    elif "SigmaEndswithModifier" in modifier_names:
                        if text_lower.endswith(search_text):
                            return True
                    elif "SigmaStartswithModifier" in modifier_names:
                        if text_lower.startswith(search_text):
                            return True
                    elif not modifiers:
                        # No modifier = exact match on a word boundary
                        for word in text_lower.split():
                            if word.strip(".,;:!?") == search_text:
                                return True
        return False

    def _apply_classifier(self, scored: list[ScoredEvent], events: list[NormalizedEvent]) -> None:
        """Populate classifier_score using the trained event-count classifier."""
        from logfilter.pipeline.normalizer import LogSourceType

        syslog_indices = [
            i for i, ev in enumerate(events)
            if ev.source_type in (LogSourceType.SYSLOG, LogSourceType.WEB,
                                  LogSourceType.FIREWALL, LogSourceType.WINEVENT)
        ]
        hdfs_indices = [i for i in range(len(events)) if i not in syslog_indices]

        with telemetry.start_as_current_span(
            "scorer.tier1.classifier",
            {"logfilter.batch_size": len(events)},
        ) as span:
            try:
                if syslog_indices and self.syslog_classifier.is_ready():
                    syslog_events = [events[i] for i in syslog_indices]
                    syslog_vectors = self._syslog_feature_vectors(syslog_events)
                    syslog_probs = self.syslog_classifier.predict_proba(syslog_vectors)
                    for idx, prob in zip(syslog_indices, syslog_probs):
                        scored[idx].classifier_score = max(0.0, min(1.0, float(prob)))

                if hdfs_indices:
                    hdfs_events = [events[i] for i in hdfs_indices]
                    feature_vectors = self._classifier_feature_vectors(hdfs_events)
                    span.set_attribute("logfilter.feature_count", int(feature_vectors.shape[1]))
                    hdfs_probs = self.classifier.predict_proba(feature_vectors)
                    for idx, prob in zip(hdfs_indices, hdfs_probs):
                        scored[idx].classifier_score = max(0.0, min(1.0, float(prob)))

                if not self.classifier.is_ready() and not self.syslog_classifier.is_ready():
                    for se in scored:
                        se.score_degraded = True
            except (ValueError, IndexError, TypeError, RuntimeError) as exc:
                telemetry.record_exception(span, exc)
                logger.warning("Classifier scoring failed; using neutral scores", error=str(exc))
                for se in scored:
                    se.classifier_score = 0.5
                    se.score_degraded = True

        escalation_indices = [
            i
            for i, se in enumerate(scored)
            if i not in syslog_indices
            and self.tier2_classifier.should_escalate(se.classifier_score)
        ]
        with telemetry.start_as_current_span(
            "scorer.tier2.classifier",
            {"logfilter.escalation_count": len(escalation_indices)},
        ) as span:
            if not escalation_indices:
                return
            if self.drift_detector is not None and self.drift_detector.is_fallback_active():
                span.set_attribute("logfilter.tier2.fallback_active", True)
                logger.warning(
                    "Tier-2 fallback active due to model drift; keeping Tier-1 scores",
                    drift_psi=round(self.drift_detector.check_drift().psi, 4),
                )
                return
            if not self.tier2_classifier.is_ready():
                span.set_attribute("logfilter.tier2.ready", False)
                logger.warning("Tier-2 classifier unavailable; keeping Tier-1 uncertain scores")
                return

            tier2_texts = [events[i].raw for i in escalation_indices]
            try:
                tier2_probs = self.tier2_classifier.predict_proba(tier2_texts)
            except (ValueError, IndexError, TypeError, RuntimeError) as exc:
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
                    hits = 0
                    for ft in feature_tokens:
                        if ft in text_tokens:
                            hits += 1
                        elif len(ft) >= 4 and any(ft in tt for tt in text_tokens):
                            hits += 1
                    threshold = 0.50 if len(feature_tokens) <= 3 else 0.65
                    if hits / len(feature_tokens) >= threshold:
                        vectors[row, col] += 1.0
        return vectors

    def _syslog_feature_vectors(self, events: list[NormalizedEvent]) -> np.ndarray:
        """Convert normalized log text into the 100-feature syslog classifier vector."""
        feature_names = tuple(self.syslog_classifier.feature_names)
        n_features = len(feature_names) or 100
        vectors = np.zeros((len(events), n_features), dtype=np.float32)
        if not feature_names:
            return vectors

        prepared = self._prepared_syslog_features(feature_names)
        for row, event in enumerate(events):
            text = f"{event.text} {event.raw}".lower()
            text_tokens = set(_TOKEN_RE.findall(text))
            for col, (feature_text, feature_tokens) in enumerate(prepared):
                if feature_text and feature_text in text:
                    vectors[row, col] += 1.0
                    continue
                if feature_tokens and text_tokens:
                    hits = 0
                    for ft in feature_tokens:
                        if ft in text_tokens:
                            hits += 1
                        elif len(ft) >= 4 and any(ft in tt for tt in text_tokens):
                            hits += 1
                    threshold = 0.50 if len(feature_tokens) <= 3 else 0.65
                    if hits / len(feature_tokens) >= threshold:
                        vectors[row, col] += 1.0
        return vectors

    def _prepared_syslog_features(
        self, feature_names: tuple[str, ...]
    ) -> list[tuple[str, tuple[str, ...]]]:
        if feature_names == self._syslog_feature_cache_names:
            return self._syslog_feature_cache_tokens

        prepared: list[tuple[str, tuple[str, ...]]] = []
        for name in feature_names:
            lowered = name.lower()
            normalized = lowered.replace("+", " ")
            tokens = tuple(
                token
                for token in _TOKEN_RE.findall(normalized)
                if len(token) > 2 and token not in _FEATURE_STOPWORDS
            )
            prepared.append((normalized, tokens))

        self._syslog_feature_cache_names = feature_names
        self._syslog_feature_cache_tokens = prepared
        return prepared

    def _prepared_classifier_features(
        self, feature_names: tuple[str, ...]
    ) -> list[tuple[str, tuple[str, ...]]]:
        if feature_names == self._feature_cache_names:
            return self._feature_cache_tokens

        prepared: list[tuple[str, tuple[str, ...]]] = []
        for name in feature_names:
            lowered = name.lower()
            normalized = lowered.replace("+", " ")
            tokens = tuple(
                token
                for token in _TOKEN_RE.findall(normalized)
                if len(token) > 2 and token not in _FEATURE_STOPWORDS
            )
            prepared.append((normalized, tokens))

        self._feature_cache_names = feature_names
        self._feature_cache_tokens = prepared
        return prepared

    def _compute_score(self, se: ScoredEvent) -> float:
        """Compute final composite threat score."""
        # Sigma match → immediate HIGH signal
        if se.sigma_matched:
            se.classifier_score = max(se.classifier_score, 0.90)

        # Only apply dedup penalty when cross-encoder actually contributed;
        # otherwise the penalty unfairly zeroes out scores for duplicates
        # that never got a cross-encoder evaluation.
        dedup = se.dedup_penalty if se.cross_encoder_max > 0 else 0.0

        score = (
            self.w_cls * se.classifier_score
            + self.w_ent * se.entity_boost
            + self.w_ce * se.cross_encoder_max
            + self.w_nov * se.novelty_score
            - dedup
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
