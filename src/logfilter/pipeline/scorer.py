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

import hashlib
import importlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog  # type: ignore[import-not-found]
import yaml

from logfilter import telemetry
from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.endpoint_novelty import EndpointNoveltyModel
from logfilter.models.flow_classifier import FlowClassifier
from logfilter.models.ner import NERModel
from logfilter.models.syslog_classifier import SyslogClassifier
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.monitoring.drift_detector import DriftDetector
from logfilter.monitoring.novelty_detector import NoveltyDetector
from logfilter.pipeline.endpoint_features import extract_endpoint_vector
from logfilter.pipeline.events import ScoredEvent
from logfilter.pipeline.flow_features import FlowWindowAggregator, extract_flow_vector
from logfilter.pipeline.normalizer import NormalizedEvent

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent
_SIGMA_TEXT_FIELDS = frozenset(
    {
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
)
_SIGMA_TEXT_MODIFIERS = frozenset({"contains", "endswith", "startswith", "exact"})


@dataclass(frozen=True, slots=True)
class _SigmaTextPattern:
    field: str
    modifier: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SigmaTextRule:
    rule_id: str
    patterns: tuple[_SigmaTextPattern, ...]


def _parse_sigma_document(document: Any) -> _SigmaTextRule | None:
    """Parse the safe text-matching subset used by the in-process scorer."""
    if not isinstance(document, dict):
        return None
    detection = document.get("detection")
    if not isinstance(detection, dict):
        return None
    condition = detection.get("condition")
    if not isinstance(condition, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", condition):
        return None
    selection = detection.get(condition)
    if not isinstance(selection, dict):
        return None

    patterns: list[_SigmaTextPattern] = []
    for field_expression, raw_values in selection.items():
        if not isinstance(field_expression, str):
            continue
        field, separator, modifier = field_expression.partition("|")
        modifier = modifier if separator else "exact"
        if field not in _SIGMA_TEXT_FIELDS or modifier not in _SIGMA_TEXT_MODIFIERS:
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        clean_values = tuple(value for value in values if isinstance(value, str) and value)
        if clean_values:
            patterns.append(_SigmaTextPattern(field, modifier, clean_values))

    if not patterns:
        return None
    rule_id = str(document.get("id") or document.get("title") or "unnamed")
    return _SigmaTextRule(rule_id, tuple(patterns))


def _load_sigma_text_rules(rules_dir: Path) -> list[_SigmaTextRule]:
    rules: list[_SigmaTextRule] = []
    for rule_path in sorted(rules_dir.glob("*.yml")):
        try:
            with rule_path.open(encoding="utf-8") as handle:
                for document in yaml.safe_load_all(handle):
                    rule = _parse_sigma_document(document)
                    if rule is not None:
                        rules.append(rule)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Sigma rule file skipped", path=str(rule_path), error=str(exc))
    return rules


_bot_classifier_module: Any = importlib.import_module("logfilter.models.bot_flow_classifier")
_flow_fusion_module: Any = importlib.import_module("logfilter.models.flow_fusion")


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


def _strict_model_loading() -> bool:
    return _enabled(os.environ.get("LOGFILTER_MODELS_STRICT"), default=False)


def _float_value(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _int_value(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bounded_probability(value: Any) -> float:
    return max(0.0, min(1.0, _float_value(value, "model probability")))


def _probability_config(value: Any, name: str) -> float:
    """Parse and validate a probability threshold from config or env."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric probability") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return parsed


def _weight_value(value: Any, name: str) -> float:
    parsed = _float_value(value, name)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
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

    def check_dedup_retrieve_and_score_novelty_batch(
        self, texts: list[str]
    ) -> list[tuple[Any, list[Any], Any]]:
        from logfilter.models.biencoder import DedupResult
        from logfilter.monitoring.novelty_detector import NoveltyResult

        return [
            (
                DedupResult(is_duplicate=False, similarity=0.0),
                [],
                NoveltyResult(score=0.0, distance=0.0, baseline_size=0),
            )
            for _ in texts
        ]


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

# Attack tool signatures and threat keywords that XGBoost misses due to sparse
# training signal.  The regex is intentionally broad — these are high-signal
# strings that appear in real-world attack logs.  Matching applies a floor
# score so the event is never buried below INFO priority.
_ATTACK_TOOL_RE = re.compile(
    r"(?:"
    # Tool names
    r"sqlmap|nikto|nmap|masscan|zgrab|gobuster|dirbuster|wpscan|metasploit|mimikatz"
    r"|"
    # Attack categories
    r"xss\s+attack|sql\s*injection|command\s+injection|reverse\s+shell"
    r"|"
    r"credential\s+dump|credential\s+theft|password\s+spray"
    r"|"
    r"c2\s+communication|c&c\s+communication|beaconing\s+behavior"
    r"|"
    r"known\s+malware|malware\s+hash|trojan|ransomware|rootkit"
    r"|"
    r"privilege\s+escalation|lateral\s+movement|data\s+exfiltration"
    r"|"
    # Suspicious processes
    r"procdump\.exe|mimikatz\.exe|powershell\.exe\s+-enc|bash\s+-i\s+>&\s*/dev/tcp"
    r"|"
    r"invoke-webrequest|invoke-expression|sekurlsa::logonpasswords"
    r"|"
    # Firewall/IDS alert keywords
    r"port\s+scan\s+detected|syn\s+flood|brute\s+force\s+detected"
    r"|"
    r"outbound\s+connection\s+to\s+evil|dns\s+query\s+to\s+.*evil"
    r"|"
    # Additional attack patterns
    r"invalid\s+user.*from\s+\d|maximum\s+authentication\s+attempts\s+exceeded"
    r"|"
    r"stealth\s+scan|tcp\s+stealth|syn\s+scan|fin.*scan"
    r"|"
    r"connection\s+flood|dos\s+threshold\s+exceeded|ddos\s+.*flood"
    r"|"
    r"<script>|alert\s*\(|or\s+1\s*=\s*1|union\s+select"
    r"|"
    r"cve-\d{4}-\d+.*exploit|exploit\s+attempt"
    r"|"
    r"failed\s+password\s+for\s+invalid\s+user"
    r")",
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
    novelty_detector : NoveltyDetector | None
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
        novelty_detector: NoveltyDetector | None = None,
        flow_classifier: FlowClassifier | None = None,
        bot_flow_classifier: Any | None = None,
        flow_aggregator: FlowWindowAggregator | None = None,
        endpoint_novelty_model: EndpointNoveltyModel | None = None,
    ) -> None:
        self._cfg = config
        self._model_version = model_version
        scoring = config.get("scoring", {})
        self.calibrated_flow_isolation = _enabled(
            scoring.get("calibrated_flow_isolation", False), default=False
        )
        weights = scoring.get("weights", {})

        self.w_cls = _weight_value(weights.get("classifier", 0.35), "classifier weight")
        self.w_ent = _weight_value(weights.get("entity_boost", 0.25), "entity weight")
        self.w_ce = _weight_value(weights.get("cross_encoder", 0.40), "cross-encoder weight")
        self.w_nov = _weight_value(weights.get("novelty", 0.0), "novelty weight")
        self.entity_boost_value = _float_value(
            scoring.get("entity_boost_value", 0.20), "entity boost"
        )
        self.dedup_penalty_value = _float_value(
            scoring.get("dedup_penalty", 0.30), "deduplication penalty"
        )

        routing = scoring.get("routing", {})
        self.threshold_low, self.threshold_medium, self.threshold_high = _routing_thresholds(
            routing
        )
        self._default_profile = (
            self.w_cls,
            self.w_ent,
            self.w_ce,
            self.w_nov,
            self.threshold_low,
            self.threshold_medium,
            self.threshold_high,
        )
        self._source_profiles: dict[str, tuple[float, ...]] = {}
        allowed_profiles = {
            "syslog",
            "winevent",
            "firewall",
            "firewall_http",
            "firewall_flow",
            "endpoint",
            "cloudtrail",
            "web",
            "generic",
        }
        for profile_name, profile_value in scoring.get("source_profiles", {}).items():
            if profile_name not in allowed_profiles or not isinstance(profile_value, dict):
                raise ValueError(f"Invalid scoring source profile: {profile_name}")
            profile_weights = profile_value.get("weights", {})
            profile_routing = {**routing, **profile_value.get("routing", {})}
            self._source_profiles[profile_name] = (
                _weight_value(
                    profile_weights.get("classifier", self.w_cls),
                    f"{profile_name} classifier weight",
                ),
                _weight_value(
                    profile_weights.get("entity_boost", self.w_ent),
                    f"{profile_name} entity weight",
                ),
                _weight_value(
                    profile_weights.get("cross_encoder", self.w_ce),
                    f"{profile_name} cross-encoder weight",
                ),
                _weight_value(
                    profile_weights.get("novelty", self.w_nov),
                    f"{profile_name} novelty weight",
                ),
                *_routing_thresholds(profile_routing),
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
                batch_size=_int_value(ner_cfg.get("batch_size", 32), "NER batch size"),
                min_confidence=_float_value(
                    ner_cfg.get("min_confidence", 0.80), "NER minimum confidence"
                ),
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
                batch_size=_int_value(biencoder_cfg.get("batch_size", 64), "BiEncoder batch size"),
                dedup_threshold=_float_value(
                    biencoder_cfg.get("dedup_threshold", 0.95), "deduplication threshold"
                ),
                dedup_window_minutes=_float_value(
                    biencoder_cfg.get("dedup_window_minutes", 5.0), "deduplication window"
                ),
                faiss_top_k=_int_value(biencoder_cfg.get("faiss_top_k", 3), "FAISS top-k"),
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
                batch_size=_int_value(
                    cross_encoder_cfg.get("batch_size", 16), "CrossEncoder batch size"
                ),
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
                window_size=_int_value(drift_cfg.get("window_size", 1000), "drift window"),
                psi_threshold=_float_value(drift_cfg.get("psi_threshold", 0.25), "PSI threshold"),
                check_interval=_int_value(
                    drift_cfg.get("check_interval", 100), "drift check interval"
                ),
                auto_fallback=_enabled(drift_cfg.get("auto_fallback", True)),
            )
        else:
            self.drift_detector = None

        novelty_cfg = config.get("monitoring", {}).get("novelty", {})
        self.novelty_detector: NoveltyDetector | None
        if novelty_detector is not None:
            self.novelty_detector = novelty_detector
        elif _enabled(novelty_cfg.get("enabled", False)):
            self.novelty_detector = NoveltyDetector(
                window_size=_int_value(novelty_cfg.get("window_size", 10000), "novelty window"),
                min_baseline=_int_value(novelty_cfg.get("min_baseline", 100), "novelty baseline"),
                warmup_events=_int_value(novelty_cfg.get("warmup_events", 500), "novelty warmup"),
                distance_scale=_float_value(
                    novelty_cfg.get("distance_scale", 2.0), "novelty distance scale"
                ),
            )
        else:
            self.novelty_detector = None

        if self.novelty_detector is not None and not isinstance(
            self.biencoder, DisabledBiEncoderModel
        ):
            self.biencoder.novelty_detector = self.novelty_detector

        self.syslog_classifier = syslog_classifier or SyslogClassifier()

        flow_cfg = models_cfg.get("flow", {})
        self.flow_enabled = flow_classifier is not None or _enabled(
            flow_cfg.get("enabled", False), default=False
        )
        self.flow_classifier = flow_classifier or FlowClassifier(
            model_path=_resolve_path(flow_cfg.get("path", "models/flow/flow_classifier.onnx")),
            scaler_path=_resolve_path(flow_cfg.get("scaler_path", "models/flow/scaler.json")),
            feature_names_path=_resolve_path(
                flow_cfg.get("feature_names_path", "models/flow/feature_names.json")
            ),
        )
        bot_cfg = flow_cfg.get("bot", {})
        self.bot_flow_enabled = bot_flow_classifier is not None or _enabled(
            bot_cfg.get("enabled", False), default=False
        )
        self.bot_flow_classifier = bot_flow_classifier or _bot_classifier_module.BotFlowClassifier(
            model_path=_resolve_path(bot_cfg.get("path", "models/flow/bot_classifier.onnx")),
            scaler_path=_resolve_path(bot_cfg.get("scaler_path", "models/flow/bot_scaler.json")),
            feature_names_path=_resolve_path(
                bot_cfg.get("feature_names_path", "models/flow/bot_feature_names.json")
            ),
            calibration_path=_resolve_path(
                bot_cfg.get("calibration_path", "models/flow/bot_calibration.json")
            ),
            target_fpr=str(bot_cfg.get("target_fpr", "0.01")),
            alert_floor=_probability_config(
                bot_cfg.get("alert_floor", 0.65), "Bot flow alert floor"
            ),
        )
        self.flow_fusion_policy = _flow_fusion_module.FlowFusionPolicy(
            calibration_path=_resolve_path(
                flow_cfg.get("fusion_calibration_path", "models/flow/fusion_calibration.json")
            ),
            target_fpr=str(flow_cfg.get("fusion_target_fpr", "0.01")),
        )
        window_cfg = scoring.get("flow_window", {})
        self.flow_aggregator = flow_aggregator
        if self.flow_aggregator is None and _enabled(window_cfg.get("enabled", False), False):
            self.flow_aggregator = FlowWindowAggregator(
                window_seconds=_float_value(window_cfg.get("window_seconds", 10.0), "flow window"),
                max_keys=_int_value(window_cfg.get("max_keys", 10_000), "flow maximum keys"),
                max_events_per_key=_int_value(
                    window_cfg.get("max_events_per_key", 5_000), "flow events per key"
                ),
                flow_threshold=_float_value(
                    window_cfg.get("flow_threshold", 200.0), "flow threshold"
                ),
                packet_threshold=_float_value(
                    window_cfg.get("packet_threshold", 20_000.0), "packet threshold"
                ),
                byte_threshold=_float_value(
                    window_cfg.get("byte_threshold", 10_000_000.0), "byte threshold"
                ),
                source_threshold=_float_value(
                    window_cfg.get("source_threshold", 50.0), "source threshold"
                ),
            )

        if self.calibrated_flow_isolation:
            if self.flow_aggregator is not None:
                raise ValueError("Calibrated flow isolation requires flow_window.enabled=false")
            if not self.flow_enabled or not self.bot_flow_enabled:
                raise ValueError("Calibrated flow isolation requires both flow heads")

        endpoint_cfg = models_cfg.get("endpoint_novelty", {})
        self.endpoint_novelty_enabled = endpoint_novelty_model is not None or _enabled(
            endpoint_cfg.get("enabled", False), default=False
        )
        self.endpoint_novelty_model = endpoint_novelty_model or EndpointNoveltyModel(
            model_path=_resolve_path(
                endpoint_cfg.get("path", "models/endpoint/endpoint_novelty.onnx")
            ),
            config_path=_resolve_path(
                endpoint_cfg.get("config_path", "models/endpoint/novelty_config.json")
            ),
            score_scale=_float_value(
                endpoint_cfg.get("score_scale", 0.03), "endpoint novelty score scale"
            ),
        )

        self._feature_cache_names: tuple[str, ...] = ()
        self._feature_cache_tokens: list[tuple[str, tuple[str, ...]]] = []
        self._syslog_feature_cache_names: tuple[str, ...] = ()
        self._syslog_feature_cache_tokens: list[tuple[str, tuple[str, ...]]] = []

        self._validate_manifest()

    def _validate_manifest(self) -> None:
        manifest_path = ROOT / "models" / "model_manifest.json"
        strict = _strict_model_loading()
        if not manifest_path.exists():
            if strict:
                raise RuntimeError(f"Model manifest not found: {manifest_path}")
            logger.warning("Model manifest not found; integrity validation skipped")
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors: list[str] = []
            for key, expected in manifest.get("models", {}).items():
                if expected.get("status") == "missing":
                    continue
                model_dir = ROOT / expected.get("path", "")
                if not model_dir.exists():
                    errors.append(f"{key}: directory not found")
                    continue
                for pattern, meta in expected.get("artifacts", {}).items():
                    artifact_path = model_dir / pattern
                    if not artifact_path.exists():
                        errors.append(f"{key}: artifact {pattern} missing")
                        continue
                    h = hashlib.sha256()
                    with open(artifact_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    actual_hash = h.hexdigest()[:16]
                    if actual_hash != meta.get("sha256"):
                        errors.append(
                            f"{key}: {pattern} hash mismatch "
                            f"(expected {meta['sha256']}, got {actual_hash})"
                        )
            if errors:
                message = "Model manifest validation failed: " + "; ".join(errors)
                if strict:
                    raise RuntimeError(message)
                for error in errors:
                    logger.warning("Model manifest mismatch", error=error)
            else:
                logger.info("Model manifest validated")
        except RuntimeError:
            raise
        except Exception as exc:
            if strict:
                raise RuntimeError("Model manifest validation failed") from exc
            logger.warning("Manifest validation skipped", error=str(exc))

    # ── public API ─────────────────────────────────────────────────────────────

    def preload_models(self) -> None:
        """Eagerly load all models to eliminate cold-start latency."""

        def _warmup(model_name: str, warmup_fn: Any) -> None:
            started = time.perf_counter()
            try:
                warmup_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model warmup failed",
                    model=model_name,
                    error=str(exc),
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                if _strict_model_loading():
                    raise
            else:
                logger.info(
                    "Model warmup completed",
                    model=model_name,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )

        logger.info("Pre-loading classifier")
        _ = self.classifier.is_ready()
        classifier_features = getattr(self.classifier, "expected_feature_count", 0) or 1
        _warmup(
            "classifier",
            lambda: self.classifier.predict_proba(
                np.zeros((1, classifier_features), dtype=np.float32)
            ),
        )
        if self.endpoint_novelty_enabled:
            logger.info("Pre-loading endpoint novelty model")
            endpoint_ready = self.endpoint_novelty_model.is_ready()
            if not endpoint_ready and _strict_model_loading():
                raise RuntimeError("endpoint novelty model failed readiness validation")
        if self.flow_enabled:
            logger.info("Pre-loading flow classifier")
            flow_ready = self.flow_classifier.is_loaded()
            if not flow_ready and _strict_model_loading():
                raise RuntimeError("flow classifier failed readiness validation")
            flow_features = self.flow_classifier.expected_feature_count or 1
            _warmup(
                "flow_classifier",
                lambda: self.flow_classifier.predict_proba(
                    np.ones((1, flow_features), dtype=np.float32)
                ),
            )
        if self.bot_flow_enabled:
            logger.info("Pre-loading Bot flow classifier")
            _ = self.bot_flow_classifier.is_loaded()
            bot_features = self.bot_flow_classifier.expected_feature_count or 1
            _warmup(
                "bot_flow_classifier",
                lambda: self.bot_flow_classifier.predict_proba(
                    np.ones((1, bot_features), dtype=np.float32)
                ),
            )
            _ = self.flow_fusion_policy.composite_threshold
        logger.info("Pre-loading syslog classifier")
        _ = self.syslog_classifier.is_ready()
        syslog_features = len(self.syslog_classifier.feature_names) or 1
        _warmup(
            "syslog_classifier",
            lambda: self.syslog_classifier.predict_proba(
                np.zeros((1, syslog_features), dtype=np.float32)
            ),
        )
        logger.info("Pre-loading tier2 classifier")
        _ = self.tier2_classifier.is_ready()
        _warmup("tier2_classifier", self.tier2_classifier.prewarm)
        logger.info("Pre-loading biencoder")
        _ = self.biencoder.check_dedup_and_retrieve_batch([])
        if not isinstance(self.biencoder, DisabledBiEncoderModel):
            _warmup(
                "biencoder",
                lambda: self.biencoder.check_dedup_and_retrieve_batch(["warmup"]),
            )
        logger.info("Pre-loading NER model")
        _ = self.ner_model.extract_batch([])
        if not isinstance(self.ner_model, DisabledNERModel):
            _warmup("ner_model", lambda: self.ner_model.extract_batch(["warmup"]))
        logger.info("Pre-loading cross encoder")
        _ = self.cross_encoder.score_batch([], [])
        if not isinstance(self.cross_encoder, DisabledCrossEncoderModel):
            _warmup(
                "cross_encoder",
                lambda: self.cross_encoder.score_batch(
                    ["warmup"],
                    [[{"id": "warmup", "name": "warmup", "description": "warmup"}]],
                ),
            )
        if self.novelty_detector is not None:
            logger.info("Pre-loading novelty detector")
            _ = self.novelty_detector.get_stats()

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
                tier3_failed = False
                novelty_results = None
                try:
                    if self.novelty_detector is not None and hasattr(
                        self.biencoder,
                        "check_dedup_retrieve_and_score_novelty_batch",
                    ):
                        combined_results = (
                            self.biencoder.check_dedup_retrieve_and_score_novelty_batch(texts)
                        )
                        bi_results = [
                            (dedup_res, candidates)
                            for dedup_res, candidates, _novelty in combined_results
                        ]
                        novelty_results = [
                            novelty for _dedup_res, _candidates, novelty in combined_results
                        ]
                    else:
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
                except Exception as exc:
                    bi_span.set_attribute("logfilter.tier3_error_stage", "biencoder")
                    bi_span.set_attribute("logfilter.tier3_error_type", type(exc).__name__)
                    self._apply_tier3_neutral_fallback(
                        scored,
                        stage="biencoder",
                        exc=exc,
                    )
                    tier3_failed = True

            # ── Novelty detection (reuses BiEncoder embeddings) ──────────────
            if self.novelty_detector is not None and not tier3_failed:
                with telemetry.start_as_current_span(
                    "scorer.novelty.score_batch",
                    {"logfilter.batch_size": len(texts)},
                ) as nov_span:
                    try:
                        if novelty_results is None:
                            novelty_results = self.biencoder.score_novelty_batch(texts)
                        for se, nov_result in zip(scored, novelty_results):
                            se.novelty_score = nov_result.score
                        nov_span.set_attribute(
                            "logfilter.novelty_baseline_size",
                            novelty_results[0].baseline_size if novelty_results else 0,
                        )
                    except Exception as exc:
                        nov_span.set_attribute("logfilter.novelty_error", type(exc).__name__)
                        logger.warning("Novelty detection failed", error=str(exc))

            # ── Tier 3: NER + CrossEncoder (skip duplicates) ───────────────────
            non_dup_indices = (
                [] if tier3_failed else [i for i, se in enumerate(scored) if not se.is_duplicate]
            )
            if non_dup_indices and not tier3_failed:
                nd_texts = [texts[i] for i in non_dup_indices]
                nd_scored = [scored[i] for i in non_dup_indices]

                # NER
                with telemetry.start_as_current_span(
                    "scorer.ner.extract_batch",
                    {"logfilter.batch_size": len(nd_texts)},
                ) as ner_span:
                    try:
                        ner_results = self.ner_model.extract_batch(nd_texts)
                        high_value_count = 0
                        for se, ner_result in zip(nd_scored, ner_results):
                            se.entities = ner_result.to_dict()
                            se.ai_entities = ner_result.flat_entity_string()
                            if ner_result.has_high_value_entities:
                                high_value_count += 1
                                se.entity_boost = self.entity_boost_value
                        ner_span.set_attribute(
                            "logfilter.high_value_entity_count", high_value_count
                        )
                    except Exception as exc:
                        ner_span.set_attribute("logfilter.tier3_error_stage", "ner")
                        ner_span.set_attribute("logfilter.tier3_error_type", type(exc).__name__)
                        self._apply_tier3_neutral_fallback(
                            scored,
                            stage="ner",
                            exc=exc,
                        )
                        tier3_failed = True

                # CrossEncoder
                if not tier3_failed:
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
                        try:
                            ce_results = self.cross_encoder.score_batch(
                                nd_texts, candidates_per_event
                            )
                            matched_count = 0
                            for se, ce_scores in zip(nd_scored, ce_results):
                                se.cross_encoder_scores = [
                                    {
                                        "id": s.technique_id,
                                        "name": s.name,
                                        "score": round(s.score, 4),
                                    }
                                    for s in ce_scores
                                ]
                                if ce_scores:
                                    matched_count += 1
                                    se.cross_encoder_max = ce_scores[0].score
                                    se.ai_mitre_technique = ce_scores[0].technique_id
                            ce_span.set_attribute(
                                "logfilter.cross_encoder_match_count", matched_count
                            )
                        except Exception as exc:
                            ce_span.set_attribute("logfilter.tier3_error_stage", "cross_encoder")
                            ce_span.set_attribute("logfilter.tier3_error_type", type(exc).__name__)
                            self._apply_tier3_neutral_fallback(
                                scored,
                                stage="cross_encoder",
                                exc=exc,
                            )

            # ── Final score + routing ──────────────────────────────────────────
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            high_count = 0
            medium_count = 0
            sigma_count = 0
            for se, ev in zip(scored, events):
                if not self._is_isolated_flow(se) and _ATTACK_TOOL_RE.search(ev.raw):
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

        Rules are parsed with ``yaml.safe_load_all`` and only the local
        text-matching subset is accepted: one named selection containing
        mappable fields with exact, contains, startswith, or endswith values.
        Complex conditions and structured-only fields are skipped.

        For full fidelity Sigma evaluation, pipe the enriched event to a
        dedicated Sigma backend (Splunk, Elastic, or a field-extracted
        normaliser) rather than relying on this in-process matcher.
        """
        sigma_cfg = self._cfg.get("scoring", {}).get("sigma", {})
        if not _enabled(sigma_cfg.get("enabled", False), default=False):
            return
        with telemetry.start_as_current_span(
            "scorer.sigma.apply",
            {"logfilter.batch_size": len(events)},
        ) as span:
            sigma_rules_dir = ROOT / "config" / "sigma_rules"
            if not sigma_rules_dir.exists():
                span.set_attribute("logfilter.sigma.rule_count", 0)
                return

            rules = _load_sigma_text_rules(sigma_rules_dir)
            span.set_attribute("logfilter.sigma.rule_count", len(rules))
            match_count = 0
            for se, ev in zip(scored, events):
                for rule in rules:
                    if self._sigma_rule_matches(rule, ev.text):
                        se.sigma_matched = True
                        se.sigma_rule_ids.append(rule.rule_id)
                        match_count += 1
            span.set_attribute("logfilter.sigma.match_count", match_count)

    def _sigma_rule_matches(self, rule: _SigmaTextRule, event_text: str) -> bool:
        """Return whether the event matches any accepted text pattern."""
        text_lower = event_text.lower()
        for pattern in rule.patterns:
            for value in pattern.values:
                search_text = value.strip().lower()
                if not search_text:
                    continue
                if pattern.modifier == "contains" and search_text in text_lower:
                    return True
                if pattern.modifier == "endswith" and text_lower.endswith(search_text):
                    return True
                if pattern.modifier == "startswith" and text_lower.startswith(search_text):
                    return True
                if pattern.modifier == "exact":
                    for word in text_lower.split():
                        if word.strip(".,;:!?") == search_text:
                            return True
        return False

    def _apply_tier3_neutral_fallback(
        self,
        scored: list[ScoredEvent],
        *,
        stage: str,
        exc: Exception,
    ) -> None:
        logger.warning(
            "Tier-3 scoring failed; using neutral downstream scores",
            stage=stage,
            error_type=type(exc).__name__,
        )
        for se in scored:
            se.is_duplicate = False
            se.dedup_similarity = 0.0
            se.attack_candidates = []
            se.entities = {}
            se.ai_entities = ""
            se.entity_boost = 0.0
            se.cross_encoder_scores = []
            se.cross_encoder_max = 0.0
            se.ai_mitre_technique = ""
            se.dedup_penalty = 0.0
            se.score_degraded = True

    def _apply_classifier(self, scored: list[ScoredEvent], events: list[NormalizedEvent]) -> None:
        """Populate classifier_score using the trained event-count classifier."""
        from logfilter.pipeline.normalizer import LogSourceType

        endpoint_indices: list[int] = []
        if self.endpoint_novelty_enabled:
            endpoint_vectors: list[np.ndarray] = []
            endpoint_candidates: list[int] = []
            for index, event in enumerate(events):
                vector = extract_endpoint_vector(event)
                if vector is not None:
                    endpoint_candidates.append(index)
                    endpoint_vectors.append(vector)
            if endpoint_candidates:
                try:
                    probabilities = self.endpoint_novelty_model.predict_proba(
                        np.vstack(endpoint_vectors)
                    )
                    for index, probability in zip(endpoint_candidates, probabilities):
                        endpoint_score = _bounded_probability(probability)
                        scored[index].endpoint_score = endpoint_score
                        scored[index].classifier_score = endpoint_score
                    endpoint_indices = endpoint_candidates
                except (ValueError, IndexError, TypeError, RuntimeError) as exc:
                    logger.warning("Endpoint novelty scoring failed", error=str(exc))
                    for index in endpoint_candidates:
                        scored[index].score_degraded = True

        flow_vectors: list[np.ndarray] = []
        flow_candidates: list[int] = []
        if self.flow_enabled or self.bot_flow_enabled:
            for index, event in enumerate(events):
                if event.source_type != LogSourceType.FIREWALL:
                    continue
                vector = extract_flow_vector(event)
                if vector is not None:
                    flow_candidates.append(index)
                    flow_vectors.append(vector)

        flow_indices: list[int] = []
        if flow_candidates:
            try:
                flow_matrix = np.vstack(flow_vectors)
                loaded_general = self.flow_enabled and self.flow_classifier.is_loaded()
                loaded_bot = self.bot_flow_enabled and self.bot_flow_classifier.is_loaded()
                general_probabilities = (
                    self.flow_classifier.predict_proba(flow_matrix)
                    if loaded_general
                    else np.zeros(len(flow_candidates), dtype=np.float32)
                )
                bot_probabilities = (
                    self.bot_flow_classifier.predict_proba(flow_matrix)
                    if loaded_bot
                    else np.zeros(len(flow_candidates), dtype=np.float32)
                )
                if loaded_general or loaded_bot:
                    for position, index in enumerate(flow_candidates):
                        flow_score = _bounded_probability(general_probabilities[position])
                        bot_score = _bounded_probability(bot_probabilities[position])
                        scored[index].flow_score = flow_score
                        scored[index].bot_score = bot_score
                        if loaded_general and loaded_bot:
                            classifier_score = self.flow_fusion_policy.calibrated_risk(
                                flow_score, bot_score
                            )
                        else:
                            classifier_score = max(flow_score, bot_score)
                        scored[index].classifier_score = classifier_score
                        if self.flow_aggregator is not None:
                            snapshot = self.flow_aggregator.update(events[index])
                            scored[index].ddos_window_score = snapshot.risk_score
                            scored[index].classifier_score = max(
                                scored[index].classifier_score, snapshot.risk_score
                            )
                    flow_indices = flow_candidates
            except (ValueError, IndexError, TypeError, RuntimeError) as exc:
                logger.warning(
                    "Flow classifier failed; falling back to text classifier", error=str(exc)
                )
                for index in flow_candidates:
                    scored[index].score_degraded = True

        flow_index_set = set(flow_indices)
        endpoint_index_set = set(endpoint_indices)
        syslog_indices = [
            index
            for index, event in enumerate(events)
            if index not in flow_index_set
            and index not in endpoint_index_set
            and event.source_type
            in (
                LogSourceType.SYSLOG,
                LogSourceType.WEB,
                LogSourceType.FIREWALL,
                LogSourceType.WINEVENT,
            )
        ]
        structured_indices = flow_index_set | endpoint_index_set | set(syslog_indices)
        hdfs_indices = [index for index in range(len(events)) if index not in structured_indices]

        with telemetry.start_as_current_span(
            "scorer.tier1.classifier",
            {"logfilter.batch_size": len(events), "logfilter.flow_count": len(flow_indices)},
        ) as span:
            try:
                if syslog_indices and self.syslog_classifier.is_ready():
                    syslog_events = [events[index] for index in syslog_indices]
                    syslog_vectors = self._syslog_feature_vectors(syslog_events)
                    syslog_probs = self.syslog_classifier.predict_proba(syslog_vectors)
                    for index, probability in zip(syslog_indices, syslog_probs):
                        scored[index].classifier_score = _bounded_probability(probability)

                if hdfs_indices:
                    hdfs_events = [events[index] for index in hdfs_indices]
                    feature_vectors = self._classifier_feature_vectors(hdfs_events)
                    span.set_attribute(
                        "logfilter.feature_count",
                        _int_value(feature_vectors.shape[1], "feature count"),
                    )
                    hdfs_probs = self.classifier.predict_proba(feature_vectors)
                    for index, probability in zip(hdfs_indices, hdfs_probs):
                        scored[index].classifier_score = _bounded_probability(probability)

                http_indices: list[int] = []
                for index, event in enumerate(events):
                    if "http_lexical_score" not in event.fields:
                        continue
                    http_score = _bounded_probability(event.fields["http_lexical_score"])
                    scored[index].http_score = http_score
                    scored[index].classifier_score = http_score
                    http_indices.append(index)

                ready = (
                    bool(flow_indices)
                    or bool(endpoint_indices)
                    or bool(http_indices)
                    or self.classifier.is_ready()
                    or self.syslog_classifier.is_ready()
                )
                if not ready:
                    for result in scored:
                        result.score_degraded = True
            except (ValueError, IndexError, TypeError, RuntimeError) as exc:
                telemetry.record_exception(span, exc)
                logger.warning("Classifier scoring failed; using neutral scores", error=str(exc))
                for result in scored:
                    result.classifier_score = 0.5
                    result.score_degraded = True

        escalation_indices = [
            index
            for index, result in enumerate(scored)
            if index not in structured_indices
            and self.tier2_classifier.should_escalate(result.classifier_score)
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
                score = _bounded_probability(prob)
                scored[index].classifier_score = score
                scored[index].tier2_score = score
                scored[index].tier2_used = True

    def _classifier_feature_vectors(self, events: list[NormalizedEvent]) -> np.ndarray:
        """
        Convert normalized log text into the bag-of-event vector expected by the
        HDFS TraceBench classifier artifacts.
        """
        feature_names = tuple(getattr(self.classifier, "feature_names", []) or ())
        n_features = len(feature_names) or _int_value(
            getattr(self.classifier, "expected_feature_count", 0) or 1,
            "classifier feature count",
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
        """Convert normalized log text into the syslog classifier feature vector (110 features)."""
        feature_names = tuple(self.syslog_classifier.feature_names)
        n_features = len(feature_names) or 100
        vectors = np.zeros((len(events), n_features), dtype=np.float32)
        if not feature_names:
            return vectors

        prepared = self._prepared_syslog_features(feature_names)
        for row, event in enumerate(events):
            # Adapters keep model-facing text bounded and redacted. Archived raw
            # payloads may contain credentials and must not re-enter this surface.
            text = event.text.lower()
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

    def _profile_for(self, se: ScoredEvent) -> tuple[float, ...]:
        profile_name = se.source_type
        if profile_name == "firewall" and "http_lexical_features" in se.fields:
            profile_name = "firewall_http"
        elif profile_name == "firewall" and any(
            key in se.fields for key in ("dpt", "duration", "fwdPackets")
        ):
            profile_name = "firewall_flow"
        return self._source_profiles.get(profile_name, self._default_profile)

    def _is_isolated_flow(self, se: ScoredEvent) -> bool:
        return (
            self.calibrated_flow_isolation
            and se.source_type == "firewall"
            and any(key in se.fields for key in ("dpt", "duration", "fwdPackets"))
        )

    def _compute_score(self, se: ScoredEvent) -> float:
        """Compute final composite threat score."""
        w_cls, w_ent, w_ce, w_nov, _, _, _ = self._profile_for(se)
        isolated_flow = self._is_isolated_flow(se)
        if isolated_flow:
            return _bounded_probability(se.classifier_score)
        # Sigma is outside the attested calibrated-flow policy.
        if se.sigma_matched and not isolated_flow:
            se.classifier_score = max(se.classifier_score, 0.90)

        # A dedup penalty is valid only when the cross-encoder has non-zero weight.
        dedup = se.dedup_penalty if se.cross_encoder_max > 0 and w_ce > 0 else 0.0

        score = (
            w_cls * se.classifier_score
            + w_ent * se.entity_boost
            + w_ce * se.cross_encoder_max
            + w_nov * se.novelty_score
            - dedup
        )
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _routing_label(self, se: ScoredEvent) -> str:
        _, _, _, _, low, medium, high = self._profile_for(se)
        if (se.sigma_matched and not self._is_isolated_flow(se)) or se.ai_threat_score >= high:
            return "HIGH"
        if se.ai_threat_score >= medium:
            return "MEDIUM"
        if se.ai_threat_score >= low:
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
        return sum(valid) / len(valid) if valid else 0.0
