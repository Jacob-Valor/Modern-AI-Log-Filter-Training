"""Unit tests for scoring orchestration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from logfilter.models.biencoder import ATTACKCandidate, BiEncoderModel, DedupResult
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel, CrossEncoderScore
from logfilter.models.ner import ExtractedEntities, NERModel
from logfilter.models.syslog_classifier import SyslogClassifier
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.monitoring.novelty_detector import NoveltyDetector, NoveltyResult
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer


class FakeClassifier(LogClassifier):
    feature_names = ["failed password from", "namenode block received"]
    expected_feature_count = 2

    def __init__(self) -> None:
        self.last_vectors: np.ndarray | None = None

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.last_vectors = feature_vectors
        return np.full(len(feature_vectors), 0.82, dtype=np.float32)

    def is_ready(self) -> bool:
        return True


class FakeSyslogClassifier(SyslogClassifier):
    def __init__(self) -> None:
        self._feature_names = ["sshd+failed password"]
        self.last_vectors: np.ndarray | None = None

    def is_ready(self) -> bool:
        return True

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.last_vectors = feature_vectors
        return np.full(len(feature_vectors), 0.82, dtype=np.float32)


class FakeTier2Classifier(Tier2Classifier):
    def should_escalate(self, tier1_prob: float) -> bool:
        _ = tier1_prob
        return False

    def is_ready(self) -> bool:
        return False

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        _ = texts
        raise AssertionError("Tier-2 should not be called in Tier-1 classifier score test")


class FakeNER(NERModel):
    def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        return [ExtractedEntities() for _ in texts]


class FakeBiEncoder(BiEncoderModel):
    def check_dedup_and_retrieve_batch(
        self, texts: list[str]
    ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
        return [
            (
                DedupResult(is_duplicate=False, similarity=0.0),
                [
                    ATTACKCandidate(
                        technique_id="T1110",
                        name="Brute Force",
                        description="Password guessing and repeated failed logons",
                        similarity=0.91,
                    )
                ],
            )
            for _ in texts
        ]


class FakeCrossEncoder(CrossEncoderModel):
    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[CrossEncoderScore]]:
        _ = candidates_per_log
        return [[CrossEncoderScore("T1110", "Brute Force", 0.0)] for _ in log_texts]


def _tier3_fault_scorer(
    *,
    ner_model: NERModel,
    biencoder: BiEncoderModel,
    cross_encoder: CrossEncoderModel,
) -> LogScorer:
    return LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 0.5,
                    "entity_boost": 0.3,
                    "cross_encoder": 0.2,
                    "novelty": 0.15,
                },
                "entity_boost_value": 0.3,
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=ner_model,
        biencoder=biencoder,
        cross_encoder=cross_encoder,
        syslog_classifier=FakeSyslogClassifier(),
    )


def _assert_neutral_tier3_fallback(scored: Any, raw_payload: str, log_text: str) -> None:
    assert scored.score_degraded is True
    assert scored.is_duplicate is False
    assert scored.dedup_similarity == pytest.approx(0.0)
    assert scored.attack_candidates == []
    assert scored.entities == {}
    assert scored.ai_entities == ""
    assert scored.entity_boost == pytest.approx(0.0)
    assert scored.cross_encoder_scores == []
    assert scored.cross_encoder_max == pytest.approx(0.0)
    assert scored.ai_mitre_technique == ""
    assert raw_payload not in log_text


def test_classifier_score_is_applied_to_composite_score() -> None:
    syslog_classifier = FakeSyslogClassifier()
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=syslog_classifier,
    )

    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )
    scored = scorer.score(event)

    assert abs(scored.classifier_score - 0.82) < 1e-6
    assert abs(scored.ai_threat_score - 0.82) < 1e-6
    assert scored.ai_priority == "MEDIUM"
    assert syslog_classifier.last_vectors is not None
    assert syslog_classifier.last_vectors.shape == (1, 1)
    assert syslog_classifier.last_vectors[0, 0] > 0


def test_preload_models_strict_mode_propagates_warmup_failure(monkeypatch) -> None:
    class BrokenClassifier(FakeClassifier):
        def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
            del feature_vectors
            raise RuntimeError("classifier warmup failed")

    monkeypatch.setenv("LOGFILTER_MODELS_STRICT", "true")
    scorer = LogScorer(
        config={"scoring": {"weights": {"classifier": 1.0}}},
        classifier=BrokenClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    with pytest.raises(RuntimeError, match="classifier warmup failed"):
        scorer.preload_models()


class RaisingNER(NERModel):
    def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        del texts
        raise AssertionError("Disabled NER should not call configured heavy model")


class RaisingBiEncoder(BiEncoderModel):
    def check_dedup_and_retrieve_batch(
        self, texts: list[str]
    ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
        del texts
        raise AssertionError("Disabled BiEncoder should not call configured heavy model")


class RaisingCrossEncoder(CrossEncoderModel):
    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[CrossEncoderScore]]:
        del log_texts, candidates_per_log
        raise AssertionError("Disabled CrossEncoder should not call configured heavy model")


def test_optional_downstream_models_can_be_disabled_for_local_validation() -> None:
    classifier = FakeClassifier()
    scorer = LogScorer(
        config={
            "models": {
                "ner": {"enabled": "false"},
                "biencoder": {"enabled": "false"},
                "cross_encoder": {"enabled": "false"},
            },
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            },
        },
        classifier=classifier,
        tier2_classifier=FakeTier2Classifier(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )
    scored = scorer.score(event)

    assert abs(scored.classifier_score - 0.82) < 1e-6
    assert scored.attack_candidates == []
    assert scored.entities["has_high_value_entities"] is False
    assert scored.entities["confidence"] == 0.0
    assert scored.cross_encoder_scores == []
    assert scored.ai_priority == "MEDIUM"


def test_routing_thresholds_accept_env_substituted_strings() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "routing": {"high": "0.90", "medium": "0.60", "low": "0.30"}
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    assert scorer.threshold_high == pytest.approx(0.90)
    assert scorer.threshold_medium == pytest.approx(0.60)
    assert scorer.threshold_low == pytest.approx(0.30)


def test_scorer_wires_tier2_uncertainty_config() -> None:
    scorer = LogScorer(
        config={
            "scoring": {"tier2": {"uncertainty_low": "0.25", "uncertainty_high": "0.75"}}
        },
        classifier=FakeClassifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    assert scorer.tier2_classifier.uncertainty_low == pytest.approx(0.25)
    assert scorer.tier2_classifier.uncertainty_high == pytest.approx(0.75)


@pytest.mark.parametrize(
    "routing, message",
    [
        ({"high": "1.20", "medium": "0.50", "low": "0.20"}, "between"),
        ({"high": "0.85", "medium": "0.50", "low": "-0.10"}, "between"),
        ({"high": "0.85", "medium": "abc", "low": "0.20"}, "numeric"),
        ({"high": "0.85", "medium": "0.20", "low": "0.20"}, "low < medium"),
        ({"high": "0.50", "medium": "0.50", "low": "0.20"}, "low < medium"),
    ],
)
def test_invalid_routing_thresholds_fail_fast(
    routing: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LogScorer(
            config={"scoring": {"routing": routing}},
            classifier=FakeClassifier(),
            tier2_classifier=FakeTier2Classifier(),
            ner_model=FakeNER(),
            biencoder=FakeBiEncoder(),
            cross_encoder=FakeCrossEncoder(),
        )


def test_enabled_parses_various_truthy_strings() -> None:
    from logfilter.pipeline.scorer import _enabled

    assert _enabled("true") is True
    assert _enabled("True") is True
    assert _enabled("TRUE") is True
    assert _enabled("yes") is True
    assert _enabled("1") is True
    assert _enabled("on") is True
    assert _enabled("false") is False
    assert _enabled("no") is False
    assert _enabled("0") is False
    assert _enabled(1) is True
    assert _enabled(0) is False
    assert _enabled(None, default=True) is True
    assert _enabled(None, default=False) is False
    assert _enabled(True) is True
    assert _enabled(False) is False


def test_score_batch_processes_multiple_events() -> None:
    classifier = FakeClassifier()
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=classifier,
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    normalizer = LogNormalizer()
    events = [
        normalizer.normalize(
            "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
        ),
        normalizer.normalize(
            "Jan 15 11:08:00 prod sshd[124]: Accepted password for admin from 10.0.0.6"
        ),
    ]
    scored = scorer.score_batch(events)

    assert len(scored) == 2
    assert all(s.classifier_score == pytest.approx(0.82) for s in scored)


class FakeTier2ThatEscalates(Tier2Classifier):
    def should_escalate(self, tier1_prob: float) -> bool:
        _ = tier1_prob
        return True

    def is_ready(self) -> bool:
        return True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        del texts
        return np.array([0.95], dtype=np.float32)


def test_tier2_escalates_when_uncertain() -> None:
    classifier = FakeClassifier()
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 0.8,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=classifier,
        tier2_classifier=FakeTier2ThatEscalates(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    # Use a non-syslog event so it goes through the HDFS classifier path
    # and is eligible for tier-2 escalation (syslog events are excluded by design).
    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)

    assert scored.tier2_used is True
    assert scored.tier2_score == pytest.approx(0.95)


def test_novelty_scoring_reuses_combined_biencoder_batch_path() -> None:
    class CombinedBiEncoder(FakeBiEncoder):
        combined_calls = 0

        def check_dedup_retrieve_and_score_novelty_batch(
            self, texts: list[str]
        ) -> list[tuple[DedupResult, list[ATTACKCandidate], NoveltyResult]]:
            self.combined_calls += 1
            return [
                (
                    DedupResult(is_duplicate=False, similarity=0.0),
                    [],
                    NoveltyResult(score=0.42, distance=0.21, baseline_size=1),
                )
                for _ in texts
            ]

        def score_novelty_batch(
            self,
            texts: list[str],
            embeddings: np.ndarray | None = None,
        ) -> list[NoveltyResult]:
            del texts, embeddings
            raise AssertionError("legacy novelty path should not be called")

    biencoder = CombinedBiEncoder()
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=biencoder,
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
        novelty_detector=NoveltyDetector(window_size=10, min_baseline=1, warmup_events=0),
    )

    scored = scorer.score(LogNormalizer().normalize("test event"))

    assert scored.novelty_score == pytest.approx(0.42)
    assert biencoder.combined_calls == 1


def test_novelty_detector_is_wired_into_real_biencoder() -> None:
    detector = NoveltyDetector(window_size=10, min_baseline=1, warmup_events=0)
    real_biencoder = BiEncoderModel()
    scorer = LogScorer(
        config={"scoring": {"weights": {"classifier": 1.0, "novelty": 0.15}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=real_biencoder,
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
        novelty_detector=detector,
    )

    assert scorer.biencoder.novelty_detector is detector


def test_preload_models_reraises_tier2_prewarm_failure_under_strict(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_MODELS_STRICT", "1")

    class BrokenWarmupTier2(FakeTier2Classifier):
        def prewarm(self) -> None:
            raise RuntimeError("tier2 inference broke during warmup")

    scorer = LogScorer(
        config={"scoring": {"weights": {"classifier": 1.0}}},
        classifier=FakeClassifier(),
        tier2_classifier=BrokenWarmupTier2(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    with pytest.raises(RuntimeError, match="tier2 inference broke during warmup"):
        scorer.preload_models()


def test_preload_models_tolerates_tier2_prewarm_failure_when_not_strict(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_MODELS_STRICT", raising=False)

    class BrokenWarmupTier2(FakeTier2Classifier):
        def prewarm(self) -> None:
            raise RuntimeError("tier2 inference broke during warmup")

    scorer = LogScorer(
        config={"scoring": {"weights": {"classifier": 1.0}}},
        classifier=FakeClassifier(),
        tier2_classifier=BrokenWarmupTier2(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    scorer.preload_models()


def test_probability_config_validates_numeric() -> None:
    from logfilter.pipeline.scorer import _probability_config

    with pytest.raises(ValueError, match="numeric"):
        _probability_config("abc", "test")


def test_probability_config_validates_range() -> None:
    from logfilter.pipeline.scorer import _probability_config

    with pytest.raises(ValueError, match="between"):
        _probability_config(1.5, "test")

    with pytest.raises(ValueError, match="between"):
        _probability_config(-0.1, "test")


def test_routing_thresholds_validates_order() -> None:
    from logfilter.pipeline.scorer import _routing_thresholds

    with pytest.raises(ValueError, match="low < medium < high"):
        _routing_thresholds({"high": 0.5, "medium": 0.6, "low": 0.2})


def test_scored_event_to_dict() -> None:
    from logfilter.pipeline.scorer import ScoredEvent

    event = ScoredEvent(
        source_type="syslog",
        timestamp="2026-01-15T11:07:53Z",
        host="prod-server01",
        raw="test",
        normalized_text="test",
        ai_threat_score=0.5,
        ai_priority="LOW",
    )
    d = event.to_dict()
    assert d["source_type"] == "syslog"
    assert d["ai_threat_score"] == 0.5


def test_model_version_paths() -> None:
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        model_version="v1",
    )
    assert scorer._model_version == "v1"


class FakeNERWithEntities(FakeNER):
    def extract_batch(self, texts: list[str]) -> list[Any]:
        result = ExtractedEntities()
        result.indicators = ["10.0.0.5"]
        result.confidence = 0.95
        result.has_high_value_entities = True
        return [result for _ in texts]


def test_tier3_biencoder_failure_uses_neutral_fallback(caplog) -> None:
    raw_payload = "RAW_PAYLOAD_SHOULD_NOT_LOG"

    class BrokenBiEncoder(FakeBiEncoder):
        def check_dedup_and_retrieve_batch(
            self, texts: list[str]
        ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
            raise RuntimeError(f"BiEncoder failed while scoring {texts[0]}")

    caplog.set_level(logging.WARNING)
    scorer = _tier3_fault_scorer(
        ner_model=FakeNERWithEntities(),
        biencoder=BrokenBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )
    event = LogNormalizer().normalize(raw_payload)

    scored = scorer.score(event)

    _assert_neutral_tier3_fallback(scored, raw_payload, caplog.text)


def test_tier3_ner_failure_uses_neutral_fallback(caplog) -> None:
    raw_payload = "RAW_PAYLOAD_SHOULD_NOT_LOG"

    class BrokenNER(FakeNER):
        def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
            raise RuntimeError(f"NER failed while scoring {texts[0]}")

    caplog.set_level(logging.WARNING)
    scorer = _tier3_fault_scorer(
        ner_model=BrokenNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )
    event = LogNormalizer().normalize(raw_payload)

    scored = scorer.score(event)

    _assert_neutral_tier3_fallback(scored, raw_payload, caplog.text)


def test_tier3_cross_encoder_failure_uses_neutral_fallback(caplog) -> None:
    raw_payload = "RAW_PAYLOAD_SHOULD_NOT_LOG"

    class BrokenCrossEncoder(FakeCrossEncoder):
        def score_batch(
            self,
            log_texts: list[str],
            candidates_per_log: list[list[dict[str, str]]],
        ) -> list[list[CrossEncoderScore]]:
            del candidates_per_log
            raise RuntimeError(f"CrossEncoder failed while scoring {log_texts[0]}")

    caplog.set_level(logging.WARNING)
    scorer = _tier3_fault_scorer(
        ner_model=FakeNERWithEntities(),
        biencoder=FakeBiEncoder(),
        cross_encoder=BrokenCrossEncoder(),
    )
    event = LogNormalizer().normalize(raw_payload)

    scored = scorer.score(event)

    _assert_neutral_tier3_fallback(scored, raw_payload, caplog.text)


class FakeDuplicateBiEncoder(FakeBiEncoder):
    def check_dedup_and_retrieve_batch(
        self, texts: list[str]
    ) -> list[tuple[Any, list[Any]]]:
        return [
            (
                DedupResult(is_duplicate=True, similarity=0.95),
                [
                    ATTACKCandidate(
                        technique_id="T1110",
                        name="Brute Force",
                        description="Test",
                        similarity=0.91,
                    )
                ],
            )
            for _ in texts
        ]


def test_duplicate_penalty_and_entity_boost() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 0.5,
                    "entity_boost": 0.3,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNERWithEntities(),
        biencoder=FakeDuplicateBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )
    scored = scorer.score(event)

    assert scored.is_duplicate is True
    assert scored.dedup_similarity == pytest.approx(0.95)


def test_entity_boost_for_non_duplicates() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 0.5,
                    "entity_boost": 0.3,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "entity_boost_value": 0.3,
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNERWithEntities(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=FakeSyslogClassifier(),
    )

    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )
    scored = scorer.score(event)

    assert scored.is_duplicate is False
    assert scored.entity_boost == pytest.approx(0.3)


def test_sigma_no_rules_dir_is_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("logfilter.pipeline.scorer.Path", lambda _p: tmp_path / "nonexistent")
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.sigma_matched is False


def test_classifier_exception_uses_neutral_scores() -> None:
    class BrokenClassifier(FakeClassifier):
        def predict_proba(self, feature_vectors):
            del feature_vectors
            raise RuntimeError("model broken")

    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=BrokenClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.classifier_score == pytest.approx(0.5)
    assert scored.score_degraded is True


def test_no_model_loaded_marks_degraded() -> None:
    class NotReadyClassifier(FakeClassifier):
        def is_ready(self) -> bool:
            return False

        def predict_proba(self, feature_vectors):
            return np.full(len(feature_vectors), 0.5, dtype=np.float32)

    class NotReadySyslogClassifier(FakeSyslogClassifier):
        def is_ready(self) -> bool:
            return False

    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=NotReadyClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
        syslog_classifier=NotReadySyslogClassifier(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.classifier_score == pytest.approx(0.5)
    assert scored.score_degraded is True


def test_healthy_classifier_not_degraded() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.score_degraded is False


def test_tier2_not_ready_keeps_tier1_scores() -> None:
    class Tier2NotReady(Tier2Classifier):
        def should_escalate(self, tier1_prob: float) -> bool:
            del tier1_prob
            return True

        def is_ready(self) -> bool:
            return False

        def predict_proba(self, texts: list[str]) -> np.ndarray:
            del texts
            raise AssertionError("should not be called")

    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=Tier2NotReady(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.tier2_used is False


def test_tier2_exception_keeps_tier1_scores() -> None:
    class Tier2Broken(Tier2Classifier):
        def should_escalate(self, tier1_prob: float) -> bool:
            del tier1_prob
            return True

        def is_ready(self) -> bool:
            return True

        def predict_proba(self, texts: list[str]) -> np.ndarray:
            del texts
            raise RuntimeError("tier2 broken")

    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=Tier2Broken(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    assert scored.tier2_used is False


def test_feature_vector_matching_branches() -> None:
    class FeatureClassifier(FakeClassifier):
        feature_names = ["exact.match", "partial token"]
        expected_feature_count = 2

    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FeatureClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("exact.match event with partial token here")
    scored = scorer.score(event)
    assert scored.classifier_score == pytest.approx(0.82)


def test_feature_cache_is_reused() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scorer.score(event)
    cached = scorer._feature_cache_names
    scorer.score(event)
    assert scorer._feature_cache_names is cached


def test_compute_score_with_sigma_match() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    scored.sigma_matched = True
    score = scorer._compute_score(scored)
    assert score >= 0.90


def test_routing_label_info() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    scored.ai_threat_score = 0.01
    label = scorer._routing_label(scored)
    assert label == "INFO"


def test_confidence_with_no_signals() -> None:
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.15,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize("test event")
    scored = scorer.score(event)
    scored.classifier_score = 0.0
    scored.entities = {}
    scored.cross_encoder_max = 0.0
    conf = scorer._confidence(scored)
    assert conf == 0.0


def test_sigma_rule_matches_contains() -> None:
    from sigma.rule import SigmaRule

    rule = SigmaRule.from_yaml("""
title: SSH Brute Force
id: 12345678-1234-1234-1234-123456789abc
logsource:
    category: process_creation
    product: linux
detection:
    sel:
        CommandLine|contains:
            - 'failed password'
    condition: sel
level: high
""")
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    assert scorer._sigma_rule_matches(rule, "user failed password from 10.0.0.5")
    assert not scorer._sigma_rule_matches(rule, "accepted password")


def test_sigma_rule_matches_startswith() -> None:
    from sigma.rule import SigmaRule

    rule = SigmaRule.from_yaml("""
title: SSH Start
logsource:
    category: process_creation
    product: linux
detection:
    sel:
        CommandLine|startswith: 'ssh '
    condition: sel
level: low
""")
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    assert scorer._sigma_rule_matches(rule, "ssh user@host")
    assert not scorer._sigma_rule_matches(rule, "sudo ssh user@host")


def test_sigma_rule_matches_endswith() -> None:
    from sigma.rule import SigmaRule

    rule = SigmaRule.from_yaml("""
title: SSH End
logsource:
    category: process_creation
    product: linux
detection:
    sel:
        CommandLine|endswith: '.exe'
    condition: sel
level: low
""")
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    assert scorer._sigma_rule_matches(rule, "malware.exe")
    assert not scorer._sigma_rule_matches(rule, "malware.bin")


def test_sigma_rule_matches_exact() -> None:
    from sigma.rule import SigmaRule

    rule = SigmaRule.from_yaml("""
title: Exact Match
logsource:
    category: process_creation
    product: linux
detection:
    sel:
        CommandLine: 'exact'
    condition: sel
level: low
""")
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    assert scorer._sigma_rule_matches(rule, "the exact command")
    assert not scorer._sigma_rule_matches(rule, "inexact")


def test_sigma_rule_skips_unmappable_field() -> None:
    from sigma.rule import SigmaRule

    rule = SigmaRule.from_yaml("""
title: Windows Image
logsource:
    category: process_creation
    product: windows
detection:
    sel:
        Image|contains: 'cmd.exe'
    condition: sel
level: low
""")
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    assert not scorer._sigma_rule_matches(rule, "cmd.exe running")


def test_sigma_rule_no_detection_returns_false() -> None:
    scorer = LogScorer(
        config={"scoring": {"routing": {"high": 0.85, "medium": 0.50, "low": 0.20}}},
        classifier=FakeClassifier(),
        tier2_classifier=FakeTier2Classifier(),
    )
    class FakeRule:
        detection = None

    assert not scorer._sigma_rule_matches(FakeRule(), "any text")
