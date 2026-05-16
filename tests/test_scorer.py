"""Unit tests for scoring orchestration."""

from __future__ import annotations

import numpy as np

from logfilter.models.biencoder import ATTACKCandidate, BiEncoderModel, DedupResult
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel, CrossEncoderScore
from logfilter.models.ner import ExtractedEntities, NERModel
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer


class FakeClassifier(LogClassifier):
    feature_names = ["failed password from", "namenode block received"]
    expected_feature_count = 2

    def __init__(self) -> None:
        self.last_vectors: np.ndarray | None = None

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.last_vectors = feature_vectors
        return np.array([0.82], dtype=np.float32)

    def is_ready(self) -> bool:
        return True


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


def test_classifier_score_is_applied_to_composite_score() -> None:
    classifier = FakeClassifier()
    scorer = LogScorer(
        config={
            "scoring": {
                "weights": {
                    "classifier": 1.0,
                    "entity_boost": 0.0,
                    "cross_encoder": 0.0,
                    "novelty": 0.0,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            }
        },
        classifier=classifier,
        tier2_classifier=FakeTier2Classifier(),
        ner_model=FakeNER(),
        biencoder=FakeBiEncoder(),
        cross_encoder=FakeCrossEncoder(),
    )

    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )
    scored = scorer.score(event)

    assert abs(scored.classifier_score - 0.82) < 1e-6
    assert abs(scored.ai_threat_score - 0.82) < 1e-6
    assert scored.ai_priority == "MEDIUM"
    assert classifier.last_vectors is not None
    assert classifier.last_vectors.shape == (1, 2)
    assert classifier.last_vectors[0, 0] > 0


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
                    "novelty": 0.0,
                },
                "routing": {"high": 0.85, "medium": 0.50, "low": 0.20},
            },
        },
        classifier=classifier,
        tier2_classifier=FakeTier2Classifier(),
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
