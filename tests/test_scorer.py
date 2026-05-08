"""Unit tests for scoring orchestration."""

from __future__ import annotations

import numpy as np

from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer


class FakeClassifier:
    feature_names = ["failed password from", "namenode block received"]
    expected_feature_count = 2

    def __init__(self) -> None:
        self.last_vectors: np.ndarray | None = None

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.last_vectors = feature_vectors
        return np.array([0.82], dtype=np.float32)

    def is_ready(self) -> bool:
        return True


class FakeNERResult:
    has_high_value_entities = False

    def to_dict(self) -> dict:
        return {"confidence": 0.0, "has_high_value_entities": False}

    def flat_entity_string(self) -> str:
        return ""


class FakeNER:
    def extract_batch(self, texts: list[str]) -> list[FakeNERResult]:
        return [FakeNERResult() for _ in texts]


class FakeDedup:
    is_duplicate = False
    similarity = 0.0


class FakeCandidate:
    technique_id = "T1110"
    name = "Brute Force"
    description = "Password guessing and repeated failed logons"
    similarity = 0.91


class FakeBiEncoder:
    def check_dedup_and_retrieve_batch(self, texts: list[str]) -> list[tuple]:
        return [(FakeDedup(), [FakeCandidate()]) for _ in texts]


class FakeCrossScore:
    technique_id = "T1110"
    name = "Brute Force"
    score = 0.0


class FakeCrossEncoder:
    def score_batch(self, log_texts: list[str], candidates_per_log: list[list[dict]]) -> list[list]:
        return [[FakeCrossScore()] for _ in log_texts]


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
