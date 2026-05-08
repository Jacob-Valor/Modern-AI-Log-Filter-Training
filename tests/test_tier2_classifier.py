"""Tests for Tier-2 transformer classifier cascade behavior."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.ner import NERModel
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer


class FakeClassifier:
    feature_names = ["failed password from"]
    expected_feature_count = 1

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        return np.full(feature_vectors.shape[0], self.probability, dtype=np.float32)

    def is_ready(self) -> bool:
        return True


class FakeTier2:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.seen_texts: list[str] = []

    def is_ready(self) -> bool:
        return True

    def should_escalate(self, tier1_prob: float) -> bool:
        return 0.10 <= tier1_prob <= 0.90

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.seen_texts = texts
        return np.full(len(texts), self.probability, dtype=np.float32)


class FakeNERResult:
    has_high_value_entities = False

    def to_dict(self) -> dict[str, float | bool]:
        return {"confidence": 0.0, "has_high_value_entities": False}

    def flat_entity_string(self) -> str:
        return ""


class FakeNER:
    def extract_batch(self, texts: list[str]) -> list[FakeNERResult]:
        return [FakeNERResult() for _ in texts]


class FakeDedup:
    is_duplicate = False
    similarity = 0.0


class FakeBiEncoder:
    def check_dedup_and_retrieve_batch(self, texts: list[str]) -> list[tuple[FakeDedup, list]]:
        return [(FakeDedup(), []) for _ in texts]


class FakeCrossEncoder:
    def score_batch(self, log_texts: list[str], candidates_per_log: list[list[dict]]) -> list[list]:
        del candidates_per_log
        return [[] for _ in log_texts]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeONNXSession:
    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, output_names, feed):  # noqa: ANN001
        del output_names
        batch_size = feed["input_ids"].shape[0]
        logits = np.array([[2.0, -2.0], [-1.0, 3.0]], dtype=np.float32)[:batch_size]
        return [logits]


class FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs) -> dict[str, np.ndarray]:  # noqa: ANN003
        del kwargs
        return {
            "input_ids": np.ones((len(texts), 4), dtype=np.int64),
            "attention_mask": np.ones((len(texts), 4), dtype=np.int64),
        }


def test_tier2_not_ready_when_artifacts_missing(tmp_path) -> None:
    classifier = Tier2Classifier(model_dir=tmp_path)

    result = classifier.predict_proba(["normal log", "failure log"])

    assert not classifier.is_ready()
    np.testing.assert_allclose(result, np.array([0.5, 0.5], dtype=np.float32))


def test_tier2_should_escalate_band() -> None:
    classifier = Tier2Classifier()

    assert not classifier.should_escalate(0.05)
    assert classifier.should_escalate(0.10)
    assert classifier.should_escalate(0.50)
    assert classifier.should_escalate(0.90)
    assert not classifier.should_escalate(0.95)


def test_tier2_predict_returns_valid_probs(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "log_classifier_tier2.onnx").write_bytes(b"fake")
    (tmp_path / "tier2_label_map.json").write_text('{"0": "normal", "1": "failure"}')
    classifier = Tier2Classifier(model_dir=tmp_path)
    classifier._tokenizer = FakeTokenizer()
    classifier._session = FakeONNXSession()
    classifier._onnx_input_names = ["input_ids", "attention_mask"]
    classifier._load_attempted = True

    result = classifier.predict_proba(["normal log", "failure log"])

    assert result.shape == (2,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert result[1] > result[0]


def test_scorer_tier2_overrides_when_uncertain() -> None:
    tier2 = FakeTier2(probability=0.9)
    scorer = _make_scorer(FakeClassifier(probability=0.5), tier2)
    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )

    scored = scorer.score(event)

    assert scored.classifier_score == pytest.approx(0.9)
    assert scored.tier2_score == pytest.approx(0.9)
    assert scored.tier2_used is True
    assert tier2.seen_texts == [event.raw]


def test_scorer_tier2_skipped_when_confident() -> None:
    tier2 = FakeTier2(probability=0.1)
    scorer = _make_scorer(FakeClassifier(probability=0.95), tier2)
    event = LogNormalizer().normalize("Jan 15 11:07:53 prod app[1]: startup completed")

    scored = scorer.score(event)

    assert scored.classifier_score == pytest.approx(0.95)
    assert scored.tier2_score == 0.0
    assert scored.tier2_used is False
    assert tier2.seen_texts == []


def test_tier2_predict_proba_empty_input_short_circuits(tmp_path) -> None:
    classifier = Tier2Classifier(model_dir=tmp_path)
    result = classifier.predict_proba([])
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_tier2_warn_degraded_only_emits_once(tmp_path, caplog) -> None:
    classifier = Tier2Classifier(model_dir=tmp_path)
    import logging as _logging

    caplog.set_level(_logging.WARNING)
    classifier._warn_degraded("first")
    classifier._warn_degraded("second")
    warning_count = sum(1 for r in caplog.records if "first" in r.getMessage())
    second_count = sum(1 for r in caplog.records if "second" in r.getMessage())
    assert warning_count <= 1
    assert second_count == 0


def test_tier2_load_label_map_malformed_json(tmp_path) -> None:
    (tmp_path / "tier2_label_map.json").write_text("{not valid json")
    classifier = Tier2Classifier(model_dir=tmp_path)
    classifier._load_label_map()
    assert classifier._failure_label_index == 1


def test_tier2_load_label_map_with_inverted_labels(tmp_path) -> None:
    (tmp_path / "tier2_label_map.json").write_text('{"0": "failure", "1": "normal"}')
    classifier = Tier2Classifier(model_dir=tmp_path)
    classifier._load_label_map()
    assert classifier._failure_label_index == 0


def test_tier2_predict_onnx_returns_neutral_when_tokenizer_missing(tmp_path) -> None:
    classifier = Tier2Classifier(model_dir=tmp_path)
    classifier._session = FakeONNXSession()
    classifier._tokenizer = None
    result = classifier._predict_onnx(["log line"])
    np.testing.assert_allclose(result, np.array([0.5], dtype=np.float32))


def test_tier2_predict_handles_session_run_exception(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "log_classifier_tier2.onnx").write_bytes(b"fake")
    (tmp_path / "tier2_label_map.json").write_text('{"0": "normal", "1": "failure"}')

    class BrokenSession:
        def get_inputs(self) -> list:
            return [FakeInput("input_ids")]

        def run(self, output_names, feed):  # noqa: ANN001
            del output_names, feed
            raise RuntimeError("ORT crashed")

    classifier = Tier2Classifier(model_dir=tmp_path)
    classifier._tokenizer = FakeTokenizer()
    classifier._session = BrokenSession()
    classifier._onnx_input_names = ["input_ids"]
    classifier._load_attempted = True

    result = classifier.predict_proba(["log line"])

    np.testing.assert_allclose(result, np.array([0.5], dtype=np.float32))


def test_tier2_logits_with_unexpected_shape_returns_neutral(tmp_path) -> None:
    classifier = Tier2Classifier(model_dir=tmp_path)
    bad_logits = np.array([[1.0]], dtype=np.float32)
    result = classifier._failure_probs_from_logits(bad_logits)
    np.testing.assert_allclose(result, np.array([0.5], dtype=np.float32))


def test_tier2_load_falls_back_to_neutral_when_transformers_missing(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "log_classifier_tier2.onnx").write_bytes(b"fake")
    (tmp_path / "tier2_label_map.json").write_text('{"0": "normal", "1": "failure"}')

    import importlib as _importlib
    real_import_module = _importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):  # noqa: ANN001
        if name == "transformers":
            raise ImportError("transformers not installed in this env")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        "logfilter.models.tier2_classifier.importlib.import_module",
        fake_import_module,
    )

    classifier = Tier2Classifier(model_dir=tmp_path)
    result = classifier.predict_proba(["log line one", "log line two"])

    assert not classifier.is_ready()
    np.testing.assert_allclose(result, np.array([0.5, 0.5], dtype=np.float32))


def test_scorer_tier2_inference_exception_keeps_tier1_scores() -> None:
    class ExplodingTier2:
        def is_ready(self) -> bool:
            return True

        def should_escalate(self, tier1_prob: float) -> bool:
            return 0.10 <= tier1_prob <= 0.90

        def predict_proba(self, texts: list[str]) -> np.ndarray:
            del texts
            raise RuntimeError("tier-2 inference exploded")

    scorer = _make_scorer(
        FakeClassifier(probability=0.5),
        cast(FakeTier2, ExplodingTier2()),
    )
    event = LogNormalizer().normalize(
        "Jan 15 11:07:53 prod sshd[123]: Failed password for root from 10.0.0.5"
    )

    scored = scorer.score(event)

    assert scored.classifier_score == pytest.approx(0.5)
    assert scored.tier2_used is False


def _make_scorer(classifier: FakeClassifier, tier2_classifier: FakeTier2) -> LogScorer:
    return LogScorer(
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
        classifier=cast(LogClassifier, classifier),
        tier2_classifier=cast(Tier2Classifier, tier2_classifier),
        ner_model=cast(NERModel, FakeNER()),
        biencoder=cast(BiEncoderModel, FakeBiEncoder()),
        cross_encoder=cast(CrossEncoderModel, FakeCrossEncoder()),
    )
