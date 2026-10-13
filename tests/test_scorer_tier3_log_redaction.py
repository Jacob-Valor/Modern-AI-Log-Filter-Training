"""Working redaction check for the tier-3 model-failure paths.

``tests/test_scorer.py::_assert_neutral_tier3_fallback`` asserts
``raw_payload not in caplog.text``. That check can never fire: the scorer logs
via ``structlog.get_logger(__name__)``, nothing in ``src/`` calls
``structlog.configure(...)`` or installs a stdlib bridge, so structlog writes
straight to stdout and ``caplog.text`` is always the empty string.

These tests do the same assertion against the stream structlog actually writes
to (``capsys``), which makes it real.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from logfilter.models.biencoder import ATTACKCandidate, BiEncoderModel, DedupResult
from logfilter.models.classifier import LogClassifier
from logfilter.models.cross_encoder import CrossEncoderModel, CrossEncoderScore
from logfilter.models.ner import ExtractedEntities, NERModel
from logfilter.models.syslog_classifier import SyslogClassifier
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

#: Distinctive string standing in for the raw log line. It must never appear in
#: any log output produced while a tier-3 model failure is handled.
RAW_PAYLOAD = "RAW_PAYLOAD_SHOULD_NOT_LOG"


class _FakeClassifier(LogClassifier):
    def __init__(self) -> None:
        self.last_vectors: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str]:
        return ["failed password from"]

    @property
    def expected_feature_count(self) -> int:
        return 1

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        self.last_vectors = feature_vectors
        return np.full(len(feature_vectors), 0.82, dtype=np.float32)

    def is_ready(self) -> bool:
        return True


class _FakeSyslogClassifier(SyslogClassifier):
    def __init__(self) -> None:
        self._feature_names = ["sshd+failed password"]

    def is_ready(self) -> bool:
        return True

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        return np.full(len(feature_vectors), 0.82, dtype=np.float32)


class _FakeTier2Classifier(Tier2Classifier):
    def should_escalate(self, tier1_prob: float) -> bool:
        _ = tier1_prob
        return False

    def is_ready(self) -> bool:
        return False

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        _ = texts
        raise AssertionError("Tier-2 must not run in this test")


class _FakeNER(NERModel):
    def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        result = ExtractedEntities()
        result.indicators = ["10.0.0.5"]
        result.confidence = 0.95
        result.has_high_value_entities = True
        return [result for _ in texts]


class _FakeBiEncoder(BiEncoderModel):
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


class _FakeCrossEncoder(CrossEncoderModel):
    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[CrossEncoderScore]]:
        _ = candidates_per_log
        return [[CrossEncoderScore("T1110", "Brute Force", 0.0)] for _ in log_texts]


class _BrokenBiEncoder(_FakeBiEncoder):
    def check_dedup_and_retrieve_batch(
        self, texts: list[str]
    ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
        raise RuntimeError(f"BiEncoder failed while scoring {texts[0]}")


class _BrokenNER(_FakeNER):
    def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        raise RuntimeError(f"NER failed while scoring {texts[0]}")


class _BrokenCrossEncoder(_FakeCrossEncoder):
    def score_batch(
        self,
        log_texts: list[str],
        candidates_per_log: list[list[dict[str, str]]],
    ) -> list[list[CrossEncoderScore]]:
        _ = candidates_per_log
        raise RuntimeError(f"CrossEncoder failed while scoring {log_texts[0]}")


#: stage name -> the constructor argument that injects the broken model.
_BROKEN_MODELS: dict[str, tuple[str, Callable[[], Any]]] = {
    "biencoder": ("biencoder", _BrokenBiEncoder),
    "ner": ("ner_model", _BrokenNER),
    "cross_encoder": ("cross_encoder", _BrokenCrossEncoder),
}


def _scorer(**overrides: Any) -> LogScorer:
    models: dict[str, Any] = {
        "ner_model": _FakeNER(),
        "biencoder": _FakeBiEncoder(),
        "cross_encoder": _FakeCrossEncoder(),
    }
    models.update(overrides)
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
        classifier=_FakeClassifier(),
        tier2_classifier=_FakeTier2Classifier(),
        syslog_classifier=_FakeSyslogClassifier(),
        **models,
    )


def _run_and_capture(scorer: LogScorer, capsys) -> tuple[Any, str]:
    """Score RAW_PAYLOAD and return (scored_event, everything-structlog-printed)."""
    capsys.readouterr()  # discard anything printed during setup
    event = LogNormalizer().normalize(RAW_PAYLOAD)
    scored = scorer.score(event)
    captured = capsys.readouterr()
    return scored, captured.out + captured.err


def test_capsys_captures_the_structlog_failure_warning(capsys) -> None:
    """Sanity check for the harness: the failure warning is observable here.

    Without this, a future structlog configuration change could silence the
    output and quietly turn every redaction assertion below into a tautology.
    """
    _scored, log_text = _run_and_capture(_scorer(biencoder=_BrokenBiEncoder()), capsys)

    assert "Tier-3 scoring failed" in log_text, (
        "structlog output was not captured; the redaction assertions would be "
        f"vacuous. captured={log_text!r}"
    )


@pytest.mark.parametrize("stage", sorted(_BROKEN_MODELS))
def test_tier3_failure_log_does_not_echo_raw_log_line(stage: str, capsys) -> None:
    """A tier-3 failure warning must never contain the raw log payload.

    This is the only check that a model-failure path does not leak the log
    line it was processing into the operator's console.
    """
    argument, factory = _BROKEN_MODELS[stage]
    scorer = _scorer(**{argument: factory()})

    scored, log_text = _run_and_capture(scorer, capsys)

    assert RAW_PAYLOAD not in log_text, (
        f"the {stage} failure path echoed the raw log payload to stdout"
    )
    # The failure was still handled rather than silently dropped.
    assert scored.score_degraded is True
    assert scored.entities == {}