"""Endpoint telemetry parsing, feature, and novelty-model tests."""

# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from logfilter.models.endpoint_novelty import EndpointNoveltyModel
from logfilter.pipeline.endpoint_features import ENDPOINT_FEATURE_NAMES, extract_endpoint_vector
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType
from logfilter.pipeline.scorer import LogScorer

_MODEL_AVAILABLE = Path("models/endpoint/endpoint_novelty.onnx").exists()


def _evil_record() -> dict:
    return {
        "timestamp": 500.0,
        "processId": 7555,
        "threadId": 7555,
        "parentProcessId": 7548,
        "userId": 1001,
        "mountNamespace": 4026531840,
        "processName": "tsm",
        "hostName": "beth-test-host",
        "eventId": 56,
        "eventName": "clone",
        "argsNum": 5,
        "returnValue": 7598,
        "args": "CLONE_VM CLONE_FS CLONE_FILES CLONE_THREAD CLONE_SETTLS",
    }


def _config() -> dict:
    return {
        "models": {
            "flow": {"enabled": False},
            "endpoint_novelty": {
                "enabled": True,
                "path": "models/endpoint/endpoint_novelty.onnx",
                "config_path": "models/endpoint/novelty_config.json",
            },
            "ner": {"enabled": False},
            "biencoder": {"enabled": False},
            "cross_encoder": {"enabled": False},
        },
        "scoring": {
            "weights": {
                "classifier": 1.0,
                "entity_boost": 0.0,
                "cross_encoder": 0.0,
                "novelty": 0.0,
            },
            "routing": {"high": 0.85, "medium": 0.5, "low": 0.2},
        },
    }


def test_beth_endpoint_record_is_normalized_without_labels():
    event = LogNormalizer().normalize(json.dumps(_evil_record()))
    assert event.source_type == LogSourceType.ENDPOINT
    assert event.host == "beth-test-host"
    assert event.fields["process_id"] == 7555
    assert "evil" not in event.fields
    vector = extract_endpoint_vector(event)
    assert vector is not None
    assert vector.shape == (len(ENDPOINT_FEATURE_NAMES),)


@pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="endpoint artifacts are provisioned with 'make train-endpoint'",
)
def test_endpoint_novelty_onnx_scores_open_set_record():
    event = LogNormalizer().normalize(json.dumps(_evil_record()))
    vector = extract_endpoint_vector(event)
    assert vector is not None
    probability = EndpointNoveltyModel().predict_proba(vector.reshape(1, -1))[0]
    assert probability > 0.85


@pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="endpoint artifacts are provisioned with 'make train-endpoint'",
)
def test_endpoint_novelty_is_integrated_into_scorer():
    event = LogNormalizer().normalize(json.dumps(_evil_record()))
    result = LogScorer(_config()).score(event)
    assert result.endpoint_score > 0.85
    assert result.classifier_score == result.endpoint_score
    assert result.ai_priority == "HIGH"


def test_missing_endpoint_artifacts_mark_scoring_degraded(tmp_path):
    model = EndpointNoveltyModel(
        model_path=tmp_path / "missing.onnx",
        config_path=tmp_path / "missing.json",
    )
    event = LogNormalizer().normalize(json.dumps(_evil_record()))
    result = LogScorer(_config(), endpoint_novelty_model=model).score(event)
    assert result.score_degraded is True
    assert result.endpoint_score == 0.0
