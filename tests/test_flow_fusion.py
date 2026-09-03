"""Tests for attested flow fusion and feature-schema enforcement."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-not-found]

from logfilter.models.flow_classifier import FlowClassifier
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES

FlowFusionPolicy: Any = importlib.import_module("logfilter.models.flow_fusion").FlowFusionPolicy


def _policy(tmp_path: Path, target: str = "0.01") -> Any:
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "fixed_fpr": {
                    "0.01": {
                        "status": "calibrated",
                        "composite_threshold": 0.5,
                        "general_probability_threshold": 0.4,
                        "bot_probability_threshold": 0.2,
                        "alert_floor": 0.65,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return FlowFusionPolicy(path, target)


def test_flow_fusion_raises_only_when_a_calibrated_head_crosses(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    assert policy.calibrated_risk(0.39, 0.19) == pytest.approx(0.39)
    assert policy.calibrated_risk(0.40, 0.01) == pytest.approx(0.65)
    assert policy.calibrated_risk(0.01, 0.20) == pytest.approx(0.65)
    assert policy.composite_threshold == pytest.approx(0.5)


def test_flow_classifier_rejects_reordered_feature_artifact(tmp_path: Path) -> None:
    names = list(FLOW_FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    features_path = tmp_path / "features.json"
    scaler_path = tmp_path / "scaler.json"
    features_path.write_text(json.dumps(names), encoding="utf-8")
    scaler_path.write_text(
        json.dumps(
            {
                "type": "MaxAbsScaler",
                "scale": [1.0] * len(names),
                "n_features_in": len(names),
            }
        ),
        encoding="utf-8",
    )
    classifier = FlowClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=scaler_path,
        feature_names_path=features_path,
    )
    with pytest.raises(ValueError, match="feature artifact order"):
        classifier.is_loaded()
