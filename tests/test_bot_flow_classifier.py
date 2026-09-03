"""Tests for validation-calibrated Bot flow risk mapping."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-not-found]

BotFlowClassifier: Any = importlib.import_module(
    "logfilter.models.bot_flow_classifier"
).BotFlowClassifier


def _classifier(tmp_path: Path, target: str = "0.01") -> Any:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "fixed_fpr": {
                    "0.001": {"status": "calibrated", "threshold": 0.4},
                    "0.01": {"status": "calibrated", "threshold": 0.2},
                    "0.05": {"status": "calibrated", "threshold": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    return BotFlowClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=tmp_path / "missing-scaler.json",
        feature_names_path=tmp_path / "missing-features.json",
        calibration_path=calibration,
        target_fpr=target,
        alert_floor=0.65,
    )


def test_probability_crossing_validation_threshold_gets_alert_floor(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path)
    assert classifier.operating_threshold == pytest.approx(0.2)
    assert classifier.calibrated_risk(0.19) == pytest.approx(0.19)
    assert classifier.calibrated_risk(0.2) == pytest.approx(0.65)


def test_fixed_fpr_target_selects_distinct_validation_threshold(tmp_path: Path) -> None:
    assert _classifier(tmp_path, target="0.001").operating_threshold == pytest.approx(0.4)
    assert _classifier(tmp_path, target="0.05").operating_threshold == pytest.approx(0.1)


def test_missing_target_is_rejected(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, target="missing")
    with pytest.raises(RuntimeError, match="Cannot load Bot calibration"):
        _ = classifier.operating_threshold
