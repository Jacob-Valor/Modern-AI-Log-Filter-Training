"""Validation-calibrated Bot-vs-rest flow classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from logfilter.models.flow_classifier import FlowClassifier


class BotFlowClassifier(FlowClassifier):
    """ONNX Bot flow head with a validation-only fixed-FPR operating point."""

    def __init__(
        self,
        model_path: str | Path = "models/flow/bot_classifier.onnx",
        scaler_path: str | Path = "models/flow/bot_scaler.json",
        feature_names_path: str | Path = "models/flow/bot_feature_names.json",
        calibration_path: str | Path = "models/flow/bot_calibration.json",
        target_fpr: str = "0.01",
        alert_floor: float = 0.65,
    ) -> None:
        super().__init__(model_path, scaler_path, feature_names_path)
        self.calibration_path = Path(calibration_path)
        self.target_fpr = target_fpr
        self.alert_floor = self._probability(alert_floor, "Bot alert floor")
        self._operating_threshold: float | None = None

    @staticmethod
    def _probability(value: Any, name: str) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not 0.0 <= converted <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
        return converted

    @property
    def operating_threshold(self) -> float:
        if self._operating_threshold is None:
            try:
                data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                row = data["fixed_fpr"][self.target_fpr]
                if row.get("status") != "calibrated" or row.get("threshold") is None:
                    raise ValueError("selected Bot fixed-FPR target is not calibrated")
                self._operating_threshold = self._probability(
                    row["threshold"], "Bot operating threshold"
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Cannot load Bot calibration: {exc}") from exc
        return self._operating_threshold

    def calibrated_risk(self, probability: Any) -> float:
        raw = self._probability(probability, "Bot probability")
        return max(raw, self.alert_floor) if raw >= self.operating_threshold else raw
