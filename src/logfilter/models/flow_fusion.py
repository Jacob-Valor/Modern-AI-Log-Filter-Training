"""Attested validation-calibrated policy for final flow-head fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FlowFusionPolicy:
    """Map general and Bot probabilities to one calibrated runtime risk."""

    def __init__(
        self,
        calibration_path: str | Path = "models/flow/fusion_calibration.json",
        target_fpr: str = "0.01",
    ) -> None:
        self.calibration_path = Path(calibration_path)
        self.target_fpr = target_fpr
        self._row: dict[str, Any] | None = None

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
    def row(self) -> dict[str, Any]:
        if self._row is None:
            try:
                data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                selected = data["fixed_fpr"][self.target_fpr]
                if selected.get("status") != "calibrated":
                    raise ValueError("selected flow fusion target is not calibrated")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Cannot load flow fusion policy: {exc}") from exc
            self._row = selected
        row = self._row
        if row is None:
            raise RuntimeError("Flow fusion policy was not initialized")
        return row

    @property
    def composite_threshold(self) -> float:
        return self._probability(self.row["composite_threshold"], "composite threshold")

    def calibrated_risk(self, general_probability: Any, bot_probability: Any) -> float:
        general = self._probability(general_probability, "general flow probability")
        bot = self._probability(bot_probability, "Bot flow probability")
        general_threshold = self._probability(
            self.row["general_probability_threshold"], "general flow threshold"
        )
        bot_threshold = self._probability(
            self.row["bot_probability_threshold"], "Bot flow threshold"
        )
        floor = self._probability(self.row["alert_floor"], "flow alert floor")
        if general >= general_threshold or bot >= bot_threshold:
            return max(general, bot, floor)
        return max(general, bot)
