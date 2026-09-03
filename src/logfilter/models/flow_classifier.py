"""Flow-aware numeric classifier wrapper.

The implementation intentionally reuses the hardened ONNX/XGBoost loader and
JSON-backed scaler used by the tier-1 classifier while keeping independent,
versioned flow artifacts.
"""

from __future__ import annotations

from pathlib import Path

from logfilter.models.classifier import LogClassifier
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[3]


class FlowClassifier(LogClassifier):
    """Binary attack probability model for numeric network-flow observations."""

    def __init__(
        self,
        model_path: str | Path = ROOT / "models" / "flow" / "flow_classifier.onnx",
        scaler_path: str | Path = ROOT / "models" / "flow" / "scaler.json",
        feature_names_path: str | Path = ROOT / "models" / "flow" / "feature_names.json",
    ) -> None:
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            feature_names_path=feature_names_path,
        )

    def _load(self) -> None:
        super()._load()
        if tuple(self._feature_names) != FLOW_FEATURE_NAMES:
            raise ValueError("Flow feature artifact order does not match runtime schema")
        expected = len(FLOW_FEATURE_NAMES)
        if self._scaler is None or self._scaler.n_features_in_ != expected:
            raise ValueError("Flow scaler width does not match runtime schema")
        if self._session is not None:
            shape = self._session.get_inputs()[0].shape
            if len(shape) < 2 or shape[1] != expected:
                raise ValueError("Flow ONNX input width does not match runtime schema")
