"""
ONNX-based log anomaly classifier wrapper.

Loads the XGBoost model exported to ONNX during the training pipeline.
Designed for high-throughput CPU inference with ONNX Runtime.

Input:  event count feature vector (2155-dim float32 array)
Output: (normal_prob, failure_prob) — uses failure_prob as threat score component.

NOTE: This classifier was trained on HDFS TraceBench data. For production
QRadar deployment it should be re-trained (or fine-tuned) on labeled logs
from your own environment. It serves as Tier-1 fast classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent  # project root


@dataclass
class SafeMaxAbsScaler:
    """Minimal, JSON-backed MaxAbsScaler runtime representation."""

    scale: np.ndarray

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        if self.scale.size == 0:
            raise ValueError("Scaler scale vector is empty")
        if not np.isfinite(self.scale).all():
            raise ValueError("Scaler scale vector contains non-finite values")
        if np.any(self.scale == 0):
            raise ValueError("Scaler scale vector contains zero values")

    @property
    def n_features_in_(self) -> int:
        return int(self.scale.size)

    @classmethod
    def from_json(cls, path: str | Path) -> SafeMaxAbsScaler:
        payload = json.loads(Path(path).read_text())
        if payload.get("type") != "MaxAbsScaler":
            raise ValueError("Unsupported scaler type")
        scale = payload.get("scale")
        if not isinstance(scale, list):
            raise ValueError("Scaler JSON must contain a list-valued scale field")
        scaler = cls(np.asarray(scale, dtype=np.float32))
        expected_features = int(payload.get("n_features_in", scaler.n_features_in_))
        if expected_features != scaler.n_features_in_:
            raise ValueError("Scaler feature count does not match scale length")
        return scaler

    @classmethod
    def from_sklearn(cls, scaler) -> SafeMaxAbsScaler:  # noqa: ANN001
        scale = getattr(scaler, "scale_", None)
        if scale is None:
            raise ValueError("Expected a fitted sklearn MaxAbsScaler with scale_")
        return cls(np.asarray(scale, dtype=np.float32))

    def to_json(self, path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "type": "MaxAbsScaler",
            "n_features_in": self.n_features_in_,
            "scale": self.scale.astype(float).tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Scaler input must be a 2D array")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Scaler expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Scaler input contains non-finite input values")
        return matrix / self.scale


class LogClassifier:
    """
    Lazy-loading wrapper around the ONNX log anomaly classifier.

    Falls back to the native XGBoost model if ONNX Runtime is unavailable.
    """

    def __init__(
        self,
        model_path: str | Path = ROOT / "models" / "log_classifier.onnx",
        scaler_path: str | Path = ROOT / "models" / "scaler.json",
        feature_names_path: str | Path = ROOT / "models" / "feature_names.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_names_path = Path(feature_names_path)

        self._session: Any | None = None  # ONNX Runtime InferenceSession
        self._xgb_model: Any | None = None  # Fallback XGBoost model
        self._scaler: SafeMaxAbsScaler | None = None
        self._feature_names: list[str] = []
        self._input_name: str = ""

    def _load(self) -> None:
        if self.scaler_path.exists():
            if self.scaler_path.suffix != ".json":
                raise ValueError(
                    f"Refusing to load unsafe scaler artifact {self.scaler_path}. "
                    "Export scaler parameters to models/scaler.json instead."
                )
            self._scaler = SafeMaxAbsScaler.from_json(self.scaler_path)
            logger.info("Scaler loaded", path=str(self.scaler_path))

        # Load feature names
        if self.feature_names_path.exists():
            self._feature_names = json.loads(self.feature_names_path.read_text())
            logger.info("Feature names loaded", count=len(self._feature_names))

        # Try ONNX Runtime first
        if self.model_path.exists() and self.model_path.suffix == ".onnx":
            try:
                import onnxruntime as rt

                sess_options = rt.SessionOptions()
                sess_options.intra_op_num_threads = 4
                self._session = rt.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                logger.info("ONNX classifier loaded", path=str(self.model_path))
                return
            except Exception as e:
                logger.warning("ONNX Runtime failed, falling back to XGBoost", error=str(e))

        # Fallback: native XGBoost
        json_path = self.model_path.parent / (self.model_path.stem + ".json")
        if json_path.exists():
            import xgboost as xgb

            self._xgb_model = xgb.XGBClassifier()
            self._xgb_model.load_model(str(json_path))
            logger.info("XGBoost classifier loaded (fallback)", path=str(json_path))
        else:
            logger.warning("No classifier model found — returning 0.5 for all events")

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Predict failure probability for a batch of feature vectors.

        Parameters
        ----------
        feature_vectors : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) — probability of failure/anomaly class
        """
        if self._session is None and self._xgb_model is None:
            self._load()

        X = feature_vectors.astype(np.float32)

        # Zero or near-zero vectors (empty/malformed logs) → neutral score
        row_sums = np.abs(X).sum(axis=1)
        neutral_mask = row_sums == 0.0
        if neutral_mask.all():
            return np.full(len(X), 0.5)

        # Apply scaler if available
        if self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)

        if self._session is not None:
            outputs = self._session.run(None, {self._input_name: X})
            # ONNX output: [label_array, probability_map]
            # The second output is a list of dicts or array
            proba_output = outputs[1]
            if isinstance(proba_output, list):
                # List of dicts like [{'0': 0.1, '1': 0.9}, ...]
                result = np.array([float(d.get("1", 0.5)) for d in proba_output])
            else:
                result = proba_output[:, 1]
        elif self._xgb_model is not None:
            result = self._xgb_model.predict_proba(X)[:, 1]
        else:
            result = np.full(len(X), 0.5)

        result[neutral_mask] = 0.5
        return result

    def predict_single(self, feature_vector: np.ndarray) -> float:
        """Predict failure probability for a single event."""
        return float(self.predict_proba(feature_vector.reshape(1, -1))[0])

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names and self.feature_names_path.exists():
            self._feature_names = json.loads(self.feature_names_path.read_text())
        return self._feature_names

    @property
    def expected_feature_count(self) -> int:
        """Best-effort input feature count for callers that need to build vectors."""
        if self.feature_names:
            return len(self.feature_names)
        if self._scaler is not None and hasattr(self._scaler, "n_features_in_"):
            return int(self._scaler.n_features_in_)
        if self._session is not None:
            shape = self._session.get_inputs()[0].shape
            if len(shape) > 1 and isinstance(shape[1], int):
                return int(shape[1])
        if self._xgb_model is not None and hasattr(self._xgb_model, "n_features_in_"):
            return int(self._xgb_model.n_features_in_)
        return 0

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded successfully."""
        if self._session is None and self._xgb_model is None:
            self._load()
        return self._session is not None or self._xgb_model is not None

    def is_ready(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._session is not None or self._xgb_model is not None
