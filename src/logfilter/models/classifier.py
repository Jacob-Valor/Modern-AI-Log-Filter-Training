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
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent  # project root


class LogClassifier:
    """
    Lazy-loading wrapper around the ONNX log anomaly classifier.

    Falls back to the native XGBoost model if ONNX Runtime is unavailable.
    """

    def __init__(
        self,
        model_path: str | Path = ROOT / "models" / "log_classifier.onnx",
        scaler_path: str | Path = ROOT / "models" / "scaler.pkl",
        feature_names_path: str | Path = ROOT / "models" / "feature_names.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_names_path = Path(feature_names_path)

        self._session = None  # ONNX Runtime InferenceSession
        self._xgb_model = None  # Fallback XGBoost model
        self._scaler = None
        self._feature_names: list[str] = []
        self._input_name: str = ""

    def _load(self) -> None:
        # Load scaler
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
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
                return np.array([float(d.get("1", 0.5)) for d in proba_output])
            else:
                return proba_output[:, 1]

        if self._xgb_model is not None:
            return self._xgb_model.predict_proba(X)[:, 1]

        # No model: return neutral probability
        return np.full(len(feature_vectors), 0.5)

    def predict_single(self, feature_vector: np.ndarray) -> float:
        """Predict failure probability for a single event."""
        return float(self.predict_proba(feature_vector.reshape(1, -1))[0])

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names and self.feature_names_path.exists():
            self._feature_names = json.loads(self.feature_names_path.read_text())
        return self._feature_names

    def is_ready(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._session is not None or self._xgb_model is not None
