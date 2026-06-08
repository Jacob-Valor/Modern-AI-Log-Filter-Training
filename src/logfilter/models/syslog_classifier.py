"""
Syslog-only classifier wrapper.

Uses a lightweight XGBoost model trained on 100 syslog features to
classify real syslog events. The main HDFS classifier produces uniform
scores for syslog events because it was trained on 2255 features where
HDFS patterns dominate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent
SYSLOG_MODEL_DIR = ROOT / "models" / "syslog"


class SyslogClassifier:
    """Lazy-loading wrapper for the syslog-only XGBoost classifier."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else SYSLOG_MODEL_DIR
        self._session: Any | None = None
        self._xgb_model: Any | None = None
        self._scaler: Any | None = None
        self._feature_names: list[str] = []

    def _load(self) -> None:
        onnx_path = self.model_dir / "log_classifier_syslog.onnx"
        json_path = self.model_dir / "log_classifier_syslog.json"
        scaler_path = self.model_dir / "scaler_syslog.json"
        features_path = self.model_dir / "feature_names_syslog.json"

        if not features_path.exists():
            logger.warning("Syslog feature names not found", path=str(features_path))
            return

        import json as _json
        self._feature_names = _json.loads(features_path.read_text())

        if scaler_path.exists():
            from logfilter.models.classifier import SafeMaxAbsScaler
            self._scaler = SafeMaxAbsScaler.from_json(scaler_path)

        if onnx_path.exists():
            try:
                import onnxruntime as rt
                self._session = rt.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                logger.info("Syslog ONNX classifier loaded", path=str(onnx_path))
                return
            except Exception as e:
                logger.warning("Syslog ONNX load failed, trying XGBoost", error=str(e))

        if json_path.exists():
            import xgboost as xgb
            self._xgb_model = xgb.XGBClassifier()
            self._xgb_model.load_model(str(json_path))
            logger.info("Syslog XGBoost classifier loaded", path=str(json_path))
        else:
            logger.warning("No syslog classifier found")

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names and not self._session and not self._xgb_model:
            self._load()
        return self._feature_names

    def is_ready(self) -> bool:
        if self._session is None and self._xgb_model is None:
            self._load()
        return self._session is not None or self._xgb_model is not None

    def predict_proba(self, feature_vectors: np.ndarray) -> np.ndarray:
        if self._session is None and self._xgb_model is None:
            self._load()

        X = feature_vectors.astype(np.float32)

        if self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)

        if self._session is not None:
            outputs = self._session.run(None, {self._input_name: X})
            proba = outputs[1]
            result = proba[:, 1] if proba.ndim > 1 else proba
        elif self._xgb_model is not None:
            proba = self._xgb_model.predict_proba(X)
            result = proba[:, 1]
        else:
            return np.full(len(feature_vectors), 0.5)

        # NOTE: zero-vector clamping was removed after retraining on WitFoo data.
        # The retrained model correctly assigns high malicious probability to
        # zero-vector events because 94% of malicious WitFoo events have no
        # matching features. Clamping to 0.5 masked this signal.
        return result
