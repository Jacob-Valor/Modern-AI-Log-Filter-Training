"""ONNX runtime wrapper for the BETH endpoint novelty model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

ROOT = Path(__file__).resolve().parents[3]
logger = structlog.get_logger(__name__)


class EndpointNoveltyModel:
    """Convert IsolationForest score-samples output into a bounded risk score."""

    def __init__(
        self,
        model_path: str | Path = ROOT / "models" / "endpoint" / "endpoint_novelty.onnx",
        config_path: str | Path = ROOT / "models" / "endpoint" / "novelty_config.json",
        score_scale: float = 0.03,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        try:
            parsed_scale = float(score_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Endpoint novelty score scale must be numeric") from exc
        self.score_scale = max(parsed_scale, 1e-6)
        self._session: Any | None = None
        self._input_name = ""
        self._score_output = "score_samples"
        self._threshold = 0.5

    def _load(self) -> None:
        if not self.model_path.exists() or not self.config_path.exists():
            logger.warning("Endpoint novelty artifacts missing")
            return
        import onnxruntime as ort

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            threshold = float(config["threshold"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid endpoint novelty configuration") from exc
        session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs = {output.name for output in session.get_outputs()}
        self._threshold = threshold
        self._session = session
        self._input_name = input_name
        if self._score_output not in outputs:
            raise RuntimeError("Endpoint novelty ONNX output score_samples is missing")

    def is_ready(self) -> bool:
        if self._session is None:
            self._load()
        return self._session is not None

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("Endpoint novelty model is unavailable")
        session = self._session
        if session is None:
            return np.full(len(vectors), 0.0, dtype=np.float32)
        values = vectors.astype(np.float32)
        outputs = session.run([self._score_output], {self._input_name: values})[0]
        anomaly_scores = -np.asarray(outputs, dtype=np.float64).reshape(-1)
        logits = np.clip((anomaly_scores - self._threshold) / self.score_scale, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities[~np.isfinite(probabilities)] = 0.0
        return probabilities.astype(np.float32)

    @property
    def threshold(self) -> float:
        if self._session is None:
            self._load()
        return self._threshold
