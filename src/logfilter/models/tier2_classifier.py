"""Tier-2 transformer classifier for uncertain Tier-1 log scores."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent


class Tier2Classifier:
    """Lazy-loading transformer classifier for Tier-1 uncertainty escalation."""

    def __init__(
        self,
        model_dir: Path = ROOT / "models" / "tier2",
        uncertainty_low: float = 0.10,
        uncertainty_high: float = 0.90,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.uncertainty_low = float(uncertainty_low)
        self.uncertainty_high = float(uncertainty_high)
        self.onnx_path = self.model_dir / "log_classifier_tier2.onnx"
        self.label_map_path = self.model_dir / "tier2_label_map.json"

        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._torch_model: Any | None = None
        self._torch: Any | None = None
        self._onnx_input_names: list[str] = []
        self._failure_label_index = 1
        self._load_attempted = False
        self._warned_unavailable = False

    def is_ready(self) -> bool:
        """Return True when artifacts exist and a tokenizer/model backend is loadable."""
        if not self._artifacts_present():
            return False
        if not self._load_attempted:
            self._load()
        return self._tokenizer is not None and (
            self._session is not None or self._torch_model is not None
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict failure probabilities for raw log texts."""
        if not texts:
            return np.array([], dtype=np.float32)

        if not self.is_ready():
            self._warn_degraded("Tier-2 classifier unavailable; returning neutral scores")
            return np.full(len(texts), 0.5, dtype=np.float32)

        try:
            if self._session is not None:
                return self._predict_onnx(texts)
            if self._torch_model is not None:  # pragma: no cover
                return self._predict_torch(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier-2 classifier inference failed", error=str(exc))

        return np.full(len(texts), 0.5, dtype=np.float32)

    def should_escalate(self, tier1_prob: float) -> bool:
        """Return True when Tier-1 probability is inside the uncertainty band."""
        return self.uncertainty_low <= float(tier1_prob) <= self.uncertainty_high

    def _artifacts_present(self) -> bool:
        tokenizer_present = any(
            (self.model_dir / name).exists()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "vocab.json",
                "spiece.model",
            )
        )
        model_present = self.onnx_path.exists() or any(
            (self.model_dir / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "config.json")
        )
        label_map_present = self.label_map_path.exists()
        ready = (
            self.model_dir.exists()
            and tokenizer_present
            and model_present
            and label_map_present
        )
        if not ready:
            self._warn_degraded(
                "Tier-2 classifier artifacts missing",
                model_dir=str(self.model_dir),
                onnx_path=str(self.onnx_path),
                label_map_path=str(self.label_map_path),
            )
        return ready

    def _load(self) -> None:
        self._load_attempted = True
        self._load_label_map()

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            self._warn_degraded("transformers unavailable for Tier-2 classifier", error=str(exc))
            return

        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception as exc:  # noqa: BLE001
            self._warn_degraded("Tier-2 tokenizer failed to load", error=str(exc))
            return

        if self.onnx_path.exists() and self._load_onnx():
            return

        self._load_torch(transformers)

    def _load_label_map(self) -> None:
        try:
            label_map = json.loads(self.label_map_path.read_text())
            for key, value in label_map.items():
                if str(value).lower() == "failure":
                    self._failure_label_index = int(key)
                    return
        except Exception as exc:  # noqa: BLE001
            self._warn_degraded("Tier-2 label map failed to load", error=str(exc))

    def _load_onnx(self) -> bool:  # pragma: no cover
        try:
            onnxruntime = importlib.import_module("onnxruntime")
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 4
            self._session = onnxruntime.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_names = [
                model_input.name for model_input in self._session.get_inputs()
            ]
            logger.info("Tier-2 ONNX classifier loaded", path=str(self.onnx_path))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier-2 ONNX load failed; falling back to PyTorch", error=str(exc))
            self._session = None
            self._onnx_input_names = []
            return False

    def _load_torch(self, transformers: Any) -> None:  # pragma: no cover
        try:
            self._torch = importlib.import_module("torch")
            self._torch_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            )
            self._torch_model.eval()
            logger.info("Tier-2 PyTorch classifier loaded", path=str(self.model_dir))
        except Exception as exc:  # noqa: BLE001
            self._warn_degraded("Tier-2 PyTorch model failed to load", error=str(exc))
            self._torch = None
            self._torch_model = None

    def _predict_onnx(self, texts: list[str]) -> np.ndarray:
        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:
            return np.full(len(texts), 0.5, dtype=np.float32)

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="np",
        )
        feed = {
            name: np.asarray(encoded[name], dtype=np.int64)
            for name in self._onnx_input_names
            if name in encoded
        }
        outputs = session.run(None, feed)
        logits = np.asarray(outputs[0], dtype=np.float32)
        return self._failure_probs_from_logits(logits)

    def _predict_torch(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        tokenizer = self._tokenizer
        torch = self._torch
        torch_model = self._torch_model
        if tokenizer is None or torch is None or torch_model is None:
            return np.full(len(texts), 0.5, dtype=np.float32)

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = torch_model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()
        return self._failure_probs_from_logits(np.asarray(logits, dtype=np.float32))

    def _failure_probs_from_logits(self, logits: np.ndarray) -> np.ndarray:
        if logits.ndim != 2 or logits.shape[1] <= self._failure_label_index:
            logger.warning("Tier-2 logits had unexpected shape", shape=list(logits.shape))
            return np.full(logits.shape[0] if logits.ndim else 1, 0.5, dtype=np.float32)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.clip(probs[:, self._failure_label_index], 0.0, 1.0).astype(np.float32)

    def _warn_degraded(self, message: str, **kwargs: Any) -> None:
        if self._warned_unavailable:
            return
        logger.warning(message, **kwargs)
        self._warned_unavailable = True
