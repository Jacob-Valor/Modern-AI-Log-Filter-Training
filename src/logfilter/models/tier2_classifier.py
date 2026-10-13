"""Tier-2 transformer classifier for uncertain Tier-1 log scores."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent.parent


def _strict_model_loading() -> bool:
    return os.environ.get("LOGFILTER_MODELS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _probability_config(value: float | str, name: str) -> float:
    """Parse and validate a probability threshold from config or env."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric probability") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return parsed


class Tier2Classifier:
    """Lazy-loading transformer classifier for Tier-1 uncertainty escalation."""

    def __init__(
        self,
        model_dir: Path = ROOT / "models" / "tier2",
        uncertainty_low: float | str = 0.10,
        uncertainty_high: float | str = 0.90,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.uncertainty_low = _probability_config(uncertainty_low, "tier2 uncertainty_low")
        self.uncertainty_high = _probability_config(uncertainty_high, "tier2 uncertainty_high")
        if self.uncertainty_low > self.uncertainty_high:
            raise ValueError("tier2 uncertainty_low must be <= uncertainty_high")
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
        self.strict_model_loading = _strict_model_loading()
        # Sequence length used at inference. Overridden from the artifact at load
        # time when the export recorded one, so training and inference cannot drift.
        self._inference_max_length = 1024

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
            # Propagate instead of returning a fabricated 0.5. A neutral-looking
            # probability is indistinguishable from a real prediction, so the caller
            # recorded it as the classifier score and marked the event tier2_used
            # without ever setting score_degraded - letting an event that was never
            # classified reach the automated-response gate. The caller already
            # handles RuntimeError by keeping the Tier-1 score and marking degraded.
            logger.warning("Tier-2 classifier inference failed", error=str(exc))
            raise RuntimeError("Tier-2 classifier inference failed") from exc

        raise RuntimeError("Tier-2 classifier has no usable inference backend")

    def prewarm(self) -> None:
        # Inference exceptions are deliberately NOT swallowed here (unlike
        # predict_proba), so a strict warmup can reject a model that loads but
        # cannot complete an inference pass.
        if not self.is_ready():
            return
        if self._session is not None:
            self._predict_onnx(["warmup"])
        elif self._torch_model is not None:  # pragma: no cover
            self._predict_torch(["warmup"])

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
            self.model_dir.exists() and tokenizer_present and model_present and label_map_present
        )
        if not ready:
            self._fail_or_warn(
                "Tier-2 classifier artifacts missing",
                model_dir=str(self.model_dir),
                onnx_path=str(self.onnx_path),
                label_map_path=str(self.label_map_path),
            )
        return ready

    def _load(self) -> None:
        self._load_attempted = True
        self._load_label_map()
        self._load_export_metadata()

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            self._fail_or_warn("transformers unavailable for Tier-2 classifier", error=str(exc))
            return

        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception as exc:  # noqa: BLE001
            self._fail_or_warn("Tier-2 tokenizer failed to load", error=str(exc))
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
            self._fail_or_warn("Tier-2 label map failed to load", error=str(exc))

    def _load_export_metadata(self) -> None:
        """Adopt the sequence length recorded at export time, when present.

        ``training/train_transformer.py`` accepts ``--max-length``; recording it in the
        artifact metadata is what stops an export at one length from silently changing
        inference at another.
        """
        for name in ("tier2_config.json", "config.json"):
            candidate = self.model_dir / name
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            value = metadata.get("inference_max_length", metadata.get("max_length"))
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                self._inference_max_length = parsed
                logger.info(
                    "Tier-2 inference max_length loaded from artifact",
                    max_length=parsed,
                    source=str(candidate),
                )
                return

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
            self._fail_or_warn("Tier-2 PyTorch model failed to load", error=str(exc))
            self._torch = None
            self._torch_model = None

    def _max_length(self) -> int:
        """Inference sequence length, recorded at export time when available.

        Training takes ``--max-length`` as a CLI flag that previously only reached
        the metrics JSON, so an export at a different length silently changed
        inference behaviour. Prefer the value recorded beside the artifact.
        """
        return int(self._inference_max_length)

    def _predict_onnx(self, texts: list[str]) -> np.ndarray:
        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:
            raise RuntimeError("Tier-2 ONNX backend is not fully initialised")

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self._max_length(),
            return_tensors="np",
        )
        missing = [name for name in self._onnx_input_names if name not in encoded]
        if missing:
            # Previously the feed was built with ``if name in encoded``, silently
            # dropping any graph-required input the tokenizer did not emit; session.run
            # then failed and the blanket handler turned the whole batch into a
            # fabricated score.
            raise RuntimeError(f"Tier-2 ONNX graph requires missing inputs: {missing}")
        feed = {
            name: np.asarray(encoded[name], dtype=np.int64) for name in self._onnx_input_names
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
            max_length=self._max_length(),
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

    def _fail_or_warn(self, message: str, **kwargs: Any) -> None:
        if self.strict_model_loading:
            raise RuntimeError(message)
        self._warn_degraded(message, **kwargs)
