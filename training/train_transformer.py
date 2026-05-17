"""
Training script — tier-2 transformer log-window classifier.

Usage:
    python training/train_transformer.py --sample-normal N --sample-failure N --epochs 2 \
        --output-dir models/tier2/

Outputs:
    models/tier2/                         (HuggingFace model/tokenizer directory)
    models/tier2/log_classifier_tier2.onnx (ONNX export for production inference)
    models/tier2/tier2_metrics.json        (precision, recall, F1, ROC-AUC)
    models/tier2/tier2_label_map.json      (label id → label name)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from training.data_loader import split_dataset  # noqa: E402
from training.text_dataset import build_windows  # noqa: E402
from training.thresholds import summarize_threshold_sweep, threshold_sweep  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("train_transformer")

LABEL_MAP = {"0": "normal", "1": "failure"}
DEFAULT_MODEL_ID = "cisco-ai/SecureBERT2.0-base"
# Override option for experiments: answerdotai/ModernBERT-base


@runtime_checkable
class TrainTokenizer(Protocol):
    """Tokenizer capabilities required by the training/export pipeline."""

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool,
        padding: str | bool,
        max_length: int,
    ) -> dict[str, Any]: ...

    def save_pretrained(self, save_directory: str | Path) -> Any: ...


@dataclass(frozen=True)
class TrainerConfig:
    """Serializable subset of CLI and derived training settings."""

    model_id: str
    sample_normal: int | None
    sample_failure: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    output_dir: str
    seed: int
    fp16: bool
    eval_steps: int
    save_steps: int
    class_weights: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tier-2 transformer log classifier")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id (default {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--sample-normal",
        type=int,
        default=None,
        help="Subsample N normal text windows (default: use all ~226K)",
    )
    parser.add_argument(
        "--sample-failure",
        type=int,
        default=None,
        help="Subsample N failure text windows (default: use all ~30K)",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs (default 2)")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per-device batch size (default 16)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate (default 2e-5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Tokenizer max length (default 1024)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "tier2",
        help="Directory for tier-2 model outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable FP16 training (default: enabled when CUDA is available)",
    )
    return parser.parse_args()


def build_hf_dataset(
    texts: np.ndarray,
    labels: np.ndarray,
    dataset_cls: Any,
    tokenizer: TrainTokenizer,
    max_length: int,
) -> Any:
    dataset = dataset_cls.from_dict(
        {"text": texts.tolist(), "labels": labels.astype(int).tolist()}
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def require_train_tokenizer(candidate: object) -> TrainTokenizer:
    """Validate that a dynamically loaded tokenizer supports training outputs."""
    if not isinstance(candidate, TrainTokenizer):
        raise TypeError("Loaded tokenizer must support tokenization and save_pretrained()")
    return candidate


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float32)
    total = float(counts.sum())
    weights = total / (len(counts) * np.maximum(counts, 1.0))
    logger.info(
        "Class balance — normal: %d  failure: %d  weights: [%.2f, %.2f]",
        int(counts[0]),
        int(counts[1]),
        float(weights[0]),
        float(weights[1]),
    )
    return weights.astype(np.float32)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    y_prob = (exp_logits / exp_logits.sum(axis=-1, keepdims=True))[:, 1]
    y_pred = np.argmax(logits, axis=-1)

    return {
        "precision": round(float(precision_score(labels, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, y_prob)), 4),
    }


def evaluate_split(trainer: Any, dataset: Any, split: str) -> dict[str, Any]:
    predictions = trainer.predict(test_dataset=dataset, metric_key_prefix=split)
    metrics = predictions.metrics
    logits = np.asarray(predictions.predictions, dtype=np.float32)
    labels = np.asarray(predictions.label_ids, dtype=np.int32)
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    y_prob = (exp_logits / exp_logits.sum(axis=-1, keepdims=True))[:, 1]
    sweep = threshold_sweep(labels, y_prob)
    result = {
        "split": split,
        "precision": round(float(metrics[f"{split}_precision"]), 4),
        "recall": round(float(metrics[f"{split}_recall"]), 4),
        "f1": round(float(metrics[f"{split}_f1"]), 4),
        "roc_auc": round(float(metrics[f"{split}_roc_auc"]), 4),
        "threshold_strategy": {
            "summary": summarize_threshold_sweep(sweep),
            "sweep": sweep,
        },
    }
    logger.info(
        "[%s]  Precision=%.4f  Recall=%.4f  F1=%.4f  ROC-AUC=%.4f",
        split,
        result["precision"],
        result["recall"],
        result["f1"],
        result["roc_auc"],
    )
    return result


def count_parameters(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def compute_eval_steps(n_train: int, batch_size: int, epochs: int, n_devices: int) -> int:
    steps_per_epoch = math.ceil(n_train / max(batch_size * max(n_devices, 1), 1))
    num_train_steps = max(steps_per_epoch * max(epochs, 1), 1)
    return max(1, math.ceil(num_train_steps / 8))


def export_onnx(model_dir: Path, output_path: Path) -> None:
    """Export via Optimum so the ONNX graph matches the saved HF checkpoint."""
    try:
        optimum_ort = importlib.import_module("optimum.onnxruntime")
        ort_model_cls = optimum_ort.ORTModelForSequenceClassification

        with tempfile.TemporaryDirectory() as tmpdir:
            ort_model = ort_model_cls.from_pretrained(model_dir, export=True)
            ort_model.save_pretrained(tmpdir)
            source = Path(tmpdir) / "model.onnx"
            output_path.write_bytes(source.read_bytes())
        logger.info("ONNX model saved to %s", output_path)
    except ImportError:
        logger.warning(
            "optimum[onnxruntime] not installed — skipping ONNX export. "
            "Install with: pip install optimum[onnxruntime] onnx onnxruntime"
        )


def make_weighted_loss_trainer(trainer_cls: Any, nn_module: Any) -> type[Any]:
    """Create a Trainer subclass after Kaggle-only dependencies are imported."""

    class WeightedLossTrainer(trainer_cls):
        """Trainer with class-weighted cross entropy for the ~7:1 normal/failure skew."""

        def __init__(self, *args: Any, class_weights: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = self.class_weights.to(logits.device)
            loss_fct = nn_module.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    transformers = importlib.import_module("transformers")
    nn_module = importlib.import_module("torch.nn")

    dataset_cls = datasets.Dataset
    auto_tokenizer = transformers.AutoTokenizer
    auto_model = transformers.AutoModelForSequenceClassification
    early_stopping_callback = transformers.EarlyStoppingCallback
    training_args_cls = transformers.TrainingArguments
    set_seed = transformers.set_seed
    weighted_trainer_cls = make_weighted_loss_trainer(transformers.Trainer, nn_module)

    fp16 = torch.cuda.is_available() if args.fp16 is None else bool(args.fp16)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Loading HDFS TraceBench text windows …")
    texts, labels, window_stats = build_windows(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
        random_state=args.seed,
    )

    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=np.int64)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, random_state=args.seed
    )

    logger.info("Loading tokenizer/model: %s", args.model_id)
    tokenizer = require_train_tokenizer(auto_tokenizer.from_pretrained(args.model_id))
    model = auto_model.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label={0: "normal", 1: "failure"},
        label2id={"normal": 0, "failure": 1},
    )
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    num_params = count_parameters(model)
    logger.info("Model parameters: %d", num_params)

    train_dataset = build_hf_dataset(
        X_train, y_train, dataset_cls, tokenizer, args.max_length
    )
    val_dataset = build_hf_dataset(X_val, y_val, dataset_cls, tokenizer, args.max_length)
    test_dataset = build_hf_dataset(
        X_test, y_test, dataset_cls, tokenizer, args.max_length
    )

    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    eval_steps = compute_eval_steps(len(y_train), args.batch_size, args.epochs, n_devices)
    class_weights_np = compute_class_weights(y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    config = TrainerConfig(
        model_id=args.model_id,
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=str(args.output_dir),
        seed=args.seed,
        fp16=fp16,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        class_weights=[round(float(v), 6) for v in class_weights_np.tolist()],
    )

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "report_to": "none",
        "logging_steps": max(1, eval_steps // 2),
        "dataloader_num_workers": 2,
        "fp16": fp16,
        "seed": args.seed,
    }
    if "save_safetensors" in inspect.signature(training_args_cls).parameters:
        training_kwargs["save_safetensors"] = True
    training_args = training_args_cls(**training_kwargs)

    trainer = weighted_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    logger.info(
        "Training transformer — epochs=%d  batch_size=%d  lr=%.6f  eval_steps=%d  fp16=%s …",
        args.epochs,
        args.batch_size,
        args.learning_rate,
        eval_steps,
        fp16,
    )
    trainer.train()

    val_metrics = evaluate_split(trainer, val_dataset, "val")
    test_metrics = evaluate_split(trainer, test_dataset, "test")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    logger.info("HuggingFace model saved to %s", args.output_dir)

    onnx_path = args.output_dir / "log_classifier_tier2.onnx"
    export_onnx(args.output_dir, onnx_path)

    label_map_path = args.output_dir / "tier2_label_map.json"
    label_map_path.write_text(json.dumps(LABEL_MAP, indent=2))
    logger.info("Label map saved to %s", label_map_path)

    metrics_path = args.output_dir / "tier2_metrics.json"
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "model_id": args.model_id,
        "num_params": num_params,
        "training_args": asdict(config),
        "window_stats": window_stats,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
