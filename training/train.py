"""
Training script — XGBoost log anomaly classifier.

Usage:
    python training/train.py [--sample-normal N] [--sample-failure N]

Outputs:
    models/log_classifier.onnx       (ONNX export for production inference)
    models/log_classifier.json       (XGBoost native format)
    models/scaler.json               (safe MaxAbsScaler parameters)
    models/feature_names.json        (ordered list of feature column names)
    models/training_metrics.json     (precision, recall, F1, ROC-AUC)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.classifier import SafeMaxAbsScaler  # noqa: E402
from training.data_loader import load_traces, scale_features, split_dataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost log anomaly classifier")
    parser.add_argument(
        "--sample-normal",
        type=int,
        default=None,
        help="Subsample N normal traces (default: use all ~226K)",
    )
    parser.add_argument(
        "--sample-failure",
        type=int,
        default=None,
        help="Subsample N failure traces (default: use all ~30K)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300, help="XGBoost n_estimators (default 300)"
    )
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max_depth (default 6)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="XGBoost learning rate (default 0.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory for model outputs",
    )
    return parser.parse_args()


def evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray, split: str) -> dict:
    """Run evaluation and log results."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)

    logger.info(
        "[%s]  Precision=%.4f  Recall=%.4f  F1=%.4f  ROC-AUC=%.4f",
        split,
        precision,
        recall,
        f1,
        roc_auc,
    )
    logger.info("\n%s", classification_report(y, y_pred, target_names=["normal", "failure"]))

    return {
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }


def export_onnx(model: xgb.XGBClassifier, n_features: int, output_path: Path) -> None:
    """Export trained XGBoost model to ONNX format via onnxmltools.

    ONNX export is mandatory for production inference (AGENTS.md: "every trained
    model must export to ONNX"). A missing onnxmltools is a hard failure, not a
    silent fallback — otherwise training would ship a model the API cannot load
    via ONNX Runtime.
    """
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "onnxmltools is required for production ONNX export but is not installed. "
            "Install it (pip install onnxmltools skl2onnx) and re-run training."
        ) from exc

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s", output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading HDFS TraceBench dataset …")
    X, y, feature_names = load_traces(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
    )

    # ── 2. Split ───────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # ── 3. Compute class weight to handle imbalance (normal >> failure) ────────
    n_normal = int((y_train == 0).sum())
    n_failure = int((y_train == 1).sum())
    scale_pos_weight = n_normal / max(n_failure, 1)
    logger.info(
        "Class balance — normal: %d  failure: %d  scale_pos_weight: %.2f",
        n_normal,
        n_failure,
        scale_pos_weight,
    )

    # ── 4. Train ───────────────────────────────────────────────────────────────
    logger.info(
        "Training XGBoost — n_estimators=%d  max_depth=%d  lr=%.3f …",
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
    )
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        use_label_encoder=False,
        tree_method="hist",  # CPU-friendly
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=50,
    )

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    val_metrics = evaluate(model, X_val_s, y_val, "val")
    test_metrics = evaluate(model, X_test_s, y_test, "test")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    # Native XGBoost JSON
    model_json_path = args.output_dir / "log_classifier.json"
    model.save_model(str(model_json_path))
    logger.info("XGBoost model saved to %s", model_json_path)

    # ONNX
    onnx_path = args.output_dir / "log_classifier.onnx"
    export_onnx(model, X_train_s.shape[1], onnx_path)

    # Scaler parameters are JSON so the API never has to unpickle model artifacts.
    scaler_path = args.output_dir / "scaler.json"
    SafeMaxAbsScaler.from_sklearn(scaler).to_json(scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    # Feature names (required to align inference-time feature vectors)
    feature_names_path = args.output_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2))
    logger.info("Feature names saved to %s", feature_names_path)

    # Metrics
    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    try:
        from logfilter.monitoring.model_registry import ModelRegistry

        registry = ModelRegistry()
        run = registry.register_run(
            model_type="tier1",
            artifact_dir=args.output_dir,
            metrics={"val": val_metrics, "test": test_metrics},
            hyperparameters={
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "sample_normal": args.sample_normal,
                "sample_failure": args.sample_failure,
            },
        )
        logger.info("Registered run %s in model registry", run.run_id)
    except Exception as exc:
        logger.warning("Failed to register run in model registry: %s", exc)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
