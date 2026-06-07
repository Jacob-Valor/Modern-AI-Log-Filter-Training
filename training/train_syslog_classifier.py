"""
Train a lightweight syslog-only classifier using 100 syslog features.

This classifier is used for real syslog events where the HDFS-dominated
2255-feature model produces uniform 0.98 scores. The syslog classifier
operates on a focused 100-feature vocabulary that actually discriminates
between normal and attack syslog events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.models.classifier import SafeMaxAbsScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train syslog-only classifier")
    p.add_argument("--n-normal", type=int, default=10000)
    p.add_argument("--n-anomaly", type=int, default=3000)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "models" / "syslog"))
    return p.parse_args()


def generate_syslog_data(
    n_normal: int, n_anomaly: int, seed: int
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Generate synthetic syslog training data using event templates."""
    from generate_syslog_data import generate_dataset

    df, y = generate_dataset(n_normal, n_anomaly, seed=seed)
    feature_names = list(df.columns)
    return df, y, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple:
    """Train XGBoost classifier on syslog features."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        precision_recall_fscore_support,
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Scale
    scaler = SafeMaxAbsScaler(
        np.abs(X_train).max(axis=0).clip(min=1.0).astype(np.float32)
    )
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Class balance
    n_normal = int((y_train == 0).sum())
    n_failure = int((y_train == 1).sum())
    scale_pos_weight = n_normal / max(n_failure, 1)

    print(f"Training: {n_normal} normal, {n_failure} failure, scale_pos_weight={scale_pos_weight:.2f}")
    print(f"Test: {(y_test == 0).sum()} normal, {(y_test == 1).sum()} failure")

    # Train
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["normal", "failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Check discrimination: what does the model predict for all-zeros?
    zeros = np.zeros((1, X.shape[1]), dtype=np.float32)
    zeros_s = scaler.transform(zeros).astype(np.float32)
    zero_proba = model.predict_proba(zeros_s)[0]
    print(f"All-zeros input: P(normal)={zero_proba[0]:.4f}, P(failure)={zero_proba[1]:.4f}")

    return model, scaler, feature_names, {
        "precision": float(precision_recall_fscore_support(y_test, y_pred, average="binary")[0]),
        "recall": float(precision_recall_fscore_support(y_test, y_pred, average="binary")[1]),
        "f1": float(precision_recall_fscore_support(y_test, y_pred, average="binary")[2]),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "train_normal": int((y_train == 0).sum()),
        "train_failure": int((y_train == 1).sum()),
        "test_normal": int((y_test == 0).sum()),
        "test_failure": int((y_test == 1).sum()),
    }


def export_model(
    model,
    scaler: SafeMaxAbsScaler,
    feature_names: list[str],
    output_dir: Path,
    metrics: dict,
) -> None:
    """Export model, scaler, and feature names."""
    import xgboost as xgb

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost native format (fallback)
    model_path = output_dir / "log_classifier_syslog.json"
    model.save_model(str(model_path))
    print(f"Saved XGBoost model: {model_path}")

    # Save scaler (BEFORE onnx in case onnx export fails)
    scaler_path = output_dir / "scaler_syslog.json"
    scaler.to_json(scaler_path)
    print(f"Saved scaler: {scaler_path}")

    # Save feature names
    features_path = output_dir / "feature_names_syslog.json"
    features_path.write_text(json.dumps(feature_names, indent=2))
    print(f"Saved feature names: {features_path}")

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")

    # Export to ONNX
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        onnx_path = output_dir / "log_classifier_syslog.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Saved ONNX model: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed ({e}) — using XGBoost fallback")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=== Generating syslog training data ===")
    df, y, feature_names = generate_syslog_data(args.n_normal, args.n_anomaly, args.seed)
    print(f"Generated: {len(df)} samples, {len(feature_names)} features")
    print(f"Normal: {(y == 0).sum()}, Anomaly: {(y == 1).sum()}")

    X = df.values.astype(np.float32)

    print("\n=== Training syslog classifier ===")
    model, scaler, feature_names, metrics = train_model(X, y, feature_names, args)

    print("\n=== Exporting ===")
    export_model(model, scaler, feature_names, output_dir, metrics)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
