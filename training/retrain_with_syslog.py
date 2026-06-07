"""
Retrain Tier-1 XGBoost classifier with combined HDFS + real syslog data.

This script:
1. Loads existing HDFS TraceBench data
2. Loads generated syslog data
3. Creates unified feature vocabulary
4. Retrains XGBoost classifier
5. Exports to ONNX with new scaler/feature names

Usage:
    python training/retrain_with_syslog.py [--syslog-normal N] [--syslog-anomaly N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.classifier import SafeMaxAbsScaler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("retrain")

PREPROCESSED_DIR = ROOT / "HDFS_v3_TraceBench" / "preprocessed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain Tier-1 with combined HDFS + syslog data"
    )
    parser.add_argument(
        "--syslog-normal", type=int, default=5000,
        help="Number of syslog normal samples (default: 5000)"
    )
    parser.add_argument(
        "--syslog-anomaly", type=int, default=1500,
        help="Number of syslog anomaly samples (default: 1500)"
    )
    parser.add_argument(
        "--sample-normal", type=int, default=None,
        help="Subsample N HDFS normal traces (None = use all)"
    )
    parser.add_argument(
        "--sample-failure", type=int, default=None,
        help="Subsample N HDFS failure traces (None = use all)"
    )
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "models",
        help="Directory for model outputs"
    )
    return parser.parse_args()


def load_hdfs_data(
    sample_normal: int | None = None,
    sample_failure: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load HDFS TraceBench data."""
    normal_path = PREPROCESSED_DIR / "normal_trace.csv"
    failure_path = PREPROCESSED_DIR / "failure_trace.csv"

    logger.info("Loading HDFS data from %s", PREPROCESSED_DIR)
    df_normal = pd.read_csv(normal_path, index_col=0)
    df_failure = pd.read_csv(failure_path, index_col=0)

    if sample_normal is not None:
        df_normal = df_normal.sample(
            n=min(sample_normal, len(df_normal)), random_state=42
        )
    if sample_failure is not None:
        df_failure = df_failure.sample(
            n=min(sample_failure, len(df_failure)), random_state=42
        )

    # Combine
    X_hdfs = pd.concat([df_normal, df_failure], ignore_index=True)
    y_hdfs = pd.Series(
        [0] * len(df_normal) + [1] * len(df_failure),
        name="label"
    )

    logger.info(
        "HDFS: %d normal + %d failure = %d total, %d features",
        len(df_normal), len(df_failure), len(X_hdfs), X_hdfs.shape[1]
    )
    return X_hdfs, y_hdfs


def load_syslog_data(
    n_normal: int = 5000,
    n_anomaly: int = 1500,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load or generate syslog data."""
    normal_path = PREPROCESSED_DIR / "real_syslog_normal.csv"
    failure_path = PREPROCESSED_DIR / "real_syslog_failure.csv"

    # Generate if not exists
    if not normal_path.exists() or not failure_path.exists():
        logger.info("Generating syslog training data...")
        from training.generate_syslog_data import generate_dataset
        df, y = generate_dataset(n_normal, n_anomaly)
        df.insert(0, "TaskID", range(len(df)))
        df[normal_path.name.replace(".csv", "")] = ...  # placeholder

        # Save
        normal_mask = y == 0
        failure_mask = y == 1
        df[normal_mask].to_csv(normal_path, index=False)
        df[failure_mask].to_csv(failure_path, index=False)

    logger.info("Loading syslog data...")
    df_normal = pd.read_csv(normal_path, index_col=0)
    df_failure = pd.read_csv(failure_path, index_col=0)

    # Combine
    X_syslog = pd.concat([df_normal, df_failure], ignore_index=True)
    y_syslog = pd.Series(
        [0] * len(df_normal) + [1] * len(df_failure),
        name="label"
    )

    logger.info(
        "Syslog: %d normal + %d failure = %d total, %d features",
        len(df_normal), len(df_failure), len(X_syslog), X_syslog.shape[1]
    )
    return X_syslog, y_syslog


def merge_datasets(
    X_hdfs: pd.DataFrame,
    y_hdfs: pd.Series,
    X_syslog: pd.DataFrame,
    y_syslog: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge HDFS and syslog datasets with unified vocabulary."""
    # Get all feature names
    hdfs_features = set(X_hdfs.columns)
    syslog_features = set(X_syslog.columns)

    # Unified vocabulary: all features from both datasets
    all_features = sorted(hdfs_features | syslog_features)
    logger.info(
        "Unified vocabulary: %d HDFS + %d syslog = %d total features",
        len(hdfs_features), len(syslog_features), len(all_features)
    )

    # Reindex both to unified feature set (fill missing with 0)
    X_hdfs_aligned = X_hdfs.reindex(columns=all_features, fill_value=0.0)
    X_syslog_aligned = X_syslog.reindex(columns=all_features, fill_value=0.0)

    # Combine
    X_combined = pd.concat([X_hdfs_aligned, X_syslog_aligned], ignore_index=True)
    y_combined = pd.concat([y_hdfs, y_syslog], ignore_index=True)

    X = X_combined.values.astype(np.float32)
    y = y_combined.values.astype(np.int32)

    logger.info("Combined: %d samples, %d features", len(y), X.shape[1])
    return X, y, all_features


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> tuple[np.ndarray, ...]:
    """Stratified train/val/test split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, stratify=y_tmp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MaxAbsScaler]:
    """Fit MaxAbsScaler on training data, apply to all splits."""
    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def evaluate(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split: str,
) -> dict:
    """Run evaluation and log results."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)

    logger.info(
        "[%s]  Precision=%.4f  Recall=%.4f  F1=%.4f  ROC-AUC=%.4f",
        split, precision, recall, f1, roc_auc
    )
    logger.info("\n%s", classification_report(
        y, y_pred, target_names=["normal", "failure"]
    ))

    return {
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }


def export_onnx(
    model: xgb.XGBClassifier,
    n_features: int,
    output_path: Path,
) -> None:
    """Export trained XGBoost model to ONNX format."""
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "onnxmltools is required for ONNX export. "
            "Install: pip install onnxmltools skl2onnx"
        ) from exc

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("ONNX model saved to %s", output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    X_hdfs, y_hdfs = load_hdfs_data(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
    )
    X_syslog, y_syslog = load_syslog_data(
        n_normal=args.syslog_normal,
        n_anomaly=args.syslog_anomaly,
    )

    # ── 2. Merge datasets ─────────────────────────────────────────────────────
    X, y, feature_names = merge_datasets(X_hdfs, y_hdfs, X_syslog, y_syslog)

    # ── 3. Split ───────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # ── 4. Compute class weight ────────────────────────────────────────────────
    n_normal = int((y_train == 0).sum())
    n_failure = int((y_train == 1).sum())
    scale_pos_weight = n_normal / max(n_failure, 1)
    logger.info(
        "Class balance — normal: %d  failure: %d  scale_pos_weight: %.2f",
        n_normal, n_failure, scale_pos_weight
    )

    # ── 5. Train XGBoost ───────────────────────────────────────────────────────
    logger.info(
        "Training XGBoost — n_estimators=%d  max_depth=%d  lr=%.3f",
        args.n_estimators, args.max_depth, args.learning_rate
    )
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=50,
    )

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    val_metrics = evaluate(model, X_val_s, y_val, "val")
    test_metrics = evaluate(model, X_test_s, y_test, "test")

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    # Native XGBoost JSON
    model_json_path = args.output_dir / "log_classifier.json"
    model.save_model(str(model_json_path))
    logger.info("XGBoost model saved to %s", model_json_path)

    # ONNX
    onnx_path = args.output_dir / "log_classifier.onnx"
    export_onnx(model, X_train_s.shape[1], onnx_path)

    # Scaler
    scaler_path = args.output_dir / "scaler.json"
    SafeMaxAbsScaler.from_sklearn(scaler).to_json(scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    # Feature names
    feature_names_path = args.output_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2))
    logger.info("Feature names saved to %s", feature_names_path)

    # Metrics
    metrics_path = args.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps({
        "val": val_metrics,
        "test": test_metrics,
        "training_data": {
            "hdfs_normal": int((y_hdfs == 0).sum()),
            "hdfs_failure": int((y_hdfs == 1).sum()),
            "syslog_normal": int((y_syslog == 0).sum()),
            "syslog_failure": int((y_syslog == 1).sum()),
            "total_features": len(feature_names),
        }
    }, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
