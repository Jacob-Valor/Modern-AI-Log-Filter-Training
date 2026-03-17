"""
Standalone evaluation script.

Loads a saved XGBoost model and scaler, evaluates on the held-out test set,
and prints a detailed classification report.

Usage:
    python training/evaluate.py [--model-dir models/]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.data_loader import load_traces, split_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved log classifier")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory with log_classifier.json and scaler.pkl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for positive (failure) class",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load artifacts
    model = xgb.XGBClassifier()
    model.load_model(str(args.model_dir / "log_classifier.json"))

    with open(args.model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data and reproduce the exact same test split
    X, y, _ = load_traces()
    _, _, X_test, _, _, y_test = split_dataset(X, y)
    X_test_s = scaler.transform(X_test)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(np.int32)

    print(f"\n{'=' * 60}")
    print(f"  Log Classifier Evaluation  (threshold={args.threshold:.2f})")
    print(f"{'=' * 60}")
    print(f"\nTest samples: {len(y_test):,}  |  Failures: {int(y_test.sum()):,}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["normal", "failure"]))
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1:       {f1_score(y_test, y_pred):.4f}")
    print(f"Precision:{precision_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix (TN FP / FN TP):\n{cm}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives  (normal correctly passed): {tn:,}")
    print(f"  False Positives (normal flagged as failure): {fp:,}")
    print(f"  False Negatives (failure missed):           {fn:,}")
    print(f"  True Positives  (failure correctly flagged): {tp:,}")
    print(f"\n  False Negative Rate (missed failures): {fn / max(fn + tp, 1):.4f}")


if __name__ == "__main__":
    main()
