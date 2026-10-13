"""
Standalone evaluation script.

Loads a saved XGBoost model and scaler parameters, evaluates on the held-out test set,
and prints a detailed classification report.

Usage:
    python training/evaluate.py [--model-dir models/]
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.models.classifier import SafeMaxAbsScaler  # noqa: E402
from training.data_loader import load_traces, split_dataset  # noqa: E402
from training.thresholds import (  # noqa: E402
    FIXED_FPR_TARGETS,
    summarize_threshold_sweep,
    threshold_metrics,
    threshold_sweep,
    thresholds_at_fixed_fpr,
)

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
        help="Directory with log_classifier.json and scaler.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for positive (failure) class",
    )
    parser.add_argument(
        "--threshold-report",
        type=Path,
        default=None,
        help="Optional JSON output path for a threshold sweep report",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="Recall target used when summarizing threshold candidates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load artifacts
    model = xgb.XGBClassifier()
    model.load_model(str(args.model_dir / "log_classifier.json"))

    scaler = SafeMaxAbsScaler.from_json(args.model_dir / "scaler.json")

    # Load data and reproduce the exact same train/val/test split used by train.py
    X, y, _ = load_traces()
    _, X_val, X_test, _, y_val, y_test = split_dataset(X, y)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Thresholds are calibrated on the validation split only; the test split is
    # scored once, at cutoffs chosen without ever looking at it.
    val_prob = model.predict_proba(X_val_s)[:, 1]
    calibration = thresholds_at_fixed_fpr(y_val, val_prob, FIXED_FPR_TARGETS)
    calibrated = [
        (key, row) for key, row in calibration.items() if row.get("threshold") is not None
    ]
    if not calibrated:
        raise RuntimeError("no validation-calibrated Tier-1 threshold could be selected")
    selected_target, selected_threshold = calibrated[0]
    logger.info(
        "Validation-calibrated threshold %.4f at target FPR %s", selected_threshold, selected_target
    )

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(np.int32)

    print(f"\n{'=' * 60}")
    print(f"  Log Classifier Evaluation  (threshold={args.threshold:.2f})")
    print(f"{'=' * 60}")
    print(f"\nTest samples: {len(y_test):,}  |  Failures: {int(y_test.sum()):,}")
    print("\nClassification Report:\n")
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

    selected_test = threshold_metrics(y_test, y_prob, selected_threshold)
    print(f"\nTest metrics at the validation-calibrated threshold {selected_threshold:.4f}:")
    print(
        f"  Precision={selected_test['precision']:.4f}  "
        f"Recall={selected_test['recall']:.4f}  F1={selected_test['f1']:.4f}  "
        f"FPR={selected_test['false_positive_rate']:.4f}"
    )

    # Sweep validation only — sweeping test would make the winner a selection
    # maximand rather than a generalisation estimate.
    sweep = threshold_sweep(y_val, val_prob)
    summary = summarize_threshold_sweep(sweep, min_recall=args.min_recall)
    print("\nThreshold strategy candidates (validation sweep):")
    print(f"  Best F1: {summary['best_f1']}")
    print(
        f"  Best precision at recall>={args.min_recall:.2f}: "
        f"{summary['best_precision_at_min_recall']}"
    )
    if args.threshold_report is not None:
        payload = {
            "model": "tier1_xgboost",
            "split": "val",
            "operating_threshold": round(float(args.threshold), 4),
            "val_fixed_fpr_calibration": calibration,
            "selected_target_fpr": selected_target,
            "selected_threshold": round(float(selected_threshold), 4),
            "test_at_selected_threshold": selected_test,
            "summary": summary,
            "sweep": sweep,
        }
        args.threshold_report.parent.mkdir(parents=True, exist_ok=True)
        args.threshold_report.write_text(json.dumps(payload, indent=2))
        print(f"\nThreshold report saved to {args.threshold_report}")

    try:
        from logfilter.monitoring.model_registry import ModelRegistry

        registry = ModelRegistry()
        run = registry.find_run_by_artifact_dir(args.model_dir)
        if run is not None:
            registry.update_metrics(
                run.run_id,
                {
                    "eval_threshold": round(float(args.threshold), 4),
                    "eval_precision": round(float(precision_score(y_test, y_pred)), 4),
                    "eval_recall": round(float(recall_score(y_test, y_pred)), 4),
                    "eval_f1": round(float(f1_score(y_test, y_pred)), 4),
                    "eval_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                    "eval_fn_rate": round(float(fn / max(fn + tp, 1)), 4),
                },
            )
            print(f"\nUpdated metrics in registry for run {run.run_id}")
    except Exception as exc:
        logger.warning("Failed to update registry metrics: %s", exc)


if __name__ == "__main__":
    main()
