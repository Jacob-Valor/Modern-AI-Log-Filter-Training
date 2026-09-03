#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Train a Bot-vs-rest flow head with validation-only threshold calibration."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from thresholds import FIXED_FPR_TARGETS, thresholds_at_fixed_fpr
from train_flow_classifier import (
    CAPTURES,
    DEFAULT_ARCHIVE,
    _exclude_seen_features,
    _hashes_for,
    load_capture,
    numeric_frame,
)

from logfilter.models.classifier import SafeMaxAbsScaler
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "flow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-negative-train", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0.0,  # type: ignore[arg-type]
    )
    return {
        "n": len(labels),
        "bot": _as_int(labels.sum(), "Bot count"),
        "rest": _as_int((labels == 0).sum(), "rest count"),
        "tp": _as_int(tp, "TP"),
        "fp": _as_int(fp, "FP"),
        "tn": _as_int(tn, "TN"),
        "fn": _as_int(fn, "FN"),
        "precision": _as_float(precision, "precision"),
        "recall": _as_float(recall, "recall"),
        "f1": _as_float(f1, "F1"),
        "balanced_accuracy": _as_float(
            balanced_accuracy_score(labels, predictions), "balanced accuracy"
        ),
        "auroc": _as_float(roc_auc_score(labels, probabilities), "AUROC"),
        "auprc": _as_float(average_precision_score(labels, probabilities), "AUPRC"),
    }


def _limit_negatives(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if len(negative) > limit:
        negative = rng.choice(negative, size=limit, replace=False)
    selected = np.concatenate((positive, negative))
    rng.shuffle(selected)
    return selected


def export_onnx(model: xgb.XGBClassifier, output: Path) -> None:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    converted = convert_xgboost(
        model,
        initial_types=[("float_input", FloatTensorType([None, len(FLOW_FEATURE_NAMES)]))],
        target_opset=15,
    )
    output.write_bytes(converted.SerializeToString())


def main() -> None:
    args = parse_args()
    if args.max_negative_train <= 0 or not 0 < args.threshold < 1:
        raise SystemExit("Training limit and threshold are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)

    frame = pd.concat([load_capture(args.archive, spec) for spec in CAPTURES], ignore_index=True)
    features = numeric_frame(frame)
    original_attack = frame["_label"].to_numpy(dtype=np.int8)
    captures = frame["_capture"].to_numpy(dtype=str)
    labels = np.logical_and(captures == "bot", original_attack == 1).astype(np.int8)
    splits = frame["_split"].to_numpy(dtype=str)
    train_idx = _limit_negatives(
        np.flatnonzero(splits == "train"), labels, args.max_negative_train, args.seed
    )
    validation_idx = np.flatnonzero(splits == "validation")
    calibration_idx = np.flatnonzero(splits == "calibration")
    test_idx = np.flatnonzero(splits == "test")
    development_hashes = _hashes_for(features, train_idx)
    validation_idx, validation_duplicates = _exclude_seen_features(
        features, validation_idx, development_hashes
    )
    development_hashes.update(_hashes_for(features, validation_idx))
    calibration_idx, calibration_duplicates = _exclude_seen_features(
        features, calibration_idx, development_hashes
    )
    development_hashes.update(_hashes_for(features, calibration_idx))
    test_idx, test_duplicates = _exclude_seen_features(features, test_idx, development_hashes)
    if not labels[calibration_idx].any() or not labels[test_idx].any():
        raise RuntimeError("Bot calibration and test positives are required")

    selection_scale = np.max(np.abs(features[train_idx]), axis=0).astype(np.float64)
    selection_scale[~np.isfinite(selection_scale) | (selection_scale == 0)] = 1.0
    selection_scaler = SafeMaxAbsScaler(scale=selection_scale)
    train_x = selection_scaler.transform(features[train_idx]).astype(np.float32)
    validation_x = selection_scaler.transform(features[validation_idx]).astype(np.float32)

    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=args.seed,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
        early_stopping_rounds=40,
    )
    model.fit(
        train_x,
        labels[train_idx],
        eval_set=[(validation_x, labels[validation_idx])],
        verbose=False,
    )
    best_iteration = _as_int(model.best_iteration, "best iteration")
    validation_probabilities = model.predict_proba(validation_x)[:, 1]

    # Early stopping selects the tree count without touching calibration/test.
    # Refit that fixed model on all development rows so temporal Bot variants
    # observed in validation are represented in the promoted artifact.
    final_development_idx = np.concatenate((train_idx, validation_idx))
    final_scale = np.max(np.abs(features[final_development_idx]), axis=0).astype(np.float64)
    final_scale[~np.isfinite(final_scale) | (final_scale == 0)] = 1.0
    scaler = SafeMaxAbsScaler(scale=final_scale)
    final_model = xgb.XGBClassifier(
        n_estimators=best_iteration + 1,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=args.seed,
        tree_method="hist",
        device="cuda",
        n_jobs=4,
    )
    final_model.fit(
        scaler.transform(features[final_development_idx]).astype(np.float32),
        labels[final_development_idx],
        verbose=False,
    )
    model = final_model
    calibration_x = scaler.transform(features[calibration_idx]).astype(np.float32)
    test_x = scaler.transform(features[test_idx]).astype(np.float32)
    calibration_probabilities = model.predict_proba(calibration_x)[:, 1]
    test_probabilities = model.predict_proba(test_x)[:, 1]
    calibration = thresholds_at_fixed_fpr(
        labels[calibration_idx], calibration_probabilities, FIXED_FPR_TARGETS
    )
    fixed_test: dict[str, Any] = {}
    for target, row in calibration.items():
        threshold = row.get("threshold")
        fixed_test[target] = (
            {"status": row["status"]}
            if threshold is None
            else _metrics(
                labels[test_idx],
                test_probabilities,
                _as_float(threshold, "calibrated threshold"),
            )
        )
    bot_capture_test = test_idx[captures[test_idx] == "bot"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "seed": args.seed,
            "target": "Bot vs rest",
            "labels_excluded_from_features": True,
            "capture_and_split_metadata_excluded_from_features": True,
            "validation_only_threshold_calibration": True,
            "fixed_fpr_targets": list(FIXED_FPR_TARGETS),
            "final_refit_includes_validation": True,
            "final_refit_rows": len(final_development_idx),
            "exact_feature_deduplication": {
                "validation_removed": validation_duplicates,
                "calibration_removed": calibration_duplicates,
                "test_removed": test_duplicates,
            },
        },
        "validation": _metrics(labels[validation_idx], validation_probabilities, args.threshold),
        "calibration": _metrics(labels[calibration_idx], calibration_probabilities, args.threshold),
        "calibration_fixed_fpr": calibration,
        "test": _metrics(labels[test_idx], test_probabilities, args.threshold),
        "test_at_validation_fixed_fpr": fixed_test,
        "bot_capture_test": _metrics(
            labels[bot_capture_test],
            test_probabilities[captures[test_idx] == "bot"],
            args.threshold,
        ),
        "best_iteration": best_iteration,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "bot_classifier.json")
    scaler.to_json(args.output_dir / "bot_scaler.json")
    (args.output_dir / "bot_feature_names.json").write_text(
        json.dumps(list(FLOW_FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "bot_development_feature_hashes.txt").write_text(
        "\n".join(sorted(development_hashes)) + "\n", encoding="ascii"
    )
    (args.output_dir / "bot_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "bot_calibration.json").write_text(
        json.dumps(
            {
                "created_utc": report["created_utc"],
                "validation_only": True,
                "fixed_fpr": calibration,
                "protocol": report["protocol"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    export_onnx(model, args.output_dir / "bot_classifier.onnx")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
