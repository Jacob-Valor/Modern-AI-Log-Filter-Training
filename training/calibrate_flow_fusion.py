#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Calibrate the final general+Bot flow score on an independent calibration partition."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from thresholds import FIXED_FPR_TARGETS
from train_flow_classifier import (
    CAPTURES,
    DEFAULT_ARCHIVE,
    DEFAULT_OUTPUT,
    _balanced_limit,
    _exclude_seen_features,
    _hashes_for,
    load_capture,
    numeric_frame,
)

from logfilter.models.classifier import SafeMaxAbsScaler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read calibration artifact {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Calibration artifact is not an object: {path}")
    return value


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _threshold_for_max_fp(probabilities: np.ndarray, benign: np.ndarray, max_fp: int) -> float:
    benign_scores = probabilities[benign]
    for candidate in np.unique(benign_scores):
        if _as_int(np.count_nonzero(benign_scores >= candidate), "head FP") <= max_fp:
            return _as_float(candidate, "head threshold")
    above_max = _as_float(np.nextafter(benign_scores.max(), np.inf), "head threshold")
    if above_max > 1.0:
        raise RuntimeError("Cannot calibrate head threshold due to score ties at 1.0")
    return above_max


def _model_probabilities(
    model_path: Path,
    scaler_path: Path,
    features: np.ndarray,
) -> np.ndarray:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = SafeMaxAbsScaler.from_json(scaler_path)
    return model.predict_proba(scaler.transform(features).astype(np.float32))[:, 1]


def main() -> None:
    seed = 20260730
    frame = pd.concat([load_capture(DEFAULT_ARCHIVE, spec) for spec in CAPTURES], ignore_index=True)
    features = numeric_frame(frame)
    labels = frame["_label"].to_numpy(dtype=np.int8)
    splits = frame["_split"].to_numpy(dtype=str)
    captures = frame["_capture"].to_numpy(dtype=str)

    general_train = _balanced_limit(np.flatnonzero(splits == "train"), labels, 75_000, seed)
    bot_labels = np.logical_and(frame["_capture"].to_numpy(dtype=str) == "bot", labels == 1).astype(
        np.int8
    )
    rng = np.random.default_rng(seed)
    all_train = np.flatnonzero(splits == "train")
    bot_positive = all_train[bot_labels[all_train] == 1]
    bot_negative = all_train[bot_labels[all_train] == 0]
    if len(bot_negative) > 50_000:
        bot_negative = rng.choice(bot_negative, size=50_000, replace=False)
    bot_train = np.concatenate((bot_positive, bot_negative))

    development_hashes = _hashes_for(features, general_train)
    development_hashes.update(_hashes_for(features, bot_train))
    early_stop = np.flatnonzero(splits == "validation")
    early_stop, _ = _exclude_seen_features(features, early_stop, development_hashes)
    development_hashes.update(_hashes_for(features, early_stop))
    calibration_idx = np.flatnonzero(splits == "calibration")
    calibration_idx, duplicates_removed = _exclude_seen_features(
        features, calibration_idx, development_hashes
    )
    calibration_features = features[calibration_idx]
    calibration_labels = labels[calibration_idx]

    general_probability = _model_probabilities(
        DEFAULT_OUTPUT / "flow_classifier.json",
        DEFAULT_OUTPUT / "scaler.json",
        calibration_features,
    )
    bot_probability = _model_probabilities(
        DEFAULT_OUTPUT / "bot_classifier.json",
        DEFAULT_OUTPUT / "bot_scaler.json",
        calibration_features,
    )
    alert_floor = 0.65
    composite_threshold = 0.5
    benign = calibration_labels == 0
    positive = calibration_labels == 1
    benign_count = _as_int(benign.sum(), "fusion benign count")
    positive_count = _as_int(positive.sum(), "fusion positive count")
    calibration_captures = captures[calibration_idx]
    targets: dict[str, Any] = {}
    for target in FIXED_FPR_TARGETS:
        key = f"{target:.4g}"
        max_total_fp = _as_int(np.floor(target * benign_count), "fusion FP budget")
        candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
        for bot_share in np.linspace(0.0, 1.0, 21):
            bot_fp_budget = _as_int(np.floor(max_total_fp * bot_share), "Bot FP budget")
            general_fp_budget = max_total_fp - bot_fp_budget
            general_threshold = _threshold_for_max_fp(
                general_probability, benign, general_fp_budget
            )
            bot_threshold = _threshold_for_max_fp(bot_probability, benign, bot_fp_budget)
            alerts = np.logical_or(
                general_probability >= general_threshold,
                bot_probability >= bot_threshold,
            )
            false_positives = _as_int(np.logical_and(benign, alerts).sum(), "fusion FP")
            if false_positives > max_total_fp:
                continue
            true_positives = _as_int(np.logical_and(positive, alerts).sum(), "fusion TP")
            family_recalls: dict[str, float] = {}
            for spec in CAPTURES:
                family_positive = np.logical_and(positive, calibration_captures == spec.name)
                family_count = _as_int(family_positive.sum(), "family positive count")
                family_tp = _as_int(np.logical_and(family_positive, alerts).sum(), "family TP")
                family_recalls[spec.name] = family_tp / max(family_count, 1)
            recalls = tuple(family_recalls[spec.name] for spec in CAPTURES)
            objective = (
                min(recalls),
                sum(recalls) / len(recalls),
                true_positives / positive_count,
                -false_positives / benign_count,
            )
            candidates.append(
                (
                    objective,
                    {
                        "status": "calibrated",
                        "target_fpr": target,
                        "composite_threshold": composite_threshold,
                        "general_probability_threshold": general_threshold,
                        "bot_probability_threshold": bot_threshold,
                        "bot_fpr_budget_share": _as_float(bot_share, "Bot budget share"),
                        "alert_floor": alert_floor,
                        "validation_benign": benign_count,
                        "validation_positive": positive_count,
                        "false_positives": false_positives,
                        "true_positives": true_positives,
                        "validation_fpr": false_positives / benign_count,
                        "validation_recall": true_positives / positive_count,
                        "validation_recall_by_capture": family_recalls,
                    },
                )
            )
        if not candidates:
            raise RuntimeError(f"No fused threshold satisfies target FPR {target}")
        targets[key] = max(candidates, key=lambda item: item[0])[1]

    artifact_paths = (
        DEFAULT_OUTPUT / "flow_classifier.onnx",
        DEFAULT_OUTPUT / "flow_classifier.json",
        DEFAULT_OUTPUT / "scaler.json",
        DEFAULT_OUTPUT / "feature_names.json",
        DEFAULT_OUTPUT / "bot_classifier.onnx",
        DEFAULT_OUTPUT / "bot_classifier.json",
        DEFAULT_OUTPUT / "bot_scaler.json",
        DEFAULT_OUTPUT / "bot_feature_names.json",
        DEFAULT_OUTPUT / "bot_calibration.json",
        DEFAULT_OUTPUT / "development_feature_hashes.txt",
        DEFAULT_OUTPUT / "bot_development_feature_hashes.txt",
    )
    report = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_only": True,
        "fused_score": "OR of validation-calibrated general and Bot heads",
        "flow_window_enabled": False,
        "other_fusion_components": False,
        "calibration_rows": len(calibration_idx),
        "exact_development_duplicates_removed": duplicates_removed,
        "source_dataset": {
            "path": str(DEFAULT_ARCHIVE.relative_to(DEFAULT_OUTPUT.parents[1])),
            "sha256": _sha256(DEFAULT_ARCHIVE),
        },
        "capture_cutoffs": {
            spec.name: {
                "train_end": spec.train_end,
                "calibration_start": spec.train_end + (spec.validation_end - spec.train_end) // 2,
                "test_start": spec.validation_end,
            }
            for spec in CAPTURES
        },
        "artifacts": {
            str(path.relative_to(DEFAULT_OUTPUT)): _sha256(path) for path in artifact_paths
        },
        "fixed_fpr": targets,
    }
    output = DEFAULT_OUTPUT / "fusion_calibration.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
