#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Train a leakage-safe, family-aware CICFlowMeter classifier.

Bot, Web Attack, and DDoS captures use immutable contiguous row cutoffs. Labels,
attack names, file identity, IPs, and timestamps are split-only metadata and are
never model features. Thresholds are calibrated on validation rows only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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

from logfilter.models.classifier import SafeMaxAbsScaler
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "demo_data" / "CIC-IDS2017" / "MachineLearningCSV.zip"
DEFAULT_OUTPUT = ROOT / "models" / "flow"
LABEL = "Label"
SOURCE_COLUMNS: dict[str, str] = {
    "destination_port": "Destination Port",
    "flow_duration_us": "Flow Duration",
    "forward_packets": "Total Fwd Packets",
    "backward_packets": "Total Backward Packets",
    "forward_bytes": "Total Length of Fwd Packets",
    "backward_bytes": "Total Length of Bwd Packets",
    "flow_bytes_per_second": "Flow Bytes/s",
    "flow_packets_per_second": "Flow Packets/s",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std",
    "syn_count": "SYN Flag Count",
    "ack_count": "ACK Flag Count",
    "rst_count": "RST Flag Count",
    "active_mean_us": "Active Mean",
    "idle_mean_us": "Idle Mean",
    "fin_count": "FIN Flag Count",
    "psh_count": "PSH Flag Count",
    "urg_count": "URG Flag Count",
    "flow_iat_mean_us": "Flow IAT Mean",
    "flow_iat_std_us": "Flow IAT Std",
    "flow_iat_max_us": "Flow IAT Max",
    "flow_iat_min_us": "Flow IAT Min",
    "forward_iat_mean_us": "Fwd IAT Mean",
    "forward_iat_std_us": "Fwd IAT Std",
    "backward_iat_mean_us": "Bwd IAT Mean",
    "backward_iat_std_us": "Bwd IAT Std",
    "packet_length_min": "Min Packet Length",
    "packet_length_max": "Max Packet Length",
    "forward_packet_length_mean": "Fwd Packet Length Mean",
    "backward_packet_length_mean": "Bwd Packet Length Mean",
    "initial_window_forward_bytes": "Init_Win_bytes_forward",
    "initial_window_backward_bytes": "Init_Win_bytes_backward",
    "down_up_ratio": "Down/Up Ratio",
}


@dataclass(frozen=True)
class CaptureSpec:
    name: str
    member_suffix: str
    attack_prefixes: tuple[str, ...]
    train_end: int
    validation_end: int


CAPTURES = (
    CaptureSpec(
        name="ddos",
        member_suffix="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        attack_prefixes=("DDoS",),
        train_end=150_000,
        validation_end=175_000,
    ),
    CaptureSpec(
        name="bot",
        member_suffix="Friday-WorkingHours-Morning.pcap_ISCX.csv",
        attack_prefixes=("Bot",),
        train_end=114_619,
        validation_end=152_826,
    ),
    CaptureSpec(
        name="web",
        member_suffix="Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        attack_prefixes=("Web Attack",),
        train_end=50_000,
        validation_end=70_000,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-train-per-class", type=int, default=75_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix}, found {len(matches)}")
    member = matches[0]
    member_path = PurePosixPath(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe ZIP member: {member}")
    return member


def _matches_attack(label: str, spec: CaptureSpec) -> bool:
    return any(label.startswith(prefix) for prefix in spec.attack_prefixes)


def load_capture(archive_path: Path, spec: CaptureSpec) -> pd.DataFrame:
    required = set(SOURCE_COLUMNS.values()) | {LABEL}
    with zipfile.ZipFile(archive_path) as archive:
        member = _member(archive, spec.member_suffix)
        with archive.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
            frame = pd.read_csv(text, usecols=lambda name: name.strip() in required)
    frame.columns = [column.strip() for column in frame.columns]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CIC columns in {spec.name}: {sorted(missing)}")
    labels = frame[LABEL].astype(str).str.strip()
    selected = (labels == "BENIGN") | labels.map(lambda label: _matches_attack(label, spec))
    frame = frame.loc[selected].copy()
    frame["_label"] = labels.loc[selected].map(lambda label: 0 if label == "BENIGN" else 1)
    frame["_capture"] = spec.name
    original_rows = frame.index.to_numpy(dtype=np.int64)
    frame["_row"] = original_rows
    calibration_start = spec.train_end + (spec.validation_end - spec.train_end) // 2
    frame["_split"] = np.select(
        (
            original_rows < spec.train_end,
            original_rows < calibration_start,
            original_rows < spec.validation_end,
        ),
        ("train", "validation", "calibration"),
        default="test",
    )
    return frame


def numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    converted = pd.DataFrame(index=frame.index)
    for feature_name in FLOW_FEATURE_NAMES:
        converted[feature_name] = pd.to_numeric(
            frame[SOURCE_COLUMNS[feature_name]], errors="coerce"
        )
    converted = converted.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return converted.loc[:, list(FLOW_FEATURE_NAMES)].to_numpy(dtype=np.float32)


def _feature_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()


def _exclude_seen_features(
    features: np.ndarray,
    indices: np.ndarray,
    seen_hashes: set[str],
) -> tuple[np.ndarray, int]:
    retained: list[int] = []
    removed = 0
    for index in indices:
        feature_hash = _feature_hash(features[index])
        if feature_hash in seen_hashes:
            removed += 1
        else:
            retained.append(_as_int(index, "feature row index"))
    return np.asarray(retained, dtype=np.int64), removed


def _hashes_for(features: np.ndarray, indices: np.ndarray) -> set[str]:
    return {_feature_hash(features[index]) for index in indices}


def _balanced_limit(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for value in (0, 1):
        candidates = indices[labels[indices] == value]
        if len(candidates) > limit:
            candidates = rng.choice(candidates, size=limit, replace=False)
        selected.append(candidates)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not an integer") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} is not finite")
    return parsed


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
    )
    return {
        "n": len(labels),
        "attack": _as_int(labels.sum(), "attack count"),
        "benign": _as_int((labels == 0).sum(), "benign count"),
        "tp": _as_int(tp, "true positives"),
        "fp": _as_int(fp, "false positives"),
        "tn": _as_int(tn, "true negatives"),
        "fn": _as_int(fn, "false negatives"),
        "precision": _as_float(precision, "precision"),
        "recall": _as_float(recall, "recall"),
        "f1": _as_float(f1, "f1"),
        "balanced_accuracy": _as_float(
            balanced_accuracy_score(labels, predictions), "balanced accuracy"
        ),
        "auroc": _as_float(roc_auc_score(labels, probabilities), "AUROC"),
        "auprc": _as_float(average_precision_score(labels, probabilities), "AUPRC"),
    }


def _fixed_fpr_test_metrics(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = thresholds_at_fixed_fpr(
        validation_labels, validation_probabilities, FIXED_FPR_TARGETS
    )
    test_results: dict[str, Any] = {}
    for target, row in calibration.items():
        threshold = row.get("threshold")
        test_results[target] = (
            {"status": row["status"]}
            if threshold is None
            else _metrics(
                test_labels,
                test_probabilities,
                _as_float(threshold, "calibrated threshold"),
            )
        )
    return calibration, test_results


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
    if args.max_train_per_class <= 0:
        raise SystemExit("Training limit must be positive")
    if not 0 < args.threshold < 1:
        raise SystemExit("Threshold must be between zero and one")
    random.seed(args.seed)
    np.random.seed(args.seed)

    frames = [load_capture(args.archive, spec) for spec in CAPTURES]
    frame = pd.concat(frames, ignore_index=True)
    features = numeric_frame(frame)
    labels = frame["_label"].to_numpy(dtype=np.int8)
    splits = frame["_split"].to_numpy(dtype=str)
    captures = frame["_capture"].to_numpy(dtype=str)
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "validation")
    calibration_idx = np.flatnonzero(splits == "calibration")
    test_idx = np.flatnonzero(splits == "test")
    train_idx = _balanced_limit(train_idx, labels, args.max_train_per_class, args.seed)
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
    for split_name, indices in (
        ("validation", validation_idx),
        ("calibration", calibration_idx),
        ("test", test_idx),
    ):
        if len(np.unique(labels[indices])) != 2:
            raise RuntimeError(f"{split_name} partition must contain both classes")

    scale = np.max(np.abs(features[train_idx]), axis=0).astype(np.float64)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    scaler = SafeMaxAbsScaler(scale=scale)
    train_x = scaler.transform(features[train_idx]).astype(np.float32)
    validation_x = scaler.transform(features[validation_idx]).astype(np.float32)
    calibration_x = scaler.transform(features[calibration_idx]).astype(np.float32)
    test_x = scaler.transform(features[test_idx]).astype(np.float32)

    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
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

    validation_probabilities = model.predict_proba(validation_x)[:, 1]
    calibration_probabilities = model.predict_proba(calibration_x)[:, 1]
    test_probabilities = model.predict_proba(test_x)[:, 1]
    calibration, fixed_fpr_test = _fixed_fpr_test_metrics(
        labels[calibration_idx],
        calibration_probabilities,
        labels[test_idx],
        test_probabilities,
    )
    family_test: dict[str, Any] = {}
    for spec in CAPTURES:
        mask = captures[test_idx] == spec.name
        family_test[spec.name] = _metrics(
            labels[test_idx][mask], test_probabilities[mask], args.threshold
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "archive": str(args.archive),
            "sha256": _sha256(args.archive),
            "members": [spec.member_suffix for spec in CAPTURES],
        },
        "protocol": {
            "seed": args.seed,
            "capture_cutoffs": {
                spec.name: {
                    "train_rows": [0, spec.train_end - 1],
                    "early_stopping_rows": [
                        spec.train_end,
                        spec.train_end + (spec.validation_end - spec.train_end) // 2 - 1,
                    ],
                    "calibration_rows": [
                        spec.train_end + (spec.validation_end - spec.train_end) // 2,
                        spec.validation_end - 1,
                    ],
                    "test_min_row_index": spec.validation_end,
                }
                for spec in CAPTURES
            },
            "threshold": args.threshold,
            "fixed_fpr_targets": list(FIXED_FPR_TARGETS),
            "label_features_excluded": True,
            "ip_timestamp_filename_features_excluded": True,
            "test_threshold_tuning": False,
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
        "test_at_validation_fixed_fpr": fixed_fpr_test,
        "test_by_capture": family_test,
        "best_iteration": _as_int(model.best_iteration, "best iteration"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "flow_classifier.json")
    scaler.to_json(args.output_dir / "scaler.json")
    (args.output_dir / "feature_names.json").write_text(
        json.dumps(list(FLOW_FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "development_feature_hashes.txt").write_text(
        "\n".join(sorted(development_hashes)) + "\n", encoding="ascii"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibration.json").write_text(
        json.dumps(
            {
                "created_utc": report["created_utc"],
                "feature_names": list(FLOW_FEATURE_NAMES),
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
    export_onnx(model, args.output_dir / "flow_classifier.onnx")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
