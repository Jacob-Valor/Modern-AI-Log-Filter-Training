"""Tests for classifier threshold strategy reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest  # type: ignore[import-not-found]

from training.thresholds import (
    benign_thresholds_at_fixed_fpr,
    summarize_threshold_sweep,
    threshold_metrics,
    threshold_sweep,
    thresholds_at_fixed_fpr,
)

ROOT = Path(__file__).parent.parent


def test_threshold_metrics_counts_and_rates() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    probabilities = np.array([0.1, 0.8, 0.4, 0.9], dtype=np.float32)

    metrics = threshold_metrics(labels, probabilities, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5


def test_threshold_sweep_summary_selects_candidates() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    probabilities = np.array([0.1, 0.2, 0.7, 0.9], dtype=np.float32)

    rows = threshold_sweep(labels, probabilities, thresholds=(0.3, 0.5, 0.8))
    summary = summarize_threshold_sweep(rows, min_recall=1.0)

    assert [row["threshold"] for row in rows] == [0.3, 0.5, 0.8]
    # 0.3 and 0.5 tie exactly on f1=1.0, tp=2, fp=0. Ties resolve toward the HIGHEST
    # threshold, so 0.5 (the most conservative tied operating point) wins. Resolving
    # to 0.3 would pick the least conservative member of the tied set.
    best_tied = max(row["threshold"] for row in rows if row["f1"] == 1.0)
    assert best_tied == 0.5
    assert summary["best_f1"]["threshold"] == best_tied
    assert summary["best_precision_at_min_recall"]["threshold"] == best_tied


def test_threshold_metrics_validate_shapes() -> None:
    with pytest.raises(ValueError, match="same number"):
        threshold_metrics(np.array([0, 1]), np.array([0.2]), threshold=0.5)

    with pytest.raises(ValueError, match="at least one sample"):
        threshold_metrics(np.array([]), np.array([]), threshold=0.5)


def test_fixed_fpr_thresholds_use_conservative_validation_order_statistics() -> None:
    benign = np.linspace(0.0, 0.999, 1000, dtype=np.float64)
    positive = np.array([0.95, 0.9999], dtype=np.float64)
    labels = np.concatenate((np.zeros(1000, dtype=np.int32), np.ones(2, dtype=np.int32)))
    scores = np.concatenate((benign, positive))

    calibrated = thresholds_at_fixed_fpr(labels, scores)

    assert calibrated["0.001"]["status"] == "calibrated"
    false_positives = [
        calibrated[target]["false_positives"] for target in ("0.001", "0.01", "0.05")
    ]
    assert all(isinstance(value, int) for value in false_positives)
    assert false_positives == [1, 10, 50]


def test_benign_only_fixed_fpr_calibration() -> None:
    scores = np.linspace(0.0, 0.999, 1000)
    calibrated = benign_thresholds_at_fixed_fpr(scores)
    assert calibrated["0.001"]["false_positives"] == 1
    assert calibrated["0.01"]["false_positives"] == 10
    assert calibrated["0.05"]["false_positives"] == 50


def test_fixed_fpr_reports_insufficient_validation_resolution() -> None:
    labels = np.array([0] * 100 + [1], dtype=np.int32)
    scores = np.linspace(0.0, 1.0, 101)

    calibrated = thresholds_at_fixed_fpr(labels, scores)

    assert calibrated["0.001"]["status"] == "insufficient_benign_samples"
    assert calibrated["0.01"]["status"] == "calibrated"


def test_fixed_fpr_rejects_invalid_validation_scores() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        thresholds_at_fixed_fpr(np.array([0, 1]), np.array([np.nan, 0.9]))


def test_threshold_cli_help_commands_import_cleanly() -> None:
    commands = (
        [sys.executable, str(ROOT / "training" / "evaluate.py"), "--help"],
        [sys.executable, str(ROOT / "scripts" / "evaluate_tier2_thresholds.py"), "--help"],
    )

    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
