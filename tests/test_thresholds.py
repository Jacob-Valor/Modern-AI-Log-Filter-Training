"""Tests for classifier threshold strategy reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.thresholds import summarize_threshold_sweep, threshold_metrics, threshold_sweep

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
    assert summary["best_f1"]["threshold"] == 0.3
    assert summary["best_precision_at_min_recall"]["threshold"] == 0.3


def test_threshold_metrics_validate_shapes() -> None:
    with pytest.raises(ValueError, match="same number"):
        threshold_metrics(np.array([0, 1]), np.array([0.2]), threshold=0.5)

    with pytest.raises(ValueError, match="at least one sample"):
        threshold_metrics(np.array([]), np.array([]), threshold=0.5)


def test_threshold_cli_help_commands_import_cleanly() -> None:
    commands = (
        [sys.executable, str(ROOT / "training" / "evaluate.py"), "--help"],
        [sys.executable, str(ROOT / "scripts" / "evaluate_tier2_thresholds.py"), "--help"],
    )

    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
