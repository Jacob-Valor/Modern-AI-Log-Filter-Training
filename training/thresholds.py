"""Threshold-sweep helpers for binary classifier evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute binary classification metrics at one probability threshold."""
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and y_prob must contain the same number of samples")
    if labels.size == 0:
        raise ValueError("threshold metrics require at least one sample")

    cutoff = float(threshold)
    predictions = (probabilities >= cutoff).astype(np.int32)
    positives = labels == 1
    negatives = labels == 0
    predicted_positive = predictions == 1
    predicted_negative = predictions == 0

    true_positive = int(np.logical_and(positives, predicted_positive).sum())
    false_positive = int(np.logical_and(negatives, predicted_positive).sum())
    true_negative = int(np.logical_and(negatives, predicted_negative).sum())
    false_negative = int(np.logical_and(positives, predicted_negative).sum())

    return {
        "threshold": round(cutoff, 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "false_positive_rate": round(false_positive / max(false_positive + true_negative, 1), 4),
        "false_negative_rate": round(false_negative / max(false_negative + true_positive, 1), 4),
        "predicted_positive_rate": round(float(predicted_positive.mean()), 4),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Return threshold metrics sorted by ascending threshold."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [threshold_metrics(y_true, y_prob, threshold) for threshold in sorted(thresholds)]


def summarize_threshold_sweep(
    rows: list[dict[str, float | int]],
    min_recall: float = 0.90,
) -> dict[str, Any]:
    """Summarize a sweep with production-oriented candidate thresholds."""
    if not rows:
        raise ValueError("threshold sweep summary requires at least one row")

    best_f1 = max(rows, key=lambda row: (float(row["f1"]), float(row["precision"])))
    recall_candidates = [row for row in rows if float(row["recall"]) >= min_recall]
    high_recall = (
        max(recall_candidates, key=lambda row: (float(row["precision"]), float(row["f1"])))
        if recall_candidates
        else None
    )
    return {
        "best_f1": best_f1,
        "min_recall_target": round(float(min_recall), 4),
        "best_precision_at_min_recall": high_recall,
    }
