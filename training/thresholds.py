"""Validation-only threshold helpers for binary classifier evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
FIXED_FPR_TARGETS = (0.001, 0.01, 0.05)


def _number(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _integer(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _rounded(value: Any, digits: int = 4) -> float:
    return round(_number(value, "metric"), digits)


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
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")

    cutoff = _number(threshold, "threshold")
    predictions = (probabilities >= cutoff).astype(np.int32)
    positives = labels == 1
    negatives = labels == 0
    predicted_positive = predictions == 1
    predicted_negative = predictions == 0

    true_positive = _integer(np.logical_and(positives, predicted_positive).sum(), "TP")
    false_positive = _integer(np.logical_and(negatives, predicted_positive).sum(), "FP")
    true_negative = _integer(np.logical_and(negatives, predicted_negative).sum(), "TN")
    false_negative = _integer(np.logical_and(positives, predicted_negative).sum(), "FN")
    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative
    actual_negative_count = false_positive + true_negative
    precision = true_positive / predicted_positive_count if predicted_positive_count else 0.0
    recall = true_positive / actual_positive_count if actual_positive_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": _rounded(cutoff),
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "false_positive_rate": _rounded(false_positive / max(actual_negative_count, 1)),
        "false_negative_rate": _rounded(false_negative / max(actual_positive_count, 1)),
        "predicted_positive_rate": _rounded(predicted_positive.mean()),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def benign_thresholds_at_fixed_fpr(
    benign_scores: np.ndarray,
    targets: tuple[float, ...] = FIXED_FPR_TARGETS,
) -> dict[str, dict[str, float | int | str | None]]:
    """Calibrate fixed-FPR thresholds from benign validation scores only."""
    benign = np.asarray(benign_scores, dtype=np.float64).reshape(-1)
    if benign.size == 0 or not np.isfinite(benign).all():
        raise ValueError("benign validation scores must be non-empty and finite")
    if np.any((benign < 0.0) | (benign > 1.0)):
        raise ValueError("benign validation scores must be probabilities")
    benign_count = _integer(benign.size, "benign count")
    output: dict[str, dict[str, float | int | str | None]] = {}
    for raw_target in targets:
        target = _number(raw_target, "fixed-FPR target")
        if not 0.0 < target < 1.0:
            raise ValueError("fixed-FPR targets must be between zero and one")
        key = f"{target:.4g}"
        minimum_benign = _integer(np.ceil(1.0 / target), "minimum benign count")
        if benign_count < minimum_benign:
            output[key] = {
                "status": "insufficient_benign_samples",
                "target_fpr": target,
                "threshold": None,
                "validation_benign": benign_count,
                "minimum_benign_required": minimum_benign,
            }
            continue
        max_false_positives = _integer(np.floor(target * benign_count), "maximum false positives")
        selected: float | None = None
        selected_fp = 0
        for candidate in np.unique(benign):
            false_positives = _integer(
                np.count_nonzero(benign >= candidate), "false-positive count"
            )
            if false_positives <= max_false_positives:
                selected = _number(candidate, "selected threshold")
                selected_fp = false_positives
                break
        if selected is None:
            above_max = _number(np.nextafter(benign.max(), np.inf), "threshold above maximum")
            if above_max <= 1.0:
                selected = above_max
            else:
                output[key] = {
                    "status": "unattainable_due_to_score_ties",
                    "target_fpr": target,
                    "threshold": None,
                    "validation_benign": benign_count,
                    "max_false_positives": max_false_positives,
                }
                continue
        output[key] = {
            "status": "calibrated",
            "target_fpr": target,
            "threshold": selected,
            "validation_fpr": selected_fp / benign_count,
            "validation_benign": benign_count,
            "max_false_positives": max_false_positives,
            "false_positives": selected_fp,
        }
    return output


def thresholds_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    targets: tuple[float, ...] = FIXED_FPR_TARGETS,
) -> dict[str, dict[str, float | int | str | None]]:
    """Select conservative thresholds using validation labels and scores only."""
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if labels.shape[0] != scores.shape[0] or labels.size == 0:
        raise ValueError("validation labels and scores must be non-empty and equally sized")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("validation scores must be finite probabilities")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("validation labels must be binary")
    benign = scores[labels == 0]
    positive = scores[labels == 1]
    if benign.size == 0 or positive.size == 0:
        raise ValueError("fixed-FPR calibration requires benign and positive validation rows")

    benign_count = _integer(benign.size, "benign count")
    positive_count = _integer(positive.size, "positive count")
    output: dict[str, dict[str, float | int | str | None]] = {}
    for raw_target in targets:
        target = _number(raw_target, "fixed-FPR target")
        if not 0.0 < target < 1.0:
            raise ValueError("fixed-FPR targets must be between zero and one")
        key = f"{target:.4g}"
        minimum_benign = _integer(np.ceil(1.0 / target), "minimum benign count")
        if benign_count < minimum_benign:
            output[key] = {
                "status": "insufficient_benign_samples",
                "target_fpr": target,
                "threshold": None,
                "validation_benign": benign_count,
                "validation_positive": positive_count,
                "minimum_benign_required": minimum_benign,
            }
            continue

        max_false_positives = _integer(np.floor(target * benign_count), "maximum false positives")
        candidates = np.unique(benign)
        selected: float | None = None
        selected_fp = 0
        for candidate in candidates:
            false_positives = _integer(
                np.count_nonzero(benign >= candidate), "false-positive count"
            )
            if false_positives <= max_false_positives:
                selected = _number(candidate, "selected threshold")
                selected_fp = false_positives
                break
        if selected is None:
            above_max = _number(np.nextafter(benign.max(), np.inf), "threshold above maximum")
            if above_max > 1.0:
                output[key] = {
                    "status": "unattainable_due_to_score_ties",
                    "target_fpr": target,
                    "threshold": None,
                    "validation_benign": benign_count,
                    "validation_positive": positive_count,
                    "max_false_positives": max_false_positives,
                }
                continue
            selected = above_max
            selected_fp = 0
        true_positives = _integer(np.count_nonzero(positive >= selected), "true-positive count")
        output[key] = {
            "status": "calibrated",
            "target_fpr": target,
            "threshold": selected,
            "validation_fpr": selected_fp / benign_count,
            "validation_recall": true_positives / positive_count,
            "validation_benign": benign_count,
            "validation_positive": positive_count,
            "max_false_positives": max_false_positives,
            "false_positives": selected_fp,
            "true_positives": true_positives,
        }
    return output


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Return threshold metrics sorted by ascending threshold."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [threshold_metrics(y_true, y_prob, threshold) for threshold in sorted(thresholds)]


def _best_f1_row(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Highest-F1 row, breaking ties toward the most conservative threshold."""
    return max(
        rows,
        key=lambda row: (
            _number(row["f1"], "F1"),
            _number(row["precision"], "precision"),
            _number(row["threshold"], "threshold"),
        ),
    )


def _best_precision_row(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Most precise row, breaking ties toward the most conservative threshold."""
    return max(
        rows,
        key=lambda row: (
            _number(row["precision"], "precision"),
            _number(row["f1"], "F1"),
            _number(row["threshold"], "threshold"),
        ),
    )


def summarize_threshold_sweep(
    rows: list[dict[str, float | int]],
    min_recall: float = 0.90,
) -> dict[str, Any]:
    """Summarize a validation sweep with production-oriented candidates.

    Callers must pass a validation sweep; sweeping the held-out test split makes
    the selected threshold a selection maximand rather than a generalisation
    estimate.

    Ties are broken toward the *highest* threshold. Sweep rows are produced in
    ascending threshold order, so without this tie-break a flat region (for
    example every row scoring ``f1 == 1.0``) would resolve to the lowest cutoff
    in the range — a flag-everything operating point that looks optimal.
    """
    if not rows:
        raise ValueError("threshold sweep summary requires at least one row")

    best_f1 = _best_f1_row(rows)
    recall_candidates = [row for row in rows if _number(row["recall"], "recall") >= min_recall]
    high_recall = _best_precision_row(recall_candidates) if recall_candidates else None
    return {
        "best_f1": best_f1,
        "min_recall_target": _rounded(min_recall),
        "best_precision_at_min_recall": high_recall,
        "tie_break": "highest_threshold",
    }
