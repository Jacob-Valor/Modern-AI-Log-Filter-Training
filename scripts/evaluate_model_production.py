"""Evaluate LogFilter scores against representative labeled production logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.config import load_config  # noqa: E402
from logfilter.pipeline.normalizer import LogNormalizer  # noqa: E402
from logfilter.pipeline.scorer import LogScorer  # noqa: E402
from training.thresholds import summarize_threshold_sweep, threshold_sweep  # noqa: E402


@dataclass(frozen=True)
class LabeledLog:
    raw: str
    label: int
    score: float | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LogFilter threat scores on representative labeled logs. "
            "Input must be CSV or JSONL with raw,label and optionally a score column."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV/JSONL labeled log dataset")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--raw-column", default="raw")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--score-column", default="", help="Use precomputed score column if present"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def load_labeled_dataset(
    path: Path,
    *,
    raw_column: str = "raw",
    label_column: str = "label",
    score_column: str = "",
) -> list[LabeledLog]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".csv":
        rows = list(csv.DictReader(path.read_text().splitlines()))
    else:
        raise ValueError("input dataset must be .csv or .jsonl")

    labeled: list[LabeledLog] = []
    for index, row in enumerate(rows, start=1):
        try:
            raw = str(row[raw_column])
            label = int(row[label_column])
            score = float(row[score_column]) if score_column else None
        except KeyError as exc:
            raise ValueError(f"missing required column {exc.args[0]!r} in row {index}") from exc
        if label not in {0, 1}:
            raise ValueError(f"label must be 0 or 1 in row {index}")
        labeled.append(LabeledLog(raw=raw, label=label, score=score))
    if not labeled:
        raise ValueError("dataset contains no labeled rows")
    return labeled


def score_rows(rows: list[LabeledLog], *, config_path: Path, batch_size: int) -> list[LabeledLog]:
    config = load_config(config_path)
    normalizer = LogNormalizer()
    scorer = LogScorer(config=config)
    scorer.preload_models()
    scored_rows: list[LabeledLog] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        normalized = [normalizer.normalize(row.raw) for row in batch]
        scored = scorer.score_batch(normalized)
        for row, scored_event in zip(batch, scored, strict=True):
            scored_rows.append(
                LabeledLog(raw=row.raw, label=row.label, score=float(scored_event.ai_threat_score))
            )
    return scored_rows


def build_report(
    rows: list[LabeledLog],
    *,
    threshold: float = 0.5,
    min_recall: float = 0.90,
) -> dict[str, Any]:
    labels = np.asarray([row.label for row in rows], dtype=np.int32)
    scores = np.asarray([_require_score(row) for row in rows], dtype=np.float32)
    predictions = (scores >= threshold).astype(np.int32)
    positives = labels == 1
    negatives = labels == 0
    predicted_positive = predictions == 1
    predicted_negative = predictions == 0

    tp = int(np.logical_and(positives, predicted_positive).sum())
    fp = int(np.logical_and(negatives, predicted_positive).sum())
    tn = int(np.logical_and(negatives, predicted_negative).sum())
    fn = int(np.logical_and(positives, predicted_negative).sum())
    roc_auc = _roc_auc_or_none(labels, scores)
    sweep = threshold_sweep(labels, scores)
    return {
        "samples": int(labels.size),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "fp_budget": {
            "false_positives": fp,
            "benign_samples": int(negatives.sum()),
            "false_positives_per_1000_benign": round(fp / max(int(negatives.sum()), 1) * 1000, 4),
        },
        "threshold_summary": summarize_threshold_sweep(sweep, min_recall=min_recall),
        "threshold_sweep": sweep,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_labeled_dataset(
        args.input,
        raw_column=args.raw_column,
        label_column=args.label_column,
        score_column=args.score_column,
    )
    if not args.score_column:
        rows = score_rows(rows, config_path=args.config, batch_size=args.batch_size)
    report = build_report(rows, threshold=args.threshold, min_recall=args.min_recall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Production-model evaluation report saved to {args.output}")
    print(f"F1 @ {args.threshold:.2f}: {report['f1']}")
    print(f"ROC-AUC: {report['roc_auc']}")
    return 0


def _require_score(row: LabeledLog) -> float:
    if row.score is None:
        raise ValueError("each row must have a score before building a report")
    return row.score


def _roc_auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


if __name__ == "__main__":
    raise SystemExit(main())
