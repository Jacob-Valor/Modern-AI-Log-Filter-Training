from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.config import load_config  # noqa: E402
from logfilter.pipeline.normalizer import LogNormalizer  # noqa: E402
from logfilter.pipeline.scorer import LogScorer  # noqa: E402


DATASET_PATH = ROOT / "scripts" / "eval_dataset.csv"
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUTPUT_PATH = ROOT / "models" / "detection_evaluation.json"
BATCH_SIZE = 64
DETECTION_THRESHOLD = 0.20


@dataclass(frozen=True)
class LabeledSample:
    raw: str
    label: int


@dataclass(frozen=True)
class ScoredSample:
    raw: str
    label: int
    score: float


def load_dataset(path: Path = DATASET_PATH) -> list[LabeledSample]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["raw", "label"]:
            raise ValueError("dataset must have exactly these columns: raw,label")
        samples: list[LabeledSample] = []
        for row_number, row in enumerate(reader, start=2):
            raw = row.get("raw", "").strip()
            if not raw:
                raise ValueError(f"missing raw log at row {row_number}")
            try:
                label = int(row.get("label", ""))
            except ValueError as exc:
                raise ValueError(f"label must be 0 or 1 at row {row_number}") from exc
            if label not in {0, 1}:
                raise ValueError(f"label must be 0 or 1 at row {row_number}")
            samples.append(LabeledSample(raw=raw, label=label))
    if not samples:
        raise ValueError("dataset contains no samples")
    return samples


def score_samples(samples: list[LabeledSample], batch_size: int = BATCH_SIZE) -> list[ScoredSample]:
    config = load_config(CONFIG_PATH)
    normalizer = LogNormalizer()
    scorer = LogScorer(config=config)
    scorer.preload_models()

    scored_samples: list[ScoredSample] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        normalized = [normalizer.normalize(sample.raw) for sample in batch]
        scored_events = scorer.score_batch(normalized)
        for sample, scored_event in zip(batch, scored_events, strict=True):
            scored_samples.append(
                ScoredSample(
                    raw=sample.raw,
                    label=sample.label,
                    score=float(scored_event.ai_threat_score),
                )
            )
    return scored_samples


def build_report(
    samples: list[ScoredSample], threshold: float = DETECTION_THRESHOLD
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    attack_scores: list[float] = []
    benign_scores: list[float] = []

    for sample in samples:
        detected = sample.score >= threshold
        if sample.label == 1:
            attack_scores.append(sample.score)
            if detected:
                tp += 1
            else:
                fn += 1
        else:
            benign_scores.append(sample.score)
            if detected:
                fp += 1
            else:
                tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    fpr = _safe_divide(fp, fp + tn)

    return {
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
        "threshold": threshold,
        "batch_size": BATCH_SIZE,
        "samples": len(samples),
        "attack_samples": len(attack_scores),
        "benign_samples": len(benign_scores),
        "detection_rate": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "score_distribution": {
            "attacks": _score_distribution(attack_scores),
            "benign": _score_distribution(benign_scores),
        },
    }


def save_report(report: dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


def print_summary(report: dict[str, Any]) -> None:
    matrix = report["confusion_matrix"]
    print(f"Detection evaluation saved to {OUTPUT_PATH}")
    print(
        f"Samples: {report['samples']} "
        f"({report['attack_samples']} attack, {report['benign_samples']} benign)"
    )
    print(f"Threshold: {report['threshold']:.2f}")
    print(f"Detection Rate / Recall: {report['detection_rate']:.4f}")
    print(f"False Positive Rate: {report['false_positive_rate']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"F1: {report['f1']:.4f}")
    print(
        "Confusion Matrix: "
        f"TP={matrix['tp']} FP={matrix['fp']} TN={matrix['tn']} FN={matrix['fn']}"
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator / denominator), 6)


def _score_distribution(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(scores)
    return {
        "count": len(scores),
        "mean": round(float(mean(ordered)), 6),
        "median": round(float(median(ordered)), 6),
        "p95": round(float(_percentile(ordered, 95)), 6),
    }


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (len(ordered_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered_values) - 1)
    fraction = rank - lower
    return ordered_values[lower] + (ordered_values[upper] - ordered_values[lower]) * fraction


def main() -> int:
    samples = load_dataset()
    scored_samples = score_samples(samples)
    report = build_report(scored_samples)
    save_report(report)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
