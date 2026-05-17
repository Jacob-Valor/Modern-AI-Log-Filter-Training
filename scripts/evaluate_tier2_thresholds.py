"""Generate a Tier-2 threshold sweep report from held-out text windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.models.tier2_classifier import Tier2Classifier  # noqa: E402
from training.data_loader import split_dataset  # noqa: E402
from training.text_dataset import build_windows  # noqa: E402
from training.thresholds import summarize_threshold_sweep, threshold_sweep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tier-2 threshold strategy")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "tier2",
        help="Directory containing Tier-2 model artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "tier2" / "tier2_threshold_report.json",
        help="JSON report output path",
    )
    parser.add_argument("--sample-normal", type=int, default=None)
    parser.add_argument("--sample-failure", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texts, labels, window_stats = build_windows(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
        random_state=args.seed,
    )
    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=np.int64)
    _, _, X_test, _, _, y_test = split_dataset(X, y, random_state=args.seed)

    classifier = Tier2Classifier(model_dir=args.model_dir)
    if not classifier.is_ready():
        print(f"Tier-2 classifier is not ready: {args.model_dir}")
        return 1

    probabilities: list[np.ndarray] = []
    test_texts = [str(text) for text in X_test.tolist()]
    for start in range(0, len(test_texts), args.batch_size):
        probabilities.append(classifier.predict_proba(test_texts[start : start + args.batch_size]))
    y_prob = np.concatenate(probabilities) if probabilities else np.array([], dtype=np.float32)
    sweep = threshold_sweep(y_test, y_prob)
    summary = summarize_threshold_sweep(sweep, min_recall=args.min_recall)
    payload = {
        "model": "tier2_transformer",
        "model_dir": str(args.model_dir),
        "split": "test",
        "test_samples": int(len(y_test)),
        "test_failures": int(y_test.sum()),
        "window_stats": window_stats,
        "summary": summary,
        "sweep": sweep,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Tier-2 threshold report saved to {args.output}")
    print(f"Best F1: {summary['best_f1']}")
    print(
        f"Best precision at recall>={args.min_recall:.2f}: "
        f"{summary['best_precision_at_min_recall']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
