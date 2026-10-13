"""Generate a Tier-2 threshold sweep report from held-out validation text windows.

The sweep runs on the *validation* split. The test split is never swept — a
threshold chosen by maximising over test rows is a selection maximand, not a
generalisation estimate.
"""

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

MIN_RECOMMENDATION_SAMPLES = 200


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
    parser.add_argument(
        "--min-recommendation-samples",
        type=int,
        default=MIN_RECOMMENDATION_SAMPLES,
        help=(
            "Refuse to emit a threshold recommendation below this many validation rows "
            f"(default {MIN_RECOMMENDATION_SAMPLES})"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _predict(classifier: Tier2Classifier, texts: list[str], batch_size: int) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        probabilities.append(classifier.predict_proba(texts[start : start + batch_size]))
    if not probabilities:
        return np.array([], dtype=np.float32)
    return np.concatenate(probabilities)


def main() -> int:
    args = parse_args()
    texts, labels, window_stats = build_windows(
        sample_normal=args.sample_normal,
        sample_failure=args.sample_failure,
        random_state=args.seed,
    )
    X = np.array(texts, dtype=object)
    y = np.array(labels, dtype=np.int64)
    _, X_val, _, _, y_val, _ = split_dataset(X, y, random_state=args.seed)

    classifier = Tier2Classifier(model_dir=args.model_dir)
    if not classifier.is_ready():
        print(f"Tier-2 classifier is not ready: {args.model_dir}")
        return 1

    y_prob = _predict(classifier, [str(text) for text in X_val.tolist()], args.batch_size)
    sweep = threshold_sweep(y_val, y_prob)
    summary = summarize_threshold_sweep(sweep, min_recall=args.min_recall)

    validation_rows = int(len(y_val))
    validation_failures = int(y_val.sum())
    reliable = (
        validation_rows >= args.min_recommendation_samples
        and validation_failures >= args.min_recommendation_samples
    )
    payload = {
        "model": "tier2_transformer",
        "model_dir": str(args.model_dir),
        "split": "val",
        "validation_samples": validation_rows,
        "validation_failures": validation_failures,
        "min_recommendation_samples": args.min_recommendation_samples,
        "recommendation_reliable": reliable,
        "tie_break": "highest_threshold",
        "window_stats": window_stats,
        "summary": summary,
        "sweep": sweep,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    if reliable:
        print(f"Tier-2 threshold report saved to {args.output}")
        print(f"Best F1: {summary['best_f1']}")
        print(
            f"Best precision at recall>={args.min_recall:.2f}: "
            f"{summary['best_precision_at_min_recall']}"
        )
        return 0

    print(f"Tier-2 threshold report saved to {args.output}")
    print(
        f"\nWARNING: only {validation_rows} validation rows "
        f"({validation_failures} failures) — below the {args.min_recommendation_samples}-row "
        "minimum for a threshold recommendation."
    )
    print(
        "The sweep above is reported for diagnostics only and MUST NOT be used to "
        "set a production threshold. Re-run with larger --sample-normal / "
        "--sample-failure values."
    )
    print(f"  diagnostic best F1: {summary['best_f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())