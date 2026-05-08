"""Verify that the Tier-2 transformer artifact can run local inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.tier2_classifier import Tier2Classifier  # noqa: E402


def main() -> int:
    model_dir = ROOT / "models" / "tier2"
    classifier = Tier2Classifier(model_dir=model_dir)

    if not classifier._artifacts_present():
        print(f"Tier-2 artifact missing or incomplete: {model_dir}")
        return 1

    texts = [
        "INFO healthcheck completed successfully for service auth-api",
        "ERROR failed password for root from 10.0.0.5 after repeated authentication failures",
    ]
    probs = classifier.predict_proba(texts)
    for text, prob in zip(texts, probs):
        print(f"failure_prob={float(prob):.4f}\t{text}")

    return 0 if len(probs) == len(texts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
