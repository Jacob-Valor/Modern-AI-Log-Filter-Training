#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Train a BETH endpoint novelty model with zero evil samples in training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from logfilter.pipeline.endpoint_features import (
    ENDPOINT_FEATURE_NAMES,
    endpoint_vector_from_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "processId",
    "threadId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "processName",
    "eventId",
    "eventName",
    "argsNum",
    "returnValue",
    "args",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "demo_data" / "BETH")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "endpoint")
    parser.add_argument("--train-benign", type=int, default=75_000)
    parser.add_argument("--validation-benign", type=int, default=25_000)
    parser.add_argument("--test-positive", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--validation-fpr", type=float, default=0.01)
    return parser.parse_args()


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=COLUMNS + ["sus", "evil"])


def _sample(
    frame: pd.DataFrame,
    *,
    benign_limit: int,
    positive_limit: int,
    seed: int,
) -> pd.DataFrame:
    positive_mask = (frame["sus"] > 0) | (frame["evil"] > 0)
    benign = frame.loc[~positive_mask]
    positive = frame.loc[positive_mask]
    if len(benign) > benign_limit:
        benign = benign.sample(n=benign_limit, random_state=seed)
    if len(positive) > positive_limit:
        positive = positive.sample(n=positive_limit, random_state=seed + 1)
    return pd.concat([benign, positive]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _vectors(frame: pd.DataFrame) -> np.ndarray:
    records = (
        dict(zip(COLUMNS, values)) for values in frame[COLUMNS].itertuples(index=False, name=None)
    )
    return np.vstack([endpoint_vector_from_mapping(record) for record in records]).astype(
        np.float32
    )


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return ((frame["sus"] > 0) | (frame["evil"] > 0)).to_numpy(dtype=np.int8)


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not an integer") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not numeric") from exc


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n": len(labels),
        "tp": _as_int(tp, "true positives"),
        "fp": _as_int(fp, "false positives"),
        "tn": _as_int(tn, "true negatives"),
        "fn": _as_int(fn, "false negatives"),
        "precision": _as_float(precision, "precision"),
        "recall": _as_float(recall, "recall"),
        "f1": _as_float(2 * precision * recall / max(precision + recall, 1e-12), "f1"),
        "fpr": _as_float(fp / max(fp + tn, 1), "false-positive rate"),
        "auroc": _as_float(roc_auc_score(labels, scores), "AUROC"),
        "auprc": _as_float(average_precision_score(labels, scores), "AUPRC"),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fpr < 0.5:
        raise SystemExit("validation_fpr must be between 0 and 0.5")
    train = _load(args.data_dir / "labelled_training_data.csv")
    validation = _load(args.data_dir / "labelled_validation_data.csv")
    test = _load(args.data_dir / "labelled_testing_data.csv")
    # Training is benign-only by construction. Suspicious and evil rows are held out.
    train = _sample(
        train,
        benign_limit=args.train_benign,
        positive_limit=0,
        seed=args.seed,
    )
    validation = _sample(
        validation,
        benign_limit=args.validation_benign,
        positive_limit=len(validation),
        seed=args.seed + 10,
    )
    test = _sample(
        test,
        benign_limit=len(test),
        positive_limit=args.test_positive,
        seed=args.seed + 20,
    )
    train_x = _vectors(train)
    validation_x = _vectors(validation)
    test_x = _vectors(test)
    validation_y = _labels(validation)
    test_y = _labels(test)
    if _labels(train).sum() != 0:
        raise RuntimeError("Endpoint novelty training set contains labelled positives")

    max_samples: Any = 4096
    model = IsolationForest(
        n_estimators=300,
        max_samples=max_samples,
        contamination="auto",
        random_state=args.seed,
        n_jobs=4,
    ).fit(train_x)
    validation_scores = -model.score_samples(validation_x)
    test_scores = -model.score_samples(test_x)
    threshold = _as_float(
        np.quantile(validation_scores[validation_y == 0], 1.0 - args.validation_fpr),
        "novelty threshold",
    )
    evil_mask = test["evil"].to_numpy() > 0
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "seed": args.seed,
            "training_positive_count": 0,
            "training_evil_count": 0,
            "validation_fpr_target": args.validation_fpr,
            "threshold": threshold,
            "label_features_excluded": True,
            "host_and_timestamp_features_excluded": True,
            "claim": "unseen-family/open-set proxy; not genuine zero-day proof",
        },
        "validation": _metrics(validation_y, validation_scores, threshold),
        "test": _metrics(test_y, test_scores, threshold),
        "test_evil_recall": _as_float((test_scores[evil_mask] >= threshold).mean(), "evil recall"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, len(ENDPOINT_FEATURE_NAMES)]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={IsolationForest: {"score_samples": True}},
    )
    onnx_path = args.output_dir / "endpoint_novelty.onnx"
    serialized = getattr(converted, "SerializeToString", None)
    if not callable(serialized):
        raise RuntimeError("ONNX conversion did not return a serializable model")
    onnx_path.write_bytes(serialized())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: test_x[:32]})
    report["onnx"] = {
        "output_names": output_names,
        "output_shapes": [list(np.asarray(value).shape) for value in outputs],
    }
    (args.output_dir / "novelty_config.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "feature_names": list(ENDPOINT_FEATURE_NAMES),
                "score_direction": "negative_score_samples",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "novelty_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
