"""Tests for leakage-resistant flow split utilities."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
flow_training = importlib.import_module("training.train_flow_classifier")


def test_exact_feature_duplicates_are_excluded_from_later_partitions() -> None:
    features = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 2.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    development = flow_training._hashes_for(features, np.array([0, 1]))

    retained, removed = flow_training._exclude_seen_features(
        features, np.array([2, 3]), development
    )

    assert retained.tolist() == [3]
    assert removed == 1


def test_float32_hash_is_stable_across_equivalent_numeric_dtypes() -> None:
    first = flow_training._feature_hash(np.array([1.0, 2.0], dtype=np.float64))
    second = flow_training._feature_hash(np.array([1.0, 2.0], dtype=np.float32))
    assert first == second
