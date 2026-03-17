"""
HDFS TraceBench data loader.

Loads normal_trace.csv (label=0) and failure_trace.csv (label=1),
returns train/val/test splits as numpy arrays.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

logger = logging.getLogger(__name__)

PREPROCESSED_DIR = Path(__file__).parent.parent / "HDFS_v3_TraceBench" / "preprocessed"


def load_traces(
    preprocessed_dir: Path = PREPROCESSED_DIR,
    sample_normal: int | None = None,
    sample_failure: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load normal and failure traces, returning (X, y, feature_names).

    Parameters
    ----------
    preprocessed_dir : Path
        Directory containing normal_trace.csv and failure_trace.csv.
    sample_normal : int | None
        If set, subsample this many normal rows (for faster development).
    sample_failure : int | None
        If set, subsample this many failure rows.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Bag-of-events count matrix (float32).
    y : np.ndarray of shape (n_samples,)
        Labels: 0 = normal, 1 = failure.
    feature_names : list[str]
        Column names (event descriptions) corresponding to X columns.
    """
    normal_path = preprocessed_dir / "normal_trace.csv"
    failure_path = preprocessed_dir / "failure_trace.csv"

    logger.info("Loading normal traces from %s …", normal_path)
    # dtype cannot be applied globally when index_col=0 (TaskID is a hex string).
    # Read without dtype, then cast feature columns only.
    df_normal = pd.read_csv(normal_path, index_col=0)
    df_normal = df_normal.astype(np.float32)
    if sample_normal is not None:
        df_normal = df_normal.sample(
            n=min(sample_normal, len(df_normal)), random_state=random_state
        )

    logger.info("Loading failure traces from %s …", failure_path)
    df_failure = pd.read_csv(failure_path, index_col=0)
    df_failure = df_failure.astype(np.float32)
    if sample_failure is not None:
        df_failure = df_failure.sample(
            n=min(sample_failure, len(df_failure)), random_state=random_state
        )

    # Sanity-check: both files must have the same columns
    assert list(df_normal.columns) == list(df_failure.columns), (
        "Column mismatch between normal_trace.csv and failure_trace.csv"
    )

    feature_names = list(df_normal.columns)

    y_normal = np.zeros(len(df_normal), dtype=np.int32)
    y_failure = np.ones(len(df_failure), dtype=np.int32)

    X = np.vstack([df_normal.values, df_failure.values])
    y = np.concatenate([y_normal, y_failure])

    logger.info(
        "Loaded %d normal + %d failure = %d total samples, %d features",
        len(y_normal),
        len(y_failure),
        len(y),
        X.shape[1],
    )
    return X, y, feature_names


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Stratified train / val / test split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split off test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split validation from remaining
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, stratify=y_tmp, random_state=random_state
    )

    logger.info(
        "Split → train=%d  val=%d  test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MaxAbsScaler]:
    """
    Fit MaxAbsScaler on training data (preserves sparsity), apply to all splits.
    Returns scaled arrays and the fitted scaler (for inference pipeline use).
    """
    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def load_event_vocabulary(preprocessed_dir: Path = PREPROCESSED_DIR) -> dict[str, int]:
    """Load eventId.json as {event_name: int_id}."""
    path = preprocessed_dir / "eventId.json"
    raw = json.loads(path.read_text())
    return {name: idx for name, idx in raw}
