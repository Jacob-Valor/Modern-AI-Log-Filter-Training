"""Tests for leakage-resistant flow split utilities."""

from __future__ import annotations

import csv
import importlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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


def _cic_archive(tmp_path: Path, timestamps: list[str], *, with_timestamp: bool = True) -> Path:
    """Build a one-member ZIP shaped like a CICFlowMeter capture CSV."""
    member = f"demo/{flow_training.CAPTURES[0].member_suffix}"
    spec = flow_training.CAPTURES[0]
    header = [*flow_training.SOURCE_COLUMNS.values(), flow_training.LABEL]
    if with_timestamp:
        header.append("Timestamp")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for index, timestamp in enumerate(timestamps):
        label = "BENIGN" if index % 2 == 0 else f"{spec.attack_prefixes[0]}-Probe"
        row = [float(index + 1)] * len(flow_training.SOURCE_COLUMNS) + [label]
        if with_timestamp:
            row.append(timestamp)
        writer.writerow(row)
    archive_path = tmp_path / "capture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, buffer.getvalue())
    return archive_path



def _cic_time(second: int) -> str:
    return f"13/07/2017:22:15:{second:02d}"


def test_load_capture_rejects_out_of_order_timestamps(tmp_path: Path) -> None:
    """The chronological split is driven by CSV row order; Timestamp must agree with it."""
    archive = _cic_archive(tmp_path, [_cic_time(5), _cic_time(1)])

    with pytest.raises(ValueError, match="monotonically increasing"):
        flow_training.load_capture(archive, flow_training.CAPTURES[0])


def test_load_capture_verifies_ordered_timestamps_when_present(tmp_path: Path) -> None:
    archive = _cic_archive(tmp_path, [_cic_time(1), _cic_time(2), _cic_time(3)])

    frame = flow_training.load_capture(archive, flow_training.CAPTURES[0])

    assert frame.attrs["timestamp_column_present"] is True
    assert frame.attrs["timestamp_monotonic_increasing"] is True
    assert "_row" not in frame.columns


def test_load_capture_records_absent_timestamp_instead_of_failing(tmp_path: Path) -> None:
    """The shipped CIC-IDS2017 captures have no Timestamp column; say so, do not crash."""
    archive = _cic_archive(tmp_path, ["", "", ""], with_timestamp=False)

    frame = flow_training.load_capture(archive, flow_training.CAPTURES[0])

    assert frame.attrs["timestamp_column_present"] is False
    assert frame.attrs["timestamp_monotonic_increasing"] is None


def test_cpu_is_always_accepted_and_needs_no_gpu() -> None:
    assert flow_training.resolve_device("cpu") == "cpu"


def test_cuda_request_fails_loudly_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import xgboost as xgb

    monkeypatch.setattr(xgb, "build_info", lambda: {"USE_CUDA": False})

    with pytest.raises(SystemExit, match="--device cpu"):
        flow_training.resolve_device("cuda")


def test_repo_relative_never_leaks_the_home_directory(tmp_path: Path) -> None:
    inside = flow_training.repo_relative(flow_training.ROOT / "demo_data" / "capture.zip")
    assert inside == str(Path("demo_data") / "capture.zip")


    outside = flow_training.repo_relative(tmp_path / "elsewhere.zip")
    assert outside == "elsewhere.zip"
