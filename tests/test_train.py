"""B14 regression: ONNX export is mandatory — never silently skipped."""

from __future__ import annotations

import sys

import pytest

import training.train as train_mod


def test_export_onnx_raises_when_onnxmltools_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "onnxmltools", None)

    class _Model:
        def save_model(self, path: str) -> None:  # pragma: no cover - must NOT be called
            raise AssertionError("native fallback save must not run when ONNX export is required")

    with pytest.raises(RuntimeError, match="onnxmltools"):
        train_mod.export_onnx(_Model(), 10, tmp_path / "log_classifier.onnx")
