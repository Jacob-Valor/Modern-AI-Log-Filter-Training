"""Tests for the syslog-only classifier wrapper using fake ONNX/XGBoost modules."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from logfilter.models.syslog_classifier import SYSLOG_MODEL_DIR, SyslogClassifier


class FakeInferenceSession:
    """Stands in for onnxruntime.InferenceSession (2-D probability output)."""

    def __init__(self, path: str, providers=None) -> None:
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, _output_names, feed):
        x = feed["input"]
        n = len(x)
        proba = np.tile([0.3, 0.7], (n, 1)).astype(np.float32)
        return [np.zeros(n, dtype=np.int64), proba]


class FakeInferenceSession1D:
    """Variant whose probability output is 1-D (proba.ndim == 1 branch)."""

    def __init__(self, path: str, providers=None) -> None:
        self.path = path
        del providers

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, _output_names, feed):
        x = feed["input"]
        return [np.zeros(len(x), dtype=np.int64), np.full(len(x), 0.8, dtype=np.float32)]


class FailingInferenceSession:
    def __init__(self, path: str, providers=None) -> None:
        del path, providers
        raise RuntimeError("onnx load boom")


class FakeXGBClassifier:
    def __init__(self) -> None:
        self.loaded_path: str | None = None

    def load_model(self, path: str) -> None:
        self.loaded_path = path

    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.4, 0.6], (n, 1)).astype(np.float32)


def _install_onnx(monkeypatch, session_cls=FakeInferenceSession) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=session_cls),
    )


def _install_xgb(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBClassifier=FakeXGBClassifier),
    )


def _write_features(model_dir, names) -> None:
    (model_dir / "feature_names_syslog.json").write_text(json.dumps(names))


def _write_scaler(model_dir, scale) -> None:
    (model_dir / "scaler_syslog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "type": "MaxAbsScaler",
                "n_features_in": len(scale),
                "scale": scale,
            }
        )
    )


def test_init_defaults() -> None:
    clf = SyslogClassifier()
    assert clf.model_dir == SYSLOG_MODEL_DIR
    assert clf._feature_names == []
    assert clf._session is None
    assert clf._xgb_model is None


def test_init_custom_dir(tmp_path) -> None:
    clf = SyslogClassifier(model_dir=tmp_path)
    assert clf.model_dir == tmp_path


def test_load_missing_feature_names_returns_early(tmp_path) -> None:
    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()
    assert clf._feature_names == []
    assert clf.is_ready() is False


def test_load_onnx_session(tmp_path, monkeypatch) -> None:
    _install_onnx(monkeypatch)
    _write_features(tmp_path, ["a", "b", "c"])
    (tmp_path / "log_classifier_syslog.onnx").write_bytes(b"onnx-bytes")

    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()

    assert clf._session is not None
    assert clf._feature_names == ["a", "b", "c"]
    assert clf.is_ready() is True


def test_load_onnx_failure_falls_back_to_xgb(tmp_path, monkeypatch) -> None:
    _install_onnx(monkeypatch, session_cls=FailingInferenceSession)
    _install_xgb(monkeypatch)
    _write_features(tmp_path, ["a", "b"])
    (tmp_path / "log_classifier_syslog.onnx").write_bytes(b"onnx-bytes")
    (tmp_path / "log_classifier_syslog.json").write_text("{}")

    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()

    assert clf._session is None
    assert isinstance(clf._xgb_model, FakeXGBClassifier)
    assert clf._xgb_model.loaded_path.endswith("log_classifier_syslog.json")


def test_load_xgb_only(tmp_path, monkeypatch) -> None:
    _install_xgb(monkeypatch)
    _write_features(tmp_path, ["a", "b"])
    (tmp_path / "log_classifier_syslog.json").write_text("{}")

    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()

    assert isinstance(clf._xgb_model, FakeXGBClassifier)


def test_load_no_model_artifacts(tmp_path) -> None:
    _write_features(tmp_path, ["a", "b"])

    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()

    assert clf._session is None
    assert clf._xgb_model is None
    assert clf.is_ready() is False


def test_strict_mode_raises_when_syslog_model_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_MODELS_STRICT", "1")
    _write_features(tmp_path, ["a", "b"])
    clf = SyslogClassifier(model_dir=tmp_path)

    with pytest.raises(RuntimeError, match="No syslog classifier found"):
        clf.predict_proba(np.ones((1, 2), dtype=np.float32))


def test_load_with_scaler(tmp_path, monkeypatch) -> None:
    _install_xgb(monkeypatch)
    _write_features(tmp_path, ["a", "b", "c"])
    _write_scaler(tmp_path, [1.0, 2.0, 4.0])
    (tmp_path / "log_classifier_syslog.json").write_text("{}")

    clf = SyslogClassifier(model_dir=tmp_path)
    clf._load()

    assert clf._scaler is not None


def test_feature_names_property_triggers_load(tmp_path, monkeypatch) -> None:
    _install_onnx(monkeypatch)
    _write_features(tmp_path, ["x", "y"])
    (tmp_path / "log_classifier_syslog.onnx").write_bytes(b"onnx-bytes")

    clf = SyslogClassifier(model_dir=tmp_path)
    assert clf.feature_names == ["x", "y"]


def test_predict_proba_neutral_without_model(tmp_path) -> None:
    _write_features(tmp_path, ["a", "b"])
    clf = SyslogClassifier(model_dir=tmp_path)

    out = clf.predict_proba(np.zeros((3, 2), dtype=np.float32))

    assert out.shape == (3,)
    assert np.allclose(out, 0.5)


def test_predict_proba_onnx_2d(tmp_path, monkeypatch) -> None:
    _install_onnx(monkeypatch)
    _write_features(tmp_path, ["a", "b"])
    (tmp_path / "log_classifier_syslog.onnx").write_bytes(b"onnx-bytes")

    clf = SyslogClassifier(model_dir=tmp_path)
    out = clf.predict_proba(np.zeros((2, 2), dtype=np.float32))

    assert out.shape == (2,)
    assert np.allclose(out, 0.7)


def test_predict_proba_onnx_1d(tmp_path, monkeypatch) -> None:
    _install_onnx(monkeypatch, session_cls=FakeInferenceSession1D)
    _write_features(tmp_path, ["a", "b"])
    (tmp_path / "log_classifier_syslog.onnx").write_bytes(b"onnx-bytes")

    clf = SyslogClassifier(model_dir=tmp_path)
    out = clf.predict_proba(np.zeros((2, 2), dtype=np.float32))

    assert np.allclose(out, 0.8)


def test_predict_proba_xgb(tmp_path, monkeypatch) -> None:
    _install_xgb(monkeypatch)
    _write_features(tmp_path, ["a", "b"])
    (tmp_path / "log_classifier_syslog.json").write_text("{}")

    clf = SyslogClassifier(model_dir=tmp_path)
    out = clf.predict_proba(np.zeros((2, 2), dtype=np.float32))

    assert out.shape == (2,)
    assert np.allclose(out, 0.6)


def test_predict_proba_with_scaler_applied(tmp_path, monkeypatch) -> None:
    _install_xgb(monkeypatch)
    _write_features(tmp_path, ["a", "b", "c"])
    _write_scaler(tmp_path, [1.0, 2.0, 4.0])
    (tmp_path / "log_classifier_syslog.json").write_text("{}")

    clf = SyslogClassifier(model_dir=tmp_path)
    out = clf.predict_proba(np.ones((2, 3), dtype=np.float32))

    assert out.shape == (2,)
    assert np.allclose(out, 0.6)
