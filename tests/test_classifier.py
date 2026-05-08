"""Tests for classifier artifact loading safety."""

from __future__ import annotations

import numpy as np
import pytest

from logfilter.models.classifier import LogClassifier, SafeMaxAbsScaler


def test_safe_max_abs_scaler_json_round_trip(tmp_path) -> None:
    path = tmp_path / "scaler.json"
    SafeMaxAbsScaler(np.array([2.0, 4.0], dtype=np.float32)).to_json(path)

    scaler = SafeMaxAbsScaler.from_json(path)

    transformed = scaler.transform(np.array([[2.0, 8.0]], dtype=np.float32))
    np.testing.assert_allclose(transformed, np.array([[1.0, 2.0]], dtype=np.float32))
    assert scaler.n_features_in_ == 2


def test_classifier_rejects_non_json_scaler_artifact(tmp_path) -> None:
    scaler_path = tmp_path / "scaler.pkl"
    scaler_path.write_bytes(b"not a safe runtime artifact")
    classifier = LogClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=scaler_path,
        feature_names_path=tmp_path / "missing-feature-names.json",
    )

    with pytest.raises(ValueError, match="Refusing to load unsafe scaler artifact"):
        classifier.predict_proba(np.zeros((1, 1), dtype=np.float32))


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        ([], "empty"),
        ([1.0, float("nan")], "non-finite"),
        ([1.0, 0.0], "zero"),
    ],
)
def test_safe_max_abs_scaler_rejects_invalid_scale(scale, message) -> None:
    with pytest.raises(ValueError, match=message):
        SafeMaxAbsScaler(np.array(scale, dtype=np.float32))


def test_safe_max_abs_scaler_rejects_invalid_json_payloads(tmp_path) -> None:
    path = tmp_path / "scaler.json"

    path.write_text('{"type": "Other", "scale": [1.0]}')
    with pytest.raises(ValueError, match="Unsupported scaler type"):
        SafeMaxAbsScaler.from_json(path)

    path.write_text('{"type": "MaxAbsScaler", "scale": "bad"}')
    with pytest.raises(ValueError, match="list-valued"):
        SafeMaxAbsScaler.from_json(path)

    path.write_text('{"type": "MaxAbsScaler", "scale": [1.0], "n_features_in": 2}')
    with pytest.raises(ValueError, match="feature count"):
        SafeMaxAbsScaler.from_json(path)


def test_safe_max_abs_scaler_from_sklearn() -> None:
    class FakeSklearnScaler:
        scale_ = np.array([2.0, 4.0], dtype=np.float32)

    scaler = SafeMaxAbsScaler.from_sklearn(FakeSklearnScaler())

    assert scaler.n_features_in_ == 2


def test_safe_max_abs_scaler_from_sklearn_requires_fitted_scaler() -> None:
    with pytest.raises(ValueError, match="fitted sklearn"):
        SafeMaxAbsScaler.from_sklearn(object())


def test_safe_max_abs_scaler_transform_validates_shape() -> None:
    scaler = SafeMaxAbsScaler(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="2D"):
        scaler.transform(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.transform(np.array([[1.0]], dtype=np.float32))


def test_classifier_returns_neutral_probability_without_model(tmp_path) -> None:
    classifier = LogClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=tmp_path / "missing-scaler.json",
        feature_names_path=tmp_path / "missing-feature-names.json",
    )

    result = classifier.predict_proba(np.zeros((2, 3), dtype=np.float32))

    np.testing.assert_allclose(result, np.array([0.5, 0.5]))
    assert not classifier.is_ready()


def test_classifier_applies_scaler_and_session_list_output(tmp_path) -> None:
    scaler_path = tmp_path / "scaler.json"
    SafeMaxAbsScaler(np.array([2.0], dtype=np.float32)).to_json(scaler_path)

    class FakeInput:
        name = "features"
        shape = [None, 1]

    class FakeSession:
        def __init__(self) -> None:
            self.seen = None

        def run(self, output_names, feed):
            self.seen = feed["features"]
            return [None, [{"0": 0.1, "1": 0.9}]]

        def get_inputs(self):
            return [FakeInput()]

    classifier = LogClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=scaler_path,
        feature_names_path=tmp_path / "missing-feature-names.json",
    )
    session = FakeSession()
    classifier._session = session
    classifier._input_name = "features"
    classifier._scaler = SafeMaxAbsScaler.from_json(scaler_path)

    result = classifier.predict_proba(np.array([[4.0]], dtype=np.float32))

    np.testing.assert_allclose(session.seen, np.array([[2.0]], dtype=np.float32))
    np.testing.assert_allclose(result, np.array([0.9]))
    assert classifier.expected_feature_count == 1
    assert classifier.is_ready()


def test_classifier_session_array_and_xgb_paths(tmp_path) -> None:
    class FakeSession:
        def run(self, output_names, feed):
            return [None, np.array([[0.2, 0.8]], dtype=np.float32)]

    classifier = LogClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=tmp_path / "missing-scaler.json",
        feature_names_path=tmp_path / "missing-feature-names.json",
    )
    classifier._session = FakeSession()
    classifier._input_name = "features"
    np.testing.assert_allclose(
        classifier.predict_proba(np.array([[1.0]], dtype=np.float32)),
        np.array([0.8], dtype=np.float32),
    )

    class FakeXGB:
        n_features_in_ = 3

        def predict_proba(self, values):
            return np.array([[0.7, 0.3]], dtype=np.float32)

    classifier._session = None
    classifier._xgb_model = FakeXGB()
    assert classifier.expected_feature_count == 3
    assert classifier.predict_single(np.array([1.0, 2.0, 3.0], dtype=np.float32)) == pytest.approx(
        0.3
    )


def test_classifier_feature_names_property_loads_json(tmp_path) -> None:
    feature_names_path = tmp_path / "feature_names.json"
    feature_names_path.write_text('["a", "b"]')
    classifier = LogClassifier(
        model_path=tmp_path / "missing.onnx",
        scaler_path=tmp_path / "missing-scaler.json",
        feature_names_path=feature_names_path,
    )

    assert classifier.feature_names == ["a", "b"]
    assert classifier.expected_feature_count == 2
