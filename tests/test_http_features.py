"""Tests for label-blind HTTP lexical features."""

from __future__ import annotations

import importlib
import math

import pytest  # type: ignore[import-not-found]

http_features = importlib.import_module("logfilter.pipeline.http_features")
HTTP_FEATURE_NAMES = http_features.HTTP_FEATURE_NAMES
extract_http_features = http_features.extract_http_features
http_lexical_score = http_features.http_lexical_score
percent_decode_layers = http_features.percent_decode_layers


@pytest.mark.parametrize(
    ("uri", "body", "feature"),
    [
        ("/item?id=1%27%20OR%201%3D1--", "", "sql_tautology_count"),
        ("/search", "q=%253Cscript%253Ealert(1)%253C/script%253E", "xss_tag_count"),
        ("/download?f=..%252f..%252fetc%252fpasswd", "", "traversal_count"),
    ],
)
def test_attack_lexical_features_survive_bounded_decoding(uri, body, feature) -> None:
    features = extract_http_features(uri, body)
    assert features[feature] > 0
    assert http_lexical_score(features) >= 0.60


def test_benign_request_has_zero_lexical_risk() -> None:
    features = extract_http_features("/products?page=2&sort=name", '{"query":"router"}')
    assert tuple(features) == HTTP_FEATURE_NAMES
    assert http_lexical_score(features) == 0.0
    assert all(math.isfinite(value) and value >= 0.0 for value in features.values())


def test_decode_depth_is_bounded() -> None:
    encoded = "%252525252525252e%252525252525252e%252525252525252f"
    layers = percent_decode_layers(encoded, max_depth=4)
    assert len(layers) <= 5
    assert extract_http_features(encoded)["decode_depth"] == 4.0


def test_invalid_decode_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        percent_decode_layers("%2e", max_depth=-1)


def test_features_do_not_retain_secret_values() -> None:
    secret = "super-secret-token-value"
    features = extract_http_features("/login", {"password": secret, "query": "normal"})
    assert secret not in repr(features)
