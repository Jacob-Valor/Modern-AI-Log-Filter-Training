"""Tests for label-blind HTTP lexical features."""

from __future__ import annotations

import importlib
import math
import time

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


# ── ReDoS regression ───────────────────────────────────────────────────────────
# The original ``select\s+.+?\s+from`` and ``['"]?\s*(?:or|and)\s+`` patterns chained
# adjacent quantifiers that could both match whitespace. Measured on this machine:
# a 1.6 KB whitespace run cost 2.7 s and grew ~n**2.5; a 64 KB run exceeded a
# 280 s timeout. Both were reachable from POST /score, whose ``raw`` field permits
# 65536 characters. These tests pin the cost, not the pattern text, so a future
# rewrite that reintroduces overlapping quantifiers fails here.


def test_sql_keyword_pattern_is_linear_on_whitespace_run() -> None:
    pattern = http_features._SQL_KEYWORD_RE
    small = "select " + " " * 8_000 + "x"
    large = "select " + " " * 32_000 + "x"

    start = time.perf_counter()
    pattern.findall(small)
    small_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    pattern.findall(large)
    large_elapsed = time.perf_counter() - start

    # 4x the input must not cost anywhere near 16x the time. A generous absolute
    # ceiling keeps the test meaningful on a loaded CI box while still failing hard
    # on quadratic behaviour (the original measured 2.7 s at 1/10th this size).
    assert large_elapsed < 0.5, (
        f"4x input took {large_elapsed:.3f}s (small took {small_elapsed:.3f}s)"
    )
    assert large_elapsed < max(small_elapsed * 10, 0.05)


def test_sql_tautology_pattern_is_linear_on_whitespace_run() -> None:
    pattern = http_features._SQL_TAUTOLOGY_RE
    small = "' or " + " " * 8_000 + "a"
    large = "' or " + " " * 32_000 + "a"

    start = time.perf_counter()
    pattern.findall(small)
    small_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    pattern.findall(large)
    large_elapsed = time.perf_counter() - start

    assert large_elapsed < 0.5, (
        f"4x input took {large_elapsed:.3f}s (small took {small_elapsed:.3f}s)"
    )
    assert large_elapsed < max(small_elapsed * 10, 0.05)


def test_extract_http_features_handles_max_size_adversarial_body() -> None:
    """A body at the documented ScoreRequest.raw limit must stay fast."""
    body = "select " + " " * 65_500 + "x"
    start = time.perf_counter()
    features = extract_http_features("/login", body)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"64 KB adversarial body took {elapsed:.3f}s"
    assert features["body_length"] == 32768.0


@pytest.mark.parametrize(
    "payload",
    [
        "1 UNION SELECT password FROM users",
        "1 union all select a from b",
        "select id,name from accounts where x=1",
        "SELECT   a,b   FROM   t",
        "select * from users",
        "SELECT\n\n  a\n FROM t",
        "insert into t values(1)",
        "update users set name=1",
        "delete from users",
        "drop table users",
        "drop database x",
        "information_schema.tables",
        "sleep(5)",
        "BENCHMARK(1)",
    ],
)
def test_sql_keyword_detection_preserved(payload: str) -> None:
    assert http_features._SQL_KEYWORD_RE.findall(payload)


@pytest.mark.parametrize(
    "payload",
    ["1' or 1=1 --", "admin' OR '1'='1", "or 1=1", "' or 'a'='a", "x AND 1=1", "' or 1 = 1"],
)
def test_sql_tautology_detection_preserved(payload: str) -> None:
    assert http_features._SQL_TAUTOLOGY_RE.findall(payload)


@pytest.mark.parametrize(
    "payload",
    ["unremarkable text", "user failed password", "", "select", "from", "or", "SELECTED", "slept"],
)
def test_sql_patterns_do_not_match_benign_text(payload: str) -> None:
    assert not http_features._SQL_KEYWORD_RE.findall(payload)
    assert not http_features._SQL_TAUTOLOGY_RE.findall(payload)
