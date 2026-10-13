"""Tests for framework-independent API security helpers."""

from __future__ import annotations

from collections import deque
from unittest.mock import Mock

import pytest

from logfilter.api.security import (
    AccessDenied,
    RedisRateLimiter,
    client_ip_from_request,
    enforce_rate_limit,
    require_configured_token,
)


def test_token_check_fails_closed_when_token_not_configured() -> None:
    with pytest.raises(AccessDenied) as exc_info:
        require_configured_token(
            "anything",
            "",
            not_configured_detail="Scoring API token is not configured",
            invalid_detail="Invalid scoring API token",
        )

    assert exc_info.value.status_code == 403


def test_token_check_rejects_invalid_token() -> None:
    with pytest.raises(AccessDenied) as exc_info:
        require_configured_token(
            "wrong-token",
            "correct-token",
            not_configured_detail="Scoring API token is not configured",
            invalid_detail="Invalid scoring API token",
        )

    assert exc_info.value.status_code == 401


def test_token_check_accepts_valid_token() -> None:
    require_configured_token(
        "correct-token",
        "correct-token",
        not_configured_detail="Scoring API token is not configured",
        invalid_detail="Invalid scoring API token",
    )


def test_rate_limit_is_enforced_per_client() -> None:
    windows: dict[str, deque[float]] = {}

    enforce_rate_limit(windows, "198.51.100.10", 2, now=100.0)
    enforce_rate_limit(windows, "198.51.100.10", 2, now=101.0)
    with pytest.raises(AccessDenied) as exc_info:
        enforce_rate_limit(windows, "198.51.100.10", 2, now=102.0)

    assert exc_info.value.status_code == 429


def test_rate_limit_expires_old_entries() -> None:
    windows: dict[str, deque[float]] = {"198.51.100.10": deque([1.0, 2.0])}

    enforce_rate_limit(windows, "198.51.100.10", 2, now=70.0)

    assert list(windows["198.51.100.10"]) == [70.0]


def test_rate_limit_can_be_disabled() -> None:
    windows: dict[str, deque[float]] = {}

    enforce_rate_limit(windows, "198.51.100.10", 0, now=100.0)

    assert windows == {}


def test_rate_limit_window_map_stays_bounded_under_distinct_client_ids() -> None:
    """A stream of one-shot client ids must not grow the window map unbounded."""
    windows: dict[str, deque[float]] = {}

    # 2k distinct ids arriving over 100 s. Without a sweep the map would hold
    # all 2k; with one it holds only the ids still inside the 60 s horizon.
    for i in range(2_000):
        client_id = f"198.51.100.{i % 250}.{i // 250}"
        enforce_rate_limit(windows, client_id, 5, now=100.0 + i * 0.05)

    assert len(windows) < 2_000


def test_rate_limit_sweep_keeps_windows_still_inside_the_horizon() -> None:
    windows: dict[str, deque[float]] = {}

    enforce_rate_limit(windows, "198.51.100.10", 5, now=100.0)
    enforce_rate_limit(windows, "198.51.100.11", 5, now=150.0)
    enforce_rate_limit(windows, "198.51.100.12", 5, now=200.0)

    assert set(windows) == {"198.51.100.11", "198.51.100.12"}
    assert list(windows["198.51.100.12"]) == [200.0]


def test_redis_rate_limit_uses_atomic_lua_script() -> None:
    client = Mock()
    script = Mock(return_value=[1, 1])
    client.register_script.return_value = script
    limiter = RedisRateLimiter(client)

    enforce_rate_limit({}, "198.51.100.10", 2, backend=limiter, now=123.456)

    client.register_script.assert_called_once()
    script.assert_called_once_with(
        keys=[
            "logfilter:rate-limit:198.51.100.10",
            "logfilter:rate-limit:198.51.100.10:seq",
        ],
        args=[2, 60.0, 123.456],
    )


def test_redis_rate_limit_raises_when_window_is_full() -> None:
    client = Mock()
    client.register_script.return_value = Mock(return_value=[0, 2])
    limiter = RedisRateLimiter(client)

    with pytest.raises(AccessDenied) as exc_info:
        enforce_rate_limit({}, "198.51.100.10", 2, backend=limiter)

    assert exc_info.value.status_code == 429


def test_client_ip_from_request_uses_rightmost_untrusted_xff_from_trusted_proxy() -> None:
    client_ip = client_ip_from_request(
        remote_addr="10.0.0.10",
        forwarded_for="198.51.100.7, 203.0.113.9, 10.0.0.11",
        trusted_proxies=["10.0.0.0/24"],
    )

    assert client_ip == "203.0.113.9"


def test_client_ip_from_request_ignores_spoofed_xff_from_untrusted_client() -> None:
    client_ip = client_ip_from_request(
        remote_addr="198.51.100.44",
        forwarded_for="203.0.113.9",
        trusted_proxies=["10.0.0.0/24"],
    )

    assert client_ip == "198.51.100.44"


def test_client_ip_from_request_falls_back_on_missing_or_invalid_xff() -> None:
    assert (
        client_ip_from_request(
            remote_addr="10.0.0.10",
            forwarded_for=None,
            trusted_proxies=["10.0.0.0/24"],
        )
        == "10.0.0.10"
    )
    assert (
        client_ip_from_request(
            remote_addr="10.0.0.10",
            forwarded_for="not-an-ip",
            trusted_proxies=["10.0.0.0/24"],
        )
        == "10.0.0.10"
    )
