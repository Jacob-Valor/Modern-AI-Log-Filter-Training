"""Tests for framework-independent API security helpers."""

from __future__ import annotations

from collections import deque

import pytest

from logfilter.api.security import AccessDenied, enforce_rate_limit, require_configured_token


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
