"""Tests for configuration helpers."""

from __future__ import annotations

from logfilter.config import resolve_env_vars


def test_resolve_env_vars_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_TEST_VALUE", raising=False)

    resolved = resolve_env_vars({"value": "${LOGFILTER_TEST_VALUE:fallback}"})

    assert resolved == {"value": "fallback"}


def test_resolve_env_vars_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_TEST_VALUE", "from-env")

    resolved = resolve_env_vars(["${LOGFILTER_TEST_VALUE:fallback}"])

    assert resolved == ["from-env"]
