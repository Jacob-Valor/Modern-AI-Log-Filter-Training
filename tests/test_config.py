"""Tests for configuration helpers."""

from __future__ import annotations

from logfilter.config import load_config, resolve_env_vars


def test_resolve_env_vars_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_TEST_VALUE", raising=False)

    resolved = resolve_env_vars({"value": "${LOGFILTER_TEST_VALUE:fallback}"})

    assert resolved == {"value": "fallback"}


def test_resolve_env_vars_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_TEST_VALUE", "from-env")

    resolved = resolve_env_vars(["${LOGFILTER_TEST_VALUE:fallback}"])

    assert resolved == ["from-env"]


def test_load_config_returns_empty_for_missing_file(tmp_path) -> None:
    assert load_config(tmp_path / "missing.yaml") == {}


def test_load_config_reads_yaml_and_resolves_environment(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("value: ${LOGFILTER_TEST_VALUE:fallback}\n")
    monkeypatch.setenv("LOGFILTER_TEST_VALUE", "from-env")

    assert load_config(path) == {"value": "from-env"}


def test_resolve_env_vars_leaves_non_string_scalars() -> None:
    assert resolve_env_vars(3) == 3
