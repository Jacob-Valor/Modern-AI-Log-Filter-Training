"""Tests for configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from logfilter.config import ConfigError, load_config, resolve_env_vars

# Mandatory placeholders in config/config.yaml. A test that reads the shipped
# default config must provide these, because load_config now fails closed when
# they are missing or empty instead of substituting a guessable literal.
MANDATORY_ENV = (
    "ES_PASSWORD",
    "LOGFILTER_API_TOKEN",
    "LOGFILTER_METRICS_TOKEN",
)


@pytest.fixture
def default_config(monkeypatch):
    """Load the shipped config/config.yaml with its mandatory env satisfied."""
    for name in MANDATORY_ENV:
        monkeypatch.setenv(name, "test-value")
    return load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


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


def test_mandatory_placeholder_raises_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_TEST_SECRET", raising=False)

    with pytest.raises(ConfigError, match="LOGFILTER_TEST_SECRET"):
        resolve_env_vars({"v": "${LOGFILTER_TEST_SECRET:?set it first}"})


def test_mandatory_placeholder_raises_when_empty(monkeypatch) -> None:
    """An empty value is a deployment error, not a valid secret."""
    monkeypatch.setenv("LOGFILTER_TEST_SECRET", "")

    with pytest.raises(ConfigError, match="set it first"):
        resolve_env_vars({"v": "${LOGFILTER_TEST_SECRET:?set it first}"})


def test_mandatory_placeholder_resolves_when_set(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_TEST_SECRET", "s3cret")

    assert resolve_env_vars("${LOGFILTER_TEST_SECRET:?set it first}") == "s3cret"


def test_mandatory_placeholder_is_not_replaced_by_its_own_message(monkeypatch) -> None:
    """Regression: ``${VAR:?message}`` used to yield the literal ``?message``.

    That produced a publicly-known Elasticsearch password taken verbatim from
    the tracked config file, so an unset ``ES_PASSWORD`` silently authenticated
    against a guessable literal instead of failing closed.
    """
    monkeypatch.delenv("ES_PASSWORD", raising=False)

    with pytest.raises(ConfigError):
        resolve_env_vars("${ES_PASSWORD:?ES_PASSWORD must be set in .env}")


def test_bare_placeholder_is_mandatory(monkeypatch) -> None:
    monkeypatch.delenv("LOGFILTER_TEST_SECRET", raising=False)

    with pytest.raises(ConfigError):
        resolve_env_vars("${LOGFILTER_TEST_SECRET}")


def test_single_colon_default_keeps_explicit_empty_value(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_TEST_VALUE", "")

    assert resolve_env_vars("${LOGFILTER_TEST_VALUE:fallback}") == ""


def test_double_colon_default_substitutes_for_empty(monkeypatch) -> None:
    monkeypatch.setenv("LOGFILTER_TEST_VALUE", "")

    assert resolve_env_vars("${LOGFILTER_TEST_VALUE:-fallback}") == "fallback"


def test_shipped_config_has_no_unresolved_mandatory_placeholder(default_config) -> None:
    """No resolved value may equal a ``?``-prefixed failure message."""

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
        elif isinstance(node, str):
            yield node

    offenders = [v for v in walk(default_config) if v.startswith("?")]
    assert offenders == []


def _config_flag(value) -> bool:
    """Interpret a config flag that may arrive as a bool or an env-substituted string.

    ``monitoring.novelty.enabled`` is quoted in config.yaml, so after placeholder
    substitution it is the string ``"false"`` rather than a bool. Treating that
    string as truthy would silently re-enable the novelty term in the ceiling
    calculation below.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _max_achievable_score(config) -> float:
    """Highest score the default profile can produce under real runtime rules.

    Mirrors ``LogScorer._compute_score``: ``entity_boost`` is the fixed
    ``entity_boost_value`` (not a probability), and the novelty term contributes
    only while its detector is enabled. Summing the raw novelty weight — as the
    previous version of this test did — overstated the ceiling by 0.15 and hid
    an unreachable ``response.block_threshold``.
    """
    scoring = config["scoring"]
    weights = scoring["weights"]
    novelty_enabled = _config_flag(config["monitoring"]["novelty"]["enabled"])
    return (
        float(weights["classifier"])
        + float(weights["entity_boost"]) * float(scoring["entity_boost_value"])
        + float(weights["cross_encoder"])
        + (float(weights["novelty"]) if novelty_enabled else 0.0)
    )


def test_default_config_novelty_term_is_inert_while_detector_disabled(default_config) -> None:
    """The novelty weight is retained in config, but its detector is off."""
    assert default_config["scoring"]["weights"]["novelty"] == 0.15
    assert not _config_flag(default_config["monitoring"]["novelty"]["enabled"])


def test_default_config_can_route_max_non_sigma_score_as_high(default_config) -> None:
    """HIGH routing must stay reachable at the real maximum score."""
    assert _max_achievable_score(default_config) >= float(
        default_config["scoring"]["routing"]["high"]
    )


def test_default_response_block_threshold_is_reachable(default_config) -> None:
    """The auto-block threshold must sit at or below the achievable maximum.

    Regression test: ``block_threshold`` was 0.85 while the maximum achievable
    score was 0.80, so the armed response engine could never fire.
    """
    assert float(default_config["response"]["block_threshold"]) <= _max_achievable_score(
        default_config
    )
