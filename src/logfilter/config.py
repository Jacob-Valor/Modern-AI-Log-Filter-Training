"""Configuration loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a mandatory ``${VAR:?message}`` placeholder cannot be resolved.

    Mandatory placeholders exist so that a missing credential fails loudly at
    startup instead of silently becoming a guessable literal. Previously the
    ``:?`` form was mis-parsed as a default, so an unset ``ES_PASSWORD``
    resolved to the literal string ``"?ES_PASSWORD must be set in .env"`` — a
    publicly-known password taken straight from the tracked config file.
    """


_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def _resolve_placeholder(raw: str) -> str:
    """Resolve a single ``${...}`` placeholder body using shell conventions.

    Supported forms:
      ``${VAR}``          mandatory — error when unset or empty
      ``${VAR:?message}`` mandatory — error when unset or empty, with ``message``
      ``${VAR:-default}`` default when unset or empty
      ``${VAR:default}``  default when unset (an explicitly empty value is kept)
    """
    name, sep, rest = raw.partition(":")
    if not sep:
        value = os.environ.get(name, "")
        if not value:
            raise ConfigError(f"{name} is required but is not set")
        return value

    if rest.startswith("?"):
        message = rest[1:].strip() or f"{name} must be set"
        value = os.environ.get(name, "")
        if not value:
            raise ConfigError(f"{name} is required: {message}")
        return value

    value = os.environ.get(name)
    if rest.startswith("-"):
        # "${VAR:-default}" substitutes the default when unset *or* empty.
        return value if value else rest[1:]
    # "${VAR:default}" substitutes only when unset; an explicit "" is kept.
    if value is not None:
        return value
    return rest


def resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${VAR:default}`` and ``${VAR:?message}`` placeholders."""
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _PLACEHOLDER.sub(lambda m: _resolve_placeholder(m.group(1)), obj)
    return obj


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load YAML config and resolve environment placeholders."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return resolve_env_vars(yaml.safe_load(f) or {})
