"""Configuration loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR:default} placeholders in config values."""
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_vars(v) for v in obj]
    if isinstance(obj, str):

        def _replace(match: re.Match) -> str:
            var, _, default = match.group(1).partition(":")
            return os.environ.get(var, default)

        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    return obj


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load YAML config and resolve environment placeholders."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return resolve_env_vars(yaml.safe_load(f) or {})
