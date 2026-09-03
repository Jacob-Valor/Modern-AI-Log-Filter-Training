#!/usr/bin/env python3
"""Reject unsafe Kubernetes Secret manifests before deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_KEYS = (
    "LOGFILTER_API_TOKEN",
    "LOGFILTER_METRICS_TOKEN",
    "ES_PASSWORD",
    "QRADAR_GATEWAY_HMAC_SECRET",
    "LOGFILTER_RESPONSE_GATEWAY_HMAC_SECRET",
)
_PLACEHOLDERS = {"", "REPLACE-ME", "changeme", "password"}


def validate_secret(path: Path) -> None:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "Secret":
        raise ValueError("manifest kind must be Secret")
    values = payload.get("stringData")
    if not isinstance(values, dict):
        raise ValueError("manifest must define stringData")

    invalid = [
        key
        for key in _REQUIRED_KEYS
        if not isinstance(values.get(key), str)
        or values[key].strip().lower() in {value.lower() for value in _PLACEHOLDERS}
    ]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"required secret values are missing or placeholders: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        validate_secret(args.manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
