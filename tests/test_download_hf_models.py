"""Tests for scripts/download_hf_models.py"""

from __future__ import annotations

from pathlib import Path

import pytest


def _script_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "download_hf_models",
        Path("scripts/download_hf_models.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _script_module()


def test_maybe_empty_returns_none_for_empty():
    assert mod._maybe_empty("") is None
    assert mod._maybe_empty(None) is None
    assert mod._maybe_empty("  ") is None


def test_maybe_empty_returns_string_for_value():
    assert mod._maybe_empty("abc123") == "abc123"
    assert mod._maybe_empty(" main ") == "main"


def test_load_config_reads_yaml_and_substitutes_env():
    cfg = mod._load_config(Path("config/config.yaml"))
    assert "models" in cfg
    assert cfg["models"]["ner"]["model_id"] == "models/ner/final"


def test_main_requires_config_or_defaults():
    # Smoke test: parsing with --help should not crash
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
