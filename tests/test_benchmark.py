from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_benchmark_script_imports() -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark", script_path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark"] = module
    spec.loader.exec_module(module)

    assert module.DEFAULT_HOST == "http://localhost:8080"
    assert module.LogFilterBenchmarkUser.single_payload()["source_type"] == "syslog"
