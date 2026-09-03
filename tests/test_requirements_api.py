"""B17 regression: the API image must declare xgboost.

classifier.py imports `xgboost` in its ONNX-fallback path, so the API
runtime image must ship it or the fallback raises ImportError in production.
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
_REQ = _ROOT / "requirements-api.txt"
_PYPROJECT = _ROOT / "pyproject.toml"


def _api_requirement_pkgs() -> list[str]:
    lines = _REQ.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _pyproject_runtime_deps() -> list[str]:
    """Extract package specs from the [project].dependencies array."""
    text = _PYPROJECT.read_text()
    start = text.index("dependencies = [")
    # Closing bracket sits on its own line; a plain ``]`` search would stop at
    # the ``]`` inside specs like ``uvicorn[standard]``.
    end = text.index("\n]", start)
    block = text[start:end]
    return [
        ln.strip().strip(",").strip('"').strip("'")
        for ln in block.splitlines()[1:]
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_xgboost_present_in_api_requirements() -> None:
    pkgs = _api_requirement_pkgs()
    assert any(p.lower().startswith("xgboost") for p in pkgs), (
        "requirements-api.txt must pin xgboost (classifier.py fallback imports it)"
    )


def test_redis_declared_in_pyproject_and_api_requirements() -> None:
    """Redis backs the distributed rate limiter.

    ``api/app.py:_configure_rate_limiter`` imports ``redis`` at runtime. It must
    be a declared runtime dependency in BOTH ``pyproject.toml`` (used by
    ``pip install -e .`` for local dev, CI, and the pytest venv) and
    ``requirements-api.txt`` (used by the API Docker image). If it is only in
    one, the other environment silently falls back to per-process rate limiting
    — ineffective behind more than one replica.
    """
    api_pkgs = _api_requirement_pkgs()
    assert any(p.lower().startswith("redis") for p in api_pkgs), (
        "requirements-api.txt must pin redis (api/app.py imports it for rate limiting)"
    )

    runtime_deps = _pyproject_runtime_deps()
    assert any(d.lower().startswith("redis") for d in runtime_deps), (
        "pyproject.toml [project].dependencies must declare redis so "
        "`pip install -e .` environments get the distributed rate limiter, "
        "matching requirements-api.txt (drift otherwise degrades to in-memory)"
    )
