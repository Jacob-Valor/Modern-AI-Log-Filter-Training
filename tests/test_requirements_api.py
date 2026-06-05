"""B17 regression: the API image must declare xgboost.

classifier.py imports `xgboost` in its ONNX-fallback path, so the API
runtime image must ship it or the fallback raises ImportError in production.
"""

from __future__ import annotations

import pathlib

_REQ = pathlib.Path(__file__).parent.parent / "requirements-api.txt"


def test_xgboost_present_in_api_requirements() -> None:
    lines = _REQ.read_text().splitlines()
    pkgs = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    assert any(p.lower().startswith("xgboost") for p in pkgs), (
        "requirements-api.txt must pin xgboost (classifier.py fallback imports it)"
    )
