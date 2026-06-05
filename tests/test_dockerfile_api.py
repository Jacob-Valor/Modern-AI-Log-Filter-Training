"""B16 regression: the API Dockerfile must not swallow model-download failure.

A `|| echo "WARNING..."` after download_hf_models.py lets the image build
succeed with missing models, defeating the self-contained-image guarantee.
"""

from __future__ import annotations

import pathlib
import re

_DOCKERFILE = pathlib.Path(__file__).parent.parent / "docker" / "api" / "Dockerfile"


def test_model_download_failure_is_not_swallowed() -> None:
    text = _DOCKERFILE.read_text()
    assert "download_hf_models.py" in text, "expected the HF model download step in the Dockerfile"
    swallow = re.search(r"download_hf_models\.py[\s\S]*?\|\|\s*echo", text)
    assert swallow is None, (
        "Dockerfile must not append '|| echo' to the download_hf_models.py RUN — "
        "a failed model download must fail the build (B16)."
    )
