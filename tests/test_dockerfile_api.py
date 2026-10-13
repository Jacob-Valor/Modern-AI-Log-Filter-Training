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
    for swallow in (
        re.search(r"download_hf_models\.py[\s\S]*?\|\|\s*echo", text),
        re.search(r"download_hf_models\.py[\s\S]*?\|\|\s*true", text),
        re.search(r"download_hf_models\.py[\s\S]*?;\s*exit\s+0", text),
        re.search(r"download_hf_models\.py[\s\S]*?\|\|\s*:\s*$", text, re.MULTILINE),
    ):
        assert swallow is None, (
            "Dockerfile must not append '|| echo'/'|| true'/'; exit 0' to the "
            "download_hf_models.py RUN — a failed model download must fail the "
            f"build (B16). matched: {swallow.group(0)[:80]!r}"
        )


def _logical_run_instructions() -> list[str]:
    """RUN instructions with backslash continuations joined into one line each."""
    logical: list[str] = []
    buffer = ""
    for raw in _DOCKERFILE.read_text().splitlines():
        stripped = raw.strip()
        if not buffer and not stripped.startswith("RUN"):
            continue
        buffer = f"{buffer} {stripped}".strip() if buffer else stripped
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        logical.append(buffer)
        buffer = ""
    return logical


def test_model_download_step_is_its_own_run_instruction() -> None:
    """The download must not carry a failure-swallowing continuation.

    Asserted on logical (backslash-joined) RUN instructions so a multi-line
    step cannot hide its exit status behind a continuation.
    """
    run_lines = [line for line in _logical_run_instructions() if "download_hf_models.py" in line]

    assert run_lines, "no RUN instruction invokes download_hf_models.py"
    for line in run_lines:
        assert "||" not in line, f"download RUN swallows failures: {line!r}"
        assert not re.search(r";\s*exit\s+0", line), f"download RUN swallows failures: {line!r}"
        assert not re.search(r"(^|&&)\s*set\s+\+e", line), (
            f"download RUN disables errexit: {line!r}"
        )
