"""T10 regression: build reproducibility for Dockerfiles and ONNX export.

Verifies:
- All Dockerfiles use uv with lockfile for pinned, reproducible installs
- All Dockerfiles pin base images to SHA256 digests (no floating tags)
- ONNX export in train_transformer.py hard-fails when optimum is missing
"""

from __future__ import annotations

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
_DOCKERFILES = [
    _ROOT / "docker" / "api" / "Dockerfile",
    _ROOT / "docker" / "collector" / "Dockerfile",
    _ROOT / "docker" / "router" / "Dockerfile",
    _ROOT / "docker" / "archive" / "Dockerfile",
]
_TRAIN_TRANSFORMER = _ROOT / "training" / "train_transformer.py"


# ── ONNX export hard-fail ──────────────────────────────────────────────


class TestOnnxExportHardFail:
    """export_onnx must raise RuntimeError when optimum is not installed."""

    def test_export_onnx_raises_on_import_error(self, tmp_path: pathlib.Path) -> None:
        """If optimum cannot be imported, export_onnx must raise, not warn."""
        import importlib
        import sys
        from unittest.mock import patch

        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "optimum.onnxruntime":
                raise ImportError("No module named 'optimum.onnxruntime'")
            return real_import(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=fake_import):
            if "training.train_transformer" in sys.modules:
                del sys.modules["training.train_transformer"]
            sys.path.insert(0, str(_ROOT))
            try:
                from training.train_transformer import export_onnx

                with pytest.raises((RuntimeError, ImportError)):
                    export_onnx(model_dir=tmp_path, output_path=tmp_path / "model.onnx")
            finally:
                sys.path.pop(0)
                if "training.train_transformer" in sys.modules:
                    del sys.modules["training.train_transformer"]

    def test_export_onnx_no_silent_skip(self) -> None:
        """export_onnx must not contain a 'skipping ONNX export' warning path."""
        source = _TRAIN_TRANSFORMER.read_text()
        assert "skipping ONNX export" not in source, (
            "export_onnx must not silently skip ONNX export — "
            "it must raise RuntimeError when optimum is unavailable"
        )


# ── Dockerfile: uv lockfile ────────────────────────────────────────────


class TestDockerfilesUseUvLockfile:
    """All Dockerfiles must use uv with --locked for reproducible installs."""

    @pytest.mark.parametrize(
        "dockerfile",
        _DOCKERFILES,
        ids=[str(p.relative_to(_ROOT)) for p in _DOCKERFILES],
    )
    def test_dockerfile_uses_uv(self, dockerfile: pathlib.Path) -> None:
        text = dockerfile.read_text()
        assert "uv " in text or "uv\t" in text or "/uv" in text, (
            f"{dockerfile.name} must install and use uv for reproducible builds"
        )

    @pytest.mark.parametrize(
        "dockerfile",
        _DOCKERFILES,
        ids=[str(p.relative_to(_ROOT)) for p in _DOCKERFILES],
    )
    def test_dockerfile_copies_uv_lock(self, dockerfile: pathlib.Path) -> None:
        text = dockerfile.read_text()
        assert "uv.lock" in text, (
            f"{dockerfile.name} must COPY uv.lock into the image for locked installs"
        )

    @pytest.mark.parametrize(
        "dockerfile",
        _DOCKERFILES,
        ids=[str(p.relative_to(_ROOT)) for p in _DOCKERFILES],
    )
    def test_dockerfile_no_bare_pip_install(self, dockerfile: pathlib.Path) -> None:
        """Dockerfiles must not use bare 'pip install' — use uv instead."""
        text = dockerfile.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Allow pip only if uv is driving it (uv pip install) or in a comment
            assert not re.match(r"^RUN\s+pip\s+install", stripped), (
                f"{dockerfile.name} uses bare 'pip install' on line: {stripped!r} — "
                "use 'uv pip install --locked' for reproducible builds"
            )


# ── Dockerfile: pinned base images ─────────────────────────────────────


class TestDockerfilesPinBaseImages:
    """All Dockerfiles must pin base images to SHA256 digests."""

    @pytest.mark.parametrize(
        "dockerfile",
        _DOCKERFILES,
        ids=[str(p.relative_to(_ROOT)) for p in _DOCKERFILES],
    )
    def test_base_image_has_digest(self, dockerfile: pathlib.Path) -> None:
        text = dockerfile.read_text()
        # Find FROM lines (not multi-stage builder aliases)
        from_lines = [ln for ln in text.splitlines() if ln.strip().upper().startswith("FROM ")]
        assert from_lines, f"{dockerfile.name} has no FROM instruction"

        for from_line in from_lines:
            # Must contain @sha256: to indicate a pinned digest
            assert "@sha256:" in from_line, (
                f"{dockerfile.name} FROM line is not pinned to a digest: {from_line.strip()!r} — "
                "use 'image:tag@sha256:...' for reproducible builds"
            )


# ── uv.lock file exists ────────────────────────────────────────────────


class TestUvLockExists:
    """The repository root must contain a uv.lock file."""

    def test_uv_lock_exists(self) -> None:
        assert (_ROOT / "uv.lock").exists(), "uv.lock must exist in the project root"
