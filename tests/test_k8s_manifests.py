from __future__ import annotations

import pathlib
import re

import pytest
import yaml

_HF_CACHE_PATH_PATTERN = re.compile(r"/app/(?:[\w.\-]+/)?hf-cache(?![-\w/])")
_EXPECTED_HF_CACHE_PATH = "/app/hf-cache"


def _k8s_manifests() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).parent.parent / "k8s"
    return sorted(root.glob("*.yaml"))


def _read_manifest(path: pathlib.Path) -> str:
    return path.read_text()


@pytest.mark.parametrize("manifest", _k8s_manifests(), ids=lambda p: p.name)
def test_k8s_manifest_is_valid_yaml(manifest: pathlib.Path) -> None:
    docs = list(yaml.safe_load_all(_read_manifest(manifest)))
    assert docs
    for doc in docs:
        if doc is None:
            continue
        assert "apiVersion" in doc
        assert "kind" in doc


@pytest.mark.parametrize("manifest", _k8s_manifests(), ids=lambda p: p.name)
def test_k8s_manifest_hf_cache_path_consistent(manifest: pathlib.Path) -> None:
    """B18 regression: every HF cache path in k8s/ must match the API image."""
    text = _read_manifest(manifest)
    found = set(_HF_CACHE_PATH_PATTERN.findall(text))
    if not found:
        return
    bad = found - {_EXPECTED_HF_CACHE_PATH}
    assert not bad, (
        f"{manifest.name} has non-canonical HF cache path(s): {sorted(bad)}. "
        f"All HF cache paths must equal {_EXPECTED_HF_CACHE_PATH}."
    )


def test_dockerfile_hf_cache_path_matches_k8s() -> None:
    """B18 regression: the API image and k8s manifests must agree on the cache path."""
    root = pathlib.Path(__file__).parent.parent
    dockerfile = (root / "docker" / "api" / "Dockerfile").read_text()
    found = set(_HF_CACHE_PATH_PATTERN.findall(dockerfile))
    assert _EXPECTED_HF_CACHE_PATH in found, (
        f"docker/api/Dockerfile must reference {_EXPECTED_HF_CACHE_PATH} for HF cache"
    )
    bad = found - {_EXPECTED_HF_CACHE_PATH}
    assert not bad, f"docker/api/Dockerfile has non-canonical HF cache path(s): {sorted(bad)}."
    compose = (root / "docker-compose.yml").read_text()
    compose_found = set(_HF_CACHE_PATH_PATTERN.findall(compose))
    assert _EXPECTED_HF_CACHE_PATH in compose_found, (
        f"docker-compose.yml must mount the hf-cache volume at {_EXPECTED_HF_CACHE_PATH}"
    )
    bad = compose_found - {_EXPECTED_HF_CACHE_PATH}
    assert not bad, (
        f"docker-compose.yml has non-canonical HF cache path(s): {sorted(bad)}."
    )
