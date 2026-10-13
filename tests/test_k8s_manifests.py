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


def test_dockerfile_hf_cache_path_matches_compose() -> None:
    """B18 regression: the API image and compose must agree on the cache path."""
    root = pathlib.Path(__file__).parent.parent
    dockerfile = (root / "docker" / "api" / "Dockerfile").read_text()
    found = set(_HF_CACHE_PATH_PATTERN.findall(dockerfile))
    assert found, (
        "docker/api/Dockerfile declares no HF cache path matching "
        f"{_HF_CACHE_PATH_PATTERN.pattern} — the guard below cannot fire."
    )
    assert _EXPECTED_HF_CACHE_PATH in found, (
        f"docker/api/Dockerfile must reference {_EXPECTED_HF_CACHE_PATH} for HF cache"
    )
    bad = found - {_EXPECTED_HF_CACHE_PATH}
    assert not bad, f"docker/api/Dockerfile has non-canonical HF cache path(s): {sorted(bad)}."
    compose = (root / "docker-compose.yml").read_text()
    compose_found = set(_HF_CACHE_PATH_PATTERN.findall(compose))
    assert compose_found, "docker-compose.yml declares no HF cache mount path."
    assert _EXPECTED_HF_CACHE_PATH in compose_found, (
        f"docker-compose.yml must mount the hf-cache volume at {_EXPECTED_HF_CACHE_PATH}"
    )
    bad = compose_found - {_EXPECTED_HF_CACHE_PATH}
    assert not bad, f"docker-compose.yml has non-canonical HF cache path(s): {sorted(bad)}."


def test_k8s_manifests_do_not_override_hf_home() -> None:
    """No k8s manifest may point the API at a different HF cache than the image.

    The image bakes ``ENV HF_HOME=/app/hf-cache``; a manifest that rewrites
    HF_HOME (or mounts a differently-named cache) silently sends model
    downloads somewhere the image does not expect. This asserts on the parsed
    YAML rather than a regex over raw text, so it fires on the real setting.
    """
    offenders: list[str] = []
    checked = 0
    for manifest in _k8s_manifests():
        for document in yaml.safe_load_all(_read_manifest(manifest)):
            if not isinstance(document, dict):
                continue
            if document.get("kind") != "Deployment":
                continue
            containers = (
                document.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            )
            for container in containers:
                env = container.get("env") or []
                for item in env:
                    checked += 1
                    if not isinstance(item, dict) or item.get("name") != "HF_HOME":
                        continue
                    value = item.get("value")
                    if value is not None and value != _EXPECTED_HF_CACHE_PATH:
                        offenders.append(f"{manifest.name}: HF_HOME={value!r}")
                for mount in container.get("volumeMounts") or []:
                    mount_path = mount.get("mountPath", "")
                    checked += 1
                    if "hf-cache" in mount_path and mount_path != _EXPECTED_HF_CACHE_PATH:
                        offenders.append(f"{manifest.name}: mountPath={mount_path!r}")

    assert checked, "no k8s Deployment env/volumeMount entries were inspected"
    assert not offenders, (
        f"k8s manifests override the image's HF cache location: {sorted(offenders)}"
    )
