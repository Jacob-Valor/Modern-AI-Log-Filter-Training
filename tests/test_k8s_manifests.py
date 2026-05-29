from __future__ import annotations

import pathlib

import pytest
import yaml


def _k8s_manifests() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).parent.parent / "k8s"
    return sorted(root.glob("*.yaml"))


@pytest.mark.parametrize("manifest", _k8s_manifests(), ids=lambda p: p.name)
def test_k8s_manifest_is_valid_yaml(manifest: pathlib.Path) -> None:
    docs = list(yaml.safe_load_all(manifest.read_text()))
    assert docs
    for doc in docs:
        if doc is None:
            continue
        assert "apiVersion" in doc
        assert "kind" in doc
