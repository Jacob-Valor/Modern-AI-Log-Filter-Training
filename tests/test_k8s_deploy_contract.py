from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
K8S_DIR = ROOT / "k8s"


def _make_target(name: str) -> tuple[str, list[str]]:
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{name}:"):
            recipe: list[str] = []
            for candidate in lines[index + 1 :]:
                if candidate and not candidate.startswith(("\t", " ")):
                    break
                if candidate.startswith("\t"):
                    recipe.append(candidate.strip())
            return line, recipe
    raise AssertionError(f"Make target {name!r} not found")


def _k8s_apply_paths() -> list[str]:
    _, recipe = _make_target("k8s-apply")
    return [
        match.group(1)
        for line in recipe
        if (match := re.search(r"kubectl apply -f (\S+)", line))
    ]


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def _collect_secret_refs(node: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(node, dict):
        secret_ref = node.get("secretKeyRef")
        if isinstance(secret_ref, dict) and secret_ref.get("name") == "logfilter-secrets":
            key = secret_ref.get("key")
            if isinstance(key, str):
                refs.add(key)
        for value in node.values():
            refs.update(_collect_secret_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.update(_collect_secret_refs(item))
    return refs


def test_k8s_apply_uses_declarative_secret() -> None:
    header, recipe = _make_target("k8s-apply")

    assert "k8s-secrets" not in header
    assert "k8s/secret.yaml" in _k8s_apply_paths()
    assert "kubectl create secret" not in "\n".join(recipe)


def test_k8s_apply_applies_namespace_before_secret() -> None:
    applied = _k8s_apply_paths()

    assert applied[0] == "k8s/namespace.yaml"
    assert applied[1] == "k8s/secret.yaml"


def test_secret_manifest_covers_all_logfilter_secret_refs() -> None:
    secret_docs = _load_yaml_documents(K8S_DIR / "secret.yaml")
    secret_keys = set(secret_docs[0]["stringData"])
    referenced_keys: set[str] = set()
    for manifest in K8S_DIR.glob("*.yaml"):
        for document in _load_yaml_documents(manifest):
            referenced_keys.update(_collect_secret_refs(document))

    assert referenced_keys <= secret_keys


def test_no_init_containers_in_api_manifests() -> None:
    for manifest in K8S_DIR.glob("*.yaml"):
        for document in _load_yaml_documents(manifest):
            if document.get("kind") == "Deployment":
                spec = document.get("spec", {}).get("template", {}).get("spec", {})
                assert "initContainers" not in spec, f"{manifest.name} has initContainers"
