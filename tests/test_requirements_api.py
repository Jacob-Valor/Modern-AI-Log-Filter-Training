"""Dependency guards for the API runtime.

The API image is built from ``uv.lock`` via ``uv export --only-group api``
(``docker/api/Dockerfile``), so the live dependency contract is the ``api``
dependency group in ``pyproject.toml``.

These tests compute the third-party import set from ``src/logfilter/`` instead
of hardcoding a package list, so a new runtime import that nobody declared fails
the suite instead of shipping an ``ImportError`` in production.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_PYPROJECT = _ROOT / "pyproject.toml"

# Import name -> distribution name, where they differ.
_IMPORT_TO_DISTRIBUTION = {
    "faiss": "faiss-cpu",
    "kafka": "kafka-python",
    "prometheus_client": "prometheus-client",
    "sentence_transformers": "sentence-transformers",
    "yaml": "PyYAML",
}

# Modules the API process never imports, though sibling modules in the same
# package do. Each maps to the group that actually installs it.
_EXCLUDED_FROM_API = {
    "click",
    "ipykernel",
    "pandas",
    "rich",
    "sklearn",
    "tqdm",
}


def _distribution_name(module: str) -> str:
    return _IMPORT_TO_DISTRIBUTION.get(module, module)


def _normalise(name: str) -> str:
    return re.split(r"[<>=!~\[\s;]", name, maxsplit=1)[0].strip().lower().replace("_", "-")


def _third_party_imports(root: Path) -> set[str]:
    """Top-level module names imported anywhere under ``root``."""
    found: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    local = {path.name for path in root.iterdir() if path.is_dir()}
    return {
        name
        for name in found
        if name not in sys.stdlib_module_names and name not in local and not name.startswith("_")
    }


def _dependency_groups() -> dict[str, set[str]]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = {
        "project": {_normalise(spec) for spec in data["project"]["dependencies"]}
    }
    for name, specs in data.get("dependency-groups", {}).items():
        groups[name] = {_normalise(spec) for spec in specs}
    for name, specs in data.get("project", {}).get("optional-dependencies", {}).items():
        groups[f"optional:{name}"] = {_normalise(spec) for spec in specs}
    return groups


def _declared_everywhere() -> set[str]:
    return set().union(*_dependency_groups().values())


@pytest.fixture(scope="module")
def api_imports() -> set[str]:
    return _third_party_imports(_SRC) - _EXCLUDED_FROM_API


def test_import_set_is_computed_not_empty(api_imports: set[str]) -> None:
    """Guard the guard: a scanner that finds nothing asserts nothing."""
    assert len(api_imports) >= 10, (
        f"third-party import scan found only {sorted(api_imports)} — "
        "the AST scanner is broken"
    )
    assert "torch" in api_imports
    assert "structlog" in api_imports


def test_project_dependencies_cover_all_runtime_imports(api_imports: set[str]) -> None:
    """Every third-party import must be a declared runtime dependency.

    An undeclared import means ``pip install -e .`` environments run without it.
    """
    declared = _dependency_groups()["project"]
    missing = sorted(
        name for name in api_imports if _normalise(_distribution_name(name)) not in declared
    )

    assert not missing, (
        f"src/logfilter imports {missing} but pyproject [project].dependencies "
        "does not declare them; every runtime import must be declared."
    )


def test_api_dependency_group_covers_api_image_imports(api_imports: set[str]) -> None:
    """The ``api`` group must ship everything the API process imports.

    The Dockerfile installs exactly ``--only-group api``, so a runtime import
    missing from that group raises ImportError inside the image — for example
    ``xgboost``, which ``models/classifier.py`` and ``models/syslog_classifier.py``
    import on the native-model fallback path.
    """
    groups = _dependency_groups()
    assert "api" in groups, "pyproject.toml is missing the [dependency-groups] api group"
    declared = groups["api"]
    missing = sorted(
        name for name in api_imports if _normalise(_distribution_name(name)) not in declared
    )

    assert not missing, (
        f"the API image installs only the 'api' dependency group, which is missing "
        f"{missing}. Either add them to [dependency-groups].api or drop the import."
    )


def test_api_dockerfile_installs_from_the_lockfile() -> None:
    """The API image must install from uv.lock, not a hand-maintained list."""
    dockerfile = (_ROOT / "docker" / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "--only-group api" in dockerfile, (
        "docker/api/Dockerfile must install the 'api' dependency group from the lockfile"
    )
    assert "--locked" in dockerfile or "--frozen" in dockerfile, (
        "docker/api/Dockerfile must pass --locked/--frozen so installs are reproducible"
    )


def test_every_requirements_file_is_actually_consumed() -> None:
    """No inert requirements file may sit at the repository root.

    A requirements file that looks authoritative but that nothing builds from
    is worse than no file at all: the next person to add a dependency edits it
    and ships nothing. Every top-level ``requirements*.txt`` must be referenced
    by a Dockerfile, a Makefile target, or a CI workflow.
    """
    consumers = [
        *sorted((_ROOT / "docker").rglob("Dockerfile*")),
        _ROOT / "Makefile",
        *sorted((_ROOT / ".github" / "workflows").glob("*.yml")),
    ]
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in consumers if path.is_file())

    orphans = [
        path.name
        for path in sorted(_ROOT.glob("requirements*.txt"))
        if path.name not in corpus
    ]

    assert not orphans, (
        f"{orphans} are not referenced by any Dockerfile, Makefile target, or CI "
        "workflow — delete them or wire them into a build"
    )


def test_redis_declared_in_project_and_api_group() -> None:
    """Redis backs the distributed rate limiter.

    ``api/app.py`` imports ``redis`` at runtime. If it is only in one of
    ``[project].dependencies`` (local dev / CI) or the ``api`` group (the image),
    the other environment silently falls back to per-process rate limiting —
    ineffective behind more than one replica.
    """
    groups = _dependency_groups()
    assert "redis" in groups["project"], "pyproject [project].dependencies must declare redis"
    assert "redis" in groups["api"], "the api dependency group must declare redis"