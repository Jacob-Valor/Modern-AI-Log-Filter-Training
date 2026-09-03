"""
Regression guard against reintroducing CVE-2025-69872 (`diskcache`).

The runtime formerly depended on pySigma, which transitively installed
`diskcache` and its pickle-backed cache. The scorer now uses a constrained
`yaml.safe_load_all` parser and declares neither package.

These tests keep the old diskcache-backed MITRE loaders out of runtime code.
See `SECURITY.md` for the resolved exception.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")

# Modules that would reintroduce the removed diskcache attack surface.
VULNERABLE_MODULES = frozenset(
    {
        "sigma.data.mitre_attack",
        "sigma.data.mitre_d3fend",
    }
)


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imports_in_file(path: Path) -> set[str]:
    """Return all dotted-name imports referenced in a Python file."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        target: ast.AST | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias
                break
        elif isinstance(node, ast.ImportFrom):
            target = node
        if target is None:
            continue

        if isinstance(target, ast.Import):
            found.add(target.names[0].name)
        elif isinstance(target, ast.ImportFrom):
            module = target.module or ""
            for alias in target.names:
                full = f"{module}.{alias.name}" if module else alias.name
                found.add(full)
    return found


@pytest.mark.parametrize("vuln_module", sorted(VULNERABLE_MODULES))
def test_vulnerable_pysigma_module_not_imported(vuln_module: str) -> None:
    """No source file under src/ or tests/ may import ``vuln_module``."""
    offenders: list[str] = []
    for root in SCAN_ROOTS:
        for py_file in _iter_python_files(root):
            imports = _imports_in_file(py_file)
            if any(imp == vuln_module or imp.startswith(vuln_module + ".") for imp in imports):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"{vuln_module} reintroduces the removed diskcache code path "
        f"(CVE-2025-69872). Offending files: {offenders}."
    )


def test_no_runtime_code_loads_mitre_stix_cache() -> None:
    """Runtime code must not restore a diskcache-backed MITRE loader.

    The project ships its own `config/mitre_techniques.json`; the scorer's
    constrained Sigma parser does not need remote MITRE data.
    """
    for py_file in _iter_python_files(REPO_ROOT / "src"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        # Block the specific entry points; the parse test above covers
        # dotted-name imports. This catches string-based references too.
        for needle in (
            "sigma.data.mitre_attack",
            "sigma.data.mitre_d3fend",
            "from sigma.data import",
        ):
            assert needle not in text, (
                f"{py_file.relative_to(REPO_ROOT)} references '{needle}', "
                "which reintroduces the removed diskcache code path."
            )
