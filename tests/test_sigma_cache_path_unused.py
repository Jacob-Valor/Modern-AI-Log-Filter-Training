"""
Defense-in-depth guard for CVE-2025-69872 (diskcache).

``pysigma`` (a transitive dependency) uses ``diskcache`` to cache MITRE
ATT&CK and D3FEND data downloaded from GitHub. The diskcache library uses
Python ``pickle`` for serialization by default, and CVE-2025-69872 (CVSS
9.8) flags that as a remote-code-execution vector if an attacker can write
to the cache directory.

We accept the CVE because our runtime never imports the vulnerable
modules (``sigma.data.mitre_attack`` / ``sigma.data.mitre_d3fend``). This
test enforces that contract by AST-walking the source tree and asserting
the vulnerable imports stay out.

See ``SECURITY.md`` (Vulnerability Exceptions → CVE-2025-69872) for the
full risk assessment and re-review trigger.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "tests")

# Modules that touch diskcache. If you find a legitimate reason to import
# them, document it in SECURITY.md and add the file to ALLOWED_FILES.
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
        f"{vuln_module} is imported by files that should not touch the "
        f"diskcache code path (CVE-2025-69872). Offending files: {offenders}. "
        f"See SECURITY.md (Vulnerability Exceptions → CVE-2025-69872)."
    )


def test_no_runtime_code_loads_mitre_stix_cache() -> None:
    """Runtime src/ must not load the diskcache-backed MITRE data loader.

    The project ships its own MITRE techniques file at
    ``config/mitre_techniques.json``; any code path that pulls MITRE
    STIX data through pysigma re-introduces the diskcache attack surface.
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
                f"{py_file.relative_to(REPO_ROOT)} references '{needle}' "
                f"which routes through the diskcache CVE-2025-69872 code path. "
                f"See SECURITY.md."
            )
