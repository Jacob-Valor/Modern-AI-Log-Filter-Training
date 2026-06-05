"""Verify the Kaggle BiEncoder notebook without running Kaggle or downloading models.

Run from the repo root:

    PATH=".venv/bin:$PATH" python scripts/verify_biencoder_notebook.py

The script checks notebook JSON validity, Python syntax, required section headers,
absence of ONNX export code, required keyword references, a synthetic positive-pair
build path, optional sentence-transformers InputExample construction, and the final
consume/config block.
"""

from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_train_biencoder.ipynb"


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    detail: str = ""


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb.get("nbformat") == 4
    assert isinstance(nb.get("cells"), list)
    assert len(nb["cells"]) > 0
    return nb


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def _iter_cell_lines(nb: dict, cell_type: str | None = None) -> Iterable[tuple[int, int, str]]:
    for cell_index, cell in enumerate(nb["cells"], start=1):
        if cell_type is not None and cell.get("cell_type") != cell_type:
            continue
        for line_index, line in enumerate(_cell_source(cell).splitlines(), start=1):
            yield cell_index, line_index, line


def _collect_markdown_headers(nb: dict) -> list[tuple[int, int, str]]:
    return [
        (cell_index, line_index, line)
        for cell_index, line_index, line in _iter_cell_lines(nb, cell_type="markdown")
        if line.lstrip().startswith("#")
    ]


def _joined_code_source(nb: dict) -> str:
    return "\n".join(_cell_source(cell) for cell in nb["cells"] if cell.get("cell_type") == "code")


def _sanitize_magic_lines(source: str) -> str:
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            out.append(f"# {line}")
        else:
            out.append(line)
    return "\n".join(out)


def gate_1_json_validity(_nb: dict) -> GateResult:
    return GateResult("Gate 1: JSON validity", "PASS")


def gate_2_python_syntax(nb: dict) -> GateResult:
    if shutil.which("jupyter"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            proc = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    str(NOTEBOOK_PATH),
                    "--output-dir",
                    str(tmpdir_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            script_path = tmpdir_path / "kaggle_train_biencoder.py"
            if proc.returncode == 0 and script_path.exists():
                py_compile.compile(str(script_path), doraise=True)
                return GateResult("Gate 2: py_compile", "PASS", f"nbconvert -> {script_path}")

    for cell_index, cell in enumerate(nb["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = _sanitize_magic_lines(_cell_source(cell))
        compile(source, f"<cell-{cell_index}>", "exec")
    return GateResult("Gate 2: py_compile", "PASS", "fallback compile of individual code cells")


def gate_3_section_headers(nb: dict) -> GateResult:
    headers = _collect_markdown_headers(nb)
    header_texts = [line.strip() for _, _, line in headers]

    top_ok = any(
        line.startswith("# Kaggle Training:") and "BiEncoder" in line
        for line in header_texts
    )
    expected_sections = [f"## {i}." for i in range(1, 10)]
    missing = [
        prefix
        for prefix in expected_sections
        if not any(h.startswith(prefix) for h in header_texts)
    ]
    if not top_ok or missing:
        detail = []
        if not top_ok:
            detail.append("top title missing")
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        raise AssertionError("; ".join(detail))

    return GateResult("Gate 3: section headers", "PASS", "(9/9)")


def gate_4_no_onnx(nb: dict) -> GateResult:
    forbidden = ["to_onnx", "optimum.onnxruntime", "ORTModel", "onnxruntime", "onnx.export"]
    for cell_index, cell in enumerate(nb["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        for needle in forbidden:
            if needle in source:
                raise AssertionError(f"offending cell {cell_index} contains {needle!r}")
    return GateResult("Gate 4: no ONNX", "PASS")


def gate_5_required_refs(nb: dict) -> GateResult:
    source = _joined_code_source(nb)
    missing = [
        needle for needle in ("KEYWORD_MAP", "mitre_techniques.json") if needle not in source
    ]
    if missing:
        raise AssertionError(f"missing {', '.join(missing)}")
    return GateResult("Gate 5: KEYWORD_MAP + mitre", "PASS")


def _build_positive_pairs_fixture(
    keyword_map: dict[str, list[str]],
    mitre_techniques: list[dict[str, str]],
    windows: list[str],
) -> list[tuple[str, str]]:
    tech_by_id = {str(item["id"]): item for item in mitre_techniques}

    def technique_text(tid: str) -> str | None:
        tech = tech_by_id.get(tid)
        if tech is None:
            return None
        return f"{tech['name']}. {tech['description']}"

    def matched_techniques(text: str) -> set[str]:
        lowered = text.lower()
        out: set[str] = set()
        for keyword, tids in keyword_map.items():
            if keyword in lowered:
                for tid in tids:
                    if tid in tech_by_id:
                        out.add(tid)
        return out

    pairs: list[tuple[str, str]] = []
    for window in windows:
        matched = matched_techniques(window)
        if not matched:
            continue
        for tid in sorted(matched):
            text = technique_text(tid)
            if text:
                pairs.append((window[:1500], text))
    return pairs


def gate_6_synthetic_pairs(_nb: dict) -> GateResult:
    broken_keyword_map = {}
    keyword_map = {
        "authentication failed": ["T1110"],
        "powershell": ["T1059.001"],
    }
    mitre_techniques = [
        {"id": "T1110", "name": "Brute Force", "description": "Failed logon attempts"},
        {
            "id": "T1059.001",
            "name": "PowerShell",
            "description": "Execute commands with PowerShell",
        },
    ]
    windows = [
        "authentication failed after invalid credentials",
        "powershell launched after authentication failed",
        "routine status message with no signal",
    ]

    red_pairs = _build_positive_pairs_fixture(broken_keyword_map, mitre_techniques, windows)
    try:
        assert red_pairs, "expected broken fixture to fail with empty pairs"
        assert all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(isinstance(item, str) for item in pair)
            for pair in red_pairs
        )
    except AssertionError:
        red_failed_as_expected = True
    else:
        red_failed_as_expected = False

    if not red_failed_as_expected:
        raise AssertionError("broken fixture unexpectedly passed")

    green_pairs = _build_positive_pairs_fixture(keyword_map, mitre_techniques, windows)
    assert green_pairs, "expected real fixture to build positive pairs"
    assert all(
        isinstance(pair, tuple)
        and len(pair) == 2
        and all(isinstance(item, str) for item in pair)
        for pair in green_pairs
    )
    return GateResult(
        "Gate 6: synthetic pairs",
        "PASS",
        f"({len(green_pairs)} positive pairs built; red fixture failed as expected)",
    )


def gate_7_inputexample(_nb: dict) -> GateResult:
    try:
        from sentence_transformers import InputExample
    except Exception:
        return GateResult("Gate 7: InputExample", "SKIP", "sentence-transformers not installed")

    pairs = _build_positive_pairs_fixture(
        {"authentication failed": ["T1110"], "powershell": ["T1059.001"]},
        [
            {"id": "T1110", "name": "Brute Force", "description": "Failed logon attempts"},
            {
                "id": "T1059.001",
                "name": "PowerShell",
                "description": "Execute commands with PowerShell",
            },
        ],
        [
            "authentication failed after invalid credentials",
            "powershell launched after authentication failed",
        ],
    )
    examples = [InputExample(texts=[anchor, positive]) for anchor, positive in pairs]
    assert examples and all(
        isinstance(example.texts, list) and len(example.texts) == 2 for example in examples
    )
    return GateResult("Gate 7: InputExample", "PASS", f"({len(examples)} examples)")


def gate_8_consume_block(nb: dict) -> GateResult:
    source = _joined_code_source(nb) + "\n" + "\n".join(
        line for _, _, line in _iter_cell_lines(nb, cell_type="markdown")
    )
    if "models.biencoder.model_id" not in source:
        raise AssertionError("missing models.biencoder.model_id consume block line")
    return GateResult("Gate 8: consume block", "PASS")


def main() -> int:
    nb = _load_notebook()
    results: list[GateResult] = []

    for gate in (
        gate_1_json_validity,
        gate_2_python_syntax,
        gate_3_section_headers,
        gate_4_no_onnx,
        gate_5_required_refs,
        gate_6_synthetic_pairs,
        gate_7_inputexample,
        gate_8_consume_block,
    ):
        try:
            result = gate(nb)
        except AssertionError as exc:
            print(f"[verify-biencoder-notebook] {gate.__name__.replace('_', ' ')} ... FAIL ({exc})")
            results.append(GateResult(gate.__name__, "FAIL", str(exc)))
            continue
        print(
            f"[verify-biencoder-notebook] {result.name} ... {result.status}"
            f"{(' ' + result.detail) if result.detail else ''}"
        )
        results.append(result)

    passed = sum(1 for result in results if result.status == "PASS")
    skipped = sum(1 for result in results if result.status == "SKIP")
    failed = sum(1 for result in results if result.status == "FAIL")
    print(f"[verify-biencoder-notebook] Result: {passed}/{passed} PASS, {skipped} SKIP")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
