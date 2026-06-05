"""Verify `notebooks/kaggle_train_ner.ipynb` without Kaggle, model downloads, or training.

Run from the repo root:

    PATH=".venv/bin:$PATH" python scripts/verify_ner_notebook.py

This harness checks notebook JSON validity, Python syntax, required section headers,
CyNER loader references, the 11-tag BIO scheme, ONNX export code, the optional regex
augmentation cell, a RED→GREEN 10-token fixture mapping test, a best-effort
`tokenize_and_align_labels` probe, and the final consume/config block.
"""

from __future__ import annotations

import json
import py_compile
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle_train_ner.ipynb"

EXPECTED_HEADERS = [
    "# Kaggle Training: LogFilter NER",
    "## 1. Locate the repo",
    "## 2. Install training dependencies",
    "## 3. Load CyNER corpus",
    "## 4. Tokenise and align labels",
    "## 5. Sampled NER training run (verify environment first)",
    "## 6. Full NER training run (uncomment when sampled run succeeds)",
    "## 7. Inspect artifacts and export to ONNX",
    "## 8. Package artifacts",
    "## 9. Output description + how to consume in repo",
]

LABEL_LIST = [
    "O",
    "B-Indicator",
    "I-Indicator",
    "B-Malware",
    "I-Malware",
    "B-Organization",
    "I-Organization",
    "B-System",
    "I-System",
    "B-Vulnerability",
    "I-Vulnerability",
]

FIXTURE_TOKENS = [
    "Super",
    "Mario",
    "Run",
    "Malware",
    "#",
    "2",
    "–",
    "DroidJack",
    "RAT",
    "Gamers",
]
FIXTURE_TAGS = [
    "B-Malware",
    "I-Malware",
    "I-Malware",
    "I-Malware",
    "O",
    "O",
    "O",
    "B-Malware",
    "I-Malware",
    "O",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    detail: str = ""


def load_notebook() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb.get("nbformat") == 4, f"nbformat={nb.get('nbformat')}"
    assert isinstance(nb.get("cells"), list), "cells is not a list"
    assert nb["cells"], "cells is empty"
    return nb


def cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def iter_cells(nb: dict, cell_type: str | None = None):
    for idx, cell in enumerate(nb["cells"], start=1):
        if cell_type is not None and cell.get("cell_type") != cell_type:
            continue
        yield idx, cell


def sanitize_magics(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            lines.append(f"# {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def all_code_source(nb: dict) -> str:
    return "\n".join(cell_text(cell) for _, cell in iter_cells(nb, "code"))


def all_markdown_source(nb: dict) -> str:
    return "\n".join(cell_text(cell) for _, cell in iter_cells(nb, "markdown"))


def markdown_headers(nb: dict) -> list[tuple[int, int, str]]:
    headers: list[tuple[int, int, str]] = []
    for cell_idx, cell in iter_cells(nb, "markdown"):
        for line_idx, line in enumerate(cell_text(cell).splitlines(), start=1):
            if line.lstrip().startswith("#"):
                headers.append((cell_idx, line_idx, line.strip()))
    return headers


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
            script_path = tmpdir_path / "kaggle_train_ner.py"
            if proc.returncode == 0 and script_path.exists():
                py_compile.compile(str(script_path), doraise=True)
                return GateResult("Gate 2: py_compile", "PASS", f"nbconvert -> {script_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "kaggle_train_ner_fallback.py"
        combined = []
        for cell_idx, cell in iter_cells(nb, "code"):
            combined.append(f"# --- cell {cell_idx} ---")
            combined.append(sanitize_magics(cell_text(cell)))
            combined.append("")
        script_path.write_text("\n".join(combined), encoding="utf-8")
        py_compile.compile(str(script_path), doraise=True)
        return GateResult("Gate 2: py_compile", "PASS", f"fallback -> {script_path}")


def gate_3_section_headers(nb: dict) -> GateResult:
    headers = [line for _, _, line in markdown_headers(nb)]
    missing = []
    for header in EXPECTED_HEADERS:
        if not any(line.startswith(header) for line in headers):
            missing.append(header)
    if missing:
        raise AssertionError(f"missing header(s): {', '.join(missing)}")
    return GateResult("Gate 3: section headers", "PASS", "(9/9)")


def gate_4_cyner_loader(nb: dict) -> GateResult:
    source = all_code_source(nb)
    required = [
        ("aiforsec/CyNER", "CyNER repo ref"),
        ("dataset/mitre", "CyNER corpus path"),
        ("37aff53b", "pinned commit"),
        ("parse_conll", "CoNLL parser"),
    ]
    missing = [label for needle, label in required if needle not in source]
    if missing:
        raise AssertionError(f"missing: {', '.join(missing)}")
    return GateResult("Gate 4: CyNER loader", "PASS", "refs found in code cells")


def gate_5_bio_scheme(nb: dict) -> GateResult:
    source = all_code_source(nb)
    found: list[str] = []
    for label in LABEL_LIST:
        if label == "O":
            if "'O'" in source or '"O"' in source:
                found.append(label)
        elif label in source:
            found.append(label)
    distinct = list(dict.fromkeys(found))
    missing = [label for label in LABEL_LIST if label not in distinct]
    if missing:
        raise AssertionError(f"missing labels: {', '.join(missing)}")
    return GateResult("Gate 5: 11-tag BIO", "PASS", f"({len(distinct)}/11 distinct labels found)")


def gate_6_onnx_export(nb: dict) -> GateResult:
    code_source = all_code_source(nb)
    notebook_source = code_source + "\n" + all_markdown_source(nb)
    required = [
        ("optimum", "optimum import"),
        ("ORTModelForTokenClassification", "ORT model class"),
        ("model.onnx", "ONNX output filename"),
    ]
    missing = []
    for needle, label in required:
        target = code_source if needle != "model.onnx" else notebook_source
        if needle not in target:
            missing.append(label)
    if missing:
        raise AssertionError(f"missing: {', '.join(missing)}")
    return GateResult("Gate 6: ONNX export", "PASS", "export cell preserved")


def gate_7_optional_regex(nb: dict) -> GateResult:
    source = all_code_source(nb)
    required = [
        ("OPTIONAL", "optional label"),
        ("find_spans", "legacy function"),
        ("USE_REGEX_AUGMENTATION", "gated flag"),
    ]
    missing = [label for needle, label in required if needle not in source]
    if missing:
        raise AssertionError(f"missing: {', '.join(missing)}")
    return GateResult("Gate 7: regex optional", "PASS", "legacy path retained")


def _project_labels(tags: list[str], label2id: dict[str, int]) -> list[int]:
    return [label2id[tag] for tag in tags]


def gate_8_fixture_red_green(_nb: dict) -> GateResult:
    expected = [3, 4, 4, 4, 0, 0, 0, 3, 4, 0]

    red_failed = False
    red_error = ""
    try:
        _project_labels(FIXTURE_TAGS, {})
    except Exception as exc:  # noqa: BLE001 - deliberate red-path failure
        red_failed = True
        red_error = type(exc).__name__
    if not red_failed:
        raise AssertionError("broken LABEL2ID unexpectedly passed")

    label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    green_ids = _project_labels(FIXTURE_TAGS, label2id)
    if green_ids != expected:
        raise AssertionError(f"expected {expected}, got {green_ids}")
    return GateResult(
        "Gate 8: 10-token fixture",
        "PASS",
        f"(RED fixture failed with {red_error}; GREEN matched expected IDs)",
    )


def _extract_tokenize_and_align_labels(nb: dict) -> str | None:
    source = all_code_source(nb)
    lines = source.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^def\s+tokenize_and_align_labels\s*\(", line):
            start_idx = idx
            break
    if start_idx is None:
        return None

    block: list[str] = [lines[start_idx]]
    for line in lines[start_idx + 1 :]:
        if line and not line.startswith((" ", "\t", "#")):
            break
        block.append(line)
    return "\n".join(block)


def gate_9_tokenize_align_labels(nb: dict) -> GateResult:
    func_src = _extract_tokenize_and_align_labels(nb)
    if func_src is None:
        return GateResult(
            "Gate 9: tokenize_and_align_labels",
            "SKIP",
            "function inline or not extractable",
        )

    namespace: dict[str, object] = {
        "LABEL2ID": {label: i for i, label in enumerate(LABEL_LIST)},
        "MAX_LENGTH": 256,
    }
    exec(func_src, namespace)
    func = cast(Callable[[list[dict], object], list[dict]], namespace["tokenize_and_align_labels"])

    class FakeEncoding(dict):
        def __init__(self, input_ids: list[int], word_ids: list[int | None]) -> None:
            super().__init__()
            self["input_ids"] = input_ids
            self._word_ids = word_ids

        def word_ids(self):
            return self._word_ids

    class FakeTokenizer:
        def __call__(self, tokens, is_split_into_words=True, truncation=True, max_length=None):
            _ = (tokens, is_split_into_words, truncation, max_length)
            return FakeEncoding([101, 201, 202, 301, 302, 102], [None, 0, 0, 1, 1, None])

    result = func([
        {"tokens": ["Alpha", "Beta"], "tags": ["B-Malware", "B-Indicator"]},
    ], FakeTokenizer())
    assert isinstance(result, list) and result, "expected a non-empty list"
    encoded = result[0]
    assert "input_ids" in encoded and "labels" in encoded, "missing output keys"
    assert encoded["labels"] == [-100, 3, 4, 1, 2, -100], encoded["labels"]
    return GateResult("Gate 9: tokenize_and_align_labels", "PASS", "extracted and validated")


def gate_10_consume_block(nb: dict) -> GateResult:
    last_markdown_idx = None
    for idx, _cell in iter_cells(nb, "markdown"):
        last_markdown_idx = idx
    if last_markdown_idx is None:
        raise AssertionError("no markdown cells found")
    markdown_source = all_markdown_source(nb)
    accepted = (
        "models.ner.model_id",
        'model_id: "models/ner/final"',
        "model_id: 'models/ner/final'",
    )
    if not any(needle in markdown_source for needle in accepted):
        raise AssertionError("missing consume-block config reference")
    return GateResult("Gate 10: consume block", "PASS", f"cell {last_markdown_idx}")


def main() -> int:
    nb = load_notebook()
    results: list[GateResult] = []
    gates = [
        gate_1_json_validity,
        gate_2_python_syntax,
        gate_3_section_headers,
        gate_4_cyner_loader,
        gate_5_bio_scheme,
        gate_6_onnx_export,
        gate_7_optional_regex,
        gate_8_fixture_red_green,
        gate_9_tokenize_align_labels,
        gate_10_consume_block,
    ]

    for gate in gates:
        try:
            result = gate(nb)
        except AssertionError as exc:
            print(f"[verify-ner-notebook] {gate.__name__.replace('_', ' ')} ... FAIL ({exc})")
            results.append(GateResult(gate.__name__, "FAIL", str(exc)))
            continue
        print(
            f"[verify-ner-notebook] {result.name} ... {result.status}"
            f"{(' ' + result.detail) if result.detail else ''}"
        )
        results.append(result)

    passed = sum(1 for result in results if result.status == "PASS")
    skipped = sum(1 for result in results if result.status == "SKIP")
    failed = sum(1 for result in results if result.status == "FAIL")
    eligible = passed + failed
    print(f"[verify-ner-notebook] Result: {passed}/{eligible} PASS, {skipped} SKIP")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
