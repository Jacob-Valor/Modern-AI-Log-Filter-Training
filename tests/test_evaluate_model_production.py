from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import evaluate_model_production as eval_prod

ROOT = Path(__file__).parent.parent


def test_load_labeled_dataset_reads_csv_with_scores(tmp_path: Path) -> None:
    dataset = tmp_path / "logs.csv"
    dataset.write_text("raw,label,score\n'failed login',1,0.9\n'normal heartbeat',0,0.1\n")

    rows = eval_prod.load_labeled_dataset(dataset, score_column="score")

    assert [row.raw for row in rows] == ["'failed login'", "'normal heartbeat'"]
    assert [row.label for row in rows] == [1, 0]
    assert [row.score for row in rows] == [0.9, 0.1]


def test_build_report_for_separable_fixture_has_perfect_metrics() -> None:
    rows = [
        eval_prod.LabeledLog(raw="failure", label=1, score=0.95),
        eval_prod.LabeledLog(raw="benign", label=0, score=0.05),
    ]

    report = eval_prod.build_report(rows, threshold=0.5, min_recall=1.0)

    assert report["samples"] == 2
    assert report["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
    assert report["roc_auc"] == 1.0
    assert report["fp_budget"]["false_positives_per_1000_benign"] == 0.0
    assert report["threshold_sweep"]


def test_build_report_handles_single_class_without_roc_auc() -> None:
    rows = [
        eval_prod.LabeledLog(raw="benign-a", label=0, score=0.1),
        eval_prod.LabeledLog(raw="benign-b", label=0, score=0.2),
    ]

    report = eval_prod.build_report(rows, threshold=0.5, min_recall=1.0)

    assert report["roc_auc"] is None
    assert report["confusion_matrix"] == {"tp": 0, "fp": 0, "tn": 2, "fn": 0}


def test_cli_writes_json_report_from_precomputed_scores(tmp_path: Path) -> None:
    dataset = tmp_path / "logs.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"raw": "failure", "label": 1, "score": 0.9}),
                json.dumps({"raw": "benign", "label": 0, "score": 0.1}),
            ]
        )
    )

    exit_code = eval_prod.main(
        [
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--score-column",
            "score",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["samples"] == 2
    assert payload["f1"] == 1.0


def test_cli_help_imports_cleanly() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_model_production.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_report_requires_scores() -> None:
    with pytest.raises(ValueError, match="score"):
        eval_prod.build_report([eval_prod.LabeledLog(raw="x", label=1, score=None)])
