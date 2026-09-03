#!/usr/bin/env python3
"""Summarize validation-targeted fixed-FPR detection-coverage evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "0.1%": ROOT / "report" / "detection_coverage_fpr_0_1pct" / "summary.json",
    "1%": ROOT / "report" / "detection_coverage_fpr_1pct" / "summary.json",
    "5%": ROOT / "report" / "detection_coverage_fpr_5pct" / "summary.json",
}
OUTPUT_JSON = ROOT / "report" / "detection_coverage_fixed_fpr_summary.json"
OUTPUT_MD = ROOT / "report" / "detection_coverage_fixed_fpr_summary.md"
FUSION_CALIBRATION = ROOT / "models" / "flow" / "fusion_calibration.json"
HTTP_CALIBRATION = ROOT / "models" / "http" / "lexical_calibration.json"
TARGET_KEYS = {"0.1%": "0.001", "1%": "0.01", "5%": "0.05"}
EXPECTED_CUTOFFS = {"cic_ddos": 175002, "cic_bot": 152828, "cic_web": 70002}
REQUIRED_IMPLEMENTATION_FILES = {
    "config/config.yaml",
    "src/logfilter/pipeline/flow_features.py",
    "src/logfilter/pipeline/http_features.py",
    "src/logfilter/pipeline/normalizer.py",
    "src/logfilter/pipeline/scorer.py",
    "src/logfilter/models/flow_classifier.py",
    "src/logfilter/models/bot_flow_classifier.py",
    "src/logfilter/models/flow_fusion.py",
    "training/thresholds.py",
    "training/train_flow_classifier.py",
    "training/train_bot_flow_classifier.py",
    "training/calibrate_flow_fusion.py",
    "training/calibrate_http_lexical.py",
    "scripts/evaluate_public_threat_scenarios.py",
    "scripts/summarize_detection_coverage.py",
    "scripts/model_manifest.py",
}
REQUIRED_DEVELOPMENT_HASH_FILES = {
    "models/flow/development_feature_hashes.txt",
    "models/flow/bot_development_feature_hashes.txt",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read evaluation {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Evaluation is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    runs = {target: _load(path) for target, path in RUNS.items()}
    fusion = _load(FUSION_CALIBRATION)
    http = _load(HTTP_CALIBRATION)
    fusion_sha256 = _sha256(FUSION_CALIBRATION)
    http_sha256 = _sha256(HTTP_CALIBRATION)
    for relative_path, expected_hash in fusion.get("artifacts", {}).items():
        if _sha256(FUSION_CALIBRATION.parent / relative_path) != expected_hash:
            raise SystemExit(f"Fusion model artifact mismatch: {relative_path}")
    source_dataset = fusion.get("source_dataset") or {}
    source_path = ROOT / source_dataset.get("path", "")
    if not source_path.is_file() or _sha256(source_path) != source_dataset.get("sha256"):
        raise SystemExit("Fusion source dataset hash mismatch")
    http_source = http.get("source") or {}
    http_source_path = ROOT / http_source.get("path", "")
    if not http_source_path.is_file() or _sha256(http_source_path) != http_source.get("sha256"):
        raise SystemExit("HTTP calibration source hash mismatch")
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "threshold_policy": (
            "Thresholds selected only from validation data at target FPR; public-test FPR is "
            "reported as observed and is not guaranteed to equal the validation target."
        ),
        "runs": {},
    }
    lines = [
        "# Detection Coverage at Validation-Targeted Fixed FPR",
        "",
        "Targets are calibrated on validation data only. Observed public-test FPR may differ",
        "under temporal/capture shift and was not used to retune thresholds.",
        "",
        "| Validation target | Scenario | Recall | Precision | F1 | Observed test FPR | AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for target, run in runs.items():
        target_key = TARGET_KEYS[target]
        run_metadata = run["run"]
        provenance = run_metadata.get("calibration_provenance") or {}
        flow_provenance = provenance.get("flow") or {}
        http_provenance = provenance.get("http") or {}
        expected_threshold = fusion["fixed_fpr"][target_key]["composite_threshold"]
        expected_http_threshold = http["fixed_fpr"][target_key]["threshold"]
        if run_metadata.get("calibration_target") != target_key:
            raise SystemExit(f"{target}: calibration target provenance mismatch")
        flow_enabled = run_metadata.get("flow_models_enabled")
        if not isinstance(flow_enabled, bool) or not flow_enabled:
            raise SystemExit(f"{target}: flow models were not enabled")
        isolated = run_metadata.get("calibrated_flow_isolation")
        if not isinstance(isolated, bool) or not isolated:
            raise SystemExit(f"{target}: calibrated flow isolation was not enabled")
        if run_metadata.get("evaluation_min_rows") != EXPECTED_CUTOFFS:
            raise SystemExit(f"{target}: held-out cutoffs do not match protocol")
        implementation_hashes = run_metadata.get("implementation_file_hashes") or {}
        if set(implementation_hashes) != REQUIRED_IMPLEMENTATION_FILES:
            raise SystemExit(f"{target}: required implementation provenance is incomplete")
        for relative_path, expected_hash in implementation_hashes.items():
            if _sha256(ROOT / relative_path) != expected_hash:
                raise SystemExit(f"{target}: implementation hash mismatch for {relative_path}")
        development_hashes = run_metadata.get("development_feature_hashes") or {}
        if set(development_hashes) != REQUIRED_DEVELOPMENT_HASH_FILES:
            raise SystemExit(f"{target}: development hash provenance is incomplete")
        for relative_path, expected_hash in development_hashes.items():
            if _sha256(ROOT / relative_path) != expected_hash:
                raise SystemExit(f"{target}: development hash mismatch for {relative_path}")
        datasets = run.get("datasets") or {}
        cic = datasets.get("CIC-IDS2017") or {}
        if _sha256(ROOT / cic.get("path", "")) != cic.get("sha256"):
            raise SystemExit(f"{target}: CIC dataset hash mismatch")
        for filename, metadata in (datasets.get("CSIC-2010") or {}).items():
            if _sha256(ROOT / "demo_data" / "CSIC-2010" / filename) != metadata.get("sha256"):
                raise SystemExit(f"{target}: CSIC dataset hash mismatch for {filename}")
        if flow_provenance.get("sha256") != fusion_sha256:
            raise SystemExit(f"{target}: fusion calibration hash mismatch")
        if http_provenance.get("sha256") != http_sha256:
            raise SystemExit(f"{target}: HTTP calibration hash mismatch")
        hashes_verified = flow_provenance.get("artifact_hashes_verified")
        if not isinstance(hashes_verified, bool) or not hashes_verified:
            raise SystemExit(f"{target}: model artifact hashes were not verified")
        if abs(run_metadata["threshold"] - expected_threshold) > 1e-12:
            raise SystemExit(f"{target}: composite threshold is not calibration-derived")
        scenario_thresholds = run_metadata.get("scenario_thresholds") or {}
        if abs(scenario_thresholds.get("csic_http", -1.0) - expected_http_threshold) > 1e-12:
            raise SystemExit(f"{target}: HTTP threshold is not calibration-derived")
        run_result = {
            "run_id": run["run"]["id"],
            "composite_threshold": run_metadata["threshold"],
            "calibration_provenance": provenance,
            "scenarios": {},
        }
        for scenario, result in run["scenarios"].items():
            metric = result["full_pipeline"]
            selected = {
                key: metric[key]
                for key in (
                    "n_attack",
                    "n_benign",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "recall",
                    "precision",
                    "f1",
                    "false_positive_rate",
                    "auroc",
                    "auprc",
                )
            }
            run_result["scenarios"][scenario] = selected
            lines.append(
                f"| {target} | {scenario} | {selected['recall']:.3f} | "
                f"{selected['precision']:.3f} | {selected['f1']:.3f} | "
                f"{selected['false_positive_rate']:.3f} | {selected['auroc']:.3f} |"
            )
        summary["runs"][target] = run_result
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "- CIC rows are held out after immutable capture-specific row cutoffs.",
            "- Bot is botnet-associated flow telemetry, not live malware execution.",
            "- CSIC uses bounded, credential-redacted structured WAF request telemetry.",
            "- Results are known-family temporal proxies, not genuine zero-day evidence.",
            "- No threshold was changed after inspecting public-test labels.",
            "",
        ]
    )
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {OUTPUT_JSON}")
    print(f"Created {OUTPUT_MD}")


if __name__ == "__main__":
    main()
