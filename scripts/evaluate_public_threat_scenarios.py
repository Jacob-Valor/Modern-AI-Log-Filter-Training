#!/usr/bin/env python3
"""Evaluate the deployed pipeline on label-blind public threat scenarios.

The evaluator keeps ground truth outside model input and writes immutable per-event
predictions plus a machine-readable summary. It covers:

* CSIC 2010 HTTP requests adapted to Apache combined access-log records;
* CIC-IDS2017 DDoS-labelled flow summaries;
* CIC-IDS2017 Web Attack flow summaries;
* CIC-IDS2017 Bot-labelled (botnet-associated) flow summaries.

CIC flow rows are serialized as neutral CEF observations. Ground-truth labels,
attack categories, MITRE mappings, and label-derived severities are never exposed
to the normalizer or scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import subprocess
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from logfilter.config import load_config
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CIC_ZIP = ROOT / "demo_data" / "CIC-IDS2017" / "MachineLearningCSV.zip"
DEFAULT_CSIC_DIR = ROOT / "demo_data" / "CSIC-2010"
DEFAULT_OUTPUT = ROOT / "report" / "public_scenario_evaluation"
DEFAULT_DEVELOPMENT_HASH_PATHS = (
    ROOT / "models" / "flow" / "development_feature_hashes.txt",
    ROOT / "models" / "flow" / "bot_development_feature_hashes.txt",
)
IMPLEMENTATION_PROVENANCE_FILES = (
    ROOT / "config" / "config.yaml",
    ROOT / "src" / "logfilter" / "pipeline" / "flow_features.py",
    ROOT / "src" / "logfilter" / "pipeline" / "http_features.py",
    ROOT / "src" / "logfilter" / "pipeline" / "normalizer.py",
    ROOT / "src" / "logfilter" / "pipeline" / "scorer.py",
    ROOT / "src" / "logfilter" / "models" / "flow_classifier.py",
    ROOT / "src" / "logfilter" / "models" / "bot_flow_classifier.py",
    ROOT / "src" / "logfilter" / "models" / "flow_fusion.py",
    ROOT / "training" / "thresholds.py",
    ROOT / "training" / "train_flow_classifier.py",
    ROOT / "training" / "train_bot_flow_classifier.py",
    ROOT / "training" / "calibrate_flow_fusion.py",
    ROOT / "training" / "calibrate_http_lexical.py",
    ROOT / "scripts" / "evaluate_public_threat_scenarios.py",
    ROOT / "scripts" / "summarize_detection_coverage.py",
    ROOT / "scripts" / "model_manifest.py",
)

REQUEST_START = re.compile(r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|TRACE|CONNECT)\s+", re.I)
CIC_LABEL_KEY = " Label"
CIC_FILE_BY_SCENARIO = {
    "cic_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "cic_web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "cic_bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
}
CIC_ATTACK_LABELS = {
    "cic_ddos": {"DDoS"},
    "cic_web": {
        "Web Attack  Brute Force",
        "Web Attack  XSS",
        "Web Attack  Sql Injection",
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection",
    },
    "cic_bot": {"Bot"},
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    scenario: str
    source_file: str
    row_number: int
    exact_label: str
    family: str
    is_attack: bool
    model_input: str
    adapter: str


@dataclass
class Prediction:
    sample_id: str
    scenario: str
    source_file: str
    row_number: int
    exact_label: str
    family: str
    is_attack: bool
    adapter: str
    model_input_sha256: str
    score: float
    predicted: bool
    threshold: float
    priority: str
    sigma_matched: bool
    sigma_rule_ids: list[str]
    classifier_score: float
    novelty_score: float
    duplicate: bool
    dedup_penalty: float
    mitre_suggestion: str
    scoring_latency_ms: float
    scoring_error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cic-zip", type=Path, default=DEFAULT_CIC_ZIP)
    parser.add_argument("--csic-dir", type=Path, default=DEFAULT_CSIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit composite threshold; cannot be combined with --calibration-target",
    )
    parser.add_argument(
        "--calibration-target",
        choices=("0.001", "0.01", "0.05"),
        default=None,
        help="Load a validation-calibrated final flow threshold and attested Bot target",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--ddos-min-row",
        type=int,
        default=175002,
        help="First CIC DDoS CSV row eligible for evaluation (held-out default: 175002)",
    )
    parser.add_argument(
        "--bot-min-row",
        type=int,
        default=152828,
        help="First CIC Bot CSV row eligible for evaluation (held-out default: 152828)",
    )
    parser.add_argument(
        "--web-min-row",
        type=int,
        default=70002,
        help="First CIC Web CSV row eligible for evaluation (held-out default: 70002)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--enable-flow-models",
        action="store_true",
        help="Enable manifest-verified general and Bot flow ONNX heads",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["csic_http", "cic_ddos", "cic_web", "cic_bot"]
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_feature_hashes(paths: tuple[Path, ...]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as exc:
            raise RuntimeError(f"Cannot read development feature hashes: {path}") from exc
        for line in lines:
            value = line.strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"Invalid feature hash in {path}")
            hashes.add(value)
    return hashes


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def clean_number(row: dict[str, str], key: str) -> tuple[str, bool]:
    raw = (row.get(key) or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "0", False
    if not math.isfinite(value):
        return "0", False
    return format(value, ".8g"), True


def clean_cef_token(row: dict[str, str], key: str) -> tuple[str, bool]:
    value = str(row.get(key, "")).strip()
    if not value or len(value) > 128 or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        return "unknown", False
    return value, True


CIC_FLOW_FIELDS = (
    ("dpt", " Destination Port"),
    ("duration", " Flow Duration"),
    ("fwdPackets", " Total Fwd Packets"),
    ("bwdPackets", " Total Backward Packets"),
    ("fwdBytes", "Total Length of Fwd Packets"),
    ("bwdBytes", " Total Length of Bwd Packets"),
    ("flowBytesPerSec", "Flow Bytes/s"),
    ("flowPacketsPerSec", " Flow Packets/s"),
    ("packetLengthMean", " Packet Length Mean"),
    ("packetLengthStd", " Packet Length Std"),
    ("synCount", " SYN Flag Count"),
    ("ackCount", " ACK Flag Count"),
    ("rstCount", " RST Flag Count"),
    ("activeMean", "Active Mean"),
    ("idleMean", "Idle Mean"),
    ("finCount", " FIN Flag Count"),
    ("pshCount", " PSH Flag Count"),
    ("urgCount", " URG Flag Count"),
    ("flowIatMean", " Flow IAT Mean"),
    ("flowIatStd", " Flow IAT Std"),
    ("flowIatMax", " Flow IAT Max"),
    ("flowIatMin", " Flow IAT Min"),
    ("fwdIatMean", " Fwd IAT Mean"),
    ("fwdIatStd", " Fwd IAT Std"),
    ("bwdIatMean", " Bwd IAT Mean"),
    ("bwdIatStd", " Bwd IAT Std"),
    ("packetLengthMin", " Min Packet Length"),
    ("packetLengthMax", " Max Packet Length"),
    ("fwdPacketLengthMean", " Fwd Packet Length Mean"),
    ("bwdPacketLengthMean", " Bwd Packet Length Mean"),
    ("initWinFwd", "Init_Win_bytes_forward"),
    ("initWinBwd", " Init_Win_bytes_backward"),
    ("downUpRatio", " Down/Up Ratio"),
)


def cic_flow_feature_hash(row: dict[str, str]) -> str:
    values: list[float] = []
    for _output_name, source_name in CIC_FLOW_FIELDS:
        raw = (row.get(source_name) or "").strip()
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value) or value < 0.0:
            value = 0.0
        values.append(min(value, 3.4028235e38))
    vector = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def neutral_cic_cef(row: dict[str, str]) -> tuple[str, int]:
    """Serialize observed CICFlowMeter values without ground-truth-derived fields."""
    invalid = 0
    source_ip, source_valid = clean_cef_token(row, " Source IP")
    destination_ip, destination_valid = clean_cef_token(row, " Destination IP")
    protocol, protocol_valid = clean_number(row, " Protocol")
    invalid += sum(not valid for valid in (source_valid, destination_valid, protocol_valid))
    extensions = [
        "act=observed",
        f"src={source_ip}",
        f"dst={destination_ip}",
        f"proto={protocol}",
    ]
    for output_name, source_name in CIC_FLOW_FIELDS:
        value, valid = clean_number(row, source_name)
        invalid += not valid
        extensions.append(f"{output_name}={value}")

    # Constant header values are adapter metadata, not inferred threat attributes.
    line = (
        "<134>Jan 01 00:00:00 flow-sensor "
        "CEF:0|UNB|CICFlowMeter|1.0|FLOW|Network flow observation|1|" + "\t".join(extensions)
    )
    return line, invalid


def reservoir_add(
    reservoirs: dict[str, list[tuple[int, dict[str, str]]]],
    seen: dict[str, int],
    label: str,
    item: tuple[int, dict[str, str]],
    limit: int,
    rng: random.Random,
) -> None:
    seen[label] += 1
    bucket = reservoirs[label]
    if len(bucket) < limit:
        bucket.append(item)
        return
    replacement = rng.randrange(seen[label])
    if replacement < limit:
        bucket[replacement] = item


def find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CIC member ending with {suffix!r}, found {matches}")
    return matches[0]


def cic_samples(
    archive_path: Path,
    scenario: str,
    per_class: int,
    seed: int,
    *,
    ddos_min_row: int = 175002,
    bot_min_row: int = 152828,
    web_min_row: int = 70002,
    excluded_feature_hashes: set[str] | None = None,
) -> tuple[list[Sample], dict[str, int]]:
    target_suffix = CIC_FILE_BY_SCENARIO[scenario]
    accepted_attacks = CIC_ATTACK_LABELS[scenario]
    rng = random.Random(seed + sum(ord(char) for char in scenario))
    reservoirs: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    audit = defaultdict(int)
    minimum_row = {
        "cic_ddos": ddos_min_row,
        "cic_bot": bot_min_row,
        "cic_web": web_min_row,
    }[scenario]

    with zipfile.ZipFile(archive_path) as archive:
        member = find_zip_member(archive, target_suffix)
        member_path = PurePosixPath(member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe archive member path: {member}")
        with archive.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)
            for row_number, row in enumerate(reader, start=2):
                audit["rows_read"] += 1
                if row_number < minimum_row:
                    audit["rows_before_evaluation_cutoff"] += 1
                    continue
                if (
                    excluded_feature_hashes
                    and cic_flow_feature_hash(row) in excluded_feature_hashes
                ):
                    audit["rows_exactly_seen_in_development_excluded"] += 1
                    continue
                label = (row.get(CIC_LABEL_KEY) or row.get("Label") or "").strip()
                if label == "BENIGN":
                    reservoir_add(reservoirs, seen, label, (row_number, row), per_class, rng)
                elif label in accepted_attacks:
                    reservoir_add(reservoirs, seen, label, (row_number, row), per_class, rng)
                else:
                    audit["rows_other_label"] += 1

    samples: list[Sample] = []
    for label, rows in reservoirs.items():
        is_attack = label != "BENIGN"
        family = (
            {
                "cic_ddos": "ddos",
                "cic_web": "web_attack",
                "cic_bot": "botnet_associated",
            }[scenario]
            if is_attack
            else "benign"
        )
        for row_number, row in rows:
            model_input, invalid_values = neutral_cic_cef(row)
            audit["invalid_or_nonfinite_values_replaced"] += invalid_values
            samples.append(
                Sample(
                    sample_id=stable_id("CIC-IDS2017", target_suffix, row_number),
                    scenario=scenario,
                    source_file=target_suffix,
                    row_number=row_number,
                    exact_label=label,
                    family=family,
                    is_attack=is_attack,
                    model_input=model_input,
                    adapter="sanitized_cef_flow_v1",
                )
            )

    audit["rows_selected"] = len(samples)
    audit["available_benign"] = seen["BENIGN"]
    audit["available_attack"] = sum(seen[label] for label in accepted_attacks)
    return samples, dict(audit)


def iter_csic_requests(path: Path) -> Iterable[tuple[int, str]]:
    current: list[str] = []
    request_number = 0
    with path.open(encoding="latin-1", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if REQUEST_START.match(line):
                if current:
                    request_number += 1
                    yield request_number, "\n".join(current)
                current = [line]
            elif current:
                current.append(line)
    if current:
        request_number += 1
        yield request_number, "\n".join(current)


def csic_to_waf(request: str) -> str | None:
    """Convert one CSIC request to label-independent structured WAF telemetry."""
    lines = request.splitlines()
    if not lines:
        return None
    parts = lines[0].split(" ", 2)
    if len(parts) != 3:
        return None
    method, target, _protocol = parts
    headers: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False
    for line in lines[1:]:
        if not in_body and not line.strip():
            in_body = True
            continue
        if in_body:
            body_lines.append(line)
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip()] = value.strip()
    envelope = {
        "event_type": "csic_waf_request",
        "source_ip": "192.0.2.1",
        "request": {
            "clientIp": "192.0.2.1",
            "method": method.upper(),
            "uri": target,
            "headers": headers,
            "body": "\n".join(body_lines),
        },
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reservoir_requests(
    path: Path, limit: int, rng: random.Random
) -> tuple[list[tuple[int, str]], int]:
    selected: list[tuple[int, str]] = []
    seen = 0
    for request_number, request in iter_csic_requests(path):
        waf_event = csic_to_waf(request)
        if waf_event is None:
            continue
        seen += 1
        if len(selected) < limit:
            selected.append((request_number, waf_event))
            continue
        replacement = rng.randrange(seen)
        if replacement < limit:
            selected[replacement] = (request_number, waf_event)
    return selected, seen


def csic_samples(directory: Path, per_class: int, seed: int) -> tuple[list[Sample], dict[str, int]]:
    normal_path = directory / "normalTrafficTest.txt"
    attack_path = directory / "anomalousTrafficTest.txt"
    rng = random.Random(seed)
    normal, normal_available = reservoir_requests(normal_path, per_class, rng)
    attack, attack_available = reservoir_requests(attack_path, per_class, rng)
    samples: list[Sample] = []
    for path, rows, label, is_attack in (
        (normal_path, normal, "normal", False),
        (attack_path, attack, "anomalous", True),
    ):
        for request_number, model_input in rows:
            samples.append(
                Sample(
                    sample_id=stable_id("CSIC-2010", path.name, request_number),
                    scenario="csic_http",
                    source_file=path.name,
                    row_number=request_number,
                    exact_label=label,
                    family="web_anomalous" if is_attack else "benign",
                    is_attack=is_attack,
                    model_input=model_input,
                    adapter="csic_request_to_structured_waf_v2",
                )
            )
    return samples, {
        "available_benign": normal_available,
        "available_attack": attack_available,
        "rows_selected": len(samples),
        "post_body_included_with_bounded_redaction": True,
        "source_ip_is_label_independent_constant": True,
    }


def deduplicate_samples(samples: list[Sample], seed: int) -> tuple[list[Sample], int, int]:
    """Select one canonical input representative without using ground truth."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.model_input].append(sample)

    deduplicated: list[Sample] = []
    conflicting_groups = 0
    for matches in grouped.values():
        ordered = sorted(matches, key=lambda sample: sample.sample_id)
        deduplicated.append(ordered[0])
        if len({sample.is_attack for sample in matches}) > 1:
            conflicting_groups += 1

    random.Random(seed).shuffle(deduplicated)
    removed = len(samples) - len(deduplicated)
    return deduplicated, removed, conflicting_groups


def assert_label_blind(sample: Sample, normalized_text: str, fields: dict[str, Any]) -> None:
    visible = (
        f"{sample.model_input}\n{normalized_text}\n{json.dumps(fields, sort_keys=True)}".lower()
    )
    forbidden = {
        "cic_ddos": ("ddos",),
        "cic_web": ("web attack", "sql injection", "xss", "brute force"),
        "cic_bot": ("botnet", " bot ", "cat=bot", "act=bot"),
        "csic_http": ("label=anomalous", "attack=true"),
    }[sample.scenario]
    leaked = [term for term in forbidden if term in visible]
    if leaked:
        raise RuntimeError(f"Ground-truth leakage for {sample.sample_id}: {leaked}")


def load_fusion_calibration(target: str) -> tuple[float, dict[str, Any]]:
    path = ROOT / "models" / "flow" / "fusion_calibration.json"
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
        row = calibration["fixed_fpr"][target]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Cannot load final flow fusion calibration") from exc
    if row.get("status") != "calibrated" or row.get("composite_threshold") is None:
        raise ValueError(f"Flow fusion target {target} is not calibrated")
    for relative_path, expected_hash in calibration.get("artifacts", {}).items():
        artifact_path = path.parent / relative_path
        actual_hash = sha256_path(artifact_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Flow calibration artifact hash mismatch: {relative_path}")
    threshold = safe_float(row["composite_threshold"], default=-1.0)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Calibrated flow fusion threshold is invalid")
    return threshold, {
        "target_fpr": target,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_path(path),
        "artifact_hashes_verified": True,
        "validation_fpr": row["validation_fpr"],
        "validation_recall": row["validation_recall"],
        "general_probability_threshold": row["general_probability_threshold"],
        "bot_probability_threshold": row["bot_probability_threshold"],
        "alert_floor": row["alert_floor"],
    }


def load_http_calibration(target: str) -> tuple[float, dict[str, Any]]:
    path = ROOT / "models" / "http" / "lexical_calibration.json"
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
        row = calibration["fixed_fpr"][target]
        source = calibration["source"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Cannot load HTTP lexical calibration") from exc
    if row.get("status") != "calibrated" or row.get("threshold") is None:
        raise ValueError(f"HTTP lexical target {target} is not calibrated")
    source_path = ROOT / source["path"]
    if sha256_path(source_path) != source["sha256"]:
        raise RuntimeError("HTTP benign calibration source hash mismatch")
    threshold = safe_float(row["threshold"], default=-1.0)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Calibrated HTTP threshold is invalid")
    return threshold, {
        "target_fpr": target,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_path(path),
        "benign_source_hash_verified": True,
        "validation_fpr": row["validation_fpr"],
    }


def configure_scorer(
    device: str,
    *,
    enable_flow_models: bool = False,
    bot_target_fpr: str | None = None,
) -> LogScorer:
    config = load_config()
    models = config.setdefault("models", {})
    for model_name in ("ner", "biencoder", "cross_encoder"):
        model = models.setdefault(model_name, {})
        model["device"] = device
        if bot_target_fpr is not None:
            model["enabled"] = False
    if enable_flow_models:
        flow = models.setdefault("flow", {})
        flow["enabled"] = True
        bot = flow.setdefault("bot", {})
        bot["enabled"] = True
        if bot_target_fpr is not None:
            bot["target_fpr"] = bot_target_fpr
            flow["fusion_target_fpr"] = bot_target_fpr
            scoring = config.setdefault("scoring", {})
            scoring["calibrated_flow_isolation"] = True
            scoring.setdefault("sigma", {})["enabled"] = False
            scoring.setdefault("flow_window", {})["enabled"] = False
    return LogScorer(config)


def score_samples(
    samples: list[Sample],
    scorer: LogScorer,
    batch_size: int,
    scenario_thresholds: dict[str, float],
) -> tuple[list[Prediction], dict[str, Any]]:
    normalizer = LogNormalizer()
    predictions: list[Prediction] = []
    started = time.perf_counter()
    normalized_count = 0

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        events = []
        valid_samples = []
        for sample in batch_samples:
            event = normalizer.normalize(sample.model_input)
            assert_label_blind(sample, event.text, event.fields)
            events.append(event)
            valid_samples.append(sample)
            normalized_count += 1
        try:
            scored = scorer.score_batch(events)
        except Exception as exc:  # preserve failed coverage rather than silently dropping rows
            for sample in valid_samples:
                threshold = scenario_thresholds[sample.scenario]
                predictions.append(
                    Prediction(
                        sample_id=sample.sample_id,
                        scenario=sample.scenario,
                        source_file=sample.source_file,
                        row_number=sample.row_number,
                        exact_label=sample.exact_label,
                        family=sample.family,
                        is_attack=sample.is_attack,
                        adapter=sample.adapter,
                        model_input_sha256=hashlib.sha256(
                            sample.model_input.encode("utf-8")
                        ).hexdigest(),
                        score=0.0,
                        predicted=False,
                        threshold=threshold,
                        priority="ERROR",
                        sigma_matched=False,
                        sigma_rule_ids=[],
                        classifier_score=0.0,
                        novelty_score=0.0,
                        duplicate=False,
                        dedup_penalty=0.0,
                        mitre_suggestion="",
                        scoring_latency_ms=0.0,
                        scoring_error=f"{type(exc).__name__}: {exc}",
                    )
                )
            continue

        for sample, result in zip(valid_samples, scored):
            threshold = scenario_thresholds[sample.scenario]
            score = safe_float(result.ai_threat_score)
            predictions.append(
                Prediction(
                    sample_id=sample.sample_id,
                    scenario=sample.scenario,
                    source_file=sample.source_file,
                    row_number=sample.row_number,
                    exact_label=sample.exact_label,
                    family=sample.family,
                    is_attack=sample.is_attack,
                    adapter=sample.adapter,
                    model_input_sha256=hashlib.sha256(
                        sample.model_input.encode("utf-8")
                    ).hexdigest(),
                    score=score,
                    predicted=score >= threshold,
                    threshold=threshold,
                    priority=str(result.ai_priority),
                    sigma_matched=bool(result.sigma_matched),
                    sigma_rule_ids=list(result.sigma_rule_ids),
                    classifier_score=safe_float(result.classifier_score),
                    novelty_score=safe_float(result.novelty_score),
                    duplicate=bool(result.is_duplicate),
                    dedup_penalty=safe_float(result.dedup_penalty),
                    mitre_suggestion=str(result.ai_mitre_technique),
                    scoring_latency_ms=safe_float(result.scoring_latency_ms),
                )
            )

    elapsed = time.perf_counter() - started
    return predictions, {
        "normalized": normalized_count,
        "scored": sum(not prediction.scoring_error for prediction in predictions),
        "scoring_failures": sum(bool(prediction.scoring_error) for prediction in predictions),
        "wall_seconds": elapsed,
        "events_per_second": len(predictions) / elapsed if elapsed else 0.0,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == total else min(1.0, centre + half_width)
    return [lower, upper]


def binary_ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return tie-aware AUROC and average precision without a sklearn runtime dependency."""
    positives = safe_int(y_true.sum())
    negatives = len(y_true) - positives
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = safe_float(ranks[y_true == 1].sum())
    auroc = (
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
        if positives and negatives
        else 0.0
    )

    descending = np.argsort(-scores, kind="mergesort")
    sorted_truth = y_true[descending]
    sorted_desc_scores = scores[descending]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_desc_scores[end] == sorted_desc_scores[start]:
            end += 1
        group_positives = safe_int(sorted_truth[start:end].sum())
        true_positives += group_positives
        false_positives += (end - start) - group_positives
        recall = true_positives / positives if positives else 0.0
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return auroc, average_precision


def metrics_for(predictions: list[Prediction], *, sigma_only: bool = False) -> dict[str, Any]:
    valid = [prediction for prediction in predictions if not prediction.scoring_error]
    y_true = np.array([prediction.is_attack for prediction in valid], dtype=int)
    if sigma_only:
        y_pred = np.array([prediction.sigma_matched for prediction in valid], dtype=int)
        scores = y_pred.astype(float)
    else:
        y_pred = np.array([prediction.predicted for prediction in valid], dtype=int)
        scores = np.array([prediction.score for prediction in valid], dtype=float)

    tp = safe_int(((y_true == 1) & (y_pred == 1)).sum())
    fp = safe_int(((y_true == 0) & (y_pred == 1)).sum())
    tn = safe_int(((y_true == 0) & (y_pred == 0)).sum())
    fn = safe_int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result: dict[str, Any] = {
        "n": len(valid),
        "n_attack": safe_int(y_true.sum()),
        "n_benign": safe_int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "balanced_accuracy": (recall + specificity) / 2,
        "alerts_per_1000_benign": 1000 * fpr,
        "recall_95ci_wilson": wilson_interval(tp, tp + fn),
        "fpr_95ci_wilson": wilson_interval(fp, fp + tn),
        "precision_95ci_wilson": wilson_interval(tp, tp + fp),
    }
    if len(set(y_true.tolist())) == 2:
        result["auroc"], result["auprc"] = binary_ranking_metrics(y_true, scores)
    return result


def per_label_metrics(predictions: list[Prediction]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    attack_predictions = [prediction for prediction in predictions if prediction.is_attack]
    for label in sorted({prediction.exact_label for prediction in attack_predictions}):
        selected = [
            prediction for prediction in attack_predictions if prediction.exact_label == label
        ]
        detected = sum(
            prediction.predicted for prediction in selected if not prediction.scoring_error
        )
        valid = sum(not prediction.scoring_error for prediction in selected)
        output[label] = {
            "n": valid,
            "detected": detected,
            "recall": detected / valid if valid else 0.0,
            "recall_95ci_wilson": wilson_interval(detected, valid),
        }
    return output


def git_workspace_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Public Threat Scenario Evaluation",
        "",
        f"Run: `{summary['run']['id']}`  ",
        f"Threshold: `{summary['run']['threshold']:.4f}` ({summary['run']['threshold_policy']})  ",
        f"Device: `{summary['run']['device']}`",
        "",
        "| Scenario | Attack | Benign | TP | FP | FN | Recall | Precision | F1 | FPR | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in summary["scenarios"].items():
        metric = result["full_pipeline"]
        lines.append(
            f"| {scenario} | {metric['n_attack']} | {metric['n_benign']} | {metric['tp']} | "
            f"{metric['fp']} | {metric['fn']} | {metric['recall']:.3f} | "
            f"{metric['precision']:.3f} | {metric['f1']:.3f} | "
            f"{metric['false_positive_rate']:.3f} | {metric.get('auroc', 0):.3f} | "
            f"{metric.get('auprc', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- CIC results are discrimination of lab-generated, labelled flow summaries, not live DDoS mitigation.",
            "- `Bot` results indicate botnet-associated network activity; no malware execution was verified.",
            "- CSIC requests use structured WAF telemetry; allowed POST bodies are bounded and credential-redacted.",
            "- Cross-dataset/held-out-family performance is an open-set proxy, not genuine zero-day evidence.",
            "- Wilson intervals are conditional flow/request-level intervals; correlated scenario rows limit generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.calibration_target and args.threshold is not None:
        raise SystemExit("--threshold cannot be combined with --calibration-target")
    calibration_provenance: dict[str, Any] | None = None
    http_threshold: float | None = None
    if args.calibration_target:
        if not args.enable_flow_models:
            raise SystemExit("--calibration-target requires --enable-flow-models")
        args.threshold, flow_provenance = load_fusion_calibration(args.calibration_target)
        http_threshold, http_provenance = load_http_calibration(args.calibration_target)
        calibration_provenance = {
            "flow": flow_provenance,
            "http": http_provenance,
        }
    elif args.threshold is None:
        args.threshold = 0.20
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excluded_feature_hashes = (
        load_development_feature_hashes(DEFAULT_DEVELOPMENT_HASH_PATHS)
        if args.enable_flow_models
        else set()
    )
    all_samples: list[Sample] = []
    audits: dict[str, dict[str, Any]] = {}
    for scenario in args.scenarios:
        if scenario == "csic_http":
            samples, audit = csic_samples(args.csic_dir, args.per_class, args.seed)
        elif scenario in CIC_FILE_BY_SCENARIO:
            samples, audit = cic_samples(
                args.cic_zip,
                scenario,
                args.per_class,
                args.seed,
                ddos_min_row=args.ddos_min_row,
                bot_min_row=args.bot_min_row,
                web_min_row=args.web_min_row,
                excluded_feature_hashes=excluded_feature_hashes,
            )
        else:
            raise SystemExit(f"Unknown scenario: {scenario}")
        samples, removed, conflicting_groups = deduplicate_samples(samples, args.seed)
        audit["exact_duplicate_inputs_removed_label_blindly"] = removed
        audit["conflicting_ground_truth_groups_observed"] = conflicting_groups
        audit["post_dedup_selected"] = len(samples)
        audits[scenario] = audit
        all_samples.extend(samples)

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    scorer = configure_scorer(
        args.device,
        enable_flow_models=args.enable_flow_models,
        bot_target_fpr=args.calibration_target,
    )
    scenario_thresholds = {scenario: args.threshold for scenario in args.scenarios}
    if http_threshold is not None and "csic_http" in scenario_thresholds:
        scenario_thresholds["csic_http"] = http_threshold
    predictions, runtime = score_samples(all_samples, scorer, args.batch_size, scenario_thresholds)

    scenario_results: dict[str, Any] = {}
    for scenario in args.scenarios:
        selected = [prediction for prediction in predictions if prediction.scenario == scenario]
        scenario_results[scenario] = {
            "audit": audits[scenario],
            "full_pipeline": metrics_for(selected),
            "sigma_only_proxy": metrics_for(selected, sigma_only=True),
            "per_attack_label": per_label_metrics(selected),
            "duplicate_predictions": sum(prediction.duplicate for prediction in selected),
        }

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = stable_id(
        timestamp,
        git_commit(),
        args.seed,
        args.threshold,
        args.calibration_target,
        calibration_provenance,
    )
    summary = {
        "schema_version": "1.0",
        "run": {
            "id": run_id,
            "created_utc": timestamp,
            "git_commit": git_commit(),
            "workspace_dirty": git_workspace_dirty(),
            "implementation_file_hashes": {
                str(path.relative_to(ROOT)): sha256_path(path)
                for path in IMPLEMENTATION_PROVENANCE_FILES
            },
            "seed": args.seed,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "scenario_thresholds": scenario_thresholds,
            "threshold_policy": (
                "validation-calibrated final fused score; no public-test tuning"
                if calibration_provenance
                else "explicit/deployed composite threshold; no public-test tuning"
            ),
            "calibration_target": args.calibration_target,
            "calibration_provenance": calibration_provenance,
            "device": args.device,
            "flow_models_enabled": args.enable_flow_models,
            "calibrated_flow_isolation": bool(calibration_provenance),
            "development_feature_hashes": {
                str(path.relative_to(ROOT)): sha256_path(path)
                for path in DEFAULT_DEVELOPMENT_HASH_PATHS
                if args.enable_flow_models
            },
            "evaluation_min_rows": {
                "cic_ddos": args.ddos_min_row,
                "cic_bot": args.bot_min_row,
                "cic_web": args.web_min_row,
            },
        },
        "datasets": {
            "CIC-IDS2017": {
                "path": str(args.cic_zip.relative_to(ROOT)),
                "sha256": sha256_path(args.cic_zip),
                "format": "CICFlowMeter CSV archive",
            },
            "CSIC-2010": {
                path.name: {"sha256": sha256_path(path), "bytes": path.stat().st_size}
                for path in sorted(args.csic_dir.glob("*.txt"))
            },
        },
        "input_policy": {
            "ground_truth_sidecar_only": True,
            "leakage_assertions_passed": True,
            "cic_adapter": "sanitized_cef_flow_v2",
            "csic_adapter": "csic_request_to_structured_waf_v2",
            "forbidden": [
                "ground-truth label",
                "attack category",
                "label-derived MITRE ID",
                "label-derived severity",
            ],
        },
        "runtime": runtime,
        "scenarios": scenario_results,
        "limitations": [
            "CIC is a lab-generated flow benchmark and does not validate live mitigation.",
            "CIC CSV rows used here have no source/destination IP, protocol, or timestamp.",
            "Bot-labelled flows are botnet-associated activity, not endpoint-confirmed malware.",
            "Held-out/cross-dataset behavior is an open-set proxy, not a genuine zero-day test.",
            "CSIC is an old synthetic HTTP corpus; adapter constants are label-independent.",
            "Allowed CSIC POST bodies are bounded and credential-redacted before model visibility.",
            "Confidence intervals do not correct for within-scenario flow correlation.",
        ],
    }

    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "thesis_results.md").write_text(markdown_report(summary), encoding="utf-8")

    print(markdown_report(summary))
    print(f"Predictions: {predictions_path}")
    print(f"Summary:     {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
