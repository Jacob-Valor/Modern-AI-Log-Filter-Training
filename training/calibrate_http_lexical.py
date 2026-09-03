#!/usr/bin/env python3
"""Calibrate HTTP lexical thresholds from CSIC benign training requests only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from thresholds import FIXED_FPR_TARGETS, benign_thresholds_at_fixed_fpr

from logfilter.pipeline.normalizer import LogNormalizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "demo_data" / "CSIC-2010" / "normalTrafficTraining.txt"
DEFAULT_OUTPUT = ROOT / "models" / "http" / "lexical_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requests(path: Path) -> Iterator[str]:
    current: list[str] = []
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ")):
                if current:
                    yield "".join(current)
                current = [line]
            elif current:
                current.append(line)
    if current:
        yield "".join(current)


def _to_waf(request: str) -> str | None:
    lines = request.splitlines()
    if not lines:
        return None
    parts = lines[0].split(" ", 2)
    if len(parts) != 3:
        return None
    method, target, _protocol = parts
    headers: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for line in lines[1:]:
        if not in_body and not line.strip():
            in_body = True
            continue
        if in_body:
            body.append(line)
        elif ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()
    return json.dumps(
        {
            "event_type": "csic_waf_request",
            "source_ip": "192.0.2.1",
            "request": {
                "clientIp": "192.0.2.1",
                "method": method.upper(),
                "uri": target,
                "headers": headers,
                "body": "\n".join(body),
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("HTTP lexical score must be numeric") from exc


def main() -> None:
    args = parse_args()
    normalizer = LogNormalizer()
    scores: list[float] = []
    skipped = 0
    for request in _requests(args.input):
        model_input = _to_waf(request)
        if model_input is None:
            skipped += 1
            continue
        event = normalizer.normalize(model_input)
        score = event.fields.get("http_lexical_score")
        if not isinstance(score, (float, int)):
            skipped += 1
            continue
        scores.append(_as_float(score))
    calibration = benign_thresholds_at_fixed_fpr(
        np.asarray(scores, dtype=np.float64), FIXED_FPR_TARGETS
    )
    report = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(args.input.relative_to(ROOT)),
            "sha256": _sha256(args.input),
            "role": "benign_validation_only",
        },
        "request_count": len(scores),
        "skipped": skipped,
        "test_labels_used": False,
        "fixed_fpr": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
