#!/usr/bin/env python3
"""Per-model throughput benchmark.

Measures events-per-second (EPS) and latency percentiles for each model
individually. Also reports the full-pipeline throughput for reference.

Usage:
    python scripts/per_model_benchmark.py [--events N] [--warmup N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import structlog

from logfilter.config import load_config
from logfilter.models.classifier import LogClassifier
from logfilter.models.syslog_classifier import SyslogClassifier
from logfilter.models.tier2_classifier import Tier2Classifier
from logfilter.models.ner import NERModel
from logfilter.models.biencoder import BiEncoderModel
from logfilter.models.cross_encoder import CrossEncoderModel

logger = structlog.get_logger(__name__)

# ── Test data ──────────────────────────────────────────────────────────────────

SYSLOG_EVENTS = [
    "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for root from 10.0.0.5 port 44382 ssh2",
    "Jan 15 11:07:54 prod-srv01 sshd[22346]: Accepted publickey for admin from 192.168.1.100 port 22 ssh2",
    "Jan 15 11:07:55 prod-srv01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=10.0.0.100 DST=10.0.0.1 PROTO=TCP SPT=443 DPT=80",
    "Jan 15 11:07:56 prod-srv01 apache2[1234]: 10.0.0.200 - - [15/Jan/2026:11:07:56 +0000] GET /admin/config HTTP/1.1 403 287",
    "Jan 15 11:07:57 prod-srv01 sudo: admin : TTY=pts/0 ; PWD=/home/admin ; USER=root ; COMMAND=/bin/cat /etc/shadow",
    "Jan 15 11:07:58 prod-srv01 sshd[22347]: Failed password for invalid user admin from 192.168.1.200 port 53942 ssh2",
    "Jan 15 11:07:59 prod-srv01 kernel: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55:66:77:88:99:00:11:22:33 SRC=10.0.0.50 DST=10.0.0.1 PROTO=UDP SPT=53 DPT=53",
    "Jan 15 11:08:00 prod-srv01 apache2[1235]: 10.0.0.201 - admin [15/Jan/2026:11:08:00 +0000] POST /wp-admin/admin-ajax.php HTTP/1.1 200 1234",
    "Jan 15 11:08:01 prod-srv01 cron[1236]: pam_unix(cron:session): session opened for user root by (uid=0)",
    "Jan 15 11:08:02 prod-srv01 sshd[22348]: Received disconnect from 10.0.0.5 port 44382:11: Bye Bye",
]

N_FEATURES_CLASSIFIER = 2255
N_FEATURES_SYSLOG = 100

# MITRE technique candidates for CrossEncoder benchmark
CANDIDATES = [
    {"id": "T1078", "name": "Valid Accounts", "description": "Adversaries may steal credentials to access valid accounts."},
    {"id": "T1059", "name": "Command and Scripting Interpreter", "description": "Adversaries may abuse command interpreters to execute commands."},
    {"id": "T1190", "name": "Exploit Public-Facing Application", "description": "Adversaries may exploit a software vulnerability to gain access."},
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def format_result(
    model_name: str,
    latencies_ms: list[float],
    events: int,
    batch_size: int,
) -> list[str]:
    if not latencies_ms:
        return [f"  {model_name}: no data"]
    total_s = sum(latencies_ms) / 1000
    eps = (events * batch_size) / total_s if total_s > 0 else 0
    return [
        f"  {model_name}:",
        f"    total events:  {events * batch_size}",
        f"    batch size:    {batch_size}",
        f"    total time:    {total_s:.2f}s",
        f"    throughput:    {eps:,.0f} eps",
        f"    mean latency:  {statistics.mean(latencies_ms):.2f}ms",
        f"    p50 latency:   {percentile(latencies_ms, 50):.2f}ms",
        f"    p95 latency:   {percentile(latencies_ms, 95):.2f}ms",
        f"    p99 latency:   {percentile(latencies_ms, 99):.2f}ms",
        f"    max latency:   {max(latencies_ms):.2f}ms",
    ]


def generate_classifier_features(n: int, n_features: int) -> np.ndarray:
    """Generate random bag-of-event feature vectors."""
    rng = np.random.default_rng(42)
    vectors = np.zeros((n, n_features), dtype=np.float32)
    for i in range(n):
        activated = rng.integers(0, n_features, size=rng.integers(1, 20))
        vectors[i, activated] = rng.uniform(0.5, 3.0, size=len(activated))
    return vectors


# ── Model benchmarks ───────────────────────────────────────────────────────────


def bench_log_classifier(
    events: int, warmup: int, batch_size: int
) -> list[str]:
    print(f"\n  Loading LogClassifier (Tier-1, ONNX XGBoost, {N_FEATURES_CLASSIFIER} features)...")
    model = LogClassifier()
    _ = model.is_ready()

    batch = generate_classifier_features(batch_size, N_FEATURES_CLASSIFIER)
    latencies: list[float] = []

    for i in range(events + warmup):
        t0 = time.perf_counter()
        _ = model.predict_proba(batch)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("LogClassifier", latencies, events, batch_size)


def bench_syslog_classifier(
    events: int, warmup: int, batch_size: int
) -> list[str]:
    print(f"\n  Loading SyslogClassifier (ONNX XGBoost, {N_FEATURES_SYSLOG} features)...")
    model = SyslogClassifier()
    _ = model.is_ready()

    batch = generate_classifier_features(batch_size, N_FEATURES_SYSLOG)
    latencies: list[float] = []

    for i in range(events + warmup):
        t0 = time.perf_counter()
        _ = model.predict_proba(batch)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("SyslogClassifier", latencies, events, batch_size)


def bench_tier2_classifier(
    events: int, warmup: int, batch_size: int
) -> list[str]:
    print(f"\n  Loading Tier2Classifier (SecureBERT2.0 → ONNX)...")
    model = Tier2Classifier()
    _ = model.is_ready()

    latencies: list[float] = []
    for i in range(events + warmup):
        batch_texts = [SYSLOG_EVENTS[j % len(SYSLOG_EVENTS)] for j in range(batch_size)]
        t0 = time.perf_counter()
        _ = model.predict_proba(batch_texts)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("Tier2Classifier", latencies, events, batch_size)


def bench_ner(events: int, warmup: int, batch_size: int) -> list[str]:
    print(f"\n  Loading NERModel (SecureBERT2.0-NER)...")
    model = NERModel(device="cpu", batch_size=batch_size)

    latencies: list[float] = []
    for i in range(events + warmup):
        batch_texts = [SYSLOG_EVENTS[j % len(SYSLOG_EVENTS)] for j in range(batch_size)]
        t0 = time.perf_counter()
        _ = model.extract_batch(batch_texts)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("NERModel", latencies, events, batch_size)


def bench_biencoder(events: int, warmup: int, batch_size: int) -> list[str]:
    print(f"\n  Loading BiEncoderModel (SecureBERT2.0-biencoder)...")
    config = load_config(ROOT / "config" / "config.yaml")
    models_cfg = config.get("models", {})
    biencoder_cfg = models_cfg.get("biencoder", {})
    model = BiEncoderModel(
        device="cpu",
        batch_size=batch_size,
        faiss_top_k=3,
        dedup_threshold=0.95,
        dedup_window_minutes=5.0,
        mitre_techniques_path=ROOT / "config" / "mitre_techniques.json",
    )

    latencies: list[float] = []
    for i in range(events + warmup):
        batch_texts = [SYSLOG_EVENTS[j % len(SYSLOG_EVENTS)] for j in range(batch_size)]
        t0 = time.perf_counter()
        _ = model.check_dedup_and_retrieve_batch(batch_texts)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("BiEncoderModel", latencies, events, batch_size)


def bench_cross_encoder(
    events: int, warmup: int, batch_size: int
) -> list[str]:
    print(f"\n  Loading CrossEncoderModel (SecureBERT2.0-cross_encoder)...")
    model = CrossEncoderModel(device="cpu", batch_size=batch_size)

    candidates_per_event = [CANDIDATES] * batch_size
    latencies: list[float] = []
    for i in range(events + warmup):
        batch_texts = [SYSLOG_EVENTS[j % len(SYSLOG_EVENTS)] for j in range(batch_size)]
        t0 = time.perf_counter()
        _ = model.score_batch(batch_texts, candidates_per_event)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("CrossEncoderModel", latencies, events, batch_size)


# ── Pipeline reference ─────────────────────────────────────────────────────────


def bench_full_pipeline(
    events: int, warmup: int, batch_size: int
) -> list[str]:
    from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType

    print(f"\n  Loading full pipeline (all models + normalizer + scorer)...")
    config = load_config(ROOT / "config" / "config.yaml")
    normalizer = LogNormalizer()
    from logfilter.pipeline.scorer import LogScorer
    scorer = LogScorer(config=config)

    latencies: list[float] = []
    for i in range(events + warmup):
        raw_events = [SYSLOG_EVENTS[j % len(SYSLOG_EVENTS)] for j in range(batch_size)]
        normalized = [normalizer.normalize(r) for r in raw_events]
        t0 = time.perf_counter()
        _ = scorer.score_batch(normalized)
        t1 = time.perf_counter()
        if i >= warmup:
            latencies.append((t1 - t0) * 1000)

    return format_result("FullPipeline", latencies, events, batch_size)


# ── Main ───────────────────────────────────────────────────────────────────────


BENCHMARKS = [
    ("Tier-1: LogClassifier (2255 features)", bench_log_classifier),
    ("Tier-1: SyslogClassifier (100 features)", bench_syslog_classifier),
    ("Tier-2: Tier2Classifier (SecureBERT2.0 ONNX)", bench_tier2_classifier),
    ("Tier-3: NERModel (SecureBERT2.0-NER)", bench_ner),
    ("Tier-2: BiEncoderModel + FAISS", bench_biencoder),
    ("Tier-3: CrossEncoderModel", bench_cross_encoder),
    ("Full Pipeline (all models combined)", bench_full_pipeline),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-model throughput benchmark for LogFilter"
    )
    parser.add_argument("--events", type=int, default=100,
                        help="Scoring iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (not counted, default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per iteration (default: 32)")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated models to benchmark (default: all)")
    args = parser.parse_args()

    print(f"{'=' * 72}")
    print(f"  Per-Model Throughput Benchmark")
    print(f"{'=' * 72}")
    print(f"  events:       {args.events}")
    print(f"  warmup:       {args.warmup}")
    print(f"  batch size:   {args.batch_size}")
    print()

    selected: list[tuple[str, Any]] = BENCHMARKS
    if args.models:
        names = [n.strip().lower() for n in args.models.split(",")]
        selected = [
            (name, fn) for name, fn in BENCHMARKS
            if any(s in name.lower() for s in names)
        ]
        if not selected:
            available = ", ".join(n for n, _ in BENCHMARKS)
            print(f"  No models matched '{args.models}'. Available: {available}")
            sys.exit(1)

    for model_name, bench_fn in selected:
        print(f"{'-' * 72}")
        print(f"  BENCHMARK: {model_name}")
        try:
            result_lines = bench_fn(args.events, args.warmup, args.batch_size)
            print("\n".join(result_lines))
            print()
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

    print(f"{'=' * 72}")
    print(f"  Benchmark complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
