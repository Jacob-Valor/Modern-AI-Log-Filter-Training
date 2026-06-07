"""Standalone pipeline throughput benchmark (no Docker required).

Measures scoring pipeline performance directly:
  - Events/second throughput
  - p50/p95/p99 latency per stage
  - Memory usage snapshot

Usage:
  python scripts/throughput_benchmark.py [--events N] [--warmup N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from logfilter.config import load_config
from logfilter.models.classifier import LogClassifier
from logfilter.models.ner import NERModel
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

SYSLOG_EVENTS = [
    "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for root from 10.0.0.5 port 44382 ssh2",
    "Jan 15 11:07:54 prod-srv01 sshd[22346]: Accepted publickey for admin from 192.168.1.100 port 22 ssh2",
    "Jan 15 11:07:55 prod-srv01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=10.0.0.100 DST=10.0.0.1 PROTO=TCP SPT=443 DPT=80",
    "Jan 15 11:07:56 prod-srv01 apache2[1234]: 10.0.0.200 - - [15/Jan/2026:11:07:56 +0000] GET /admin/config HTTP/1.1 403 287",
    "Jan 15 11:07:57 prod-srv01 sudo: admin : TTY=pts/0 ; PWD=/home/admin ; USER=root ; COMMAND=/bin/cat /etc/shadow",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def format_latency(name: str, latencies: list[float]) -> list[str]:
    if not latencies:
        return [f"  {name}: no data"]
    return [
        f"  {name}:",
        f"    count:  {len(latencies)}",
        f"    mean:   {statistics.mean(latencies):.2f}ms",
        f"    p50:    {percentile(latencies, 50):.2f}ms",
        f"    p95:    {percentile(latencies, 95):.2f}ms",
        f"    p99:    {percentile(latencies, 99):.2f}ms",
        f"    max:    {max(latencies):.2f}ms",
    ]


def run_benchmark(events: int, warmup: int, batch_size: int) -> None:
    config = load_config(ROOT / "config" / "config.yaml")
    normalizer = LogNormalizer()
    scorer = LogScorer(config=config)

    raw_events = [SYSLOG_EVENTS[i % len(SYSLOG_EVENTS)] for i in range(events + warmup)]

    print(f"\n{'=' * 60}")
    print(f"  Pipeline Throughput Benchmark")
    print(f"{'=' * 60}")
    print(f"  events:     {events}")
    print(f"  warmup:     {warmup}")
    print(f"  batch_size: {batch_size}")
    print()

    normalize_latencies: list[float] = []
    score_latencies: list[float] = []
    e2e_latencies: list[float] = []

    for i, raw in enumerate(raw_events):
        t0 = time.perf_counter()
        normalized = normalizer.normalize(raw)
        t1 = time.perf_counter()
        scored = scorer.score(normalized)
        t2 = time.perf_counter()

        if i >= warmup:
            normalize_latencies.append((t1 - t0) * 1000)
            score_latencies.append((t2 - t1) * 1000)
            e2e_latencies.append((t2 - t0) * 1000)

    total_time = sum(e2e_latencies) / 1000
    throughput = events / total_time if total_time > 0 else 0

    lines = [
        "RESULTS",
        f"  total_time:    {total_time:.2f}s",
        f"  throughput:    {throughput:.0f} events/sec",
        "",
    ]
    lines.extend(format_latency("normalize", normalize_latencies))
    lines.append("")
    lines.extend(format_latency("score (full pipeline)", score_latencies))
    lines.append("")
    lines.extend(format_latency("end-to-end", e2e_latencies))

    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline throughput benchmark")
    parser.add_argument("--events", type=int, default=500, help="Events to score")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup events (not counted)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size hint")
    args = parser.parse_args()
    run_benchmark(args.events, args.warmup, args.batch_size)


if __name__ == "__main__":
    main()
