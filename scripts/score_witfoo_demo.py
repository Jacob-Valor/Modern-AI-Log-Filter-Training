#!/usr/bin/env python3
"""
Demo: Score WitFoo Precinct6 events through the AI Log Filter pipeline.

Reads JSONL files (by source type), normalizes, scores, and reports results.
Usage:
  # Score a single source type
  python scripts/score_witfoo_demo.py --source-type winevent --max-events 100

  # Score all source types sequentially
  python scripts/score_witfoo_demo.py --all --max-events 200

  # Score via the running API
  python scripts/score_witfoo_demo.py --source-type syslog --api http://localhost:8080 --api-token $LOGFILTER_API_TOKEN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logfilter.api.schemas import ScoreRequest
from logfilter.config import load_config
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType
from logfilter.pipeline.scorer import LogScorer

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

ROOT = Path(__file__).parent.parent


def load_events(source_type: str, max_events: int = 0, data_dir: str | None = None) -> list[dict]:
    """Load JSONL events for a given source type."""
    if data_dir is None:
        data_dir = str(ROOT / "demo_data" / "witfoo" / "jsonl")

    path = Path(data_dir) / f"{source_type}.jsonl"
    if not path.exists():
        print(f"  [SKIP] No JSONL file for {source_type} at {path}")
        return []

    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
                if max_events and len(events) >= max_events:
                    break

    print(f"  Loaded {len(events)} events from {path.name}")
    return events


def score_direct(events: list[dict], batch_size: int = 200) -> list[dict]:
    """Score events directly using the pipeline's LogScorer."""
    print(f"\n  Loading config and models...")
    config = load_config()
    normalizer = LogNormalizer()

    # Load the scorer (may download HF models on first run)
    scorer = LogScorer(config)
    scorer.preload_models()
    print(f"  Models loaded: classifier={scorer.classifier.is_ready()}, "
          f"syslog_classifier={scorer.syslog_classifier.is_ready()}, "
          f"tier2={scorer.tier2_classifier.is_ready()}")

    results: list[dict] = []
    total = len(events)
    t0 = time.perf_counter()

    for start in range(0, total, batch_size):
        batch = events[start:start + batch_size]
        normalized = []
        for ev in batch:
            src_hint = ev.get("source_type")
            if src_hint:
                src_type = getattr(LogSourceType, src_hint.upper(), None)
            else:
                src_type = None
            normalized.append(normalizer.normalize(ev["raw"], source_type_hint=src_type))

        scored = scorer.score_batch(normalized)

        for se, ev in zip(scored, batch):
            results.append({
                "source_type": se.source_type,
                "ai_threat_score": round(se.ai_threat_score, 4),
                "ai_priority": se.ai_priority,
                "ai_confidence": round(se.ai_confidence, 4),
                "classifier_score": round(se.classifier_score, 4),
                "sigma_matched": se.sigma_matched,
                "is_duplicate": se.is_duplicate,
                "tier2_used": se.tier2_used,
                "entity_boost": se.entity_boost,
                "cross_encoder_max": round(se.cross_encoder_max, 4),
                "score_degraded": se.score_degraded,
                "normalized_text": se.normalized_text[:200],
                # Metadata from input
                "_label": ev.get("_label", ""),
                "_message_type": ev.get("_message_type", ""),
                "_pipeline": ev.get("_pipeline", ""),
                "_suspicion_score": ev.get("_suspicion_score", 0),
                "_attack_techniques": ev.get("_attack_techniques", ""),
            })

        print(f"    Scored {min(start + batch_size, total)}/{total} "
              f"({(time.perf_counter() - t0) / max(1, start + batch_size) * 1000:.1f} ms/event)")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {len(results)} events in {elapsed:.1f}s "
          f"({len(results) / elapsed:.1f} ev/s)")

    return results


def score_via_api(events: list[dict], api_url: str, api_token: str, batch_size: int = 200) -> list[dict]:
    """Score events via the running FastAPI endpoint."""
    import httpx

    headers = {"X-API-Token": api_token, "Content-Type": "application/json"}
    results: list[dict] = []
    total = len(events)
    t0 = time.perf_counter()

    with httpx.Client(base_url=api_url, headers=headers, timeout=120.0) as client:
        for start in range(0, total, batch_size):
            batch = events[start:start + batch_size]
            payload = {
                "events": [
                    {"raw": ev["raw"], "source_type": ev.get("source_type")}
                    for ev in batch
                ]
            }
            resp = client.post("/score/batch", json=payload)
            resp.raise_for_status()
            data = resp.json()
            for r in data["results"]:
                results.append(r)

            print(f"    Scored {min(start + batch_size, total)}/{total}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {len(results)} events in {elapsed:.1f}s "
          f"({len(results) / elapsed:.1f} ev/s)")
    return results


def print_summary(results: list[dict], label: str) -> None:
    """Print a summary of scoring results."""
    if not results:
        print(f"\n  [{label}] No results.")
        return

    priority_counts = Counter(r["ai_priority"] for r in results)
    degraded = sum(1 for r in results if r.get("score_degraded"))
    sigma = sum(1 for r in results if r.get("sigma_matched"))
    duplicates = sum(1 for r in results if r.get("is_duplicate"))
    tier2 = sum(1 for r in results if r.get("tier2_used"))

    scores = [r["ai_threat_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    # By ground-truth label
    if "_label" in results[0]:
        print(f"\n  [{label}] Results by ground-truth label:")
        by_label: dict[str, list[dict]] = {}
        for r in results:
            lbl = r.get("_label", "unknown")
            by_label.setdefault(lbl, []).append(r)
        for lbl, group in sorted(by_label.items()):
            avg = sum(r["ai_threat_score"] for r in group) / len(group)
            print(f"    {lbl:12s}: {len(group):4d} events, avg score {avg:.3f}")

    print(f"\n  [{label}] Summary ({len(results)} events):")
    print(f"    Priority distribution: {dict(priority_counts)}")
    print(f"    Avg threat score:      {avg_score:.3f}")
    print(f"    Sigma matched:         {sigma}")
    print(f"    Duplicates:            {duplicates}")
    print(f"    Tier-2 used:           {tier2}")
    print(f"    Degraded scores:       {degraded}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WitFoo logs through AI Log Filter pipeline")
    parser.add_argument("--source-type", default="winevent",
                        choices=["winevent", "syslog", "firewall", "web", "all"],
                        help="Source type to score (default: winevent)")
    parser.add_argument("--max-events", type=int, default=100,
                        help="Max events per source type (default: 100)")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--api", help="API base URL (e.g. http://localhost:8080)")
    parser.add_argument("--api-token", help="X-API-Token for API scoring")
    parser.add_argument("--data-dir", help="Path to JSONL data directory")
    args = parser.parse_args()

    source_types = ["winevent", "syslog", "firewall", "web"] if args.source_type == "all" else [args.source_type]

    for st in source_types:
        events = load_events(st, args.max_events, args.data_dir)
        if not events:
            continue

        print(f"\n── Scoring {st} ({len(events)} events) ──")

        if args.api:
            results = score_via_api(events, args.api, args.api_token or "", args.batch_size)
        else:
            results = score_direct(events, args.batch_size)

        print_summary(results, st)


if __name__ == "__main__":
    main()
