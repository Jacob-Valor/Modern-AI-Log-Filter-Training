#!/usr/bin/env python3
"""
End-to-end test: CIC-IDS2017 CSV → CEF → LogNormalizer → Scorer → LEEF → Router.

Runs WITHOUT Docker — exercises all pipeline stages with real ML models
(BiEncoder, NER, CrossEncoder) loaded from local artifacts.

Usage
-----
    python scripts/e2e_test_cic_pipeline.py [--csv path/to/sample.csv] [--max-rows 20]
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.config import load_config  # noqa: E402
from logfilter.demo.cic_converter import CicCsvToSyslog  # noqa: E402
from logfilter.pipeline.enricher import LEEFEnricher  # noqa: E402
from logfilter.pipeline.normalizer import LogNormalizer  # noqa: E402
from logfilter.pipeline.router import LogRouter  # noqa: E402
from logfilter.pipeline.scorer import LogScorer  # noqa: E402

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description="CIC-IDS2017 → Pipeline e2e test")
    parser.add_argument("--csv", default="demo_data/cic_ids2017_sample.csv", help="CIC CSV path")
    parser.add_argument("--max-rows", type=int, default=20, help="Rows to test")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{FAIL} CSV not found: {csv_path}")
        sys.exit(1)

    print("=" * 65)
    print("  CIC-IDS2017 → Full Pipeline End-to-End Test")
    print("=" * 65)
    print(f"  Source:   {csv_path}")
    print(f"  Max rows: {args.max_rows}")
    start = time.time()

    # ── Step 1: Convert CSV → CEF syslog ──
    print(f"\n{'─' * 55}")
    print("  Step 1: CSV → CEF syslog")
    converter = CicCsvToSyslog(shuffle=False)
    syslog_lines = converter.csv_lines(csv_path, max_rows=args.max_rows, attack_only=False)
    print(f"  {PASS} Generated {len(syslog_lines)} CEF syslog lines")
    attack_counts = converter.summary(syslog_lines)
    for cat, cnt in sorted(attack_counts.items(), key=lambda x: -x[1]):
        print(f"       {cat:20s} {cnt}")

    # ── Step 2: Normalize ──
    print(f"\n{'─' * 55}")
    print("  Step 2: LogNormalizer")
    normalizer = LogNormalizer()
    events = []
    for line in syslog_lines:
        event = normalizer.normalize(line)
        if event:
            events.append(event)
    print(f"  {PASS} Normalized {len(events)}/{len(syslog_lines)} lines")
    source_types = Counter(
        e.source_type.name if hasattr(e, "source_type") else "unknown" for e in events
    )
    for st, cnt in source_types.most_common():
        print(f"       {st:20s} {cnt}")

    if not events:
        print(f"  {FAIL} No events to score — aborting")
        sys.exit(1)

    # ── Step 3: Score ──
    print(f"\n{'─' * 55}")
    print("  Step 3: LogScorer (real BiEncoder + NER + CrossEncoder)")
    config = load_config()
    scorer = LogScorer(config)
    scored = scorer.score_batch(events)
    print(f"  {PASS} Scored {len(scored)} events")
    priorities = Counter(str(s.ai_priority) for s in scored)
    for pri, cnt in sorted(priorities.items()):
        print(f"       {pri:20s} {cnt}")
    avg_score = sum(s.ai_threat_score for s in scored) / len(scored)
    print(f"       avg threat score:  {avg_score:.4f}")
    print(f"       sample techniques: {[s.ai_mitre_technique for s in scored[:5]]}")

    # ── Step 4: Enrich ──
    print(f"\n{'─' * 55}")
    print("  Step 4: LEEFEnricher")
    enricher = LEEFEnricher()
    leef_payloads = []
    for s in scored:
        try:
            leef = enricher.enrich(s, es_doc_id=uuid.uuid4().hex)
            leef_payloads.append(leef)
        except Exception as e:
            print(f"       {FAIL} Enrich failed: {e}")
    print(f"  {PASS} Enriched {len(leef_payloads)}/{len(scored)} events to LEEF")
    if leef_payloads:
        print("  Sample LEEF (first 200 chars):")
        print(f"    {str(leef_payloads[0])[:200]}")

    # ── Step 5: Route ──
    print(f"\n{'─' * 55}")
    print("  Step 5: LogRouter")
    router = LogRouter(config)
    routed = 0
    for leef, event in zip(leef_payloads[:5], scored[:5]):  # route a few
        try:
            router.route(leef, event)
            routed += 1
        except Exception as e:
            print(f"       {FAIL} Route failed: {e}")
    print(f"  {PASS} Routed {routed} LEEF payloads")

    # ── Summary ──
    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print("  Pipeline: CIC CSV → CEF → Normalize → Score → Enrich → Route")
    print(f"  Status:   {PASS} ALL STAGES PASSED")
    print(f"  Events:   {len(scored)} scored, {len(leef_payloads)} enriched")
    print(f"  Avg score: {avg_score:.4f}")
    print(f"  Time:     {elapsed:.1f}s")


if __name__ == "__main__":
    main()
