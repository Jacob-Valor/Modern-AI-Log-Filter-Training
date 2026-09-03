#!/usr/bin/env python3
"""
Presentation demo: AI Log Filter processing CIC-IDS2017 attack data.

Shows the full pipeline in a polished terminal walkthrough:
  1. Attack data ingestion (CIC CSV → CEF syslog)
  2. Log normalization
  3. AI scoring (ONNX + SecureBERT2.0 models)
  4. LEEF enrichment with threat scores

Usage
-----
    # Full demo (recommended for presentation)
    python scripts/presentation_demo.py

    # Show only N events (faster)
    python scripts/presentation_demo.py --events 10

    # CSV → CEF conversion only (skip ML models)
    python scripts/presentation_demo.py --convert-only

    # Use a different CIC CSV file
    python scripts/presentation_demo.py --csv path/to/your.csv

Requires
--------
    pip install -e ".[dev]"   (already done)

    Model artifacts under models/ (already present):
      models/log_classifier.onnx
      models/scaler.json
      models/biencoder/final/
      models/ner/final/
      models/cross_encoder/final/
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
from logfilter.pipeline.scorer import LogScorer, ScoredEvent  # noqa: E402

# ── Terminal styling ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

LOGO = f"""{BOLD}{CYAN}
  ╔══════════════════════════════════════════════════════════╗
  ║         Modern AI Log Filter — Attack Demo              ║
  ║   CIC-IDS2017  →  CEF  →  ONNX  →  SecureBERT2.0       ║
  ╚══════════════════════════════════════════════════════════╝{RESET}
"""

SEP = f"{DIM}{'─' * 60}{RESET}\n"

# Attack color map for display
ATK_COLORS: dict[str, str] = {
    "DDOS": RED,
    "DOS": RED,
    "BOTNET": MAGENTA,
    "HEARTBLEED": RED,
    "WEB ATTACK": YELLOW,
    "BRUTE FORCE": YELLOW,
    "INFILTRATION": MAGENTA,
    "PORT SCAN": BLUE,
    "BENIGN": GREEN,
}

PRIO_COLORS: dict[str, str] = {
    "HIGH": RED,
    "MEDIUM": YELLOW,
    "LOW": BLUE,
    "INFO": GREEN,
}


def c(text: str, color: str = WHITE) -> str:
    return f"{color}{text}{RESET}"


def print_step(n: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}  ▸ Step {n}: {title}{RESET}")
    print(SEP)


def print_ok(text: str) -> None:
    print(f"  {c('✓', GREEN)} {text}")


def print_info(text: str) -> None:
    print(f"    {c('·', DIM)} {text}")


def print_event_table(events: list[ScoredEvent], count: int = 8) -> None:
    """Print scored events as a formatted table."""
    header = (
        f"  {c('#', DIM):>3s}  "
        f"{c('Attack Type', BOLD):20s}  "
        f"{c('Score', BOLD):7s}  "
        f"{c('Priority', BOLD):9s}  "
        f"{c('MITRE', BOLD):12s}  "
        f"{c('Entities', BOLD):25s}"
    )
    print(
        f"\n  {c('Scored Events', BOLD)}{DIM} (showing {min(count, len(events))}/{len(events)}){RESET}"
    )
    print(f"  {DIM}{'─' * 90}{RESET}")
    print(header)
    print(f"  {DIM}{'─' * 90}{RESET}")

    for i, ev in enumerate(events[:count]):
        cat = getattr(ev, "attack_category", "")
        if not cat:
            # Derive from label or event data
            cat = "N/A"
        atk_color = ATK_COLORS.get(cat.upper(), WHITE)
        prio_color = PRIO_COLORS.get(str(ev.ai_priority), WHITE)

        entities = (ev.ai_entities or "")[:25] if hasattr(ev, "ai_entities") else ""
        technique = ev.ai_mitre_technique or ""

        print(
            f"  {c(f'{i + 1:>3d}', DIM)}  "
            f"{c(f'{cat[:20]:20s}', atk_color)}  "
            f"{c(f'{ev.ai_threat_score:.4f}', prio_color)}  "
            f"{c(f'{str(ev.ai_priority):9s}', prio_color)}  "
            f"{c(f'{technique[:12]:12s}', DIM)}  "
            f"{c(entities[:25], DIM)}"
        )
    print(f"  {DIM}{'─' * 90}{RESET}")


def print_cef_sample(line: str) -> None:
    """Display a CEF line with color-highlighted fields."""
    print(f"\n  {BOLD}CEF Syslog Output:{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")

    # Show syslog header
    syslog_end = line.find("CEF:")
    if syslog_end > 0:
        print(f"  {DIM}{line[:syslog_end]}{RESET}")

    # CEF header
    cef_parts = line[syslog_end:].split("|")
    vendor_info = "|".join(cef_parts[:5])
    severity = cef_parts[5] if len(cef_parts) > 5 else "?"
    rest = "|".join(cef_parts[6:]) if len(cef_parts) > 6 else ""
    print(f"  {c('CEF Header:', BOLD)} {vendor_info}", end="")
    print(f" | {c(f'sev={severity}', RED)} |")

    # CEF extensions (key=value pairs)
    key_fields = {}
    for p in rest.split("\t"):
        if "=" in p:
            k, v = p.split("=", 1)
            key_fields[k] = v

    def show(k: str, label: str, color: str = WHITE) -> None:
        if k in key_fields:
            print(f"    {c(label, DIM)} {c(key_fields[k], color)}")

    show("src", "Source IP     → ")
    show("dst", "Destination   → ")
    show("spt", "Source Port   → ")
    show("dpt", "Dest Port     → ")
    show("proto", "Protocol      → ")
    show("act", "Action        → ", YELLOW)
    show("cs1", "MITRE Tactic  → ", MAGENTA)
    show("cs2", "Attack Cat    → ", RED)
    show("cat", "Label         → ", YELLOW)
    show("flexNumber1", "Severity      → ", RED)
    print(f"  {DIM}{'─' * 60}{RESET}")


def print_leef_sample(leef_str: str) -> None:
    """Display a LEEF payload with key fields highlighted."""
    print(f"\n  {BOLD}LEEF Output (QRadar-ready):{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")

    # Parse tab-separated fields after the LEEF header
    if "\t" in leef_str:
        header_end = leef_str.index("\t")
        print(f"  {c(leef_str[:header_end], GREEN)}")
        rest = leef_str[header_end + 1 :]
        pairs = rest.split("\t")
        for pair in pairs:
            if "=" in pair:
                k, v = pair.split("=", 1)
                if "score" in k.lower() or "priority" in k.lower():
                    print(f"    {c(k, BOLD)}={c(v, YELLOW)}")
                elif "mitre" in k.lower():
                    print(f"    {c(k, BOLD)}={c(v, MAGENTA)}")
                elif "entity" in k.lower():
                    print(f"    {c(k, BOLD)}={c(v, CYAN)}")
                else:
                    print(f"    {DIM}{k}={v}{RESET}")
    else:
        print(f"  {leef_str[:200]}")
    print(f"  {DIM}{'─' * 60}{RESET}")


def summary_box(items: list[tuple[str, str, str]]) -> None:
    """Print a summary box with (label, value, color) tuples."""
    print(f"\n  {BOLD}{c('═══ RESULTS ═══', GREEN)}{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    for label, value, color in items:
        print(f"  {c(f'{label:25s}', BOLD)} {c(value, color)}")
    print(f"  {DIM}{'─' * 60}{RESET}")


# ── Main demo ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Presentation demo: AI Log Filter pipeline")
    parser.add_argument("--csv", default="demo_data/cic_ids2017_sample.csv", help="CIC CSV path")
    parser.add_argument("--events", type=int, default=30, help="Number of events to process")
    parser.add_argument("--convert-only", action="store_true", help="CSV → CEF only (skip ML)")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--auto", action="store_true", help="Auto-advance without Enter keys")
    args = parser.parse_args()

    if args.no_color:
        globals()["c"] = lambda text, color="": text

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"  ✗ CSV not found: {csv_path}")
        print("  Run: python scripts/download_cic_ids2017.py 2017 --sample")
        sys.exit(1)

    # Clear screen
    print("\033[2J\033[H", end="")

    print(LOGO)

    if not args.auto:
        input(f"  {DIM}Press Enter to start the demo...{RESET}")
    print()

    total_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: CSV → CEF
    # ══════════════════════════════════════════════════════════════════════════
    print_step(1, "Attack Data Ingestion  —  CIC-IDS2017 CSV  →  CEF Syslog")

    print_info(f"Source: {csv_path}")
    print_info("Reading labelled network flows from Canadian Institute for Cybersecurity...")

    converter = CicCsvToSyslog(shuffle=True)
    syslog_lines = converter.csv_lines(csv_path, max_rows=args.events, attack_only=False)
    attack_counts = converter.summary(syslog_lines)

    total_events = len(syslog_lines)
    total_attacks = sum(v for k, v in attack_counts.items() if k != "BENIGN")
    total_benign = attack_counts.get("BENIGN", 0)

    print_ok(f"Generated {total_events} CEF syslog events")
    print_info(f"  {c('Attack events:', RED)} {total_attacks}")
    print_info(f"  {c('Benign events:', GREEN)} {total_benign}")

    print(f"\n  {BOLD}Attack Distribution:{RESET}")
    for cat, cnt in sorted(attack_counts.items(), key=lambda x: -x[1]):
        color = ATK_COLORS.get(cat, WHITE)
        bar = "█" * max(1, cnt * 30 // max(attack_counts.values()))
        print(f"    {c(f'{cat:20s}', color)} {c(f'{cnt:3d}', BOLD)}  {c(bar, DIM)}")

    # Show a sample CEF line
    if syslog_lines:
        # Find a DDoS or attack line for the sample
        sample_line = next(
            (line for line in syslog_lines if "cs2=DDOS" in line or "cs2=DOS" in line),
            syslog_lines[0],
        )
        print_cef_sample(sample_line)

    if not args.auto:
        input(f"\n  {DIM}Press Enter to continue...{RESET}")

    if args.convert_only:
        print(
            f"\n  {c('✓', GREEN)} Conversion complete. Run without --convert-only for full pipeline.\n"
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Normalize
    # ══════════════════════════════════════════════════════════════════════════
    print_step(2, "Log Normalization  —  CEF  →  Structured Events")

    normalizer = LogNormalizer()
    events = []
    for i, line in enumerate(syslog_lines):
        event = normalizer.normalize(line)
        if event:
            events.append(event)

    print_ok(f"Normalized {len(events)}/{total_events} events")
    source_types = Counter(
        e.source_type.name if hasattr(e, "source_type") else "unknown" for e in events
    )
    for st, cnt in source_types.most_common():
        print_info(f"{st}: {cnt} events")

    if not args.auto:
        input(f"\n  {DIM}Press Enter to load AI models and score...{RESET}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Score
    # ══════════════════════════════════════════════════════════════════════════
    print_step(3, "AI Threat Scoring  —  ONNX  +  SecureBERT2.0  (BiEncoder, NER, CrossEncoder)")

    print_info("Loading model artifacts...")
    print_info("  • Tier-1: XGBoost (ONNX)  —  fast pattern matching")
    print_info("  • Tier-2: SecureBERT2.0   —  NER entity extraction")
    print_info("  • Tier-2: CrossEncoder     —  ATT&CK technique relevance")
    print_info("  • Tier-2: BiEncoder        —  near-duplicate detection")
    print_info(f"  Events to score: {len(events)}")

    config = load_config()
    scorer = LogScorer(config)
    scored = scorer.score_batch(events)

    priorities = Counter(str(s.ai_priority) for s in scored)
    avg_score = sum(s.ai_threat_score for s in scored) / len(scored) if scored else 0.0

    print_ok(f"Scored {len(scored)} events")

    # Priority distribution
    print(f"\n  {BOLD}Threat Priority Distribution:{RESET}")
    for pri in ["HIGH", "MEDIUM", "LOW", "INFO"]:
        cnt_p = priorities.get(pri, 0)
        color = PRIO_COLORS.get(pri, WHITE)
        bar = "█" * cnt_p if cnt_p else "─"
        print(f"    {c(f'{pri:10s}', color)} {c(f'{cnt_p:3d}', BOLD)}  {c(bar, DIM)}")

    print(f"\n  {BOLD}Average threat score: {c(f'{avg_score:.4f}', YELLOW)}")

    # Show MITRE techniques found
    techniques = Counter(s.ai_mitre_technique for s in scored if s.ai_mitre_technique)
    if techniques:
        print(f"\n  {BOLD}MITRE ATT&CK Techniques Detected:{RESET}")
        for tech, cnt in techniques.most_common(8):
            print(f"    {c(f'{tech:15s}', MAGENTA)} {c(f'{cnt}', DIM)} events")

    # Show event table
    print_event_table(scored)

    if not args.auto:
        input(f"\n  {DIM}Press Enter to see LEEF enrichment...{RESET}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Enrich
    # ══════════════════════════════════════════════════════════════════════════
    print_step(4, "LEEF Enrichment  —  QRadar-ready Output")

    enricher = LEEFEnricher()
    leef_payloads: list[str] = []
    for s in scored:
        try:
            leef = enricher.enrich(s, es_doc_id=uuid.uuid4().hex)
            leef_payloads.append(leef)
        except Exception:
            pass

    print_ok(f"Generated {len(leef_payloads)} LEEF 2.0 payloads")

    if leef_payloads:
        # Find a HIGH priority event for display
        high_leef = None
        for s, leef in zip(scored, leef_payloads):
            if str(s.ai_priority) == "HIGH" or s.ai_threat_score > 0.8:
                high_leef = leef
                break
        if high_leef is None and leef_payloads:
            high_leef = leef_payloads[0]
        if high_leef:
            print_leef_sample(high_leef)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    print(f"\n{SEP}")

    summary_box(
        [
            ("Total events", str(total_events), WHITE),
            ("Attacks detected", str(total_attacks), RED),
            ("Events scored", str(len(scored)), CYAN),
            ("Average threat score", f"{avg_score:.4f}", YELLOW),
            ("MITRE techniques", str(len(techniques)), MAGENTA),
            ("Model pipeline", "ONNX + SecureBERT2.0", GREEN),
            ("Processing time", f"{total_elapsed:.1f}s", DIM),
        ]
    )

    print(f"\n  {BOLD}{GREEN}Demo complete!{RESET}")
    print(f"\n  {DIM}For the full Docker stack (Kafka, ES, Grafana):{RESET}")
    print(f"    {DIM}cp .env.example .env  # fill in secrets{RESET}")
    print(f"    {DIM}make up{RESET}")
    print(f"\n  {DIM}Then feed attack data live:{RESET}")
    print(f"    {DIM}python scripts/demo_cic_ids2017.py {csv_path} --port 5140{RESET}")
    print()


if __name__ == "__main__":
    main()
