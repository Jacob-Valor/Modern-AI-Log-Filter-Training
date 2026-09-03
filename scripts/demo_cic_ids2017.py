#!/usr/bin/env python3
"""
CIC-IDS2017 Attack Feed Demo

Converts CIC-IDS2017/2018 labelled intrusion-detection CSV files into CEF syslog
messages and feeds them into the logfilter pipeline via UDP.

The output is formatted as CEF syslog (RFC 3164) so the LogNormalizer's ``_try_cef``
parser correctly classifies events as ``LogSourceType.FIREWALL``.

Demo scenarios
--------------
    1. **DDoS flood** — feed DDoS rows only::

        python scripts/demo_cic_ids2017.py path/to/csv --filter DDOS --max-rows 1000 --delay 0.0

    2. **Botnet + brute-force** — mix attack types::

        python scripts/demo_cic_ids2017.py path/to/csv --filter BOTNET --filter "BRUTE FORCE" --max-rows 500

    3. **Full pipeline test** — send to live collector::

        python scripts/demo_cic_ids2017.py path/to/csv --max-rows 2000 --port 5140 --delay 0.01

    4. **Just write to file** (no network)::

        python scripts/demo_cic_ids2017.py path/to/csv --max-rows 500 --output demo_data/demo_attack_logs.syslog

    5. **DDoS attack burst** — high volume for threshold testing::

        python scripts/demo_cic_ids2017.py path/to/csv --filter DDOS --max-rows 5000 --loop 3

Requirements
------------
    pandas (already in project deps)

Security note
-------------
This script sends *simulated* attack data generated from research datasets.
It does NOT perform actual attacks. The output is CEF syslog text, safe for
pipeline testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.demo.cic_converter import CicCsvToSyslog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feed CIC-IDS2017 attack data into the logfilter pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", type=str, help="Path to CIC-IDS2017 CSV file")
    parser.add_argument("--max-rows", type=int, default=500, help="Rows to convert (default: 500)")
    parser.add_argument(
        "--attack-only",
        action="store_true",
        default=True,
        help="Skip BENIGN rows (default: true)",
    )
    parser.add_argument("--no-attack-only", dest="attack_only", action="store_false")
    parser.add_argument(
        "--filter",
        action="append",
        dest="attack_types",
        help="Only include specific attack types (can repeat). Options: DDOS, DOS, BOTNET, BRUTE FORCE, HEARTBLEED, WEB ATTACK, PORT SCAN, INFILTRATION",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write syslog to file (instead of sending)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Collector host (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=5140, help="Collector UDP port (default: 5140)")
    parser.add_argument(
        "--delay", type=float, default=0.01, help="Seconds between syslog lines (default: 0.01)"
    )
    parser.add_argument("--loop", type=int, default=1, help="Repeat the feed N times (default: 1)")
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Show available attack types and exit",
    )
    return parser.parse_args()


def print_banner(
    csv_path: str, num_lines: int, attack_counts: dict[str, int], args: argparse.Namespace
) -> None:
    print()
    print("=" * 65)
    print("  CIC-IDS2017 Attack Data Feed")
    print("=" * 65)
    print(f"  Source:     {csv_path}")
    print(f"  Events:     {num_lines}")
    print("  Attack distribution:")
    for cat, cnt in sorted(attack_counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, cnt * 40 // max(attack_counts.values())) if attack_counts else ""
        print(f"    {cat:20s} {cnt:5d}  {bar}")
    if args.output:
        print(f"  Output:     {args.output}")
    else:
        print(f"  Sending to: {args.host}:{args.port}")
        print(f"  Delay:      {args.delay}s per event")
    if args.loop > 1:
        print(f"  Loop:       {args.loop}x")
    print()


def main() -> None:
    args = parse_args()

    if args.list_types:
        from logfilter.demo.cic_converter import _ATTACK_FAMILIES

        print("Available attack types (use with --filter):")
        seen = set()
        for cat in sorted(set(_ATTACK_FAMILIES.values())):
            if cat not in seen and cat != "BENIGN":
                print(f"  {cat}")
                seen.add(cat)
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    converter = CicCsvToSyslog(shuffle=True)

    print(f"Reading {csv_path}...")
    lines = converter.csv_lines(
        csv_path,
        max_rows=args.max_rows,
        attack_only=args.attack_only,
        attack_types=args.attack_types,
    )

    if not lines:
        print("No attack rows found. Try:")
        print("  --attack-only to skip BENIGN rows")
        print("  --filter DDOS to select specific attack types")
        print("  Check the CSV has a 'Label' column")
        sys.exit(1)

    attack_counts = converter.summary(lines)
    print_banner(str(csv_path), len(lines), attack_counts, args)

    total_sent = 0
    for loop_n in range(args.loop):
        if args.loop > 1:
            print(f"  Loop {loop_n + 1}/{args.loop}")

        if args.output:
            output_path = args.output
            if args.loop > 1:
                p = Path(output_path)
                stem = p.stem
                output_path = str(p.with_name(f"{stem}_loop{loop_n + 1}{p.suffix}"))
            converter.write_lines(lines, output_path)
            print(f"  ✓ Wrote {len(lines)} syslog lines to {output_path}")
        else:
            sent = converter.feed_udp(
                lines,
                host=args.host,
                port=args.port,
                delay=args.delay,
            )
            total_sent += sent
            print(f"  ✓ Sent {sent} CEF syslog events to {args.host}:{args.port}")

    if not args.output:
        print(f"\n  Total sent: {total_sent}")
    print()


if __name__ == "__main__":
    main()
