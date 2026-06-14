from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from logfilter.config import load_config
from logfilter.pipeline.normalizer import LogNormalizer, NormalizedEvent
from logfilter.pipeline.scorer import LogScorer

OUTPUT_PATH = ROOT / "models" / "benchmark_local.json"

SYSLOG_MESSAGES = [
    "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root from 10.0.0.5 port 44382 ssh2",
    "Jan 15 11:07:54 prod-srv01 sshd[124]: Accepted publickey for admin from 192.168.1.100 port 22 ssh2",
    "Jan 15 11:07:55 prod-srv01 sudo: admin : TTY=pts/0 ; PWD=/home/admin ; USER=root ; COMMAND=/bin/cat /etc/shadow",
    "Jan 15 11:07:56 prod-srv01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=10.0.0.100 DST=10.0.0.1 PROTO=TCP SPT=443 DPT=80",
    "Jan 15 11:07:57 prod-srv01 apache2[1234]: 10.0.0.200 - - [15/Jan/2026:11:07:57 +0000] \"GET /admin/config HTTP/1.1\" 403 287",
    "Jan 15 11:07:58 prod-srv01 nginx[2345]: 172.16.0.9 - - [15/Jan/2026:11:07:58 +0000] \"POST /login HTTP/1.1\" 401 182",
    "<34>1 2026-01-15T11:07:59Z fw01 app01 1234 ID47 [exampleSDID@32473 iut=3 eventSource=Application] SSH login failed for user root",
    "<134>Jan 15 11:08:00 prod-srv01 firewall: src=10.0.0.44 dst=10.0.0.1 proto=tcp sport=51514 dport=3389 action=deny",
    '{"EventID":4625,"ProviderName":"Microsoft-Windows-Security-Auditing","Message":"An account failed to log on","Computer":"WIN-SRV01"}',
    '{"EventID":4688,"ProviderName":"Microsoft-Windows-Security-Auditing","Message":"A new process has been created","Computer":"WIN-SRV01"}',
    '{"Records":[{"eventSource":"signin.amazonaws.com","eventName":"ConsoleLogin","sourceIPAddress":"203.0.113.10","userAgent":"Mozilla/5.0","responseElements":{"ConsoleLogin":"Failure"}}]}',
    '{"eventTime":"2026-01-15T11:08:01Z","eventSource":"ec2.amazonaws.com","eventName":"AuthorizeSecurityGroupIngress","awsRegion":"us-east-1","sourceIPAddress":"198.51.100.7"}',
    '{"timestamp":"2026-01-15T11:08:02Z","host":"endpoint-01","event_type":"process","process_name":"powershell.exe","command_line":"powershell -enc SQBFAFgA"}',
    '{"timestamp":"2026-01-15T11:08:03Z","host":"endpoint-02","event_type":"file","file_path":"C:\\Windows\\System32\\drivers\\etc\\hosts","action":"modified"}',
    "CEF:0|Cisco|ASA|9.16|302013|Built outbound TCP connection|5|src=10.0.0.15 dst=198.51.100.20 spt=51514 dpt=443 proto=TCP act=permit",
    "CEF:0|Palo Alto Networks|PAN-OS|10.2|THREAT-1|Virus Detected|8|src=10.0.0.45 dst=203.0.113.55 fileName=dropper.exe severity=high",
    "Mar 15 11:08:04 db01 postgres[4321]: connection authorized: user=appdb database=logs host=10.0.0.30",
    "Mar 15 11:08:05 auth01 sshd[888]: Invalid user oracle from 203.0.113.12 port 60214",
    "Mar 15 11:08:06 proxy01 haproxy[222]: backend app_pool has no server available!",
    "Mar 15 11:08:07 app01 java[999]: WARN Authentication token expired for session 7f3c2d",
    "Mar 15 11:08:08 backup01 cron[1111]: (root) CMD (rsync -a /data /backup)",
    "Mar 15 11:08:09 bastion01 sshd[777]: Received disconnect from 198.51.100.88 port 55231:11: disconnected by user",
    "Mar 15 11:08:10 ids01 suricata[909]: [1:2010935:3] ET SCAN Potential SSH Scan [Classification: Attempted Information Leak] [Priority: 2] {TCP} 203.0.113.99:51514 -> 10.0.0.12:22",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = round((p / 100.0) * (len(ordered) - 1))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def make_batch(raw_batch: list[str], normalizer: LogNormalizer) -> list[NormalizedEvent]:
    return [normalizer.normalize(raw) for raw in raw_batch]


def build_raw_events(total: int) -> list[str]:
    return [SYSLOG_MESSAGES[i % len(SYSLOG_MESSAGES)] for i in range(total)]


def run_benchmark(samples: int, batch_size: int, warmup: int) -> dict[str, Any]:
    config = load_config(ROOT / "config" / "config.yaml")
    normalizer = LogNormalizer()
    scorer = LogScorer(config)
    scorer.preload_models()

    warmup_events = build_raw_events(warmup * batch_size)
    for raw_batch in chunked(warmup_events, batch_size):
        normalized_batch = make_batch(raw_batch, normalizer)
        scorer.score_batch(normalized_batch)

    timed_events = build_raw_events(samples)
    batch_latencies_ms: list[float] = []

    for raw_batch in chunked(timed_events, batch_size):
        normalized_batch = make_batch(raw_batch, normalizer)
        batch_start = time.perf_counter()
        scorer.score_batch(normalized_batch)
        batch_end = time.perf_counter()
        batch_latencies_ms.append((batch_end - batch_start) * 1000.0)
    total_seconds = sum(batch_latencies_ms) / 1000.0

    total_events = samples
    avg_per_event_latency_ms = (total_seconds / total_events * 1000.0) if total_events else 0.0

    report: dict[str, Any] = {
        "samples": samples,
        "batch_size": batch_size,
        "warmup_batches": warmup,
        "timed_batches": len(batch_latencies_ms),
        "total_events": total_events,
        "total_time_seconds": total_seconds,
        "throughput_events_per_second": (total_events / total_seconds) if total_seconds > 0 else 0.0,
        "avg_per_event_latency_ms": avg_per_event_latency_ms,
        "batch_latency_ms": {
            "mean": statistics.mean(batch_latencies_ms) if batch_latencies_ms else 0.0,
            "p50": percentile(batch_latencies_ms, 50),
            "p95": percentile(batch_latencies_ms, 95),
            "p99": percentile(batch_latencies_ms, 99),
            "min": min(batch_latencies_ms) if batch_latencies_ms else 0.0,
            "max": max(batch_latencies_ms) if batch_latencies_ms else 0.0,
        },
        "batch_latencies_ms": batch_latencies_ms,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    batch = cast(dict[str, float], report["batch_latency_ms"])

    print("\n=== Local Scoring Benchmark ===")
    print(f"samples:                  {report['samples']}")
    print(f"batch_size:               {report['batch_size']}")
    print(f"warmup_batches:           {report['warmup_batches']}")
    print(f"timed_batches:            {report['timed_batches']}")
    print(f"total_time_seconds:       {report['total_time_seconds']:.4f}")
    print(f"throughput_events_per_sec:{report['throughput_events_per_second']:.2f}")
    print(f"avg_per_event_latency_ms: {report['avg_per_event_latency_ms']:.3f}")
    print("batch_latency_ms:")
    print(f"  mean: {batch['mean']:.3f}")
    print(f"  p50:  {batch['p50']:.3f}")
    print(f"  p95:  {batch['p95']:.3f}")
    print(f"  p99:  {batch['p99']:.3f}")
    print(f"  min:  {batch['min']:.3f}")
    print(f"  max:  {batch['max']:.3f}")
    print(f"saved: {OUTPUT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local LogFilter scoring pipeline.")
    parser.add_argument("--samples", type=int, default=200, help="Timed events to score")
    parser.add_argument("--batch-size", type=int, default=50, help="Events per batch")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup batches")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    return args


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.samples, args.batch_size, args.warmup)
    print_summary(report)


if __name__ == "__main__":
    main()
