"""Real over-the-wire HTTP benchmark for the /score endpoint.

Unlike scripts/benchmark_local.py (in-process pipeline call) or a
TestClient-based check (in-process ASGI transport, no socket), this
script launches a real `uvicorn` worker as a subprocess and drives it
with real HTTP requests over a real TCP socket via `httpx`. This is
the correct way to reproduce an "API HTTP latency / throughput" claim
that's meant to include network + ASGI + HTTP framing overhead.

Usage:
  python scripts/api_benchmark_http.py [--requests N] [--warmup N] [--concurrency N]

Writes results to models/benchmark_api_http.json (same shape convention
as models/benchmark_local.json) so the numbers are a checked-in artifact,
not a one-off terminal paste.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "models" / "benchmark_api_http.json"

HOST = "127.0.0.1"
PORT = 8099
TOKEN = "benchmark-token"
BASE_URL = f"http://{HOST}:{PORT}"

PAYLOAD = {
    "source_type": "syslog",
    "timestamp": "2026-01-15T11:07:53Z",
    "host": "prod-srv01",
    "raw": "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root from 10.0.0.5 port 44382 ssh2",
}


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[idx]


def wait_for_health(client: httpx.Client, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = client.get(f"{BASE_URL}/health", timeout=2.0)
            if r.status_code in (200, 503) and r.json().get("models_loaded", {}).get("scorer"):
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"API did not become ready in {timeout_s}s (last error: {last_err})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1],
        help="One or more concurrency levels to benchmark, e.g. --concurrency 1 4 8 16",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["LOGFILTER_API_TOKEN"] = TOKEN
    env["PYTHONPATH"] = str(ROOT / "src")
    # Benchmark measures the scoring pipeline only, not ES archiving or
    # distributed rate limiting. Elasticsearch is disabled by rewriting the
    # password key to an empty value in the scratch config below (see the
    # placeholder removal there), because config.yaml declares ES_PASSWORD
    # mandatory via "${ES_PASSWORD:?...}" and load_config() now fails closed
    # when it is unset or empty.
    env["LOGFILTER_METRICS_TOKEN"] = TOKEN
    env.pop("REDIS_URL", None)

    # app.py loads config from the hardcoded relative path "config/config.yaml"
    # (relative to CWD), so run uvicorn from a scratch dir with a config.yaml
    # whose rate limit is high enough not to 429 mid-benchmark, rather than
    # mutating the tracked config file.
    import shutil
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="api_benchmark_"))
    (workdir / "config").mkdir()
    (workdir / "models").symlink_to(ROOT / "models")
    src_cfg = ROOT / "config" / "config.yaml"
    cfg_text = src_cfg.read_text() if src_cfg.exists() else ""
    if "rate_limit_per_minute" in cfg_text:
        cfg_text = re.sub(
            r"rate_limit_per_minute:\s*\d+", "rate_limit_per_minute: 100000", cfg_text
        )
    else:
        cfg_text += "\napi:\n  rate_limit_per_minute: 100000\n"

    # Strip the mandatory "${ES_PASSWORD:?...}" placeholder entirely so the scratch
    # config needs no Elasticsearch credentials. Leaving it in place would abort
    # startup now that load_config() fails closed on an unset or empty value.
    cfg_text = re.sub(
        r'^(\s*)password:\s*"\$\{ES_PASSWORD:\?[^"]*\}"\s*$',
        r'\1password: ""',
        cfg_text,
        flags=re.MULTILINE,
    )
    (workdir / "config" / "config.yaml").write_text(cfg_text)

    print(f"Starting uvicorn subprocess on {HOST}:{PORT} (cwd={workdir}) ...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "logfilter.api.app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=str(workdir),
        env=env,
    )

    try:
        headers = {"X-API-Token": TOKEN}
        limits = httpx.Limits(
            max_connections=max(args.concurrency) + 5,
            max_keepalive_connections=max(args.concurrency) + 5,
        )

        with httpx.Client(limits=limits) as warm_client:
            wait_for_health(warm_client)
            print(f"Warming up ({args.warmup} requests) ...")
            for _ in range(args.warmup):
                warm_client.post(f"{BASE_URL}/score", json=PAYLOAD, headers=headers, timeout=30.0)

        def one_request(client: httpx.Client) -> float | None:
            t0 = time.perf_counter()
            try:
                r = client.post(f"{BASE_URL}/score", json=PAYLOAD, headers=headers, timeout=30.0)
                r.raise_for_status()
            except httpx.HTTPError:
                return None
            return (time.perf_counter() - t0) * 1000

        runs = []
        for concurrency in args.concurrency:
            print(f"Timing {args.requests} requests at concurrency={concurrency} ...")
            with (
                httpx.Client(limits=limits) as client,
                ThreadPoolExecutor(max_workers=concurrency) as pool,
            ):
                t_start = time.perf_counter()
                raw_results = list(pool.map(one_request, [client] * args.requests))
                total_s = time.perf_counter() - t_start

            latencies_ms = [v for v in raw_results if v is not None]
            failures = len(raw_results) - len(latencies_ms)

            run = {
                "concurrency": concurrency,
                "requests": args.requests,
                "failures": failures,
                "total_time_seconds": total_s,
                "throughput_events_per_second": len(latencies_ms) / total_s if total_s else 0.0,
            }
            if latencies_ms:
                run["latency_ms"] = {
                    "mean": statistics.mean(latencies_ms),
                    "p50": percentile(latencies_ms, 50),
                    "p95": percentile(latencies_ms, 95),
                    "p99": percentile(latencies_ms, 99),
                    "min": min(latencies_ms),
                    "max": max(latencies_ms),
                }
            if failures:
                print(
                    f"  WARNING: {failures}/{args.requests} requests failed at concurrency={concurrency}"
                )
            runs.append(run)

        result = {
            "warmup": args.warmup,
            "transport": "real HTTP over TCP socket (uvicorn subprocess + httpx client)",
            "runs": runs,
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2))

        print(json.dumps(result, indent=2))
        print(f"\nWrote results to {OUTPUT_PATH}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
