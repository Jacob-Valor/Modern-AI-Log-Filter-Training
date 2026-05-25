"""Locust-based benchmark runner for the LogFilter scoring API."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, cast

try:
    locust_module = importlib.import_module("locust")
    locust_env_module = importlib.import_module("locust.env")
    HttpUser: Any = locust_module.HttpUser
    between: Any = locust_module.between
    task: Any = locust_module.task
    Environment: Any = locust_env_module.Environment
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without dev extras
    LOCUST_IMPORT_ERROR: ModuleNotFoundError | None = exc

    class HttpUser:  # type: ignore[no-redef]
        """Fallback base class so smoke imports work before dev extras are installed."""

    def between(_min_wait: float, _max_wait: float) -> None:
        return None

    def task(_weight: int = 1):
        def decorator(func: Any) -> Any:
            return func

        return decorator

    Environment = None  # type: ignore[assignment]
else:
    LOCUST_IMPORT_ERROR = None


DEFAULT_HOST = "http://localhost:8080"
DEFAULT_USERS = 10
DEFAULT_SPAWN_RATE = 2
DEFAULT_DURATION_SECONDS = 60
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_COUNT = 10
SYSLOG_PAYLOAD = (
    "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root "
    "from 10.0.0.5 port 44382 ssh2"
)


@dataclass
class BenchmarkCounters:
    """Request metadata that Locust does not aggregate as event counts."""

    total_events: int = 0
    endpoint_events: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    status_codes: Counter[str] = field(default_factory=Counter)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, endpoint: str, event_count: int, status_code: int | None) -> None:
        with self._lock:
            self.total_events += event_count
            self.endpoint_events[endpoint] += event_count
            status_key = str(status_code) if status_code else "connection_error"
            self.status_codes[f"{endpoint}:{status_key}"] += 1


COUNTERS = BenchmarkCounters()


class LogFilterBenchmarkUser(HttpUser):
    """Locust user that exercises both single-event and batch scoring endpoints."""

    wait_time = between(0.05, 0.2)
    api_token: str | None = os.getenv("LOGFILTER_API_TOKEN")
    batch_size: int = DEFAULT_BATCH_SIZE
    batch_count: int = DEFAULT_BATCH_COUNT

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-API-Token"] = self.api_token
        return headers

    @classmethod
    def configure(cls, api_token: str | None, batch_size: int, batch_count: int) -> None:
        cls.api_token = api_token
        cls.batch_size = batch_size
        cls.batch_count = batch_count

    @staticmethod
    def single_payload() -> dict[str, str]:
        return {"raw": SYSLOG_PAYLOAD, "source_type": "syslog"}

    @classmethod
    def batch_payload(cls) -> dict[str, list[dict[str, str]]]:
        return {"events": [cls.single_payload() for _ in range(cls.batch_size)]}

    @task(2)
    def score_single_event(self) -> None:
        with self.client.post(
            "/score",
            json=self.single_payload(),
            headers=self.headers,
            name="/score",
            catch_response=True,
        ) as response:
            status_code = getattr(response, "status_code", None)
            COUNTERS.record("/score", 1, status_code)
            if status_code != 200:
                response.failure(f"unexpected status {status_code}")

    @task(1)
    def score_batch_events(self) -> None:
        payload = self.batch_payload()
        for _ in range(self.batch_count):
            with self.client.post(
                "/score/batch",
                json=payload,
                headers=self.headers,
                name="/score/batch",
                catch_response=True,
            ) as response:
                status_code = getattr(response, "status_code", None)
                COUNTERS.record("/score/batch", self.batch_size, status_code)
                if status_code != 200:
                    response.failure(f"unexpected status {status_code}")


@dataclass(frozen=True)
class BenchmarkConfig:
    host: str
    api_token: str | None
    users: int
    spawn_rate: float
    duration: int
    batch_size: int
    batch_count: int


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark the LogFilter API with Locust.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API host URL")
    parser.add_argument(
        "--api-token",
        default=os.getenv("LOGFILTER_API_TOKEN"),
        help="API token for X-API-Token; defaults to LOGFILTER_API_TOKEN",
    )
    parser.add_argument("--users", type=int, default=DEFAULT_USERS, help="Concurrent users")
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=DEFAULT_SPAWN_RATE,
        help="Users spawned per second",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Events per /score/batch request; API maximum is 200",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=DEFAULT_BATCH_COUNT,
        help="Batch requests sent each time the batch task runs",
    )
    args = parser.parse_args(argv)

    if args.users < 1:
        parser.error("--users must be at least 1")
    if args.spawn_rate <= 0:
        parser.error("--spawn-rate must be greater than 0")
    if args.duration < 1:
        parser.error("--duration must be at least 1 second")
    if not 1 <= args.batch_size <= 200:
        parser.error("--batch-size must be between 1 and 200")
    if args.batch_count < 1:
        parser.error("--batch-count must be at least 1")

    return BenchmarkConfig(
        host=args.host.rstrip("/"),
        api_token=args.api_token,
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        batch_count=args.batch_count,
    )


def percentile(entry: Any, percentile_value: float) -> float:
    if entry.num_requests == 0:
        return 0.0
    return float(entry.get_response_time_percentile(percentile_value))


def endpoint_entries(stats: Any) -> list[Any]:
    return [entry for (_name, method), entry in sorted(stats.entries.items()) if method == "POST"]


def format_report(config: BenchmarkConfig, stats: Any, elapsed_seconds: float) -> str:
    total = stats.total
    elapsed = max(elapsed_seconds, 0.001)
    total_requests = total.num_requests
    failed_requests = total.num_failures
    requests_per_second = total_requests / elapsed
    events_per_second = COUNTERS.total_events / elapsed
    error_rate = (failed_requests / total_requests * 100.0) if total_requests else 0.0

    lines = [
        "LOGFILTER_BENCHMARK_REPORT",
        f"host: {config.host}",
        f"users: {config.users}",
        f"spawn_rate: {config.spawn_rate}",
        f"duration_seconds: {config.duration}",
        f"batch_size: {config.batch_size}",
        f"batch_count: {config.batch_count}",
        f"total_requests: {total_requests}",
        f"failed_requests: {failed_requests}",
        f"requests_sec: {requests_per_second:.2f}",
        f"events_sec: {events_per_second:.2f}",
        f"error_rate_percent: {error_rate:.2f}",
        f"p50_ms: {percentile(total, 0.50):.2f}",
        f"p95_ms: {percentile(total, 0.95):.2f}",
        f"p99_ms: {percentile(total, 0.99):.2f}",
        "endpoint,requests,failures,events,requests_sec,p50_ms,p95_ms,p99_ms,error_rate_percent",
    ]

    for entry in endpoint_entries(stats):
        endpoint_elapsed_rps = entry.num_requests / elapsed
        endpoint_error_rate = (
            entry.num_failures / entry.num_requests * 100.0 if entry.num_requests else 0.0
        )
        lines.append(
            ",".join(
                [
                    entry.name,
                    str(entry.num_requests),
                    str(entry.num_failures),
                    str(COUNTERS.endpoint_events[entry.name]),
                    f"{endpoint_elapsed_rps:.2f}",
                    f"{percentile(entry, 0.50):.2f}",
                    f"{percentile(entry, 0.95):.2f}",
                    f"{percentile(entry, 0.99):.2f}",
                    f"{endpoint_error_rate:.2f}",
                ]
            )
        )

    if COUNTERS.status_codes:
        lines.append("status_codes:")
        for status_key, count in sorted(COUNTERS.status_codes.items()):
            lines.append(f"  {status_key}: {count}")

    return "\n".join(lines)


def ensure_locust_available() -> None:
    if LOCUST_IMPORT_ERROR is None:
        return
    raise SystemExit(
        "locust is required for benchmarking. Install dev extras with: pip install -e '.[dev]'"
    ) from LOCUST_IMPORT_ERROR


def run_benchmark(config: BenchmarkConfig) -> int:
    ensure_locust_available()
    gevent = importlib.import_module("gevent")

    LogFilterBenchmarkUser.configure(config.api_token, config.batch_size, config.batch_count)
    environment = cast(Any, Environment)(user_classes=[LogFilterBenchmarkUser], host=config.host)
    runner = environment.create_local_runner()

    start = time.perf_counter()
    try:
        runner.start(user_count=config.users, spawn_rate=config.spawn_rate)
        gevent.spawn_later(config.duration, runner.quit)
        runner.greenlet.join()
    except KeyboardInterrupt:
        runner.quit()
    finally:
        elapsed = time.perf_counter() - start
        runner.stop()

    sys.stdout.write(format_report(config, environment.stats, elapsed))
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    return run_benchmark(config)


if __name__ == "__main__":
    raise SystemExit(main())
