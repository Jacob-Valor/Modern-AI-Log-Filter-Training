"""Syslog collector entrypoint.

Receives raw syslog over UDP/TCP and publishes JSON envelopes to Kafka.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, start_http_server

from logfilter import telemetry
from logfilter.config import load_config
from logfilter.kafka.producer import LogProducer
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.security.network import CIDRAllowlist

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
_collector_received_total = Counter(
    "logfilter_collector_received_total",
    "Raw syslog messages received by collector",
    ["protocol"],
)
_collector_published_total = Counter(
    "logfilter_collector_published_total",
    "Events successfully published to Kafka",
    ["protocol"],
)
_collector_dropped_total = Counter(
    "logfilter_collector_dropped_total",
    "Events dropped by collector",
    ["reason"],
)
_collector_spool_depth = Gauge(
    "logfilter_collector_spool_depth",
    "Number of events queued in the NDJSON spool",
)


@dataclass(frozen=True)
class CollectorSettings:
    listen_host: str
    listen_port: int
    allowed_cidrs: CIDRAllowlist
    bootstrap_servers: str | list[str]
    raw_topic: str
    kafka_config: dict[str, Any]
    max_tcp_line_bytes: int = 1_048_576
    max_tcp_connections: int = 256
    spool_path: Path | None = None
    max_spool_bytes: int = 50_000_000
    spool_drain_interval: float = 30.0
    metrics_port: int = 9100


def _settings() -> CollectorSettings:
    config = load_config()
    kafka_cfg = config.get("kafka", {})
    topics = kafka_cfg.get("topics", {})
    spool_path_str = os.environ.get("SYSLOG_SPOOL_PATH")
    return CollectorSettings(
        listen_host=os.environ.get("SYSLOG_LISTEN_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("SYSLOG_LISTEN_PORT", "5140")),
        allowed_cidrs=CIDRAllowlist.from_csv(
            os.environ.get("SYSLOG_ALLOWED_CIDRS", "127.0.0.1/32,::1/128")
        ),
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        raw_topic=topics.get("raw_logs", "raw-logs"),
        kafka_config=kafka_cfg,
        max_tcp_line_bytes=int(os.environ.get("SYSLOG_MAX_TCP_LINE_BYTES", "1048576")),
        max_tcp_connections=int(os.environ.get("SYSLOG_MAX_TCP_CONNECTIONS", "256")),
        spool_path=Path(spool_path_str) if spool_path_str else None,
        max_spool_bytes=int(os.environ.get("SYSLOG_MAX_SPOOL_BYTES", "50000000")),
        spool_drain_interval=float(os.environ.get("SYSLOG_SPOOL_DRAIN_INTERVAL", "30.0")),
        metrics_port=int(os.environ.get("SYSLOG_METRICS_PORT", "9100")),
    )


class BoundedNDJSONSpool:
    """Thread-safe bounded NDJSON spool for failed producer sends.

    Writes records as one JSON object per line. When the file exceeds
    *max_bytes*, oldest lines are truncated until the file fits.
    """

    def __init__(self, path: Path, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _read_lines(self) -> list[str]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return f.readlines()

    def _write_lines(self, lines: list[str]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

    def _enforce_bound(self, lines: list[str]) -> list[str]:
        while lines:
            total = sum(len(line.encode("utf-8")) for line in lines)
            if total <= self._max_bytes:
                break
            lines.pop(0)
        return lines

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            lines = self._read_lines()
            lines.append(line)
            lines = self._enforce_bound(lines)
            self._write_lines(lines)
            _collector_spool_depth.set(len(lines))

    def drain(self, callback: Callable[[dict[str, Any]], None]) -> int:
        """Replay spooled records via *callback*; return count replayed."""
        with self._lock:
            lines = self._read_lines()
            if not lines:
                return 0
            replayed = 0
            remaining: list[str] = []
            for line in lines:
                try:
                    record = json.loads(line)
                    callback(record)
                    replayed += 1
                except Exception:  # noqa: BLE001
                    remaining.append(line)
            self._write_lines(remaining)
            _collector_spool_depth.set(len(remaining))
            return replayed


class SyslogCollector:
    """Small UDP/TCP syslog listener backed by Kafka."""

    def __init__(self, settings: CollectorSettings) -> None:
        self.settings = settings
        self.normalizer = LogNormalizer()
        self.producer = LogProducer(
            bootstrap_servers=settings.bootstrap_servers,
            topic=settings.raw_topic,
            kafka_config=settings.kafka_config,
        )
        self.stop_event = threading.Event()
        self._tcp_slots = threading.BoundedSemaphore(settings.max_tcp_connections)
        self._spool: BoundedNDJSONSpool | None = None
        if settings.spool_path:
            self._spool = BoundedNDJSONSpool(
                path=settings.spool_path,
                max_bytes=settings.max_spool_bytes,
            )

    def publish(self, raw: str, peer_host: str, protocol: str) -> None:
        _collector_received_total.labels(protocol=protocol).inc()
        with telemetry.start_as_current_span(
            "collector.syslog.publish",
            {
                "logfilter.collector.peer_host": peer_host,
                "logfilter.collector.protocol": protocol,
                "messaging.destination.name": self.settings.raw_topic,
            },
        ) as span:
            raw = raw.strip()
            if not raw:
                span.set_attribute("logfilter.collector.dropped", "empty")
                _collector_dropped_total.labels(reason="empty").inc()
                return
            try:
                allowed = self.settings.allowed_cidrs.allows(peer_host)
            except ValueError:
                allowed = False
            if not allowed:
                span.set_attribute("logfilter.collector.dropped", "disallowed_peer")
                _collector_dropped_total.labels(reason="disallowed_peer").inc()
                logger.warning("Rejected syslog event from disallowed source", peer_host=peer_host)
                return

            normalized = self.normalizer.normalize(raw)
            host = normalized.host if normalized.host != "unknown" else peer_host
            telemetry.set_span_attributes(
                span,
                {
                    "logfilter.host": host,
                    "logfilter.source_type": normalized.source_type.value,
                },
            )
            try:
                self.producer.send(
                    raw_log=raw,
                    source_type=normalized.source_type.value,
                    host=host,
                    metadata={"collector_peer": peer_host, "collector_protocol": protocol},
                )
                _collector_published_total.labels(protocol=protocol).inc()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Kafka producer failed, spooling event", error=str(exc))
                _collector_dropped_total.labels(reason="kafka_failure").inc()
                if self._spool is not None:
                    self._spool.write(
                        {
                            "raw": raw,
                            "source_type": normalized.source_type.value,
                            "host": host,
                            "metadata": {
                                "collector_peer": peer_host,
                                "collector_protocol": protocol,
                            },
                            "ingest_ts": time.time(),
                        }
                    )
                else:
                    raise

    def serve_udp(self) -> None:  # pragma: no cover
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.bind((self.settings.listen_host, self.settings.listen_port))
            logger.info(
                "UDP syslog listener started",
                host=self.settings.listen_host,
                port=self.settings.listen_port,
            )
            while not self.stop_event.is_set():
                try:
                    data, addr = sock.recvfrom(65535)
                except TimeoutError:
                    continue
                except OSError as exc:
                    if not self.stop_event.is_set():
                        logger.error("UDP listener error", error=str(exc))
                    continue

                with telemetry.start_as_current_span(
                    "collector.syslog.receive_udp",
                    {
                        "network.transport": "udp",
                        "logfilter.collector.peer_host": addr[0],
                    },
                ) as span:
                    raw = data.decode("utf-8", errors="replace")
                    try:
                        self.publish(raw, peer_host=addr[0], protocol="udp")
                    except Exception as exc:  # noqa: BLE001
                        telemetry.record_exception(span, exc)
                        logger.error("Failed to publish UDP syslog event", error=str(exc))

    def serve_tcp_client(self, conn: socket.socket, peer_host: str) -> None:  # pragma: no cover
        with conn:
            conn.settimeout(1.0)
            buffer = b""
            while not self.stop_event.is_set():
                try:
                    chunk = conn.recv(65535)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not chunk:
                    break

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > self.settings.max_tcp_line_bytes:
                        logger.warning(
                            "Dropped oversized TCP syslog line",
                            peer_host=peer_host,
                            max_bytes=self.settings.max_tcp_line_bytes,
                        )
                        return
                    with telemetry.start_as_current_span(
                        "collector.syslog.receive_tcp",
                        {
                            "network.transport": "tcp",
                            "logfilter.collector.peer_host": peer_host,
                        },
                    ) as span:
                        raw = line.decode("utf-8", errors="replace")
                        try:
                            self.publish(raw, peer_host=peer_host, protocol="tcp")
                        except Exception as exc:  # noqa: BLE001
                            telemetry.record_exception(span, exc)
                            logger.error("Failed to publish TCP syslog event", error=str(exc))
                if len(buffer) > self.settings.max_tcp_line_bytes:
                    logger.warning(
                        "Dropped oversized TCP syslog buffer",
                        peer_host=peer_host,
                        max_bytes=self.settings.max_tcp_line_bytes,
                    )
                    return

            if buffer.strip():
                if len(buffer) > self.settings.max_tcp_line_bytes:
                    logger.warning(
                        "Dropped oversized TCP syslog line",
                        peer_host=peer_host,
                        max_bytes=self.settings.max_tcp_line_bytes,
                    )
                    return
                with telemetry.start_as_current_span(
                    "collector.syslog.receive_tcp",
                    {
                        "network.transport": "tcp",
                        "logfilter.collector.peer_host": peer_host,
                    },
                ) as span:
                    raw = buffer.decode("utf-8", errors="replace")
                    try:
                        self.publish(raw, peer_host=peer_host, protocol="tcp")
                    except Exception as exc:  # noqa: BLE001
                        telemetry.record_exception(span, exc)
                        logger.error("Failed to publish TCP syslog event", error=str(exc))

    def _serve_tcp_client_with_slot(self, conn: socket.socket, peer_host: str) -> None:
        try:
            self.serve_tcp_client(conn, peer_host)
        finally:
            self._tcp_slots.release()

    def _start_tcp_client_if_slot_available(
        self,
        conn: socket.socket,
        peer_host: str,
    ) -> threading.Thread | None:
        if not self._tcp_slots.acquire(blocking=False):
            conn.close()
            logger.warning(
                "Rejected TCP syslog connection; connection limit reached",
                peer_host=peer_host,
                max_connections=self.settings.max_tcp_connections,
            )
            return None
        thread = threading.Thread(
            target=self._serve_tcp_client_with_slot,
            args=(conn, peer_host),
            daemon=True,
        )
        thread.start()
        return thread

    def serve_tcp(self) -> None:  # pragma: no cover
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1.0)
            sock.bind((self.settings.listen_host, self.settings.listen_port))
            sock.listen(128)
            logger.info(
                "TCP syslog listener started",
                host=self.settings.listen_host,
                port=self.settings.listen_port,
            )
            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                except TimeoutError:
                    continue
                except OSError as exc:
                    if not self.stop_event.is_set():
                        logger.error("TCP listener error", error=str(exc))
                    continue

                self._start_tcp_client_if_slot_available(conn, addr[0])

    def _drain_spool(self) -> int:
        if self._spool is None:
            return 0
        def _replay(record: dict[str, Any]) -> None:
            self.producer.send(
                raw_log=record["raw"],
                source_type=record.get("source_type", "generic"),
                host=record.get("host", "unknown"),
                metadata=record.get("metadata", {}),
            )
        replayed = self._spool.drain(_replay)
        if replayed:
            logger.info("Replayed spooled events", replayed=replayed)
        return replayed

    def _spool_drain_loop(self) -> None:  # pragma: no cover
        while not self.stop_event.wait(self.settings.spool_drain_interval):
            if self.stop_event.is_set():
                break
            try:
                self._drain_spool()
            except Exception as exc:  # noqa: BLE001
                logger.error("Spool drain failed", error=str(exc))

    def run(self) -> None:  # pragma: no cover
        start_http_server(self.settings.metrics_port)
        logger.info(
            "Collector metrics server started",
            port=self.settings.metrics_port,
        )

        threads = [
            threading.Thread(target=self.serve_udp, daemon=True),
            threading.Thread(target=self.serve_tcp, daemon=True),
        ]
        if self._spool is not None:
            threads.append(
                threading.Thread(target=self._spool_drain_loop, daemon=True)
            )
        for thread in threads:
            thread.start()

        logger.info("Syslog collector started", topic=self.settings.raw_topic)
        try:
            while not self.stop_event.wait(1.0):
                pass
        finally:
            self.stop_event.set()
            self.producer.close()
            logger.info("Syslog collector stopped")


def main() -> None:  # pragma: no cover
    collector = SyslogCollector(_settings())

    def _shutdown(signum, frame):  # noqa: ANN001
        collector.stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    collector.run()


if __name__ == "__main__":  # pragma: no cover
    main()
