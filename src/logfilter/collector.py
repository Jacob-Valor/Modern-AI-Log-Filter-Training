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

    Durability / performance properties:
      - Writes are O(1): a persistent append handle plus in-memory byte and
        line counters mean the common path never re-reads the whole file. The
        full scan only happens when the bound is actually exceeded.
      - Every full rewrite goes to a sibling temp file and is moved into place
        with :func:`os.replace`, so a SIGKILL mid-rewrite cannot leave a
        truncated or mid-line spool.
      - ``drain`` replays records *outside* the lock: the replay callback
        blocks (Kafka send retries), so holding the lock across it would stall
        every syslog handler thread for the whole outage.
    """

    # Quarantine is bounded to this fraction of *max_bytes*, with a floor so
    # that small test/dev spools still get a usable quarantine file.
    _BAD_BYTES_FRACTION = 0.1
    _BAD_BYTES_FLOOR = 1_048_576

    def __init__(
        self,
        path: Path,
        max_bytes: int,
        *,
        bad_max_bytes: int | None = None,
        drain_batch_size: int = 1000,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if drain_batch_size < 1:
            raise ValueError("drain_batch_size must be >= 1")
        self._path = path
        self._bad_path = path.with_name(f"{path.name}.bad")
        self._max_bytes = max_bytes
        if bad_max_bytes is None:
            bad_max_bytes = max(int(max_bytes * self._BAD_BYTES_FRACTION), self._BAD_BYTES_FLOOR)
        if bad_max_bytes < 1:
            raise ValueError("bad_max_bytes must be >= 1")
        self._bad_max_bytes = bad_max_bytes
        self._drain_batch_size = drain_batch_size
        self._lock = threading.Lock()
        self._handle: Any | None = None
        self._bytes = 0
        self._lines = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._scan_state()
        _collector_spool_depth.set(self._lines)

    # ── internal state ─────────────────────────────────────────────────────────

    def _scan_state(self) -> None:
        """Recompute byte/line counters from disk. Caller must hold ``self._lock``."""
        self._close_handle()
        if not self._path.exists():
            self._bytes = 0
            self._lines = 0
            return
        self._bytes = self._path.stat().st_size
        with self._path.open("rb") as f:
            self._lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))

    def _sync_state(self) -> None:
        """Rescan only when the on-disk size disagrees with our counters.

        Cheap in the steady state (a single ``stat``); it catches an external
        writer, a replaced file, or a file left behind by a crashed process.
        Caller must hold ``self._lock``.
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            size = -1
        if size != self._bytes:
            self._scan_state()

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _read_lines(self) -> list[str]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return f.readlines()

    def _rewrite(self, lines: list[str]) -> None:
        """Atomically replace the spool contents with *lines*.

        Writes a temp file in the same directory and ``os.replace()``s it into
        position, so a crash mid-rewrite leaves the previous spool intact
        instead of a truncated one. Caller must hold ``self._lock``.
        """
        self._close_handle()
        tmp_path = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}")
        payload = [line.encode("utf-8") for line in lines]
        try:
            with tmp_path.open("wb") as f:
                f.writelines(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise
        self._bytes = sum(len(chunk) for chunk in payload)
        self._lines = len(lines)

    def _compact(self) -> None:
        """Drop oldest lines until the spool fits within *max_bytes*.

        O(n) in the number of retained lines, but only runs once per bound
        crossing. Caller must hold ``self._lock``.
        """
        lines = self._read_lines()
        total = self._bytes
        dropped = 0
        while lines and total > self._max_bytes:
            total -= len(lines.pop(0).encode("utf-8"))
            dropped += 1
        if dropped:
            logger.warning("Spool bound exceeded; evicting oldest records", dropped=dropped)
            self._rewrite(lines)

    # ── public API ─────────────────────────────────────────────────────────────

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._sync_state()
            if self._handle is None:
                self._handle = self._path.open("a", encoding="utf-8")
            self._handle.write(line)
            self._handle.flush()
            self._bytes += len(line.encode("utf-8"))
            self._lines += 1
            if self._bytes > self._max_bytes:
                self._compact()
            _collector_spool_depth.set(self._lines)

    def drain(self, callback: Callable[[dict[str, Any]], None]) -> int:
        """Replay up to ``drain_batch_size`` spooled records via *callback*.

        The spool lock is held only to snapshot and to rewrite the file — never
        across ``callback``, which blocks on Kafka sends during an outage.
        Returns the number of records successfully replayed.
        """
        with self._lock:
            self._sync_state()
            lines = self._read_lines()
            if not lines:
                return 0
            batch = lines[: self._drain_batch_size]

        replayed = 0
        failed_indexes: list[int] = []
        quarantined: list[str] = []
        for index, line in enumerate(batch):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                quarantined.append(line)
                _collector_dropped_total.labels(reason="spool_malformed").inc()
                continue
            try:
                callback(record)
                replayed += 1
            except Exception:  # noqa: BLE001
                failed_indexes.append(index)

        with self._lock:
            if quarantined:
                self._quarantine(quarantined)
            self._settle_drain(batch, failed_indexes)

        return replayed

    # ── drain helpers (all called with ``self._lock`` held) ───────────────────

    def _settle_drain(self, batch: list[str], failed_indexes: list[int]) -> None:
        """Remove replayed records from the spool, keeping failed ones.

        Re-reads the file because ``write()`` may have appended while the
        callback was running. Only the still-matching prefix of *batch* is
        removed, so records compacted away meanwhile are not miscounted and
        freshly written records are never dropped.
        """
        current = self._read_lines()
        matched = 0
        while (
            matched < len(batch)
            and matched < len(current)
            and current[matched] == batch[matched]
        ):
            matched += 1

        failed = set(failed_indexes)
        # Failed records inside the surviving prefix must be put back; failed
        # records compacted away while we replayed are gone by design.
        keep_head = [batch[i] for i in sorted(failed) if i < matched]
        remaining = keep_head + current[matched:]
        self._rewrite(remaining)
        _collector_spool_depth.set(len(remaining))

    def _quarantine(self, lines: list[str]) -> None:
        """Append malformed lines to the ``.bad`` sibling, bounded by size."""
        with self._bad_path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
        self._enforce_bad_bound()

    def _enforce_bad_bound(self) -> None:
        """Evict oldest quarantined lines once the ``.bad`` file is too big."""
        try:
            size = self._bad_path.stat().st_size
        except OSError:
            return
        if size <= self._bad_max_bytes:
            return
        with self._bad_path.open("r", encoding="utf-8") as f:
            bad_lines = f.readlines()
        total = size
        dropped = 0
        while bad_lines and total > self._bad_max_bytes:
            total -= len(bad_lines.pop(0).encode("utf-8"))
            dropped += 1
        logger.warning(
            "Quarantine bound exceeded; evicting oldest malformed records",
            dropped=dropped,
            max_bytes=self._bad_max_bytes,
        )
        tmp_path = self._bad_path.with_name(f"{self._bad_path.name}.tmp.{os.getpid()}")
        payload = [line.encode("utf-8") for line in bad_lines]
        try:
            with tmp_path.open("wb") as f:
                f.writelines(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._bad_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def close(self) -> None:
        """Release the append handle."""
        with self._lock:
            self._close_handle()


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

    def _publish_tcp_payload(self, payload: bytes, peer_host: str) -> None:
        with telemetry.start_as_current_span(
            "collector.syslog.receive_tcp",
            {
                "network.transport": "tcp",
                "logfilter.collector.peer_host": peer_host,
            },
        ) as span:
            raw = payload.decode("utf-8", errors="replace")
            try:
                self.publish(raw, peer_host=peer_host, protocol="tcp")
            except Exception as exc:  # noqa: BLE001
                telemetry.record_exception(span, exc)
                logger.error("Failed to publish TCP syslog event", error=str(exc))

    def _pending_octet_frame(self, buffer: bytes) -> bool:
        if not buffer[:1].isdigit():
            return False
        space_index = buffer.find(b" ")
        newline_index = buffer.find(b"\n")
        if space_index == -1 or (newline_index != -1 and newline_index < space_index):
            return False
        prefix = buffer[:space_index]
        if not prefix.isdigit():
            return False
        frame_len = int(prefix)
        return (
            frame_len <= self.settings.max_tcp_line_bytes
            and len(buffer) < space_index + 1 + frame_len
        )

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
                while buffer:
                    if buffer[:1].isdigit():
                        space_index = buffer.find(b" ")
                        newline_index = buffer.find(b"\n")
                        if space_index != -1 and (
                            newline_index == -1 or space_index < newline_index
                        ):
                            prefix = buffer[:space_index]
                            if prefix.isdigit():
                                frame_len = int(prefix)
                                if frame_len > self.settings.max_tcp_line_bytes:
                                    logger.warning(
                                        "Dropped oversized TCP syslog frame",
                                        peer_host=peer_host,
                                        max_bytes=self.settings.max_tcp_line_bytes,
                                    )
                                    return
                                frame_end = space_index + 1 + frame_len
                                if len(buffer) < frame_end:
                                    break
                                self._publish_tcp_payload(
                                    buffer[space_index + 1 : frame_end],
                                    peer_host,
                                )
                                buffer = buffer[frame_end:]
                                continue

                    if b"\n" not in buffer:
                        break
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > self.settings.max_tcp_line_bytes:
                        logger.warning(
                            "Dropped oversized TCP syslog line",
                            peer_host=peer_host,
                            max_bytes=self.settings.max_tcp_line_bytes,
                        )
                        return
                    self._publish_tcp_payload(line, peer_host)

                if len(buffer) > self.settings.max_tcp_line_bytes and not self._pending_octet_frame(
                    buffer
                ):
                    logger.warning(
                        "Dropped oversized TCP syslog buffer",
                        peer_host=peer_host,
                        max_bytes=self.settings.max_tcp_line_bytes,
                    )
                    return

            if buffer.strip():
                if self._pending_octet_frame(buffer):
                    logger.warning(
                        "Dropped incomplete TCP syslog frame",
                        peer_host=peer_host,
                        max_bytes=self.settings.max_tcp_line_bytes,
                    )
                    return
                if len(buffer) > self.settings.max_tcp_line_bytes:
                    logger.warning(
                        "Dropped oversized TCP syslog line",
                        peer_host=peer_host,
                        max_bytes=self.settings.max_tcp_line_bytes,
                    )
                    return
                self._publish_tcp_payload(buffer, peer_host)

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
                # drain() replays at most one bounded batch per call; keep going
                # until the backlog is empty or a whole batch fails to replay
                # (Kafka still down — retry on the next tick).
                while not self.stop_event.is_set():
                    if self._drain_spool() == 0:
                        break
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
            threads.append(threading.Thread(target=self._spool_drain_loop, daemon=True))
        for thread in threads:
            thread.start()

        logger.info("Syslog collector started", topic=self.settings.raw_topic)
        try:
            while not self.stop_event.wait(1.0):
                pass
        finally:
            self.stop_event.set()
            for thread in threads:
                thread.join(timeout=5.0)
            self.producer.close()
            if self._spool is not None:
                self._spool.close()
            logger.info("Syslog collector stopped")


def main() -> None:  # pragma: no cover
    collector = SyslogCollector(_settings())

    def _shutdown(_signum, _frame):  # noqa: ANN001
        collector.stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    collector.run()


if __name__ == "__main__":  # pragma: no cover
    main()
