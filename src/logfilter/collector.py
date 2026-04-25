"""Syslog collector entrypoint.

Receives raw syslog over UDP/TCP and publishes JSON envelopes to Kafka.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
from dataclasses import dataclass

import structlog

from logfilter.config import load_config
from logfilter.kafka.producer import LogProducer
from logfilter.pipeline.normalizer import LogNormalizer

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CollectorSettings:
    listen_host: str
    listen_port: int
    bootstrap_servers: str | list[str]
    raw_topic: str


def _settings() -> CollectorSettings:
    config = load_config()
    kafka_cfg = config.get("kafka", {})
    topics = kafka_cfg.get("topics", {})
    return CollectorSettings(
        listen_host=os.environ.get("SYSLOG_LISTEN_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("SYSLOG_LISTEN_PORT", "5140")),
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        raw_topic=topics.get("raw_logs", "raw-logs"),
    )


class SyslogCollector:
    """Small UDP/TCP syslog listener backed by Kafka."""

    def __init__(self, settings: CollectorSettings) -> None:
        self.settings = settings
        self.normalizer = LogNormalizer()
        self.producer = LogProducer(
            bootstrap_servers=settings.bootstrap_servers,
            topic=settings.raw_topic,
        )
        self.stop_event = threading.Event()

    def publish(self, raw: str, peer_host: str, protocol: str) -> None:
        raw = raw.strip()
        if not raw:
            return

        normalized = self.normalizer.normalize(raw)
        host = normalized.host if normalized.host != "unknown" else peer_host
        self.producer.send(
            raw_log=raw,
            source_type=normalized.source_type.value,
            host=host,
            metadata={"collector_peer": peer_host, "collector_protocol": protocol},
        )

    def serve_udp(self) -> None:
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

                raw = data.decode("utf-8", errors="replace")
                try:
                    self.publish(raw, peer_host=addr[0], protocol="udp")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to publish UDP syslog event", error=str(exc))

    def serve_tcp_client(self, conn: socket.socket, peer_host: str) -> None:
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
                    raw = line.decode("utf-8", errors="replace")
                    try:
                        self.publish(raw, peer_host=peer_host, protocol="tcp")
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to publish TCP syslog event", error=str(exc))

            if buffer.strip():
                raw = buffer.decode("utf-8", errors="replace")
                try:
                    self.publish(raw, peer_host=peer_host, protocol="tcp")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to publish TCP syslog event", error=str(exc))

    def serve_tcp(self) -> None:
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

                thread = threading.Thread(
                    target=self.serve_tcp_client,
                    args=(conn, addr[0]),
                    daemon=True,
                )
                thread.start()

    def run(self) -> None:
        threads = [
            threading.Thread(target=self.serve_udp, daemon=True),
            threading.Thread(target=self.serve_tcp, daemon=True),
        ]
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


def main() -> None:
    collector = SyslogCollector(_settings())

    def _shutdown(signum, frame):  # noqa: ANN001
        collector.stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    collector.run()


if __name__ == "__main__":
    main()
