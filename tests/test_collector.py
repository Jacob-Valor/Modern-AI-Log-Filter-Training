"""Tests for the syslog collector boundary logic."""

from __future__ import annotations

import json
import socket
from typing import cast

import pytest

from logfilter import collector as collector_module
from logfilter.collector import CollectorSettings, SyslogCollector
from logfilter.kafka.producer import LogProducer
from logfilter.security.network import CIDRAllowlist


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    def send(self, **kwargs) -> None:
        self.sent.append(kwargs)

    def close(self) -> None:
        self.closed = True


def _collector(allowed_cidrs: str = "10.0.0.0/24") -> tuple[SyslogCollector, FakeProducer]:
    collector = SyslogCollector(
        CollectorSettings(
            listen_host="127.0.0.1",
            listen_port=5140,
            allowed_cidrs=CIDRAllowlist.from_csv(allowed_cidrs),
            bootstrap_servers="localhost:9092",
            raw_topic="raw-logs",
            kafka_config={},
            max_tcp_line_bytes=16,
            max_tcp_connections=2,
        )
    )
    producer = FakeProducer()
    collector.producer = cast(LogProducer, producer)
    return collector, producer


def test_publish_rejects_disallowed_peer() -> None:
    collector, producer = _collector("10.0.0.0/24")

    collector.publish("Jan 15 host sshd: Failed password", peer_host="192.0.2.10", protocol="udp")

    assert producer.sent == []


def test_publish_rejects_malformed_peer_address() -> None:
    collector, producer = _collector("10.0.0.0/24")

    collector.publish("raw", peer_host="not-an-ip", protocol="udp")

    assert producer.sent == []


def test_settings_reads_environment_and_config(monkeypatch) -> None:
    monkeypatch.setenv("SYSLOG_LISTEN_HOST", "127.0.0.1")
    monkeypatch.setenv("SYSLOG_LISTEN_PORT", "15140")
    monkeypatch.setenv("SYSLOG_ALLOWED_CIDRS", "10.0.0.0/24")
    monkeypatch.setattr(
        collector_module,
        "load_config",
        lambda: {
            "kafka": {
                "bootstrap_servers": "kafka:29092",
                "topics": {"raw_logs": "raw"},
            }
        },
    )

    settings = collector_module._settings()

    assert settings.listen_host == "127.0.0.1"
    assert settings.listen_port == 15140
    assert settings.allowed_cidrs.allows("10.0.0.5")
    assert settings.bootstrap_servers == "kafka:29092"
    assert settings.raw_topic == "raw"
    assert settings.kafka_config["bootstrap_servers"] == "kafka:29092"


def test_publish_uses_normalized_host_when_present() -> None:
    collector, producer = _collector("10.0.0.0/24")

    collector.publish(
        "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root",
        peer_host="10.0.0.5",
        protocol="tcp",
    )

    assert producer.sent[0]["host"] == "prod-srv01"
    assert producer.sent[0]["source_type"] == "syslog"
    assert producer.sent[0]["metadata"] == {
        "collector_peer": "10.0.0.5",
        "collector_protocol": "tcp",
    }


def test_publish_uses_peer_when_host_unknown() -> None:
    collector, producer = _collector("10.0.0.0/24")

    collector.publish("unstructured log text", peer_host="10.0.0.5", protocol="udp")

    assert producer.sent[0]["host"] == "10.0.0.5"


def test_publish_ignores_empty_messages() -> None:
    collector, producer = _collector("10.0.0.0/24")

    collector.publish("  ", peer_host="10.0.0.5", protocol="udp")

    assert producer.sent == []


class FakeTCPConnection:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False
        self.timeout = 0.0

    def __enter__(self) -> FakeTCPConnection:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recv(self, _size: int) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True


def test_tcp_client_drops_oversized_unterminated_buffer() -> None:
    collector, producer = _collector("10.0.0.0/24")
    connection = FakeTCPConnection([b"x" * 17])

    collector.serve_tcp_client(cast(socket.socket, connection), "10.0.0.5")

    assert producer.sent == []
    assert connection.closed is True


def test_tcp_client_drops_oversized_line() -> None:
    collector, producer = _collector("10.0.0.0/24")
    connection = FakeTCPConnection([b"x" * 17 + b"\n"])

    collector.serve_tcp_client(cast(socket.socket, connection), "10.0.0.5")

    assert producer.sent == []


def test_tcp_client_accepts_rfc6587_octet_counted_frames() -> None:
    collector, producer = _collector("10.0.0.0/24")
    first = b"host1 event1"
    second = b"host2 event2"
    payload = f"{len(first)} ".encode() + first + f"{len(second)} ".encode() + second
    connection = FakeTCPConnection([payload[:8], payload[8:]])

    collector.serve_tcp_client(cast(socket.socket, connection), "10.0.0.5")

    assert [record["raw_log"] for record in producer.sent] == [
        "host1 event1",
        "host2 event2",
    ]


def test_tcp_connection_slots_reject_when_full() -> None:
    collector, _producer = _collector("10.0.0.0/24")
    assert collector._tcp_slots.acquire(blocking=False)
    assert collector._tcp_slots.acquire(blocking=False)

    connection = FakeTCPConnection([])

    assert collector._start_tcp_client_if_slot_available(
        cast(socket.socket, connection), "10.0.0.5"
    ) is None
    assert connection.closed is True


class BrokenProducer:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    def send(self, **_kwargs) -> None:
        raise RuntimeError("kafka down")

    def close(self) -> None:
        self.closed = True


def test_publish_spools_on_producer_failure(tmp_path) -> None:
    spool_path = tmp_path / "spool.ndjson"
    collector = SyslogCollector(
        CollectorSettings(
            listen_host="127.0.0.1",
            listen_port=5140,
            allowed_cidrs=CIDRAllowlist.from_csv("10.0.0.0/24"),
            bootstrap_servers="localhost:9092",
            raw_topic="raw-logs",
            kafka_config={},
            spool_path=spool_path,
            max_spool_bytes=1_000_000,
        )
    )
    producer = BrokenProducer()
    collector.producer = cast(LogProducer, producer)

    collector.publish("test event", peer_host="10.0.0.5", protocol="udp")

    assert producer.sent == []
    assert spool_path.exists()
    lines = spool_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["raw"] == "test event"
    assert record["host"] == "10.0.0.5"


def test_publish_raises_when_no_spool_and_producer_fails() -> None:
    collector = SyslogCollector(
        CollectorSettings(
            listen_host="127.0.0.1",
            listen_port=5140,
            allowed_cidrs=CIDRAllowlist.from_csv("10.0.0.0/24"),
            bootstrap_servers="localhost:9092",
            raw_topic="raw-logs",
            kafka_config={},
        )
    )
    producer = BrokenProducer()
    collector.producer = cast(LogProducer, producer)

    with pytest.raises(RuntimeError, match="kafka down"):
        collector.publish("test event", peer_host="10.0.0.5", protocol="udp")


def test_bounded_spool_enforces_max_bytes(tmp_path) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=25)

    spool.write({"n": 1})
    spool.write({"n": 2})
    spool.write({"n": 3})

    lines = spool_path.read_text().strip().split("\n")
    assert len(lines) < 3
    records = [json.loads(line) for line in lines]
    assert records[-1]["n"] == 3


def test_drain_replays_and_removes_successful_records(tmp_path) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)

    spool.write({"raw": "event1", "host": "h1"})
    spool.write({"raw": "event2", "host": "h2"})

    replayed: list[dict] = []

    def callback(record: dict) -> None:
        replayed.append(record)

    count = spool.drain(callback)
    assert count == 2
    assert len(replayed) == 2
    assert replayed[0]["raw"] == "event1"
    assert replayed[1]["raw"] == "event2"
    assert spool_path.read_text().strip() == ""


def test_drain_keeps_records_that_raise(tmp_path) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)

    spool.write({"raw": "event1", "host": "h1"})
    spool.write({"raw": "event2", "host": "h2"})

    def callback(record: dict) -> None:
        if record["raw"] == "event1":
            raise RuntimeError("boom")

    count = spool.drain(callback)
    assert count == 1
    lines = spool_path.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["raw"] == "event1"


def test_drain_quarantines_malformed_spool_lines(tmp_path) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool_path.write_text('{"raw": "good"}\nnot-json\n', encoding="utf-8")

    replayed: list[dict] = []

    def callback(record: dict) -> None:
        replayed.append(record)

    count = spool.drain(callback)

    assert count == 1
    assert replayed == [{"raw": "good"}]
    assert spool_path.read_text(encoding="utf-8") == ""
    assert (tmp_path / "spool.ndjson.bad").read_text(encoding="utf-8") == "not-json\n"


def _reset_collector_counters() -> None:
    from logfilter.collector import (
        _collector_dropped_total,
        _collector_published_total,
        _collector_received_total,
    )
    for c in (_collector_received_total, _collector_published_total):
        c._metrics.clear()
    _collector_dropped_total._metrics.clear()


def test_prometheus_received_increments_on_publish() -> None:
    from logfilter.collector import _collector_received_total

    _reset_collector_counters()
    collector, _producer = _collector("10.0.0.0/24")

    collector.publish("Jan 15 host sshd: Failed password", peer_host="10.0.0.5", protocol="tcp")

    assert _collector_received_total.labels(protocol="tcp")._value.get() == 1


def test_prometheus_published_increments_on_success() -> None:
    from logfilter.collector import _collector_published_total

    _reset_collector_counters()
    collector, _producer = _collector("10.0.0.0/24")

    collector.publish("Jan 15 host sshd: Failed password", peer_host="10.0.0.5", protocol="udp")

    assert _collector_published_total.labels(protocol="udp")._value.get() == 1


def test_prometheus_dropped_empty_increments_on_empty() -> None:
    from logfilter.collector import _collector_dropped_total

    _reset_collector_counters()
    collector, _producer = _collector("10.0.0.0/24")

    collector.publish("  ", peer_host="10.0.0.5", protocol="udp")

    assert _collector_dropped_total.labels(reason="empty")._value.get() == 1


def test_prometheus_dropped_peer_increments_on_disallowed() -> None:
    from logfilter.collector import _collector_dropped_total

    _reset_collector_counters()
    collector, _producer = _collector("10.0.0.0/24")

    collector.publish("raw", peer_host="192.0.2.10", protocol="tcp")

    assert _collector_dropped_total.labels(reason="disallowed_peer")._value.get() == 1


def test_prometheus_dropped_kafka_increments_on_producer_failure(tmp_path) -> None:
    from logfilter.collector import _collector_dropped_total

    _reset_collector_counters()
    spool_path = tmp_path / "spool.ndjson"
    collector = SyslogCollector(
        CollectorSettings(
            listen_host="127.0.0.1",
            listen_port=5140,
            allowed_cidrs=CIDRAllowlist.from_csv("10.0.0.0/24"),
            bootstrap_servers="localhost:9092",
            raw_topic="raw-logs",
            kafka_config={},
            spool_path=spool_path,
            max_spool_bytes=1_000_000,
        )
    )
    collector.producer = cast(LogProducer, BrokenProducer())

    collector.publish("test event", peer_host="10.0.0.5", protocol="udp")

    assert _collector_dropped_total.labels(reason="kafka_failure")._value.get() == 1


def test_prometheus_received_increments_before_drops() -> None:
    from logfilter.collector import (
        _collector_dropped_total,
        _collector_published_total,
        _collector_received_total,
    )

    _reset_collector_counters()
    collector, _producer = _collector("10.0.0.0/24")

    collector.publish("  ", peer_host="10.0.0.5", protocol="udp")

    assert _collector_received_total.labels(protocol="udp")._value.get() == 1
    assert _collector_dropped_total.labels(reason="empty")._value.get() == 1
    assert _collector_published_total.labels(protocol="udp")._value.get() == 0


def test_metrics_server_returns_200_on_slash_metrics() -> None:
    import socket

    from prometheus_client import start_http_server

    server = start_http_server(0)
    port = server[0].server_address[1]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.sendall(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            resp = s.recv(4096)
        assert b"200 OK" in resp
    finally:
        server[0].shutdown()


def test_metrics_server_exposes_collector_counters() -> None:
    import socket

    from prometheus_client import start_http_server

    from logfilter.collector import (
        _collector_dropped_total,
        _collector_published_total,
        _collector_received_total,
    )

    _reset_collector_counters()
    _collector_received_total.labels(protocol="tcp").inc(3)
    _collector_published_total.labels(protocol="tcp").inc(2)
    _collector_dropped_total.labels(reason="empty").inc(1)

    server = start_http_server(0)
    port = server[0].server_address[1]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.sendall(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            body = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                body += chunk
        assert b"logfilter_collector_received_total" in body
        assert b"logfilter_collector_published_total" in body
        assert b"logfilter_collector_dropped_total" in body
    finally:
        server[0].shutdown()
