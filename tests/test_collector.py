"""Tests for the syslog collector boundary logic."""

from __future__ import annotations

import json
import socket
import threading
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

    assert (
        collector._start_tcp_client_if_slot_available(cast(socket.socket, connection), "10.0.0.5")
        is None
    )
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


def test_spool_write_does_not_rescan_the_file_per_record(tmp_path, monkeypatch) -> None:
    """Appending must be O(1): no full read/rewrite of the spool per record.

    The old implementation re-read every line, re-encoded all of them to
    re-sum their lengths, then truncated and rewrote the whole file on each
    write, so the nth write cost O(n) under the lock.
    """
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    # A pre-seeded spool of 10k lines; steady-state appends must still scan
    # nothing — the byte/line counters are maintained in memory.
    spool_path.write_text(
        "".join(json.dumps({"n": i}) + "\n" for i in range(10_000)), encoding="utf-8"
    )
    seeded_size = spool_path.stat().st_size
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=10_000_000)

    reads = {"count": 0}
    rewrites = {"count": 0}
    original_read = spool._read_lines
    original_rewrite = spool._rewrite

    def counting_read() -> list[str]:
        reads["count"] += 1
        return original_read()

    def counting_rewrite(lines: list[str]) -> None:
        rewrites["count"] += 1
        original_rewrite(lines)

    monkeypatch.setattr(spool, "_read_lines", counting_read)
    monkeypatch.setattr(spool, "_rewrite", counting_rewrite)

    for n in range(1000):
        spool.write({"n": n})

    assert reads["count"] == 0
    assert rewrites["count"] == 0
    appended = sum(len((json.dumps({"n": n}) + "\n").encode()) for n in range(1000))
    assert spool_path.stat().st_size == seeded_size + appended
    assert len(spool_path.read_text(encoding="utf-8").splitlines()) == 11_000
    spool.close()


def test_spool_compacts_only_when_bound_exceeded(tmp_path) -> None:
    """A full-file rewrite happens on bound crossings, not on every write."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    record_bytes = len((json.dumps({"n": 0}) + "\n").encode())
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=record_bytes * 5)

    rewrites: list[list[str]] = []
    original_rewrite = spool._rewrite

    def recording_rewrite(lines: list[str]) -> None:
        rewrites.append(list(lines))
        original_rewrite(lines)

    spool._rewrite = recording_rewrite  # type: ignore[method-assign]

    # Five records fit exactly; nothing is rewritten until the sixth crosses.
    for n in range(5):
        spool.write({"n": n})
    assert rewrites == []

    spool.write({"n": 5})
    assert len(rewrites) == 1
    assert [json.loads(line)["n"] for line in rewrites[0]] == [1, 2, 3, 4, 5]
    spool.close()


def test_spool_rewrite_leaves_no_partial_file_on_crash(tmp_path, monkeypatch) -> None:
    """A failure mid-rewrite must leave the previous spool intact (temp+rename)."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool.write({"raw": "survivor"})

    def boom(_lines: list[str]) -> None:
        raise OSError("simulated crash mid-rewrite")

    monkeypatch.setattr(spool, "_rewrite", boom)
    with pytest.raises(OSError, match="simulated crash"):
        spool._settle_drain(['{"raw": "survivor"}\n'], [])

    assert spool_path.read_text(encoding="utf-8") == '{"raw": "survivor"}\n'
    assert list(tmp_path.glob("*.tmp.*")) == []
    spool.close()


def test_spool_rewrite_uses_temp_file_then_rename(tmp_path, monkeypatch) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool.write({"raw": "old"})

    seen: dict[str, object] = {}
    real_replace = collector_module.os.replace

    def spy_replace(src, dst) -> None:
        # The destination still holds the old contents at rename time, and the
        # source is a sibling temp file — that is what makes this atomic.
        seen["tmp_existed"] = collector_module.os.path.exists(str(src))
        seen["old_contents"] = collector_module.os.path.exists(str(dst))
        real_replace(src, dst)
        seen["replaced"] = True

    monkeypatch.setattr(collector_module.os, "replace", spy_replace)
    spool._rewrite(['{"raw": "new"}\n'])

    assert seen["tmp_existed"] is True
    assert seen["old_contents"] is True
    assert seen["replaced"] is True
    assert spool_path.read_text(encoding="utf-8") == '{"raw": "new"}\n'
    spool.close()


def test_drain_callback_can_reenter_write_without_deadlock(tmp_path) -> None:
    """drain() must not hold the spool lock while invoking the replay callback."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool.write({"raw": "event1"})
    spool.write({"raw": "event2"})

    errors: list[BaseException] = []

    def callback(record: dict) -> None:
        # Re-entering write() from the callback would deadlock if drain held
        # the (non-reentrant) lock across the callback.
        try:
            spool.write({"raw": "from-callback"})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=lambda: spool.drain(callback))
    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive(), "drain() deadlocked: it held the lock across the callback"
    assert errors == []
    records = [json.loads(line) for line in spool_path.read_text(encoding="utf-8").splitlines()]
    assert {"raw": "from-callback"} in records
    assert {"raw": "event1"} not in records
    assert {"raw": "event2"} not in records
    spool.close()


def test_drain_replays_at_most_one_batch(tmp_path) -> None:
    """drain() bounds its work so an outage cannot stall the drain thread for hours."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000, drain_batch_size=3)
    for n in range(10):
        spool.write({"n": n})

    replayed: list[dict] = []
    assert spool.drain(replayed.append) == 3
    assert [r["n"] for r in replayed] == [0, 1, 2]
    assert len(spool_path.read_text(encoding="utf-8").strip().split("\n")) == 7

    # Successive batches drain the rest in order.
    assert spool.drain(replayed.append) == 3
    assert spool.drain(replayed.append) == 3
    assert spool.drain(replayed.append) == 1
    assert spool.drain(replayed.append) == 0
    assert [r["n"] for r in replayed] == list(range(10))
    spool.close()


def test_drain_keeps_failed_records_written_during_replay(tmp_path) -> None:
    """Concurrent writes during a replay are preserved, not clobbered."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool.write({"raw": "event1"})

    def callback(record: dict) -> None:
        spool.write({"raw": "arrived-during-replay"})

    spool.drain(callback)

    records = [json.loads(line) for line in spool_path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"raw": "arrived-during-replay"}]
    spool.close()


def test_drain_retains_failed_record_written_during_replay(tmp_path) -> None:
    """A record whose replay failed stays in the spool even if a new one lands."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000)
    spool.write({"raw": "event1"})

    def callback(_record: dict) -> None:
        spool.write({"raw": "arrived-during-replay"})
        raise RuntimeError("kafka still down")

    assert spool.drain(callback) == 0

    records = [json.loads(line) for line in spool_path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"raw": "event1"}, {"raw": "arrived-during-replay"}]
    spool.close()


def test_quarantine_file_is_bounded_and_evicts_oldest(tmp_path) -> None:
    """The .bad quarantine must not grow without bound during an outage."""
    from logfilter.collector import BoundedNDJSONSpool

    spool_path = tmp_path / "spool.ndjson"
    bad_path = tmp_path / "spool.ndjson.bad"
    spool = BoundedNDJSONSpool(path=spool_path, max_bytes=1_000_000, bad_max_bytes=40)

    for n in range(6):
        spool_path.write_text(f'not-json-{n}\n', encoding="utf-8")
        spool.drain(lambda _record: None)

    assert bad_path.stat().st_size <= 40
    contents = bad_path.read_text(encoding="utf-8").splitlines()
    # Oldest-first eviction: the most recent malformed lines survive.
    assert contents == ["not-json-3", "not-json-4", "not-json-5"]
    spool.close()


def test_quarantine_default_bound_scales_with_spool(tmp_path) -> None:
    from logfilter.collector import BoundedNDJSONSpool

    small = BoundedNDJSONSpool(path=tmp_path / "a.ndjson", max_bytes=1_000_000)
    large = BoundedNDJSONSpool(path=tmp_path / "b.ndjson", max_bytes=500_000_000)
    assert small._bad_max_bytes == 1_048_576
    assert large._bad_max_bytes == 50_000_000
    small.close()
    large.close()


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
