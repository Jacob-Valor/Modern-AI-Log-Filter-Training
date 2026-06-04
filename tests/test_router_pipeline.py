"""Tests for QRadar routing decisions and sender orchestration."""

from __future__ import annotations

from logfilter.pipeline import router as router_module
from logfilter.pipeline.router import LogRouter, SyslogSender
from logfilter.pipeline.scorer import ScoredEvent


def _scored(priority: str = "HIGH", score: float = 0.9) -> ScoredEvent:
    return ScoredEvent(
        source_type="syslog",
        timestamp="2026-01-15T11:07:53Z",
        host="prod",
        raw="raw",
        normalized_text="normalized",
        ai_priority=priority,
        ai_threat_score=score,
    )


class FakeSender:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[str, str]] = []
        self.closed = False

    def send(self, message: str, priority: str) -> None:
        if self.fail:
            raise OSError("send failed")
        self.sent.append((message, priority))

    def send_batch(self, messages: list[tuple[str, str]]) -> int:
        self.sent.extend(messages)
        return len(messages)

    def close(self) -> None:
        self.closed = True


def test_route_forwards_in_enrich_only_mode() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "enrich_only"}}, sender=sender)

    decision = router.route("leef", _scored("INFO", 0.01))

    assert decision.forward_to_qradar
    assert sender.sent == [("leef", "INFO")]


def test_route_suppresses_info_in_suppress_low_mode() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "suppress_low"}}, sender=sender)

    decision = router.route("leef", _scored("INFO", 0.01))

    assert not decision.forward_to_qradar
    assert sender.sent == []


def test_route_returns_decision_even_when_sender_fails() -> None:
    router = LogRouter({"qradar": {"mode": "enrich_only"}}, sender=FakeSender(fail=True))

    decision = router.route("leef", _scored("HIGH", 0.9))

    assert decision.forward_to_qradar


def test_route_batch_sends_only_forwarded_events() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "suppress_low"}}, sender=sender)

    decisions = router.route_batch(
        ["low", "high"],
        [_scored("INFO", 0.01), _scored("HIGH", 0.9)],
    )

    assert [decision.forward_to_qradar for decision in decisions] == [False, True]
    assert sender.sent == [("high", "HIGH")]


def test_router_close_closes_sender() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "enrich_only"}}, sender=sender)

    router.close()

    assert sender.closed


def test_syslog_sender_disconnect_swallows_close_error() -> None:
    class BadSocket:
        def close(self) -> None:
            raise OSError("already closed")

    sender = SyslogSender("localhost", protocol="udp")
    sender._sock = BadSocket()  # type: ignore[assignment]

    sender._disconnect()

    assert sender._sock is None


class FakeSocket:
    instances: list[FakeSocket] = []

    def __init__(self, *args, **kwargs) -> None:
        self.timeout = None
        self.connected = None
        self.sent_to = []
        self.sent_all = []
        self.closed = False
        FakeSocket.instances.append(self)

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, address) -> None:
        self.connected = address

    def sendto(self, payload, address) -> None:
        self.sent_to.append((payload, address))

    def sendall(self, payload) -> None:
        self.sent_all.append(payload)

    def close(self) -> None:
        self.closed = True


def test_syslog_sender_udp_send_uses_sendto(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    sender = SyslogSender("qradar", port=1514, protocol="udp")
    sender.send("message", "HIGH")

    fake = FakeSocket.instances[0]
    assert fake.sent_to[0][1] == ("qradar", 1514)
    assert b"message" in fake.sent_to[0][0]


def test_syslog_sender_tcp_send_uses_octet_counted_frame(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    sender = SyslogSender("qradar", port=1514, protocol="tcp")
    sender.send("message", "MEDIUM")

    fake = FakeSocket.instances[0]
    assert fake.connected == ("qradar", 1514)
    assert fake.sent_all[0].split(b" ", 1)[0].isdigit()
    assert b"message" in fake.sent_all[0]


def test_syslog_sender_send_batch_counts_successes(monkeypatch) -> None:
    sender = SyslogSender("qradar", protocol="udp")
    calls = []

    def fake_send(message: str, priority: str) -> None:
        calls.append((message, priority))
        if message == "bad":
            raise OSError("failed")

    monkeypatch.setattr(sender, "send", fake_send)

    sent = sender.send_batch([("ok", "HIGH"), ("bad", "LOW")])

    assert sent == 1
    assert calls == [("ok", "HIGH"), ("bad", "LOW")]


def test_syslog_sender_tls_protocol_creates_context(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    sender = SyslogSender("qradar", port=1514, protocol="tls")
    assert sender._tls_context is not None


def test_syslog_sender_tls_wraps_socket(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    class FakeTLSContext:
        def wrap_socket(self, sock, server_hostname=None):
            wrapped = FakeSocket()
            wrapped.wrapped = True
            return wrapped

    sender = SyslogSender("qradar", port=1514, protocol="tls")
    sender._tls_context = FakeTLSContext()
    sender.send("message", "HIGH")

    assert FakeSocket.instances[1].wrapped is True


def test_syslog_sender_tcp_exception_triggers_retry(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    sender = SyslogSender("qradar", port=1514, protocol="tcp")
    sender._sock = FakeSocket()

    class BrokenSocket:
        def sendall(self, payload):
            raise BrokenPipeError("connection lost")

        def close(self):
            pass

    sender._sock = BrokenSocket()

    sender.send("message", "HIGH")
    assert len(FakeSocket.instances) == 2


def test_syslog_sender_udp_exception_triggers_retry(monkeypatch) -> None:
    FakeSocket.instances = []
    monkeypatch.setattr(router_module.socket, "socket", FakeSocket)

    sender = SyslogSender("qradar", port=1514, protocol="udp")

    class BrokenSocket:
        def sendto(self, payload, address):
            raise OSError("network down")

        def close(self):
            pass

    sender._sock = BrokenSocket()

    sender.send("message", "HIGH")
    assert len(FakeSocket.instances) == 1


def test_router_decide_suppress_low_non_info() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "suppress_low"}}, sender=sender)

    decision = router.decide(_scored("LOW", 0.3))
    assert decision.forward_to_qradar is True

    decision = router.decide(_scored("MEDIUM", 0.6))
    assert decision.forward_to_qradar is True


def test_router_route_batch_no_forward_events() -> None:
    sender = FakeSender()
    router = LogRouter({"qradar": {"mode": "suppress_low"}}, sender=sender)

    decisions = router.route_batch(
        ["info1", "info2"],
        [_scored("INFO", 0.01), _scored("INFO", 0.01)],
    )

    assert all(not d.forward_to_qradar for d in decisions)
    assert sender.sent == []
