"""Tests for the syslog collector boundary logic."""

from __future__ import annotations

from logfilter import collector as collector_module
from logfilter.collector import CollectorSettings, SyslogCollector
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
        )
    )
    producer = FakeProducer()
    collector.producer = producer  # type: ignore[assignment]
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
