"""Coverage for the defect-remediation paths added to previously untested branches.

Each test here corresponds to a specific fixed defect and asserts observable
behaviour, not implementation detail. They live together because the fixes share
a theme: every one of them replaces a path that silently produced a plausible but
wrong value with one that fails loudly or degrades explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest  # type: ignore[import-not-found]

from logfilter.api import security as api_security
from logfilter.http_gateway import RESERVED_ENVELOPE_KEYS, SignedHttpGatewayClient
from logfilter.models.flow_classifier import FlowClassifier
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES
from logfilter.security import network as netsec

# ── Flow classifier: must not fabricate a score with no backend ────────────────


def _write_flow_bundle(directory: Path, *, with_model: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_names.json").write_text(json.dumps(list(FLOW_FEATURE_NAMES)))
    (directory / "scaler.json").write_text(
        json.dumps(
            {
                "type": "MaxAbsScaler",
                "n_features_in": len(FLOW_FEATURE_NAMES),
                "scale": [1.0] * len(FLOW_FEATURE_NAMES),
            }
        )
    )
    if with_model:
        (directory / "flow_classifier.onnx").write_bytes(b"not-a-real-onnx-graph")


def test_flow_classifier_raises_when_bundle_has_no_model(tmp_path) -> None:
    """Feature names + a matching scaler but no model must not silently score 0.5.

    The ONNX-width validation used to sit inside ``if self._session is not None``,
    so this state skipped validation entirely and every event got a fabricated 0.5.
    """
    bundle = tmp_path / "flow"
    _write_flow_bundle(bundle, with_model=False)

    classifier = FlowClassifier(
        model_path=bundle / "flow_classifier.onnx",
        scaler_path=bundle / "scaler.json",
        feature_names_path=bundle / "feature_names.json",
    )
    with pytest.raises(ValueError, match="no usable inference backend"):
        classifier._load()


def test_flow_classifier_rejects_reordered_feature_names(tmp_path) -> None:
    bundle = tmp_path / "flow"
    _write_flow_bundle(bundle, with_model=True)
    reordered = list(FLOW_FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    (bundle / "feature_names.json").write_text(json.dumps(reordered))

    classifier = FlowClassifier(
        model_path=bundle / "flow_classifier.onnx",
        scaler_path=bundle / "scaler.json",
        feature_names_path=bundle / "feature_names.json",
    )
    with pytest.raises(ValueError, match="feature artifact order"):
        classifier._load()


def test_flow_classifier_rejects_scaler_width_mismatch(tmp_path) -> None:
    bundle = tmp_path / "flow"
    _write_flow_bundle(bundle, with_model=True)
    wrong = len(FLOW_FEATURE_NAMES) + 5
    (bundle / "scaler.json").write_text(
        json.dumps({"type": "MaxAbsScaler", "n_features_in": wrong, "scale": [1.0] * wrong})
    )

    classifier = FlowClassifier(
        model_path=bundle / "flow_classifier.onnx",
        scaler_path=bundle / "scaler.json",
        feature_names_path=bundle / "feature_names.json",
    )
    with pytest.raises(ValueError, match="scaler width"):
        classifier._load()


# ── Signed gateway: canonical string and envelope key ownership ───────────────


class _RecordingClient:
    """Captures the request and replays a canned acknowledgement."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> Any:
        self.calls.append({"url": url, "body": content, "headers": headers})

        class _Response:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return self._payload

        return _Response(self.receipt)

    def close(self) -> None:
        return None


def _gateway(
    client: _RecordingClient, url: str = "https://gw.example/block"
) -> SignedHttpGatewayClient:
    return SignedHttpGatewayClient(
        url=url,
        hmac_secret="x" * 32,
        client=client,  # type: ignore[arg-type]
    )


def test_gateway_rejects_short_secret() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        SignedHttpGatewayClient(url="https://gw.example/block", hmac_secret="tooshort")


def test_gateway_rejects_non_https_url() -> None:
    with pytest.raises(ValueError, match="absolute https"):
        SignedHttpGatewayClient(url="http://gw.example/block", hmac_secret="x" * 32)


def test_gateway_rejects_mismatched_cert_pair(tmp_path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("cert")
    with pytest.raises(ValueError, match="configured together"):
        SignedHttpGatewayClient(
            url="https://gw.example/block", hmac_secret="x" * 32, client_cert_file=str(cert)
        )


def test_gateway_rejects_missing_ca_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SignedHttpGatewayClient(
            url="https://gw.example/block",
            hmac_secret="x" * 32,
            ca_file=str(tmp_path / "absent.pem"),
        )


def test_gateway_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        SignedHttpGatewayClient(
            url="https://gw.example/block", hmac_secret="x" * 32, timeout_seconds=0
        )


@pytest.mark.parametrize("bad_id", ["", "x" * 129])
def test_gateway_rejects_out_of_range_request_id(bad_id: str) -> None:
    client = _RecordingClient({"request_id": bad_id, "status": "accepted"})
    gateway = _gateway(client)
    with pytest.raises(ValueError, match="1-128 characters"):
        gateway.send(request_id=bad_id, payload={"action": "block_ip"})


@pytest.mark.parametrize("reserved", sorted(RESERVED_ENVELOPE_KEYS))
def test_gateway_rejects_reserved_envelope_keys(reserved: str) -> None:
    """A payload must never be able to shadow the signed request_id."""
    client = _RecordingClient({"request_id": "rid", "status": "accepted"})
    gateway = _gateway(client)
    with pytest.raises(ValueError, match="reserved keys"):
        gateway.send(request_id="rid", payload={"action": "block_ip", reserved: "attacker"})


def test_gateway_accepts_duplicate_receipt() -> None:
    client = _RecordingClient({"request_id": "rid", "status": "duplicate"})
    gateway = _gateway(client)
    receipt = gateway.send(request_id="rid", payload={"action": "block_ip"})
    assert receipt.status == "duplicate"


def test_gateway_envelope_keys_always_come_from_the_gateway() -> None:
    client = _RecordingClient({"request_id": "rid", "status": "accepted"})
    gateway = _gateway(client)

    gateway.send(request_id="rid", payload={"action": "block_ip", "target": "10.0.0.1"})

    sent = json.loads(client.calls[0]["body"])
    assert sent["request_id"] == "rid"
    assert sent["schema_version"] == "1.0"
    assert client.calls[0]["headers"]["Idempotency-Key"] == "rid"


def test_gateway_rejects_receipt_with_mismatched_request_id() -> None:
    client = _RecordingClient({"request_id": "other", "status": "accepted"})
    gateway = _gateway(client)
    with pytest.raises(RuntimeError, match="request_id mismatch"):
        gateway.send(request_id="rid", payload={"action": "block_ip"})


def test_gateway_rejects_non_accepted_status() -> None:
    client = _RecordingClient({"request_id": "rid", "status": "queued"})
    gateway = _gateway(client)
    with pytest.raises(RuntimeError, match="did not durably accept"):
        gateway.send(request_id="rid", payload={"action": "block_ip"})


def test_gateway_rejects_unparseable_receipt() -> None:
    class _BadJson(_RecordingClient):
        def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> Any:
            self.calls.append({"url": url, "body": content, "headers": headers})

            class _Response:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"unexpected": "shape"}

            return _Response()

    gateway = _gateway(_BadJson({"request_id": "rid", "status": "accepted"}))
    with pytest.raises(RuntimeError, match="invalid acknowledgement"):
        gateway.send(request_id="rid", payload={"action": "block_ip"})


def test_gateway_does_not_close_an_injected_client() -> None:
    client = _RecordingClient({"request_id": "rid", "status": "accepted"})
    gateway = _gateway(client)
    gateway.close()  # must not raise; the gateway does not own the client


# ── Rate limiter: the window map must stay bounded ────────────────────────────


def test_in_memory_rate_limiter_sweeps_stale_client_windows() -> None:
    limiter = api_security.InMemoryRateLimiter({})
    limiter.enforce("client-a", 5, now=0.0)
    assert "client-a" in limiter._windows

    # A much later request from a different client triggers the sweep; the stale
    # entry is dropped instead of accumulating forever.
    limiter.enforce("client-b", 5, now=3600.0)

    assert "client-a" not in limiter._windows
    assert "client-b" in limiter._windows


def test_in_memory_rate_limiter_raises_past_limit() -> None:
    limiter = api_security.InMemoryRateLimiter({})
    for _ in range(3):
        limiter.enforce("client", 3, now=1.0)
    with pytest.raises(api_security.AccessDenied) as excinfo:
        limiter.enforce("client", 3, now=1.0)
    assert excinfo.value.status_code == 429


def test_require_configured_token_fails_closed_when_unset() -> None:
    with pytest.raises(api_security.AccessDenied) as excinfo:
        api_security.require_configured_token(
            "anything", "", not_configured_detail="unset", invalid_detail="bad"
        )
    assert excinfo.value.status_code == 403


def test_require_configured_token_rejects_wrong_token() -> None:
    with pytest.raises(api_security.AccessDenied) as excinfo:
        api_security.require_configured_token(
            "wrong", "r" * 32, not_configured_detail="unset", invalid_detail="bad"
        )
    assert excinfo.value.status_code == 401


def test_require_configured_token_accepts_matching_token() -> None:
    api_security.require_configured_token(
        "r" * 32, "r" * 32, not_configured_detail="unset", invalid_detail="bad"
    )


def test_require_configured_token_rejects_empty_provided_value() -> None:
    with pytest.raises(api_security.AccessDenied) as excinfo:
        api_security.require_configured_token(
            "", "r" * 32, not_configured_detail="unset", invalid_detail="bad"
        )
    assert excinfo.value.status_code == 401


# ── Client IP handling: forwarded headers are only honoured from trusted peers ─


def test_client_ip_ignores_forwarded_header_from_untrusted_peer() -> None:
    resolved = api_security.client_ip_from_request(
        remote_addr="203.0.113.9", forwarded_for="10.0.0.1", trusted_proxies=[]
    )
    assert resolved == "203.0.113.9"


def test_client_ip_uses_forwarded_header_from_trusted_proxy() -> None:
    resolved = api_security.client_ip_from_request(
        remote_addr="10.0.0.1", forwarded_for="198.51.100.7", trusted_proxies=["10.0.0.0/8"]
    )
    assert resolved == "198.51.100.7"


def test_client_ip_rejects_malformed_forwarded_value() -> None:
    resolved = api_security.client_ip_from_request(
        remote_addr="10.0.0.1", forwarded_for="not-an-ip", trusted_proxies=["10.0.0.0/8"]
    )
    assert resolved == "10.0.0.1"


def test_client_ip_with_no_forwarded_header() -> None:
    resolved = api_security.client_ip_from_request(
        remote_addr="10.0.0.1", forwarded_for=None, trusted_proxies=["10.0.0.0/8"]
    )
    assert resolved == "10.0.0.1"


def test_parse_trusted_proxy_networks_returns_empty_for_none() -> None:
    assert api_security._parse_trusted_proxy_networks(None) == []


def test_parse_trusted_proxy_networks_accepts_csv_and_skips_garbage() -> None:
    networks = api_security._parse_trusted_proxy_networks("10.0.0.0/8, 192.168.0.0/16")
    assert len(networks) == 2
    # A malformed entry is dropped rather than aborting the whole allowlist, so one
    # bad value cannot silently disable trusted-proxy handling for the rest.
    assert api_security._parse_trusted_proxy_networks("not-a-cidr, 10.0.0.0/8") != []
    assert len(api_security._parse_trusted_proxy_networks("not-a-cidr, 10.0.0.0/8")) == 1
    assert api_security._parse_trusted_proxy_networks("not-a-cidr") == []


# ── Collector network allowlist ───────────────────────────────────────────────


def test_collector_allows_configured_cidr() -> None:
    allowlist = netsec.CIDRAllowlist.from_csv("10.0.0.0/8, 192.168.0.0/16")
    assert allowlist.allows("10.1.2.3") is True
    assert allowlist.allows("192.168.5.5") is True


def test_collector_rejects_address_outside_cidr() -> None:
    allowlist = netsec.CIDRAllowlist.from_csv("10.0.0.0/8")
    assert allowlist.allows("198.51.100.5") is False


def test_collector_rejects_malformed_address() -> None:
    allowlist = netsec.CIDRAllowlist.from_csv("10.0.0.0/8")
    with pytest.raises(ValueError):
        allowlist.allows("not-an-ip")


def test_collector_rejects_empty_allowlist() -> None:
    with pytest.raises(ValueError, match="at least one network"):
        netsec.CIDRAllowlist.from_csv("  ,  ")


# ── Endpoint / bot feature extractors must stay finite ───────────────────────


def _flow_event() -> Any:
    """A real CEF flow line, so the vector is built through the normalizer."""
    from logfilter.pipeline.normalizer import LogNormalizer

    line = (
        "CEF:0|UNB|CICFlowMeter|1.0|FLOW|Network flow observation|1|"
        "src=192.0.2.1\tdst=198.51.100.10\tdpt=443\tproto=TCP\tduration=1000\t"
        "fwdPackets=100\tbwdPackets=20\tfwdBytes=50000\tbwdBytes=10000\t"
        "flowBytesPerSec=60000\tflowPacketsPerSec=120\tpacketLengthMean=500\t"
        "packetLengthStd=50\tsynCount=10\tackCount=8\trstCount=0\t"
        "activeMean=500\tidleMean=100"
    )
    return LogNormalizer().normalize(line)


def test_flow_vector_is_finite_and_correct_width() -> None:
    from logfilter.pipeline.flow_features import extract_flow_vector

    vector = extract_flow_vector(_flow_event(), minimum_present=8)
    assert vector is not None
    assert vector.shape == (len(FLOW_FEATURE_NAMES),)
    assert np.all(np.isfinite(vector))


def test_flow_vector_returns_none_when_too_sparse() -> None:
    """Too few present fields must yield None, not a mostly-zero vector."""
    from logfilter.pipeline.flow_features import extract_flow_vector

    assert extract_flow_vector(_flow_event(), minimum_present=10_000) is None


# ── QRadar transport construction (build_qradar_sender) ───────────────────────


def test_build_qradar_sender_defaults_to_tcp() -> None:
    from logfilter.pipeline.router import SyslogSender, build_qradar_sender

    sender = build_qradar_sender({"syslog_host": "siem.example", "syslog_port": 514})
    assert isinstance(sender, SyslogSender)
    assert sender.port == 514
    sender.close()


@pytest.mark.parametrize("protocol", ["udp", "tls"])
def test_build_qradar_sender_accepts_supported_protocols(protocol: str) -> None:
    from logfilter.pipeline.router import build_qradar_sender

    sender = build_qradar_sender({"syslog_protocol": protocol})
    assert sender.protocol == protocol
    sender.close()


def test_build_qradar_sender_rejects_unknown_protocol() -> None:
    """An unknown protocol must fail closed at build time, not silently fall back."""
    from logfilter.pipeline.router import build_qradar_sender

    with pytest.raises(ValueError, match="must be udp, tcp, tls, or https_ack"):
        build_qradar_sender({"syslog_protocol": "carrier-pigeon"})


def test_build_qradar_sender_builds_acknowledged_sender_for_https_ack() -> None:
    from logfilter.pipeline.router import AcknowledgedHttpSender, build_qradar_sender

    sender = build_qradar_sender(
        {
            "syslog_protocol": "https_ack",
            "gateway": {"url": "https://gw.example/qradar", "hmac_secret": "x" * 32},
        }
    )
    assert isinstance(sender, AcknowledgedHttpSender)


def test_build_qradar_sender_rejects_https_ack_with_bad_gateway() -> None:
    """https_ack with an unusable gateway must not silently degrade to plain syslog."""
    from logfilter.pipeline.router import build_qradar_sender

    with pytest.raises(ValueError):
        build_qradar_sender(
            {"syslog_protocol": "https_ack", "gateway": {"url": "", "hmac_secret": ""}}
        )


# ── Auto-response execution gating ────────────────────────────────────────────


def _high_priority_event(*, source_ip: str = "203.0.113.10") -> Any:
    from logfilter.pipeline.events import ScoredEvent

    return ScoredEvent(
        source_type="firewall",
        timestamp="",
        host="fw",
        raw="attack",
        normalized_text="attack",
        fields={"src": source_ip},
        ai_threat_score=0.95,
        ai_priority="HIGH",
        classifier_score=0.95,
    )


class _RecordingFirewall:
    def __init__(self) -> None:
        self.blocked: list[tuple[str, str, str]] = []
        self.quarantined: list[tuple[str, str, str]] = []

    def block_ip(self, ip: str, reason: str, action_id: str) -> None:
        self.blocked.append((ip, reason, action_id))

    def quarantine_host(self, host: str, reason: str, action_id: str) -> None:
        self.quarantined.append((host, reason, action_id))


class _RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def notify(self, message: str, priority: str, action_id: str) -> None:
        self.sent.append((message, priority, action_id))


def test_dry_run_never_executes() -> None:
    from logfilter.response.auto_response import AutoResponseEngine

    firewall = _RecordingFirewall()
    engine = AutoResponseEngine(firewall=firewall, notifier=None, dry_run=True)
    actions = engine.decide(_high_priority_event())

    results = engine.execute(actions)

    assert firewall.blocked == []
    assert all(result.status == "recommended" for result in results)


def test_kill_switch_suppresses_armed_actions() -> None:
    """The kill switch must win over an armed engine."""
    from logfilter.response.auto_response import AutoResponseEngine

    firewall = _RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=_RecordingNotifier(),
        dry_run=False,
        kill_switch=True,
        allowed_networks=("203.0.113.0/24",),
    )
    results = engine.execute(engine.decide(_high_priority_event()))

    assert firewall.blocked == []
    assert all(result.status == "suppressed" for result in results)


def test_armed_mode_blocks_allowed_ip_and_dedupes_repeat() -> None:
    """An executed action must not be repeated, even across identical calls."""
    from logfilter.response.auto_response import AutoResponseEngine

    firewall = _RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=_RecordingNotifier(),
        dry_run=False,
        kill_switch=False,
        allowed_networks=("203.0.113.0/24",),
    )
    event = _high_priority_event()

    engine.execute(engine.decide(event, event_ref="ref-1"))
    assert len(firewall.blocked) == 1, "armed mode should have executed the block exactly once"
    executed_ids = {action_id for _, _, action_id in firewall.blocked}
    assert executed_ids, "the firewall must have received a signed action id"

    second = engine.execute(engine.decide(event, event_ref="ref-1"))
    assert len(firewall.blocked) == 1, "idempotent action_id must not execute a second time"
    assert all(r.status == "duplicate" for r in second)


def test_armed_mode_refuses_ip_outside_allowlist() -> None:
    from logfilter.response.auto_response import AutoResponseEngine

    firewall = _RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=_RecordingNotifier(),
        dry_run=False,
        kill_switch=False,
        allowed_networks=("10.0.0.0/8",),
    )
    engine.execute(engine.decide(_high_priority_event(source_ip="198.51.100.9")))
    assert firewall.blocked == []


def test_executed_id_memory_is_bounded() -> None:
    """_executed_ids must not grow without bound over the process lifetime."""
    from logfilter.response.auto_response import AutoResponseEngine

    engine = AutoResponseEngine(dry_run=True)
    cap = engine.MAX_EXECUTED_IDS
    for index in range(cap + 250):
        engine._remember_executed(f"id-{index}")

    assert len(engine._executed_ids) <= cap
    # The most recent ids are the ones retained.
    assert f"id-{cap + 249}" in engine._executed_ids
    assert "id-0" not in engine._executed_ids


# ── HTTP feature helpers ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        ({"a": "1"}, "a=1"),
        (["x", "y"], "x&y"),
        (42, "42"),
    ],
)
def test_body_text_coerces_every_input_shape(raw: Any, expected: str) -> None:
    from logfilter.pipeline.http_features import _body_text

    assert _body_text(raw) == expected


def test_nested_body_text_is_flattened() -> None:
    from logfilter.pipeline.http_features import _body_text

    # A nested list is joined without repeating the parent key.
    assert _body_text({"user": "bob", "ids": [1, 2]}) == "user=bob&ids=1&2"


def test_finite_float_rejects_non_numeric_and_non_finite() -> None:
    from logfilter.pipeline.http_features import _finite_float

    assert _finite_float("2.5") == pytest.approx(2.5)
    with pytest.raises(ValueError, match="must be numeric"):
        _finite_float("not-a-number")
    with pytest.raises(ValueError, match="must be finite"):
        _finite_float(float("inf"))


def test_entropy_of_empty_string_is_zero() -> None:
    from logfilter.pipeline.http_features import _entropy

    assert _entropy("") == 0.0
    assert _entropy("aaaa") == pytest.approx(0.0, abs=1e-9)
    assert _entropy("ab") > 0.0
