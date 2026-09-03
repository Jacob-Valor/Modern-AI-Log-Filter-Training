"""Behavioral contracts for signed, acknowledged security gateways."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest

from logfilter.http_gateway import SignedHttpGatewayClient
from logfilter.pipeline.router import AcknowledgedHttpSender
from logfilter.response.auto_response import SignedResponseGateway

_SECRET = "s" * 32


def _client(handler) -> SignedHttpGatewayClient:
    transport = httpx.MockTransport(handler)
    return SignedHttpGatewayClient(
        url="https://gateway.example/v1/events",
        hmac_secret=_SECRET,
        client=httpx.Client(transport=transport),
    )


def test_gateway_signs_canonical_body_and_requires_matching_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        request_id = request.headers["Idempotency-Key"]
        timestamp = request.headers["X-LogFilter-Timestamp"]
        body_digest = hashlib.sha256(request.content).hexdigest()
        canonical = f"{timestamp}\n{request_id}\n{body_digest}".encode()
        expected = hmac.new(_SECRET.encode(), canonical, hashlib.sha256).hexdigest()
        assert request.headers["X-LogFilter-Signature"] == f"v1={expected}"
        assert json.loads(request.content)["request_id"] == request_id
        return httpx.Response(200, json={"request_id": request_id, "status": "accepted"})

    receipt = _client(handler).send(request_id="event-1", payload={"message": "leef"})

    assert receipt.request_id == "event-1"
    assert receipt.status == "accepted"


def test_gateway_treats_receiver_duplicate_as_durable_success() -> None:
    client = _client(
        lambda request: httpx.Response(
            200,
            json={"request_id": request.headers["Idempotency-Key"], "status": "duplicate"},
        )
    )

    assert client.send(request_id="event-1", payload={}).status == "duplicate"


def test_gateway_rejects_mismatched_or_non_durable_receipt() -> None:
    mismatch = _client(
        lambda request: httpx.Response(200, json={"request_id": "other", "status": "accepted"})
    )
    with pytest.raises(RuntimeError, match="request_id mismatch"):
        mismatch.send(request_id="event-1", payload={})

    rejected = _client(
        lambda request: httpx.Response(
            200,
            json={"request_id": request.headers["Idempotency-Key"], "status": "queued"},
        )
    )
    with pytest.raises(RuntimeError, match="did not durably accept"):
        rejected.send(request_id="event-1", payload={})


def test_gateway_requires_https_and_strong_secret() -> None:
    with pytest.raises(ValueError, match="https"):
        SignedHttpGatewayClient(url="http://gateway.example", hmac_secret=_SECRET)
    with pytest.raises(ValueError, match="32 bytes"):
        SignedHttpGatewayClient(url="https://gateway.example", hmac_secret="short")


def test_qradar_sender_uses_raw_log_reference_as_idempotency_key() -> None:
    calls = []

    class RecordingClient:
        def send(self, *, request_id, payload):
            calls.append((request_id, payload))

        def close(self) -> None:
            pass

    sender = AcknowledgedHttpSender(RecordingClient())  # type: ignore[arg-type]
    sender.send("LEEF:2.0|event", "HIGH", event_id="a" * 64)

    assert calls[0][0] == "a" * 64
    assert calls[0][1]["event_id"] == "a" * 64
    assert calls[0][1]["message"] == "LEEF:2.0|event"


def test_response_adapter_preserves_authoritative_action_id() -> None:
    calls = []

    class RecordingClient:
        def send(self, *, request_id, payload):
            calls.append((request_id, payload))

        def close(self) -> None:
            pass

    adapter = SignedResponseGateway(RecordingClient())  # type: ignore[arg-type]
    adapter.block_ip("203.0.113.7", "high score", "action-1")

    assert calls == [
        (
            "action-1",
            {"action": "block_ip", "target": "203.0.113.7", "reason": "high score"},
        )
    ]
