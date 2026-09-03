"""Authenticated, acknowledged HTTPS delivery for external security gateways."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class GatewayReceipt:
    """Durable receiver acknowledgement for one idempotent request."""

    request_id: str
    status: str


class SignedHttpGatewayClient:
    """Send canonically signed JSON and require an idempotent receiver receipt."""

    _ACCEPTED_STATUSES = frozenset({"accepted", "duplicate"})

    def __init__(
        self,
        *,
        url: str,
        hmac_secret: str,
        timeout_seconds: float = 10.0,
        ca_file: str = "",
        client_cert_file: str = "",
        client_key_file: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("gateway URL must be an absolute https:// URL")
        if len(hmac_secret.encode("utf-8")) < 32:
            raise ValueError("gateway HMAC secret must contain at least 32 bytes")
        if timeout_seconds <= 0:
            raise ValueError("gateway timeout must be positive")
        if bool(client_cert_file) != bool(client_key_file):
            raise ValueError("gateway client certificate and key must be configured together")
        for label, value in (
            ("CA file", ca_file),
            ("client certificate", client_cert_file),
            ("client key", client_key_file),
        ):
            if value and not Path(value).is_file():
                raise ValueError(f"gateway {label} does not exist: {value}")

        self.url = url
        self._secret = hmac_secret.encode("utf-8")
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            verify: bool | str = ca_file or True
            cert: tuple[str, str] | None = (
                (client_cert_file, client_key_file) if client_cert_file else None
            )
            self._client = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds),
                verify=verify,
                cert=cert,
            )

    def _signed_headers(self, request_id: str, body: bytes, timestamp: int) -> dict[str, str]:
        body_digest = hashlib.sha256(body).hexdigest()
        canonical = f"{timestamp}\n{request_id}\n{body_digest}".encode()
        signature = hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()
        return {
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
            "X-LogFilter-Request-ID": request_id,
            "X-LogFilter-Timestamp": str(timestamp),
            "X-LogFilter-Signature": f"v1={signature}",
        }

    def send(self, *, request_id: str, payload: dict[str, Any]) -> GatewayReceipt:
        """Return only after the receiver confirms durable acceptance or a duplicate."""
        if not request_id or len(request_id) > 128:
            raise ValueError("gateway request_id must contain 1-128 characters")
        envelope = {"schema_version": "1.0", "request_id": request_id, **payload}
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = self._signed_headers(request_id, body, int(time.time()))
        response = self._client.post(self.url, content=body, headers=headers)
        response.raise_for_status()
        try:
            receipt = response.json()
            receipt_id = str(receipt["request_id"])
            receipt_status = str(receipt["status"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("gateway returned an invalid acknowledgement") from exc
        if receipt_id != request_id:
            raise RuntimeError("gateway acknowledgement request_id mismatch")
        if receipt_status not in self._ACCEPTED_STATUSES:
            raise RuntimeError(f"gateway did not durably accept request: {receipt_status}")
        return GatewayReceipt(request_id=receipt_id, status=receipt_status)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
