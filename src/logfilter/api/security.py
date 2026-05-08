"""Framework-independent API security helpers."""

from __future__ import annotations

import hmac
import time
from collections import deque


class AccessDenied(Exception):
    """Raised when a request fails an API access-control check."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def require_configured_token(
    provided: str | None,
    expected: str,
    *,
    not_configured_detail: str,
    invalid_detail: str,
) -> None:
    """Validate a configured secret using constant-time comparison."""
    if not expected:
        raise AccessDenied(status_code=403, detail=not_configured_detail)

    provided_bytes = (provided or "").encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if not hmac.compare_digest(provided_bytes, expected_bytes):
        raise AccessDenied(status_code=401, detail=invalid_detail)


def enforce_rate_limit(
    windows: dict[str, deque[float]],
    client_id: str,
    limit_per_minute: int,
    *,
    now: float | None = None,
) -> None:
    """Apply a sliding one-minute request limit for a client identifier."""
    if limit_per_minute <= 0:
        return

    current_time = time.monotonic() if now is None else now
    window = windows.setdefault(client_id, deque())
    cutoff = current_time - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= limit_per_minute:
        raise AccessDenied(status_code=429, detail="Rate limit exceeded")
    window.append(current_time)
