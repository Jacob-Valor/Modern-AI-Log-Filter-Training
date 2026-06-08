"""Framework-independent API security helpers."""

from __future__ import annotations

import hmac
import ipaddress
import time
from collections import deque
from collections.abc import MutableMapping
from typing import Any, Protocol


class RateLimitBackend(Protocol):
    """Backend interface for rate-limit enforcement."""

    def enforce(
        self,
        client_id: str,
        limit_per_minute: int,
        *,
        now: float | None = None,
    ) -> None:
        """Enforce the rate limit or raise :class:`AccessDenied`."""


class InMemoryRateLimiter:
    """Per-process sliding-window limiter backed by deques."""

    def __init__(self, windows: MutableMapping[str, deque[float]]) -> None:
        self._windows = windows

    def enforce(
        self,
        client_id: str,
        limit_per_minute: int,
        *,
        now: float | None = None,
    ) -> None:
        if limit_per_minute <= 0:
            return

        current_time = time.monotonic() if now is None else now
        window = self._windows.setdefault(client_id, deque())
        cutoff = current_time - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit_per_minute:
            raise AccessDenied(status_code=429, detail="Rate limit exceeded")
        window.append(current_time)


_REDIS_RATE_LIMIT_LUA = """
local key = KEYS[1]
local seq_key = KEYS[2]
local limit = tonumber(ARGV[1])
local window_seconds = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

if not now then
    local redis_time = redis.call('TIME')
    now = tonumber(redis_time[1]) + (tonumber(redis_time[2]) / 1000000)
end

local cutoff = now - window_seconds
redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
local current = redis.call('ZCARD', key)
if current >= limit then
    return {0, current}
end

local member = tostring(now) .. '-' .. tostring(redis.call('INCR', seq_key))
redis.call('ZADD', key, now, member)
local ttl = math.ceil(window_seconds) + 1
redis.call('EXPIRE', key, ttl)
redis.call('EXPIRE', seq_key, ttl)
return {1, current + 1}
"""


class RedisRateLimiter:
    """Distributed sliding-window limiter backed by Redis sorted sets."""

    def __init__(self, client: Any, *, key_prefix: str = "logfilter:rate-limit") -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._script = client.register_script(_REDIS_RATE_LIMIT_LUA)

    def _window_key(self, client_id: str) -> str:
        return f"{self._key_prefix}:{client_id}"

    def enforce(
        self,
        client_id: str,
        limit_per_minute: int,
        *,
        now: float | None = None,
    ) -> None:
        if limit_per_minute <= 0:
            return

        window_key = self._window_key(client_id)
        sequence_key = f"{window_key}:seq"
        result = self._script(
            keys=[window_key, sequence_key],
            args=[limit_per_minute, 60.0, now if now is not None else ""],
        )
        allowed = int(result[0]) if isinstance(result, (list, tuple)) else int(result)
        if allowed == 0:
            raise AccessDenied(status_code=429, detail="Rate limit exceeded")


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
    backend: RateLimitBackend | None = None,
) -> None:
    """Apply a sliding one-minute request limit for a client identifier."""
    limiter = backend or InMemoryRateLimiter(windows)
    limiter.enforce(client_id, limit_per_minute, now=now)


def client_ip_from_request(
    *,
    remote_addr: str,
    forwarded_for: str | None,
    trusted_proxies: list[str] | tuple[str, ...] | str | None,
) -> str:
    trusted_networks = _parse_trusted_proxy_networks(trusted_proxies)
    if not forwarded_for or not _ip_in_networks(remote_addr, trusted_networks):
        return remote_addr

    hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
    if not hops:
        return remote_addr
    for hop in reversed(hops):
        if not _valid_ip(hop):
            return remote_addr
        if not _ip_in_networks(hop, trusted_networks):
            return hop
    return hops[0] if _valid_ip(hops[0]) else remote_addr


def _parse_trusted_proxy_networks(
    trusted_proxies: list[str] | tuple[str, ...] | str | None,
) -> list[ipaddress._BaseNetwork]:
    if trusted_proxies is None:
        return []
    if isinstance(trusted_proxies, str):
        values = [value.strip() for value in trusted_proxies.split(",")]
    else:
        values = [str(value).strip() for value in trusted_proxies]
    networks: list[ipaddress._BaseNetwork] = []
    for value in values:
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            continue
    return networks


def _valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _ip_in_networks(value: str, networks: list[ipaddress._BaseNetwork]) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(address in network for network in networks)
