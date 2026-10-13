from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
import structlog
from elasticsearch import exceptions as es_exceptions
from kafka.errors import KafkaError

from logfilter.utils.circuit_breaker import CircuitBreakerOpen

logger = structlog.get_logger(__name__)

_VALID_SECURITY_PROTOCOLS = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}
_USERNAME_PASSWORD_MECHANISMS = {"PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"}

DEFAULT_MAX_POLL_INTERVAL_MS = 900_000
DEFAULT_SESSION_TIMEOUT_MS = 45_000

_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429})


def _int_config(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def kafka_consumer_health_kwargs(kafka_config: Mapping[str, Any] | None) -> dict[str, int]:
    """Group-membership timings that keep slow batches from triggering rebalances."""
    kafka_config = kafka_config or {}
    return {
        "max_poll_interval_ms": _int_config(
            kafka_config.get("max_poll_interval_ms"), DEFAULT_MAX_POLL_INTERVAL_MS
        ),
        "session_timeout_ms": _int_config(
            kafka_config.get("session_timeout_ms"), DEFAULT_SESSION_TIMEOUT_MS
        ),
    }


def exhausted_dlq_topic(dlq_topic: str | None) -> str | None:
    """Terminal topic for DLQ records that used up their replay budget."""
    if not dlq_topic:
        return None
    return f"{dlq_topic}-exhausted"


def decode_kafka_value(value: bytes | None) -> Any:
    """Tolerant value deserializer — undecodable payloads become ``None``.

    Deserialisation happens inside ``poll()``; letting a JSON error escape would
    crash the consumer on a single poison message and re-poll it forever.
    """
    if value is None:
        return None
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Discarding undecodable Kafka message value", error=str(exc))
        return None


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status carried by an exception, from any of its usual homes."""
    candidates = (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "meta", None), "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
    return None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, CircuitBreakerOpen):
        return True
    status = _status_code(exc)
    if isinstance(status, int):
        # An explicit status is authoritative: httpx raises HTTPStatusError for
        # every non-2xx, so the transport fallback below would call a 400
        # "retryable" and keep a permanently bad record alive forever.
        return status in _TRANSIENT_STATUS_CODES or status >= 500
    return isinstance(
        exc,
        (
            httpx.HTTPError,
            es_exceptions.TransportError,
            KafkaError,
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )


def is_transient_failure(exc: BaseException) -> bool:
    """True when a failure is environmental and the records must be retried as-is.

    Retrying means: no DLQ publish, no offset commit, and a short pause. Only
    permanent, record-attributable failures may discard a record.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_transient(current):
            return True
        current = current.__cause__
    return False


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _bool_config(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean config value: {value!r}")


def kafka_security_kwargs(kafka_config: Mapping[str, Any] | None) -> dict[str, Any]:
    kafka_config = kafka_config or {}
    security = kafka_config.get("security", {}) or {}
    if not isinstance(security, Mapping):
        raise ValueError("kafka.security must be a mapping")

    protocol = _clean(security.get("protocol") or "PLAINTEXT").upper()
    if protocol not in _VALID_SECURITY_PROTOCOLS:
        raise ValueError(f"Unsupported Kafka security protocol: {protocol}")

    kwargs: dict[str, Any] = {"security_protocol": protocol}

    sasl = security.get("sasl", {}) or {}
    ssl = security.get("ssl", {}) or {}
    if not isinstance(sasl, Mapping):
        raise ValueError("kafka.security.sasl must be a mapping")
    if not isinstance(ssl, Mapping):
        raise ValueError("kafka.security.ssl must be a mapping")

    if "SASL" in protocol:
        mechanism = _clean(sasl.get("mechanism")).upper()
        if not mechanism:
            raise ValueError("Kafka SASL protocol requires kafka.security.sasl.mechanism")
        kwargs["sasl_mechanism"] = mechanism
        if mechanism in _USERNAME_PASSWORD_MECHANISMS:
            username = _clean(sasl.get("username"))
            password = _clean(sasl.get("password"))
            if not username or not password:
                raise ValueError("Kafka SASL username/password are required for PLAIN/SCRAM")
            kwargs["sasl_plain_username"] = username
            kwargs["sasl_plain_password"] = password

    if "SSL" in protocol:
        kwargs["ssl_check_hostname"] = _bool_config(ssl.get("check_hostname"), True)
        for config_key, kwarg_key in (
            ("cafile", "ssl_cafile"),
            ("certfile", "ssl_certfile"),
            ("keyfile", "ssl_keyfile"),
            ("password", "ssl_password"),
        ):
            value = _clean(ssl.get(config_key))
            if value:
                kwargs[kwarg_key] = value

    return kwargs
