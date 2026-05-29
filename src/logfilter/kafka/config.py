from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_VALID_SECURITY_PROTOCOLS = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}
_USERNAME_PASSWORD_MECHANISMS = {"PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"}


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
