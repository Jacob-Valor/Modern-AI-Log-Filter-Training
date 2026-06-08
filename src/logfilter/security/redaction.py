"""Configurable redaction helpers for raw-log archival.

The default policy masks credentials and direct personal identifiers that are
high-risk in long-retention archives while preserving IP addresses and hostnames,
because those fields are core SIEM investigation signals. Operators can enable
IP/hostname masking via configuration for stricter privacy environments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionConfig:
    """Runtime redaction policy for archived raw payloads."""

    enabled: bool = True
    redact_emails: bool = True
    redact_credit_cards: bool = True
    redact_secrets: bool = True
    redact_ip_addresses: bool = False
    redact_hostnames: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> RedactionConfig:
        if not values:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: _as_bool(value) for key, value in values.items() if key in allowed}
        return cls(**kwargs)


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)\b(Authorization\s*:\s*Bearer\s+)([^\s,;]+)")
_KEY_VALUE_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret|slack)\s*[:=]\s*"
    r"(\"[^\"]+\"|'[^']+'|[^\s,;]+)"
)
_AWS_ACCESS_KEY_RE = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+\b")
_CREDIT_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IPV4_RE = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")
_IPV6_RE = re.compile(r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-f:])")
_HOST_VALUE_RE = re.compile(r"(?i)\b(host|hostname|fqdn)\s*=\s*([^\s,;]+)")


def redact(
    text: str,
    *,
    config: RedactionConfig | None = None,
    enabled: bool | None = None,
) -> str:
    """Return ``text`` with sensitive values masked according to ``config``."""

    policy = config or RedactionConfig()
    if enabled is not None:
        policy = RedactionConfig(
            enabled=enabled,
            redact_emails=policy.redact_emails,
            redact_credit_cards=policy.redact_credit_cards,
            redact_secrets=policy.redact_secrets,
            redact_ip_addresses=policy.redact_ip_addresses,
            redact_hostnames=policy.redact_hostnames,
        )
    if not policy.enabled or not text:
        return text

    redacted = text
    if policy.redact_secrets:
        redacted = _PRIVATE_KEY_RE.sub("<PRIVATE_KEY>", redacted)
        redacted = _BEARER_RE.sub(r"\1<REDACTED>", redacted)
        redacted = _KEY_VALUE_SECRET_RE.sub(lambda match: f"{match.group(1)}=<REDACTED>", redacted)
        redacted = _AWS_ACCESS_KEY_RE.sub("<REDACTED>", redacted)
        redacted = _SLACK_TOKEN_RE.sub("<REDACTED>", redacted)

    if policy.redact_emails:
        redacted = _EMAIL_RE.sub("<EMAIL>", redacted)

    if policy.redact_credit_cards:
        redacted = _redact_credit_cards(redacted)

    if policy.redact_ip_addresses:
        redacted = _IPV4_RE.sub("<IP>", redacted)
        redacted = _IPV6_RE.sub("<IP>", redacted)

    if policy.redact_hostnames:
        redacted = _HOST_VALUE_RE.sub(lambda match: f"{match.group(1)}=<HOST>", redacted)

    return redacted


def _redact_credit_cards(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19 and _passes_luhn(digits):
            return "<CREDIT_CARD>"
        return candidate

    return _CREDIT_CARD_CANDIDATE_RE.sub(replace, text)


def _passes_luhn(digits: str) -> bool:
    checksum = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
