from __future__ import annotations

import time

from logfilter.security.redaction import (
    MAX_REDACT_CHARS,
    TRUNCATION_MARKER,
    RedactionConfig,
    redact,
)


def test_redact_masks_default_sensitive_values_without_masking_ip_or_host() -> None:
    raw = (
        "user=alice email=alice@example.com src=10.0.0.5 host=edge-router-1 "
        "password=hunter2 token=AKIAABCDEFGHIJKLMNOP card=4111-1111-1111-1111"
    )

    redacted = redact(raw)

    assert "alice@example.com" not in redacted
    assert "hunter2" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "4111-1111-1111-1111" not in redacted
    assert "<EMAIL>" in redacted
    assert "password=<REDACTED>" in redacted
    assert "token=<REDACTED>" in redacted
    assert "<CREDIT_CARD>" in redacted
    assert "10.0.0.5" in redacted
    assert "edge-router-1" in redacted


def test_redact_masks_bearer_slack_and_private_key_material() -> None:
    raw = (
        "Authorization: Bearer abc.def.ghi slack=xoxb-1234567890-secret "
        "-----BEGIN PRIVATE KEY-----\nsecret-key-data\n-----END PRIVATE KEY-----"
    )

    redacted = redact(raw)

    assert "abc.def.ghi" not in redacted
    assert "xoxb-1234567890-secret" not in redacted
    assert "secret-key-data" not in redacted
    assert "Authorization: Bearer <REDACTED>" in redacted
    assert "slack=<REDACTED>" in redacted
    assert "<PRIVATE_KEY>" in redacted


def test_redact_can_mask_ip_addresses_and_hostnames_when_configured() -> None:
    raw = "src=10.0.0.5 dst=2001:db8::1 host=edge-router-1 fqdn=api.example.internal"
    config = RedactionConfig(redact_ip_addresses=True, redact_hostnames=True)

    redacted = redact(raw, config=config)

    assert "10.0.0.5" not in redacted
    assert "2001:db8::1" not in redacted
    assert "edge-router-1" not in redacted
    assert "api.example.internal" not in redacted
    assert "src=<IP>" in redacted
    assert "dst=<IP>" in redacted
    assert "host=<HOST>" in redacted
    assert "fqdn=<HOST>" in redacted


def test_redact_is_idempotent_and_can_be_disabled() -> None:
    raw = "email=bob@example.com password=s3cr3t"

    redacted = redact(raw)

    assert redact(redacted) == redacted
    assert redact(raw, config=RedactionConfig(enabled=False)) == raw
    assert redact(raw, enabled=False) == raw


def test_redact_does_not_mask_invalid_credit_card_like_numbers() -> None:
    raw = "ticket=4111-1111-1111-1112"

    assert redact(raw) == raw


def test_redact_caps_adversarial_input_and_redacts_within_the_cap() -> None:
    """A 64 KiB adversarial payload must not dominate redact cost or leak secrets."""
    adversarial = ("token=AKIAABCDEFGHIJKLMNOP a@b.com 1" * 3000)[: MAX_REDACT_CHARS * 2]

    start = time.perf_counter()
    redacted = redact(adversarial)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert len(redacted) <= MAX_REDACT_CHARS + len(TRUNCATION_MARKER)
    assert TRUNCATION_MARKER in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "a@b.com" not in redacted
