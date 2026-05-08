"""Tests for network access-control helpers."""

from __future__ import annotations

import pytest

from logfilter.security.network import CIDRAllowlist


def test_cidr_allowlist_accepts_member_address() -> None:
    allowlist = CIDRAllowlist.from_csv("10.0.0.0/24,192.168.1.10/32")

    assert allowlist.allows("10.0.0.42")
    assert allowlist.allows("192.168.1.10")


def test_cidr_allowlist_rejects_non_member_address() -> None:
    allowlist = CIDRAllowlist.from_csv("10.0.0.0/24")

    assert not allowlist.allows("10.0.1.42")


def test_cidr_allowlist_rejects_empty_configuration() -> None:
    with pytest.raises(ValueError, match="at least one network"):
        CIDRAllowlist.from_csv(" , ")
