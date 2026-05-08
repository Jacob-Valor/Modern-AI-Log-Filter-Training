"""Network access-control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network
from typing import TypeAlias

IPNetwork: TypeAlias = IPv4Network | IPv6Network


@dataclass(frozen=True)
class CIDRAllowlist:
    """CIDR-based source address allowlist."""

    networks: tuple[IPNetwork, ...]

    @classmethod
    def from_csv(cls, value: str) -> CIDRAllowlist:
        networks = tuple(
            ip_network(item.strip(), strict=False)
            for item in value.split(",")
            if item.strip()
        )
        if not networks:
            raise ValueError("CIDR allowlist must contain at least one network")
        return cls(networks=networks)

    def allows(self, host: str) -> bool:
        address = ip_address(host)
        return any(address in network for network in self.networks)
