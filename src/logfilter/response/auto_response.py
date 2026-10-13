"""Policy-controlled automated response through an acknowledged HTTPS gateway.

The engine remains safe by default: dry-run mode only recommends actions. Armed
mode requires explicit allowlists, a kill switch setting, and a real adapter.
Every gateway request carries a deterministic idempotency key so scoring retries
cannot repeat an already accepted firewall, EDR, or SOC action.
"""

from __future__ import annotations

import hashlib
import ipaddress
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

import structlog

from logfilter.http_gateway import SignedHttpGatewayClient
from logfilter.pipeline.events import ScoredEvent

logger = structlog.get_logger(__name__)


class ActionType(str, Enum):
    BLOCK_IP = "block_ip"
    QUARANTINE_HOST = "quarantine_host"
    NOTIFY_SOC = "notify_soc"


@dataclass(frozen=True)
class ResponseAction:
    action: ActionType
    target: str
    reason: str
    priority: str
    dry_run: bool
    executed_at: float = field(default_factory=time.time)
    action_id: str = ""
    event_ref: str = ""
    status: str = "recommended"


class FirewallClient(Protocol):
    """Production firewall/EDR enforcement contract."""

    def block_ip(self, ip: str, reason: str, action_id: str) -> None: ...
    def quarantine_host(self, host: str, reason: str, action_id: str) -> None: ...


class SocNotifier(Protocol):
    """Production SOC notification contract."""

    def notify(self, message: str, priority: str, action_id: str) -> None: ...


class NullFirewallClient:
    """No-op client used only in dry-run mode."""

    def block_ip(self, ip: str, reason: str, action_id: str) -> None:
        logger.info(
            "NullFirewallClient.block_ip (no-op)", ip=ip, reason=reason, action_id=action_id
        )

    def quarantine_host(self, host: str, reason: str, action_id: str) -> None:
        logger.info(
            "NullFirewallClient.quarantine_host (no-op)",
            host=host,
            reason=reason,
            action_id=action_id,
        )


class NullSocNotifier:
    """No-op notifier used only in dry-run mode."""

    def notify(self, message: str, priority: str, action_id: str) -> None:
        logger.info(
            "NullSocNotifier.notify (no-op)",
            message=message,
            priority=priority,
            action_id=action_id,
        )


class SignedResponseGateway:
    """Firewall, EDR, and SOC adapter backed by one signed HTTPS gateway."""

    def __init__(self, client: SignedHttpGatewayClient) -> None:
        self.client = client

    def block_ip(self, ip: str, reason: str, action_id: str) -> None:
        self.client.send(
            request_id=action_id,
            payload={"action": ActionType.BLOCK_IP.value, "target": ip, "reason": reason},
        )

    def quarantine_host(self, host: str, reason: str, action_id: str) -> None:
        self.client.send(
            request_id=action_id,
            payload={
                "action": ActionType.QUARANTINE_HOST.value,
                "target": host,
                "reason": reason,
            },
        )

    def notify(self, message: str, priority: str, action_id: str) -> None:
        self.client.send(
            request_id=action_id,
            payload={
                "action": ActionType.NOTIFY_SOC.value,
                "message": message,
                "priority": priority,
            },
        )

    def close(self) -> None:
        self.client.close()


class AutoResponseEngine:
    """
    Decides and (optionally) executes automated response actions for
    HIGH/MEDIUM priority scored events.

    Parameters
    ----------
    firewall : FirewallClient
        Client used to block IPs / quarantine hosts. Defaults to a no-op
        client — pass a real implementation to actually enforce actions.
    notifier : SocNotifier
        Client used to notify the SOC. Defaults to a no-op notifier.
    dry_run : bool
        When True (default), actions are logged and returned but NOT
        executed against firewall/notifier clients. Must be explicitly
        set False to arm real enforcement.
    block_threshold : float
        Minimum ai_threat_score required to trigger BLOCK_IP.
    quarantine_priorities : set[str]
        ai_priority values that trigger host quarantine.
    """

    #: Maximum number of acknowledged action ids retained for deduplication.
    #: Bounded so a long-lived process cannot accumulate one 64-char digest per
    #: executed action (~31M/year at 60/min). Idempotency is guaranteed for the
    #: most recent ``MAX_EXECUTED_IDS`` acknowledgements; older ids are evicted
    #: oldest-first and may be executed again.
    MAX_EXECUTED_IDS = 100_000

    def __init__(
        self,
        firewall: FirewallClient | None = None,
        notifier: SocNotifier | None = None,
        dry_run: bool = True,
        block_threshold: float = 0.85,
        quarantine_priorities: frozenset[str] = frozenset({"HIGH"}),
        allowed_networks: tuple[str, ...] = (),
        allowed_hosts: frozenset[str] = frozenset(),
        kill_switch: bool = False,
        max_actions_per_minute: int = 60,
    ) -> None:
        self.firewall = firewall or NullFirewallClient()
        self.notifier = notifier or NullSocNotifier()
        if not dry_run and (
            isinstance(self.firewall, NullFirewallClient)
            or isinstance(self.notifier, NullSocNotifier)
        ):
            raise ValueError("Armed response mode requires non-null firewall and SOC adapters")
        self.dry_run = dry_run
        self.block_threshold = block_threshold
        self.quarantine_priorities = quarantine_priorities
        try:
            self.allowed_networks = tuple(ipaddress.ip_network(value) for value in allowed_networks)
        except ValueError as exc:
            raise ValueError("Invalid response allowlist network") from exc
        self.allowed_hosts = allowed_hosts
        self.kill_switch = kill_switch
        if max_actions_per_minute <= 0:
            raise ValueError("max_actions_per_minute must be positive")
        self.max_actions_per_minute = max_actions_per_minute
        self._executed_ids: set[str] = set()
        self._executed_order: deque[str] = deque()
        self._execution_times: list[float] = []
        self._inflight_ids: set[str] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _action_id(action: ActionType, target: str, reason: str, event_ref: str = "") -> str:
        event_identity = event_ref or reason
        payload = f"{event_identity}|{action.value}|{target}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _remember_executed(self, action_id: str) -> None:
        """Record an acknowledged id, evicting oldest entries past the bound."""
        if action_id in self._executed_ids:
            return
        self._executed_ids.add(action_id)
        self._executed_order.append(action_id)
        while len(self._executed_order) > self.MAX_EXECUTED_IDS:
            self._executed_ids.discard(self._executed_order.popleft())

    def _ip_allowed(self, value: str) -> str | None:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return None
        protected = (
            address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
        )
        if protected:
            return None
        if not self.dry_run and not any(address in network for network in self.allowed_networks):
            return None
        return str(address)

    def _host_allowed(self, value: str) -> bool:
        return bool(value) and (self.dry_run or value in self.allowed_hosts)

    def _action(
        self,
        action: ActionType,
        target: str,
        reason: str,
        priority: str,
        event_ref: str,
    ) -> ResponseAction:
        return ResponseAction(
            action=action,
            target=target,
            reason=reason,
            priority=priority,
            dry_run=self.dry_run,
            action_id=self._action_id(action, target, reason, event_ref),
            event_ref=event_ref,
        )

    def decide(self, scored: ScoredEvent, event_ref: str = "") -> list[ResponseAction]:
        """Determine which response actions apply, without executing them."""
        actions: list[ResponseAction] = []

        if scored.ai_priority not in ("HIGH", "MEDIUM"):
            return actions

        reason = (
            f"priority={scored.ai_priority} score={scored.ai_threat_score:.3f} "
            f"mitre={scored.ai_mitre_technique or 'n/a'}"
        )

        source_value = (
            scored.fields.get("src_ip")
            or scored.fields.get("src")
            or scored.fields.get("sourceAddress")
        )
        src_ip = self._ip_allowed(str(source_value)) if source_value else None
        destructive_allowed = not scored.score_degraded and not scored.is_duplicate
        if src_ip and destructive_allowed and scored.ai_threat_score >= self.block_threshold:
            actions.append(
                self._action(
                    ActionType.BLOCK_IP,
                    src_ip,
                    reason,
                    scored.ai_priority,
                    event_ref,
                )
            )

        endpoint_source = scored.source_type == "endpoint"
        if (
            destructive_allowed
            and endpoint_source
            and scored.ai_priority in self.quarantine_priorities
            and self._host_allowed(scored.host)
        ):
            actions.append(
                self._action(
                    ActionType.QUARANTINE_HOST,
                    scored.host,
                    reason,
                    scored.ai_priority,
                    event_ref,
                )
            )

        actions.append(
            self._action(
                ActionType.NOTIFY_SOC,
                scored.host or src_ip or "unknown",
                reason,
                scored.ai_priority,
                event_ref,
            )
        )
        return actions

    def execute(self, actions: list[ResponseAction]) -> list[ResponseAction]:
        """Execute actions under authoritative engine policy, never caller flags."""
        results: list[ResponseAction] = []
        for action in actions:
            authoritative_id = self._action_id(
                action.action, action.target, action.reason, action.event_ref
            )
            if self.dry_run or action.dry_run:
                logger.info(
                    "Auto-response suppressed",
                    mode="DRY_RUN",
                    action=action.action.value,
                    target=action.target,
                    action_id=authoritative_id,
                )
                results.append(replace(action, status="recommended"))
                continue
            if self.kill_switch:
                logger.info(
                    "Auto-response suppressed",
                    mode="KILL_SWITCH",
                    action=action.action.value,
                    target=action.target,
                    action_id=authoritative_id,
                )
                results.append(replace(action, status="suppressed"))
                continue

            if action.action == ActionType.BLOCK_IP:
                canonical_ip = self._ip_allowed(action.target)
                if canonical_ip is None or canonical_ip != action.target:
                    logger.warning(
                        "Auto-response target rejected",
                        action=action.action.value,
                        target=action.target,
                    )
                    results.append(replace(action, status="rejected"))
                    continue
            elif action.action == ActionType.QUARANTINE_HOST and not self._host_allowed(
                action.target
            ):
                logger.warning(
                    "Auto-response target rejected",
                    action=action.action.value,
                    target=action.target,
                )
                results.append(replace(action, status="rejected"))
                continue

            now = time.time()
            with self._lock:
                cutoff = now - 60.0
                self._execution_times = [
                    value for value in self._execution_times if value >= cutoff
                ]
                if authoritative_id in self._executed_ids:
                    logger.info("Auto-response duplicate suppressed", action_id=authoritative_id)
                    results.append(replace(action, status="duplicate"))
                    continue
                if authoritative_id in self._inflight_ids:
                    logger.info("Auto-response already in flight", action_id=authoritative_id)
                    results.append(replace(action, status="in_flight"))
                    continue
                if len(self._execution_times) >= self.max_actions_per_minute:
                    logger.warning("Auto-response rate limit reached", action_id=authoritative_id)
                    results.append(replace(action, status="rate_limited"))
                    continue
                self._inflight_ids.add(authoritative_id)
                self._execution_times.append(now)

            try:
                if action.action == ActionType.BLOCK_IP:
                    self.firewall.block_ip(action.target, action.reason, authoritative_id)
                elif action.action == ActionType.QUARANTINE_HOST:
                    self.firewall.quarantine_host(action.target, action.reason, authoritative_id)
                elif action.action == ActionType.NOTIFY_SOC:
                    self.notifier.notify(
                        f"[{action.priority}] {action.reason} (target={action.target})",
                        action.priority,
                        authoritative_id,
                    )
            except Exception:
                with self._lock:
                    self._inflight_ids.discard(authoritative_id)
                raise
            with self._lock:
                self._inflight_ids.discard(authoritative_id)
                self._remember_executed(authoritative_id)
            logger.info(
                "Auto-response executed",
                action=action.action.value,
                target=action.target,
                action_id=authoritative_id,
            )
            results.append(replace(action, status="acknowledged"))
        return results

    def handle(self, scored: ScoredEvent, event_ref: str = "") -> list[ResponseAction]:
        """Convenience: decide + execute in one call."""
        return self.execute(self.decide(scored, event_ref))
