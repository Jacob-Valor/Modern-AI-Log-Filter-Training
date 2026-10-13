"""Tests for the automated response prototype (src/logfilter/response)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from logfilter.pipeline.events import ScoredEvent
from logfilter.response.auto_response import ActionType, AutoResponseEngine


def _scored(
    priority: str, score: float, src_ip: str | None = "10.0.0.5", host: str = "web01"
) -> ScoredEvent:
    return ScoredEvent(
        source_type="endpoint",
        timestamp="2026-01-01T00:00:00Z",
        host=host,
        raw="raw log line",
        normalized_text="normalized",
        fields={"src_ip": src_ip} if src_ip else {},
        ai_threat_score=score,
        ai_priority=priority,
        ai_mitre_technique="T1110",
    )


class RecordingFirewall:
    def __init__(self) -> None:
        self.blocked_ips: list[str] = []
        self.quarantined_hosts: list[str] = []

    def block_ip(self, ip: str, reason: str, action_id: str) -> None:
        self.blocked_ips.append(ip)

    def quarantine_host(self, host: str, reason: str, action_id: str) -> None:
        self.quarantined_hosts.append(host)


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, message: str, priority: str, action_id: str) -> None:
        self.messages.append(message)


def test_low_priority_triggers_no_actions():
    engine = AutoResponseEngine()
    actions = engine.decide(_scored("LOW", 0.2))
    assert actions == []


def test_high_priority_high_score_triggers_block_quarantine_notify():
    engine = AutoResponseEngine(block_threshold=0.85)
    actions = engine.decide(_scored("HIGH", 0.9))
    kinds = {a.action for a in actions}
    assert kinds == {ActionType.BLOCK_IP, ActionType.QUARANTINE_HOST, ActionType.NOTIFY_SOC}


def test_medium_priority_below_block_threshold_only_notifies():
    engine = AutoResponseEngine(block_threshold=0.85)
    actions = engine.decide(_scored("MEDIUM", 0.6))
    kinds = {a.action for a in actions}
    assert kinds == {ActionType.NOTIFY_SOC}


def test_dry_run_default_does_not_call_firewall():
    firewall = RecordingFirewall()
    notifier = RecordingNotifier()
    engine = AutoResponseEngine(firewall=firewall, notifier=notifier)  # dry_run=True default

    engine.handle(_scored("HIGH", 0.95))

    assert firewall.blocked_ips == []
    assert firewall.quarantined_hosts == []
    assert notifier.messages == []

    with pytest.raises(ValueError, match="firewall and SOC"):
        AutoResponseEngine(firewall=firewall, dry_run=False)
    with pytest.raises(ValueError, match="firewall and SOC"):
        AutoResponseEngine(notifier=notifier, dry_run=False)


def test_armed_mode_calls_firewall_and_notifier():
    firewall = RecordingFirewall()
    notifier = RecordingNotifier()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=notifier,
        dry_run=False,
        block_threshold=0.85,
        allowed_networks=("203.0.113.0/24",),
        allowed_hosts=frozenset({"db01"}),
    )

    engine.handle(_scored("HIGH", 0.95, src_ip="203.0.113.7", host="db01"))

    assert firewall.blocked_ips == ["203.0.113.7"]
    assert firewall.quarantined_hosts == ["db01"]
    assert len(notifier.messages) == 1


def test_no_src_ip_skips_block_but_still_quarantines_and_notifies():
    firewall = RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=RecordingNotifier(),
        dry_run=False,
        block_threshold=0.85,
        allowed_hosts=frozenset({"web01"}),
    )

    engine.handle(_scored("HIGH", 0.95, src_ip=None))

    assert firewall.blocked_ips == []
    assert firewall.quarantined_hosts == ["web01"]


def test_action_flag_cannot_bypass_engine_dry_run():
    firewall = RecordingFirewall()
    engine = AutoResponseEngine(firewall=firewall)
    action = engine.decide(_scored("HIGH", 0.95))[0]
    forged = type(action)(
        action=action.action,
        target=action.target,
        reason=action.reason,
        priority=action.priority,
        dry_run=False,
    )
    engine.execute([forged])
    assert firewall.blocked_ips == []


def test_armed_mode_rejects_target_outside_allowlist():
    firewall = RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=RecordingNotifier(),
        dry_run=False,
        allowed_networks=("10.0.0.0/24",),
        allowed_hosts=frozenset({"web01"}),
    )
    engine.handle(_scored("HIGH", 0.95, src_ip="203.0.113.9"))
    assert firewall.blocked_ips == []


def test_execute_revalidates_forged_targets_against_allowlists():
    firewall = RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=RecordingNotifier(),
        dry_run=False,
        allowed_networks=("10.0.0.0/24",),
        allowed_hosts=frozenset({"safe-host"}),
    )
    actions = engine.decide(_scored("HIGH", 0.95, src_ip="10.0.0.5", host="safe-host"))
    block = next(action for action in actions if action.action == ActionType.BLOCK_IP)
    quarantine = next(action for action in actions if action.action == ActionType.QUARANTINE_HOST)
    forged_block = type(block)(
        action=block.action,
        target="203.0.113.99",
        reason=block.reason,
        priority=block.priority,
        dry_run=False,
    )
    forged_quarantine = type(quarantine)(
        action=quarantine.action,
        target="victim-host",
        reason=quarantine.reason,
        priority=quarantine.priority,
        dry_run=False,
    )
    engine.execute([forged_block, forged_quarantine])
    assert firewall.blocked_ips == []
    assert firewall.quarantined_hosts == []


def test_armed_mode_suppresses_duplicate_actions():
    firewall = RecordingFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=RecordingNotifier(),
        dry_run=False,
        allowed_networks=("203.0.113.0/24",),
        allowed_hosts=frozenset({"db01"}),
    )
    scored = _scored("HIGH", 0.95, src_ip="203.0.113.7", host="db01")
    first = engine.handle(scored, "raw-ref")
    second = engine.handle(scored, "raw-ref")
    assert firewall.blocked_ips == ["203.0.113.7"]
    assert firewall.quarantined_hosts == ["db01"]
    assert {action.status for action in first} == {"acknowledged"}
    assert {action.status for action in second} == {"duplicate"}


def test_executed_id_cache_stays_bounded_and_keeps_recent_idempotency(monkeypatch):
    """Acknowledged ids must be evicted oldest-first once past the documented bound."""
    firewall = RecordingFirewall()
    notifier = RecordingNotifier()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=notifier,
        dry_run=False,
        allowed_networks=("203.0.113.0/24",),
        allowed_hosts=frozenset({"db01"}),
        max_actions_per_minute=100_000,
    )
    monkeypatch.setattr(engine, "MAX_EXECUTED_IDS", 5)

    scored = _scored("HIGH", 0.95, src_ip="203.0.113.7", host="db01")
    for i in range(5):
        engine.handle(scored, f"raw-ref-{i}")

    assert len(engine._executed_ids) == 5
    assert len(engine._executed_order) == 5

    # A sixth distinct event pushes the map one entry past the bound.
    engine.handle(scored, "raw-ref-5")

    assert len(engine._executed_ids) == 5
    assert len(engine._executed_order) == 5

    # The most recent acknowledgement is still deduplicated.
    again = engine.handle(scored, "raw-ref-5")

    assert {action.status for action in again} == {"duplicate"}
    assert firewall.blocked_ips == ["203.0.113.7"] * 6


def test_failed_gateway_action_is_retryable():
    class FailOnceFirewall(RecordingFirewall):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def block_ip(self, ip: str, reason: str, action_id: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("gateway unavailable")
            super().block_ip(ip, reason, action_id)

    firewall = FailOnceFirewall()
    engine = AutoResponseEngine(
        firewall=firewall,
        notifier=RecordingNotifier(),
        dry_run=False,
        allowed_networks=("203.0.113.0/24",),
    )
    block = next(
        action
        for action in engine.decide(_scored("HIGH", 0.95, src_ip="203.0.113.7"))
        if action.action == ActionType.BLOCK_IP
    )
    with pytest.raises(RuntimeError, match="gateway unavailable"):
        engine.execute([block])
    engine.execute([block])

    assert firewall.attempts == 2
    assert firewall.blocked_ips == ["203.0.113.7"]


def test_event_reference_stabilizes_action_ids_across_scoring_retries():
    first = _scored("HIGH", 0.95, src_ip="203.0.113.7")
    second = replace(first, timestamp="2026-01-01T00:01:00Z", ai_threat_score=0.96)
    engine = AutoResponseEngine()

    first_ids = [action.action_id for action in engine.decide(first, "raw-ref-1")]
    second_ids = [action.action_id for action in engine.decide(second, "raw-ref-1")]
    different_ref_ids = [action.action_id for action in engine.decide(first, "raw-ref-2")]

    assert first_ids == second_ids
    assert first_ids != different_ref_ids
