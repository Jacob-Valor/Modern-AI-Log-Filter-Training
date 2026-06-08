from __future__ import annotations

from typing import Any

from logfilter.pipeline.events import ScoredEvent


def _event(**overrides: object) -> ScoredEvent:
    defaults: dict[str, Any] = {
        "source_type": "syslog",
        "timestamp": "2026-06-08T01:00:00Z",
        "host": "edge-router-1",
        "raw": "user=alice password=secret from 10.0.0.5",
        "normalized_text": "user login failed",
    }
    defaults.update(overrides)
    return ScoredEvent(**defaults)


def test_scored_event_defaults_are_isolated_between_instances() -> None:
    first = _event()
    second = _event()

    first.sigma_rule_ids.append("rule-1")
    first.fields["user"] = "alice"
    first.attack_candidates.append({"technique_id": "T1110"})
    first.entities["ip"] = ["10.0.0.5"]
    first.cross_encoder_scores.append({"technique_id": "T1110", "score": 0.9})

    assert second.sigma_rule_ids == []
    assert second.fields == {}
    assert second.attack_candidates == []
    assert second.entities == {}
    assert second.cross_encoder_scores == []


def test_to_dict_rounds_scores_and_excludes_raw_payload() -> None:
    scored = _event(
        dedup_similarity=0.123456,
        classifier_score=0.987654,
        tier2_score=0.456789,
        entity_boost=0.111119,
        cross_encoder_max=0.876543,
        novelty_score=0.222229,
        dedup_penalty=0.333339,
        ai_threat_score=0.654321,
        ai_confidence=0.444449,
        scoring_latency_ms=12.3456,
    )

    payload = scored.to_dict()

    assert payload["dedup_similarity"] == 0.1235
    assert payload["classifier_score"] == 0.9877
    assert payload["tier2_score"] == 0.4568
    assert payload["entity_boost"] == 0.1111
    assert payload["cross_encoder_max"] == 0.8765
    assert payload["novelty_score"] == 0.2222
    assert payload["dedup_penalty"] == 0.3333
    assert payload["ai_threat_score"] == 0.6543
    assert payload["ai_confidence"] == 0.4444
    assert payload["scoring_latency_ms"] == 12.35
    assert "raw" not in payload


def test_to_dict_preserves_structured_tier_fields() -> None:
    scored = _event(
        sigma_matched=True,
        sigma_rule_ids=["sigma-1"],
        is_duplicate=True,
        fields={"event_id": "4625"},
        attack_candidates=[{"technique_id": "T1110", "score": 0.7}],
        entities={"USER": ["alice"]},
        cross_encoder_scores=[{"technique_id": "T1110", "score": 0.8}],
        ai_priority="HIGH",
        ai_mitre_technique="T1110",
        ai_entities="USER=alice",
    )

    payload = scored.to_dict()

    assert payload["source_type"] == "syslog"
    assert payload["host"] == "edge-router-1"
    assert payload["fields"] == {"event_id": "4625"}
    assert payload["sigma_matched"] is True
    assert payload["sigma_rule_ids"] == ["sigma-1"]
    assert payload["is_duplicate"] is True
    assert payload["attack_candidates"] == [{"technique_id": "T1110", "score": 0.7}]
    assert payload["entities"] == {"USER": ["alice"]}
    assert payload["cross_encoder_scores"] == [{"technique_id": "T1110", "score": 0.8}]
    assert payload["ai_priority"] == "HIGH"
    assert payload["ai_mitre_technique"] == "T1110"
    assert payload["ai_entities"] == "USER=alice"
