"""Unit tests for the LEEF enricher."""

from __future__ import annotations

from logfilter.pipeline.enricher import LEEFEnricher
from logfilter.pipeline.scorer import ScoredEvent


def _make_scored(**kwargs) -> ScoredEvent:
    defaults = dict(
        source_type="syslog",
        timestamp="2026-01-15T11:07:53Z",
        host="prod-server01",
        raw="Jan 15 11:07:53 prod-server01 sshd: Failed password for root from 10.0.0.5",
        normalized_text="Host prod-server01: Failed password from 10.0.0.5",
        fields={"src_ip": "10.0.0.5", "user": "root"},
        ai_threat_score=0.87,
        ai_priority="HIGH",
        ai_mitre_technique="T1110.001",
        ai_entities="10.0.0.5",
        ai_confidence=0.82,
        sigma_matched=False,
        is_duplicate=False,
        dedup_similarity=0.12,
        entities={
            "confidence": 0.91,
            "has_high_value_entities": True,
            "indicators": ["10.0.0.5"],
            "malware": [],
            "vulnerabilities": [],
        },
        cross_encoder_scores=[{"id": "T1110.001", "name": "Password Guessing", "score": 0.82}],
        sigma_rule_ids=[],
        classifier_score=0.76,
        entity_boost=0.20,
        cross_encoder_max=0.82,
        novelty_score=0.5,
        dedup_penalty=0.0,
        scoring_latency_ms=45.3,
        attack_candidates=[],
    )
    defaults.update(kwargs)
    return ScoredEvent(**defaults)


class TestLEEFEnricher:
    def setup_method(self):
        self.enricher = LEEFEnricher(vendor="TestCo", product="TestFilter", version="1.0")

    def test_leef_header_format(self):
        scored = _make_scored()
        leef = self.enricher.enrich(scored)
        assert leef.startswith("LEEF:2.0|TestCo|TestFilter|1.0|")

    def test_ai_threat_score_present(self):
        scored = _make_scored(ai_threat_score=0.92)
        leef = self.enricher.enrich(scored)
        assert "ai_threat_score=0.9200" in leef

    def test_ai_priority_present(self):
        scored = _make_scored(ai_priority="HIGH")
        leef = self.enricher.enrich(scored)
        assert "ai_priority=HIGH" in leef

    def test_mitre_technique_present(self):
        scored = _make_scored(ai_mitre_technique="T1021.002")
        leef = self.enricher.enrich(scored)
        assert "ai_mitre_technique=T1021.002" in leef

    def test_dedup_flag_false(self):
        scored = _make_scored(is_duplicate=False)
        leef = self.enricher.enrich(scored)
        assert "ai_dedup_flag=false" in leef

    def test_dedup_flag_true(self):
        scored = _make_scored(is_duplicate=True)
        leef = self.enricher.enrich(scored)
        assert "ai_dedup_flag=true" in leef

    def test_sigma_match_flag(self):
        scored = _make_scored(sigma_matched=True, sigma_rule_ids=["rule-001"])
        leef = self.enricher.enrich(scored)
        assert "ai_sigma_match=true" in leef
        assert "rule-001" in leef

    def test_raw_log_ref_embedded(self):
        scored = _make_scored()
        leef = self.enricher.enrich(scored, es_doc_id="abc123xyz")
        assert "raw_log_ref=abc123xyz" in leef

    def test_raw_log_b64_fallback_when_no_doc_id(self):
        scored = _make_scored()
        leef = self.enricher.enrich(scored, es_doc_id="")
        assert "raw_log_b64=" in leef

    def test_tab_separates_attributes(self):
        scored = _make_scored()
        leef = self.enricher.enrich(scored)
        # The attribute section (after header) should have tab-separated kv pairs
        header, _, attrs = leef.partition("|^|")
        assert "\t" in attrs

    def test_pipe_in_value_is_sanitised(self):
        # A raw log containing | should not break the LEEF header parsing
        scored = _make_scored(ai_mitre_technique="T1059|evil")
        leef = self.enricher.enrich(scored)
        # The LEEF header section should not have extra pipes after the delimiter
        header_section = leef.split("|^|")[0]
        # 4 pipes in the pre-delimiter section: LEEF:2.0|Vendor|Product|Version|EventID
        # (the ^| separator is split on, so it's excluded from header_section)
        assert header_section.count("|") == 4

    def test_batch_enrichment(self):
        events = [_make_scored(), _make_scored(ai_threat_score=0.3, ai_priority="LOW")]
        leefs = self.enricher.enrich_batch(events)
        assert len(leefs) == 2
        assert "0.8700" in leefs[0]
        assert "0.3000" in leefs[1]

    def test_user_field_mapped(self):
        scored = _make_scored(fields={"user": "admin", "src_ip": "10.0.0.1"})
        leef = self.enricher.enrich(scored)
        assert "usrName=admin" in leef

    def test_src_ip_mapped(self):
        scored = _make_scored(fields={"src_ip": "192.168.1.50"})
        leef = self.enricher.enrich(scored)
        assert "src=192.168.1.50" in leef

    def test_dst_ip_mapped(self):
        scored = _make_scored(fields={"dst_ip": "192.168.1.100"})
        leef = self.enricher.enrich(scored)
        assert "dst=192.168.1.100" in leef

    def test_src_port_mapped(self):
        scored = _make_scored(fields={"src_port": "443"})
        leef = self.enricher.enrich(scored)
        assert "srcPort=443" in leef

    def test_dst_port_mapped(self):
        scored = _make_scored(fields={"dst_port": "80"})
        leef = self.enricher.enrich(scored)
        assert "dstPort=80" in leef

    def test_protocol_mapped(self):
        scored = _make_scored(fields={"protocol": "tcp"})
        leef = self.enricher.enrich(scored)
        assert "proto=tcp" in leef

    def test_no_src_ip_skips_mapping(self):
        scored = _make_scored(fields={})
        leef = self.enricher.enrich(scored)
        assert "src=" not in leef

    def test_host_unknown_skips_timestamp(self):
        scored = _make_scored(host="unknown")
        leef = self.enricher.enrich(scored)
        assert "devTime=" not in leef

    def test_empty_host_skips_timestamp(self):
        scored = _make_scored(host="")
        leef = self.enricher.enrich(scored)
        assert "devTime=" not in leef

    def test_enrich_batch_with_doc_ids(self):
        events = [_make_scored(), _make_scored(ai_threat_score=0.3)]
        leefs = self.enricher.enrich_batch(events, es_doc_ids=["doc1", "doc2"])
        assert len(leefs) == 2
        assert "raw_log_ref=doc1" in leefs[0]
        assert "raw_log_ref=doc2" in leefs[1]

    def test_entities_none_fallback(self):
        scored = _make_scored(entities=None)
        leef = self.enricher.enrich(scored)
        assert "ai_ner_confidence=0.0000" in leef
