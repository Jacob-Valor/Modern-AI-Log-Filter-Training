"""
End-to-end pipeline smoke test.

Exercises the full stack:
  LogNormalizer → (mocked scorer) → LEEFEnricher → RoutingDecision

The SecureBERT2.0 models (NER, BiEncoder, CrossEncoder) are not downloaded
in the test environment, so the scorer is run in a stub mode that bypasses
the heavy ML models but exercises all the data structures, LEEF formatting,
and routing logic.

Also validates the ONNX classifier directly on a zero-vector input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.pipeline.enricher import LEEFEnricher  # noqa: E402
from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType  # noqa: E402
from logfilter.pipeline.router import LogRouter  # noqa: E402
from logfilter.pipeline.scorer import LogScorer, ScoredEvent  # noqa: E402

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"


def banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Normalizer smoke test ───────────────────────────────────────────────────


def test_normalizer() -> bool:
    banner("1. LogNormalizer — multi-format parsing")
    normalizer = LogNormalizer()
    ok = True

    cases = [
        (
            "syslog",
            "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for "
            "root from 10.0.0.5 port 44382 ssh2",
            LogSourceType.SYSLOG,
            "prod-srv01",
        ),
        (
            "winevent",
            json.dumps(
                {
                    "EventID": "4625",
                    "Computer": "WIN01",
                    "Message": "Account failed logon",
                    "TimeCreated": "2026-01-15",
                }
            ),
            LogSourceType.WINEVENT,
            "WIN01",
        ),
        (
            "CEF firewall",
            "CEF:0|Palo Alto|NGFW|10.0|threat|SSH brute force|8|"
            "src=10.0.0.5 dst=192.168.1.1 spt=55000 dpt=22",
            LogSourceType.FIREWALL,
            None,
        ),
        (
            "CloudTrail",
            json.dumps(
                {
                    "eventSource": "iam.amazonaws.com",
                    "eventName": "CreateUser",
                    "userIdentity": {"userName": "attacker"},
                    "sourceIPAddress": "1.2.3.4",
                    "awsRegion": "us-east-1",
                    "eventTime": "2026-01-15T11:00:00Z",
                }
            ),
            LogSourceType.CLOUDTRAIL,
            "1.2.3.4",
        ),
        (
            "Apache web log",
            '10.0.0.5 - - [15/Jan/2026:11:07:53 +0000] "GET /wp-admin HTTP/1.1" 403 512',
            LogSourceType.WEB,
            "10.0.0.5",
        ),
    ]

    for name, raw, expected_type, expected_host in cases:
        ev = normalizer.normalize(raw)
        type_ok = ev.source_type == expected_type
        host_ok = (expected_host is None) or (ev.host == expected_host)
        status = PASS if (type_ok and host_ok) else FAIL
        if not (type_ok and host_ok):
            ok = False
        print(f"  {status} {name}: type={ev.source_type.value}, host={ev.host!r}")
        if not type_ok:
            print(f"      Expected type {expected_type.value}, got {ev.source_type.value}")
        if not host_ok:
            print(f"      Expected host {expected_host!r}, got {ev.host!r}")

    return ok


# ── 2. ONNX classifier smoke test ──────────────────────────────────────────────


def test_onnx_classifier() -> bool:
    banner("2. ONNX Classifier — inference on zero vector")
    import numpy as np

    from logfilter.models.classifier import LogClassifier

    classifier = LogClassifier(
        model_path=ROOT / "models" / "log_classifier.onnx",
        scaler_path=ROOT / "models" / "scaler.json",
        feature_names_path=ROOT / "models" / "feature_names.json",
    )

    if not (ROOT / "models" / "log_classifier.onnx").exists():
        print(f"  {INFO} ONNX model not found — skipping (run 'make train' first)")
        return True

    # Zero vector — should classify as normal (no anomalous events)
    n_features = len(classifier.feature_names)
    zero_vec = np.zeros((1, n_features), dtype=np.float32)
    prob = classifier.predict_proba(zero_vec)
    is_normal = float(prob[0]) < 0.5
    status = PASS if is_normal else FAIL
    print(
        f"  {status} Zero vector → failure_prob={float(prob[0]):.4f} (expected < 0.5, i.e. normal)"
    )

    # Feature names loaded
    feat_ok = n_features > 0
    status2 = PASS if feat_ok else FAIL
    print(f"  {status2} Feature names loaded: {n_features} features")

    return is_normal and feat_ok


# ── 3. LEEF enricher smoke test ───────────────────────────────────────────────


def test_leef_enricher() -> bool:
    banner("3. LEEFEnricher — LEEF 2.0 payload construction")
    enricher = LEEFEnricher(vendor="TestCo", product="AIPreprocessor", version="1.0")
    ok = True

    scored = ScoredEvent(
        source_type="syslog",
        timestamp="2026-01-15T11:07:53Z",
        host="prod-srv01",
        raw="Jan 15 11:07:53 prod-srv01 sshd: Failed password for root from 10.0.0.5",
        normalized_text="Host prod-srv01: Failed password from 10.0.0.5",
        fields={"src_ip": "10.0.0.5", "user": "root"},
        ai_threat_score=0.87,
        ai_priority="HIGH",
        ai_mitre_technique="T1110.001",
        ai_entities="10.0.0.5",
        ai_confidence=0.83,
        sigma_matched=False,
        is_duplicate=False,
        dedup_similarity=0.1,
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

    leef = enricher.enrich(scored, es_doc_id="es-doc-abc123")
    checks = [
        ("LEEF:2.0 header", leef.startswith("LEEF:2.0|TestCo|AIPreprocessor|1.0|")),
        ("ai_threat_score present", "ai_threat_score=0.8700" in leef),
        ("ai_priority=HIGH", "ai_priority=HIGH" in leef),
        ("ai_mitre_technique", "ai_mitre_technique=T1110.001" in leef),
        ("raw_log_ref", "raw_log_ref=es-doc-abc123" in leef),
        ("src field mapped", "src=10.0.0.5" in leef),
        ("usrName mapped", "usrName=root" in leef),
        (
            "tab-delimited attrs",
            # B19: LEEF header declares \t as the delimiter (|\t|) and the
            # attribute section is joined with \t. A correctly formatted LEEF
            # with N attrs has N+1 tab characters (1 in the header, 1 per attr
            # boundary). The previous check used "|^|" which is no longer in
            # the format — LEEF now uses \t for both header delimiter and
            # attribute separator to avoid parser ambiguity.
            leef.count("\t") > 1,
        ),
    ]

    for name, result in checks:
        status = PASS if result else FAIL
        if not result:
            ok = False
        print(f"  {status} {name}")

    print("\n  LEEF preview (first 200 chars):")
    print(f"  {leef[:200]}…")

    return ok


# ── 4. Router logic smoke test ────────────────────────────────────────────────


def test_router_logic() -> bool:
    banner("4. LogRouter — routing decision logic (no network)")
    ok = True

    # Mock the SyslogSender to avoid real network connections
    mock_sender = MagicMock()
    mock_sender.send.return_value = None
    mock_sender.send_batch.return_value = 3

    router_enrich = LogRouter(
        config={
            "qradar": {
                "mode": "enrich_only",
                "syslog_host": "localhost",
                "syslog_port": 514,
                "syslog_protocol": "tcp",
            }
        },
        sender=mock_sender,
    )
    router_suppress = LogRouter(
        config={
            "qradar": {
                "mode": "suppress_low",
                "syslog_host": "localhost",
                "syslog_port": 514,
                "syslog_protocol": "tcp",
            }
        },
        sender=mock_sender,
    )

    def _scored(priority: str, score: float) -> ScoredEvent:
        return ScoredEvent(
            source_type="syslog",
            timestamp="",
            host="h",
            raw="r",
            normalized_text="t",
            ai_priority=priority,
            ai_threat_score=score,
        )

    cases = [
        ("enrich_only + INFO event → forward", router_enrich, _scored("INFO", 0.05), True),
        ("enrich_only + HIGH event → forward", router_enrich, _scored("HIGH", 0.92), True),
        ("suppress_low + INFO event → suppress", router_suppress, _scored("INFO", 0.05), False),
        ("suppress_low + HIGH event → forward", router_suppress, _scored("HIGH", 0.92), True),
        ("suppress_low + MEDIUM event → forward", router_suppress, _scored("MEDIUM", 0.65), True),
    ]

    for name, router, scored, expected_fwd in cases:
        decision = router.decide(scored)
        result = decision.forward_to_qradar == expected_fwd
        status = PASS if result else FAIL
        if not result:
            ok = False
        print(f"  {status} {name}: forward={decision.forward_to_qradar}")

    return ok


# ── 5. Scorer with real models ──────────────────────────────────────────────


def test_scorer_real() -> bool:
    banner("5. LogScorer — full tiered pipeline (real models)")
    ok = True

    normalizer = LogNormalizer()
    enricher = LEEFEnricher()

    from logfilter.config import load_config

    full_config = load_config(ROOT / "config" / "config.yaml")

    config = {
        "scoring": full_config["scoring"],
        "models": full_config["models"],
    }

    scorer = LogScorer(config=config)

    raw = (
        "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for "
        "root from 10.0.0.5 port 44382 ssh2"
    )
    normalized = normalizer.normalize(raw)
    scored = scorer.score(normalized)
    leef = enricher.enrich(scored, es_doc_id="es-doc-smoke-test")

    checks = [
        ("score > 0", scored.ai_threat_score > 0),
        ("score <= 1", scored.ai_threat_score <= 1.0),
        (
            "priority is HIGH/MEDIUM/LOW/INFO",
            scored.ai_priority in {"HIGH", "MEDIUM", "LOW", "INFO"},
        ),
        ("mitre technique set", bool(scored.ai_mitre_technique)),
        ("entity extracted", bool(scored.ai_entities)),
        ("LEEF produced", leef.startswith("LEEF:2.0|")),
        ("source_type preserved", scored.source_type == "syslog"),
        ("host preserved", scored.host == "prod-srv01"),
    ]

    for name, result in checks:
        status = PASS if result else FAIL
        if not result:
            ok = False
        print(f"  {status} {name}")

    print(f"\n  ai_threat_score={scored.ai_threat_score:.4f}  priority={scored.ai_priority}")
    print(f"  ai_mitre_technique={scored.ai_mitre_technique}  ai_entities={scored.ai_entities}")

    return ok


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "=" * 60)
    print("  AI Log Filter — End-to-End Pipeline Smoke Test")
    print("=" * 60)

    results = {
        "normalizer": test_normalizer(),
        "onnx_classifier": test_onnx_classifier(),
        "leef_enricher": test_leef_enricher(),
        "router_logic": test_router_logic(),
        "scorer_real": test_scorer_real(),
    }

    banner("Summary")
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        if not passed:
            all_pass = False
        print(f"  {status} {name}")

    print()
    if all_pass:
        print("  All smoke tests passed.")
    else:
        print("  Some tests FAILED — see output above.")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
