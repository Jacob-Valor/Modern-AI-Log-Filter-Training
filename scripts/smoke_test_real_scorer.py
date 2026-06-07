"""
Real-model pipeline smoke test.

Exercises the scoring pipeline with actual ONNX models:
  - Tier-1: LogClassifier (XGBoost → ONNX)
  - Tier-2: Tier2Classifier (SecureBERT2.0 → ONNX, uncertain-band escalation)

NER, BiEncoder, and CrossEncoder are disabled via the built-in
Disabled* stubs (they require HuggingFace downloads not present in CI).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.biencoder import BiEncoderModel  # noqa: E402
from logfilter.models.classifier import LogClassifier  # noqa: E402
from logfilter.models.cross_encoder import CrossEncoderModel  # noqa: E402
from logfilter.models.ner import NERModel  # noqa: E402
from logfilter.models.tier2_classifier import Tier2Classifier  # noqa: E402
from logfilter.pipeline.enricher import LEEFEnricher  # noqa: E402
from logfilter.pipeline.normalizer import LogNormalizer  # noqa: E402
from logfilter.pipeline.scorer import LogScorer  # noqa: E402

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"


def banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def test_real_scorer() -> bool:
    banner("Real-model scorer — Tier-1 + Tier-2 ONNX inference")
    ok = True

    # ── Load real models ────────────────────────────────────────────────────
    classifier = LogClassifier(
        model_path=ROOT / "models" / "log_classifier.onnx",
        scaler_path=ROOT / "models" / "scaler.json",
        feature_names_path=ROOT / "models" / "feature_names.json",
    )
    tier2 = Tier2Classifier(model_dir=ROOT / "models" / "tier2")

    print(f"  {INFO} Tier-1 ready: {classifier.is_ready()}")
    print(f"  {INFO} Tier-2 ready: {tier2.is_ready()}")

    # ── Load real HF models ────────────────────────────────────────────────
    print(f"  {INFO} Loading NER model (cisco-ai/SecureBERT2.0-NER)…")
    ner = NERModel(
        model_id="cisco-ai/SecureBERT2.0-NER",
        device="cpu",
        batch_size=32,
        min_confidence=0.80,
        revision="792db5b",
    )
    print(f"  {INFO} Loading BiEncoder (models/biencoder/final)…")
    biencoder = BiEncoderModel(
        model_id=str(ROOT / "models" / "biencoder" / "final"),
        device="cpu",
        batch_size=64,
        dedup_threshold=0.95,
        dedup_window_minutes=5.0,
        faiss_top_k=3,
        mitre_techniques_path=str(ROOT / "config" / "mitre_techniques.json"),
    )
    print(f"  {INFO} Loading CrossEncoder (cisco-ai/SecureBERT2.0-cross_encoder)…")
    cross_encoder = CrossEncoderModel(
        model_id="cisco-ai/SecureBERT2.0-cross_encoder",
        device="cpu",
        batch_size=16,
        revision="960b923",
    )

    # ── Build scorer with ALL real models ──────────────────────────────────
    config = {
        "scoring": {
            "weights": {
                "classifier": 0.35,
                "entity_boost": 0.25,
                "cross_encoder": 0.40,
                "novelty": 0.0,
            },
            "entity_boost_value": 0.20,
            "dedup_penalty": 0.30,
            "routing": {"high": 0.80, "medium": 0.50, "low": 0.20},
            "tier2": {"uncertainty_low": 0.10, "uncertainty_high": 0.90},
        },
        "models": {
            "classifier": {"path": "models/log_classifier.onnx"},
            "biencoder": {"enabled": True},
            "ner": {"enabled": True},
            "cross_encoder": {"enabled": True},
        },
    }

    scorer = LogScorer(
        config=config,
        classifier=classifier,
        tier2_classifier=tier2,
        ner_model=ner,
        biencoder=biencoder,
        cross_encoder=cross_encoder,
    )

    normalizer = LogNormalizer()
    enricher = LEEFEnricher()

    # ── Test cases ──────────────────────────────────────────────────────────
    test_cases = [
        (
            "SSH brute force (failure-leaning)",
            "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for "
            "root from 10.0.0.5 port 44382 ssh2",
        ),
        (
            "Normal SSH login (benign-leaning)",
            "Jan 15 11:08:01 prod-srv01 sshd[22400]: Accepted publickey for "
            "deploy from 10.0.0.12 port 52301 ssh2",
        ),
        (
            "HDFS getFileInfo (normal HDFS trace)",
            "getFileInfo+success: return(ow[class=class org.apache.hadoop.hdfs"
            ".protocol.HdfsFileStatus\n"
            "getBlockLocations+success: return(ow[class=class org.apache.hadoop"
            ".hdfs.protocol.LocatedBlocks\n"
            "bestNode+success: chosen bestnode = dn1 in nodes = [dn1, dn2, dn3]",
        ),
        (
            "Multiple failed logins (suspicious)",
            "Jan 15 11:09:01 prod-srv01 sshd[22500]: Failed password for "
            "invalid user admin from 192.168.1.100 port 33442 ssh2\n"
            "Jan 15 11:09:02 prod-srv01 sshd[22501]: Failed password for "
            "invalid user admin from 192.168.1.100 port 33443 ssh2\n"
            "Jan 15 11:09:03 prod-srv01 sshd[22502]: Failed password for "
            "invalid user root from 192.168.1.100 port 33444 ssh2",
        ),
    ]

    for name, raw in test_cases:
        t0 = time.perf_counter()
        ev = normalizer.normalize(raw)
        scored = scorer.score(ev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        leef = enricher.enrich(scored, es_doc_id="smoke-test")

        tier2_flag = " [T2]" if scored.tier2_used else ""
        print(
            f"  {PASS if scored.ai_threat_score >= 0 else FAIL} "
            f"{name}{tier2_flag}"
        )
        print(
            f"    score={scored.ai_threat_score:.4f}  "
            f"priority={scored.ai_priority}  "
            f"tier1={scored.classifier_score:.4f}  "
            f"latency={elapsed_ms:.1f}ms"
        )
        if scored.tier2_used:
            print(f"    tier2_score={scored.tier2_score:.4f}")
        print(f"    mitre={scored.ai_mitre_technique!r}  entities={scored.ai_entities!r}")

        # Basic sanity checks
        score_ok = 0.0 <= scored.ai_threat_score <= 1.0
        priority_ok = scored.ai_priority in {"HIGH", "MEDIUM", "LOW", "INFO"}
        leef_ok = leef.startswith("LEEF:2.0|")

        if not (score_ok and priority_ok and leef_ok):
            ok = False
            print(
                f"    {FAIL} sanity check failed: "
                f"score_ok={score_ok} priority_ok={priority_ok} "
                f"leef_ok={leef_ok}"
            )

    return ok


def main() -> None:
    print("\n" + "=" * 60)
    print("  AI Log Filter — Real-Model Smoke Test")
    print("=" * 60)

    passed = test_real_scorer()

    banner("Summary")
    status = PASS if passed else FAIL
    print(f"  {status} real_scorer")
    print()
    if passed:
        print("  All real-model smoke tests passed.")
    else:
        print("  Some tests FAILED — see output above.")
    print()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
