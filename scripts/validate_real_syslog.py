"""
Validate the scoring pipeline against real syslog patterns.

Tests the full pipeline (normalizer → scorer → enricher) with representative
production log events covering common attack vectors and benign traffic.
Reports per-category accuracy so you can assess real-world readiness.
"""

from __future__ import annotations

import json
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
from logfilter.pipeline.normalizer import LogNormalizer  # noqa: E402
from logfilter.pipeline.scorer import LogScorer  # noqa: E402

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
INFO = "\033[94m·\033[0m"

# ── Test cases by category ────────────────────────────────────────────────────
# Each entry: (category, raw_log, expected_priority_range, description)
# priority_range is a set of acceptable priorities. Empty = any.
TEST_CASES: list[tuple[str, str, set[str], str]] = [
    # ── SSH brute force ──────────────────────────────────────────────────────
    (
        "ssh_brute_force",
        "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password "
        "for root from 10.0.0.5 port 44382 ssh2",
        set(),
        "Single failed SSH password attempt",
    ),
    (
        "ssh_brute_force",
        "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password for "
        "invalid user admin from 192.168.1.100 port 33442 ssh2",
        set(),
        "Failed SSH with invalid user",
    ),
    (
        "ssh_brute_force",
        "Jan 15 11:07:53 prod-srv01 sshd[22345]: Connection closed by "
        "authenticating user root 10.0.0.5 port 44382 [preauth]",
        set(),
        "SSH connection closed during auth",
    ),
    (
        "ssh_brute_force",
        "Jan 15 11:07:53 prod-srv01 sshd[22345]: Disconnecting "
        "authenticating user root 10.0.0.5 port 44382: "
        "Too many authentication failures [preauth]",
        set(),
        "Too many SSH auth failures",
    ),
    # ── SSH successful login ─────────────────────────────────────────────────
    (
        "ssh_login",
        "Jan 15 11:08:01 prod-srv01 sshd[22400]: Accepted publickey "
        "for deploy from 10.0.0.12 port 52301 ssh2: RSA SHA256:abc123",
        set(),
        "Successful SSH public key login",
    ),
    (
        "ssh_login",
        "Jan 15 11:08:01 prod-srv01 sshd[22400]: pam_unix(sshd:session"
        "): session opened for user deploy by (uid=0)",
        set(),
        "PAM session opened for SSH user",
    ),
    # ── Web attacks ──────────────────────────────────────────────────────────
    (
        "web_attack",
        '10.0.0.5 - - [15/Jan/2026:11:07:53 +0000] "GET /wp-admin HTTP/1.1" 403 512',
        set(),
        "WordPress admin access attempt (403)",
    ),
    (
        "web_attack",
        "10.0.0.5 - - [15/Jan/2026:11:07:53 +0000] "
        "\"GET /../../../etc/passwd HTTP/1.1\" 400 342",
        set(),
        "Path traversal attempt",
    ),
    (
        "web_attack",
        "10.0.0.5 - - [15/Jan/2026:11:07:53 +0000] "
        "\"POST /api/login HTTP/1.1\" 200 128 "
        "\"-\" \"sqlmap/1.5\"",
        set(),
        "SQL injection tool detected in UA",
    ),
    # ── Normal web traffic ───────────────────────────────────────────────────
    (
        "web_normal",
        "10.0.0.12 - - [15/Jan/2026:11:07:53 +0000] "
        "\"GET /index.html HTTP/1.1\" 200 5120",
        set(),
        "Normal HTTP GET for index.html",
    ),
    (
        "web_normal",
        "10.0.0.12 - - [15/Jan/2026:11:07:53 +0000] "
        "\"GET /api/health HTTP/1.1\" 200 16 "
        "\"-\" \"ELB-HealthChecker/2.0\"",
        set(),
        "Health check from load balancer",
    ),
    # ── Firewall / IDS ───────────────────────────────────────────────────────
    (
        "firewall",
        "CEF:0|Palo Alto|NGFW|10.0|threat|SSH brute force|8|"
        "src=10.0.0.5 dst=192.168.1.1 spt=55000 dpt=22",
        set(),
        "CEF firewall SSH brute force alert",
    ),
    (
        "firewall",
        "CEF:0|Snort|IDS|2.9|attempted-admin|"
        "EternalBlue MS17-010|10|"
        "src=10.0.0.99 dst=10.0.0.1 spt=4444 dpt=445",
        set(),
        "IDS EternalBlue exploit attempt",
    ),
    # ── Authentication failures ──────────────────────────────────────────────
    (
        "auth_failure",
        "Jan 15 11:07:53 prod-srv01 sudo: user : "
        "TTY=pts/0 ; PWD=/home/user ; USER=root ; "
        "COMMAND=/bin/cat /etc/shadow",
        set(),
        "sudo attempt to read shadow file",
    ),
    (
        "auth_failure",
        "Jan 15 11:07:53 prod-srv01 login: "
        "FAILED LOGIN 2 FROM 10.0.0.5 FOR admin, "
        "Authentication failure",
        set(),
        "Login authentication failure",
    ),
    # ── Windows events ───────────────────────────────────────────────────────
    (
        "windows_event",
        json.dumps({
            "EventID": "4625",
            "Computer": "DC01",
            "Message": (
                "An account failed to log on. Subject: SYSTEM. "
                "Target: admin. Logon Type: 3. Failure Reason: "
                "Unknown user or bad password."
            ),
            "TimeCreated": "2026-01-15T11:07:53Z",
        }),
        set(),
        "Windows Event 4625 - failed logon",
    ),
    (
        "windows_event",
        json.dumps({
            "EventID": "4720",
            "Computer": "DC01",
            "Message": (
                "A user account was created. Subject: "
                "administrator. New Account: backdoor"
            ),
            "TimeCreated": "2026-01-15T11:07:53Z",
        }),
        set(),
        "Windows Event 4720 - user account created",
    ),
    # ── HDFS traces ──────────────────────────────────────────────────────────
    (
        "hdfs_trace",
        "getFileInfo+success: return(ow[class=class "
        "org.apache.hadoop.hdfs.protocol.HdfsFileStatus\n"
        "getBlockLocations+success: return(ow[class=class "
        "org.apache.hadoop.hdfs.protocol.LocatedBlocks\n"
        "bestNode+success: chosen bestnode = dn1 "
        "in nodes = [dn1, dn2, dn3]",
        set(),
        "Normal HDFS trace - getFileInfo",
    ),
]


def build_scorer() -> LogScorer:
    classifier = LogClassifier(
        model_path=ROOT / "models" / "log_classifier.onnx",
        scaler_path=ROOT / "models" / "scaler.json",
        feature_names_path=ROOT / "models" / "feature_names.json",
    )
    tier2 = Tier2Classifier(model_dir=ROOT / "models" / "tier2")
    ner = NERModel(
        model_id="cisco-ai/SecureBERT2.0-NER",
        device="cpu",
        batch_size=32,
        min_confidence=0.80,
        revision="792db5b",
    )
    biencoder = BiEncoderModel(
        model_id=str(ROOT / "models" / "biencoder" / "final"),
        device="cpu",
        batch_size=64,
        dedup_threshold=0.95,
        dedup_window_minutes=5.0,
        faiss_top_k=3,
        mitre_techniques_path=str(ROOT / "config" / "mitre_techniques.json"),
    )
    cross_encoder = CrossEncoderModel(
        model_id="cisco-ai/SecureBERT2.0-cross_encoder",
        device="cpu",
        batch_size=16,
        revision="960b923",
    )
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
    return LogScorer(
        config=config,
        classifier=classifier,
        tier2_classifier=tier2,
        ner_model=ner,
        biencoder=biencoder,
        cross_encoder=cross_encoder,
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("  AI Log Filter — Real Syslog Validation")
    print("=" * 60)

    normalizer = LogNormalizer()

    print(f"\n  {INFO} Loading models (first run takes ~15s)…")
    t0 = time.perf_counter()
    scorer = build_scorer()
    scorer.preload_models()
    load_time = (time.perf_counter() - t0) * 1000
    print(f"  {INFO} Models loaded in {load_time:.0f}ms\n")

    results: dict[str, dict[str, list]] = {}
    latencies: list[float] = []
    total = len(TEST_CASES)

    for i, (category, raw, expected_prios, desc) in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        ev = normalizer.normalize(raw)
        scored = scorer.score(ev)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if category not in results:
            results[category] = {"pass": [], "fail": []}

        has_entities = bool(scored.ai_entities)
        has_score = scored.ai_threat_score > 0
        pri_ok = (not expected_prios) or (scored.ai_priority in expected_prios)

        passed = has_score and pri_ok

        tag = PASS if passed else FAIL
        print(f"  [{i}/{total}] {tag} {desc}")
        print(
            f"    score={scored.ai_threat_score:.4f}  "
            f"priority={scored.ai_priority}  "
            f"mitre={scored.ai_mitre_technique!r}  "
            f"entities={'yes' if has_entities else 'no'}  "
            f"latency={elapsed_ms:.0f}ms"
        )
        if scored.tier2_used:
            print(f"    tier2_score={scored.tier2_score:.4f} (escalated from tier1)")

        entry = {"desc": desc, "score": scored.ai_threat_score, "priority": scored.ai_priority}
        if passed:
            results[category]["pass"].append(entry)
        else:
            results[category]["fail"].append(entry)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    print(f"\n{'─' * 60}")
    print("  Summary by Category")
    print(f"{'─' * 60}")
    for cat, cat_results in sorted(results.items()):
        n_pass = len(cat_results["pass"])
        n_fail = len(cat_results["fail"])
        n_total = n_pass + n_fail
        ratio = n_pass / n_total if n_total else 0
        status = PASS if ratio >= 0.5 else WARN
        print(f"  {status} {cat}: {n_pass}/{n_total} passed ({ratio:.0%})")
        for f in cat_results["fail"]:
            print(
                f"      {FAIL} {f['desc']} "
                f"(score={f['score']:.4f}, priority={f['priority']})"
            )

    print(f"\n{'─' * 60}")
    print("  Latency")
    print(f"{'─' * 60}")
    print(f"  avg={avg_latency:.0f}ms  p99={p99_latency:.0f}ms  total_events={total}")

    all_pass = all(len(v["fail"]) == 0 for v in results.values())
    if all_pass:
        print("\n  All categories pass.")
    else:
        print("\n  Some categories have failures — review above.")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
