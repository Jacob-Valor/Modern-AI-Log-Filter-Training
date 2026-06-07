"""
Load test for the scoring pipeline.

Measures throughput (events/sec) and latency percentiles by running
a fixed number of events through the full pipeline in-process.
No Docker or network required.
"""

from __future__ import annotations

import statistics
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

SAMPLE_EVENTS = [
    "Jan 15 11:07:53 prod-srv01 sshd[22345]: Failed password "
    "for root from 10.0.0.5 port 44382 ssh2",
    "Jan 15 11:08:01 prod-srv01 sshd[22400]: Accepted publickey "
    "for deploy from 10.0.0.12 port 52301 ssh2",
    "10.0.0.5 - - [15/Jan/2026:11:07:53 +0000] \"GET /wp-admin HTTP/1.1\" 403 512",
    "10.0.0.12 - - [15/Jan/2026:11:07:53 +0000] \"GET /index.html HTTP/1.1\" 200 5120",
    "CEF:0|Palo Alto|NGFW|10.0|threat|SSH brute force|8|"
    "src=10.0.0.5 dst=192.168.1.1 spt=55000 dpt=22",
    "getFileInfo+success: return(ow[class=class "
    "org.apache.hadoop.hdfs.protocol.HdfsFileStatus",
    "Jan 15 11:07:53 prod-srv01 sudo: user : TTY=pts/0 ; "
    "PWD=/home/user ; USER=root ; COMMAND=/bin/cat /etc/shadow",
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


def percentile(data: list[float], p: float) -> float:
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def main() -> None:
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("\n" + "=" * 60)
    print(f"  AI Log Filter — Load Test ({n_events} events)")
    print("=" * 60)

    normalizer = LogNormalizer()
    enricher = LEEFEnricher()

    print("\n  Loading models …")
    t0 = time.perf_counter()
    scorer = build_scorer()
    scorer.preload_models()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Models loaded in {load_ms:.0f}ms\n")

    latencies: list[float] = []
    errors = 0

    print(f"  Running {n_events} events …")
    t_start = time.perf_counter()

    for i in range(n_events):
        raw = SAMPLE_EVENTS[i % len(SAMPLE_EVENTS)]
        t0 = time.perf_counter()
        try:
            ev = normalizer.normalize(raw)
            scored = scorer.score(ev)
            _ = enricher.enrich(scored, es_doc_id=f"load-test-{i}")
        except Exception:
            errors += 1
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    total_ms = (time.perf_counter() - t_start) * 1000

    latencies.sort()
    throughput = (n_events / total_ms) * 1000 if total_ms > 0 else 0

    print(f"\n{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    print(f"  Total events:  {n_events}")
    print(f"  Total time:    {total_ms:.0f}ms")
    print(f"  Throughput:    {throughput:.1f} events/sec")
    print(f"  Errors:        {errors}")
    print()
    print("  Latency percentiles:")
    for p in [50, 90, 95, 99]:
        v = percentile(latencies, p)
        print(f"    p{p:2d}: {v:.1f}ms")
    print(f"    min: {latencies[0]:.1f}ms")
    print(f"    max: {latencies[-1]:.1f}ms")
    print(f"    avg: {statistics.mean(latencies):.1f}ms")
    print()


if __name__ == "__main__":
    main()
