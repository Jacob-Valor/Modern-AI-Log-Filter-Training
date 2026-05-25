# scripts/ KNOWLEDGE BASE

## OVERVIEW

Operational and validation scripts. Run locally or in CI. Not part of the installed package.

## STRUCTURE

```
scripts/
├── benchmark.py              # Latency benchmark for scoring pipeline
├── certs/                    # Self-signed certs for local TLS testing
├── evaluate_tier2_thresholds.py  # Tier-2 precision/recall sweep
├── smoke_test_pipeline.py    # End-to-end validation (no Docker required)
└── verify_tier2_artifact.py  # Check ONNX + tokenizer files exist
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run smoke test | `smoke_test_pipeline.py` | Validates normalizer → classifier → enricher → router |
| Benchmark scoring | `benchmark.py` | Times `score_batch()` with mocked models |
| Tune tier-2 thresholds | `evaluate_tier2_thresholds.py` | Outputs precision/recall/F1 per cutoff |
| Verify model artifacts | `verify_tier2_artifact.py` | Checks `models/tier2/` has required files |

## CONVENTIONS

- Scripts are runnable directly: `python scripts/smoke_test_pipeline.py`
- All scripts exit 0 on success, non-zero on failure (CI-friendly)
- Smoke test mocks heavy ML models — runs in seconds
- Benchmark prints human-readable latency percentiles

## ANTI-PATTERNS

- **Never** add training code here — use `training/` instead
- **Never** commit production secrets — scripts only read from env/config
- **Never** require Docker for validation — smoke test must run standalone
