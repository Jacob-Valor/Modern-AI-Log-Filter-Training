# tests/ KNOWLEDGE BASE

## OVERVIEW

pytest suite with 90% coverage gate. Flat structure — one test file per source module, not mirrored to `src/` depth.

## STRUCTURE

```
tests/
├── __init__.py
├── test_api_app.py           # FastAPI endpoint tests (mocked scorer)
├── test_api_security.py      # Token + rate limit tests
├── test_archive.py           # Elasticsearch archive helpers
├── test_archive_consumer.py  # Kafka consumer wiring
├── test_benchmark.py         # Benchmark script imports
├── test_biencoder.py         # BiEncoder dedup + retrieval
├── test_circuit_breaker.py   # Circuit breaker state machine
├── test_classifier.py        # SafeMaxAbsScaler + LogClassifier
├── test_collector.py         # Syslog receiver logic
├── test_config.py            # YAML env-var resolution
├── test_enricher.py          # LEEF payload construction
├── test_kafka_consumer.py    # Consumer batch + commit
├── test_kafka_producer.py    # Producer send + flush
├── test_kafka_router.py      # Router API client
├── test_model_wrappers.py    # NER + cross-encoder helpers
├── test_network_security.py  # CIDR allowlist
├── test_normalizer.py        # Syslog/CEF/JSON parsing
├── test_router_pipeline.py   # Routing decisions + syslog sender
├── test_scorer.py            # Scoring orchestration
├── test_telemetry.py         # OTEL noop fallback paths
├── test_thresholds.py        # Tier-2 threshold metrics
└── test_tier2_classifier.py  # Tier2 uncertainty band
```

## CONVENTIONS

- Coverage target: ≥90% (`tool.coverage.report.fail_under = 90` in pyproject.toml)
- Use `tmp_path` fixture for filesystem isolation
- Use `monkeypatch` for dependency injection (onnxruntime, xgboost, transformers)
- Mock external services (Kafka, Elasticsearch) — never connect in unit tests
- Fake model classes replace real ONNX/Transformers models for speed

## ANTI-PATTERNS

- **Never** import real models in tests — use fakes or monkeypatch
- **Never** test against live Docker services in unit tests
- **Never** skip coverage on new code — every module must have tests
- **Never** use `print()` in tests — assertions should be self-describing
