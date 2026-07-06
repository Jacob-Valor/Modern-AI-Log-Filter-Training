# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-25
**Commit:** 51bc95f
**Branch:** dev

## OVERVIEW

AI-powered log preprocessing pipeline for IBM QRadar SIEM. Ingests syslog-like events, archives them, scores them with a tiered ML pipeline, and emits enriched LEEF payloads for downstream routing.

Core stack: Python 3.10+, FastAPI, Kafka, Elasticsearch, XGBoost, Transformers (SecureBERT2.0), ONNX Runtime, Docker.

## STRUCTURE

```
.
├── src/logfilter/          # Main source code
│   ├── api/                # FastAPI scoring service
│   ├── models/             # ONNX model wrappers (classifier, tier2, biencoder, NER, cross-encoder)
│   ├── pipeline/           # Scoring pipeline (normalizer → scorer → enricher → router)
│   ├── kafka/              # Producer/consumer abstractions
│   ├── security/           # Network security helpers
│   ├── mitre/              # ATT&CK technique mappings
│   ├── utils/              # Circuit breaker + shared utilities
│   ├── collector.py        # UDP/TCP syslog receiver
│   ├── archive_consumer.py # Elasticsearch archive consumer
│   ├── kafka_router.py     # Kafka-to-Kafka routing
│   └── config.py           # YAML config loader with ${VAR:default} resolution
├── training/               # Training scripts for XGBoost and transformers
├── tests/                  # pytest suite (coverage target: 90%)
├── notebooks/              # Kaggle training notebooks
├── scripts/                # Utility scripts (smoke tests, verification)
├── config/                 # Runtime YAML configs
├── models/                 # Output directory for trained artifacts
├── HDFS_v3_TraceBench/     # Preprocessed HDFS dataset
├── docker-compose.yml      # Full stack (API, Kafka, ES, Grafana)
├── Dockerfile*             # Multi-stage builds
├── Makefile                # Dev/test/train automation
├── pyproject.toml          # Packaging, ruff, pytest, coverage config
└── requirements-api.txt    # API Docker image deps (torch CPU + transformers)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new model | `src/logfilter/models/` | Implement `load()` + `predict()` + `to_onnx()`, add to `models/__init__.py` |
| Change scoring weights | `config/config.yaml` | `weights:` section controls tier blending |
| Add API endpoint | `src/logfilter/api/app.py` + `schemas.py` | Follow Pydantic v2 patterns |
| Change Kafka consumer | `src/logfilter/kafka/consumer.py` | Manual commit after successful batch |
| Change Kafka producer | `src/logfilter/kafka/producer.py` | Retry via `tenacity` |
| Add circuit breaker | `src/logfilter/utils/circuit_breaker.py` | States: CLOSED → OPEN → HALF_OPEN |
| Train tier-1 | `make train` or `training/train.py` | Outputs ONNX + JSON to `models/` |
| Train tier-2 | `training/train_transformer.py` | Fine-tunes SecureBERT2.0-base |
| Add Kafka topic | `src/logfilter/kafka_router.py` | Uses `kafka-python` with manual commits |
| Rebuild Docker | `make build` | Multi-stage Dockerfile for production |
| Run smoke test | `make smoke-pipeline` | End-to-end via `scripts/smoke_test_pipeline.py` |

## CONVENTIONS

- **Ruff**: line-length 100, target py310, rules E/F/I/UP
- **Imports**: `from __future__ import annotations` in every module
- **Logging**: `structlog` in runtime, standard `logging` in training
- **Paths**: `Path(__file__)` chains to project root; never hardcode absolute paths
- **Config**: YAML with `${ENV_VAR:default}` placeholder resolution via `config.py`
- **Security env vars**: Use `${VAR:?error message}` for required tokens in config.yaml, not `${VAR:}` (empty default), to prevent fail-open auth paths
- **Models**: ONNX Runtime for inference; native format kept for retraining
- **Security**: API tokens via `X-API-Token` / `X-Admin-Token` headers; no default secrets

## ANTI-PATTERNS (THIS PROJECT)

- **Never** hardcode model paths — always resolve from `Path(__file__)` chain
- **Never** use `print()` in runtime code — use `structlog.get_logger()`
- **Never** commit `.env` — only `.env.example` with placeholder values
- **Never** skip ONNX export — every trained model must export to ONNX for production
- **Never** use unbounded Kafka consumer loops — always implement graceful shutdown
- **Coverage must stay ≥90%** — `tool.coverage.report.fail_under = 90` in pyproject.toml

## UNIQUE STYLES

- **SafeMaxAbsScaler**: Custom JSON-backed scaler with finite-value validation (replaces sklearn at runtime)
- **3-tier scoring**: Sigma rules (fast) → BiEncoder + FAISS (dedup + ATT&CK retrieval) → NER + CrossEncoder (precise)
- **Kaggle self-bootstrap**: Notebooks clone the repo from GitHub if not found locally
- **Dataset auto-link**: Notebooks search `/kaggle/input` for trace files and symlink them

## COMMANDS

```bash
make dev-install      # Install all deps + dev tools
make train            # Train tier-1 XGBoost (SAMPLE_N=50000 default)
make test             # Run pytest suite
make test-coverage    # Run tests with 90% coverage gate
make lint             # Ruff lint
make format           # Ruff format
make verify           # lint + test-coverage + smoke-pipeline
make up               # Start full Docker stack
make down             # Stop Docker stack
```

## NOTES

- HDFS TraceBench data is in `HDFS_v3_TraceBench/preprocessed/` — ~2.2GB total
- Kaggle dataset: `jacobvalor/hdfs-tracebench-preprocessed-logs` (public)
- Docker stack binds Kafka and API to localhost by default
- OpenAPI docs disabled unless `LOGFILTER_ENABLE_DOCS=1`

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
