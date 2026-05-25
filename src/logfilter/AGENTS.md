# src/logfilter/ KNOWLEDGE BASE

## OVERVIEW

Core runtime package. Contains the full log processing pipeline from ingestion to scoring to routing.

## STRUCTURE

```
src/logfilter/
├── __init__.py
├── config.py              # YAML loader with env var resolution
├── collector.py           # Syslog UDP/TCP receiver → Kafka producer
├── archive_consumer.py    # Kafka consumer → Elasticsearch indexer
├── kafka_router.py        # Routes scored events to downstream topics
├── api/                   # FastAPI scoring service
├── models/                # ONNX model wrappers
├── pipeline/              # Scoring pipeline stages
├── kafka/                 # Producer/consumer abstractions
├── security/              # Network security helpers
└── mitre/                 # ATT&CK technique mappings
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add config key | `config.py` + `config/config.yaml` | Supports `${VAR:default}` syntax |
| Change syslog port | `collector.py` | `DEFAULT_SYSLOG_PORT = 514` |
| Add Kafka routing rule | `kafka_router.py` | Topic mapping in `route_event()` |
| Fix archive indexing | `archive_consumer.py` | Uses `elasticsearch` client with bulk helper |
| Change API security | `api/security.py` | Token enforcement + rate limiting |

## CONVENTIONS

- All top-level modules use `structlog.get_logger(__name__)`
- `config.py` is the single source of truth for runtime config loading
- Kafka connections use manual commit mode (no auto-commit)
- Elasticsearch bulk indexing with retry via `tenacity`

## ANTI-PATTERNS

- **Never** import training modules here — runtime has no training deps
- **Never** use blocking I/O in FastAPI endpoints — always `async` for I/O bound ops
- **Never** log raw event payloads at INFO level — security/privacy risk
