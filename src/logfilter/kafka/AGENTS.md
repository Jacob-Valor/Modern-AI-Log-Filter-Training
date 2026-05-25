# src/logfilter/kafka/ KNOWLEDGE BASE

## OVERVIEW

Producer and consumer abstractions around `kafka-python`. Used by the collector, archive consumer, and Kafka router.

## STRUCTURE

```
src/logfilter/kafka/
├── __init__.py
├── producer.py    # LogProducer — wraps KafkaProducer with retry
└── consumer.py    # ArchiveConsumer + ScorerConsumer — manual commit pattern
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change producer batching | `producer.py` | `batch_size_bytes`, `linger_ms` constructor params |
| Add consumer | `consumer.py` | Extend `BaseConsumer` with `process_batch()` |
| Fix commit logic | `consumer.py` | Manual commit after successful batch processing |
| Change retry policy | `producer.py` | Uses `tenacity` with exponential backoff |
| Add telemetry | Both | Kafka header context injection via `telemetry` |

## CONVENTIONS

- Manual commit mode only — no auto-commit
- Consumers implement graceful shutdown via `stop()` + threading.Event
- Producer batches are flushed on `close()`
- All Kafka errors are caught and logged; never swallowed silently

## ANTI-PATTERNS

- **Never** use auto-commit — always manual commit after successful processing
- **Never** commit before writing to Elasticsearch — risk of data loss
- **Never** create unbounded consumer loops — always check `self._running`
- **Never** ignore `KafkaError` — log and retry or circuit-break
