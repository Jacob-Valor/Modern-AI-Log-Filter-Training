# src/logfilter/pipeline/ KNOWLEDGE BASE

## OVERVIEW

Scoring pipeline stages. Events flow: `normalizer` → `scorer` → `enricher` → `router`.

## STRUCTURE

```
src/logfilter/pipeline/
├── __init__.py
├── normalizer.py          # Syslog → structured NormalizedEvent
├── scorer.py              # 3-tier AI scoring (Sigma → BiEncoder → NER+CrossEncoder)
├── enricher.py            # NormalizedEvent → LEEF payload
├── router.py              # Route by score threshold to downstream
└── archive.py             # Elasticsearch archive helpers
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change score formula | `scorer.py` | `compute_score()` applies weighted blend |
| Adjust tier thresholds | `scorer.py` | `TIER1_LOW=0.10`, `TIER1_HIGH=0.90` |
| Add new entity type | `scorer.py` + `enricher.py` | Update `ScoredEvent` dataclass |
| Change LEEF format | `enricher.py` | `LEEFEnricher.to_leef()` |
| Add normalization rule | `normalizer.py` | `LogNormalizer.normalize()` |
| Change routing logic | `router.py` | Score threshold → topic mapping |

## CONVENTIONS

- `ScoredEvent` dataclass is the central data structure — extend here first
- All scoring weights read from `config/config.yaml`
- `normalizer.py` supports multiple `LogSourceType` variants
- `archive.py` uses Elasticsearch bulk API with retry

## ANTI-PATTERNS

- **Never** mutate `NormalizedEvent` after scoring — create new `ScoredEvent`
- **Never** skip tier-1 before tier-2 — cascade order is enforced
- **Never** hardcode LEEF field names — use `enricher.py` constants
