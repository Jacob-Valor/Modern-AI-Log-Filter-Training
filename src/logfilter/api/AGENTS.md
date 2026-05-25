# src/logfilter/api/ KNOWLEDGE BASE

## OVERVIEW

FastAPI scoring service. Exposes `/score`, `/score/batch`, `/health`, `/metrics`, and admin `/reload` endpoints.

## STRUCTURE

```
src/logfilter/api/
├── __init__.py
├── app.py          # FastAPI application + lifespan + endpoints
├── schemas.py      # Pydantic v2 request/response models
└── security.py     # Token validation + rate limiting (framework-independent)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add endpoint | `app.py` | Follow existing pattern with `@app.post` + Pydantic schema |
| Change request validation | `schemas.py` | Pydantic v2 `BaseModel` with `Field()` constraints |
| Change auth logic | `security.py` | `require_configured_token()` uses `hmac.compare_digest` |
| Adjust rate limits | `app.py` | `RATE_LIMIT_PER_MINUTE` default is 60 |
| Add response field | `schemas.py` + `scorer.py` | Update `ScoreResponse` dataclass first |
| Enable OpenAPI docs | `app.py` | Set `LOGFILTER_ENABLE_DOCS=1` |

## CONVENTIONS

- All endpoints use Pydantic v2 schemas for request/response validation
- Security is framework-independent — `security.py` can be reused outside FastAPI
- Rate limiting is in-memory per-process (not distributed)
- Admin endpoints require `X-Admin-Token`; scoring requires `X-API-Token`
- Metrics endpoint returns Prometheus text format

## ANTI-PATTERNS

- **Never** use `async def` for CPU-bound scoring — call `scorer.score()` in `run_in_threadpool`
- **Never** skip schema validation — always define a Pydantic model for request bodies
- **Never** log raw event payloads at INFO level — security/privacy risk
- **Never** return stack traces in HTTP responses — log them internally
