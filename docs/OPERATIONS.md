# Operations notes

This repository is safe local scaffolding, not a complete production deployment.
Keep production-specific secrets, certificate material, ingress hostnames, and
alert thresholds outside the repository and inject them through your deployment
platform.

## Local health checks

- API readiness: `curl -fsS http://localhost:8080/health`
- Prometheus metrics: `curl -fsS http://localhost:8080/metrics`
- Admin JSON metric snapshot: `curl -fsS -H "X-Admin-Token: $LOGFILTER_ADMIN_TOKEN" http://localhost:8080/metrics/snapshot`
- Docker stack status: `docker compose ps`

The Compose file includes health checks for Kafka, Elasticsearch, and the API.
The collector, archive consumer, and router are long-running workers; monitor
their container health through process state, restart count, and structured logs
unless dedicated worker health endpoints are added later.

## Verification before deployment

Run these checks from a clean checkout before promoting images or model
artifacts:

```bash
make verify
make audit
python scripts/verify_tier2_artifact.py
```

Run `make smoke-test` only after the API stack is running and `.env` contains
real local tokens.

## Production boundaries

- Terminate API TLS at a trusted ingress, load balancer, or service mesh.
- Keep Kafka, Elasticsearch, Prometheus, and Grafana private to the cluster or
  management network.
- Rotate `LOGFILTER_API_TOKEN`, `LOGFILTER_ADMIN_TOKEN`, `ES_PASSWORD`, and
  `GRAFANA_ADMIN_PASSWORD` through a secret manager; never commit `.env` files.
- Replace local plaintext Kafka with broker-side SASL/TLS before crossing trust
  boundaries. Configure via KAFKA_SECURITY_PROTOCOL and related environment
  variables; the default is PLAINTEXT for local development only.
- Set service CPU and memory limits based on measured ingest rate, model size,
  and queue depth. The API has a conservative Compose limit; workers still need
  production sizing from load tests.
- Pre-download or package Hugging Face model artifacts for offline and
  repeatable production starts.
