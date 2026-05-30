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
## HuggingFace model packaging

The scoring service downloads SecureBERT2.0 weights from HuggingFace Hub at
first use. For air-gapped, reproducible, or faster cold starts, pre-download
the models and pin exact revisions.

### Pin revisions in config

Set per-model revision hashes in `config/config.yaml` or via environment:

```bash
LOGFILTER_NER_REVISION=abc123def456
LOGFILTER_BIENCODER_REVISION=abc123def456
LOGFILTER_CROSS_ENCODER_REVISION=abc123def456
```

Leave empty to use the default branch head.

### Docker image (baked models)

The API Dockerfile runs `scripts/download_hf_models.py` during the build.
Models are cached in `/app/models/hf-cache` and the image is self-contained.
Override the target directory with the `HF_CACHE_DIR` build arg:

```bash
docker build -f docker/api/Dockerfile \
  --build-arg HF_CACHE_DIR=/app/models/hf-cache \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t logfilter-api:latest .
```

### Kubernetes initContainer (runtime fetch)

If you do not bake models into the image, the `api.yaml` Deployment includes an
`initContainer` that downloads models to a shared `emptyDir` volume before the
main container starts. Set `HF_TOKEN` in `k8s/secret.yaml` if any model is
gated or private.

### Manual pre-download

```bash
python scripts/download_hf_models.py \
  --config config/config.yaml \
  --cache-dir /mnt/hf-cache \
  --model all
```

The script reads model IDs and revisions from config, so a single command
synchronises the cache with the running configuration.
