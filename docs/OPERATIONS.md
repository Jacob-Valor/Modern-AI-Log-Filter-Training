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

## High-availability Compose overlay

`docker-compose.prod.yml` is an overlay for local HA validation, not a complete
cluster deployment. Start it with the base stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The overlay adds `kafka-2`, `kafka-3`, `elasticsearch-2`, and
`elasticsearch-3`; sets Kafka internal replication factors and minimum ISR for
three brokers; and configures API/archive/router Kafka clients to use all three
brokers. Validate the rendered configuration before promotion:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config -q
```

Keep this overlay private: Kafka is still PLAINTEXT inside the Compose network.
For production trust boundaries, replace it with broker-side SASL/TLS, managed
Kafka/Elasticsearch, or Kubernetes manifests wired to your platform secrets.

## Rollback procedure

If a deployment causes degraded service (high latency, model load failures,
or incorrect scoring), roll back using the procedure below.

### Docker Compose (single host)

```bash
# 1. Identify the last known good image tag
export LAST_GOOD_TAG=$(docker images logfilter-api --format '{{.Tag}}' | head -n 2 | tail -n 1)

# 2. Stop the failing service
docker compose stop logfilter-api

# 3. Revert to the last known good image
docker compose up -d logfilter-api --no-deps --force-recreate

# 4. Verify health
curl -fsS http://localhost:8080/health
```

### Kubernetes (cluster)

```bash
# 1. Roll back the API deployment to the previous revision
kubectl rollout undo deployment/logfilter-api -n logfilter

# 2. Monitor rollout status
kubectl rollout status deployment/logfilter-api -n logfilter

# 3. Verify pods are healthy
kubectl get pods -n logfilter -l app.kubernetes.io/name=logfilter-api

# 4. Verify service health via ingress
curl -fsS https://api.yourdomain.com/health
```

### Model artifact rollback

If the issue is a bad model artifact (ONNX/scaler) rather than code:

```bash
# 1. Restore previous model PVC snapshot or re-run training
make train

# 2. Restart the API to pick up the restored artifacts
kubectl rollout restart deployment/logfilter-api -n logfilter
```

### Emergency circuit breaker

If the scorer is failing but you need to preserve log flow:

```bash
# Scale the router to zero to stop consuming from Kafka
kubectl scale deployment/logfilter-router --replicas=0 -n logfilter

# Events remain in Kafka raw-logs until the issue is resolved
```

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
Models are cached in `/app/hf-cache` and the image is self-contained.
Override the target directory with the `HF_CACHE_DIR` build arg:

```bash
docker build -f docker/api/Dockerfile \
  --build-arg HF_CACHE_DIR=/app/hf-cache \
  -t logfilter-api:latest .
```

For private or gated models, set the `HF_TOKEN` build arg:

```bash
docker build -f docker/api/Dockerfile \
  --build-arg HF_CACHE_DIR=/app/hf-cache \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t logfilter-api:latest .
```

### Kubernetes (models mounted from PVC)

Models are baked into the Docker image at build time, so the `api.yaml`
Deployment has no `initContainer` for model download. The image must
include the models (build with `make build` or the snippet above); the
container starts self-contained.

If you do not bake models into the image, mount a shared PVC pre-populated
with `scripts/download_hf_models.py --cache-dir /app/hf-cache --model all`
and set `HF_HOME=/app/hf-cache` on the Deployment.

### Manual pre-download

```bash
python scripts/download_hf_models.py \
  --config config/config.yaml \
  --cache-dir /app/hf-cache \
  --model all
```

The script reads model IDs and revisions from config, so a single command
synchronises the cache with the running configuration.

## Monitoring runbook

Prometheus rules and a Grafana dashboard ship with the repository.

### Alerts

| Alert | Severity | Meaning | Action |
|---|---|---|---|
| `LogfilterAPIDown` | critical | API unreachable | Check pod status, ingress, and API logs |
| `LogfilterModelNotLoaded` | warning | ONNX/scaler missing | Verify `models/` PVC mount and artifact presence |
| `LogfilterHighLatencyP99` | warning | p99 > 500 ms | Scale replicas or reduce batch size; check CPU throttling |
| `LogfilterHighLatencyP95` | warning | p95 > 200 ms | Same as above; lower threshold for early warning |
| `LogfilterNoEventsScored` | warning | Zero throughput | Verify collector is sending, Kafka topics exist, consumer lag |
| `LogfilterHighDuplicateRate` | warning | >50% duplicates | Check dedup window or upstream log replay |
| `LogfilterHighSigmaMatchRate` | info | >80% Sigma matches | Review rule calibration; may indicate noisy rules |
| `LogfilterHighThreatVolume` | info | >10 HIGH/sec | Escalate to SOC if sustained |

### Dashboard

Open Grafana at `http://localhost:3000` (Docker) or your cluster ingress.
The "LogFilter — AI Scoring Pipeline" dashboard auto-provisions on startup.

Panels:
- **API Up** — binary health indicator
- **Scoring p99 Latency** — tail latency from `logfilter_scoring_latency_ms`
- **Duplicate Rate** — ratio of deduplicated events
- **Models Loaded** — green when all ONNX artifacts are ready
- **Events Scored / sec** — throughput by priority
- **Scoring Latency** — p50/p95/p99 timeseries
- **Duplicate & Sigma Match Rate** — anomaly indicators
- **Threat Score Distribution** — p99 threat score trend

## Service Level Objectives

The table below defines formal SLOs, SLIs, and error budgets for the LogFilter
scoring pipeline.

| SLO | SLI | Target | Error Budget | Measurement Window |
|---|---|---|---|---|
| Latency | p99 scoring latency | < 500 ms | 1% of requests may exceed | 30 days |
| Latency | p95 scoring latency | < 200 ms | 5% of requests may exceed | 30 days |
| Availability | API health endpoint success rate | > 99.9% | 0.1% downtime | 30 days |
| Throughput | Events scored per second | > 1000 rps | < 10% degradation | 30 days |
| Model Freshness | Time since last model reload | < 7 days | N/A | Continuous |

### Alert severity and burn rate

Prometheus burn-rate alerts fire when error budgets are consumed faster than
expected:

- **Fast burn (14.4x)**: page on-call within 2 hours of budget exhaustion.
  Fires when < 99% of requests are under 500ms over a 1-hour window.
- **Slow burn (2x)**: create ticket within 3 days of budget exhaustion.
  Fires when < 99% of requests are under 500ms over a 6-hour window.

### Dashboard panels for SLO tracking

- **Scoring p99 Latency** — tracks against 500ms SLO line
- **Error Budget Remaining** — visualises remaining budget for the current window
- **API Availability** — binary up/down indicator with 99.9% target
- **Burn Rate** — rate of budget consumption over 1h and 6h windows

## Disaster recovery

### Complete cluster loss

If the entire K8s cluster or Docker host is lost:

```bash
# 1. Restore Elasticsearch data from snapshot
# Ensure your ES cluster has snapshot repositories configured.
# See: https://www.elastic.co/guide/en/elasticsearch/reference/current/snapshots.html

# 2. Redeploy the stack from a clean checkout
git clone <repo> /opt/logfilter
cd /opt/logfilter
make verify
kubectl apply -f k8s/

# 3. Re-download HuggingFace models
kubectl exec deployment/logfilter-api -n logfilter -- \
  python scripts/download_hf_models.py --config config/config.yaml

# 4. Verify all pods are healthy
kubectl get pods -n logfilter
kubectl rollout status deployment/logfilter-api -n logfilter
```

### Kafka data loss

If Kafka logs are lost (e.g. volume corruption):

- **Raw events are still archived** in Elasticsearch (see
  `docs/OPERATIONS.md` chain-of-custody section; daily indices under
  `raw-logs-*`).
- **No automated backfill tool exists** between Elasticsearch and Kafka.
  Replay must be scripted per deployment using the `scripts/` utilities
  (`replay_archive_to_kafka.py` is a TODO; track via the project issue
  tracker).
- **Forensic recovery** is unaffected: the `raw_log_ref` in any LEEF
  payload still resolves to the original raw event via ES `get_by_id`.
- **Forwarded events to QRadar** (the LEEF stream) are NOT recoverable
  from Kafka once the topic is gone. To re-emit to QRadar, query
  Elasticsearch for the original raw logs, re-score them through the API
  (`POST /score`), and forward the resulting LEEF payload via the router.

This is a known gap; do not rely on Kafka as the system-of-record.

### Model artifact corruption

If ONNX/scaler artifacts are corrupted:

```bash
# Re-train from the latest validated dataset
make train
make evaluate

# Verify the new artifacts
python scripts/verify_tier2_artifact.py

# Promote to production
kubectl rollout restart deployment/logfilter-api -n logfilter
```

### Customising thresholds

Edit `config/prometheus/alerts.yml` and redeploy:

```bash
# Docker Compose
docker compose restart prometheus

# Kubernetes
kubectl create configmap logfilter-alerts \
  --from-file=config/prometheus/alerts.yml -n logfilter --dry-run=client -o yaml | kubectl apply -f -
# Then reload Prometheus (if your cluster exposes the reload endpoint)
curl -X POST http://prometheus:9090/-/reload
```
