# Security Policy

## Supported Scope

This repository is a local/development deployment template for the LogFilter
pipeline. Production deployments must place public traffic behind the owning
organization's ingress, firewall, TLS, identity, and monitoring controls.

## Security Defaults

- Required secrets have no unsafe defaults.
- Scoring endpoints require `X-API-Token`.
- The collector accepts only `SYSLOG_ALLOWED_CIDRS`; the default is loopback.
- Metrics endpoints require `X-Metrics-Token` (or `Authorization: Bearer <token>`).
- All token checks fail closed when the token is not configured.
- API docs and OpenAPI JSON are disabled unless `LOGFILTER_ENABLE_DOCS=1`.
- Kafka, API, Elasticsearch, Kibana, Prometheus, Grafana, and collector ports
  are bound to `127.0.0.1` in the default Compose stack.
- Python service containers run as non-root users, drop Linux capabilities,
  prevent privilege escalation, and use read-only root filesystems. Specifically
  the `archive`, `logfilter-api`, `collector`, and `router` services:
  `docker-compose.yml` merges the `&python-service-security` anchor
  (`no-new-privileges:true`, `cap_drop: ALL`, `read_only: true`, tmpfs `/tmp`)
  into each of them.
- Model artifact integrity is checked against `models/model_manifest.json` at API
  startup. Mismatches are a logged warning by default; set `LOGFILTER_MODELS_STRICT=1`
  to make a mismatch fatal and refuse startup.
- Runtime scaler artifacts are JSON-only; pickle/joblib scaler loading is
  rejected.

## Production Checklist

Before exposing this system outside a developer machine:

1. Terminate TLS at a trusted ingress or service mesh (not self-signed dev certs).
   Replace `scripts/certs/` with certificates from your PKI (cert-manager, Let's
   Encrypt, or cloud ACM). The nginx config lives at `config/nginx/nginx.conf`.
2. Restrict collector access to trusted log forwarder CIDRs (`SYSLOG_ALLOWED_CIDRS`).
3. Keep Elasticsearch, Prometheus, Grafana, Kafka, and Kibana behind private
   network boundaries — the Compose stack binds them to `127.0.0.1` by default.
4. Rotate `LOGFILTER_API_TOKEN`, `LOGFILTER_METRICS_TOKEN`, `ES_PASSWORD`,
   `GRAFANA_ADMIN_PASSWORD`, `QRADAR_GATEWAY_HMAC_SECRET`, and
   `LOGFILTER_RESPONSE_GATEWAY_HMAC_SECRET` through your secret manager (Vault,
   AWS SM, GCP SM). Never store secrets in `.env` on disk in production.
5. Set `CORS_ALLOW_ORIGINS` to your actual frontend origin(s). The API refuses
   to start with localhost CORS origins in production mode (when
   `LOGFILTER_ENABLE_DOCS` is off). Leave empty for headless deployments.
6. Configure Prometheus to authenticate against `/metrics` using the bearer
   token file. The Compose stack writes `LOGFILTER_METRICS_TOKEN` to
   `/etc/prometheus/secrets/metrics-bearer-token` at Prometheus startup; the
   scrape config references `bearer_token_file`. If you manage Prometheus
   separately, add `bearer_token_file` or `authorization` to your scrape config.
7. Enable Kafka SASL_SSL for broker-side TLS and authentication before
   crossing trust boundaries. The Compose stack uses PLAINTEXT inside the
   private network — acceptable for local development but not for production.
8. Run `make verify` and `make audit` in CI before deploys.
9. Build images in CI from a clean checkout and scan them with your container
   scanner.
10. Treat model artifacts as deployable code; only promote artifacts from trusted
    training jobs. Run `make model-manifest-validate` on every deploy, and set
    `LOGFILTER_MODELS_STRICT=1` in production so a hash mismatch fails startup
    instead of only logging a warning.
11. For high availability, layer `docker-compose.prod.yml` over the base stack
    (`docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`)
    to expand Kafka to 3 brokers, ES to 3 data nodes, and raise replication
    factors. See README for details.

## Reporting

Do not put credentials, production logs, or sensitive customer data in issues.
Report vulnerabilities privately through your normal internal security channel.

## Vulnerability Exceptions

No active vulnerability exceptions. `make audit` and CI run `pip-audit`
without ignored advisories.

The former `diskcache`/pySigma exception was removed by replacing pySigma with
the scorer's constrained `yaml.safe_load_all` text-rule parser. The former
PyTorch exception was removed by upgrading to the patched 2.13 release line.

## Incident Response Runbook

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | Pipeline down, data loss | 15 min | Collector unreachable, Kafka broker down, ES cluster red |
| P2 | Degraded throughput | 1 hour | Consumer lag > 10k, API p99 > 2s, scoring errors > 1% |
| P3 | Non-urgent anomaly | 4 hours | Model drift detected, disk usage > 80%, cert expiry < 30d |

### P1: Pipeline Down

**Symptoms**: Collector unreachable, Kafka producer errors, API returning 5xx.

1. Check service health: `docker compose ps` or `kubectl get pods`
2. Check Kafka: `kafka-topics.sh --bootstrap-server localhost:9092 --list`
3. Check Elasticsearch: `curl -s localhost:9200/_cluster/health?pretty`
4. Check logs: `docker compose logs -f --tail=100 <service>`
5. If Kafka is down: restart broker, check disk space, verify advertised listeners
6. If ES is down: check heap usage, disk space, cluster state
7. If collector is down: check SYSLOG_ALLOWED_CIDRS, port bindings

**Escalation**: If restart doesn't restore within 15 min, page on-call.

### P2: Degraded Throughput

**Symptoms**: Consumer lag growing, API latency spiking, scoring errors.

1. Check consumer lag: `kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe <group>`
2. Check API metrics: `curl -s localhost:8080/metrics`
3. Check model loading: scorer logs show "Model manifest mismatch" warnings
4. If lag is growing: increase consumer instances or batch size
5. If API is slow: check ONNX session thread count, memory pressure
6. If scoring errors: check model manifest, verify artifacts match

### P3: Non-Urgent Anomaly

**Symptoms**: Model drift, disk usage high, cert expiry approaching.

1. Model drift: check DriftDetector logs, retrain if PSI > 0.25
2. Disk usage: `df -h`, clean old ES indices or Kafka retention
3. Cert expiry: rotate certs before 30-day window

## Scaling Playbook

### Horizontal Scaling

- **Collector**: Scale replicas behind load balancer; each binds to its own port
- **API**: Scale replicas behind load balancer; stateless, no sticky sessions
- **Kafka**: Add brokers, increase partitions per topic
- **Elasticsearch**: Add data nodes, rebalance shards

### Vertical Scaling

- **API**: Increase `intra_op_num_threads` for ONNX sessions (currently 4)
- **Kafka**: Increase `num.io.threads` and `num.network.threads`
- **Elasticsearch**: Increase heap (50% of available RAM, max 31g)

### Capacity Planning

| Component | Metric | Threshold | Action |
|-----------|--------|-----------|--------|
| Kafka | Consumer lag | > 10,000 | Scale consumers or increase batch size |
| Elasticsearch | Disk usage | > 80% | Add nodes or increase retention policy |
| API | p99 latency | > 2s | Scale replicas or optimize model |
| Collector | Drops/sec | > 0 | Scale collectors or increase buffer |
| All | CPU usage | > 80% sustained | Scale horizontally |

## TLS Termination

The stack includes an nginx reverse proxy that terminates TLS in front of the API.

**Dev setup** (self-signed certs):

```bash
bash scripts/certs/generate_services_tls.sh
docker compose up nginx
```

The nginx service proxies HTTPS on port 443 to the API on port 8080.
HTTP on port 80 redirects to HTTPS.

**Production**: Replace self-signed certs with certificates from your PKI
(e.g. cert-manager, Let's Encrypt, or cloud-provider ACM). The nginx config
lives at `config/nginx/nginx.conf`.

All service ports (Kafka, Elasticsearch, Prometheus, Grafana, Redis, collector
metrics) remain bound to `127.0.0.1` by default. Expose them only through
trusted network paths.

## Elasticsearch ILM

Raw-log indices use an ILM policy to prevent unbounded disk growth.

```bash
# Apply the policy after ES is healthy:
ES_PASSWORD=<your-password> bash scripts/setup_es_ilm.sh
```

Policy phases:

| Phase | Trigger | Action |
|-------|---------|--------|
| Hot | 0d | Rollover at 30GB or 7d |
| Warm | 30d | Shrink to 1 shard, force-merge |
| Delete | 90d | Permanently remove indices |

Adjust retention in `scripts/setup_es_ilm.sh` if your compliance window
requires longer or shorter retention.

## Backup & Restore

Snapshot-based backups for Elasticsearch raw-log data:

```bash
# Create a snapshot:
ES_PASSWORD=<your-password> bash scripts/es_backup.sh

# List available snapshots:
curl -u elastic:<password> http://localhost:9200/_snapshot/logfilter-backups/_all

# Restore a snapshot:
ES_PASSWORD=<your-password> bash scripts/es_restore.sh <snapshot-name>
```

The default snapshot repository is `logfilter-backups` (filesystem-based).
For production, configure an S3 or GCS snapshot repository and schedule
backups via cron or your orchestration layer.

## Alertmanager

Prometheus alerts route to Alertmanager, which forwards to a configurable
webhook receiver.

**Default receivers**:

- `default-webhook` — all alerts (4h repeat interval)
- `critical-webhook` — critical alerts (1h repeat interval)

Set `ALERT_WEBHOOK_URL` in `.env` to point at your alerting endpoint
(PagerDuty, Slack, Opsgenie, etc.). Without this, alerts evaluate but are
not delivered.

**Alertmanager UI**: `http://localhost:9093` (dev only, localhost-bound).

## Resource Limits

All services have `deploy.resources.limits` in `docker-compose.yml`:

| Service | Memory | CPUs |
|---------|--------|------|
| logfilter-api | 8G | 4.0 |
| collector | 2G | 2.0 |
| router | 2G | 2.0 |
| archive | 2G | 2.0 |

Adjust these for your node capacity. The API limit accounts for ONNX model
memory (SecureBERT2.0 cross-encoder + NER + biencoder).
