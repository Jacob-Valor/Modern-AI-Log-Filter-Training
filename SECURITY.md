# Security Policy

## Supported Scope

This repository is a local/development deployment template for the LogFilter
pipeline. Production deployments must place public traffic behind the owning
organization's ingress, firewall, TLS, identity, and monitoring controls.

## Security Defaults

- Required secrets have no unsafe defaults.
- Scoring endpoints require `X-API-Token`.
- Admin endpoints require `X-Admin-Token`.
- API docs and OpenAPI JSON are disabled unless `LOGFILTER_ENABLE_DOCS=1`.
- Kafka, API, Elasticsearch, Kibana, Prometheus, Grafana, and collector ports
  are bound to `127.0.0.1` in the default Compose stack.
- The collector accepts only `SYSLOG_ALLOWED_CIDRS`; the default is loopback.
- Python service containers run as non-root users, drop Linux capabilities,
  prevent privilege escalation, and use read-only root filesystems.
- Runtime scaler artifacts are JSON-only; pickle/joblib scaler loading is
  rejected.

## Production Checklist

Before exposing this system outside a developer machine:

1. Terminate TLS at a trusted ingress or service mesh.
2. Restrict collector access to trusted log forwarder CIDRs.
3. Keep Elasticsearch, Prometheus, Grafana, Kafka, and Kibana private.
4. Rotate `LOGFILTER_API_TOKEN`, `LOGFILTER_ADMIN_TOKEN`, `ES_PASSWORD`, and
   `GRAFANA_ADMIN_PASSWORD` through your secret manager.
5. Run `make verify` and `make audit` in CI before deploys.
6. Build images in CI from a clean checkout and scan them with your container
   scanner.
7. Treat model artifacts as deployable code; only promote artifacts from trusted
   training jobs.

## Reporting

Do not put credentials, production logs, or sensitive customer data in issues.
Report vulnerabilities privately through your normal internal security channel.

## Vulnerability Exceptions

### CVE-2025-69872 — `diskcache` (transitive via `pysigma`)

**Status**: Documented exception. `make audit` ignores this CVE with
`--ignore-vuln CVE-2025-69872`.

**What it is**: `diskcache 5.6.3` (the latest available release) uses Python
`pickle` for serialization by default. If an attacker can write to the cache
directory, they can achieve arbitrary code execution when the application
reads from the cache. CVSS 3.1 base score: 9.8 (Critical).

**Why we accept it**:

1. `diskcache` is a transitive dependency of `pysigma` (required for Sigma
   rule loading). It is not a direct dependency.
2. `pysigma` is used in this project **only** for `sigma.collection`
   (Sigma rule parsing and matching). We do **not** import
   `sigma.data.mitre_attack` or `sigma.data.mitre_d3fend` — the two modules
   that use `diskcache`. MITRE ATT&CK data is loaded from
   `config/mitre_techniques.json` (project-owned, version-controlled), not
   from pySigma's MITRE STIX cache.
3. A defense-in-depth test (`tests/test_sigma_cache_path_unused.py`) asserts
   the vulnerable modules are never imported in the runtime code path.
4. Even if the vulnerable path were reached, our container hardening
   narrows the attack surface: services run as non-root, use read-only root
   filesystems, and drop Linux capabilities. The default `diskcache` cache
   directory is created via `tempfile.mkdtemp` (mode 0700) inside the
   container's writable layer.

**No upstream fix exists** as of audit date. Snyk and OSV both list "no
fixed version" for `diskcache` regarding CVE-2025-69872. The diskcache
maintainer's position is that the default `tempfile.mkdtemp(0o700)` directory
is the trust boundary and that the CVE is a documentation concern, not a
code defect (see grantjenks/python-diskcache issue #357).

**Re-review trigger**: When `diskcache` ships a release that addresses
CVE-2025-69872, remove the `--ignore-vuln CVE-2025-69872` flag from
`make audit` (Makefile line 88+) and re-run the gate. The defense-in-depth
test in `tests/test_sigma_cache_path_unused.py` should remain as a guard
against accidental introduction of the vulnerable code path.

**Alternatives considered**:

- Pin `pysigma` to a version that does not require `diskcache`. Rejected:
  pySigma has required diskcache for MITRE data caching since 0.10; older
  versions have other transitive issues.
- Switch to a Sigma rule engine that does not require pySigma. Rejected:
  rewrites the Sigma matching layer; out of scope for a security patch.
- Override `pysigma`'s MITRE cache with a JSON-only backend. Possible but
  requires vendoring a fork; deferred to upstream fix.

This exception is reviewed on every dependency bump.

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
