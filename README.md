# Modern AI Log Filter Training

AI-powered log preprocessing for QRadar-style SIEM pipelines. The project ingests
raw syslog-like events, archives them, scores them with a tiered ML pipeline, and
emits enriched LEEF payloads for downstream routing.

## Architecture

```text
syslog clients
    |
    v
collector (:5140) ──► Kafka raw-logs ──► archive ──► Elasticsearch
    |                                         │
    |  Prometheus :9100/metrics               v
    |                                   nginx (:443) ──► scoring API (:8080)
    |  DLQ ◄── Kafka logfilter-dlq             |
    |                                           v
    |                                    router ──► QRadar / downstream SIEM
    |
    └──► spool (on Kafka failure)

    Prometheus ──► Alertmanager (:9093) ──► webhook receiver
```

Core services:

- `collector`: receives UDP/TCP syslog, publishes normalized envelopes to Kafka, exposes Prometheus metrics on `:9100/metrics`.
- `nginx`: TLS reverse proxy in front of the API (self-signed dev certs; use your PKI in production).
- `logfilter-api`: exposes `/score`, `/score/batch`, `/health`, and `/metrics` for AI scoring.
- `archive`: persists raw events to Elasticsearch before scoring decisions; DLQ on persistent failure.
- `router`: consumes Kafka events, calls the scoring API, and forwards LEEF output.
- `training`: trains and exports the HDFS TraceBench classifier artifacts.
- `alertmanager`: receives Prometheus alerts and forwards to a configurable webhook receiver.

## Dual-Classifier Architecture

The scoring pipeline uses a cascade of two classifiers with source-type routing. Syslog, web, firewall, and Windows events route to a dedicated syslog classifier (110-feature XGBoost trained on sparse real-world patterns). All other events (HDFS TraceBench traces, generic) route to the HDFS classifier (2255-feature XGBoost). Tier-2 escalation is applied only to HDFS-classified events. Feature counts are those recorded in `models/model_manifest.json` (`onnx_input_features`), which matches the length of `models/feature_names.json` and `models/syslog/feature_names_syslog.json`.

```
event ──► source-type router
              │
              ├─ syslog / web / firewall / winevent ──► syslog classifier (XGBoost, 110 feat)
              │                                              │
              │                                              └─► final score (no tier-2)
              │
              └─ generic / other ──► HDFS classifier (XGBoost, 2255 feat)
                                         │
                                         ├─ p < 0.10 (confident benign)  ──► trust tier-1
                                         ├─ p > 0.90 (confident failure) ──► trust tier-1
                                         └─ 0.10 ≤ p ≤ 0.90 (uncertain) ──► tier-2 (SecureBERT2.0 / ONNX, ~50ms)
                                                                                │
                                                                                └─► override classifier_score
```

The syslog classifier was retrained on sparse-feature data (WitFoo Precinct6) to handle real-world events that activate far fewer features than the dense HDFS training set. See `training/retrain_syslog_witfoo.py` for the retraining script — it is the only producer of `models/syslog/*`.

> **Known limitation — zero-vector dominance.** An event that activates none of the
> 110 syslog features produces an all-zero feature vector, which collapses every
> distinct such event into a single model input. The zero-vector clamp was removed
> after this retraining (`src/logfilter/models/syslog_classifier.py`), so a zero
> vector is scored by the model rather than forced to a neutral 0.5 — and the model
> scores it **benign**: `models/syslog/training_metrics.json` records
> `zero_vector_p_malicious: 0.0009724` for that same run. Because syslog routing has
> no tier-2 fallback, the majority of production syslog traffic that matches no
> known feature is classified by this single all-zero vector and receives a low
> threat score. Treat syslog-classified scores for feature-less events as
> effectively uninformative; Sigma rules and tier-3 components remain the usable
> signal there.

Before changing production thresholds, generate a Tier-2 threshold report:

```bash
PYTHONPATH=src python scripts/evaluate_tier2_thresholds.py
```

That report shows precision/recall/F1 and false-positive/false-negative tradeoffs across candidate cutoffs. **Treat its recommendations as directional only:** the committed report (`models/tier2/tier2_threshold_report.json`) was produced from a 3-sample test split with 1 failure (`test_samples: 3`, `test_failures: 1`). On samples that small, the `best_f1` candidate degenerates to the lowest cutoff — the flag-everything point — so do **not** select operating points from it. Use it only to confirm the sweep mechanics run, and choose real thresholds from a representative labelled log sample.

Runtime thresholds are configurable without changing code. Defaults preserve the shipped cascade
behavior; invalid ranges fail startup instead of being silently clamped.

```bash
LOGFILTER_TIER2_UNCERTAINTY_LOW=0.10
LOGFILTER_TIER2_UNCERTAINTY_HIGH=0.90
LOGFILTER_SCORE_HIGH=0.80
LOGFILTER_SCORE_MEDIUM=0.50
LOGFILTER_SCORE_LOW=0.20
```

Only adjust these after reviewing a threshold report from representative logs. Routing thresholds
must satisfy `0.0 <= low < medium < high <= 1.0`; Tier-2 uncertainty thresholds must satisfy
`0.0 <= uncertainty_low <= uncertainty_high <= 1.0`.

## Scoring Formula

The final threat score is a weighted blend of all model components:

```
ai_threat_score = (0.35 × classifier_score)
                + (0.25 × entity_boost)
                + (0.40 × cross_encoder_max)
                + (0.15 × novelty_score)
                - dedup_penalty
```

| Component | Weight | Source | Purpose |
|-----------|--------|--------|---------|
| classifier_score | 0.35 | Tier 1 XGBoost | Fast pattern matching |
| entity_boost | 0.25 | Tier 3 NER | IOC/malware/CVE detection |
| cross_encoder_max | 0.40 | Tier 3 CrossEncoder | ATT&CK technique relevance |
| novelty_score | 0.15 | Tier 3 Novelty | Zero-day/rare event detection |
| dedup_penalty | -0.30 | Tier 2 BiEncoder | Near-duplicate suppression, applied where the cross-encoder carries weight |

Routing thresholds: HIGH (≥0.80), MEDIUM (≥0.50), LOW (≥0.20), INFO (<0.20).

### Scoring overrides and guards

The formula and thresholds above are the **default** source profile. Five behaviours
change the effective result, and all five matter when you tune `LOGFILTER_SCORE_HIGH`:

- **Sigma override.** When a Sigma rule matches, the scorer floors the classifier
  term at `0.90` *before* computing the score, and the routing step forces `HIGH`
  regardless of the final score. No threshold can pull a Sigma-matched event below
  HIGH. The floor is computed in a local and does not mutate the event's reported
  `classifier_score`. Set `LOGFILTER_SIGMA_ENABLED=false` to disable this behaviour.
- **Source-profile overrides.** `config/config.yaml` defines `source_profiles` for
  `firewall_flow` and `firewall_http`. These **replace the base weights entirely**
  (`classifier: 1.0`, `entity_boost: 0.0`, `cross_encoder: 0.0`, `novelty: 0.0`)
  and carry their own routing thresholds, so `LOGFILTER_SCORE_*` does not apply to
  them. Tune `LOGFILTER_FLOW_SCORE_*` and `LOGFILTER_HTTP_SCORE_*` instead.
- **dedup_penalty scope.** The penalty compensates for Tier 3 being skipped on a
  near-duplicate, so it applies exactly where the cross-encoder carries weight —
  i.e. under the default profile. It is `0.0` under `firewall_flow` and
  `firewall_http`, where Tier 3 was never part of the score to begin with.
- **Non-finite scores degrade; they do not alert.** If any model component yields
  NaN or Inf, the composite score becomes `0.0` and the event is flagged
  `score_degraded` rather than being clamped upward. (`max(0.0, min(1.0, nan))`
  evaluates to `1.0` in CPython, so a naive clamp turns a broken computation into a
  maximum-threat verdict.) A degraded event therefore routes as INFO, not HIGH.
- **A missing or broken model marks the event degraded.** When the syslog head
  cannot serve an event, it is scored by the HDFS head *and* flagged
  `score_degraded`; when Tier-2 is unavailable or its inference fails, the escalated
  events keep their Tier-1 scores and are likewise flagged. In both cases the score
  you see is a best-effort fallback rather than a model verdict.

**`score_degraded` is what blocks automated response.** In armed response mode a
degraded (or duplicate) event is never the subject of a destructive action such as
an IP block, regardless of how high its score is. Watch it in the LEEF `degraded`
field and in the API's `score_degraded` response attribute.

`firewall` events select a profile by content: those carrying
`http_lexical_features` route to `firewall_http`, those carrying `dpt`, `duration`,
or `fwdPackets` route to `firewall_flow`, and everything else falls back to the
default profile. When the calibrated flow-isolation policy is active, a
`firewall_flow` event's score is the bounded classifier probability alone — no
entity, cross-encoder, novelty, Sigma, or dedup term applies.

### LEEF Output Fields

The enricher produces LEEF 2.0 payloads with these AI custom properties:

| Field | Description |
|-------|-------------|
| `ai_threat_score` | 0.0–1.0 composite threat score |
| `ai_priority` | HIGH / MEDIUM / LOW / INFO |
| `ai_mitre_technique` | Top ATT&CK technique ID (e.g., T1021.002) |
| `ai_entities` | Comma-separated IOCs/malware/CVEs extracted by NER |
| `ai_ner_confidence` | NER model confidence (0.0–1.0) |
| `ai_dedup_flag` | true/false (was this a near-duplicate event?) |
| `ai_sigma_match` | true/false (matched a Sigma detection rule?) |
| `ai_sigma_rules` | Comma-separated matched rule IDs |
| `ai_source_type` | Event source type (syslog, web, firewall, etc.) |
| `ai_scoring_latency_ms` | Total scoring latency in milliseconds |
| `ai_novelty_score` | 0.0–1.0 novelty score (0=normal, 1=highly novel) |
| `degraded` | 1 if scoring degraded, 0 otherwise |
| `raw_log_ref` | Elasticsearch document ID for chain-of-custody |

## Novelty Detection

Novelty detection identifies rare/unusual events that don't match known patterns, enabling detection of zero-day attacks and emerging threats. It reuses the existing BiEncoder embeddings (768-dim) and maintains a rolling baseline of "normal" events.

**How it works:**
1. Collects embeddings into a rolling 10,000-event window
2. Computes centroid of "normal" embeddings
3. Measures cosine distance from centroid for each new event
4. High distance = novel event (score 0.0–1.0)

**Enable novelty detection:**

```bash
LOGFILTER_NOVELTY_ENABLED=true make up
```

**Configuration:**

Only the on/off switch is environment-driven; the remaining knobs are literals in
`config/config.yaml` (`monitoring.novelty`) and require a config edit plus a
restart:

```dotenv
LOGFILTER_NOVELTY_ENABLED=false
```

```yaml
# config/config.yaml — monitoring.novelty
window_size: 10000        # baseline window size (events)
min_baseline: 100        # events required before scoring activates
warmup_events: 500       # events collected before any score is emitted
distance_scale: 2.0      # cosine distance → novelty score scaling
```

**Tuning for false positives:**
- Start with weight=0.0, gradually increase to 0.15 (`scoring.weights.novelty`)
- Monitor `ai_novelty_score` in LEEF output (target: < 0.1 for normal events)
- If false positives are high, decrease `distance_scale` or increase `warmup_events`
  in `config/config.yaml`

## Security Defaults

The local Docker stack requires explicit secrets in `.env`; unsafe default
passwords are not provided.

Required values:

```bash
LOGFILTER_API_TOKEN=replace-with-openssl-rand-base64-32
LOGFILTER_METRICS_TOKEN=replace-with-openssl-rand-base64-32
ES_PASSWORD=replace-with-openssl-rand-base64-32
GRAFANA_ADMIN_PASSWORD=replace-with-openssl-rand-base64-32
QRADAR_GATEWAY_HMAC_SECRET=replace-with-openssl-rand-base64-32
LOGFILTER_RESPONSE_GATEWAY_HMAC_SECRET=replace-with-openssl-rand-base64-32
```

The scoring API requires `X-API-Token`. The metrics endpoint requires
`X-Metrics-Token` (or `Authorization: Bearer <token>`). All token checks fail
closed: the endpoint returns 403 when the token is not configured and 401 when
the token is wrong.
Kafka and the API are bound to localhost by default in `docker-compose.yml`.

OpenAPI docs are disabled unless `LOGFILTER_ENABLE_DOCS=1`. The collector is
local-only by default and only accepts peers in `SYSLOG_ALLOWED_CIDRS`.

`CORS_ALLOW_ORIGINS` must be set to your actual frontend origin(s) in
production. The API refuses to start with localhost CORS origins when
`LOGFILTER_ENABLE_DOCS` is off. Leave empty for headless deployments (no
browser clients).

### Acknowledged security gateways

Production QRadar delivery uses `QRADAR_PROTOCOL=https_ack`. Each LEEF payload is
signed with HMAC-SHA256 and carries its stable `raw_log_ref` as both
`request_id` and `Idempotency-Key`. The receiver must durably persist that ID
before returning `{"request_id":"...","status":"accepted"}`; a replay returns
`status=duplicate`. Any other response prevents the Kafka offset from being
committed. TCP/TLS/UDP syslog remains available but retains at-least-once
duplicate risk because it has no durable receiver acknowledgement.

Automated response uses the same signed acknowledgement contract at
`LOGFILTER_RESPONSE_GATEWAY_URL`. `LOGFILTER_RESPONSE_MODE=dry_run` only returns
recommendations. Armed mode requires a 32-byte secret, HTTPS URL, explicit CIDR
and host allowlists, and `LOGFILTER_RESPONSE_KILL_SWITCH=false`. Optional private
CA and mTLS paths are available in `.env.example`. Direct `/score` clients that
need retry idempotency must reuse a 64-character SHA-256 `raw_log_ref`; Kafka
router calls receive this stable reference from topic/partition/offset.

## Local Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
make dev-install
cp .env.example .env
```

Edit `.env` and replace every placeholder secret before starting Docker.

```bash
make up
make smoke-test
```

### High-availability Compose overlay

For a local high-availability topology, layer `docker-compose.prod.yml` over the
base stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The overlay expands Kafka and Elasticsearch to three nodes, raises Kafka topic
replication factors to `3`, and points API/archive/router clients at all Kafka
brokers. It is still Compose-based scaffolding: Kafka remains PLAINTEXT inside
the private Compose network, so use broker-side SASL/TLS or a managed Kafka
service before crossing trust boundaries.

## Validation

Run the local engineering gate before opening a change:

```bash
make verify
```

That runs:

- Ruff linting (`lint`)
- The test suite under the coverage gate from `pyproject.toml`
  (`test-coverage`; `tool.coverage.report.fail_under = 90`, so the run fails if
  measured branch coverage falls below the gate)
- End-to-end pipeline smoke test (`smoke-pipeline` →
  `scripts/smoke_test_pipeline.py`), which runs the normalizer, ONNX tier-1
  classifier, LEEF enricher, and router logic against a mocked syslog sender —
  then scores a real event through the **full tiered pipeline using the installed
  BiEncoder, NER, and CrossEncoder artifacts** (`models/biencoder/final`,
  `models/ner/final`, `models/cross_encoder/final`).

Because the smoke test loads real transformers, it needs the `models/` artifacts
installed and takes roughly 10 seconds; the final `scorer_real` section is what
cannot run without them.

Run a dependency audit separately:

```bash
make audit
```

See [SECURITY.md](SECURITY.md) for deployment hardening and reporting guidance.

## Training

Train the classifier from HDFS TraceBench data:

```bash
make train
make evaluate
```

Train tier-2 transformer models (NER + CrossEncoder) on Kaggle:

```bash
# Upload and run notebooks (full training, ~2h on T4 GPU)
kaggle kernels push -p notebooks -t kaggle_train_ner
kaggle kernels push -p notebooks -t kaggle_train_cross_encoder
```

After training, install artifacts and validate the model manifest:

```bash
make model-manifest-generate   # recompute SHA-256 hashes
make model-manifest-validate   # verify hashes match installed artifacts
```

Runtime artifacts are written under `models/`. The API expects the scaler as
safe JSON (`models/scaler.json`), not a pickle/joblib artifact.

## Monitoring & Observability

### Prometheus Metrics

| Service | Endpoint | Metrics |
|---------|----------|---------|
| API | `:8080/metrics` | `logfilter_events_total`, `logfilter_scoring_latency_ms`, `logfilter_threat_score`, `logfilter_model_loaded`, `logfilter_drift_psi` |
| Collector | `:9100/metrics` | `logfilter_collector_received_total`, `logfilter_collector_published_total`, `logfilter_collector_dropped_total`, `logfilter_collector_spool_depth` |

Novelty detection stats are available via the `NoveltyDetector.get_stats()` method:
- `baseline_size`: Current number of embeddings in the rolling window
- `event_count`: Total events processed
- `avg_novelty_score`: Average novelty score across all events

Collector env var: `SYSLOG_METRICS_PORT` (default: `9100`).

### Alert Rules

Prometheus alerts are in `config/prometheus/alerts.yml`:

- **LogfilterDeadManSwitch** — monitoring pipeline liveness probe
- **LogfilterAPIDown** — API unreachable >1 min
- **LogfilterNoEventsScored** — no scoring activity for 10 min
- **LogfilterCollectorDown** — collector receives no events for 5 min
- **LogfilterCollectorPublishLag** — events arriving but none reach Kafka
- **LogfilterCollectorHighDropRate** — >30% drop rate for 10 min
- **LogfilterCollectorSpoolGrowing** — spool non-empty for 10 min
- **LogfilterDLQDepthHigh** — DLQ has >100 unprocessed messages
- **LogfilterHighLatencyP99/P95** — SLO breach on scoring latency
- **LogfilterLatencyBudgetBurnFast/Slow** — error-budget burn-rate alerts

### Grafana

Dashboards are provisioned from `config/grafana/dashboards/`. Access at
`http://localhost:3000` (credentials from `.env`).

## Model Artifacts

The manifest tracks nine model groups. Regenerate this listing from
`models/model_manifest.json` rather than trusting a copy:

```bash
PYTHONPATH=src python scripts/model_manifest.py validate
```

```text
models/
├── model_manifest.json          # SHA-256 hashes + sizes + feature counts, all 9 groups
├── log_classifier.onnx          # Tier-1 HDFS XGBoost classifier (2255 features)
├── scaler.json                  # SafeMaxAbsScaler (JSON, not pickle)
├── feature_names.json           # 2255 HDFS feature names
├── syslog/                      # Syslog-routed Tier-1 (110 features)
│   ├── log_classifier_syslog.onnx
│   ├── scaler_syslog.json
│   ├── feature_names_syslog.json
│   ├── log_classifier_syslog.json
│   └── training_metrics.json    # includes zero_vector_p_malicious
├── ner/final/                   # SecureBERT 2.0 NER
│   ├── model.safetensors        # ~571 MB
│   ├── model.onnx               # ~571 MB
│   ├── ner_label_map.json
│   └── tokenizer.json / special_tokens_map.json
├── biencoder/final/             # SecureBERT 2.0 BiEncoder (dedup + ATT&CK retrieval)
│   ├── model.safetensors        # ~568 MB
│   └── tokenizer.json / sentence_bert_config.json
├── cross_encoder/final/         # SecureBERT 2.0 CrossEncoder (ATT&CK relevance)
│   ├── model.safetensors        # ~571 MB
│   └── tokenizer.json / sentence_bert_config.json
├── tier2/                       # Tier-2 SecureBERT 2.0 classifier (uncertain band)
│   ├── log_classifier_tier2.onnx
│   ├── model.safetensors
│   └── tier2_label_map.json / tier2_threshold_report.json
├── flow/                        # CICFlowMeter flow head + bot head (33 features each)
│   ├── flow_classifier.onnx     # general flow head
│   ├── bot_classifier.onnx      # Bot-vs-rest head
│   ├── fusion_calibration.json  # attested general/bot fusion policy
│   ├── bot_calibration.json / calibration.json
│   ├── bot_scaler.json / scaler.json
│   ├── bot_feature_names.json / feature_names.json
│   └── development_feature_hashes.txt / bot_development_feature_hashes.txt
├── http/                        # WAF/HTTP lexical calibration
│   └── lexical_calibration.json
└── endpoint/                    # BETH endpoint IsolationForest novelty (136 features)
    ├── endpoint_novelty.onnx
    └── novelty_config.json / novelty_metrics.json
```

### Manifest integrity validation

The scorer validates `models/model_manifest.json` at startup: for every tracked
artifact it recomputes the SHA-256 digest, compares it against the stored value,
and reports each mismatch. A mismatch is **non-fatal by default** — it is logged as
a warning so corrupted or stale artifacts are detectable without taking the service
down. Set `LOGFILTER_MODELS_STRICT=1` to make a mismatch fatal instead: model
loading then raises and the service refuses to start.

Run the standalone check before a deploy:

```bash
make model-manifest-validate
```

## Throughput Benchmark

Run a standalone benchmark without Docker:

```bash
make benchmark-standalone
```

This measures p50/p95/p99 latency and events/sec against local model artifacts.
Override sample count and concurrency:

```bash
make benchmark-standalone BENCHMARK_ARGS="--samples 10000 --concurrency 8"
```

## Repository Layout

```text
config/       Runtime configuration and observability config
docker/       Per-service Dockerfiles (api, archive, collector, router, nginx) — single-stage
scripts/      Local validation and smoke-test scripts
src/          Python package source
tests/        Unit, security-contract, and integration tests
training/     Model training and evaluation entrypoints
notebooks/    Optional training notebooks and artifact notes
docs/         Operations and threat-upgrade notes
models/       Trained artifacts, indexed by models/model_manifest.json
uv.lock       Resolved dependency graph — all four Python images build from this via `uv export --frozen`
```

Python image dependencies are resolved from `uv.lock` at build time (`uv export
--frozen --only-group <api|archive|collector|router>`); no requirements file is
consumed by a Dockerfile, and `.dockerignore` keeps everything except `src/`,
`config/`, `pyproject.toml`, `uv.lock`, and `scripts/download_hf_models.py` out of
the build context.

`HDFS_v3_TraceBench/` holds the preprocessed dataset and is gitignored — it
exists only on machines where the dataset was downloaded (~5 GB).

## Production Notes

Before production, place public-facing services behind a trusted ingress with
TLS, restrict syslog ingestion to trusted source networks, and decide whether
Prometheus/Grafana/Elasticsearch should be private-only or protected by your
organization's access-control layer.

### TLS Termination

An nginx reverse proxy terminates TLS on port 443. Generate dev certs:

```bash
bash scripts/certs/generate_services_tls.sh
docker compose up nginx
```

For production, replace self-signed certs with your PKI (cert-manager, Let's
Encrypt, cloud ACM). See `config/nginx/nginx.conf`.

### Distributed Rate Limiting

The API uses Redis-backed rate limiting when `REDIS_URL` is set (configured in
`docker-compose.yml` by default). Without Redis, rate limiting falls back to
per-process counters — effective only behind a single replica.

### Elasticsearch ILM

Raw-log indices roll over at 30 GB or 7 days, move to warm at 30 days, and
delete at 90 days. Apply the policy after ES is healthy:

```bash
ES_PASSWORD=<your-password> bash scripts/setup_es_ilm.sh
```

### Backup & Restore

```bash
# Create snapshot:
ES_PASSWORD=<your-password> bash scripts/es_backup.sh

# List snapshots:
ES_PASSWORD=<your-password> bash scripts/es_backup.sh list

# Restore:
ES_PASSWORD=<your-password> bash scripts/es_restore.sh <snapshot-name>
```

### Alertmanager

Prometheus alerts route to Alertmanager (`:9093`) which forwards to a webhook
receiver. Set `ALERT_WEBHOOK_URL` in `.env` to deliver alerts. Without it,
alerts evaluate but are not delivered.
