# Modern AI Log Filter Training

AI-powered log preprocessing for QRadar-style SIEM pipelines. The project ingests
raw syslog-like events, archives them, scores them with a tiered ML pipeline, and
emits enriched LEEF payloads for downstream routing.

## Architecture

```text
syslog clients
    |
    v
collector -> Kafka raw-logs -> archive -> Elasticsearch
                         |
                         v
                   scoring API
                         |
                         v
router -> QRadar / downstream SIEM
```

Core services:

- `collector`: receives UDP/TCP syslog and publishes normalized envelopes to Kafka.
- `logfilter-api`: exposes `/score` and `/score/batch` for AI scoring.
- `archive`: persists raw events to Elasticsearch before scoring decisions.
- `router`: consumes Kafka events, calls the scoring API, and forwards LEEF output.
- `training`: trains and exports the HDFS TraceBench classifier artifacts.

## Dual-Classifier Architecture

The scoring pipeline uses a cascade of two classifiers with source-type routing. Syslog, web, firewall, and Windows events route to a dedicated syslog classifier (100-feature XGBoost trained on sparse real-world patterns). All other events (HDFS TraceBench traces, generic) route to the HDFS classifier (2255-feature XGBoost). Tier-2 escalation is applied only to HDFS-classified events.

```
event ──► source-type router
              │
              ├─ syslog / web / firewall / winevent ──► syslog classifier (XGBoost, 100 feat)
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

The syslog classifier was retrained with sparse-feature data (1-7 features per sample) to handle real-world events that activate far fewer features than the dense HDFS training set. Events with no matching features receive a neutral 0.5 score. See `training/retrain_syslog_sparse.py` for the retraining script.

Before changing production thresholds, generate a Tier-2 threshold report:

```bash
PYTHONPATH=src python scripts/evaluate_tier2_thresholds.py
```

That report shows precision/recall/F1 and false-positive/false-negative tradeoffs across candidate cutoffs.

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

## Security Defaults

The local Docker stack requires explicit secrets in `.env`; unsafe default
passwords are not provided.

Required values:

```bash
LOGFILTER_ADMIN_TOKEN=replace-with-openssl-rand-base64-32
LOGFILTER_API_TOKEN=replace-with-openssl-rand-base64-32
ES_PASSWORD=replace-with-openssl-rand-base64-32
GRAFANA_ADMIN_PASSWORD=replace-with-openssl-rand-base64-32
```

The scoring API requires `X-API-Token`. The admin reload endpoint requires
`X-Admin-Token`. Kafka and the API are bound to localhost by default in
`docker-compose.yml`.

OpenAPI docs are disabled unless `LOGFILTER_ENABLE_DOCS=1`. The collector is
local-only by default and only accepts peers in `SYSLOG_ALLOWED_CIDRS`.

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

## Validation

Run the fast local engineering gate:

```bash
make verify
```

That runs:

- Ruff linting
- Unit tests
- End-to-end pipeline smoke test with heavy ML models mocked

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

Runtime artifacts are written under `models/`. The API expects the scaler as
safe JSON (`models/scaler.json`), not a pickle/joblib artifact.

## Repository Layout

```text
config/       Runtime configuration and observability config
docker/       Service Dockerfiles
scripts/      Local validation and smoke-test scripts
src/          Python package source
tests/        Unit and security-contract tests
training/     Model training and evaluation entrypoints
notebooks/    Optional training notebooks and artifact notes
```

## Production Notes

Before production, place public-facing services behind a trusted ingress with
TLS, restrict syslog ingestion to trusted source networks, and decide whether
Prometheus/Grafana/Elasticsearch should be private-only or protected by your
organization's access-control layer.
