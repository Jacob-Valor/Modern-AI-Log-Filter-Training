# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

Remediation pass over the scoring, ingestion, and documentation surface. The
defect classes fixed:

### Versioning note

`pyproject.toml` declares `version = "0.1.0"`, but the repository's git tags
(`v0.1.5`, `v0.1.16`) are ahead of it. The `0.1.0` entry below is a baseline
description of initial capability, not a claim that the tagged releases
correspond to that version string. Reconciling the package version with the tag
history is tracked as separate work.

### Fixed

- **Data loss on transient failure.** Ingestion and archive paths could drop or
  fail to persist events on transient Kafka/Elasticsearch errors instead of
  retrying and durably spooling.
- **ReDoS.** The HTTP request-feature extractors used regular expressions with
  adjacent optional quantifiers over attacker-controlled request bodies, allowing
  catastrophic backtracking on crafted input. `_SQL_KEYWORD_RE` and
  `_SQL_TAUTOLOGY_RE` in `src/logfilter/pipeline/http_features.py` both carried the
  same overlapping-quantifier construction and were fixed together using disjoint
  character classes and bounded quantifiers.
- **Cross-encoder double sigmoid.** The cross-encoder score was passed through a
  logistic transform twice, compressing legitimate scores toward 0.5 and
  flattening the ATT&CK relevance signal.
- **NaN clamp.** A non-finite component score was clamped with
  `max(0.0, min(1.0, score))`, which evaluates to `1.0` for NaN in CPython because
  `nan < 1.0` is False. A broken computation therefore became a maximum-threat
  verdict that routed to HIGH. Non-finite scores now yield `0.0` with
  `score_degraded = True`, and the event is excluded from destructive automated
  response.
- **Dead dedup penalty.** The `-0.30` `dedup_penalty` term was documented but
  unreachable: its guard required `se.cross_encoder_max > 0`, and that field is only
  set on non-duplicates while the penalty is only set on duplicates, so the
  subtraction was always `0.0`. The guard is now the cross-encoder's profile weight
  alone, which makes the term reachable exactly where Tier 3 was skipped for a
  near-duplicate.
- **Sigma floor side effect.** The Sigma match floored `se.classifier_score` in
  place during score computation, so a second call or any later inspection of the
  event saw an inflated classifier score that no model had produced. The floor is
  now a local within the scoring step and leaves the event untouched.
- **Missing model reported as benign.** When the syslog head could not serve an
  event, no classifier ran and `classifier_score` kept its `0.0` default with
  `score_degraded` left `False` — a missing artifact surfaced as a confident INFO
  verdict. Such events are now routed to the HDFS head and flagged degraded, as are
  events whose Tier-2 escalation is unavailable or fails.
- **Train/test leakage.** Flow-model training and evaluation splits shared
  duplicate feature rows and capture-specific identity, inflating held-out
  metrics. Splits are now deduplicated on canonical float32 features with
  labels, IPs, timestamps, and attack names excluded from the feature set.
- **Coverage gate vs. committed report.** The committed `coverage.xml` reported a
  stale figure that the `fail_under = 90` gate would not have accepted.
  Documentation now describes the gate rather than a hardcoded percentage.

### Changed

- Model manifest hashes are full SHA-256 digests rather than 16-character
  truncations.
- Model manifest validation is fatal when `LOGFILTER_MODELS_STRICT=1`; it remains
  a logged warning otherwise.
- The admin token path (`X-Admin-Token`, `LOGFILTER_ADMIN_TOKEN`) and the CORS
  allowance for it were removed. No admin or reload endpoint exists or is
  planned; scoring uses `X-API-Token` and metrics use `X-Metrics-Token`.
  The stale `LOGFILTER_ADMIN_TOKEN` entries that CI still exported have also been
  dropped — they set a variable nothing reads.

### Documentation

- Corrected the syslog classifier's zero-vector behaviour and disclosed
  zero-vector dominance as a known limitation.
- Corrected feature counts (syslog 110, HDFS 2255) against
  `models/model_manifest.json`.
- Documented the Sigma routing override, the `firewall_flow` / `firewall_http`
  source-profile weight and threshold overrides, and the dedup-penalty guard, so
  operators know when `LOGFILTER_SCORE_HIGH` does not apply.
- Regenerated the `models/` artifact tree from the manifest.
- `.env.example` enumerates the environment variables referenced by
  `config/config.yaml`, the Compose files, and the runtime. One gap was closed in
  this release: `LOGFILTER_ES_INDEX_REPLICAS` (read at `config/config.yaml` and
  set by the HA overlay for multi-node clusters) was previously missing.
- Added `CHANGELOG.md` and removed volatile generated metadata (generation date,
  commit hash, branch) from `AGENTS.md`.

## [0.1.0]

Baseline. Initial ingest → archive → score → LEEF pipeline: syslog collector,
Kafka transport with spool fallback and DLQ, Elasticsearch archiver, FastAPI
scoring service, QRadar router, tiered XGBoost/SecureBERT2.0 model cascade,
Docker Compose stack, Prometheus/Grafana observability, and the acknowledged
HMAC-signed QRadar delivery gateway.
