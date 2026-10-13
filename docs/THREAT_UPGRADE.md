# Threat Detection Upgrade

This upgrade adds domain-specific detectors without changing the evidence boundary:
public labelled data supports offline detection claims, not live mitigation, malware
execution, or genuine zero-day claims.

## Components

- Numeric CICFlowMeter XGBoost model exported to ONNX (`models/flow/`).
- Bounded, thread-safe 10-second destination/protocol flow window.
- Structured ModSecurity/generic WAF JSON parser with bounded, content-type-aware
  POST body extraction and credential redaction.
- BETH endpoint feature adapter plus an IsolationForest ONNX novelty model trained
  with zero suspicious/evil-labelled rows.
- Response policy hardening: authoritative DRY_RUN, execution-time allowlist checks,
  protected IP rejection, kill switch, local rate cap and duplicate suppression.

All new runtime models default to disabled. Enable only after training and manifest
validation:

```bash
make download-cic
make train-flow
make download-beth
make train-endpoint
.venv/bin/python scripts/model_manifest.py validate
```

Then set:

```dotenv
LOGFILTER_FLOW_ENABLED=true
LOGFILTER_BOT_FLOW_ENABLED=true
LOGFILTER_FLOW_WINDOW_ENABLED=true
LOGFILTER_ENDPOINT_NOVELTY_ENABLED=true
```

`LOGFILTER_BOT_FLOW_ENABLED=true` is required, not optional. The attested
general/Bot fusion only exists when the Bot head is loaded; with it off (the
default), bot traffic is scored by the general head alone, which has 0% Bot recall
at its default threshold. Without this flag the Bot column in the scenario table
below is unreachable. `docker-compose.prod.yml` sets it to `false` explicitly —
override it there too when promoting to production.

The in-process DDoS window is appropriate for single-worker validation only. A
multi-worker/HA deployment must externalize window state to Kafka Streams, Redis,
or an equivalent state store before the signal can participate in enforcement.

## Verified Results

### Flow model development split

Source: immutable, capture-specific CIC-IDS2017 DDoS, Bot and Web partitions.
Training, early-stopping validation, independent calibration and test ranges do not
overlap. For the Bot head, early stopping selects the tree count on validation,
then the promoted model is refit to that fixed tree count on training + validation;
calibration and test remain untouched. Exact canonical float32 feature duplicates
are removed from every later partition. Labels, attack names, IP addresses,
timestamps and file identity are excluded from the 33 model features.

- Independent test rows after duplicate removal: 180,161
- Recall: 96.82%
- Precision: 98.86%
- F1: 97.83%
- False positives: 168 / 165,069 benign
- AUROC: 0.99635

The aggregate hides material capture differences: the general head has 0% Bot
recall at its default threshold. A separate Bot-vs-rest head is therefore fused
through validation-only policies. These are temporal/capture-held-out results, not
cross-dataset validation.

### Attested fixed-FPR public scenario runs

Thresholds are selected only from independent calibration partitions. The values
0.1%, 1% and 5% are **validation targets**, not guarantees for shifted public-test
data. For calibrated flow runs, the final score is structurally isolated to the
attested general/Bot fusion: Sigma, text-model components, attack-tool floors,
dedup penalties and the local flow window cannot alter it.

| Validation target | Run ID | DDoS recall/FPR | Web recall/FPR | Bot recall/FPR | CSIC recall/FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.1% | `8d2e6e60cd6e523cc9fe84d2` | 100% / 0% | 87.3% / 0.2% | 94.6% / 0% | 15.6% / 0% |
| 1% | `4be17913e0fa085eb9bbb8fa` | 100% / 0.6% | 92.9% / 1.2% | 100% / 1.0% | 15.6% / 0% |
| 5% | `2e07408b75c00a9743696fe9` | 100% / 2.8% | 93.9% / 2.8% | 100% / 4.8% | 15.6% / 0% |

Reproducing any row of this table requires `LOGFILTER_BOT_FLOW_ENABLED=true`. The
Bot recall/FPR column is unreachable at the default (`false`), because the general
head alone scores 0% Bot recall at its default threshold — the numbers above are
fusion results, not general-head results. Reproduce them with a manifest-validated
artifact set and `LOGFILTER_MODELS_STRICT=1`; see [SECURITY.md](../SECURITY.md).

The evaluator excludes exact feature rows seen by either flow head during model
development. Deduplication chooses representatives without labels. Reports attest
and independently rehash native and deployed models, calibration artifacts,
development hashes, CIC/CSIC source files and the required implementation files.
Artifacts: `report/detection_coverage_fpr_*` and
`report/detection_coverage_fixed_fpr_summary.json`.

### Endpoint open-set proxy

Source: predefined BETH train/validation/test files. The IsolationForest training
set contains zero `sus` and zero `evil` rows. Host, timestamp and labels are not
features. The threshold targets 1% validation-benign FPR.

- Test recall: 86.30%
- Test precision: 97.61%
- Test F1: 91.61%
- Test FPR: 3.02%
- Test AUROC: 0.94867
- Evil-labelled recall: 93.26%

This is an unseen-family/open-set proxy. It is not proof of genuine zero-day
detection or live malware execution.

## Response Safety Boundary

The API now wires block-IP, endpoint-quarantine, and SOC-notification actions to
a signed HTTPS gateway. Armed mode requires:

- explicit permitted CIDRs and endpoint asset IDs;
- HMAC-SHA256 request signing over canonical JSON plus optional mTLS;
- execution-time target revalidation;
- a kill switch and bounded action rate;
- a receiver that durably persists `request_id` and returns `accepted` or
  `duplicate` before the API reports an action as acknowledged;
- analyst-approved staging assets.

The gateway's durable idempotency ledger prevents scoring retries from repeating
accepted actions across API workers. Local rate state is still per-process; HA
rate enforcement, provider operation reconciliation, TTL rollback, and
multi-party approval remain required before broad production enforcement.
