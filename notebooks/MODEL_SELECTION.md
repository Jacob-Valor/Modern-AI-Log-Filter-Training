# Tier-2 Model Selection Rationale

This document explains the design decisions behind the tier-2 transformer classifier in the Modern AI Log Filter pipeline. The tier-2 model complements the tier-1 XGBoost classifier to provide higher-fidelity decisions in the uncertainty band.

## Why Tier-2 is Necessary

### Failure Modes of Tier-1 Alone

The tier-1 classifier consumes bag-of-events count vectors (one integer per event template, 2,255 dimensions total). This representation is fast and effective for HDFS TraceBench, but has fundamental limitations:

1. **Template brittleness**. Tier-1 sees event identifiers, not text. Any novel log line that fails to match the existing template vocabulary becomes invisible or is collapsed to a generic "unmatched" bucket.

2. **Semantic blindness**. Phrases like "disk full" and "out of disk space" are distinct templates with unrelated IDs, yet semantically identical. Tier-1 cannot exploit this redundancy.

3. **Intensity misinterpretation**. A task with 10 instances of event E5 and another with 100 instances both have the same non-zero bit. The count scalar helps, but the relative significance of counts is learned from the training distribution and may not transfer.

4. **Generalization across log surfaces**. The HDFS TraceBench vocabulary is specific to Hadoop. Production syslog streams contain nginx, kernel, application, and container logs with entirely different template spaces. A bag-of-events model trained on HDFS cannot classify these without retemplating.

Tier-2 addresses these gaps by consuming reconstructed log text, not counts. A transformer pretrained on broad natural language and fine-tuned on security/CTI domains can interpret novel phrases, recognize semantic equivalence, and transfer to new log surfaces with minimal retraining.

## Model Evaluation and Comparison

We evaluated four candidate transformers for the tier-2 role. The selection criteria were: inference latency on CPU, context window size, pretraining domain relevance, and license compatibility.

| Model | Params | Context | License | Pretraining Domain | Last Updated | Why pick / Why not pick |
|-------|--------|---------|---------|-------------------|--------------|------------------------|
| [cisco-ai/SecureBERT2.0-base](https://huggingface.co/cisco-ai/SecureBERT2.0-base) | 149.7M | 8192 | Apache-2.0 | Security/CTI threat reports + codebases | 2025-10-30 | **PRIMARY**: Domain match (security logs) + ModernBERT speed. Pretraining corpus includes threat intel, CVE descriptions, and security code. **Context caveat**: the published card lists 1024, but the shipped artifact's `config.json` records `max_position_embeddings: 8192`; the 1024 figure used by the original run was an inference/training *setting* (`training_args.max_length`), not a model limit. See "Inference Length" below. |
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149.7M | 8192 | Apache-2.0 | 2T tokens English+code | 2025-01-15 | **FALLBACK**: Longest context (8K), ModernBERT architecture, no security-specific pretraining. Use when tier-2 must ingest very long windows or when SecureBERT2.0 proves unstable. [Model docs](https://huggingface.co/docs/transformers/main/en/model_doc/modernbert) |
| [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | ~184M | 512 | MIT | BookCorpus+Wikipedia | 2022-09-22 | **NOT picked**: Heavier than alternatives, short 512-token context, older architecture. Disqualified because the text-window builder can emit 256 lines; with 12 tokens per line average, we approach the context limit rapidly. |
| [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | 67.0M | 512 | Apache-2.0 | BookCorpus+Wikipedia | 2024-05-06 | **SPEED FALLBACK**: Use if latency matters more than accuracy. 40% fewer parameters than base BERT, but 512 context and no domain pretraining limits effectiveness on security logs. |

### Why SecureBERT2.0 Base Was Chosen

SecureBERT2.0-base is the primary tier-2 model for three reasons:

1. **Domain alignment**. The model is pretrained on security-relevant text: threat reports, CVE descriptions, exploit code, and CTI bulletins. Log lines like "DataXceiver error processing WRITE_BLOCK operation" are closer to this domain than to general English.

2. **Architectural currency**. Built on ModernBERT (2025), not BERT (2018). The attention implementation and positional encoding are more efficient for sequence classification.

3. **Operational fit**. 149M parameters and an 8192-token positional table balance against DeBERTa-v3 (184M, 512 ctx) and DistilBERT (67M, 512 ctx). On Kaggle T4 GPU a forward pass under 50ms is achievable, which fits the cascade latency budget.

### Inference Length — The Most Consequential Setting

`max_length` is not a detail. On HDFS TraceBench, failure windows run to ~2100+ subword tokens while the p50 is ~160. Measured on the held-out test split with the shipped 1024-token artifact:

| Bucket | Windows | Recall | F1 |
|---|---|---|---|
| Fits within 1024 tokens | 564 | **1.0000** | **1.0000** |
| Truncated at 1024 | 636 | **0.8059** | 0.8916 |

Overall recall decomposes exactly as `0.9074 × 0.8059 + 0.0926 × 1.0 = 0.824`. At a 1024 cap, **~90% of failure windows are truncated** and the discriminative exception lines — which sit at the *end* of the window, median token position 2113 — are cut away. No backbone change can recover tokens that were never fed to the model, so recall is capped near 0.82 regardless of architecture.

At `max_length = 4096`, fewer than 1% of failure windows are truncated.

**What to do:** the committed `models/tier2/` artifact is still the 1024-trained one, so it will keep scoring ~0.82 until regenerated. Retrain via `notebooks/kaggle_train_transformer_READY.ipynb`, which now trains at `--max-length 4096` and packages `tier2_config.json` (recording the length) alongside the ONNX graph. `Tier2Classifier._load_export_metadata()` reads that file, so serving follows training automatically instead of falling back to a hardcoded default — which is what let the original 1024 truncation go unnoticed.

Reordering the window builder so the exception lines come first does **not** work: it recovers only 23 of 1353 truncated signal-bearing windows (1.7%). The signal genuinely lies past the cut.

The SecureBERT2.0 paper ([arXiv:2510.00240](https://arxiv.org/abs/2510.00240)) reports F1=0.89 on the log anomaly detection benchmark, competitive with domain-specific models trained from scratch.

## Fallback Strategy

| Condition | Fallback Model | Trigger |
|-----------|---------------|---------|
| SecureBERT2.0 unavailable or export fails | ModernBERT-base | `--model-id answerdotai/ModernBERT-base` |
| Latency SLA < 20ms per inference | DistilBERT-base | `--model-id distilbert/distilbert-base-uncased --max-length 256`. **Caveat**: its 512-token window is worse than the 1024 this pipeline already suffers from — at 512, ~99.9% of failure windows are truncated. Treat this as a last resort, not a like-for-like speedup. |
| Context > 4096 tokens required | ModernBERT-base | Native 8K context handles long traces without truncation. |

The fallback selection is exposed via CLI flags in `training/train_transformer.py`. The notebook defaults to SecureBERT2.0-base.

## Data Flow

The tier-2 training pipeline reconstructs text windows from the same preprocessed files used by tier-1:

```
eventId.json (2155 templates) ─┐
                               ├─► text_dataset.build_windows ─► (text, label) ─► AutoTokenizer ─► HF Dataset ─► Trainer
normal_trace.csv counts ───────┤
failure_trace.csv counts ──────┘
```

Step-by-step:

1. **Load vocabulary**. `eventId.json` maps event IDs to template strings like "Receiving block src: /10.251.74.62:35977 dest: /10.251.74.62:50010".

2. **Expand counts**. For each task row in `normal_trace.csv` or `failure_trace.csv`, the integer count for each event is expanded into that many repetitions of the template string. A count of 5 for event E12 emits the E12 template 5 times.

3. **Cap and truncate**. Per-task emissions are capped at 256 total lines (default) with per-event repeats clipped at 16. If a task exceeds the cap, events are ranked by count and the highest-count templates are retained preferentially.

4. **Join and tokenize**. Lines are joined with newlines into a single string per task. The tokenizer maps this to input IDs and attention masks, truncating or padding to `max_length`. The notebook trains at **4096**; 1024 truncates ~90% of failure windows (see "Inference Length").

5. **HF Dataset**. The result is a HuggingFace `Dataset` with `input_ids`, `attention_mask`, and `labels` columns, suitable for `transformers.Trainer`.

The design preserves the tier-1 sampling interface (`--sample-normal`, `--sample-failure`) so both models can be trained on identical data subsets for fair comparison.

## Cascade Design

Tier-1 and tier-2 operate in a cascade. Tier-1 runs first because it is fast (ONNX runtime, milliseconds on CPU). Tier-2 runs only when tier-1 is uncertain.

```
syslog event ──► tier-1 (XGBoost / ONNX, ms)
                    │
                    ├─ p < 0.10 (confident benign)  ──► trust tier-1
                    ├─ p > 0.90 (confident failure) ──► trust tier-1
                    └─ 0.10 ≤ p ≤ 0.90 (uncertain) ──► tier-2 (SecureBERT2.0 / ONNX, ~50ms)
                                                           │
                                                           └─► override classifier_score
```

### Uncertainty Band Selection

The band 0.10-0.90 is intentionally conservative. On the HDFS TraceBench test set, tier-1 achieves precision=0.9997, recall=0.9997, F1=0.9997, ROC-AUC=1.0 (source: `models/training_metrics.json`). This is not because the model is perfect, but because the dataset is separable: failure traces emit specific exception templates that normal traces never do. The tier-1 uncertainty band is expected to be nearly empty on this dataset.

In production, with noisier logs and novel templates, the uncertainty band will populate. The 0.10-0.90 range should be tuned based on observed false-positive and false-negative rates. A narrower band (0.30-0.70) reduces tier-2 invocations but risks missing edge cases. A wider band (0.05-0.95) catches more edge cases but increases latency and compute cost.

### Integration Points

The cascade logic lives in the scoring API (`logfilter-api`). The API loads both models at startup:

- Tier-1: `models/log_classifier.onnx` (XGBoost via skl2onnx)
- Tier-2: `models/tier2/log_classifier_tier2.onnx` (SecureBERT2.0 via optimum)

Both exports use ONNX Runtime for consistent inference. The API decides tier-2 invocation based on `tier1_proba`; tier-2 output, if computed, overrides the final score.

## Limitations and Known Caveats

### Dataset-Specific Tier-1 Metrics

The recorded tier-1 metrics (`models/training_metrics.json`) are precision=0.9997, recall=0.9997, F1=0.9997, ROC-AUC=1.0 on the test split. They are an artefact of HDFS TraceBench, not a benchmark of the system: the split is derived from 226,767 normal and 29,817 failure traces, and failure traces emit exception templates (e.g. "java.io.IOException: Connection reset by peer") that never appear in normal traces. The bag-of-events representation therefore achieves near-perfect separability.

> An earlier revision of this document reported "precision=1.0, recall=1.0, F1=1.0 on 60K test samples: 50K normal + 10K failure". Those figures were wrong twice over: they did not match `models/training_metrics.json`, and the "50K + 10K" figures are the **training subsample**, not the test split. Always source metrics from the committed artifact rather than restating them here.

This is not expected to hold in production. Real-world logs contain:

- **Novel templates**: Log lines from new code paths the training data never saw.
- **Semantic variation**: Different wording for the same underlying condition.
- **Noise**: Malformed lines, truncated messages, injected fields.
- **Concept drift**: System behavior changes over time; failure modes evolve.

Tier-2's value emerges in these conditions. The transformer can interpret novel text, recognize semantic equivalence across template boundaries, and generalize across log surfaces without explicit retemplating.

### Cascade Band Conservatism

The 0.10-0.90 uncertainty band is a starting point. In production:

- If false negatives are costly (missed incidents), widen the band to capture more tier-1 uncertainty.
- If false positives are costly (alert fatigue), narrow the band to let tier-1 handle more decisions alone.
- If latency is constrained, raise the threshold for tier-2 invocation or downsample to DistilBERT.

The band is not learned; it is a configuration parameter in the API.

### Threshold Strategy Report

Before production use, generate a threshold sweep for the trained Tier-2 model instead of
changing runtime thresholds by intuition. The report evaluates precision, recall, F1,
false-positive rate, and false-negative rate across candidate probability cutoffs:

```bash
PYTHONPATH=src python scripts/evaluate_tier2_thresholds.py \
  --model-dir models/tier2 \
  --output models/tier2/tier2_threshold_report.json
```

Use the report to choose the operating point:

- **Alert-fatigue posture:** prefer a higher threshold when false positives are
  operationally expensive.
- **Missed-incident posture:** prefer a lower threshold, with an explicit recall
  target, when false negatives are more expensive.

### Current Report

The committed report (`models/tier2/tier2_threshold_report.json`) was regenerated
from **841 validation windows containing 240 failures**, comfortably above the
script's 200-row / 200-failure reliability floor
(`recommendation_reliable: true`). It is swept on the **validation** split; the
test split is never swept, because a threshold chosen by maximising over test rows
is a selection maximand rather than a generalisation estimate.

| Threshold | Precision | Recall | F1 | FP | FN | Posture |
|---|---|---|---|---|---|---|
| 0.05 | 0.8681 | 0.9875 | 0.9240 | 36 | 3 | flag almost everything |
| 0.25 | 0.8722 | 0.9667 | 0.9170 | 34 | 8 | missed-incident posture |
| **0.45** | **0.8992** | **0.9292** | 0.9139 | 25 | 17 | highest precision meeting the 0.90 recall target |
| **0.50+** | **0.9950** | **0.8375** | 0.9095 | **1** | 39 | **alert-fatigue posture** |

Two things stand out:

1. **`best_f1` still collapses onto the lowest cutoff (0.05) — the flag-everything
   point.** That is not a bug; F1 rewards recall on an 88:12 normal/failure skew, so
   the sweep will always prefer alerting more. Do not adopt `best_f1` blindly.
2. **Precision jumps from 0.8992 to 0.9950 between 0.45 and 0.50**, and false
   positives fall from 25 to 1. If alert fatigue is the binding constraint, 0.50 is
   the far better operating point — it buys back 24 false positives for 22 additional
   misses.

The flat recall of 0.8375 from 0.50 upward is the truncation ceiling from
"Inference Length" showing up here too: raising the Tier-2 threshold cannot recover
the failure windows whose signal was cut off before the model ever saw it.

> **Sample-size caveat.** An earlier committed report was generated from a 3-sample
> split containing 1 failure, in which every threshold row read 1.0. That report has
> been replaced. If you regenerate with small `--sample-normal` / `--sample-failure`
> values, check `recommendation_reliable` in the output before trusting any field —
> the script sets it false and prints a warning below its own floor.

Do not change the default `0.10`-`0.90` cascade band or routing thresholds until this report has
been reviewed against production-like logs. The Kaggle HDFS split is useful for reproducibility,
but production syslog distributions may shift both the best Tier-2 cutoff and the ideal cascade
band.

Runtime configuration lives under `scoring:` in `config/config.yaml` and can be overridden with
environment variables:

| Runtime setting | Config key | Environment variable | Default |
|---|---|---|---|
| Tier-2 lower escalation bound | `scoring.tier2.uncertainty_low` | `LOGFILTER_TIER2_UNCERTAINTY_LOW` | `0.10` |
| Tier-2 upper escalation bound | `scoring.tier2.uncertainty_high` | `LOGFILTER_TIER2_UNCERTAINTY_HIGH` | `0.90` |
| HIGH priority score threshold | `scoring.routing.high` | `LOGFILTER_SCORE_HIGH` | `0.80` |
| MEDIUM priority score threshold | `scoring.routing.medium` | `LOGFILTER_SCORE_MEDIUM` | `0.50` |
| LOW priority score threshold | `scoring.routing.low` | `LOGFILTER_SCORE_LOW` | `0.20` |

Startup validation requires `0.0 <= low < medium < high <= 1.0` for routing and
`0.0 <= uncertainty_low <= uncertainty_high <= 1.0` for Tier-2 escalation. Invalid values should
be treated as deployment configuration errors, not as values to clamp at runtime.

### Text Window Truncation

The text-window builder caps emissions at 256 lines per task and 16 repeats per event. Long-tail tasks with hundreds of distinct events or thousands of repetitions are truncated. The truncation strategy (keep highest-count events) preserves signal for most failure modes, but tasks with diffuse, low-frequency error scattering may lose information.

If production logs exhibit this pattern, increase `max_lines` (requires longer context model like ModernBERT-base) or redesign the windowing strategy to sample rather than truncate.

### Transfer to New Log Domains

The primary value proposition of tier-2 is domain transfer. HDFS TraceBench is a convenience for development; production syslog streams will differ. Transfer learning with SecureBERT2.0 requires:

1. **Template alignment**: Either reconstruct text windows from the new log domain (if structured like TraceBench) or ingest raw syslog directly.
2. **Label acquisition**: Tier-2 fine-tuning requires labeled examples. In a production SIEM, this comes from analyst verdicts on tier-1 uncertain cases.
3. **Retraining cadence**: As log sources evolve, tier-2 should be retrained on accumulated labeled data. The Kaggle notebook workflow supports this.

## Citations

- SecureBERT2.0 model: [cisco-ai/SecureBERT2.0-base](https://huggingface.co/cisco-ai/SecureBERT2.0-base)
- SecureBERT2.0 paper: [arXiv:2510.00240](https://arxiv.org/abs/2510.00240)
- ModernBERT-base: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- ModernBERT documentation: [HuggingFace Model Docs](https://huggingface.co/docs/transformers/main/en/model_doc/modernbert)
- DeBERTa-v3-base: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- DistilBERT-base-uncased: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- HDFS TraceBench paper: [TraceBench: An Open Data Set for Trace-oriented Monitoring](http://zbchen.github.io/Papers_files/cloudcom2014.pdf), CloudCom 2014
- Loghub paper: [Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics](https://arxiv.org/abs/2008.06448), ISSRE 2023

## References

- `training/text_dataset.py` — text window builder implementation
- `training/train_transformer.py` — tier-2 training script
- `notebooks/kaggle_train_transformer_READY.ipynb` — Kaggle execution notebook (Tier-2; replaces the earlier `kaggle_train_transformer.ipynb`)
