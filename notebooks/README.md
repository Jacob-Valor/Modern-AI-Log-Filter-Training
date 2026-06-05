# Notebooks

Kaggle and exploratory notebooks live here.

Keep reusable training logic in `training/`. Notebooks should call the scripts or import their helpers, then export generated artifacts into `models/`.

## Inventory

| Notebook | Trains | Output dir | Consumed by |
|---|---|---|---|
| `kaggle_train_classifier.ipynb` | Tier-1 XGBoost classifier (bag-of-events) | `models/` | `src/logfilter/models/classifier.py` |
| `kaggle_train_transformer.ipynb` | Tier-2 SecureBERT2.0-base classifier (text windows) | `models/tier2/` | `src/logfilter/models/tier2_classifier.py` |
| `kaggle_pretrain_mlm.ipynb` | Continued MLM pre-training of SecureBERT2.0-base on log corpus | `models/securebert2-logs-mlm/final/` | All downstream encoders below (optional but recommended) |
| `kaggle_train_ner.ipynb` | SecureBERT2.0-NER fine-tuned on CyNER (MIT, arXiv:2204.05754; pinned commit `37aff53b`); 5 entity types match wrapper exactly; per-class seqeval F1 reported; domain-transfer yield check required before promotion | `models/ner/final/` | `src/logfilter/models/ner.py` |
| `kaggle_train_cross_encoder.ipynb` | SecureBERT2.0-cross_encoder fine-tuned on log↔ATT&CK pairs | `models/cross_encoder/final/` | `src/logfilter/models/cross_encoder.py` |
| `kaggle_train_biencoder.ipynb` | SecureBERT2.0-biencoder on (log window, ATT&CK technique) positive pairs via MultipleNegativesRankingLoss; optional SimCSE cell for unsupervised dedup sharpening; optional BYO pairs loader | `models/biencoder/final/` (sentence-transformers directory; no ONNX export) | `src/logfilter/models/biencoder.py` (one-line `config.yaml` `model_id` swap) |

## Recommended training order

The Tier-1 and Tier-2 notebooks are independent and can run any time. The downstream notebooks form a small dependency graph:

```text
kaggle_pretrain_mlm.ipynb  ──►  models/securebert2-logs-mlm/final/
                                 │
                ┌────────────────┼──────────────────────────────┐
                ▼                ▼                ▼              ▼
   kaggle_train_transformer  kaggle_train_ner  kaggle_train_cross_encoder  kaggle_train_biencoder
   (re-train Tier-2 on        (set MODEL_ID    (set MODEL_ID to log-adapted  (set MODEL_ID to
    log-adapted base for       to log-adapted   base in the model-load cell)   log-adapted base)
    higher Tier-2 quality)     base)
```

The MLM step is optional but produces a domain-adapted encoder that lifts every downstream head. Each downstream notebook documents how to switch its `MODEL_ID` constant from the published Cisco variant to the local MLM-adapted directory.

`kaggle_train_ner`, `kaggle_train_cross_encoder`, and `kaggle_train_biencoder` are all Tier-3 fine-tunes with no ordering dependency between them; run them in any order after the MLM base is ready.

## Per-notebook flow

Each Kaggle notebook follows the same nine-section template:

1. Locate the repo (works whether running locally or under `/kaggle/working`).
2. Install dependencies (`%pip install -q ...`).
3. Attach the HDFS TraceBench preprocessed dataset (symlink from `/kaggle/input` if needed).
4. Sanity-preview the inputs (text windows, regex labels, or pair samples — depending on the head).
5. Run a **sampled** training job first to verify environment, GPU, and artifact paths.
6. Run the **full** training job (commented by default — uncomment after the sampled run succeeds).
7. Inspect artifacts and any per-task metrics file.
8. Package the artifacts as a zip under `/kaggle/working/`.
9. Output description + how to consume in the repo.

Run sampled first, full second. Download the generated artifact zip from `/kaggle/working/` once the run is green.

## Generated artifacts (gitignored)

```text
# Tier-1
models/log_classifier.onnx
models/log_classifier.json
models/scaler.json
models/feature_names.json
models/training_metrics.json

# Tier-2
models/tier2/config.json
models/tier2/model.safetensors
models/tier2/tokenizer.json
models/tier2/log_classifier_tier2.onnx
models/tier2/tier2_metrics.json
models/tier2/tier2_label_map.json

# MLM-adapted base
models/securebert2-logs-mlm/final/{config.json,model.safetensors,tokenizer.json,...}

# NER
models/ner/final/{config.json,model.safetensors,model.onnx,ner_metrics.json,ner_label_map.json,tokenizer.json}

# CrossEncoder
models/cross_encoder/final/{config.json,model.safetensors,tokenizer.json,cross_encoder_metrics.json}

# BiEncoder (sentence-transformers directory — no ONNX export; runtime loads natively)
models/biencoder/final/{modules.json,config_sentence_transformers.json,sentence_bert_config.json,model.safetensors,tokenizer.json,biencoder_metrics.json}
```

The ONNX exports are production classifiers. The HuggingFace directories remain useful for further fine-tuning and reproducible export. Keep artifacts under their respective `models/<head>/final/` subdirectories so the existing wrapper code in `src/logfilter/models/` can find them with one config-file change.

The BiEncoder artifacts are a sentence-transformers directory, not ONNX. The runtime wrapper loads them natively via `sentence-transformers`; no export step is needed.

## Data sources

| Dataset | Labels | Size | Notes |
|---|---|---|---|
| HDFS TraceBench | Line-level Anomaly / NotAnomaly | ~2.2 GB | Preprocessed; `HDFS_v3_TraceBench/preprocessed/` |
| MITRE ATT&CK techniques | N/A (retrieval corpus) | ~50 techniques | `config/mitre_techniques.json` |
| CyNER | Token-level BIO (5 entity types) | ~107K tokens | MIT licence; arXiv:2204.05754; pinned commit `37aff53b` |

See [MODEL_SELECTION.md](MODEL_SELECTION.md) for model selection rationale, cascade design, and limitations.
