# Kaggle Tier-2 Transformer Training Guide

**Notebook:** `notebooks/kaggle_train_transformer_READY.ipynb`  
**Base model:** `cisco-ai/SecureBERT2.0-base`  
**Dataset:** `jacobvalor/hdfs-tracebench-preprocessed-logs`  
**Known-good run:** Kaggle kernel version 8, full data, 2 epochs

---

## Quick Start (Kaggle UI)

### 1. Upload Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **New Notebook**
3. **File → Upload Notebook**
4. Select `notebooks/kaggle_train_transformer_READY.ipynb`

### 2. Attach Dataset
1. In the right panel, click **Add Data**
2. Search: `jacobvalor/hdfs-tracebench-preprocessed-logs`
3. Click **Add**

### 3. Enable GPU
1. Right panel → **Settings**
2. **Accelerator → GPU T4 x2** if available
3. **Save**

The notebook intentionally sets `CUDA_VISIBLE_DEVICES=0`, so it uses one visible GPU even when Kaggle provides two. This avoids HuggingFace `DataParallel` wrapping, which is not compatible with the current custom weighted-loss trainer.

### 4. Run Cells (in order)

| Section | Purpose | Time |
|---|---|---|
| 1 | Locate repo and set Kaggle compatibility env vars | 5 sec |
| 2 | Install ABI-compatible dependencies | 2-5 min |
| 2b | Patch cloned training script for Kaggle GPU compatibility | 5 sec |
| 3 | Attach dataset | 5 sec |
| 4 | Select base model | 5 sec |
| 5 | Sanity preview | 30 sec |
| **6** | **Sampled training** | **~15-25 min** |
| 7 | Full training (uncomment after #6 succeeds) | ~2-4 hours |
| 8 | Inspect artifacts | 5 sec |
| 9 | Package zip | 10 sec |

**Do not skip the sampled run.** It verifies data paths, tokenization, GPU training, ONNX export, and artifact packaging before you spend a multi-hour session on the full run.

---

## Kaggle CLI Workflow

`notebooks/kernel-metadata.json` is configured for the READY notebook and the HDFS dataset. To submit from the repo root:

```bash
kaggle kernels push -p notebooks --accelerator NvidiaTeslaT4
kaggle kernels status jacobvalor/logfilter-tier-2-transformer-training
```

When the run completes:

```bash
mkdir -p kaggle-output
kaggle kernels output   jacobvalor/logfilter-tier-2-transformer-training   -p kaggle-output   --force   --file-pattern 'logfilter-tier2-artifacts\.zip'
```

Extract the zip into the repo root:

```bash
python - <<'PY'
from pathlib import Path
import zipfile


def safe_extract(zip_file: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe archive member path: {member.filename}")
    zip_file.extractall(destination)


zip_path = Path('kaggle-output/logfilter-tier2-artifacts.zip')
with zipfile.ZipFile(zip_path) as zf:
    safe_extract(zf, Path('.'))
PY
```

---

## What the Sampled Run Does

- Trains on **50,000 normal** + **10,000 failure** text windows
- Runs **1 epoch**
- Uses batch size **4**
- Outputs to `models/tier2/`

If this succeeds, you should see:

- `log_classifier_tier2.onnx` (production ONNX export)
- `tier2_metrics.json` (precision, recall, F1, ROC-AUC)
- HuggingFace model/tokenizer files for further fine-tuning

## What the Full Run Does

- Trains on all HDFS TraceBench windows (~226K normal + ~30K failure)
- Runs **2 epochs**
- Uses the same output directory and overwrites sampled artifacts

To run it manually in the notebook, remove the `#` comments from the full-training cell after the sampled run succeeds.

The known-good full run produced these test metrics:

```text
Precision: 0.9922
Recall:    0.8078
F1:        0.8906
ROC-AUC:   0.9956
```

## Phase 5: Threshold Strategy

The metrics above use the model's default class decision. Before production, generate a
threshold sweep so you can choose between fewer false alerts and fewer missed failures:

```bash
PYTHONPATH=src python scripts/evaluate_tier2_thresholds.py \
  --model-dir models/tier2 \
  --output models/tier2/tier2_threshold_report.json
```

The report includes:

- `best_f1`: the best balanced threshold on the held-out split.
- `best_precision_at_min_recall`: the highest-precision threshold that still meets the recall
  target, controlled by `--min-recall`.
- Per-threshold confusion counts and false-positive/false-negative rates.

Keep the runtime cascade defaults unchanged until you review this report against logs that
represent your deployment environment.

After choosing an operating point, set runtime thresholds through `config/config.yaml` or
environment variables rather than editing code:

```bash
LOGFILTER_TIER2_UNCERTAINTY_LOW=0.10
LOGFILTER_TIER2_UNCERTAINTY_HIGH=0.90
LOGFILTER_SCORE_HIGH=0.85
LOGFILTER_SCORE_MEDIUM=0.50
LOGFILTER_SCORE_LOW=0.20
```

The service validates these at startup. Routing thresholds must be ordered
`low < medium < high`, and all threshold values must stay within `[0.0, 1.0]`.

## Optional MLM-Adapted Base Model

The READY notebook defaults to `cisco-ai/SecureBERT2.0-base`, but it will automatically prefer a local log-adapted MLM artifact when one is present. Detection order is:

1. `models/securebert2-logs-mlm/final/`
2. `models/securebert2-logs-mlm-sample/final/`
3. `cisco-ai/SecureBERT2.0-base`

Use the full `securebert2-logs-mlm/final/` path for serious retraining. The sampled path is only a quick environment check.

---

## Troubleshooting

### `numpy.dtype size changed` / pandas import failure
This is a binary compatibility mismatch between Kaggle's image and the scientific Python stack. Run the notebook's dependency cell exactly as written; it installs `numpy`, `pandas`, `scipy`, and `scikit-learn` together, then imports `numpy` and `pandas` as an ABI check.

### `GPUTooOldForTriton` on Tesla P100
ModernBERT can invoke `torch.compile`, which routes through Triton. Kaggle sometimes assigns P100 GPUs, and P100 has CUDA capability 6.0 while Triton requires >=7.0. The notebook sets `TORCHDYNAMO_DISABLE=1` and injects a small `torch.compile` no-op shim into the cloned training script before training starts.

### `DataParallel object has no attribute config`
Kaggle T4 x2 can expose two GPUs. HuggingFace may wrap the model in `DataParallel`, but this project's custom weighted-loss trainer reads `model.config`. The notebook sets `CUDA_VISIBLE_DEVICES=0` to force single-GPU training until the trainer is made DataParallel-aware.

### Out of Memory
Reduce batch size in both sampled and full training commands:

```python
'--batch-size', '2'
```

### Dataset Not Found
Make sure the dataset `jacobvalor/hdfs-tracebench-preprocessed-logs` is attached. The notebook also has a `kagglehub` fallback, but attaching the dataset in the UI or `dataset_sources` is more reliable.

### Training Too Slow
Check GPU availability:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

---

## After Training

1. Download `logfilter-tier2-artifacts.zip` from Kaggle outputs.
2. Extract it into your local repo root so it restores `models/tier2/`.
3. Verify Tier-2 ONNX inference through the production wrapper:

   ```bash
   PYTHONPATH=src python - <<'PY'
   from pathlib import Path
   from logfilter.models.tier2_classifier import Tier2Classifier

   clf = Tier2Classifier(model_dir=Path('models/tier2'))
   print('ready=', clf.is_ready())
   print(clf.predict_proba(['Failed password for root from 10.0.0.5']).tolist())
   PY
   ```

4. Run the pipeline smoke test:

   ```bash
   PYTHONPATH=src python scripts/smoke_test_pipeline.py
   ```

`training/evaluate.py` evaluates the Tier-1 XGBoost classifier only; do not use it for Tier-2 ONNX artifacts.

---

## Model Architecture Notes

- **Input:** Reconstructed log text windows up to 1024 tokens
- **Architecture:** SecureBERT2.0-base, ModernBERT-based, 149.6M parameters
- **Output:** Binary classification: normal vs failure
- **Loss:** Class-weighted cross entropy for imbalanced normal/failure windows
- **Export:** ONNX via Optimum for production inference

The Tier-2 model is invoked in the Tier-1 uncertainty band (`0.10`-`0.90`) to provide higher-fidelity decisions when the fast XGBoost model is uncertain.
