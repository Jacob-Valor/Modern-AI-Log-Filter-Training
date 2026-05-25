# training/ KNOWLEDGE BASE

## OVERVIEW

Training scripts for the two-tier classifier and auxiliary models. All scripts output to `models/` in ONNX + native formats.

## STRUCTURE

```
training/
├── train.py               # Tier-1: XGBoost on bag-of-events count vectors
├── train_transformer.py   # Tier-2: SecureBERT2.0 fine-tuning on text windows
├── data_loader.py         # HDFS TraceBench CSV loader + train/val/test split
├── text_dataset.py        # Reconstructs text windows from count vectors
├── evaluate.py            # Evaluate saved models on test set
└── requirements.txt       # Training-specific deps (overlaps with main)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change XGBoost params | `train.py` | `parse_args()` has all hyperparameters |
| Change transformer params | `train_transformer.py` | `TrainerConfig` dataclass |
| Fix data loading | `data_loader.py` | Loads `normal_trace.csv` + `failure_trace.csv` |
| Fix text window building | `text_dataset.py` | Reconstructs from `eventId.json` vocabulary |
| Evaluate model | `evaluate.py` | Runs on test split, outputs metrics JSON |
| Add new training script | Create file here | Follow `train.py` CLI pattern |

## CONVENTIONS

- All scripts accept `--sample-normal` and `--sample-failure` for fast dev iterations
- ONNX export is the final step of every training script
- `SafeMaxAbsScaler.to_json()` is used instead of sklearn pickle for scaler persistence
- Training uses standard `logging`; no `structlog` here

## ANTI-PATTERNS

- **Never** commit trained model binaries to git — add `models/*.onnx` to `.gitignore`
- **Never** use raw sklearn scaler pickle — always export to JSON via `SafeMaxAbsScaler`
- **Never** skip evaluation step — every script runs eval and writes `metrics.json`
- **Never** train on full data without sampled run first — sampled is the sanity check

## NOTES

- HDFS data lives in `HDFS_v3_TraceBench/preprocessed/`
- Kaggle notebooks in `notebooks/` call these scripts via subprocess
- Text windows are deterministic: event-id order, capped at 256 lines per task
