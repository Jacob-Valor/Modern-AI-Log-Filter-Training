# src/logfilter/models/ KNOWLEDGE BASE

## OVERVIEW

ONNX model wrappers for production inference. Each model loads from `models/` and exposes a uniform `predict()` interface.

## STRUCTURE

```
src/logfilter/models/
├── __init__.py
├── classifier.py          # Tier-1: XGBoost → ONNX (bag-of-events)
├── tier2_classifier.py    # Tier-2: SecureBERT2.0 → ONNX (text windows)
├── biencoder.py           # Sentence-transformer embeddings + FAISS index
├── ner.py                 # Named entity recognition on log text
└── cross_encoder.py       # Log↔ATT&CK relevance scoring
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Fix tier-1 inference | `classifier.py` | Loads `log_classifier.onnx` + `scaler.json` |
| Fix tier-2 inference | `tier2_classifier.py` | Loads `log_classifier_tier2.onnx` + tokenizer |
| Add embedding model | `biencoder.py` | Uses `sentence-transformers` + `faiss-cpu` |
| Add NER model | `ner.py` | Fine-tuned SecureBERT2.0-NER |
| Add cross-encoder | `cross_encoder.py` | Fine-tuned SecureBERT2.0-cross_encoder |
| Re-export to ONNX | Any model file | Follow existing `to_onnx()` pattern |

## CONVENTIONS

- Every model class must implement: `load()` (class method), `predict()` (instance method)
- ONNX export is mandatory — native format is only for retraining
- `SafeMaxAbsScaler` in `classifier.py` replaces sklearn at runtime for safety
- Model paths resolved via `Path(__file__)` chain to project root

## ANTI-PATTERNS

- **Never** load models in `__init__.py` — lazy load in service startup or first request
- **Never** use sklearn in production inference — always ONNX Runtime
- **Never** forget input validation — all `predict()` methods validate shapes/ranges
