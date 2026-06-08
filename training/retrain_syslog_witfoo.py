"""
Retrain SyslogClassifier on real WitFoo Precinct6 production logs.

Replaces the old synthetic-data-trained model with one trained on real SOC log
feature vectors, fixing the 46% accuracy from domain shift.

Usage:
    python training/retrain_syslog_witfoo.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.classifier import SafeMaxAbsScaler  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────

WITFOO_PARQUET = ROOT / "demo_data" / "witfoo" / "signals" / "signals.parquet"
FEATURES_PATH = ROOT / "models" / "syslog" / "feature_names_syslog.json"
OUTPUT_DIR = ROOT / "models" / "syslog"

# Token regex matching scorer.py _TOKEN_RE
_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")
_FEATURE_STOPWORDS = frozenset({"the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her",
                                 "was", "one", "our", "out", "has", "have", "been", "some", "same", "also", "its"})

# ── Config ─────────────────────────────────────────────────────────────────

SEED = 42
TEST_SIZE = 0.20
MAX_TRAIN_EVENTS = 50000          # cap to keep training fast (~30s)
RNG = np.random.default_rng(SEED)

# ── Feature extraction (mirrors scorer._syslog_feature_vectors) ─────────────


def prepare_features(feature_names: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Pre-process feature names into (normalized_text, tokens) tuples."""
    prepared: list[tuple[str, tuple[str, ...]]] = []
    for name in feature_names:
        lowered = name.lower()
        normalized = lowered.replace("+", " ")
        tokens = tuple(
            t for t in _TOKEN_RE.findall(normalized)
            if len(t) > 2 and t not in _FEATURE_STOPWORDS
        )
        prepared.append((normalized, tokens))
    return prepared


def extract_feature_vector(
    raw_text: str,
    prepared: list[tuple[str, tuple[str, ...]]],
    n_features: int,
) -> np.ndarray:
    """Convert a single raw log message into a 100-element binary feature vector.

    Mirrors scorer._syslog_feature_vectors exactly.
    """
    vec = np.zeros(n_features, dtype=np.float32)
    text = raw_text.lower()
    text_tokens = set(_TOKEN_RE.findall(text))

    for col, (feature_text, feature_tokens) in enumerate(prepared):
        # Exact substring match first
        if feature_text and feature_text in text:
            vec[col] = 1.0
            continue

        # Token-overlap match
        if feature_tokens and text_tokens:
            hits = 0
            for ft in feature_tokens:
                if ft in text_tokens:
                    hits += 1
                elif len(ft) >= 4 and any(ft in tt for tt in text_tokens):
                    hits += 1
            threshold = 0.50 if len(feature_tokens) <= 3 else 0.65
            if hits / len(feature_tokens) >= threshold:
                vec[col] = 1.0

    return vec


# ── Data loading ──────────────────────────────────────────────────────────


def load_witfoo_data(max_samples: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    """Load WitFoo parquet, sample stratified, extract feature vectors.

    Returns (X, y) where X is (n_samples, 100) binary feature vectors
    and y is binary labels (0=benign, 1=malicious/suspicious).
    """
    print(f"Loading WitFoo data from {WITFOO_PARQUET}...")
    df = pd.read_parquet(WITFOO_PARQUET)

    # Filter to events with messages and valid labels
    df = df[
        df["message_sanitized"].notna()
        & (df["message_sanitized"] != "")
        & df["label_binary"].notna()
    ].copy()
    print(f"  {len(df):,} events with messages and labels")

    # Map label_binary to binary: benign=0, malicious+suspicious=1
    df["y"] = np.where(df["label_binary"] == "benign", 0, 1)

    # Print label distribution
    benign = int((df["y"] == 0).sum())
    malicious = int((df["y"] == 1).sum())
    print(f"  Benign: {benign:,}  Malicious/Suspicious: {malicious:,}")

    # Balanced sampling (50/50) for maximum discrimination.
    # 85% of benign vs 6% of malicious events activate features.
    # Zero-vector → high malicious probability is the correct Bayes-optimal
    # prediction. Classifier weight (0.35) in composite score keeps FPR manageable.
    benign_df = df[df["y"] == 0]
    malicious_df = df[df["y"] == 1]

    n_benign = min(len(benign_df), max_samples // 2)
    n_malicious = min(len(malicious_df), max_samples // 2)

    benign_sample = benign_df.sample(n=n_benign, random_state=SEED)
    malicious_sample = malicious_df.sample(n=n_malicious, random_state=SEED)

    sample_df = pd.concat([benign_sample, malicious_sample], ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Training sample: {len(sample_df)} events "
          f"({int((sample_df['y']==0).sum())} benign, "
          f"{int((sample_df['y']==1).sum())} malicious)")

    # Load features
    feature_names: list[str] = json.loads(FEATURES_PATH.read_text())
    n_features = len(feature_names)
    print(f"  Features: {n_features}")

    prepared = prepare_features(feature_names)

    # Extract feature vectors
    print(f"  Extracting feature vectors for {len(sample_df)} events...")
    t0 = time.time()

    X_list: list[np.ndarray] = []
    for _, row in sample_df.iterrows():
        raw = row["message_sanitized"]
        vec = extract_feature_vector(raw, prepared, n_features)
        X_list.append(vec)

    X = np.array(X_list, dtype=np.float32)
    y = sample_df["y"].values.astype(np.int32)

    elapsed = time.time() - t0
    nonzero_frac = np.count_nonzero(X.sum(axis=1)) / len(X)
    avg_features = np.mean(np.count_nonzero(X, axis=1))
    print(f"  Feature extraction: {elapsed:.1f}s "
          f"({len(X) / elapsed:.0f} events/sec)")
    print(f"  Events with any feature activated: {nonzero_frac*100:.1f}%")
    print(f"  Avg features per event: {avg_features:.2f}")

    return X, y, feature_names


# ── Training ──────────────────────────────────────────────────────────────


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple:
    """Train XGBoost classifier with real-world feature vectors."""
    import xgboost as xgb
    from sklearn.metrics import (
        classification_report,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    # ── Split ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # ── Scale ───────────────────────────────────────────────────────────────
    max_abs = np.max(np.abs(X_train), axis=0).astype(np.float32)
    max_abs[max_abs == 0] = 1.0
    scaler = SafeMaxAbsScaler(max_abs)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ── Class distribution ──────────────────────────────────────────────────
    n_normal = int((y_train == 0).sum())
    n_failure = int((y_train == 1).sum())
    print(f"\n  Train: {n_normal} benign, {n_failure} malicious "
          f"(malicious rate: {n_failure/max(n_normal+n_failure,1)*100:.1f}%)")
    print(f"  Test:  {(y_test==0).sum()} benign, {(y_test==1).sum()} malicious")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Training XGBoost (n_estimators=200, max_depth=6, lr=0.08)...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    print(f"\n  === Evaluation on held-out test set ===")
    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    # Zero-vector baseline
    zeros = np.zeros((1, X.shape[1]), dtype=np.float32)
    zero_s = scaler.transform(zeros).astype(np.float32)
    zero_proba = model.predict_proba(zero_s)[0]
    print(f"  Zero-vector → P(benign)={zero_proba[0]:.4f} "
          f"P(malicious)={zero_proba[1]:.4f}")

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "train_normal": n_normal,
        "train_failure": n_failure,
        "test_normal": int((y_test == 0).sum()),
        "test_failure": int((y_test == 1).sum()),
        "zero_vector_p_malicious": float(zero_proba[1]),
    }

    return model, scaler, metrics


# ── Export ─────────────────────────────────────────────────────────────────


def export_model(model, scaler: SafeMaxAbsScaler, feature_names: list[str],
                 metrics: dict) -> None:
    """Export trained model to ONNX + scaler + metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Native XGBoost JSON (fallback)
    model_path = OUTPUT_DIR / "log_classifier_syslog.json"
    model.save_model(str(model_path))
    print(f"\n  Saved XGBoost model: {model_path.name}")

    # Scaler JSON
    scaler_path = OUTPUT_DIR / "scaler_syslog.json"
    scaler.to_json(scaler_path)
    print(f"  Saved scaler: {scaler_path.name}")

    # Feature names (unchanged, but save for reference)
    features_path = OUTPUT_DIR / "feature_names_syslog.json"
    features_path.write_text(json.dumps(feature_names, indent=2))
    print(f"  Saved feature names: {features_path.name}")

    # Metrics
    metrics["training_source"] = "WitFoo Precinct6"
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics: {metrics_path.name}")

    # ONNX export
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
        onnx_model =         convert_xgboost(model, initial_types=initial_type)
        onnx_path = OUTPUT_DIR / "log_classifier_syslog.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  Saved ONNX model: {onnx_path.name}")

        # Verify ONNX
        import onnxruntime as rt
        n_feat = len(feature_names)
        sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        zeros = scaler.transform(np.zeros((1, n_feat), dtype=np.float32)).astype(np.float32)
        outputs = sess.run(None, {input_name: zeros})
        proba = outputs[1][0]
        print(f"  ONNX verification (zero-vector): "
              f"P(benign)={proba[0]:.4f} P(malicious)={proba[1]:.4f}")

        # Test with a known feature active
        vec = np.zeros((1, n_feat), dtype=np.float32)
        vec[0, 0] = 1.0
        vec_s = scaler.transform(vec).astype(np.float32)
        outputs2 = sess.run(None, {input_name: vec_s})
        proba2 = outputs2[1][0]
        print(f"  ONNX verification (feature[0]=1): "
              f"P(benign)={proba2[0]:.4f} P(malicious)={proba2[1]:.4f}")

    except Exception as e:
        print(f"  ONNX export failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("SyslogClassifier Retraining with WitFoo Precinct6")
    print("=" * 60)

    # 1. Load data and extract features
    print("\n[1/4] Loading WitFoo data & extracting feature vectors...")
    X, y, feature_names = load_witfoo_data(max_samples=MAX_TRAIN_EVENTS)

    # 2. Train model
    print("\n[2/4] Training XGBoost classifier...")
    model, scaler, metrics = train_model(X, y, feature_names)

    # 3. Export
    print("\n[3/4] Exporting model artifacts...")
    export_model(model, scaler, feature_names, metrics)

    # 4. Summary
    print(f"\n[4/4] Done!")
    print(f"  Test precision: {metrics['precision']:.3f}")
    print(f"  Test recall:    {metrics['recall']:.3f}")
    print(f"  Test F1:        {metrics['f1']:.3f}")
    print(f"  Test ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"  Zero-vector → malicious: {metrics['zero_vector_p_malicious']:.3f}")


if __name__ == "__main__":
    main()
