"""Retrain syslog classifier with sparse-feature data for real-world generalization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logfilter.models.classifier import SafeMaxAbsScaler  # noqa: E402

FEATURES_PATH = ROOT / "models" / "syslog" / "feature_names_syslog.json"
OUTPUT_DIR = ROOT / "models" / "syslog"


def _load_features() -> list[str]:
    return json.loads(FEATURES_PATH.read_text())


def _make_sparse_normal(
    n_features: int, rng: np.random.Generator, feature_names: list[str]
) -> np.ndarray:
    """Generate a sparse normal-event feature vector (1-4 benign features)."""
    vec = np.zeros(len(feature_names), dtype=np.float32)
    benign_indices = [
        feature_names.index(f)
        for f in [
            "systemd+Started", "dhclient+lease", "get+request+returned",
            "post+request+returned", "cron+CROND+CMD", "sshd+accepted password",
            "sshd+accepted publickey", "sshd+pam_unix session opened",
            "eventid+4624+successful logon", "iptables+ACCEPT+ESTABLISHED",
            "ufw+ALLOW+OUT", "getfileinfo+success",
        ]
        if f in feature_names
    ]
    chosen = rng.choice(
        benign_indices, size=min(n_features, len(benign_indices)), replace=False
    )
    vec[chosen] = 1.0
    return vec


def _make_sparse_failure(
    n_features: int, rng: np.random.Generator, feature_names: list[str]
) -> np.ndarray:
    """Generate a sparse failure-event feature vector (1-4 attack features)."""
    vec = np.zeros(len(feature_names), dtype=np.float32)
    attack_indices = [
        feature_names.index(f)
        for f in [
            "sshd+failed password", "sql injection+select union",
            "xss+script+query string", "snort+eternalblue+MS17-010",
            "pam_unix+authentication failure", "ufw+BLOCK+IN",
            "iptables+DROP+INPUT", "iptables+DROP+FORWARD",
            "sqlmap+scanner", "snort+brute force",
            "eventid+4625+failed logon", "eventid+1102+audit log cleared",
            "sshd+failed password invalid user",
            "sshd+disconnecting authenticating too many",
            "directory traversal+passwd", "command injection+exec",
            "429+rate limit exceeded", "login failed+401 unauthorized",
        ]
        if f in feature_names
    ]
    chosen = rng.choice(
        attack_indices, size=min(n_features, len(attack_indices)), replace=False
    )
    vec[chosen] = 1.0
    return vec


def _make_dense_normal(
    rng: np.random.Generator, feature_names: list[str]
) -> np.ndarray:
    """Generate a dense normal-event vector (3-5 features)."""
    return _make_sparse_normal(rng.integers(3, 6), rng, feature_names)


def _make_dense_failure(
    rng: np.random.Generator, feature_names: list[str]
) -> np.ndarray:
    """Generate a dense failure-event vector (3-7 features)."""
    return _make_sparse_failure(rng.integers(3, 8), rng, feature_names)


def generate_dataset(
    n_normal: int = 12000, n_failure: int = 6000
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data with both sparse and dense feature vectors."""
    rng = np.random.default_rng(42)
    feature_names = _load_features()

    X, y = [], []

    for _ in range(int(n_normal * 0.40)):
        X.append(_make_sparse_normal(rng.integers(1, 4), rng, feature_names))
        y.append(0)

    for _ in range(int(n_normal * 0.30)):
        X.append(_make_dense_normal(rng, feature_names))
        y.append(0)

    for _ in range(int(n_normal * 0.30)):
        vec = _make_sparse_normal(1, rng, feature_names)
        attack_idx = rng.choice([
            feature_names.index(f)
            for f in [
                "sshd+failed password", "sql injection+select union",
                "ufw+BLOCK+IN", "pam_unix+authentication failure",
            ]
            if f in feature_names
        ])
        n_extra = rng.integers(1, 3)
        extra_attack = rng.choice([
            feature_names.index(f)
            for f in [
                "sshd+failed password", "sql injection+select union",
                "xss+script+query string", "ufw+BLOCK+IN",
            ]
            if f in feature_names
        ], size=min(n_extra, 2), replace=False)
        vec[attack_idx] = 1.0
        vec[extra_attack] = 1.0
        X.append(vec)
        y.append(0)

    for _ in range(int(n_failure * 0.50)):
        X.append(_make_sparse_failure(rng.integers(1, 4), rng, feature_names))
        y.append(1)

    for _ in range(int(n_failure * 0.50)):
        X.append(_make_dense_failure(rng, feature_names))
        y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train(X: np.ndarray, y: np.ndarray) -> tuple[xgb.XGBClassifier, SafeMaxAbsScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    max_abs = np.max(np.abs(X_train), axis=0).astype(np.float32)
    max_abs[max_abs == 0] = 1.0
    scaler = SafeMaxAbsScaler(max_abs)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=float(np.sum(y_train == 0)) / max(1, np.sum(y_train == 1)),
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["normal", "failure"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    zeros = scaler.transform(
        np.zeros((1, X.shape[1]), dtype=np.float32)
    ).astype(np.float32)
    proba_zeros = model.predict_proba(zeros)[0]
    print(
        f"\nAll-zeros: P(normal)={proba_zeros[0]:.4f}"
        f" P(failure)={proba_zeros[1]:.4f}"
    )

    feature_names = json.loads(FEATURES_PATH.read_text())
    vec = np.zeros((1, len(feature_names)), dtype=np.float32)
    vec[0, feature_names.index("sshd+failed password")] = 1.0
    vec_s = scaler.transform(vec).astype(np.float32)
    proba_attack = model.predict_proba(vec_s)[0]
    print(
        f"sshd+failed password only: P(normal)={proba_attack[0]:.4f}"
        f" P(failure)={proba_attack[1]:.4f}"
    )

    vec2 = np.zeros((1, len(feature_names)), dtype=np.float32)
    vec2[0, feature_names.index("systemd+Started")] = 1.0
    vec2_s = scaler.transform(vec2).astype(np.float32)
    proba_normal = model.predict_proba(vec2_s)[0]
    print(
        f"systemd+Started only: P(normal)={proba_normal[0]:.4f}"
        f" P(failure)={proba_normal[1]:.4f}"
    )

    return model, scaler


def export_onnx(model: xgb.XGBClassifier, scaler: SafeMaxAbsScaler) -> None:
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, 100]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        onnx_path = OUTPUT_DIR / "log_classifier_syslog.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        print(f"\nExported ONNX to {onnx_path}")
    except ImportError:
        print("\nonnxmltools not available, saving as XGBoost JSON only")
        model.save_model(str(OUTPUT_DIR / "log_classifier_syslog.json"))

    scaler.to_json(str(OUTPUT_DIR / "scaler_syslog.json"))
    print(f"Saved scaler to {OUTPUT_DIR / 'scaler_syslog.json'}")

    onnx_path = OUTPUT_DIR / "log_classifier_syslog.onnx"
    if onnx_path.exists():
        sess = rt.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        zeros = scaler.transform(
            np.zeros((1, 100), dtype=np.float32)
        ).astype(np.float32)
        outputs = sess.run(None, {input_name: zeros})
        proba = outputs[1][0]
        print(
            f"ONNX verification (all-zeros): P(normal)={proba[0]:.4f}"
            f" P(failure)={proba[1]:.4f}"
        )

        vec = np.zeros((1, 100), dtype=np.float32)
        vec[0, 0] = 1.0
        vec_s = scaler.transform(vec).astype(np.float32)
        outputs2 = sess.run(None, {input_name: vec_s})
        proba2 = outputs2[1][0]
        print(
            f"ONNX verification (feature[0]=1): P(normal)={proba2[0]:.4f}"
            f" P(failure)={proba2[1]:.4f}"
        )


def main() -> None:
    print("Generating sparse-feature training data...")
    X, y = generate_dataset()
    print(f"Dataset: {len(X)} samples, {np.sum(y == 0)} normal, {np.sum(y == 1)} failure")
    print(f"Avg features per sample: {np.mean(np.count_nonzero(X, axis=1)):.1f}")

    print("\nTraining...")
    model, scaler = train(X, y)

    print("\nExporting...")
    export_onnx(model, scaler)


if __name__ == "__main__":
    main()
