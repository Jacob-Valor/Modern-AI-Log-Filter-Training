"""Generate or validate the model manifest.

The manifest tracks artifact hashes, feature counts, and training metadata
to prevent drift between trained models and runtime consumers.

Usage:
  python scripts/model_manifest.py generate   # create/update models/model_manifest.json
  python scripts/model_manifest.py validate   # check artifacts match manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

MODEL_DIRS = {
    "classifier": ROOT / "models",
    "syslog_classifier": ROOT / "models" / "syslog",
    "ner": ROOT / "models" / "ner" / "final",
    "biencoder": ROOT / "models" / "biencoder" / "final",
    "cross_encoder": ROOT / "models" / "cross_encoder" / "final",
    "tier2": ROOT / "models" / "tier2",
}

ARTIFACT_PATTERNS = {
    "classifier": ["log_classifier.onnx", "scaler.json", "feature_names.json"],
    "syslog_classifier": ["log_classifier_syslog.onnx"],
    "ner": ["config.json", "model.safetensors", "model.onnx"],
    "biencoder": ["config.json", "model.safetensors"],
    "cross_encoder": ["config.json", "model.safetensors"],
    "tier2": ["log_classifier_tier2.onnx"],
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def count_onnx_features(model_path: Path) -> int | None:
    if not model_path.exists():
        return None
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        if len(shape) > 1 and isinstance(shape[1], int):
            return shape[1]
    except Exception:
        pass
    return None


def count_feature_names(feature_names_path: Path) -> int | None:
    if not feature_names_path.exists():
        return None
    try:
        names = json.loads(feature_names_path.read_text())
        return len(names)
    except Exception:
        return None


def gather_model_info(model_key: str, model_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(model_dir.relative_to(ROOT)), "artifacts": {}}
    patterns = ARTIFACT_PATTERNS.get(model_key, [])

    for pattern in patterns:
        artifact_path = model_dir / pattern
        if artifact_path.exists():
            info["artifacts"][pattern] = {
                "sha256": file_sha256(artifact_path),
                "size_mb": file_size_mb(artifact_path),
            }

    if model_key == "classifier":
        onnx_path = model_dir / "log_classifier.onnx"
        fn_path = model_dir / "feature_names.json"
        onnx_features = count_onnx_features(onnx_path)
        fn_features = count_feature_names(fn_path)
        if onnx_features is not None:
            info["onnx_input_features"] = onnx_features
        if fn_features is not None:
            info["feature_names_count"] = fn_features
        if onnx_features is not None and fn_features is not None:
            info["feature_count_match"] = onnx_features == fn_features

    if model_key == "syslog_classifier":
        onnx_path = model_dir / "log_classifier_syslog.onnx"
        onnx_features = count_onnx_features(onnx_path)
        if onnx_features is not None:
            info["onnx_input_features"] = onnx_features

    return info


def generate_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {"schema_version": 1, "models": {}}

    for key, model_dir in MODEL_DIRS.items():
        if model_dir.exists():
            manifest["models"][key] = gather_model_info(key, model_dir)
        else:
            manifest["models"][key] = {"path": str(model_dir.relative_to(ROOT)), "status": "missing"}

    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for key, expected in manifest.get("models", {}).items():
        if expected.get("status") == "missing":
            continue

        model_dir = ROOT / expected.get("path", "")
        if not model_dir.exists():
            errors.append(f"{key}: directory {model_dir} not found")
            continue

        for pattern, meta in expected.get("artifacts", {}).items():
            artifact_path = model_dir / pattern
            if not artifact_path.exists():
                errors.append(f"{key}: artifact {pattern} missing")
                continue
            actual_hash = file_sha256(artifact_path)
            if actual_hash != meta.get("sha256"):
                errors.append(
                    f"{key}: {pattern} hash mismatch "
                    f"(expected {meta['sha256']}, got {actual_hash})"
                )

        if key == "classifier" and "feature_count_match" in expected:
            onnx_path = model_dir / "log_classifier.onnx"
            fn_path = model_dir / "feature_names.json"
            onnx_features = count_onnx_features(onnx_path)
            fn_features = count_feature_names(fn_path)
            if onnx_features is not None and fn_features is not None:
                if onnx_features != fn_features:
                    errors.append(
                        f"{key}: feature count mismatch "
                        f"(ONNX={onnx_features}, feature_names={fn_features})"
                    )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model manifest management")
    parser.add_argument("action", choices=["generate", "validate"], help="Action to perform")
    args = parser.parse_args(argv)

    if args.action == "generate":
        manifest = generate_manifest()
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Manifest written to {MANIFEST_PATH}")
        for key, info in manifest["models"].items():
            status = info.get("status", "ok")
            artifacts = len(info.get("artifacts", {}))
            print(f"  {key}: {status} ({artifacts} artifacts)")
        return 0

    if args.action == "validate":
        if not MANIFEST_PATH.exists():
            print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
            print("Run: python scripts/model_manifest.py generate")
            return 1
        manifest = json.loads(MANIFEST_PATH.read_text())
        errors = validate_manifest(manifest)
        if errors:
            print("VALIDATION FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("All models match manifest.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
