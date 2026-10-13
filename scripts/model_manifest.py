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
import importlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
_flow_features: Any = importlib.import_module("logfilter.pipeline.flow_features")
CANONICAL_FLOW_FEATURES = list(_flow_features.FLOW_FEATURE_NAMES)

MODEL_DIRS = {
    "classifier": ROOT / "models",
    "syslog_classifier": ROOT / "models" / "syslog",
    "ner": ROOT / "models" / "ner" / "final",
    "biencoder": ROOT / "models" / "biencoder" / "final",
    "cross_encoder": ROOT / "models" / "cross_encoder" / "final",
    "tier2": ROOT / "models" / "tier2",
    "flow_classifier": ROOT / "models" / "flow",
    "http_lexical": ROOT / "models" / "http",
    "endpoint_novelty": ROOT / "models" / "endpoint",
}

ARTIFACT_PATTERNS = {
    "classifier": ["log_classifier.onnx", "scaler.json", "feature_names.json"],
    "syslog_classifier": ["log_classifier_syslog.onnx"],
    "ner": ["config.json", "model.safetensors", "model.onnx"],
    "biencoder": ["config.json", "model.safetensors"],
    "cross_encoder": ["config.json", "model.safetensors"],
    "tier2": ["log_classifier_tier2.onnx"],
    "flow_classifier": [
        "flow_classifier.onnx",
        "flow_classifier.json",
        "scaler.json",
        "feature_names.json",
        "metrics.json",
        "calibration.json",
        "development_feature_hashes.txt",
        "fusion_calibration.json",
        "bot_classifier.onnx",
        "bot_classifier.json",
        "bot_scaler.json",
        "bot_feature_names.json",
        "bot_metrics.json",
        "bot_calibration.json",
        "bot_development_feature_hashes.txt",
    ],
    "http_lexical": ["lexical_calibration.json"],
    "endpoint_novelty": [
        "endpoint_novelty.onnx",
        "novelty_config.json",
        "novelty_metrics.json",
    ],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Cannot hash model artifact {path}") from exc
    return digest.hexdigest()


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def count_onnx_features(model_path: Path) -> int | None:
    if not model_path.exists():
        return None
    try:
        import onnxruntime as rt  # type: ignore[import-not-found]

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

    if model_key == "flow_classifier":
        onnx_features = count_onnx_features(model_dir / "flow_classifier.onnx")
        named_features = count_feature_names(model_dir / "feature_names.json")
        if onnx_features is not None:
            info["onnx_input_features"] = onnx_features
        if named_features is not None:
            info["feature_names_count"] = named_features
        if onnx_features is not None and named_features is not None:
            info["feature_count_match"] = onnx_features == named_features
        bot_onnx_features = count_onnx_features(model_dir / "bot_classifier.onnx")
        bot_named_features = count_feature_names(model_dir / "bot_feature_names.json")
        if bot_onnx_features is not None:
            info["bot_onnx_input_features"] = bot_onnx_features
        if bot_named_features is not None:
            info["bot_feature_names_count"] = bot_named_features
        if bot_onnx_features is not None and bot_named_features is not None:
            info["bot_feature_count_match"] = bot_onnx_features == bot_named_features
        try:
            flow_names = json.loads((model_dir / "feature_names.json").read_text())
            bot_names = json.loads((model_dir / "bot_feature_names.json").read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            info["canonical_feature_order_match"] = False
            info["bot_canonical_feature_order_match"] = False
        else:
            info["canonical_feature_order_match"] = flow_names == CANONICAL_FLOW_FEATURES
            info["bot_canonical_feature_order_match"] = bot_names == CANONICAL_FLOW_FEATURES

    if model_key == "endpoint_novelty":
        onnx_features = count_onnx_features(model_dir / "endpoint_novelty.onnx")
        if onnx_features is not None:
            info["onnx_input_features"] = onnx_features

    return info


def generate_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {"schema_version": 1, "models": {}}

    for key, model_dir in MODEL_DIRS.items():
        if model_dir.exists():
            manifest["models"][key] = gather_model_info(key, model_dir)
        else:
            manifest["models"][key] = {
                "path": str(model_dir.relative_to(ROOT)),
                "status": "missing",
            }

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
                    f"{key}: {pattern} hash mismatch (expected {meta['sha256']}, got {actual_hash})"
                )

        if key == "flow_classifier":
            for artifact in ("feature_names.json", "bot_feature_names.json"):
                try:
                    names = json.loads((model_dir / artifact).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(f"{key}: cannot read {artifact}: {exc}")
                    continue
                if names != CANONICAL_FLOW_FEATURES:
                    errors.append(f"{key}: {artifact} order differs from runtime schema")

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
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: Cannot read manifest: {exc}")
            return 1
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
