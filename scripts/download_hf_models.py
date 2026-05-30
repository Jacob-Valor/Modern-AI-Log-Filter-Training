#!/usr/bin/env python3
"""Pre-download HuggingFace models to a local cache for offline/air-gapped use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    raw = path.read_text()
    # Simple env-var substitution: ${VAR:default} -> value or default
    import os
    import re

    def _sub(m: Any) -> str:
        var = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var, default)

    resolved = re.sub(r"\$\{([^}:]+)(?::([^}]*))?\}", _sub, raw)
    return yaml.safe_load(resolved)


def _maybe_empty(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _download(
    model_id: str,
    cache_dir: Path,
    revision: str | None = None,
    library: str = "transformers",
) -> None:
    """Download a model snapshot to the local cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(
            "huggingface_hub is required. Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Downloading {model_id} (library={library}, revision={revision or 'default'}) ...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        revision=revision,
        local_files_only=False,
    )
    print(f"  -> cached in {cache_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download HuggingFace models for offline/air-gapped deployments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Target cache dir (default: read from config HF_HOME, else ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--model",
        choices=["ner", "biencoder", "cross_encoder", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token (for gated/private models).",
    )
    args = parser.parse_args(argv)

    if args.token:
        import os

        os.environ.setdefault("HF_TOKEN", args.token)

    cfg = _load_config(args.config)
    models_cfg = cfg.get("models", {})

    cache_dir = args.cache_dir
    if cache_dir is None:
        global_cache = _maybe_empty(cfg.get("cache_dir", ""))
        if global_cache:
            cache_dir = Path(global_cache)
        else:
            cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, str, str | None, str]] = []
    for name, library in (
        ("ner", "transformers"),
        ("biencoder", "sentence-transformers"),
        ("cross_encoder", "sentence-transformers"),
    ):
        if args.model != "all" and args.model != name:
            continue
        mcfg = models_cfg.get(name, {})
        model_id = mcfg.get("model_id", "")
        if not model_id:
            print(f"Skipping {name}: no model_id in config", file=sys.stderr)
            continue
        revision = _maybe_empty(mcfg.get("revision", ""))
        targets.append((name, model_id, revision, library))

    if not targets:
        print("No models configured for download.", file=sys.stderr)
        return 1

    errors = 0
    for name, model_id, revision, library in targets:
        try:
            _download(model_id, cache_dir, revision=revision, library=library)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"\n{errors}/{len(targets)} downloads failed.", file=sys.stderr)
        return 1

    print(f"\nAll {len(targets)} model(s) cached in {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
