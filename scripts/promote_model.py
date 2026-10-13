"""Model promotion script.

Promotes a registered staging run to production by:
1. Updating the registry status
2. Copying artifacts to the production directory

Promotion is entirely local. The API exposes no reload endpoint (`app.py`
registers only /score, /score/batch, /health, /health/drift, /metrics and
/metrics/snapshot), so there is nothing to notify; a running API picks up the
new artifacts on its next restart.

Usage:
    python scripts/promote_model.py --run-id <run_id> [--production-dir models/production]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logfilter.monitoring.model_registry import ModelRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model run to production")
    parser.add_argument("--run-id", required=True, help="Registry run_id to promote")
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("models/production"),
        help="Destination directory for production artifacts",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Override path to registry.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = ModelRegistry(args.registry_path) if args.registry_path else ModelRegistry()

    run = registry.get_run(args.run_id)
    if run is None:
        print(f"ERROR: Run {args.run_id} not found in registry", file=sys.stderr)
        sys.exit(1)

    if run.status == "production":
        print(f"Run {args.run_id} is already in production")
        return

    # Promote in registry
    registry.promote_to_production(args.run_id)

    # Copy artifacts to production directory
    artifact_dir = Path(run.artifact_dir)
    if not artifact_dir.exists():
        print(
            f"WARNING: Artifact directory {artifact_dir} does not exist. "
            "Registry updated but artifacts not copied.",
            file=sys.stderr,
        )
        sys.exit(2)

    args.production_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing production artifacts
    for item in args.production_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Copy new artifacts
    for item in artifact_dir.iterdir():
        dest = args.production_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Promoted run {args.run_id} to production")
    print(f"  model_type: {run.model_type}")
    print(f"  artifacts:  {args.production_dir}")
    print(f"  metrics:    {run.metrics}")
    print("  reload:     not applicable (the API exposes no reload endpoint)")




if __name__ == "__main__":
    main()
