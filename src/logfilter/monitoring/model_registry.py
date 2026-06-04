"""Lightweight JSON-based model registry for tracking training runs and promotions.

No external services required — everything is stored in a single JSON file
(``models/registry.json`` by default).  Supports run registration, metric
updates, production promotion, and archival.

Concurrency-safe via an exclusive ``fcntl.flock`` held on a dedicated lock file
across each read-modify-write, so concurrent processes/threads cannot interleave
or lose updates.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RegistryRun:
    """A single model training run entry."""

    run_id: str
    timestamp: str
    model_type: str  # "tier1" | "tier2" | "ner" | "biencoder" | "cross_encoder"
    artifact_dir: str
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    status: str = "staging"  # staging | production | archived
    version: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryRun:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


class ModelRegistry:
    """JSON-backed model registry with atomic updates.

    Parameters
    ----------
    registry_path : Path
        Path to the JSON registry file (default: ``models/registry.json``).
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        project_root = Path(__file__).parent.parent.parent.parent
        self.registry_path = Path(registry_path or project_root / "models" / "registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Dedicated lock file: never replaced/unlinked, so flock on it stays valid
        # across the atomic os.replace() in _write().
        self._lock_path = self.registry_path.with_name(self.registry_path.name + ".lock")

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_run(
        self,
        model_type: str,
        artifact_dir: Path | str,
        metrics: dict[str, Any] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        version: str = "",
        notes: str = "",
    ) -> RegistryRun:
        """Register a new training run and return its record."""
        run_id = self._generate_run_id()
        run = RegistryRun(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            artifact_dir=str(artifact_dir),
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            status="staging",
            version=version or run_id,
            notes=notes,
        )
        with self._locked():
            data = self._read()
            data["runs"].append(run.to_dict())
            self._write(data)
        logger.info("Registered model run", run_id=run_id, model_type=model_type)
        return run

    def get_run(self, run_id: str) -> RegistryRun | None:
        """Retrieve a single run by ID."""
        with self._locked():
            data = self._read()
        for run_data in data["runs"]:
            if run_data["run_id"] == run_id:
                return RegistryRun.from_dict(run_data)
        return None

    def list_runs(
        self,
        model_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RegistryRun]:
        """List runs with optional filtering."""
        with self._locked():
            data = self._read()
        results: list[RegistryRun] = []
        for run_data in reversed(data["runs"]):
            if model_type is not None and run_data["model_type"] != model_type:
                continue
            if status is not None and run_data["status"] != status:
                continue
            results.append(RegistryRun.from_dict(run_data))
            if len(results) >= limit:
                break
        return results

    def get_production(self, model_type: str | None = None) -> RegistryRun | None:
        """Return the current production run (optionally filtered by model_type)."""
        with self._locked():
            data = self._read()
        for run_data in reversed(data["runs"]):
            if run_data["status"] != "production":
                continue
            if model_type is not None and run_data["model_type"] != model_type:
                continue
            return RegistryRun.from_dict(run_data)
        return None

    def promote_to_production(self, run_id: str) -> RegistryRun:
        """Promote a staging run to production and archive the previous one.

        Only the previous production run of the *same* model type is archived, so
        each tier (tier1/tier2/ner/…) can have its own production run concurrently.
        """
        with self._locked():
            data = self._read()
            promoted: dict[str, Any] | None = None
            for run_data in data["runs"]:
                if run_data["run_id"] == run_id:
                    promoted = run_data
                    break

            if promoted is None:
                raise ValueError(f"Run {run_id} not found")

            for run_data in data["runs"]:
                if (
                    run_data["run_id"] != run_id
                    and run_data["status"] == "production"
                    and run_data["model_type"] == promoted["model_type"]
                ):
                    run_data["status"] = "archived"
                    logger.info(
                        "Archived previous production run",
                        run_id=run_data["run_id"],
                        model_type=run_data["model_type"],
                    )

            promoted["status"] = "production"
            self._write(data)
        logger.info(
            "Promoted run to production",
            run_id=run_id,
            model_type=promoted["model_type"],
        )
        return RegistryRun.from_dict(promoted)

    def archive_run(self, run_id: str) -> RegistryRun:
        """Mark a run as archived."""
        with self._locked():
            data = self._read()
            for run_data in data["runs"]:
                if run_data["run_id"] == run_id:
                    run_data["status"] = "archived"
                    self._write(data)
                    logger.info("Archived run", run_id=run_id)
                    return RegistryRun.from_dict(run_data)
        raise ValueError(f"Run {run_id} not found")

    def update_metrics(self, run_id: str, metrics: dict[str, Any]) -> RegistryRun:
        """Update metrics for an existing run (e.g. after evaluation)."""
        with self._locked():
            data = self._read()
            for run_data in data["runs"]:
                if run_data["run_id"] == run_id:
                    run_data["metrics"] = {**run_data.get("metrics", {}), **metrics}
                    self._write(data)
                    return RegistryRun.from_dict(run_data)
        raise ValueError(f"Run {run_id} not found")

    def find_run_by_artifact_dir(self, artifact_dir: Path | str) -> RegistryRun | None:
        """Find a run by its artifact directory path."""
        target = Path(artifact_dir).resolve()
        with self._locked():
            data = self._read()
        for run_data in reversed(data["runs"]):
            if Path(run_data["artifact_dir"]).resolve() == target:
                return RegistryRun.from_dict(run_data)
        return None

    # ── Internal ─────────────────────────────────────────────────────────────────

    def _generate_run_id(self) -> str:
        now = datetime.now(timezone.utc)
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive lock across a whole read-modify-write.

        The lock is taken on a dedicated ``*.lock`` file rather than on the data
        file itself, so the atomic ``os.replace()`` in :meth:`_write` cannot swap
        out the inode another caller is holding the lock on.
        """
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read(self) -> dict[str, Any]:
        """Read registry state. Caller must hold :meth:`_locked`."""
        if not self.registry_path.exists():
            return {"schema_version": "1.0", "runs": []}
        content = self.registry_path.read_text(encoding="utf-8")
        if not content.strip():
            return {"schema_version": "1.0", "runs": []}
        return json.loads(content)

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically persist registry state. Caller must hold :meth:`_locked`."""
        tmp_path = Path(str(self.registry_path) + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.registry_path)
