"""Tests for the lightweight JSON-based model registry."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from logfilter.monitoring.model_registry import ModelRegistry, RegistryRun


@pytest.fixture
def tmp_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry.json")


class TestRegistryInit:
    def test_default_path(self) -> None:
        reg = ModelRegistry()
        assert "models/registry.json" in str(reg.registry_path)

    def test_custom_path(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "custom.json")
        assert reg.registry_path == tmp_path / "custom.json"


class TestRegisterRun:
    def test_register_tier1(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(
            model_type="tier1",
            artifact_dir="models/runs/test-run",
            metrics={"f1": 0.95},
            hyperparameters={"n_estimators": 300},
        )
        assert run.model_type == "tier1"
        assert run.status == "staging"
        assert run.metrics["f1"] == 0.95
        assert run.run_id.startswith("20")

    def test_register_generates_unique_ids(self, tmp_registry: ModelRegistry) -> None:
        r1 = tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        r2 = tmp_registry.register_run(model_type="tier1", artifact_dir="b")
        assert r1.run_id != r2.run_id

    def test_register_persists(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(
            model_type="tier2", artifact_dir="dir", metrics={"acc": 0.9}
        )
        reg2 = ModelRegistry(tmp_registry.registry_path)
        retrieved = reg2.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.metrics["acc"] == 0.9


class TestGetRun:
    def test_get_existing(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="dir")
        found = tmp_registry.get_run(run.run_id)
        assert found is not None
        assert found.run_id == run.run_id

    def test_get_missing(self, tmp_registry: ModelRegistry) -> None:
        assert tmp_registry.get_run("nonexistent") is None


class TestListRuns:
    def test_list_all(self, tmp_registry: ModelRegistry) -> None:
        tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        tmp_registry.register_run(model_type="tier2", artifact_dir="b")
        runs = tmp_registry.list_runs()
        assert len(runs) == 2

    def test_filter_by_model_type(self, tmp_registry: ModelRegistry) -> None:
        tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        tmp_registry.register_run(model_type="tier2", artifact_dir="b")
        assert len(tmp_registry.list_runs(model_type="tier1")) == 1
        assert len(tmp_registry.list_runs(model_type="tier2")) == 1

    def test_filter_by_status(self, tmp_registry: ModelRegistry) -> None:
        r1 = tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        tmp_registry.promote_to_production(r1.run_id)
        tmp_registry.register_run(model_type="tier1", artifact_dir="b")
        assert len(tmp_registry.list_runs(status="production")) == 1
        assert len(tmp_registry.list_runs(status="staging")) == 1

    def test_limit(self, tmp_registry: ModelRegistry) -> None:
        for i in range(5):
            tmp_registry.register_run(model_type="tier1", artifact_dir=f"dir{i}")
        assert len(tmp_registry.list_runs(limit=3)) == 3

    def test_reverse_chronological_order(self, tmp_registry: ModelRegistry) -> None:
        r1 = tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        r2 = tmp_registry.register_run(model_type="tier1", artifact_dir="b")
        runs = tmp_registry.list_runs()
        assert runs[0].run_id == r2.run_id
        assert runs[1].run_id == r1.run_id


class TestPromoteToProduction:
    def test_promote_staging(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="dir")
        promoted = tmp_registry.promote_to_production(run.run_id)
        assert promoted.status == "production"

    def test_promote_archives_previous(self, tmp_registry: ModelRegistry) -> None:
        r1 = tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        tmp_registry.promote_to_production(r1.run_id)
        r2 = tmp_registry.register_run(model_type="tier1", artifact_dir="b")
        tmp_registry.promote_to_production(r2.run_id)
        assert tmp_registry.get_run(r1.run_id).status == "archived"
        assert tmp_registry.get_run(r2.run_id).status == "production"

    def test_promote_not_found(self, tmp_registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            tmp_registry.promote_to_production("fake-id")

    def test_promote_already_production(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="dir")
        tmp_registry.promote_to_production(run.run_id)
        promoted = tmp_registry.promote_to_production(run.run_id)
        assert promoted.status == "production"


class TestGetProduction:
    def test_get_production(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="dir")
        tmp_registry.promote_to_production(run.run_id)
        prod = tmp_registry.get_production()
        assert prod is not None
        assert prod.run_id == run.run_id

    def test_get_production_by_model_type(self, tmp_registry: ModelRegistry) -> None:
        r1 = tmp_registry.register_run(model_type="tier1", artifact_dir="a")
        r2 = tmp_registry.register_run(model_type="tier2", artifact_dir="b")
        tmp_registry.promote_to_production(r1.run_id)
        tmp_registry.promote_to_production(r2.run_id)
        assert tmp_registry.get_production("tier1").run_id == r1.run_id
        assert tmp_registry.get_production("tier2").run_id == r2.run_id

    def test_get_production_none(self, tmp_registry: ModelRegistry) -> None:
        assert tmp_registry.get_production() is None


class TestArchiveRun:
    def test_archive(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="dir")
        archived = tmp_registry.archive_run(run.run_id)
        assert archived.status == "archived"

    def test_archive_not_found(self, tmp_registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            tmp_registry.archive_run("fake-id")


class TestUpdateMetrics:
    def test_update(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(
            model_type="tier1", artifact_dir="dir", metrics={"f1": 0.90}
        )
        updated = tmp_registry.update_metrics(run.run_id, {"f1": 0.95, "acc": 0.99})
        assert updated.metrics["f1"] == 0.95
        assert updated.metrics["acc"] == 0.99

    def test_update_not_found(self, tmp_registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            tmp_registry.update_metrics("fake-id", {"f1": 0.95})


class TestFindRunByArtifactDir:
    def test_find(self, tmp_registry: ModelRegistry) -> None:
        run = tmp_registry.register_run(model_type="tier1", artifact_dir="models/run-1")
        found = tmp_registry.find_run_by_artifact_dir("models/run-1")
        assert found is not None
        assert found.run_id == run.run_id

    def test_find_missing(self, tmp_registry: ModelRegistry) -> None:
        assert tmp_registry.find_run_by_artifact_dir("nonexistent") is None


class TestThreadSafety:
    def test_concurrent_register(self, tmp_registry: ModelRegistry) -> None:
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for i in range(10):
                    tmp_registry.register_run(
                        model_type="tier1", artifact_dir=f"dir{n}-{i}"
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(tmp_registry.list_runs()) == 50


class TestRegistryRun:
    def test_roundtrip(self) -> None:
        run = RegistryRun(
            run_id="r1",
            timestamp="2025-01-01T00:00:00Z",
            model_type="tier1",
            artifact_dir="dir",
            metrics={"f1": 0.9},
            status="staging",
        )
        d = run.to_dict()
        restored = RegistryRun.from_dict(d)
        assert restored.run_id == "r1"
        assert restored.metrics["f1"] == 0.9

    def test_from_dict_ignores_extra(self) -> None:
        restored = RegistryRun.from_dict(
            {
                "run_id": "r1",
                "timestamp": "t",
                "model_type": "tier1",
                "artifact_dir": "dir",
                "extra_field": "ignored",
            }
        )
        assert restored.run_id == "r1"
