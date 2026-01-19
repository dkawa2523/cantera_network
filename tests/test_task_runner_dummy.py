from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from rxn_platform.core import ArtifactManifest, make_artifact_id
from rxn_platform.errors import TaskError
from rxn_platform.registry import Registry
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.runner import run_task, run_task_from_config


def _dummy_task(cfg: Mapping[str, object], *, store: ArtifactStore):
    artifact_id = make_artifact_id(
        inputs={"task": "dummy"},
        config=dict(cfg),
        code={"version": "0.0.0"},
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="features",
        id=artifact_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs={"task": "dummy"},
        config=dict(cfg),
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / "result.txt").write_text("ok", encoding="utf-8")

    return store.ensure(manifest, writer=_writer)


def test_run_task_executes_dummy(tmp_path: Path) -> None:
    registry = Registry()
    registry.register("task", "dummy.task", _dummy_task)
    store = ArtifactStore(tmp_path / "artifacts")
    cfg = {"task": {"name": "dummy.task"}}

    result = run_task("dummy.task", cfg, store=store, registry=registry)

    assert result.path.exists()
    assert (result.path / "result.txt").read_text(encoding="utf-8") == "ok"


def test_run_task_wraps_errors(tmp_path: Path) -> None:
    registry = Registry()

    def _boom(cfg: Mapping[str, object], *, store: ArtifactStore) -> None:
        raise ValueError("boom")

    registry.register("task", "boom.task", _boom)
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TaskError):
        run_task("boom.task", {"task": {"name": "boom.task"}}, store=store, registry=registry)


def test_run_task_from_config_uses_store_root(tmp_path: Path) -> None:
    registry = Registry()
    registry.register("task", "dummy.task", _dummy_task)
    store_root = tmp_path / "artifacts"
    cfg = {"task": {"name": "dummy.task"}, "store": {"root": str(store_root)}}

    result = run_task_from_config(cfg, registry=registry)

    assert result.path == store_root / "features" / result.manifest.id
