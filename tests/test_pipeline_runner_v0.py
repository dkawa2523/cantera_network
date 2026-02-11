from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from rxn_platform.core import ArtifactManifest, make_artifact_id
from rxn_platform.errors import TaskError
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry
from rxn_platform.store import ArtifactStore


def _dummy_task(cfg: Mapping[str, object], *, store: ArtifactStore):
    config = dict(cfg)
    inputs = dict(config.get("inputs", {}))
    artifact_id = make_artifact_id(
        inputs=inputs,
        config=config,
        code={"version": "0.0.0"},
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="features",
        id=artifact_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs=inputs,
        config=config,
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / "result.txt").write_text("ok", encoding="utf-8")

    return store.ensure(manifest, writer=_writer)


def test_pipeline_runner_executes_steps(tmp_path: Path) -> None:
    registry = Registry()
    registry.register("task", "dummy.step", _dummy_task)
    store = ArtifactStore(tmp_path / "artifacts")
    runner = PipelineRunner(store=store, registry=registry, config_dir=tmp_path)

    pipeline_cfg = {
        "steps": [
            {"id": "first", "task": "dummy.step", "params": {"value": 1}},
            {
                "id": "second",
                "task": "dummy.step",
                "inputs": {"previous": "artifact-123"},
                "params": {"value": 2},
            },
        ]
    }

    results = runner.run(pipeline_cfg)

    assert set(results.keys()) == {"first", "second"}
    first_path = store.artifact_dir("features", results["first"])
    second_path = store.artifact_dir("features", results["second"])
    assert (first_path / "result.txt").read_text(encoding="utf-8") == "ok"
    assert (second_path / "result.txt").read_text(encoding="utf-8") == "ok"


def test_pipeline_runner_reports_step_failure(tmp_path: Path) -> None:
    registry = Registry()

    def _boom(cfg: Mapping[str, object], *, store: ArtifactStore) -> None:
        raise ValueError("boom")

    registry.register("task", "boom.step", _boom)
    runner = PipelineRunner(
        store=ArtifactStore(tmp_path / "artifacts"),
        registry=registry,
        config_dir=tmp_path,
    )
    pipeline_cfg = {"steps": [{"id": "explode", "task": "boom.step"}]}

    with pytest.raises(TaskError) as exc:
        runner.run(pipeline_cfg)

    assert "explode" in str(exc.value)
