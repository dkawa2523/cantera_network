from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from rxn_platform.core import ArtifactManifest, make_artifact_id, make_run_id
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


def test_pipeline_runner_resolves_step_refs(tmp_path: Path) -> None:
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
                "inputs": {"from_first": "@first"},
            },
            {
                "id": "third",
                "task": "dummy.step",
                "inputs": {"from_prev": "@last"},
            },
        ]
    }

    results = runner.run(pipeline_cfg)

    second_manifest = store.open_manifest("features", results["second"])
    third_manifest = store.open_manifest("features", results["third"])

    assert second_manifest.inputs["from_first"] == results["first"]
    assert third_manifest.inputs["from_prev"] == results["second"]


def test_pipeline_runner_writes_pipeline_run_artifact(tmp_path: Path) -> None:
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
                "inputs": {"from_first": "@first"},
            },
        ]
    }

    results = runner.run(pipeline_cfg)
    pipeline_run_id = make_run_id(pipeline_cfg, exclude_keys=("hydra",))
    pipeline_dir = store.artifact_dir("pipelines", pipeline_run_id)
    results_path = pipeline_dir / "results.json"

    manifest = store.open_manifest("pipelines", pipeline_run_id)
    assert manifest.id == pipeline_run_id
    assert results_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["pipeline_run_id"] == pipeline_run_id
    assert payload["results"] == results
    assert [step["id"] for step in payload["steps"]] == ["first", "second"]
    for step in payload["steps"]:
        assert "started_at" in step
        assert "ended_at" in step
        assert "elapsed_seconds" in step
