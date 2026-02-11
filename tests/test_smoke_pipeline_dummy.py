from __future__ import annotations

from pathlib import Path

from rxn_platform.core import make_run_id
from rxn_platform.hydra_utils import compose_config, resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_smoke_pipeline_dummy_creates_run_artifact(tmp_path: Path) -> None:
    cfg = compose_config(config_path=_config_dir(), config_name="default")
    resolved = resolve_config(cfg)
    assert "pipeline" in resolved, "expected pipeline config in default"
    assert isinstance(
        resolved.get("pipeline"), dict
    ), "expected pipeline config to be a mapping"

    store = ArtifactStore(tmp_path / "artifacts")
    runner = PipelineRunner(store=store)

    try:
        results = runner.run(resolved)
    except Exception as exc:
        raise AssertionError(f"pipeline run failed: {exc}") from exc

    assert "sim" in results, "pipeline results missing 'sim' step"
    run_id = results["sim"]
    assert isinstance(run_id, str) and run_id, (
        "pipeline result for 'sim' must be a non-empty string"
    )

    assert store.exists("runs", run_id), (
        f"expected run manifest for {run_id} to exist"
    )
    run_manifest = store.open_manifest("runs", run_id)
    assert run_manifest.kind == "runs", "run manifest kind should be 'runs'"
    run_dir = store.artifact_dir("runs", run_id)
    state_path = run_dir / "state.zarr" / "dataset.json"
    assert state_path.exists(), "expected run dataset at state.zarr/dataset.json"

    pipeline_cfg = dict(resolved["pipeline"])
    common_cfg = resolved.get("common")
    if isinstance(common_cfg, dict):
        pipeline_cfg["common"] = dict(common_cfg)
    pipeline_run_id = make_run_id(pipeline_cfg, exclude_keys=("hydra",))
    assert store.exists("pipelines", pipeline_run_id), (
        f"expected pipeline manifest for {pipeline_run_id} to exist"
    )
    pipeline_manifest = store.open_manifest("pipelines", pipeline_run_id)
    assert pipeline_manifest.kind == "pipelines", (
        "pipeline manifest kind should be 'pipelines'"
    )
    results_path = store.artifact_dir("pipelines", pipeline_run_id) / "results.json"
    assert results_path.exists(), "expected pipeline results.json to be written"
