from __future__ import annotations

from pathlib import Path

from rxn_platform.hydra_utils import compose_config, resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_p2_integration_dummy_pipeline(tmp_path: Path) -> None:
    store_root = tmp_path / "artifacts"
    cfg = compose_config(
        config_path=_config_dir(),
        config_name="defaults",
        overrides=[
            f"store.root={store_root}",
            "pipeline=p2_smoke",
            "sim=dummy",
        ],
    )
    resolved = resolve_config(cfg)

    store = ArtifactStore(store_root)
    runner = PipelineRunner(store=store)
    results = runner.run(resolved)

    assert "opt" in results
    assert "assim" in results
    assert "sensitivity" in results
    assert "reduction" in results
    assert "validation" in results
    assert "benchmark" in results

    opt_id = results["opt"]
    opt_manifest = store.read_manifest("optimization", opt_id)
    assert opt_manifest.parents
    opt_dir = store.artifact_dir("optimization", opt_id)
    assert (opt_dir / "history.parquet").exists()
    assert (opt_dir / "pareto.parquet").exists()

    assim_id = results["assim"]
    assim_manifest = store.read_manifest("assimilation", assim_id)
    assert assim_manifest.parents
    assim_dir = store.artifact_dir("assimilation", assim_id)
    assert (assim_dir / "posterior.parquet").exists()
    assert (assim_dir / "misfit_history.parquet").exists()

    sens_id = results["sensitivity"]
    sens_manifest = store.read_manifest("sensitivity", sens_id)
    assert sens_manifest.parents
    sens_dir = store.artifact_dir("sensitivity", sens_id)
    assert (sens_dir / "sensitivity.parquet").exists()

    reduction_id = results["reduction"]
    reduction_manifest = store.read_manifest("reduction", reduction_id)
    assert sens_id in reduction_manifest.parents
    reduction_dir = store.artifact_dir("reduction", reduction_id)
    assert (reduction_dir / "mechanism_patch.yaml").exists()

    validation_id = results["validation"]
    validation_manifest = store.read_manifest("validation", validation_id)
    assert reduction_id in validation_manifest.parents
    validation_dir = store.artifact_dir("validation", validation_id)
    assert (validation_dir / "metrics.parquet").exists()

    benchmark_id = results["benchmark"]
    benchmark_manifest = store.read_manifest("reports", benchmark_id)
    for expected in (opt_id, assim_id, validation_id):
        assert expected in benchmark_manifest.parents
    benchmark_dir = store.artifact_dir("reports", benchmark_id)
    assert (benchmark_dir / "index.html").exists()
