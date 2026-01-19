from __future__ import annotations

from pathlib import Path

from rxn_platform.hydra_utils import compose_config, resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore
from rxn_platform.validators import (
    validate_feature_artifact,
    validate_graph_artifact,
    validate_observable_artifact,
    validate_run_artifact,
    validate_sensitivity_artifact,
)


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_p1_integration_dummy_pipeline(tmp_path: Path) -> None:
    store_root = tmp_path / "artifacts"
    cfg = compose_config(
        config_path=_config_dir(),
        config_name="defaults",
        overrides=[
            f"store.root={store_root}",
            "pipeline=p1_smoke",
            "sim=dummy",
        ],
    )
    resolved = resolve_config(cfg)

    store = ArtifactStore(store_root)
    runner = PipelineRunner(store=store)
    results = runner.run(resolved)

    assert "sim" in results
    assert "observables" in results
    assert "graph" in results
    assert "features" in results
    assert "sensitivity" in results
    assert "viz_ds" in results
    assert "viz_chem" in results

    run_id = results["sim"]
    obs_id = results["observables"]
    graph_id = results["graph"]
    feat_id = results["features"]
    sens_id = results["sensitivity"]

    validate_run_artifact(store.artifact_dir("runs", run_id))
    validate_observable_artifact(
        store.artifact_dir("observables", obs_id),
        table_columns=["run_id", "observable", "value", "unit", "meta_json"],
    )
    validate_graph_artifact(store.artifact_dir("graphs", graph_id))
    validate_feature_artifact(
        store.artifact_dir("features", feat_id),
        table_columns=["run_id", "feature", "value", "unit", "meta_json"],
    )
    validate_sensitivity_artifact(
        store.artifact_dir("sensitivity", sens_id),
        table_columns=[
            "run_id",
            "target",
            "reaction_id",
            "reaction_index",
            "value",
            "unit",
            "rank",
            "meta_json",
        ],
    )

    for report_id in (results["viz_ds"], results["viz_chem"]):
        report_dir = store.artifact_dir("reports", report_id)
        assert (report_dir / "index.html").exists()
