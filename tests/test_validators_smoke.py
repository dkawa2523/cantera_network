import json

import pytest

from rxn_platform.backends.base import RunDataset, dump_run_dataset
from rxn_platform.core import ArtifactManifest, dump_manifest
from rxn_platform.errors import ValidationError
from rxn_platform.validators import (
    validate_feature_artifact,
    validate_graph_artifact,
    validate_observable_artifact,
    validate_run_artifact,
    validate_sensitivity_artifact,
)


def _write_manifest(path, *, kind: str, artifact_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-17T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )
    dump_manifest(path / "manifest.yaml", manifest)


def test_validate_artifacts_smoke(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, kind="runs", artifact_id="run-1")
    dataset = RunDataset(
        coords={
            "time": {"dims": ["time"], "data": [0.0, 1.0]},
            "species": {"dims": ["species"], "data": ["A", "B"]},
        },
        data_vars={"T": {"dims": ["time"], "data": [300.0, 301.0]}},
        attrs={"units": {"time": "s"}, "model": "dummy"},
    )
    dump_run_dataset(dataset, run_dir / "state.zarr")
    validate_run_artifact(run_dir)

    obs_dir = tmp_path / "observables" / "obs-1"
    obs_dir.mkdir(parents=True)
    _write_manifest(obs_dir, kind="observables", artifact_id="obs-1")
    (obs_dir / "values.parquet").write_text("stub\n", encoding="utf-8")
    validate_observable_artifact(
        obs_dir,
        table_columns=["run_id", "observable", "value", "unit", "meta_json"],
    )

    graph_dir = tmp_path / "graphs" / "graph-1"
    graph_dir.mkdir(parents=True)
    _write_manifest(graph_dir, kind="graphs", artifact_id="graph-1")
    graph_payload = json.dumps({"nodes": [], "edges": []})
    (graph_dir / "graph.json").write_text(graph_payload + "\n", encoding="utf-8")
    validate_graph_artifact(graph_dir)

    feature_dir = tmp_path / "features" / "feature-1"
    feature_dir.mkdir(parents=True)
    _write_manifest(feature_dir, kind="features", artifact_id="feature-1")
    (feature_dir / "features.parquet").write_text("stub\n", encoding="utf-8")
    validate_feature_artifact(
        feature_dir,
        table_columns=["run_id", "value"],
    )

    sensitivity_dir = tmp_path / "sensitivity" / "sens-1"
    sensitivity_dir.mkdir(parents=True)
    _write_manifest(sensitivity_dir, kind="sensitivity", artifact_id="sens-1")
    (sensitivity_dir / "sensitivity.parquet").write_text(
        "stub\n", encoding="utf-8"
    )
    validate_sensitivity_artifact(
        sensitivity_dir,
        table_columns=["reaction_id", "value"],
    )


def test_validate_observable_reports_missing_columns(tmp_path) -> None:
    obs_dir = tmp_path / "observables" / "obs-2"
    obs_dir.mkdir(parents=True)
    _write_manifest(obs_dir, kind="observables", artifact_id="obs-2")
    (obs_dir / "values.parquet").write_text("stub\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        validate_observable_artifact(
            obs_dir,
            table_columns=["run_id", "observable", "value"],
        )

    message = str(exc.value)
    assert "values.parquet:unit" in message
    assert "values.parquet:meta_json" in message
