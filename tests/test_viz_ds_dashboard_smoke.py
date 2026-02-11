from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks import viz as viz_task


def _make_manifest(
    kind: str,
    artifact_id: str,
    *,
    config: Optional[dict[str, Any]] = None,
    parents: Optional[list[str]] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-18T00:00:00Z",
        parents=parents or [],
        inputs=inputs or {},
        config=config or {"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def _write_json_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    payload = {"columns": columns, "rows": rows}
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_viz_ds_dashboard_smoke(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    run_1 = _make_manifest(
        "runs",
        "run-1",
        config={"sim": {"initial": {"T": 700.0, "P": 101325.0}}},
    )
    run_2 = _make_manifest(
        "runs",
        "run-2",
        config={"sim": {"initial": {"T": 800.0, "P": 120000.0}}},
    )
    store.ensure(run_1)
    store.ensure(run_2)

    obs_manifest = _make_manifest(
        "observables",
        "obs-1",
        parents=[run_1.id, run_2.id],
        inputs={"runs": [run_1.id, run_2.id], "observables": ["objective"]},
    )

    def _write_obs(base_dir: Path) -> None:
        rows = [
            {
                "run_id": run_1.id,
                "observable": "objective",
                "value": 1.0,
                "unit": "1",
                "meta_json": "{}",
            },
            {
                "run_id": run_2.id,
                "observable": "objective",
                "value": 2.0,
                "unit": "1",
                "meta_json": "{}",
            },
        ]
        _write_json_table(
            base_dir / "values.parquet",
            ["run_id", "observable", "value", "unit", "meta_json"],
            rows,
        )

    store.ensure(obs_manifest, writer=_write_obs)

    sens_manifest = _make_manifest(
        "sensitivity",
        "sens-1",
        parents=[run_1.id],
        inputs={"runs": [run_1.id], "targets": ["objective"]},
    )

    def _write_sens(base_dir: Path) -> None:
        rows = [
            {
                "run_id": run_1.id,
                "target": "objective",
                "reaction_id": "R1",
                "reaction_index": 1,
                "value": 0.5,
                "unit": "1",
                "rank": 1,
                "meta_json": "{}",
                "condition_id": "c1",
            },
            {
                "run_id": run_1.id,
                "target": "objective",
                "reaction_id": "R2",
                "reaction_index": 2,
                "value": -0.2,
                "unit": "1",
                "rank": 2,
                "meta_json": "{}",
                "condition_id": "c1",
            },
        ]
        _write_json_table(
            base_dir / "sensitivity.parquet",
            [
                "run_id",
                "target",
                "reaction_id",
                "reaction_index",
                "value",
                "unit",
                "rank",
                "meta_json",
                "condition_id",
            ],
            rows,
        )

    store.ensure(sens_manifest, writer=_write_sens)

    cfg = {
        "viz": {
            "name": "ds_dashboard",
            "title": "DS Dashboard Smoke",
            "dashboard": "ds",
            "inputs": [
                {"kind": "runs", "id": run_1.id},
                {"kind": "runs", "id": run_2.id},
                {"kind": "observables", "id": obs_manifest.id},
                {"kind": "sensitivity", "id": sens_manifest.id},
                {"kind": "runs", "id": "missing-run"},
            ],
            "condition_fields": ["sim.initial.T", "sim.initial.P"],
            "objective": {
                "observables": ["objective"],
                "condition_field": "sim.initial.T",
            },
            "sensitivity": {
                "targets": ["objective"],
                "top_n": 5,
                "condition_id": "c1",
            },
            "placeholders": ["Convergence"],
        }
    }

    result = viz_task.ds_dashboard(cfg, store=store)

    html = (result.path / "index.html").read_text(encoding="utf-8")
    assert "Condition Distribution" in html
    assert "Objective Overview" in html
    assert "Sensitivity Heatmap" in html
    assert "runs/missing-run" in html
