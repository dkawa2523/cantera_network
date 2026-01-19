from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401

from rxn_platform.backends.base import RunDataset, dump_run_dataset
from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _read_values(path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _write_run_artifact(store: ArtifactStore, run_id: str, dataset: RunDataset) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id=run_id,
        created_at="2026-01-17T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )

    def _writer(base_dir) -> None:
        dump_run_dataset(dataset, base_dir / "state.zarr")

    store.ensure(manifest, writer=_writer)


def test_observable_coverage_missing_ok(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 2.0, "steps": 3},
            "species": ["A", "B"],
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    obs_task = get("task", "observables.run")
    obs_cfg = {
        "inputs": {"run_id": sim_result.manifest.id},
        "params": {"observables": [{"name": "coverage_summary"}]},
    }
    obs_result = obs_task(obs_cfg, store=store)

    rows = _read_values(obs_result.path / "values.parquet")
    assert rows

    for row in rows:
        assert row["observable"].startswith("cov.")
        assert math.isnan(row["value"])
        meta = json.loads(row["meta_json"])
        assert meta.get("status") == "missing_coverage"


def test_observable_coverage_values_present(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = "run-coverage"
    dataset = RunDataset(
        coords={
            "time": {"dims": ["time"], "data": [0.0, 1.0, 2.0]},
            "surface_species": {
                "dims": ["surface_species"],
                "data": ["S1", "S2"],
            },
        },
        data_vars={
            "coverage": {
                "dims": ["time", "surface_species"],
                "data": [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.3, 0.7],
                ],
            }
        },
        attrs={"units": {"time": "s", "coverage": "fraction"}, "model": "dummy"},
    )
    _write_run_artifact(store, run_id, dataset)

    obs_task = get("task", "observables.run")
    obs_cfg = {
        "inputs": {"run_id": run_id},
        "params": {
            "observables": [
                {
                    "name": "coverage_summary",
                    "params": {"stats": ["last", "mean", "max", "integral"]},
                }
            ]
        },
    }
    obs_result = obs_task(obs_cfg, store=store)

    rows = _read_values(obs_result.path / "values.parquet")
    by_name = {row["observable"]: row for row in rows}

    assert "cov.S1.last" in by_name
    assert "cov.S1.integral" in by_name
    assert "cov.S2.mean" in by_name

    assert math.isclose(by_name["cov.S1.last"]["value"], 0.3, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(
        by_name["cov.S1.integral"]["value"],
        0.4,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(by_name["cov.S2.mean"]["value"], 0.8, rel_tol=0.0, abs_tol=1.0e-9)
    assert by_name["cov.S1.integral"]["unit"] == "fraction*s"
