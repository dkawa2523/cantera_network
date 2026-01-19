from __future__ import annotations

import json
import math
from typing import Any

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


def test_observable_film_thickness_proxy(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = "run-film-proxy"
    dataset = RunDataset(
        coords={
            "time": {"dims": ["time"], "data": [0.0, 1.0, 2.0]},
            "species": {"dims": ["species"], "data": ["SiH4", "H2"]},
        },
        data_vars={
            "net_production_rates": {
                "dims": ["time", "species"],
                "data": [
                    [-1.0, 0.2],
                    [-2.0, 0.1],
                    [-3.0, 0.0],
                ],
            }
        },
        attrs={
            "units": {"time": "s", "net_production_rates": "mol/m3/s"},
            "model": "dummy",
        },
    )
    _write_run_artifact(store, run_id, dataset)

    obs_task = get("task", "observables.run")
    obs_cfg = {
        "inputs": {"run_id": run_id},
        "params": {
            "observables": [
                {
                    "name": "film_thickness",
                    "params": {
                        "proxy": {
                            "data_var": "net_production_rates",
                            "species": "SiH4",
                            "sign": -1.0,
                        }
                    },
                }
            ]
        },
    }
    obs_result = obs_task(obs_cfg, store=store)

    rows = _read_values(obs_result.path / "values.parquet")
    by_name = {row["observable"]: row for row in rows}

    assert "film.thickness" in by_name
    row = by_name["film.thickness"]
    assert math.isclose(row["value"], 4.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert row["unit"] == "mol/m3/s*s"

    meta = json.loads(row["meta_json"])
    assert meta.get("status") == "proxy"
    assert meta.get("source") == "data_vars.net_production_rates"
    assert meta.get("proxy_species") == "SiH4"
    assert math.isclose(meta.get("sign", 0.0), -1.0, rel_tol=0.0, abs_tol=1.0e-12)
