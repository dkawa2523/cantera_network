from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401

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


def _load_dataset_payload(run_path) -> dict[str, Any]:
    dataset_path = run_path / "state.zarr" / "dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _compute_stats(series: list[float], time_values: list[float]) -> dict[str, float]:
    last = series[-1]
    mean = sum(series) / float(len(series))
    max_value = max(series)
    integral = 0.0
    for index in range(1, len(series)):
        dt = time_values[index] - time_values[index - 1]
        integral += 0.5 * (series[index] + series[index - 1]) * dt
    return {
        "last": last,
        "mean": mean,
        "max": max_value,
        "integral": integral,
    }


def test_gas_composition_species_stats(tmp_path) -> None:
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
        "params": {
            "observables": [
                {
                    "name": "gas_composition",
                    "params": {
                        "species": ["A"],
                        "stats": ["last", "mean", "max", "integral"],
                    },
                }
            ]
        },
    }
    obs_result = obs_task(obs_cfg, store=store)

    rows = _read_values(obs_result.path / "values.parquet")
    by_name = {row["observable"]: row for row in rows}
    assert set(by_name.keys()) == {
        "gas.A.last",
        "gas.A.mean",
        "gas.A.max",
        "gas.A.integral",
    }

    payload = _load_dataset_payload(sim_result.path)
    time_values = [float(value) for value in payload["coords"]["time"]["data"]]
    series = [float(row[0]) for row in payload["data_vars"]["X"]["data"]]
    expected = _compute_stats(series, time_values)

    for stat in ("last", "mean", "max", "integral"):
        row = by_name[f"gas.A.{stat}"]
        assert math.isclose(row["value"], expected[stat], rel_tol=0.0, abs_tol=1.0e-9)
        if stat == "integral":
            assert row["unit"] == "mole_fraction*s"
        else:
            assert row["unit"] == "mole_fraction"


def test_gas_composition_top_n(tmp_path) -> None:
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
        "params": {
            "observables": [
                {
                    "name": "gas_composition",
                    "params": {"top_n": 1, "rank_by": "mean", "stats": ["last"]},
                }
            ]
        },
    }
    obs_result = obs_task(obs_cfg, store=store)

    rows = _read_values(obs_result.path / "values.parquet")
    observables = {row["observable"] for row in rows}
    assert observables == {"gas.B.last"}
