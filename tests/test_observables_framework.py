from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401

from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable, RunDatasetView


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


class MeanTemperatureObservable(Observable):
    name = "test.mean_T"
    requires = ("T",)

    def compute(self, run_dataset: RunDatasetView, cfg: dict[str, Any]) -> dict[str, Any]:
        series = run_dataset.data_vars.get("T", {}).get("data", [])
        values = [float(value) for value in series]
        mean = sum(values) / float(len(values))
        return {"value": mean, "unit": "K", "meta": {"method": "mean"}}


class MissingObservable(Observable):
    name = "test.missing"
    requires = ("not_present",)

    def compute(self, run_dataset: RunDatasetView, cfg: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("missing observable compute should not be called")


def test_observables_run_handles_missing_inputs(tmp_path) -> None:
    register("observable", "test.mean_T", MeanTemperatureObservable(), overwrite=True)
    register("observable", "test.missing", MissingObservable(), overwrite=True)

    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 2.0, "steps": 3},
            "initial": {"T": 100.0},
            "ramp": {"T": 10.0},
            "species": ["A", "B"],
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    obs_task = get("task", "observables.run")
    obs_cfg = {
        "inputs": {"run_id": sim_result.manifest.id},
        "params": {"observables": ["test.mean_T", "test.missing"]},
    }
    obs_result = obs_task(obs_cfg, store=store)

    assert obs_result.manifest.kind == "observables"
    assert sim_result.manifest.id in obs_result.manifest.inputs.get("runs", [])

    values_path = obs_result.path / "values.parquet"
    rows = _read_values(values_path)
    assert rows

    by_name = {row["observable"]: row for row in rows}
    assert "test.mean_T" in by_name
    assert "test.missing" in by_name

    mean_row = by_name["test.mean_T"]
    assert mean_row["run_id"] == sim_result.manifest.id
    assert math.isclose(mean_row["value"], 110.0, rel_tol=0.0, abs_tol=1.0e-6)

    missing_row = by_name["test.missing"]
    assert missing_row["run_id"] == sim_result.manifest.id
    assert math.isnan(missing_row["value"])
    meta = json.loads(missing_row["meta_json"])
    assert meta.get("status") in {"missing_input", "skipped"}
    assert any("not_present" in item for item in meta.get("missing", []))
