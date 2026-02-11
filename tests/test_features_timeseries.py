from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.features  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _read_features(path) -> list[dict[str, Any]]:
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


def test_timeseries_summary_features_with_missing_variable(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 2.0, "steps": 3},
            "initial": {"T": 100.0, "P": 1000.0},
            "ramp": {"T": 10.0, "P": 100.0},
            "species": ["A", "B"],
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    feat_task = get("task", "features.run")
    feat_cfg = {
        "inputs": {"run_id": sim_result.manifest.id},
        "params": {
            "features": [
                {
                    "name": "timeseries_summary",
                    "params": {
                        "variables": [
                            "T",
                            {"name": "X", "species": ["A"]},
                            "missing_var",
                        ],
                        "stats": ["mean", "max", "min", "last", "integral"],
                    },
                }
            ]
        },
    }
    feat_result = feat_task(feat_cfg, store=store)

    assert feat_result.manifest.kind == "features"
    assert sim_result.manifest.id in feat_result.manifest.inputs.get("runs", [])

    features_path = feat_result.path / "features.parquet"
    rows = _read_features(features_path)
    assert rows

    by_name = {row["feature"]: row for row in rows}
    assert "T.mean" in by_name
    assert "T.integral" in by_name
    assert "X.A.mean" in by_name
    assert "missing_var.mean" in by_name

    mean_row = by_name["T.mean"]
    assert math.isclose(mean_row["value"], 110.0, rel_tol=0.0, abs_tol=1.0e-6)

    integral_row = by_name["T.integral"]
    assert math.isclose(integral_row["value"], 220.0, rel_tol=0.0, abs_tol=1.0e-6)

    missing_row = by_name["missing_var.mean"]
    assert math.isnan(missing_row["value"])
    meta = json.loads(missing_row["meta_json"])
    assert meta.get("status") == "missing_variable"
    assert any("data_vars.missing_var" == item for item in meta.get("missing", []))
