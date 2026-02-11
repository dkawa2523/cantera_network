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


def test_rop_wdot_summary_features_with_outputs(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 2.0, "steps": 3},
            "initial": {"T": 100.0, "P": 1000.0},
            "ramp": {"T": 10.0, "P": 100.0},
            "species": ["A", "B", "C"],
            "reactions": ["R1", "R2", "R3"],
            "outputs": {"rop": True, "wdot": True},
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    feat_task = get("task", "features.run")
    feat_cfg = {
        "inputs": {"run_id": sim_result.manifest.id},
        "params": {
            "features": [
                {"name": "rop_wdot_summary", "params": {"top_n": 2}},
            ]
        },
    }
    feat_result = feat_task(feat_cfg, store=store)

    rows = _read_features(feat_result.path / "features.parquet")
    by_name = {row["feature"]: row for row in rows}

    assert "rop_net.R3.integral" in by_name
    assert "rop_net.R3.max" in by_name
    assert "net_production_rates.C.integral" in by_name
    assert "net_production_rates.C.max" in by_name

    rop_integral = by_name["rop_net.R3.integral"]["value"]
    assert math.isclose(rop_integral, 0.12, rel_tol=0.0, abs_tol=1.0e-6)

    wdot_max = by_name["net_production_rates.C.max"]["value"]
    assert math.isclose(wdot_max, 0.009, rel_tol=0.0, abs_tol=1.0e-9)


def test_rop_wdot_summary_features_missing_optional(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
            "initial": {"T": 100.0, "P": 1000.0},
            "ramp": {"T": 0.0, "P": 0.0},
            "species": ["A", "B"],
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    feat_task = get("task", "features.run")
    feat_cfg = {
        "inputs": {"run_id": sim_result.manifest.id},
        "params": {
            "features": [
                {"name": "rop_wdot_summary", "params": {"top_n": 1}},
            ]
        },
    }
    feat_result = feat_task(feat_cfg, store=store)

    rows = _read_features(feat_result.path / "features.parquet")
    by_name = {row["feature"]: row for row in rows}

    assert "rop_net.integral" in by_name
    assert math.isnan(by_name["rop_net.integral"]["value"])
    meta = json.loads(by_name["rop_net.integral"]["meta_json"])
    assert meta.get("status") == "missing_variable"

    assert "net_production_rates.A.integral" in by_name
    assert math.isnan(by_name["net_production_rates.A.integral"]["value"])
