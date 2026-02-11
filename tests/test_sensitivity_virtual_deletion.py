from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.sensitivity  # noqa: F401

from rxn_platform.core import make_run_id
from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


class DummyMultiplierObservable(Observable):
    name = "dummy_multiplier"

    def compute(self, run_dataset, cfg):
        multipliers = run_dataset.attrs.get("reaction_multipliers") or []
        total = 0.0
        for entry in multipliers:
            if not isinstance(entry, dict):
                continue
            weight = 1.0
            if "index" in entry:
                try:
                    weight = float(entry["index"]) + 1.0
                except (TypeError, ValueError):
                    weight = 1.0
            multiplier = entry.get("multiplier", 0.0)
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError):
                multiplier_value = 0.0
            total += multiplier_value * weight
        return {
            "observable": "dummy.multiplier_sum",
            "value": total,
            "unit": "1",
        }


def _read_sensitivity(path) -> list[dict[str, Any]]:
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


def _base_run_id(sim_cfg: dict[str, Any]) -> str:
    manifest_cfg = {"sim": sim_cfg, "inputs": {}, "params": {}}
    return make_run_id(manifest_cfg, exclude_keys=("hydra",))


def test_sensitivity_virtual_deletion_stability(tmp_path) -> None:
    register("observable", DummyMultiplierObservable.name, DummyMultiplierObservable(), overwrite=True)
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sensitivity.multiplier_fd")

    sim_a = {
        "name": "dummy",
        "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        "reaction_multipliers": [
            {"index": 0, "multiplier": 1.0},
            {"index": 1, "multiplier": 2.0},
        ],
    }
    sim_b = {
        "name": "dummy",
        "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        "reaction_multipliers": [
            {"index": 0, "multiplier": 2.0},
            {"index": 1, "multiplier": 4.0},
        ],
    }

    cfg = {
        "sim": [sim_a, sim_b],
        "params": {
            "reactions": [0, 1],
            "observables": ["dummy_multiplier"],
            "targets": ["dummy.multiplier_sum"],
            "mode": "virtual_deletion",
            "rank_by": "abs",
        },
    }

    result = task(cfg, store=store)
    table_path = result.path / "sensitivity.parquet"
    rows = _read_sensitivity(table_path)
    rows = [row for row in rows if row.get("target") == "dummy.multiplier_sum"]
    assert len(rows) == 4

    run_a = _base_run_id(sim_a)
    run_b = _base_run_id(sim_b)

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        run_id = row.get("run_id")
        reaction_index = row.get("reaction_index")
        if not isinstance(run_id, str) or reaction_index is None:
            continue
        by_key[(run_id, int(reaction_index))] = row

    assert math.isclose(by_key[(run_a, 0)]["impact"], -1.0, abs_tol=1.0e-6)
    assert math.isclose(by_key[(run_a, 1)]["impact"], -4.0, abs_tol=1.0e-6)
    assert math.isclose(by_key[(run_b, 0)]["impact"], -2.0, abs_tol=1.0e-6)
    assert math.isclose(by_key[(run_b, 1)]["impact"], -8.0, abs_tol=1.0e-6)
    assert int(by_key[(run_a, 1)]["rank"]) == 1
    assert int(by_key[(run_b, 1)]["rank"]) == 1

    meta = json.loads(by_key[(run_a, 0)]["meta_json"])
    stability = meta.get("rank_stability", {})
    assert stability.get("status") == "computed"
    assert math.isclose(stability.get("spearman_mean"), 1.0, abs_tol=1.0e-6)
    assert math.isclose(stability.get("top_k_jaccard_mean"), 1.0, abs_tol=1.0e-6)
