from __future__ import annotations

import json
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.optimization  # noqa: F401

from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


def _read_table(path) -> list[dict[str, Any]]:
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


def _multiplier_value(entries: list[dict[str, Any]], index: int) -> float:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("index") != index:
            continue
        try:
            return float(entry.get("multiplier", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


class DummyObjectiveMax(Observable):
    name = "dummy.obj_max"

    def compute(self, run_dataset, cfg):
        multipliers = run_dataset.attrs.get("reaction_multipliers") or []
        value = _multiplier_value(list(multipliers), 0)
        return {"observable": "dummy.obj_max", "value": value, "unit": "1"}


class DummyObjectiveMin(Observable):
    name = "dummy.obj_min"

    def compute(self, run_dataset, cfg):
        multipliers = run_dataset.attrs.get("reaction_multipliers") or []
        value = _multiplier_value(list(multipliers), 1)
        return {"observable": "dummy.obj_min", "value": value, "unit": "1"}


def _dominates(
    candidate: list[float],
    other: list[float],
    directions: list[str],
) -> bool:
    strictly_better = False
    for value, other_value, direction in zip(candidate, other, directions):
        if direction == "min":
            if value > other_value:
                return False
            if value < other_value:
                strictly_better = True
        else:
            if value < other_value:
                return False
            if value > other_value:
                strictly_better = True
    return strictly_better


def _pareto_front_ids(
    samples: list[tuple[int, list[float]]],
    directions: list[str],
) -> list[int]:
    front: list[int] = []
    for sample_id, values in samples:
        dominated = False
        for other_id, other_values in samples:
            if other_id == sample_id:
                continue
            if _dominates(other_values, values, directions):
                dominated = True
                break
        if not dominated:
            front.append(sample_id)
    return front


def test_opt_pareto_archive(tmp_path) -> None:
    register(
        "observable",
        DummyObjectiveMax.name,
        DummyObjectiveMax(),
        overwrite=True,
    )
    register(
        "observable",
        DummyObjectiveMin.name,
        DummyObjectiveMin(),
        overwrite=True,
    )
    task = get("task", "optimization.random_search")

    samples = 6
    cfg = {
        "common": {"seed": 11},
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "observables": {
            "observables": [DummyObjectiveMax.name, DummyObjectiveMin.name],
        },
        "params": {
            "samples": samples,
            "objectives": [
                {"target": "dummy.obj_max", "direction": "max"},
                {"target": "dummy.obj_min", "direction": "min"},
            ],
            "search_space": {
                "multipliers": [
                    {"index": 0, "low": 0.0, "high": 1.0},
                    {"index": 1, "low": 0.0, "high": 1.0},
                ]
            },
        },
    }

    store = ArtifactStore(tmp_path / "artifacts")
    result = task(cfg, store=store)

    history_rows = _read_table(result.path / "history.parquet")
    pareto_rows = _read_table(result.path / "pareto.parquet")

    objective_order = ["dummy.obj_max", "dummy.obj_min"]
    objective_names = {row.get("objective_name") for row in history_rows}
    assert objective_names == set(objective_order)

    directions = {}
    samples_by_id: dict[int, dict[str, float]] = {}
    for row in history_rows:
        sample_id = int(row["sample_id"])
        objective_name = row["objective_name"]
        samples_by_id.setdefault(sample_id, {})[objective_name] = float(row["objective"])
        directions[objective_name] = row["direction"]

    samples_list: list[tuple[int, list[float]]] = []
    for sample_id, values in samples_by_id.items():
        assert set(values.keys()) == set(objective_order)
        vector = [values[name] for name in objective_order]
        samples_list.append((sample_id, vector))

    expected_front = _pareto_front_ids(
        samples_list,
        [directions[name] for name in objective_order],
    )
    pareto_sample_ids = {int(row["sample_id"]) for row in pareto_rows}

    assert pareto_sample_ids == set(expected_front)
    assert len(pareto_rows) == len(expected_front) * len(objective_order)
