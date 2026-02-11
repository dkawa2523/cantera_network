from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.sensitivity  # noqa: F401

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


def test_sensitivity_multiplier_fd_dummy(tmp_path) -> None:
    register("observable", DummyMultiplierObservable.name, DummyMultiplierObservable(), overwrite=True)
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sensitivity.multiplier_fd")

    eps = 0.1
    cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "params": {
            "reactions": [0, 1],
            "observables": ["dummy_multiplier"],
            "targets": ["dummy.multiplier_sum"],
            "eps": eps,
            "definition": "dlogk",
            "rank_by": "abs",
        },
    }

    result = task(cfg, store=store)

    assert result.manifest.kind == "sensitivity"

    table_path = result.path / "sensitivity.parquet"
    rows = _read_sensitivity(table_path)
    assert rows
    rows = [row for row in rows if row.get("target") == "dummy.multiplier_sum"]
    assert len(rows) == 2

    by_index = {}
    for row in rows:
        index_value = row.get("reaction_index")
        if index_value is None:
            continue
        by_index[int(index_value)] = row

    expected_scale = math.log1p(eps)
    expected_0 = (1.0 + eps) / expected_scale
    expected_1 = (1.0 + eps) * 2.0 / expected_scale

    assert math.isclose(
        by_index[0]["value"], expected_0, rel_tol=0.0, abs_tol=1.0e-6
    )
    assert math.isclose(
        by_index[1]["value"], expected_1, rel_tol=0.0, abs_tol=1.0e-6
    )
    assert int(by_index[1]["rank"]) == 1
