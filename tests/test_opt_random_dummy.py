from __future__ import annotations

import json
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.optimization  # noqa: F401

from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


class DummyMultiplierObservable(Observable):
    name = "dummy_multiplier_opt"

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


def _read_history(path) -> list[dict[str, Any]]:
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


def test_opt_random_dummy(tmp_path) -> None:
    register(
        "observable",
        DummyMultiplierObservable.name,
        DummyMultiplierObservable(),
        overwrite=True,
    )
    task = get("task", "optimization.random_search")

    samples = 4
    cfg = {
        "common": {"seed": 123},
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "observables": {
            "observables": [DummyMultiplierObservable.name],
        },
        "params": {
            "samples": samples,
            "objective": {"target": "dummy.multiplier_sum", "direction": "max"},
            "search_space": {
                "multipliers": [
                    {"index": 0, "low": 0.5, "high": 1.5},
                    {"index": 1, "low": 0.5, "high": 1.5},
                ]
            },
        },
    }

    store_a = ArtifactStore(tmp_path / "artifacts_a")
    store_b = ArtifactStore(tmp_path / "artifacts_b")

    result_a = task(cfg, store=store_a)
    result_b = task(cfg, store=store_b)

    assert result_a.manifest.kind == "optimization"
    assert result_b.manifest.kind == "optimization"

    rows_a = _read_history(result_a.path / "history.parquet")
    rows_b = _read_history(result_b.path / "history.parquet")

    assert len(rows_a) == samples
    assert len(rows_b) == samples

    rows_a = sorted(rows_a, key=lambda row: row.get("sample_id", 0))
    rows_b = sorted(rows_b, key=lambda row: row.get("sample_id", 0))
    assert rows_a == rows_b

    parsed = json.loads(rows_a[0]["params_json"])
    assert "multipliers" in parsed
    assert rows_a[0]["objective_name"] == "dummy.multiplier_sum"
