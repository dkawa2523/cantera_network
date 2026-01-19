from __future__ import annotations

import json
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.optimization  # noqa: F401

from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


class DummyMultiFidelityObservable(Observable):
    name = "dummy_multi_fidelity"

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
            "observable": "dummy.multi_fidelity_sum",
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


def test_multi_fidelity_opt_dummy(tmp_path) -> None:
    register(
        "observable",
        DummyMultiFidelityObservable.name,
        DummyMultiFidelityObservable(),
        overwrite=True,
    )
    task = get("task", "optimization.multi_fidelity")

    samples = 5
    high_samples = 2
    cfg = {
        "common": {"seed": 7},
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "observables": {
            "observables": [DummyMultiFidelityObservable.name],
        },
        "params": {
            "samples": samples,
            "objective": {"target": "dummy.multi_fidelity_sum", "direction": "max"},
            "search_space": {
                "multipliers": [
                    {"index": 0, "low": 0.5, "high": 1.5},
                    {"index": 1, "low": 0.5, "high": 1.5},
                ]
            },
            "multi_fidelity": {
                "high_fidelity_samples": high_samples,
                "patch": {
                    "schema_version": 1,
                    "reaction_multipliers": [
                        {"index": 1, "multiplier": 0.25}
                    ],
                },
            },
        },
    }

    store = ArtifactStore(tmp_path / "artifacts")
    result = task(cfg, store=store)

    rows = _read_history(result.path / "history.parquet")
    low_rows = [row for row in rows if row.get("fidelity") == "low"]
    high_rows = [row for row in rows if row.get("fidelity") == "high"]

    assert len(low_rows) == samples
    assert len(high_rows) == high_samples

    summary = json.loads(
        (result.path / "fidelity_summary.json").read_text(encoding="utf-8")
    )
    assert summary["low_fidelity_samples"] == samples
    assert summary["high_fidelity_samples"] == high_samples
