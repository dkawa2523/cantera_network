from __future__ import annotations

import json
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.assimilation  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401

from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


class DummyMultiplierObservable(Observable):
    name = "dummy_multiplier_esmda"

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
            try:
                multiplier_value = float(entry.get("multiplier", 0.0))
            except (TypeError, ValueError):
                multiplier_value = 0.0
            total += multiplier_value * weight
        return {
            "observable": "dummy.multiplier_sum",
            "value": total,
            "unit": "1",
        }


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


def test_assim_esmda_dummy(tmp_path) -> None:
    register(
        "observable",
        DummyMultiplierObservable.name,
        DummyMultiplierObservable(),
        overwrite=True,
    )
    task = get("task", "assimilation.esmda")

    iterations = 2
    ensemble_size = 4
    alpha_schedule = [2.0, 2.0]
    cfg = {
        "common": {"seed": 123},
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "observables": {"observables": [DummyMultiplierObservable.name]},
        "assimilation": {
            "parameters": [
                {
                    "index": 0,
                    "name": "m0",
                    "prior": {"type": "uniform", "low": 0.5, "high": 1.5},
                },
                {
                    "index": 1,
                    "name": "m1",
                    "prior": {"type": "uniform", "low": 0.5, "high": 1.5},
                },
            ],
            "observed": [
                {
                    "observable": "dummy.multiplier_sum",
                    "value": 2.0,
                    "weight": 1.0,
                    "noise": 1.0,
                }
            ],
            "params": {
                "ensemble_size": ensemble_size,
                "iterations": iterations,
                "alpha_schedule": alpha_schedule,
                "ridge": 1.0e-6,
            },
        },
    }

    store = ArtifactStore(tmp_path / "artifacts")
    result = task(cfg, store=store)

    assert result.manifest.kind == "assimilation"
    assert result.manifest.inputs.get("alpha_schedule") == alpha_schedule

    posterior_rows = _read_table(result.path / "posterior.parquet")
    misfit_rows = _read_table(result.path / "misfit_history.parquet")

    assert len(misfit_rows) == iterations
    assert {row.get("iteration") for row in misfit_rows} == {0, 1}

    forecast_rows = [
        row for row in posterior_rows if row.get("stage") == "forecast"
    ]
    analysis_rows = [
        row for row in posterior_rows if row.get("stage") == "analysis"
    ]

    assert len(forecast_rows) == iterations * ensemble_size
    assert len(analysis_rows) == iterations * ensemble_size

    parsed = json.loads(forecast_rows[0]["params_json"])
    assert set(parsed.keys()) == {"m0", "m1"}
