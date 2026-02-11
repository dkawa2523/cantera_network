from __future__ import annotations

import json
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.reduction  # noqa: F401

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
            multiplier = entry.get("multiplier", 0.0)
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError):
                multiplier_value = 0.0
            total += multiplier_value
        return {
            "observable": "dummy.multiplier_sum",
            "value": total,
            "unit": "1",
        }


def _read_metrics(path) -> list[dict[str, Any]]:
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


def test_reduction_validation_loop_dummy(tmp_path) -> None:
    register(
        "observable",
        DummyMultiplierObservable.name,
        DummyMultiplierObservable(),
        overwrite=True,
    )
    store = ArtifactStore(tmp_path / "artifacts")

    mechanism_payload = {
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "species": ["A", "B"],
                "reactions": "all",
            }
        ],
        "species": [{"name": "A"}, {"name": "B"}],
        "reactions": [
            {"id": "R1", "equation": "A => B", "rate-constant": {"A": 1.0}},
        ],
    }
    mechanism_path = tmp_path / "mechanism.yaml"
    mechanism_path.write_text(
        json.dumps(mechanism_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    pipeline_cfg = {
        "steps": [
            {
                "id": "sim",
                "task": "sim.run",
                "sim": {"name": "dummy", "time": {"start": 0.0, "stop": 1.0, "steps": 2}},
            },
            {
                "id": "obs",
                "task": "observables.run",
                "inputs": {"runs": "@sim"},
                "params": {"observables": ["dummy_multiplier"]},
            },
        ]
    }

    patch_small = {
        "schema_version": 1,
        "reaction_multipliers": [{"reaction_id": "R1", "multiplier": 0.05}],
    }
    patch_large = {
        "schema_version": 1,
        "reaction_multipliers": [{"reaction_id": "R1", "multiplier": 0.5}],
    }

    task = get("task", "reduction.validate")
    cfg = {
        "reduction": {
            "mechanism": str(mechanism_path),
            "validation": {
                "pipeline": pipeline_cfg,
                "patches": [patch_small, patch_large],
                "metric": "abs",
                "tolerance": 0.1,
                "sim_step_id": "sim",
                "observables_step_id": "obs",
            },
        }
    }

    result = task(cfg, store=store)

    assert result.manifest.kind == "validation"
    assert result.manifest.inputs["passed"] is True
    assert result.manifest.inputs["selected_patch"]["patch_index"] == 0

    rows = _read_metrics(result.path / "metrics.parquet")
    assert rows

    by_patch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "observable":
            continue
        patch_index = row.get("patch_index")
        if patch_index is None:
            continue
        by_patch.setdefault(int(patch_index), []).append(row)

    assert by_patch[0]
    assert by_patch[1]
    assert all(row.get("passed") for row in by_patch[0])
    assert any(not row.get("passed") for row in by_patch[1])
