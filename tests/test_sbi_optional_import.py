from __future__ import annotations

import json

import pytest

sbi = pytest.importorskip("sbi")

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.features  # noqa: F401
import rxn_platform.tasks.sbi  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def test_sbi_snpe_minimal(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sbi.run")

    cfg = {
        "seed": 7,
        "sbi": {
            "inputs": {
                "sim": {
                    "name": "dummy",
                    "time": {"start": 0.0, "stop": 1.0, "steps": 3},
                    "initial": {"T": 100.0, "P": 1000.0},
                    "ramp": {"T": 5.0, "P": 0.0},
                    "species": ["A", "B"],
                },
                "features": {
                    "features": [
                        {
                            "name": "timeseries_summary",
                            "params": {"variables": ["T"], "stats": ["mean"]},
                        }
                    ]
                },
            },
            "params": {
                "parameters": [
                    {"name": "multiplier_0", "index": 0, "low": 0.9, "high": 1.1}
                ],
                "num_simulations": 2,
                "max_epochs": 1,
                "posterior_samples": 2,
                "missing_strategy": "zero",
            },
        },
    }

    result = task(cfg, store=store)

    payload = json.loads(
        (result.path / "sbi_result.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "trained"
    assert payload["num_simulations"] == 2
    assert payload["num_parameters"] == 1
    assert payload["num_features"] == 1
    assert len(payload["posterior_samples"]) == 2
