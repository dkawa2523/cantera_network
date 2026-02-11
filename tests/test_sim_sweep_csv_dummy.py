from __future__ import annotations

import csv
import json
from pathlib import Path

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_sim_sweep_csv_creates_run_set_and_reuses(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    conditions = tmp_path / "conds.csv"
    _write_csv(
        conditions,
        [
            {"case_id": "c0", "t_end": "1.0"},
            {"case_id": "c1", "t_end": "2.0"},
        ],
    )

    task = get("task", "sim.sweep_csv")
    cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 0.5, "steps": 3},
            "species": ["A", "B"],
            "reactions": ["R1", "R2"],
            "outputs": {"include_rop": True},
        },
        "params": {
            "conditions_file": str(conditions),
            "case_mode": "all",
            "time_grid_policy": "max",
        },
    }

    result1 = task(cfg, store=store)
    assert result1.manifest.kind == "run_sets"
    assert (result1.path / "runs.json").exists()

    payload = json.loads((result1.path / "runs.json").read_text(encoding="utf-8"))
    assert payload["case_ids"] == ["c0", "c1"]
    assert len(payload["run_ids"]) == 2
    assert payload["time_grid_policy"] == "max"

    # Ensure runs exist.
    for run_id in payload["run_ids"]:
        assert store.exists("runs", run_id)

    # Second run should reuse the same RunSet without backend calls.
    result2 = task(cfg, store=store)
    assert result2.manifest.id == result1.manifest.id
    assert result2.reused is True

