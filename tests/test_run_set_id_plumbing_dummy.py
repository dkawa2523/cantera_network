from __future__ import annotations

import csv
from pathlib import Path

import pytest

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.features  # noqa: F401
import rxn_platform.tasks.gnn_dataset  # noqa: F401
import rxn_platform.tasks.graphs  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.errors import ConfigError
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_run_set_id_plumbing_across_tasks_dummy(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    store = ArtifactStore(tmp_path / "artifacts")
    conditions = tmp_path / "conds.csv"
    _write_csv(
        conditions,
        [
            {"case_id": "c0", "t_end": "1.0"},
            {"case_id": "c1", "t_end": "2.0"},
        ],
    )

    sweep = get("task", "sim.sweep_csv")
    run_set = sweep(
        {
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
        },
        store=store,
    )

    # graphs.temporal_flux consumes run_set_id and must not raise time-grid mismatch.
    graph_task = get("task", "graphs.temporal_flux")
    graph = graph_task(
        {
            "inputs": {
                "run_set_id": run_set.manifest.id,
            },
            "params": {
                "windowing": {"type": "fixed", "count": 2},
                "rop": {"var": "rop_net", "use_abs": True},
                "reaction_map": {"R1": ["A"], "R2": ["B"]},
            },
        },
        store=store,
    )
    assert graph.manifest.kind == "graphs"

    # gnn_dataset.temporal_graph_pyg consumes run_set_id.
    gnn_task = get("task", "gnn_dataset.temporal_graph_pyg")
    dataset = gnn_task(
        {
            "inputs": {"run_set_id": run_set.manifest.id, "graph_id": graph.manifest.id},
            "params": {"node_features": ["X"], "split": {"train_ratio": 0.8, "val_ratio": 0.2}},
        },
        store=store,
    )
    assert dataset.manifest.kind == "gnn_datasets"

    # features.gnn_importance consumes run_set_id.
    imp_task = get("task", "features.gnn_importance")
    importance = imp_task(
        {
            "inputs": {
                "run_set_id": run_set.manifest.id,
                "graph_id": graph.manifest.id,
                "dataset_id": dataset.manifest.id,
            },
            "params": {"output": {"include_reactions": True, "include_species": False}},
        },
        store=store,
    )
    assert importance.manifest.kind == "features"

    # observables.run consumes run_set_id.
    obs_task = get("task", "observables.run")
    obs = obs_task(
        {"inputs": {"run_set_id": run_set.manifest.id}, "params": {"observables": [{"name": "ignition_delay", "params": {}}]}},
        store=store,
    )
    assert obs.manifest.kind == "observables"

    # Conflict: specifying run_set_id and run_ids together must be rejected.
    with pytest.raises(ConfigError):
        graph_task(
            {"inputs": {"run_set_id": run_set.manifest.id, "run_ids": ["x"]}, "params": {}},
            store=store,
        )
