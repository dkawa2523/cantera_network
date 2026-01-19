from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.gnn_dataset  # noqa: F401
import rxn_platform.tasks.graphs  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _read_node_features(path) -> list[dict[str, Any]]:
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


def test_gnn_dataset_export_dummy(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    sim_task = get("task", "sim.run")

    sim_cfg = {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 3},
            "species": ["A", "B"],
            "outputs": {"include_wdot": True},
        }
    }
    sim_result = sim_task(sim_cfg, store=store)

    graph_task = get("task", "graphs.from_run")
    graph_cfg = {"graphs": {"inputs": {"run_id": sim_result.manifest.id}}}
    graph_result = graph_task(graph_cfg, store=store)

    gnn_task = get("task", "gnn_dataset.export")
    gnn_cfg = {
        "gnn_dataset": {
            "inputs": {
                "run_id": sim_result.manifest.id,
                "graph_id": graph_result.manifest.id,
            },
            "params": {
                "node_features": [
                    "X",
                    {"name": "wdot", "data_var": "net_production_rates"},
                ]
            },
        }
    }
    result = gnn_task(gnn_cfg, store=store)

    assert result.manifest.kind == "gnn_datasets"
    meta = json.loads((result.path / "dataset.json").read_text(encoding="utf-8"))
    assert meta["source"]["run_ids"] == [sim_result.manifest.id]

    node_order = meta["nodes"]["order"]
    assert node_order
    assert any(node_id == "species_A" for node_id in node_order)

    node_meta = {entry["id"]: entry for entry in meta["nodes"]["meta"]}
    assert node_meta["species_A"]["species"] == "A"

    files_meta = meta.get("files", {})
    table_name = files_meta.get("node_features_table", "node_features.parquet")
    rows = _read_node_features(result.path / table_name)
    assert rows

    run_payload = json.loads(
        (sim_result.path / "state.zarr" / "dataset.json").read_text(encoding="utf-8")
    )
    expected_x = run_payload["data_vars"]["X"]["data"][0][0]
    expected_wdot = run_payload["data_vars"]["net_production_rates"]["data"][0][0]

    def _find_row(feature: str) -> dict[str, Any]:
        for row in rows:
            if (
                row.get("run_id") == sim_result.manifest.id
                and int(row.get("time_index", -1)) == 0
                and row.get("node_id") == "species_A"
                and row.get("feature") == feature
            ):
                return row
        raise AssertionError(f"Row not found for feature={feature}")

    row_x = _find_row("X")
    assert math.isclose(float(row_x["value"]), expected_x, rel_tol=0.0, abs_tol=1e-12)

    row_wdot = _find_row("wdot")
    assert math.isclose(
        float(row_wdot["value"]),
        expected_wdot,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
