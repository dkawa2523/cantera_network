from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks import viz as viz_task


def _make_manifest(
    kind: str,
    artifact_id: str,
    *,
    config: Optional[dict[str, Any]] = None,
    parents: Optional[list[str]] = None,
    inputs: Optional[dict[str, Any]] = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-02-08T00:00:00Z",
        parents=parents or [],
        inputs=inputs or {},
        config=config or {"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def _write_run_dataset(base_dir: Path) -> None:
    payload = {
        "coords": {
            "time": {"dims": ["time"], "data": [0.0, 1.0, 2.0]},
            "reaction": {"dims": ["reaction"], "data": ["R1", "R2"]},
            "species": {"dims": ["species"], "data": ["A", "B"]},
        },
        "data_vars": {
            "rop_net": {
                "dims": ["time", "reaction"],
                "data": [[1.0, -0.5], [0.8, -0.4], [0.6, -0.3]],
            },
            "net_production_rates": {
                "dims": ["time", "species"],
                "data": [[0.02, -0.02], [0.03, -0.03], [0.04, -0.04]],
            },
            "X": {
                "dims": ["time", "species"],
                "data": [[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]],
            },
        },
        "attrs": {"units": {"time": "s"}},
    }
    dataset_dir = base_dir / "state.zarr"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_graph_payload(base_dir: Path) -> None:
    payload = {
        "kind": "stoichiometric_matrix",
        "bipartite": {
            "data": {
                "nodes": [
                    {"id": "S:A", "kind": "species", "label": "A", "species": "A"},
                    {"id": "S:B", "kind": "species", "label": "B", "species": "B"},
                    {
                        "id": "R:R1",
                        "kind": "reaction",
                        "reaction_id": "R1",
                        "reaction_equation": "A => B",
                        "reaction_index": 0,
                        "label": "A => B",
                    },
                    {
                        "id": "R:R2",
                        "kind": "reaction",
                        "reaction_id": "R2",
                        "reaction_equation": "B => A",
                        "reaction_index": 1,
                        "label": "B => A",
                    },
                ],
                "links": [
                    {"source": "S:A", "target": "R:R1", "stoich": -1.0},
                    {"source": "R:R1", "target": "S:B", "stoich": 1.0},
                    {"source": "S:B", "target": "R:R2", "stoich": -1.0},
                    {"source": "R:R2", "target": "S:A", "stoich": 1.0},
                ],
            }
        },
    }
    (base_dir / "graph.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_reduction_patch(reaction_id: str) -> Any:
    return {
        "schema_version": 1,
        "disabled_reactions": [{"reaction_id": reaction_id}],
        "reaction_multipliers": [],
    }


def _write_validation_metrics(base_dir: Path, *, patch_ids: list[str]) -> None:
    rows = []
    for patch_index, patch_id in enumerate(patch_ids):
        # Each patch gets a few metrics rows.
        for k in range(3):
            rows.append(
                {
                    "patch_index": patch_index,
                    "patch_id": patch_id,
                    "status": "ok",
                    "passed": True if patch_index == 0 else (k < 2),
                    "abs_diff": 0.01 * float(k + 1 + patch_index),
                    "baseline_value": 1.0,
                    "reduced_value": 1.0 + 0.01 * float(k + 1 + patch_index),
                }
            )
    payload = {"columns": sorted({key for row in rows for key in row.keys()}), "rows": rows}
    (base_dir / "metrics.parquet").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_viz_benchmark_report_runstore_exports_smoke(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "exp1" / "demo_bench"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "demo_bench"}, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    store = ArtifactStore(run_root / "artifacts")

    run_manifest = _make_manifest("runs", "run-1")
    store.ensure(run_manifest, writer=_write_run_dataset)

    graph_manifest = _make_manifest("graphs", "graph-1")
    store.ensure(graph_manifest, writer=_write_graph_payload)

    reduction_ids = ["red-1", "red-2"]
    for idx, reduction_id in enumerate(reduction_ids):
        patch_payload = _write_reduction_patch("R1" if idx == 0 else "R2")
        red_manifest = _make_manifest(
            "reduction",
            reduction_id,
            inputs={"threshold": {"top_k": 250 - 50 * idx}},
        )

        def _writer(base_dir: Path, payload: dict[str, Any] = patch_payload) -> None:
            (base_dir / "mechanism_patch.yaml").write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        store.ensure(red_manifest, writer=_writer)

    val_manifest = _make_manifest(
        "validation",
        "val-1",
        inputs={
            "patches": [
                {"patch_index": 0, "reduction_id": reduction_ids[0]},
                {"patch_index": 1, "reduction_id": reduction_ids[1]},
            ],
            "selected_patch": {"patch_index": 1, "reduction_id": reduction_ids[1]},
        },
    )
    store.ensure(val_manifest, writer=lambda base_dir: _write_validation_metrics(base_dir, patch_ids=reduction_ids))

    cfg = {
        "viz": {
            "name": "benchmark_report_smoke",
            "title": "Benchmark Report RunStore Export Smoke",
            "dashboard": "benchmark",
            "chart_backend": "svg",
            "graphviz": {
                "top_n": 2,
                "max_nodes": 20,
                "max_edges": 40,
                "engine": "dot",
                "export_dot": True,
                "export_svg": False,
            },
            "inputs": {
                "runs": run_manifest.id,
                "graphs": [graph_manifest.id],
                "reduction": reduction_ids,
                "validation": [val_manifest.id],
            },
        }
    }

    viz_task.benchmark_report(cfg, store=store)

    network_dir = run_root / "viz" / "network"
    assert network_dir.exists()
    assert (network_dir / "index.json").exists()
    assert sorted(network_dir.glob("*.dot"))

    reduction_dir = run_root / "viz" / "reduction"
    assert reduction_dir.exists()
    assert (reduction_dir / "index.json").exists()
    assert sorted(reduction_dir.glob("*.svg"))

