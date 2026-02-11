from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rxn_platform.tasks.features  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
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


def _ensure_temporal_graph(store: ArtifactStore, graph_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=graph_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "kind": "temporal_flux",
            "source": {"run_ids": ["run0"]},
            "reactions": {"count": 3, "order": ["R1", "R2", "R3"]},
            "reaction_stats": {
                "activity": {"type": "abs_rop_sum", "values": [10.0, 5.0, 1.0], "total": 16.0}
            },
        }
        _write_json(base_dir / "graph.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_patch(store: ArtifactStore, reduction_id: str, multipliers: list[dict[str, Any]]) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=reduction_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "schema_version": 1,
            "disabled_reactions": [],
            "reaction_multipliers": multipliers,
        }
        _write_yaml(base_dir / "mechanism_patch.yaml", payload)

    store.ensure(manifest, writer=_writer)


def test_reduction_cheap_metrics_retained_and_jaccard(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    graph_id = "g0001"
    red_a = "r0001"
    red_b = "r0002"
    _ensure_temporal_graph(store, graph_id)
    _ensure_patch(store, red_a, [{"index": 0, "multiplier": 0.0}])
    _ensure_patch(store, red_b, [{"index": 1, "multiplier": 0.5}])

    task = get("task", "features.reduction_cheap_metrics")
    result = task(
        {"inputs": {"graph_flux_id": graph_id, "patches": [red_a, red_b]}, "params": {"top_k": 2}},
        store=store,
    )

    rows = _read_rows(result.path / "features.parquet")
    by_reduction: dict[str, dict[str, float]] = {}
    for row in rows:
        meta = json.loads(row.get("meta_json") or "{}")
        rid = meta.get("reduction_id")
        if not isinstance(rid, str):
            continue
        by_reduction.setdefault(rid, {})[row["feature"]] = float(row["value"])

    assert abs(by_reduction[red_a]["cheap.flux_coverage_retained"] - (6.0 / 16.0)) < 1.0e-12
    assert abs(by_reduction[red_a]["cheap.top_path_jaccard"] - (1.0 / 3.0)) < 1.0e-12
    assert abs(by_reduction[red_b]["cheap.flux_coverage_retained"] - (13.5 / 16.0)) < 1.0e-12
    assert abs(by_reduction[red_b]["cheap.top_path_jaccard"] - 1.0) < 1.0e-12

