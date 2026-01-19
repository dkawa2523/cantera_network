from __future__ import annotations

import json
import math
from typing import Any

import rxn_platform.tasks.dimred  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _store_artifact(store: ArtifactStore, *, kind: str, artifact_id: str, files: dict) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-01T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )

    def _writer(base_dir):
        for name, payload in files.items():
            path = base_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    store.ensure(manifest, writer=_writer)


def _store_sensitivity(
    store: ArtifactStore,
    artifact_id: str,
    rows: list[dict[str, Any]],
) -> None:
    payload = {"rows": rows}
    _store_artifact(
        store,
        kind="sensitivity",
        artifact_id=artifact_id,
        files={"sensitivity.parquet": payload},
    )


def _read_subspace(path) -> list[dict[str, Any]]:
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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def test_active_subspace_dummy(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    _store_sensitivity(
        store,
        "sens1",
        [
            {
                "run_id": "run_a",
                "target": "t1",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 1.0,
            },
            {
                "run_id": "run_a",
                "target": "t1",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 1.0,
            },
            {
                "run_id": "run_b",
                "target": "t1",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 2.0,
            },
            {
                "run_id": "run_b",
                "target": "t1",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 2.0,
            },
        ],
    )

    task = get("task", "dimred.active_subspace")
    cfg = {
        "dimred": {
            "inputs": {"sensitivity": "sens1"},
            "params": {"k": 1},
        }
    }
    result = task(cfg, store=store)

    assert result.manifest.kind == "subspaces"
    rows = _read_subspace(result.path / "subspace.parquet")
    assert rows

    component_ids = {_coerce_int(row.get("component"), -1) for row in rows}
    assert component_ids == {0}
    comp_rows = [row for row in rows if _coerce_int(row.get("component"), -1) == 0]
    assert len(comp_rows) == 2

    by_param = {row.get("param_id"): row for row in comp_rows}
    assert "R1" in by_param and "R2" in by_param
    load_r1 = float(by_param["R1"]["loading"])
    load_r2 = float(by_param["R2"]["loading"])
    ratio = abs(load_r1 / load_r2) if load_r2 != 0.0 else math.inf
    assert 0.8 <= ratio <= 1.25
