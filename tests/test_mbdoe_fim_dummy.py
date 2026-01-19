from __future__ import annotations

import json
from typing import Any

import rxn_platform.tasks.doe  # noqa: F401

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


def _read_design(path) -> list[dict[str, Any]]:
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


def test_mbdoe_fim_ranking_dummy(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    _store_sensitivity(
        store,
        "sens1",
        [
            {
                "run_id": "cond_a",
                "condition_id": "cond_a",
                "target": "t1",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 1.0,
            },
            {
                "run_id": "cond_a",
                "condition_id": "cond_a",
                "target": "t1",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 0.0,
            },
            {
                "run_id": "cond_a",
                "condition_id": "cond_a",
                "target": "t2",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 0.0,
            },
            {
                "run_id": "cond_a",
                "condition_id": "cond_a",
                "target": "t2",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 1.0,
            },
            {
                "run_id": "cond_b",
                "condition_id": "cond_b",
                "target": "t1",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 0.5,
            },
            {
                "run_id": "cond_b",
                "condition_id": "cond_b",
                "target": "t1",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 0.0,
            },
            {
                "run_id": "cond_b",
                "condition_id": "cond_b",
                "target": "t2",
                "reaction_id": "R1",
                "reaction_index": 0,
                "value": 0.0,
            },
            {
                "run_id": "cond_b",
                "condition_id": "cond_b",
                "target": "t2",
                "reaction_id": "R2",
                "reaction_index": 1,
                "value": 0.5,
            },
        ],
    )

    task = get("task", "doe.fim_rank")
    cfg = {
        "doe": {
            "inputs": {"sensitivity": "sens1"},
            "params": {
                "metric": "d_opt",
                "targets": ["t1", "t2"],
            },
        }
    }
    result = task(cfg, store=store)

    assert result.manifest.kind == "designs"
    rows = _read_design(result.path / "design.parquet")
    assert len(rows) == 2

    ranked = sorted(rows, key=lambda row: row.get("rank", 0))
    assert ranked[0]["condition_id"] == "cond_a"
    assert ranked[0]["score"] > ranked[1]["score"]
