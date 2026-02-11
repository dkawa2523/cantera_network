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
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _ensure_dummy_run(store: ArtifactStore, run_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id=run_id,
        created_at="2026-02-10T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        dataset_dir = base_dir / "state.zarr"
        payload = {
            "coords": {
                "time": {"dims": ["time"], "data": [0.0, 1.0]},
                "species": {"dims": ["species"], "data": ["A", "B"]},
            },
            "data_vars": {
                "X": {
                    "dims": ["time", "species"],
                    "data": [
                        [0.2, 0.8],
                        [0.2, 0.8],
                    ],
                }
            },
            "attrs": {"units": {"time": "s", "X": "mole_fraction"}},
        }
        _write_json(dataset_dir / "dataset.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_identity_mapping(store: ArtifactStore, mapping_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=mapping_id,
        created_at="2026-02-10T00:00:00Z",
        parents=[],
        inputs={"mode": "superstate_mapping"},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        # Identity mapping where clusters include members and mapping repeats them.
        # Previously, _resolve_superstate_mapping would double-count members, causing
        # purity to drop from 1.0 to 0.5 for singleton clusters.
        payload = {
            "schema_version": 1,
            "kind": "superstate_mapping",
            "clusters": [
                {"superstate_id": 0, "name": "S000", "members": ["A"]},
                {"superstate_id": 1, "name": "S001", "members": ["B"]},
            ],
            "mapping": [
                {"species": "A", "superstate_id": 0},
                {"species": "B", "superstate_id": 1},
            ],
        }
        _write_json(base_dir / "mapping.json", payload)

    store.ensure(manifest, writer=_writer)


def test_superstate_merge_quality_dedups_cluster_members(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = "run0001"
    mapping_id = "map0001"
    _ensure_dummy_run(store, run_id)
    _ensure_identity_mapping(store, mapping_id)

    task = get("task", "features.superstate_merge_quality")
    result = task(
        {"inputs": {"run_id": run_id, "mapping_id": mapping_id}},
        store=store,
    )

    rows = _read_rows(result.path / "features.parquet")
    by_name = {row["feature"]: row for row in rows if row.get("run_id") == run_id}
    assert abs(by_name["merge.cluster_purity.mean"]["value"] - 1.0) < 1.0e-12

