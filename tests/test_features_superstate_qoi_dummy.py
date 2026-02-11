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


def _ensure_dummy_run(store: ArtifactStore, run_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id=run_id,
        created_at="2026-02-08T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        # Use legacy layout for minimal tests: artifacts/runs/<id>/state.zarr/dataset.json
        dataset_dir = base_dir / "state.zarr"
        payload = {
            "coords": {
                "time": {"dims": ["time"], "data": [0.0, 1.0, 2.0]},
                "species": {"dims": ["species"], "data": ["CO", "CO2", "H2"]},
            },
            "data_vars": {
                "X": {
                    "dims": ["time", "species"],
                    "data": [
                        [0.10, 0.20, 0.70],
                        [0.20, 0.20, 0.60],
                        [0.30, 0.10, 0.60],
                    ],
                },
                "T": {"dims": ["time"], "data": [1000.0, 1100.0, 1500.0]},
            },
            "attrs": {"units": {"time": "s", "X": "mole_fraction", "T": "K"}},
        }
        _write_json(dataset_dir / "dataset.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_mapping_json(store: ArtifactStore, mapping_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=mapping_id,
        created_at="2026-02-08T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "schema_version": 1,
            "kind": "superstate_mapping",
            "clusters": [
                {"superstate_id": 0, "name": "S000", "members": ["CO", "H2"]},
                {"superstate_id": 1, "name": "S001", "members": ["CO2"]},
            ],
            "mapping": [
                {"species": "CO", "superstate_id": 0},
                {"species": "H2", "superstate_id": 0},
                {"species": "CO2", "superstate_id": 1},
            ],
        }
        _write_json(base_dir / "mapping.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_node_lumping_json(store: ArtifactStore, mapping_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=mapping_id,
        created_at="2026-02-08T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "schema_version": 1,
            "kind": "node_lumping",
            "clusters": [
                {"cluster_id": 0, "members": ["CO", "H2"], "representative": "CO"},
                {"cluster_id": 1, "members": ["CO2"], "representative": "CO2"},
            ],
            "mapping": [
                {"species": "CO", "cluster_id": 0},
                {"species": "H2", "cluster_id": 0},
                {"species": "CO2", "cluster_id": 1},
            ],
        }
        _write_json(base_dir / "node_lumping.json", payload)

    store.ensure(manifest, writer=_writer)


def test_superstate_qoi_from_mapping_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = "run0001"
    mapping_id = "map0001"
    _ensure_dummy_run(store, run_id)
    _ensure_mapping_json(store, mapping_id)

    task = get("task", "features.superstate_qoi")
    result = task(
        {"inputs": {"run_id": run_id, "mapping_id": mapping_id}},
        store=store,
    )

    rows = _read_rows(result.path / "features.parquet")
    by_name = {row["feature"]: row for row in rows if row.get("run_id") == run_id}

    # CO cluster contains CO + H2. Last: 0.30 + 0.60 = 0.90.
    assert abs(by_name["qoi.CO_final_super"]["value"] - 0.90) < 1.0e-12
    assert abs(by_name["qoi.CO2_final_super"]["value"] - 0.10) < 1.0e-12
    assert abs(by_name["qoi.T_peak"]["value"] - 1500.0) < 1.0e-12
    # max dT/dt occurs between t=1 and t=2 (400 K / 1 s), timestamp is t=2.
    assert abs(by_name["qoi.ignition_delay"]["value"] - 2.0) < 1.0e-12


def test_superstate_qoi_from_node_lumping_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_id = "run0002"
    mapping_id = "nl0001"
    _ensure_dummy_run(store, run_id)
    _ensure_node_lumping_json(store, mapping_id)

    task = get("task", "features.superstate_qoi")
    result = task(
        {"inputs": {"run_id": run_id, "mapping_id": mapping_id}},
        store=store,
    )

    rows = _read_rows(result.path / "features.parquet")
    by_name = {row["feature"]: row for row in rows if row.get("run_id") == run_id}
    assert abs(by_name["qoi.CO_final_super"]["value"] - 0.90) < 1.0e-12
    assert abs(by_name["qoi.CO2_final_super"]["value"] - 0.10) < 1.0e-12
