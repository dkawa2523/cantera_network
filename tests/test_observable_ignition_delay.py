from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rxn_platform.tasks.observables  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _make_manifest(kind: str, artifact_id: str) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-02-08T00:00:00Z",
        parents=[],
        inputs={},
        config={"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def _write_run_dataset(base_dir: Path) -> None:
    payload = {
        "coords": {"time": {"dims": ["time"], "data": [0.0, 1.0, 2.0, 3.0]}},
        "data_vars": {"T": {"dims": ["time"], "data": [300.0, 400.0, 1000.0, 1100.0]}},
        "attrs": {"units": {"time": "s", "T": "K"}},
    }
    dataset_dir = base_dir / "state.zarr"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def test_observables_ignition_delay_task(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_manifest = _make_manifest("runs", "run-ign")
    store.ensure(run_manifest, writer=_write_run_dataset)

    task = get("task", "observables.run")
    cfg = {
        "observables": {
            "inputs": {"runs": [run_manifest.id]},
            "params": {"observables": [{"name": "ignition_delay"}]},
        }
    }
    result = task(cfg, store=store)

    rows = _read_rows(result.path / "values.parquet")
    matches = [row for row in rows if row.get("observable") == "ignition_delay"]
    assert matches
    assert matches[0]["value"] == 2.0
