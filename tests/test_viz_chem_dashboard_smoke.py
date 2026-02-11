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
        created_at="2026-01-18T00:00:00Z",
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
            "species": {"dims": ["species"], "data": ["A", "B"]},
        },
        "data_vars": {
            "X": {
                "dims": ["time", "species"],
                "data": [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
            },
            "net_production_rates": {
                "dims": ["time", "species"],
                "data": [[0.01, -0.02], [0.02, -0.01], [0.03, 0.0]],
            },
        },
        "attrs": {"units": {"time": "s", "X": "mole_fraction"}},
    }
    dataset_dir = base_dir / "state.zarr"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_viz_chem_dashboard_smoke(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    run_manifest = _make_manifest("runs", "run-1")
    store.ensure(run_manifest, writer=_write_run_dataset)

    cfg = {
        "viz": {
            "name": "chem_dashboard",
            "title": "Chem Dashboard Smoke",
            "dashboard": "chem",
            "inputs": [{"kind": "runs", "id": run_manifest.id}],
            "species": {"top_n": 2},
        }
    }

    result = viz_task.chem_dashboard(cfg, store=store)

    html = (result.path / "index.html").read_text(encoding="utf-8")
    assert "Species Time Series" in html
    assert "Rate-of-Production Ranking" in html
    assert "No ROP data available." in html
    assert "Mechanism Networks (Graphviz)" in html
    assert "No graphviz networks available." in html
