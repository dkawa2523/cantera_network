from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _ensure_mapping(store: ArtifactStore, mapping_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=mapping_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={"mode": "superstate_mapping"},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "schema_version": 1,
            "kind": "superstate_mapping",
            "source": {"graph_id": "g0001"},
            "guards": {"protected_species": ["CO"]},
            "superstates": [
                {"superstate_id": 0, "name": "S000", "representative": "CO", "members": ["CO", "H2"]},
            ],
            "clusters": [
                {"superstate_id": 0, "name": "S000", "representative": "CO", "members": ["CO", "H2"]},
            ],
            "mapping": [
                {"species": "CO", "superstate_id": 0, "representative": "CO"},
                {"species": "H2", "superstate_id": 0, "representative": "CO"},
            ],
            "composition_meta": [
                {"species": "CO", "elements": {"C": 1, "O": 1}},
                {"species": "H2", "elements": {"H": 2}},
            ],
        }
        _write_json(base_dir / "mapping.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_merge_quality(store: ArtifactStore, features_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="features",
        id=features_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        rows = [
            {
                "run_id": "run0",
                "feature": "merge.superstate_purity",
                "value": 0.4,
                "unit": "",
                "meta_json": json.dumps(
                    {"superstate_id": 0, "superstate": "S000", "member_count": 2},
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        ]
        _write_json(base_dir / "features.parquet", {"rows": rows})

    store.ensure(manifest, writer=_writer)


def test_repair_mapping_split_protected_species_singleton(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    mapping_id = "map0001"
    features_id = "feat0001"
    _ensure_mapping(store, mapping_id)
    _ensure_merge_quality(store, features_id)

    task = get("task", "reduction.repair_mapping_split")
    result = task(
        {
            "inputs": {"mapping_id": mapping_id, "merge_quality_id": features_id},
            "params": {"policy": {"purity_min": 0.5}},
        },
        store=store,
    )

    repaired = json.loads((result.path / "mapping.json").read_text(encoding="utf-8"))
    mapping = repaired.get("mapping") or []
    assert isinstance(mapping, list)
    by_species = {entry["species"]: entry["superstate_id"] for entry in mapping}
    assert by_species["CO"] != by_species["H2"]

