import json

import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _ensure_dummy_node_lumping(store: ArtifactStore, artifact_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=artifact_id,
        created_at="2026-02-10T00:00:00Z",
        parents=[],
        inputs={"mode": "node_lumping"},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir) -> None:
        payload = {
            "schema_version": 1,
            "kind": "node_lumping",
            "clusters": [
                {
                    "cluster_id": 0,
                    "members": ["HNCO", "HOCN", "HCNO", "CO2"],
                    "representative": "CO2",
                }
            ],
            # Representative-only mapping (what signature_split should overcome).
            "mapping": [
                {"species": "HNCO", "representative": "CO2"},
                {"species": "HOCN", "representative": "CO2"},
                {"species": "HCNO", "representative": "CO2"},
                {"species": "CO2", "representative": "CO2"},
            ],
            "species": [
                {"species": "HNCO", "phase": "gas", "degree": 10},
                {"species": "HCNO", "phase": "gas", "degree": 5},
                {"species": "HOCN", "phase": "gas", "degree": 1},
                {"species": "CO2", "phase": "gas", "degree": 2},
            ],
        }
        (base_dir / "node_lumping.json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    store.ensure(manifest, writer=_writer)


def test_node_lumping_prune_signature_split_merges_same_signature(tmp_path) -> None:
    base = ct.Solution("gri30.yaml")
    required = {"HNCO", "HOCN", "HCNO", "CO2"}
    if not required.issubset(set(base.species_names)):
        pytest.skip("gri30.yaml does not contain required isomer species for this test.")

    store = ArtifactStore(tmp_path / "artifacts")
    lumping_id = "nl0001"
    _ensure_dummy_node_lumping(store, lumping_id)

    prune_task = get("task", "reduction.node_lumping_prune")

    rep_res = prune_task(
        {
            "inputs": {"node_lumping": lumping_id},
            "mechanism": "gri30.yaml",
            "params": {
                "merge_mode": "cluster_representative",
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "require_same_phase": True,
                "cache_bust": "rep",
            },
        },
        store=store,
    )
    rep_metrics = json.loads((rep_res.path / "metrics.json").read_text(encoding="utf-8"))

    split_res = prune_task(
        {
            "inputs": {"node_lumping": lumping_id},
            "mechanism": "gri30.yaml",
            "params": {
                "merge_mode": "signature_split",
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "require_same_phase": True,
                "cache_bust": "split",
            },
        },
        store=store,
    )
    split_metrics = json.loads((split_res.path / "metrics.json").read_text(encoding="utf-8"))

    assert rep_metrics["accepted_merges"] == 0
    assert split_metrics["accepted_merges"] >= 2

