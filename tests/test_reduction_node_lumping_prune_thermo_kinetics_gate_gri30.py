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


def test_node_lumping_prune_thermo_kinetics_gate_reduces_merges(tmp_path) -> None:
    base = ct.Solution("gri30.yaml")
    required = {"HNCO", "HOCN", "HCNO", "CO2"}
    if not required.issubset(set(base.species_names)):
        pytest.skip("gri30.yaml does not contain required isomer species for this test.")

    store = ArtifactStore(tmp_path / "artifacts")
    lumping_id = "nl_gate_0001"
    _ensure_dummy_node_lumping(store, lumping_id)

    prune_task = get("task", "reduction.node_lumping_prune")

    # Baseline: merge within same composition signature, without thermo/kinetics gating.
    base_res = prune_task(
        {
            "inputs": {"node_lumping": lumping_id},
            "mechanism": "gri30.yaml",
            "params": {
                "merge_mode": "signature_split",
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "require_same_phase": True,
                "cache_bust": "base",
            },
        },
        store=store,
    )
    base_metrics = json.loads((base_res.path / "metrics.json").read_text(encoding="utf-8"))
    assert base_metrics["accepted_merges"] >= 2

    # Thermo gate: use a strict threshold to ensure candidates are rejected and the skip reason is recorded.
    thermo_res = prune_task(
        {
            "inputs": {"node_lumping": lumping_id},
            "mechanism": "gri30.yaml",
            "params": {
                "merge_mode": "signature_split",
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "require_same_phase": True,
                "thermo_constraints": {
                    "enabled": True,
                    "T_grid": [1000.0],
                    "P_ref": 101325.0,
                    "max_rel_cp": 0.0,
                    "max_rel_h": 0.0,
                    "max_rel_s": 0.0,
                    "missing_strategy": "skip",
                },
                "cache_bust": "thermo",
            },
        },
        store=store,
    )
    thermo_metrics = json.loads((thermo_res.path / "metrics.json").read_text(encoding="utf-8"))
    assert thermo_metrics["thermo_constraints"]["enabled"] is True
    assert thermo_metrics["accepted_merges"] <= base_metrics["accepted_merges"]
    assert (
        thermo_metrics["skip_reasons"].get("thermo_mismatch", 0) > 0
        or thermo_metrics["skip_reasons"].get("thermo_missing", 0) > 0
    )

    import yaml

    patch = yaml.safe_load((thermo_res.path / "mechanism_patch.yaml").read_text(encoding="utf-8"))
    assert isinstance(patch, dict)
    groups = ((patch.get("state_merge") or {}).get("signature_groups") or [])
    assert any(isinstance(g, dict) and "members_info" in g for g in groups)

    # Kinetics gate: strict threshold; should reject and record reason.
    kin_res = prune_task(
        {
            "inputs": {"node_lumping": lumping_id},
            "mechanism": "gri30.yaml",
            "params": {
                "merge_mode": "signature_split",
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "require_same_phase": True,
                "kinetics_constraints": {
                    "enabled": True,
                    "T_grid": [1000.0],
                    "P_ref": 101325.0,
                    "X_ref": "N2:0.79, O2:0.21",
                    "eps": 1.0e-300,
                    "species_signature": "adjacent_logk_mean",
                    "max_abs_log10k_diff": 0.0,
                    "min_adj_size": 1,
                    "missing_strategy": "skip",
                },
                "cache_bust": "kinetics",
            },
        },
        store=store,
    )
    kin_metrics = json.loads((kin_res.path / "metrics.json").read_text(encoding="utf-8"))
    assert kin_metrics["kinetics_constraints"]["enabled"] is True
    assert kin_metrics["accepted_merges"] <= base_metrics["accepted_merges"]
    assert (
        kin_metrics["skip_reasons"].get("kinetics_mismatch", 0) > 0
        or kin_metrics["skip_reasons"].get("kinetics_missing", 0) > 0
    )

