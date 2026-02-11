from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.features  # noqa: F401
import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


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


def test_reduction_validate_inject_inputs_overrides_pipeline_step_inputs(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    mapping_id = "map0001"
    _ensure_mapping_json(store, mapping_id)

    # Mechanism file is required by the reduction.validate interface, but dummy sim ignores it.
    mechanism_path = tmp_path / "mechanism.yaml"
    mechanism_payload = {
        "phases": [{"name": "gas", "thermo": "ideal-gas", "species": ["A"], "reactions": "all"}],
        "species": [{"name": "A"}],
        "reactions": [
            {"id": "R1", "equation": "A => A", "rate-constant": {"A": 1.0}},
        ],
    }
    mechanism_path.write_text(
        json.dumps(mechanism_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    inner_pipeline_cfg = {
        "steps": [
            {
                "id": "sim",
                "task": "sim.run",
                "sim": {
                    "name": "dummy",
                    "time": {"start": 0.0, "stop": 2.0, "steps": 3},
                    "species": ["CO", "CO2", "H2"],
                    "ramp": {"T": 200.0},
                },
            },
            {
                "id": "qoi",
                "task": "features.superstate_qoi",
                "inputs": {
                    "run_id": "@sim",
                    # If inject_inputs is not applied, this will fail to resolve.
                    "mapping_id": "MAPPING_ID_REQUIRED",
                },
                "params": {
                    "targets": [
                        {"name": "CO_final_super", "member_species": "CO", "stat": "last"},
                        {"name": "CO2_final_super", "member_species": "CO2", "stat": "last"},
                    ],
                    "include_temperature_qoi": True,
                    "include_ignition_delay": True,
                },
            },
        ]
    }

    patch = {
        "schema_version": 1,
        "reaction_multipliers": [{"index": 0, "multiplier": 1.0}],
    }

    task = get("task", "reduction.validate")
    cfg = {
        "reduction": {
            "mechanism": str(mechanism_path),
            "inputs": {"mapping_id": mapping_id},
            "validation": {
                "pipeline": inner_pipeline_cfg,
                "patches": [patch],
                "metric": "abs",
                "tolerance": 1.0e-12,
                "sim_step_id": "sim",
                "features_step_id": "qoi",
                "use_multipliers_only": True,
                "inject_inputs": {"qoi": {"mapping_id": "$inputs.mapping_id"}},
            },
        }
    }

    result = task(cfg, store=store)
    assert result.manifest.kind == "validation"
    assert result.manifest.inputs["passed"] is True
    report_path = result.path / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "level0" in report
    assert "level1" in report
    assert "level2" in report
