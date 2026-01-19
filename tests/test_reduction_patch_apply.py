import json

import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import load_config
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def test_reduction_patch_apply_creates_mechanism(tmp_path) -> None:
    mechanism_payload = {
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "species": ["A", "B", "C"],
                "reactions": "all",
            }
        ],
        "species": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        "reactions": [
            {"id": "R1", "equation": "A => B", "rate-constant": {"A": 1.0}},
            {"equation": "B => C", "rate-constant": {"A": 2.0}},
        ],
    }
    mechanism_path = tmp_path / "mechanism.yaml"
    mechanism_path.write_text(
        json.dumps(mechanism_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    patch_payload = {
        "schema_version": 1,
        "disabled_reactions": [{"reaction_id": "R1"}],
        "reaction_multipliers": [{"reaction_id": "B => C", "multiplier": 0.5}],
    }
    patch_path = tmp_path / "patch.yaml"
    patch_path.write_text(
        json.dumps(patch_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "reduction.apply")
    cfg = {"reduction": {"mechanism": str(mechanism_path), "patch": str(patch_path)}}
    result = task(cfg, store=store)

    reduced_path = result.path / "mechanism.yaml"
    assert reduced_path.exists()
    reduced_payload = load_config(reduced_path)
    assert len(reduced_payload["reactions"]) == 1
    assert reduced_payload["reactions"][0]["equation"] == "B => C"

    original_payload = load_config(mechanism_path)
    assert len(original_payload["reactions"]) == 2

    patch_in_manifest = result.manifest.inputs["patch"]
    assert patch_in_manifest["disabled_reactions"] == [{"reaction_id": "R1"}]
    assert patch_in_manifest["reaction_multipliers"][0]["multiplier"] == 0.5

    patch_file_payload = load_config(result.path / "mechanism_patch.yaml")
    assert patch_file_payload["reaction_multipliers"][0]["reaction_id"] == "B => C"
