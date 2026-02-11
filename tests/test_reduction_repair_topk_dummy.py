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


def _write_yaml(path: Path, payload: Any) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _ensure_patch(store: ArtifactStore, reduction_id: str, disabled_count: int) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=reduction_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        disabled = [{"index": i} for i in range(disabled_count)]
        payload = {"schema_version": 1, "disabled_reactions": disabled, "reaction_multipliers": []}
        _write_yaml(base_dir / "mechanism_patch.yaml", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_patch_with_state_merge(
    store: ArtifactStore,
    reduction_id: str,
    *,
    disabled_count: int,
    merged_species: int,
) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=reduction_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        disabled = [{"index": i} for i in range(disabled_count)]
        species_to_rep = {f"S{i}": "S0" for i in range(1, max(merged_species, 0) + 1)}
        payload = {
            "schema_version": 1,
            "disabled_reactions": disabled,
            "reaction_multipliers": [],
            "state_merge": {"species_to_representative": species_to_rep},
        }
        _write_yaml(base_dir / "mechanism_patch.yaml", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_validation(store: ArtifactStore, validation_id: str, patch_ids: list[str]) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="validation",
        id=validation_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={
            "patches": [{"patch_index": idx, "reduction_id": rid} for idx, rid in enumerate(patch_ids)]
        },
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        rows = []
        for idx, rid in enumerate(patch_ids):
            # Two metrics per patch; both pass.
            for j in range(2):
                rows.append(
                    {
                        "patch_index": idx,
                        "patch_id": rid,
                        "passed": True,
                        "status": "ok",
                        "kind": "feature",
                        "name": f"qoi.{j}",
                        "unit": "",
                        "meta_json": "{}",
                        "item_index": j,
                        "baseline_value": 1.0,
                        "reduced_value": 1.0,
                        "abs_diff": 0.0,
                        "rel_diff": 0.0,
                    }
                )
        _write_json(base_dir / "metrics.parquet", {"rows": rows})

    store.ensure(manifest, writer=_writer)


def test_repair_topk_selects_max_disabled_when_all_pass(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    red_a = "red_a"
    red_b = "red_b"
    val_id = "val0001"
    _ensure_patch(store, red_a, disabled_count=1)
    _ensure_patch(store, red_b, disabled_count=2)
    _ensure_validation(store, val_id, [red_a, red_b])

    task = get("task", "reduction.repair_topk")
    result = task(
        {"inputs": {"validation_id": val_id}, "params": {"policy": {"target_pass_rate": 1.0}}},
        store=store,
    )

    repair = json.loads((result.path / "repair.json").read_text(encoding="utf-8"))
    assert repair["selected_reduction_id"] == red_b

    import yaml

    patch = yaml.safe_load((result.path / "mechanism_patch.yaml").read_text(encoding="utf-8"))
    assert isinstance(patch, dict)
    assert len(patch.get("disabled_reactions") or []) == 2


def test_repair_topk_selects_max_merged_species_when_objective_set(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    red_a = "red_merge_a"
    red_b = "red_merge_b"
    val_id = "val_merge_0001"
    _ensure_patch_with_state_merge(store, red_a, disabled_count=10, merged_species=1)
    _ensure_patch_with_state_merge(store, red_b, disabled_count=1, merged_species=3)
    _ensure_validation(store, val_id, [red_a, red_b])

    task = get("task", "reduction.repair_topk")
    result = task(
        {
            "inputs": {"validation_id": val_id},
            "params": {"policy": {"objective": "merged_species", "target_pass_rate": 1.0}},
        },
        store=store,
    )

    repair = json.loads((result.path / "repair.json").read_text(encoding="utf-8"))
    assert repair["selected_reduction_id"] == red_b
