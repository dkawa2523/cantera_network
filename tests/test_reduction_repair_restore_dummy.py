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


def _ensure_reduction_patch(store: ArtifactStore, reduction_id: str, disabled_indices: list[int]) -> None:
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
        payload = {
            "schema_version": 1,
            "disabled_reactions": [{"index": int(i)} for i in disabled_indices],
            "reaction_multipliers": [],
        }
        _write_yaml(base_dir / "mechanism_patch.yaml", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_run_set(store: ArtifactStore, run_set_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="run_sets",
        id=run_set_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        payload = {
            "schema_version": 1,
            "kind": "run_set",
            "case_to_run": {"c000": "run000"},
        }
        _write_json(base_dir / "runs.json", payload)

    store.ensure(manifest, writer=_writer)


def _ensure_features(store: ArtifactStore, features_id: str) -> None:
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
                "run_id": "run000",
                "feature": "rop_net.RX0.integral",
                "value": 10.0,
                "unit": "",
                "meta_json": json.dumps({"reaction_index": 0}),
            },
            {
                "run_id": "run000",
                "feature": "rop_net.RX3.integral",
                "value": 9.0,
                "unit": "",
                "meta_json": json.dumps({"reaction_index": 3}),
            },
        ]
        # Use the JSON-table fallback that tasks.common.read_table_rows supports.
        _write_json(base_dir / "features.parquet", {"rows": rows})

    store.ensure(manifest, writer=_writer)


def _ensure_validation(store: ArtifactStore, validation_id: str, patch_id: str) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="validation",
        id=validation_id,
        created_at="2026-02-09T00:00:00Z",
        parents=[],
        inputs={"patches": [{"patch_index": 0, "reduction_id": patch_id}]},
        config={},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )

    def _writer(base_dir: Path) -> None:
        # One failing row for c000.
        rows = [
            {
                "patch_index": 0,
                "patch_id": patch_id,
                "passed": False,
                "status": "fail",
                "kind": "observable",
                "name": "gas.CO2.last",
                "unit": "",
                "meta_json": "{}",
                "item_index": 0,
                "case_id": "c000",
                "baseline_value": 1.0,
                "reduced_value": 2.0,
                "abs_diff": 1.0,
                "rel_diff": 1.0,
                "metric": "rel",
                "tolerance": 0.5,
                "baseline_run_id": "baseline",
                "reduced_run_id": "reduced",
                "baseline_artifact_id": "base",
                "reduced_artifact_id": "red",
            }
        ]
        _write_json(base_dir / "metrics.parquet", {"rows": rows})

    store.ensure(manifest, writer=_writer)


def test_repair_restore_restores_top_importance_for_failing_case(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    base_patch = "red_base"
    val_id = "val0001"
    run_set_id = "rs0001"
    feat_id = "feat0001"

    _ensure_reduction_patch(store, base_patch, disabled_indices=[0, 1, 2, 3, 4])
    _ensure_validation(store, val_id, patch_id=base_patch)
    _ensure_run_set(store, run_set_id)
    _ensure_features(store, feat_id)

    task = get("task", "reduction.repair_restore")
    result = task(
        {
            "inputs": {
                "base_reduction_id": base_patch,
                "validation_id": val_id,
                "features_id": feat_id,
                "run_set_id": run_set_id,
            },
            "params": {"restore_per_case": 1, "target_patch_id": base_patch},
        },
        store=store,
    )

    import yaml

    patch = yaml.safe_load((result.path / "mechanism_patch.yaml").read_text(encoding="utf-8"))
    disabled = patch.get("disabled_reactions") or []
    disabled_indices = sorted(int(e["index"]) for e in disabled)
    # reaction_index=0 is the highest-importance; it should be restored (removed from disabled).
    assert 0 not in disabled_indices
    assert len(disabled_indices) == 4

