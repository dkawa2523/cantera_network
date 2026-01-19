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


def _write_json_table(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    payload = {"columns": columns, "rows": rows}
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_benchmark_report_smoke(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    opt_manifest = _make_manifest(
        "optimization",
        "opt-1",
        inputs={"sample_count": 3, "run_ids": ["run-1", "run-2", "run-3"]},
    )

    def _write_opt(base_dir: Path) -> None:
        rows = [
            {
                "sample_id": 0,
                "run_id": "run-1",
                "observable_id": "obs-1",
                "objective_name": "objective",
                "objective": 1.2,
                "direction": "min",
                "params_json": "{}",
                "meta_json": "{}",
            },
            {
                "sample_id": 1,
                "run_id": "run-2",
                "observable_id": "obs-2",
                "objective_name": "objective",
                "objective": 0.8,
                "direction": "min",
                "params_json": "{}",
                "meta_json": "{}",
            },
        ]
        _write_json_table(
            base_dir / "history.parquet",
            [
                "sample_id",
                "run_id",
                "observable_id",
                "objective_name",
                "objective",
                "direction",
                "params_json",
                "meta_json",
            ],
            rows,
        )

    store.ensure(opt_manifest, writer=_write_opt)

    assim_manifest = _make_manifest(
        "assimilation",
        "assim-1",
        inputs={"iterations": 2, "ensemble_size": 3},
    )

    def _write_assim(base_dir: Path) -> None:
        rows = [
            {
                "iteration": 0,
                "mean_misfit": 1.5,
                "min_misfit": 1.2,
                "max_misfit": 1.7,
                "valid_count": 3,
                "total_count": 3,
                "status": "ok",
                "message": "",
            },
            {
                "iteration": 1,
                "mean_misfit": 0.9,
                "min_misfit": 0.8,
                "max_misfit": 1.1,
                "valid_count": 3,
                "total_count": 3,
                "status": "ok",
                "message": "",
            },
        ]
        _write_json_table(
            base_dir / "misfit_history.parquet",
            [
                "iteration",
                "mean_misfit",
                "min_misfit",
                "max_misfit",
                "valid_count",
                "total_count",
                "status",
                "message",
            ],
            rows,
        )

    store.ensure(assim_manifest, writer=_write_assim)

    reduction_manifest = _make_manifest("reduction", "red-1")

    def _write_reduction(base_dir: Path) -> None:
        patch_payload = {
            "schema_version": 1,
            "disabled_reactions": [{"reaction_id": "R1"}, {"reaction_id": "R2"}],
            "reaction_multipliers": [],
        }
        (base_dir / "mechanism_patch.yaml").write_text(
            json.dumps(patch_payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    store.ensure(reduction_manifest, writer=_write_reduction)

    val_manifest = _make_manifest(
        "validation",
        "val-1",
        inputs={
            "selected_patch": {"patch_index": 0, "reduction_id": reduction_manifest.id},
            "patches": [{"patch_index": 0, "reduction_id": reduction_manifest.id}],
        },
    )

    def _write_validation(base_dir: Path) -> None:
        rows = [
            {
                "patch_index": 0,
                "patch_id": reduction_manifest.id,
                "passed": True,
                "status": "ok",
                "kind": "observable",
                "name": "objective",
                "unit": "1",
                "meta_json": "{}",
                "item_index": 0,
                "baseline_value": 1.0,
                "reduced_value": 1.05,
                "abs_diff": 0.05,
                "rel_diff": 0.05,
                "metric": "abs",
                "tolerance": 0.1,
                "baseline_run_id": "run-1",
                "reduced_run_id": "run-2",
                "baseline_artifact_id": "obs-1",
                "reduced_artifact_id": "obs-2",
            }
        ]
        _write_json_table(
            base_dir / "metrics.parquet",
            [
                "patch_index",
                "patch_id",
                "passed",
                "status",
                "kind",
                "name",
                "unit",
                "meta_json",
                "item_index",
                "baseline_value",
                "reduced_value",
                "abs_diff",
                "rel_diff",
                "metric",
                "tolerance",
                "baseline_run_id",
                "reduced_run_id",
                "baseline_artifact_id",
                "reduced_artifact_id",
            ],
            rows,
        )

    store.ensure(val_manifest, writer=_write_validation)

    cfg = {
        "viz": {
            "name": "benchmark_report",
            "title": "Benchmark Smoke",
            "dashboard": "benchmark",
            "groups": {
                "baseline": {
                    "optimization": [opt_manifest.id],
                    "validation": [val_manifest.id],
                },
                "advanced": {"assimilation": [assim_manifest.id]},
            },
            "reduction": {"baseline_reactions": 4},
            "inputs": [{"kind": "optimization", "id": "missing-opt"}],
        }
    }

    result = viz_task.benchmark_report(cfg, store=store)

    html = (result.path / "index.html").read_text(encoding="utf-8")
    assert "Benchmark Summary" in html
    assert "Optimization Artifacts" in html
    assert "Assimilation Artifacts" in html
    assert "Validation Artifacts" in html
    assert "optimization/missing-opt" in html
