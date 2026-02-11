import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

RECIPES = [
    "smoke",
    "build_temporal_graph",
    "build_gnn_dataset",
    "reduce_amore_search",
    "reduce_learnck_style",
    "reduce_gnn_importance_prune",
    "benchmark_compare",
    "validate_mech",
]
REQUIRES_XARRAY = {"reduce_learnck_style", "benchmark_compare"}

EXPECTED_ARTIFACTS = {
    "reduce_amore_search": ("amore", "reduction", "mechanism_patch.yaml"),
    "reduce_learnck_style": ("learnck", "reduction", "mechanism_patch.yaml"),
    "reduce_gnn_importance_prune": (
        "prune",
        "reduction",
        "mechanism_patch.yaml",
    ),
    "validate_mech": ("validate", "validation", "metrics.parquet"),
}


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[1] / "src"
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(src_root)
    return env


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rxn_platform.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_cli_env(),
    )


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_smoke_cli(tmp_path: Path, recipe: str) -> None:
    if recipe in REQUIRES_XARRAY:
        pytest.importorskip("xarray")
    repo_root = Path(__file__).resolve().parents[1]
    run_id = f"{recipe}_smoke"
    run_root = tmp_path / "runs" / run_id
    result = _run_cli(
        "run",
        f"recipe={recipe}",
        f"run_id={run_id}",
        f"run.root={run_root}",
        "common.seed=1",
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run_id

    metrics_path = run_root / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("status") == "ok"

    expected = EXPECTED_ARTIFACTS.get(recipe)
    if expected:
        step_id, kind, filename = expected
        artifact_id = payload["results"][step_id]
        artifact_path = run_root / "artifacts" / kind / artifact_id / filename
        assert artifact_path.exists()
