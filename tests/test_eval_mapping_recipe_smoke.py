import json
import os
from pathlib import Path
import subprocess
import sys


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


def _run_recipe(tmp_path: Path, repo_root: Path, recipe: str, run_id: str) -> dict:
    run_root = tmp_path / "runs" / run_id
    result = _run_cli(
        "run",
        f"recipe={recipe}",
        f"run_id={run_id}",
        f"run.root={run_root}",
        "common.seed=2",
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_eval_mapping_recipe_smoke(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("xarray")
    repo_root = Path(__file__).resolve().parents[1]

    gnn_payload = _run_recipe(tmp_path, repo_root, "reduce_gnn_pool_temporal", "gnn_pool")
    cnr_payload = _run_recipe(tmp_path, repo_root, "reduce_cnr_coarse", "cnr")

    gnn_mapping = gnn_payload["results"]["mapping"]
    gnn_graph = gnn_payload["results"]["graph"]
    cnr_mapping = cnr_payload["results"]["mapping"]
    cnr_graph = cnr_payload["results"]["graph"]

    gnn_root = Path(gnn_payload["run_root"]) / "artifacts"
    cnr_root = Path(cnr_payload["run_root"]) / "artifacts"

    assert (gnn_root / "reduction" / gnn_mapping / "mapping.json").exists()
    assert (cnr_root / "reduction" / cnr_mapping / "mapping.json").exists()

    eval_root = tmp_path / "runs" / "eval_mapping"
    eval_result = _run_cli(
        "run",
        "recipe=eval_mapping",
        "run_id=eval_mapping",
        f"run.root={eval_root}",
        f"eval_mapping.gnn.mapping_id={gnn_mapping}",
        f"eval_mapping.gnn.graph_id={gnn_graph}",
        f"eval_mapping.gnn.store_root={gnn_root}",
        f"eval_mapping.compare.mapping_id={cnr_mapping}",
        f"eval_mapping.compare.graph_id={cnr_graph}",
        f"eval_mapping.compare.store_root={cnr_root}",
        cwd=repo_root,
    )
    assert eval_result.returncode == 0, eval_result.stderr
    eval_payload = json.loads(eval_result.stdout)

    eval_id = eval_payload["results"]["mapping_eval"]
    metrics_path = (
        Path(eval_payload["run_root"]) / "artifacts" / "validation" / eval_id / "metrics.json"
    )
    assert metrics_path.exists()
