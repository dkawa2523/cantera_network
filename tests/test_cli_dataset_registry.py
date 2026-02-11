import json
import os
from pathlib import Path
import subprocess
import sys

from rxn_platform.run_store import (
    write_run_config,
    write_run_manifest,
    write_run_metrics,
)


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


def _write_runstore_entry(run_root: Path) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": run_root.name,
        "exp": run_root.parent.name,
        "created_at": "2026-02-01T00:00:00Z",
        "recipe": "smoke",
        "store_root": str(run_root / "artifacts"),
        "simulator": "dummy",
        "mechanism_hash": "deadbeef",
        "conditions_hash": "cafebabe",
        "qoi_spec_hash": "abcd1234",
    }
    write_run_manifest(run_root, manifest)
    write_run_config(run_root, {"sim": {"name": "dummy"}})
    write_run_metrics(run_root, {"schema_version": 1, "status": "ok", "results": {}})


def test_dataset_register_accepts_run_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runs" / "default" / "demo_run"
    _write_runstore_entry(run_root)

    datasets_root = tmp_path / "datasets"
    result = _run_cli(
        "dataset",
        "register",
        "--run-id",
        "demo_run",
        "--exp",
        "default",
        "--run-root",
        str(run_root),
        "--root",
        str(datasets_root),
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "demo_run"
    registry_path = datasets_root / "registry.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry.get("datasets")
