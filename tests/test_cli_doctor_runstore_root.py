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


def test_doctor_accepts_runstore_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runstore_root = tmp_path / "runstore"
    store_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs" / "default" / "doctor_smoke"

    result = _run_cli(
        "doctor",
        "--runstore-root",
        str(runstore_root),
        f"store.root={store_root}",
        f"run.root={run_root}",
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
