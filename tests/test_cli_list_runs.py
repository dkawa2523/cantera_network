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


def test_list_runs_respects_runstore_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runstore" / "exp" / "run1"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        "{\"schema_version\":1,\"run_id\":\"run1\",\"exp\":\"exp\",\"created_at\":\"2026-02-01T00:00:00Z\"}\n",
        encoding="utf-8",
    )
    result = _run_cli(
        "list-runs",
        "--runstore-root",
        str(tmp_path / "runstore"),
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert "exp/run1" in result.stdout
