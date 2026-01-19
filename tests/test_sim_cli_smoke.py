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


def _extract_id(stdout: str, label: str) -> str:
    token = f"{label}="
    for line in stdout.splitlines():
        if token in line:
            return line.split(token, 1)[-1].strip()
    raise AssertionError(f"{label} not found in output: {stdout!r}")


def test_sim_run_cli_creates_artifact_and_cache(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store_root = tmp_path / "artifacts"
    result = _run_cli(
        "sim",
        "run",
        f"store.root={store_root}",
        "sim=dummy",
        "sim.time.steps=2",
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    run_id = _extract_id(result.stdout, "run_id")
    assert run_id
    assert (store_root / "runs" / run_id / "manifest.yaml").exists()

    second = _run_cli(
        "sim",
        "run",
        f"store.root={store_root}",
        "sim=dummy",
        "sim.time.steps=2",
        cwd=repo_root,
    )
    assert second.returncode == 0, second.stderr
    assert "Cache hit" in second.stderr
    assert run_id == _extract_id(second.stdout, "run_id")


def test_sim_viz_cli_creates_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store_root = tmp_path / "artifacts"
    run = _run_cli(
        "sim",
        "run",
        f"store.root={store_root}",
        "sim=dummy",
        "sim.time.steps=2",
        cwd=repo_root,
    )
    assert run.returncode == 0, run.stderr
    run_id = _extract_id(run.stdout, "run_id")

    result = _run_cli(
        "sim",
        "viz",
        run_id,
        "--root",
        str(store_root),
        "--title",
        "Smoke Report",
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    report_id = _extract_id(result.stdout, "report_id")
    report_path = store_root / "reports" / report_id / "index.html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert run_id in html


def test_sim_validate_cli_reports_error(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "defaults.yaml").write_text("defaults:\n  - _self_\n", encoding="utf-8")

    result = _run_cli(
        "sim",
        "validate",
        "--config-path",
        str(config_dir),
        "--config-name",
        "defaults",
        cwd=repo_root,
    )
    assert result.returncode != 0
    assert "sim config is missing" in result.stderr.lower()
