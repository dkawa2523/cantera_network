import subprocess
import sys
from pathlib import Path


def _list_svg(path: Path) -> list[Path]:
    return sorted(path.glob("*.svg"))


def test_runstore_outputs_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runs" / "exp1" / "demo01"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rxn_platform.cli",
            "run",
            "recipe=viz_report",
            "sim=dummy",
            "run_id=demo01",
            "exp=exp1",
            f"run.root={run_root}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert (run_root / "manifest.json").exists()
    assert (run_root / "config_resolved.yaml").exists()
    assert (run_root / "metrics.json").exists()
    assert (run_root / "summary.json").exists()

    sim_dir = run_root / "sim" / "timeseries.zarr"
    assert sim_dir.exists()

    graph_meta = run_root / "graphs" / "meta.json"
    assert graph_meta.exists()

    viz_root = run_root / "viz"
    assert (viz_root / "index.html").exists()

    network_dir = viz_root / "network"
    assert network_dir.exists()
    assert (network_dir / "index.json").exists()
    assert sorted(network_dir.glob("*.dot"))

    ts_dir = viz_root / "timeseries"
    red_dir = viz_root / "reduction"
    assert ts_dir.exists()
    assert red_dir.exists()
    assert _list_svg(ts_dir)
    assert _list_svg(red_dir)
