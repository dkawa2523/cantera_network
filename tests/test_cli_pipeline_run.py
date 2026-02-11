import json
import subprocess
import sys
from pathlib import Path


def test_cli_pipeline_run_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store_root = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rxn_platform.cli",
            "pipeline",
            "run",
            f"store.root={store_root}",
            "pipeline=smoke",
            "sim=dummy",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "sim" in payload
    assert isinstance(payload["sim"], str) and payload["sim"]
