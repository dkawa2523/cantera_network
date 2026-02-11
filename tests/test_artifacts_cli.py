import json
import os
from pathlib import Path
import subprocess
import sys

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore


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


def _make_manifest(kind: str, artifact_id: str) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs={},
        config={"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def test_artifacts_ls_lists_kinds_and_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.ensure(_make_manifest("runs", "run-1"))
    store.ensure(_make_manifest("reports", "rep-1"))

    result = _run_cli(
        "artifacts",
        "--root",
        str(store.root),
        "ls",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "runs:" in result.stdout
    assert "run-1" in result.stdout
    assert "reports:" in result.stdout
    assert "rep-1" in result.stdout


def test_artifacts_show_outputs_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = _make_manifest("runs", "run-1")
    store.ensure(manifest)

    result = _run_cli(
        "artifacts",
        "--root",
        str(store.root),
        "show",
        "runs",
        "run-1",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == "run-1"
    assert payload["kind"] == "runs"


def test_artifacts_show_missing_artifact_is_clear(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.root.mkdir(parents=True, exist_ok=True)

    result = _run_cli(
        "artifacts",
        "--root",
        str(store.root),
        "show",
        "runs",
        "missing",
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "Artifact runs/missing not found" in result.stderr
