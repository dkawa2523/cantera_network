import pytest

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore


def _make_manifest(kind: str = "runs", artifact_id: str = "run-1") -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs={"run_id": "x"},
        config={"simulation": {"seed": 123}},
        code={"git_commit": "abc123", "dirty": False, "version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def test_artifact_store_roundtrip(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = _make_manifest()
    artifact_dir = store.write_artifact(
        manifest,
        data_files={"data.txt": "hello"},
    )

    assert artifact_dir == tmp_path / "artifacts" / "runs" / "run-1"
    assert store.exists("runs", "run-1")

    loaded = store.read_manifest("runs", "run-1")
    assert loaded.to_dict() == manifest.to_dict()
    assert (artifact_dir / "data.txt").read_text(encoding="utf-8") == "hello"

    with pytest.raises(FileExistsError):
        store.write_artifact(
            manifest,
            data_files={"data.txt": "overwrite"},
        )
