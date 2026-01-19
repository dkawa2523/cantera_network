from pathlib import Path

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore, apply_parents_inputs


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


def test_store_ensure_reuses_artifact(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = _make_manifest()
    calls: list[Path] = []

    def writer(base_dir: Path) -> None:
        calls.append(base_dir)
        (base_dir / "data.txt").write_text("hello", encoding="utf-8")

    result = store.ensure(manifest, writer=writer)

    assert result.reused is False
    assert result.path == tmp_path / "artifacts" / "runs" / "run-1"
    assert (result.path / "data.txt").read_text(encoding="utf-8") == "hello"
    assert len(calls) == 1

    def writer_fail(_: Path) -> None:
        raise AssertionError("ensure should reuse the existing artifact.")

    reused = store.ensure(manifest, writer=writer_fail)

    assert reused.reused is True
    assert reused.path == result.path
    assert len(calls) == 1


def test_apply_parents_inputs_merges_provenance() -> None:
    manifest = _make_manifest(artifact_id="child-1")
    parent = _make_manifest(artifact_id="parent-1")

    updated = apply_parents_inputs(
        manifest,
        parents=[parent, "parent-2", "parent-1"],
        inputs={"extra": "value", "run_id": "x"},
    )

    assert updated.parents == ["parent-1", "parent-2"]
    assert updated.inputs == {"run_id": "x", "extra": "value"}
    assert manifest.parents == []
