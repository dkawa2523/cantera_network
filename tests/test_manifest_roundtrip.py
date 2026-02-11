import json

from rxn_platform.core import ArtifactManifest, dump_manifest, load_manifest


def test_manifest_roundtrip(tmp_path) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id="run-1",
        created_at="2026-01-17T00:00:00Z",
        parents=[],
        inputs={"run_id": "x"},
        config={"simulation": {"seed": 123}},
        code={"git_commit": "abc123", "dirty": False, "version": "0.0.0"},
        provenance={"python": "3.11"},
    )
    path = tmp_path / "manifest.yaml"
    dump_manifest(path, manifest)
    loaded = load_manifest(path)
    assert manifest.to_dict() == loaded.to_dict()


def test_manifest_rejects_unknown_fields(tmp_path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "runs",
        "id": "run-1",
        "created_at": "2026-01-17T00:00:00Z",
        "parents": [],
        "inputs": {},
        "config": {},
        "code": {},
        "provenance": {},
        "extra": "nope",
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_manifest(path)
    except ValueError as exc:
        assert "Unknown fields" in str(exc)
    else:
        assert False, "Expected ValueError for unknown fields."
