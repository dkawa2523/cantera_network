from __future__ import annotations

from pathlib import Path

from rxn_platform.core import ArtifactManifest
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks import viz as viz_task


def _make_manifest(kind: str = "runs", artifact_id: str = "run-1") -> ArtifactManifest:
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


def test_viz_base_creates_report_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    input_manifest = _make_manifest()
    store.ensure(input_manifest)

    cfg = {
        "task": {"name": "viz.base"},
        "viz": {
            "name": "base",
            "title": "Smoke Report",
            "dashboard": "base",
            "inputs": [{"kind": "runs", "id": input_manifest.id}],
        },
    }

    result = viz_task.run(cfg, store=store)

    report_manifest = store.read_manifest("reports", result.manifest.id)
    assert result.path == store.root / "reports" / result.manifest.id
    assert report_manifest.inputs["artifacts"][0]["id"] == input_manifest.id

    html = (result.path / "index.html").read_text(encoding="utf-8")
    assert "Input Artifacts" in html
    assert input_manifest.id in html
    assert "Report Manifest" in html
    assert '"dashboard": "base"' in html
