import json

import pytest

np = pytest.importorskip("numpy")

import rxn_platform.tasks.graphs  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _make_graph_manifest(graph_id: str) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=graph_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs={"graph": "demo"},
        config={"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def _write_demo_graph(store: ArtifactStore, graph_id: str) -> None:
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {"name": "demo"},
        "nodes": [
            {"id": "A", "kind": "species"},
            {"id": "B", "kind": "species"},
            {"id": "C", "kind": "species"},
        ],
        "links": [
            {"source": "A", "target": "B", "weight": 2.0},
            {"source": "B", "target": "C", "weight": 1.0},
        ],
    }
    manifest = _make_graph_manifest(graph_id)

    def _writer(base_dir):
        (base_dir / "graph.json").write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    store.ensure(manifest, writer=_writer)


def test_graph_laplacian_properties(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    graph_id = "graph-demo"
    _write_demo_graph(store, graph_id)

    task = get("task", "graphs.laplacian")
    cfg = {"graphs": {"laplacian": {"graph_id": graph_id, "normalized": True}}}
    result = task(cfg, store=store)

    meta = json.loads((result.path / "graph.json").read_text(encoding="utf-8"))
    assert meta["nodes"]["order"] == ["A", "B", "C"]
    assert meta["laplacian"]["normalized"] is True

    with np.load(result.path / "laplacian.npz") as data:
        laplacian = data["laplacian"]
        degree = data["degree"]
        laplacian_norm = data["laplacian_norm"]

    assert degree.shape == (3,)
    assert laplacian.shape == (3, 3)
    assert laplacian_norm.shape == (3, 3)

    assert np.allclose(laplacian, laplacian.T)
    eigvals = np.linalg.eigvalsh(laplacian)
    assert eigvals.min() >= -1e-8

    assert np.allclose(laplacian_norm, laplacian_norm.T)
    eigvals_norm = np.linalg.eigvalsh(laplacian_norm)
    assert eigvals_norm.min() >= -1e-8
