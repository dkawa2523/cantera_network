import json

import rxn_platform.tasks.graphs  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore
from rxn_platform.validators import validate_graph_artifact


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
            {"id": "D", "kind": "species"},
        ],
        "links": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},
        ],
    }
    manifest = _make_graph_manifest(graph_id)

    def _writer(base_dir):
        (base_dir / "graph.json").write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    store.ensure(manifest, writer=_writer)


def test_graph_analytics_smoke(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    graph_id = "graph-demo"
    _write_demo_graph(store, graph_id)

    task = get("task", "graphs.analyze")
    cfg = {
        "graphs": {
            "analysis": {
                "graph_id": graph_id,
                "top_n": 2,
                "max_components": 2,
                "max_component_nodes": 5,
            }
        }
    }

    result = task(cfg, store=store)
    validate_graph_artifact(result.path)

    payload = json.loads((result.path / "graph.json").read_text(encoding="utf-8"))
    analysis = payload["analysis"]

    assert analysis["scc"]["count"] == 2
    assert analysis["communities"]["count"] == 2

    ranking = analysis["centrality"]["degree"]["ranking"]
    assert len(ranking) == 2
    node_ids = {entry["node_id"] for entry in ranking}
    assert node_ids.issubset({"A", "B", "C"})
