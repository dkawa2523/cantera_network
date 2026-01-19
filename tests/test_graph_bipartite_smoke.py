import json

import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.tasks.graphs  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.graphs import build_bipartite_graph, build_stoich


def test_build_bipartite_graph_structure() -> None:
    solution = ct.Solution("gri30.yaml")
    result = build_stoich(solution)
    graph = build_bipartite_graph(result)

    assert graph["directed"] is True

    nodes = graph["nodes"]
    links = graph["links"]
    node_ids = {node["id"] for node in nodes}
    species_nodes = [node for node in nodes if node.get("kind") == "species"]
    reaction_nodes = [node for node in nodes if node.get("kind") == "reaction"]

    assert len(species_nodes) == len(result.species)
    assert len(reaction_nodes) == len(result.reaction_ids)
    assert all(node["id"].startswith("species_") for node in species_nodes)
    assert all(node["id"].startswith("reaction_") for node in reaction_nodes)

    for link in links:
        assert link["source"] in node_ids
        assert link["target"] in node_ids
        assert link["source"].startswith("species_")
        assert link["target"].startswith("reaction_")
        assert "stoich" in link
        assert link["role"] in ("reactant", "product")
        expected_role = "reactant" if link["stoich"] < 0 else "product"
        assert link["role"] == expected_role


def test_graphs_task_writes_bipartite(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "graphs.stoich")
    cfg = {"graphs": {"mechanism": "gri30.yaml"}}

    result = task(cfg, store=store)
    meta = json.loads((result.path / "graph.json").read_text(encoding="utf-8"))

    assert "bipartite" in meta
    bipartite = meta["bipartite"]
    assert bipartite["format"] == "node_link"
    data = bipartite["data"]
    assert data["directed"] is True
    assert data["nodes"]
    assert data["links"]
