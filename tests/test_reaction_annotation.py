from __future__ import annotations

import pytest

ct = pytest.importorskip("cantera")

from rxn_platform.tasks.graphs import (
    annotate_reactions,
    build_bipartite_graph,
    build_stoich,
)


def _find_reaction_node(graph: dict[str, object], reaction_id: str) -> dict[str, object]:
    for node in graph["nodes"]:
        if node.get("kind") == "reaction" and node.get("reaction_id") == reaction_id:
            return node
    raise AssertionError(f"reaction node not found: {reaction_id}")


def test_annotate_reactions_fields() -> None:
    solution = ct.Solution("gri30.yaml")
    annotations = annotate_reactions(solution)

    assert len(annotations) == solution.n_reactions
    assert any(
        annotation["reaction_type"] != "unknown"
        for annotation in annotations.values()
    )
    for annotation in annotations.values():
        assert "reaction_type" in annotation
        assert "order" in annotation
        assert "reversible" in annotation
        assert "duplicate" in annotation
        assert "reaction_type_reason" in annotation
        if annotation["reaction_type"] == "unknown":
            assert annotation["reaction_type_reason"]


def test_bipartite_graph_includes_reaction_annotations() -> None:
    solution = ct.Solution("gri30.yaml")
    result = build_stoich(solution)
    annotations = annotate_reactions(solution)
    graph = build_bipartite_graph(result, reaction_annotations=annotations)

    reaction_id = result.reaction_ids[0]
    node = _find_reaction_node(graph, reaction_id)
    assert node["reaction_type"] == annotations[reaction_id]["reaction_type"]
    assert node["order"] == annotations[reaction_id]["order"]
    assert node["reversible"] == annotations[reaction_id]["reversible"]
    assert node["duplicate"] == annotations[reaction_id]["duplicate"]
