from __future__ import annotations

import pytest

ct = pytest.importorskip("cantera")

from rxn_platform.tasks.graphs import annotate_species, build_bipartite_graph, build_stoich


def _find_species_node(graph: dict[str, object], label: str) -> dict[str, object]:
    for node in graph["nodes"]:
        if node.get("kind") == "species" and node.get("label") == label:
            return node
    raise AssertionError(f"species node not found: {label}")


def test_annotate_species_fields() -> None:
    solution = ct.Solution("gri30.yaml")
    annotations = annotate_species(solution)

    h2o = annotations["H2O"]
    assert h2o["formula"] == "H2O"
    assert h2o["elements"]["H"] == pytest.approx(2.0)
    assert h2o["elements"]["O"] == pytest.approx(1.0)
    assert h2o["charge"] == 0.0
    assert h2o["phase"] == "gas"
    assert h2o["state"] in ("neutral", "radical", "ion", "unknown")
    assert h2o["is_inferred"] is True
    assert "state" in h2o["inferred_fields"]


def test_bipartite_graph_includes_annotations() -> None:
    solution = ct.Solution("gri30.yaml")
    result = build_stoich(solution)
    annotations = annotate_species(solution)
    graph = build_bipartite_graph(result, species_annotations=annotations)

    h2o_node = _find_species_node(graph, "H2O")
    assert h2o_node["formula"] == annotations["H2O"]["formula"]
    assert h2o_node["elements"] == annotations["H2O"]["elements"]
    assert h2o_node["charge"] == annotations["H2O"]["charge"]
    assert h2o_node["phase"] == annotations["H2O"]["phase"]
    assert h2o_node["state"] == annotations["H2O"]["state"]
    assert h2o_node["is_inferred"] is True
