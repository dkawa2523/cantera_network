import json

import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.tasks.graphs  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.graphs import build_stoich
from rxn_platform.validators import validate_graph_artifact


def test_build_stoich_shapes() -> None:
    solution = ct.Solution("gri30.yaml")
    result = build_stoich(solution)

    assert result.matrix.shape[0] == len(result.species)
    assert result.matrix.shape[1] == len(result.reaction_ids)
    assert len(result.reaction_equations) == result.matrix.shape[1]


def test_graphs_task_writes_artifact(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "graphs.stoich")
    cfg = {"graphs": {"mechanism": "gri30.yaml"}}

    result = task(cfg, store=store)
    assert result.manifest.kind == "graphs"
    validate_graph_artifact(result.path)

    meta = json.loads((result.path / "graph.json").read_text(encoding="utf-8"))
    assert meta["shape"][0] == len(meta["species"])
    assert meta["shape"][1] == len(meta["reactions"])
    assert (result.path / "stoich.npz").exists()
    first = meta["reactions"][0]
    assert "id" in first
    assert "equation" in first
