import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.tasks.graphs  # noqa: F401
import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def test_node_lumping_prune_writes_loadable_mechanism(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    graph_task = get("task", "graphs.stoich")
    graph_res = graph_task({"graphs": {"mechanism": "gri30.yaml"}}, store=store)

    lump_task = get("task", "reduction.node_lumping")
    lump_res = lump_task({"inputs": {"graph_id": graph_res.manifest.id}}, store=store)

    prune_task = get("task", "reduction.node_lumping_prune")
    prune_res = prune_task(
        {
            "inputs": {"node_lumping": lump_res.manifest.id},
            "mechanism": "gri30.yaml",
            "params": {
                "protected_species": ["CO", "CO2"],
                "require_same_composition": True,
                "require_same_charge": True,
                "cache_bust": "test",
            },
        },
        store=store,
    )

    mech_path = prune_res.path / "mechanism.yaml"
    assert mech_path.exists()

    base = ct.Solution("gri30.yaml")
    reduced = ct.Solution(str(mech_path))

    assert reduced.n_species <= base.n_species
    assert "CO" in reduced.species_names
    assert "CO2" in reduced.species_names

