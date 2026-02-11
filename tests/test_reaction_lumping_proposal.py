import json

import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import ArtifactManifest, load_config
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _store_graph(store: ArtifactStore, artifact_id: str, payload: dict) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=artifact_id,
        created_at="2026-01-01T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )

    def _writer(base_dir):
        path = base_dir / "graph.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    store.ensure(manifest, writer=_writer)


def test_reaction_lumping_proposal_reaction_type(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    graph_payload = {
        "bipartite": {
            "data": {
                "nodes": [
                    {"id": "species_A", "kind": "species", "label": "A"},
                    {"id": "species_B", "kind": "species", "label": "B"},
                    {"id": "species_C", "kind": "species", "label": "C"},
                    {"id": "species_D", "kind": "species", "label": "D"},
                    {"id": "species_E", "kind": "species", "label": "E"},
                    {
                        "id": "reaction_1",
                        "kind": "reaction",
                        "reaction_id": "R1",
                        "reaction_type": "gas",
                    },
                    {
                        "id": "reaction_2",
                        "kind": "reaction",
                        "reaction_id": "R2",
                        "reaction_type": "gas",
                    },
                    {
                        "id": "reaction_3",
                        "kind": "reaction",
                        "reaction_id": "R3",
                        "reaction_type": "gas",
                    },
                    {
                        "id": "reaction_4",
                        "kind": "reaction",
                        "reaction_id": "R4",
                        "reaction_type": "surface",
                    },
                ],
                "links": [
                    {"source": "species_A", "target": "reaction_1", "role": "reactant"},
                    {"source": "species_B", "target": "reaction_1", "role": "reactant"},
                    {"source": "species_C", "target": "reaction_1", "role": "product"},
                    {"source": "species_A", "target": "reaction_2", "role": "reactant"},
                    {"source": "species_B", "target": "reaction_2", "role": "reactant"},
                    {"source": "species_D", "target": "reaction_2", "role": "product"},
                    {"source": "species_E", "target": "reaction_3", "role": "reactant"},
                    {"source": "species_C", "target": "reaction_3", "role": "product"},
                    {"source": "species_A", "target": "reaction_4", "role": "reactant"},
                    {"source": "species_B", "target": "reaction_4", "role": "reactant"},
                    {"source": "species_C", "target": "reaction_4", "role": "product"},
                ],
            }
        }
    }
    _store_graph(store, "graph1", graph_payload)

    task = get("task", "reduction.reaction_lumping")
    cfg = {
        "reduction": {
            "inputs": {"graph": "graph1"},
            "reaction_lumping": {
                "threshold": 0.8,
                "similarity": {"mode": "reactants"},
            },
        }
    }
    result = task(cfg, store=store)

    payload = load_config(result.path / "reaction_lumping.json")
    mapping = {entry["reaction_id"]: entry["representative"] for entry in payload["mapping"]}
    rep_r1 = mapping["R1"]
    assert rep_r1 == mapping["R2"]
    assert rep_r1 in {"R1", "R2"}
    assert mapping["R3"] == "R3"
    assert mapping["R4"] == "R4"
    assert mapping["R4"] != rep_r1

    pair = next(
        item
        for item in payload["similarity"]["pairs"]
        if {item["reaction_id_a"], item["reaction_id_b"]} == {"R1", "R2"}
    )
    assert set(pair["components"].keys()) == {"reactants", "products", "union"}
    assert payload["similarity"]["mode"] == "reactants"

    cluster = next(
        item for item in payload["clusters"] if set(item["members"]) == {"R1", "R2"}
    )
    assert cluster["reaction_type"] == "gas"
