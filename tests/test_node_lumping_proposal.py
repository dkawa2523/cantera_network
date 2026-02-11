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


def test_node_lumping_proposal(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    graph_payload = {
        "bipartite": {
            "data": {
                "nodes": [
                    {
                        "id": "species_A",
                        "kind": "species",
                        "label": "A",
                        "elements": {"H": 2},
                        "charge": 0.0,
                        "phase": "gas",
                        "state": "neutral",
                    },
                    {
                        "id": "species_B",
                        "kind": "species",
                        "label": "B",
                        "elements": {"H": 2},
                        "charge": 0.0,
                        "phase": "gas",
                        "state": "neutral",
                    },
                    {
                        "id": "species_C",
                        "kind": "species",
                        "label": "C",
                        "elements": {"H": 1},
                        "charge": 1.0,
                        "phase": "gas",
                        "state": "ion",
                    },
                    {"id": "reaction_1", "kind": "reaction", "reaction_id": "R1"},
                    {"id": "reaction_2", "kind": "reaction", "reaction_id": "R2"},
                ],
                "links": [
                    {"source": "species_A", "target": "reaction_1"},
                    {"source": "species_B", "target": "reaction_1"},
                    {"source": "species_C", "target": "reaction_2"},
                ],
            }
        }
    }
    _store_graph(store, "graph1", graph_payload)

    task = get("task", "reduction.node_lumping")
    cfg = {"reduction": {"inputs": {"graph": "graph1"}, "node_lumping": {"threshold": 0.9}}}
    result = task(cfg, store=store)

    payload = load_config(result.path / "node_lumping.json")
    mapping = {entry["species"]: entry["representative"] for entry in payload["mapping"]}
    rep_ab = mapping["A"]
    assert rep_ab == mapping["B"]
    assert rep_ab in {"A", "B"}
    assert mapping["C"] == "C"

    pair = next(
        item
        for item in payload["similarity"]["pairs"]
        if {item["species_a"], item["species_b"]} == {"A", "B"}
    )
    assert set(pair["components"].keys()) == {
        "elements",
        "charge",
        "phase",
        "state",
        "reaction_type_profile",
        "neighbor_reaction",
    }

    cluster = next(
        item for item in payload["clusters"] if set(item["members"]) == {"A", "B"}
    )
    assert cluster["selection"]["metric"] == "degree"

    # Shared mapping output for downstream consumers (projection QoI, etc.).
    mapping_payload = load_config(result.path / "mapping.json")
    assert mapping_payload["kind"] == "superstate_mapping"
