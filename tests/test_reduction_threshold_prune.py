import json

import rxn_platform.tasks.reduction  # noqa: F401

from rxn_platform.core import ArtifactManifest, load_config
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _store_artifact(store: ArtifactStore, *, kind: str, artifact_id: str, files: dict) -> None:
    manifest = ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at="2026-01-01T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )

    def _writer(base_dir):
        for name, payload in files.items():
            path = base_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    store.ensure(manifest, writer=_writer)


def _store_sensitivity(
    store: ArtifactStore,
    artifact_id: str,
    rows: list[dict],
) -> None:
    payload = {"rows": rows}
    _store_artifact(
        store,
        kind="sensitivity",
        artifact_id=artifact_id,
        files={"sensitivity.parquet": payload},
    )


def _store_graph(store: ArtifactStore, artifact_id: str, nodes: list[dict]) -> None:
    payload = {"bipartite": {"data": {"nodes": nodes}}}
    _store_artifact(
        store,
        kind="graphs",
        artifact_id=artifact_id,
        files={"graph.json": payload},
    )


def test_reduction_threshold_prune_score_threshold_protection(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    _store_sensitivity(
        store,
        "sens1",
        [
            {"reaction_id": "R1", "reaction_index": 0, "value": 0.05},
            {"reaction_id": "R2", "reaction_index": 1, "value": 0.15},
            {"reaction_id": "R3", "reaction_index": 2, "value": 1.5},
        ],
    )
    _store_graph(
        store,
        "graph1",
        [
            {
                "id": "reaction_1",
                "kind": "reaction",
                "reaction_id": "R1",
                "reaction_index": 0,
                "reaction_type": "surface",
            },
            {
                "id": "reaction_2",
                "kind": "reaction",
                "reaction_id": "R2",
                "reaction_index": 1,
                "reaction_type": "gas",
            },
            {
                "id": "reaction_3",
                "kind": "reaction",
                "reaction_id": "R3",
                "reaction_index": 2,
                "reaction_type": "gas",
            },
        ],
    )

    task = get("task", "reduction.threshold_prune")
    cfg = {
        "reduction": {
            "inputs": {"sensitivity": "sens1", "graph": "graph1"},
            "threshold": {"score_lt": 0.2},
            "protect": {"reaction_types": ["surface"]},
        }
    }
    result = task(cfg, store=store)

    patch_payload = load_config(result.path / "mechanism_patch.yaml")
    assert patch_payload["disabled_reactions"] == [{"reaction_id": "R2"}]
    assert patch_payload["reaction_multipliers"] == []


def test_reduction_threshold_prune_top_k(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    _store_sensitivity(
        store,
        "sens2",
        [
            {"reaction_id": "R1", "reaction_index": 0, "value": 0.05},
            {"reaction_id": "R2", "reaction_index": 1, "value": 0.15},
            {"reaction_id": "R3", "reaction_index": 2, "value": 1.5},
        ],
    )

    task = get("task", "reduction.threshold_prune")
    cfg = {"reduction": {"inputs": {"sensitivity": "sens2"}, "threshold": {"top_k": 1}}}
    result = task(cfg, store=store)

    patch_payload = load_config(result.path / "mechanism_patch.yaml")
    disabled = {entry["reaction_id"] for entry in patch_payload["disabled_reactions"]}
    assert disabled == {"R1", "R2"}
