import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _base_cfg() -> dict:
    return {
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        }
    }


def test_multipliers_change_run_id(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sim.run")

    cfg_a = _base_cfg()
    cfg_a["sim"]["reaction_multipliers"] = [
        {"index": 0, "multiplier": 0.5}
    ]
    cfg_b = _base_cfg()
    cfg_b["sim"]["reaction_multipliers"] = [
        {"index": 0, "multiplier": 0.75}
    ]

    result_a = task(cfg_a, store=store)
    result_b = task(cfg_b, store=store)

    assert result_a.manifest.id != result_b.manifest.id


def test_multipliers_order_keeps_run_id(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sim.run")

    cfg_a = _base_cfg()
    cfg_a["sim"]["reaction_multipliers"] = [
        {"index": 1, "multiplier": 1.5},
        {"index": 0, "multiplier": 0.25},
    ]
    cfg_b = _base_cfg()
    cfg_b["sim"]["reaction_multipliers"] = [
        {"index": 0, "multiplier": 0.25},
        {"index": 1, "multiplier": 1.5},
    ]

    result_a = task(cfg_a, store=store)
    result_b = task(cfg_b, store=store)

    assert result_a.manifest.id == result_b.manifest.id
    assert result_b.reused


def test_manifest_records_multipliers(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sim.run")

    cfg = _base_cfg()
    cfg["sim"]["reaction_multipliers"] = [
        {"index": 1, "multiplier": 2.0}
    ]
    cfg["sim"]["disabled_reactions"] = [0]

    result = task(cfg, store=store)

    assert result.manifest.inputs["reaction_multipliers"] == [
        {"index": 0, "multiplier": 0.0},
        {"index": 1, "multiplier": 2.0},
    ]
