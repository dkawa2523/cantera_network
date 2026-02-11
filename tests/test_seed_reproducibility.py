from __future__ import annotations

from pathlib import Path

from rxn_platform.hydra_utils import compose_config, resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def _run_mapping(store_root: Path, seed: int) -> tuple[str, str]:
    cfg = compose_config(
        config_path=_config_dir(),
        config_name="default",
        overrides=[
            f"store.root={store_root}",
            "recipe=reduce_cnr_coarse",
            f"common.seed={seed}",
        ],
    )
    resolved = resolve_config(cfg)
    store = ArtifactStore(store_root)
    runner = PipelineRunner(store=store)
    results = runner.run(resolved)
    mapping_id = results["mapping"]
    mapping_path = store.artifact_dir("reduction", mapping_id) / "mapping.json"
    return mapping_id, mapping_path.read_text(encoding="utf-8")


def test_seed_reproducibility(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("xarray")
    first_id, first_payload = _run_mapping(tmp_path / "first", seed=123)
    second_id, second_payload = _run_mapping(tmp_path / "second", seed=123)

    assert first_id == second_id
    assert first_payload == second_payload
