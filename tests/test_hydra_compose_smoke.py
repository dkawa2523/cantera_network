from pathlib import Path

from rxn_platform.core import ArtifactManifest
from rxn_platform.hydra_utils import (
    attach_hydra_config,
    compose_config,
    format_config,
    resolve_config,
    seed_everything,
)


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_hydra_compose_defaults() -> None:
    cfg = compose_config(
        config_path=_config_dir(),
        config_name="defaults",
        overrides=["sim=placeholder", "task=placeholder", "pipeline=placeholder"],
    )
    resolved = resolve_config(cfg)
    assert resolved["common"]["seed"] == 0
    assert resolved["sim"]["name"] == "placeholder"
    assert resolved["task"]["name"] == "placeholder"
    assert resolved["pipeline"]["name"] == "placeholder"
    rendered = format_config(cfg)
    assert "common:" in rendered


def test_seed_and_manifest_attachment() -> None:
    cfg = compose_config(config_path=_config_dir(), config_name="defaults")
    seed = seed_everything(cfg)
    assert seed == 0
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id="run-1",
        created_at="2026-01-17T00:00:00Z",
        parents=[],
        inputs={},
        config={},
        code={},
        provenance={},
    )
    updated = attach_hydra_config(manifest, cfg)
    assert "hydra" in updated.config
