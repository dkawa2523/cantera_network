"""Structured config schema for Hydra."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any


@dataclass
class CommonConfig:
    seed: int = 0


@dataclass
class RunConfig:
    exp: str = "${exp}"
    run_id: str = "${run_id}"
    root: str = "runs/${run.exp}/${run.run_id}"


@dataclass
class StoreConfig:
    root: str = "${run.root}/artifacts"


@dataclass
class RecipeConfig:
    name: str = "smoke"
    description: str = ""
    notes: str = ""


@dataclass
class AppConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    exp: str = "default"
    run_id: str = "auto"
    use_surrogate: bool = False
    clearml: bool = False
    dry_run: bool = False
    run: RunConfig = field(default_factory=RunConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    # Ad-hoc mechanism metadata/paths used by benchmark pipelines (ex: mechanism.path).
    # Keep keys explicit so `mechanism.path=...` overrides work under Hydra struct mode.
    mechanism: dict[str, Any] = field(default_factory=lambda: {"path": ""})
    # Top-level knobs used by assimilation pipelines (ex: assimilation.obs_file).
    assimilation: dict[str, Any] = field(
        default_factory=lambda: {"obs_file": "benchmarks/assets/observations/gri30_netbench_obs.csv"}
    )
    sim: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    compare: dict[str, Any] = field(default_factory=dict)
    eval_mapping: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    try:
        from hydra.core.config_store import ConfigStore
    except ModuleNotFoundError:
        return
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Hydra config store import failed: %s", exc
        )
        return
    cs = ConfigStore.instance()
    try:
        cs.store(group="schema", name="base", node=AppConfig, package="_global_")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Hydra config store registration failed: %s", exc
        )


__all__ = ["AppConfig", "register_configs"]
