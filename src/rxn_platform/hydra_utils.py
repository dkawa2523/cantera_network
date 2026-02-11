"""Hydra config composition and manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import random
from typing import Any, Optional, Union

from rxn_platform.core import ArtifactManifest
from rxn_platform.errors import ConfigError

try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional dependency
    compose = None
    initialize_config_dir = None
    GlobalHydra = None
    OmegaConf = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

DEFAULT_CONFIG_PATH = "configs"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_SEED_PATHS = ("common.seed", "seed")


def _require_hydra() -> None:
    if compose is None or initialize_config_dir is None or GlobalHydra is None:
        raise ConfigError("hydra-core is required to compose configs.")


def _normalize_overrides(overrides: Optional[Sequence[str]]) -> list[str]:
    if not overrides:
        return []
    normalized: list[str] = []
    skip_next = False
    for idx, item in enumerate(overrides):
        if skip_next:
            skip_next = False
            continue
        if not item or item == "--":
            continue
        if item.startswith("--use_surrogate"):
            if item == "--use_surrogate":
                value = "true"
                if idx + 1 < len(overrides):
                    candidate = overrides[idx + 1]
                    if candidate and not candidate.startswith("-") and "=" not in candidate:
                        value = candidate
                        skip_next = True
                normalized.append(f"use_surrogate={value}")
            else:
                if "=" in item:
                    _, value = item.split("=", 1)
                    value = value or "true"
                else:
                    value = "true"
                normalized.append(f"use_surrogate={value}")
            continue
        if item == "--clearml" or item.startswith("--clearml="):
            if item == "--clearml":
                value = "true"
                if idx + 1 < len(overrides):
                    candidate = overrides[idx + 1]
                    if candidate and not candidate.startswith("-") and "=" not in candidate:
                        value = candidate
                        skip_next = True
                normalized.append(f"clearml={value}")
            else:
                _, value = item.split("=", 1)
                value = value or "true"
                normalized.append(f"clearml={value}")
            continue
        if (
            item in {"--dry_run", "--dry-run"}
            or item.startswith("--dry_run=")
            or item.startswith("--dry-run=")
        ):
            flag = "--dry_run" if item.startswith("--dry_run") else "--dry-run"
            if item == flag:
                value = "true"
                if idx + 1 < len(overrides):
                    candidate = overrides[idx + 1]
                    if candidate and not candidate.startswith("-") and "=" not in candidate:
                        value = candidate
                        skip_next = True
                normalized.append(f"dry_run={value}")
            else:
                _, value = item.split("=", 1)
                value = value or "true"
                normalized.append(f"dry_run={value}")
            continue
        normalized.append(item)
    return normalized


def _normalize_config_name(config_name: str) -> str:
    if config_name.endswith((".yaml", ".yml")):
        return Path(config_name).stem
    return config_name


def compose_config(
    *,
    config_path: Union[Path, str] = DEFAULT_CONFIG_PATH,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: Optional[Sequence[str]] = None,
) -> Any:
    _require_hydra()
    try:
        from rxn_platform.config.schema import register_configs
    except Exception as exc:  # pragma: no cover - config registration should be safe
        raise ConfigError(f"Failed to load structured config registry: {exc}") from exc
    register_configs()
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    if not config_dir.exists():
        raise ConfigError(f"Config directory not found: {config_dir}")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name=_normalize_config_name(config_name),
            overrides=_normalize_overrides(overrides),
        )


def resolve_config(cfg: Any) -> dict[str, Any]:
    if OmegaConf is None:
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise ConfigError("OmegaConf is required to resolve Hydra config.")
    if not OmegaConf.is_config(cfg):
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise ConfigError("OmegaConf is required to resolve Hydra config.")
    resolved = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    if not isinstance(resolved, dict):
        raise ConfigError("Resolved config must be a mapping.")
    return resolved


def format_config(cfg: Any) -> str:
    if OmegaConf is None:
        return json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    return OmegaConf.to_yaml(cfg, resolve=True)


def _select_from_mapping(cfg: Mapping[str, Any], path: str) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _extract_seed(cfg: Any, seed_paths: Sequence[str]) -> Optional[int]:
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        for path in seed_paths:
            value = OmegaConf.select(cfg, path, default=None)
            if value is not None:
                return value
    if isinstance(cfg, Mapping):
        for path in seed_paths:
            value = _select_from_mapping(cfg, path)
            if value is not None:
                return value
    return None


def seed_everything(
    cfg: Any,
    *,
    seed_paths: Sequence[str] = DEFAULT_SEED_PATHS,
) -> Optional[int]:
    seed = _extract_seed(cfg, seed_paths)
    if seed is None:
        return None
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Seed must be an integer, got {seed!r}.") from exc
    random.seed(seed_int)
    if np is not None:
        np.random.seed(seed_int)
    return seed_int


def attach_hydra_config(
    manifest: ArtifactManifest,
    cfg: Any,
) -> ArtifactManifest:
    if not hasattr(manifest, "to_dict"):
        raise TypeError("manifest must be an ArtifactManifest instance.")
    resolved = resolve_config(cfg)
    payload = manifest.to_dict()
    config = dict(payload.get("config", {}))
    config["hydra"] = resolved
    payload["config"] = config
    return ArtifactManifest.from_dict(payload)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_SEED_PATHS",
    "compose_config",
    "resolve_config",
    "format_config",
    "seed_everything",
    "attach_hydra_config",
]
