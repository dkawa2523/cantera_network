"""Optimization task: random search baseline with history + Pareto archive."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
import logging
import math
import platform
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import (
    ArtifactManifest,
    load_config,
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
)
from rxn_platform.errors import ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import Task

try:  # Optional dependency.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:  # Optional dependency.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pq = None

try:  # Optional dependency.
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_SAMPLES = 8
DEFAULT_DIRECTION = "min"
DEFAULT_AGGREGATE = "first"
DEFAULT_MISSING_STRATEGY = "nan"
DEFAULT_HIGH_FIDELITY_SAMPLES = 2
DEFAULT_FIDELITY_SELECTION = "topk"
PATCH_SCHEMA_VERSION = 1
REDUCTION_PATCH_FILENAME = "mechanism_patch.yaml"
REDUCED_MECHANISM_FILENAME = "mechanism.yaml"
REQUIRED_COLUMNS = (
    "sample_id",
    "run_id",
    "observable_id",
    "objective_name",
    "objective",
    "direction",
    "params_json",
    "meta_json",
)


@dataclass(frozen=True)
class ConditionSpec:
    path: str
    low: float
    high: float
    dtype: str


@dataclass(frozen=True)
class MultiplierSpec:
    reaction_id: Optional[str]
    index: Optional[int]
    low: float
    high: float

    def key(self) -> tuple[str, Any]:
        if self.index is not None:
            return ("index", self.index)
        return ("reaction_id", self.reaction_id)


@dataclass(frozen=True)
class ObjectiveSpec:
    target: str
    direction: str
    aggregate: str


@dataclass(frozen=True)
class SampleResult:
    sample_id: int
    run_id: str
    observable_id: str
    params_payload: dict[str, Any]
    objective_values: list[float]
    objective_meta: list[dict[str, Any]]
    valid: bool


def _utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _resolve_cfg(cfg: Any) -> dict[str, Any]:
    try:
        resolved = resolve_config(cfg)
    except ConfigError:
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise
    return resolved


def _extract_optimization_cfg(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "optimization" in cfg:
        opt_cfg = cfg.get("optimization")
        if not isinstance(opt_cfg, Mapping):
            raise ConfigError("optimization config must be a mapping.")
        return dict(cfg), dict(opt_cfg)
    return dict(cfg), dict(cfg)


def _extract_params(opt_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = opt_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("optimization.params must be a mapping.")
    return dict(params)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _extract_sim_cfg(opt_cfg: Mapping[str, Any]) -> dict[str, Any]:
    sim_cfg = opt_cfg.get("sim")
    if sim_cfg is None:
        inputs = opt_cfg.get("inputs")
        if isinstance(inputs, Mapping) and "sim" in inputs:
            sim_cfg = inputs.get("sim")
    if not isinstance(sim_cfg, Mapping):
        raise ConfigError("optimization sim config must be provided as a mapping.")
    return dict(sim_cfg)


def _normalize_observables_cfg(
    raw: Any,
    *,
    missing_strategy: Optional[str],
) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        if "params" in raw:
            params = raw.get("params")
            if not isinstance(params, Mapping):
                raise ConfigError("observables.params must be a mapping.")
            config = dict(params)
        else:
            config = dict(raw)
        config.pop("inputs", None)
        if "observables" not in config and "observables" in raw:
            config["observables"] = raw.get("observables")
    else:
        config = {"observables": raw}
    if missing_strategy is not None:
        config.setdefault("missing_strategy", missing_strategy)
    return config


def _extract_observables_cfg(
    opt_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    observables_raw = None
    for source in (params, opt_cfg):
        if "observables" in source:
            observables_raw = source.get("observables")
            break
        if "observable" in source:
            observables_raw = source.get("observable")
            break
    if observables_raw is None:
        raise ConfigError("optimization observables must be provided.")

    missing_strategy = None
    for source in (params, opt_cfg):
        if "missing_strategy" in source:
            missing_strategy = source.get("missing_strategy")
            break
    if missing_strategy is None:
        missing_strategy = DEFAULT_MISSING_STRATEGY
    if not isinstance(missing_strategy, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = missing_strategy.strip().lower()
    if strategy not in {"nan", "skip"}:
        raise ConfigError("missing_strategy must be 'nan' or 'skip'.")
    return _normalize_observables_cfg(
        observables_raw,
        missing_strategy=strategy,
    )


def _coerce_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer.")
    return value


def _extract_sample_count(opt_cfg: Mapping[str, Any], params: Mapping[str, Any]) -> int:
    count: Any = None
    for source in (params, opt_cfg):
        for key in ("samples", "n_samples", "num_samples", "iterations"):
            if key in source:
                count = source.get(key)
                break
        if count is not None:
            break
    if count is None:
        return DEFAULT_SAMPLES
    if isinstance(count, bool):
        raise ConfigError("samples must be an integer.")
    try:
        value = int(count)
    except (TypeError, ValueError) as exc:
        raise ConfigError("samples must be an integer.") from exc
    if value <= 0:
        raise ConfigError("samples must be positive.")
    return value


def _extract_multi_fidelity_cfg(
    opt_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    for source in (params, opt_cfg):
        if "multi_fidelity" in source:
            value = source.get("multi_fidelity")
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError("multi_fidelity config must be a mapping.")
            return dict(value)
    return {}


def _extract_high_fidelity_samples(
    mf_cfg: Mapping[str, Any],
    sample_count: int,
) -> int:
    raw: Any = None
    for key in (
        "high_fidelity_samples",
        "high_fidelity",
        "high_fidelity_count",
        "top_k",
        "topK",
        "n_high_fidelity",
    ):
        if key in mf_cfg:
            raw = mf_cfg.get(key)
            break
    if raw is None:
        return min(DEFAULT_HIGH_FIDELITY_SAMPLES, sample_count)
    if isinstance(raw, bool):
        raise ConfigError("high_fidelity_samples must be an integer.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("high_fidelity_samples must be an integer.") from exc
    if value < 0:
        raise ConfigError("high_fidelity_samples must be >= 0.")
    return min(value, sample_count)


def _extract_seed(cfg: Mapping[str, Any]) -> int:
    for path in ("common.seed", "seed"):
        current: Any = cfg
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if current is None:
            continue
        if isinstance(current, bool):
            raise ConfigError("seed must be an integer.")
        try:
            return int(current)
        except (TypeError, ValueError) as exc:
            raise ConfigError("seed must be an integer.") from exc
    return 0


def _normalize_direction(value: Any) -> str:
    if value is None:
        return DEFAULT_DIRECTION
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("direction must be a non-empty string.")
    key = value.strip().lower()
    if key in {"min", "minimize", "minimise"}:
        return "min"
    if key in {"max", "maximize", "maximise"}:
        return "max"
    raise ConfigError("direction must be 'min' or 'max'.")


def _normalize_aggregate(value: Any) -> str:
    if value is None:
        return DEFAULT_AGGREGATE
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("aggregate must be a non-empty string.")
    key = value.strip().lower()
    if key in {"first", "mean", "min", "max", "sum"}:
        return key
    raise ConfigError("aggregate must be one of: first, mean, min, max, sum.")


def _normalize_objective_entry(value: Any, label: str) -> ObjectiveSpec:
    if isinstance(value, str):
        target = _require_nonempty_str(value, label)
        direction = _normalize_direction(None)
        aggregate = _normalize_aggregate(None)
        return ObjectiveSpec(target=target, direction=direction, aggregate=aggregate)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a string or mapping.")
    target = value.get("target") or value.get("observable")
    target = _require_nonempty_str(target, f"{label}.target")
    direction = _normalize_direction(value.get("direction"))
    aggregate = _normalize_aggregate(value.get("aggregate"))
    return ObjectiveSpec(target=target, direction=direction, aggregate=aggregate)


def _extract_objectives(
    opt_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[ObjectiveSpec]:
    objective_raw = None
    for source in (params, opt_cfg):
        if "objectives" in source:
            objective_raw = source.get("objectives")
            break
    if objective_raw is None:
        for source in (params, opt_cfg):
            if "objective" in source:
                objective_raw = source.get("objective")
                break
            if "target" in source:
                objective_raw = source.get("target")
                break
    if objective_raw is None:
        raise ConfigError("optimization objective must be provided.")
    if isinstance(objective_raw, Sequence) and not isinstance(
        objective_raw, (str, bytes, bytearray, Mapping)
    ):
        if not objective_raw:
            raise ConfigError("objectives must not be empty.")
        specs: list[ObjectiveSpec] = []
        for index, entry in enumerate(objective_raw):
            specs.append(_normalize_objective_entry(entry, f"objectives[{index}]"))
        return specs
    return [_normalize_objective_entry(objective_raw, "objective")]


def _coerce_numeric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number.") from exc


def _normalize_condition_specs(value: Any) -> list[ConditionSpec]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("conditions must be a list of mappings.")
    specs: list[ConditionSpec] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"conditions[{index}] must be a mapping.")
        path = entry.get("path") or entry.get("name")
        path = _require_nonempty_str(path, f"conditions[{index}].path")
        low = _coerce_numeric(entry.get("low", entry.get("min")), f"{path}.low")
        high = _coerce_numeric(entry.get("high", entry.get("max")), f"{path}.high")
        if high < low:
            raise ConfigError(f"{path} range must satisfy low <= high.")
        dtype = entry.get("dtype") or entry.get("type") or entry.get("kind")
        if entry.get("integer") is True:
            dtype = "int"
        if dtype is None:
            dtype = "float"
        if not isinstance(dtype, str):
            raise ConfigError(f"{path} dtype must be a string.")
        dtype_key = dtype.strip().lower()
        if dtype_key not in {"float", "int"}:
            raise ConfigError(f"{path} dtype must be 'float' or 'int'.")
        specs.append(ConditionSpec(path=path, low=low, high=high, dtype=dtype_key))
    return specs


def _normalize_multiplier_specs(value: Any) -> list[MultiplierSpec]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("multipliers must be a list of mappings.")
    specs: list[MultiplierSpec] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"multipliers[{index}] must be a mapping.")
        reaction_id = entry.get("reaction_id") or entry.get("reaction")
        idx = _coerce_optional_int(entry.get("index"), f"multipliers[{index}].index")
        if reaction_id is None and idx is None:
            raise ConfigError(f"multipliers[{index}] must set reaction_id or index.")
        if reaction_id is not None and idx is not None:
            raise ConfigError(f"multipliers[{index}] must set only one of reaction_id or index.")
        if reaction_id is not None:
            reaction_id = _require_nonempty_str(reaction_id, f"multipliers[{index}].reaction_id")
        low = _coerce_numeric(entry.get("low", entry.get("min")), f"multipliers[{index}].low")
        high = _coerce_numeric(entry.get("high", entry.get("max")), f"multipliers[{index}].high")
        if high < low:
            raise ConfigError("multiplier range must satisfy low <= high.")
        specs.append(
            MultiplierSpec(reaction_id=reaction_id, index=idx, low=low, high=high)
        )
    return specs


def _extract_search_space(
    opt_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[list[ConditionSpec], list[MultiplierSpec]]:
    space: Any = None
    for source in (params, opt_cfg):
        if "search_space" in source:
            space = source.get("search_space")
            break
        if "space" in source:
            space = source.get("space")
            break
    if space is None:
        conditions = None
        multipliers = None
        for source in (params, opt_cfg):
            if "conditions" in source:
                conditions = source.get("conditions")
            if "multipliers" in source:
                multipliers = source.get("multipliers")
        condition_specs = _normalize_condition_specs(conditions)
        multiplier_specs = _normalize_multiplier_specs(multipliers)
        if not condition_specs and not multiplier_specs:
            raise ConfigError("search_space must include conditions or multipliers.")
        return condition_specs, multiplier_specs

    if not isinstance(space, Mapping):
        raise ConfigError("search_space must be a mapping.")
    condition_specs = _normalize_condition_specs(space.get("conditions"))
    multiplier_specs = _normalize_multiplier_specs(space.get("multipliers"))
    if not condition_specs and not multiplier_specs:
        raise ConfigError("search_space must include conditions or multipliers.")
    return condition_specs, multiplier_specs


def _set_nested_value(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts:
        raise ConfigError("condition path must be non-empty.")
    current = payload
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        if not isinstance(node, Mapping):
            raise ConfigError(f"condition path {path!r} conflicts with existing value.")
        if not isinstance(node, dict):
            node = dict(node)
            current[part] = node
        current = node
    current[parts[-1]] = value


def _multiplier_sort_key(entry: Mapping[str, Any]) -> tuple[int, Any]:
    if "index" in entry:
        return (0, entry["index"])
    return (1, entry["reaction_id"])


def _build_multiplier_map(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, Any], float]:
    mapping: dict[tuple[str, Any], float] = {}
    for entry in entries:
        if "index" in entry:
            key = ("index", entry["index"])
        else:
            key = ("reaction_id", entry["reaction_id"])
        mapping[key] = float(entry.get("multiplier", 1.0))
    return mapping


def _rebuild_multipliers(mapping: Mapping[tuple[str, Any], float]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, multiplier in mapping.items():
        kind, value = key
        entry: dict[str, Any] = {"multiplier": multiplier}
        if kind == "index":
            entry["index"] = value
        else:
            entry["reaction_id"] = value
        entries.append(entry)
    return sorted(entries, key=_multiplier_sort_key)


def _normalize_sim_cfg(sim_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, Any], float]]:
    normalized = dict(sim_cfg)
    try:
        multipliers = normalize_reaction_multipliers(sim_cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"reaction multipliers are invalid: {exc}") from exc
    if multipliers:
        normalized["reaction_multipliers"] = list(multipliers)
    else:
        normalized.pop("reaction_multipliers", None)
    normalized.pop("disabled_reactions", None)
    return normalized, _build_multiplier_map(multipliers)


def _extract_reduction_cfg(mf_cfg: Mapping[str, Any]) -> dict[str, Any]:
    reduction_cfg: dict[str, Any] = {}
    for key in ("reduction", "reduced", "reduced_model", "low_fidelity"):
        if key in mf_cfg:
            value = mf_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            reduction_cfg = dict(value)
            break
    for key in (
        "patch",
        "patches",
        "mechanism_patch",
        "patch_file",
        "reduction_id",
        "id",
        "mechanism",
        "mechanism_path",
        "solution",
    ):
        if key in mf_cfg and key not in reduction_cfg:
            reduction_cfg[key] = mf_cfg.get(key)
    return reduction_cfg


def _coerce_path(value: Any, label: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _load_patch_payload(patch_value: Any) -> tuple[dict[str, Any], Optional[str]]:
    if isinstance(patch_value, Mapping):
        return dict(patch_value), None
    if isinstance(patch_value, Path):
        patch_value = str(patch_value)
    if isinstance(patch_value, str):
        patch_path = Path(patch_value)
        if not patch_path.exists():
            raise ConfigError(f"patch file not found: {patch_value}")
        payload = load_config(patch_path)
        if not isinstance(payload, Mapping):
            raise ConfigError("patch file must contain a mapping.")
        return dict(payload), str(patch_path)
    raise ConfigError("patch must be a mapping or path to YAML/JSON.")


def _normalize_patch_payload(
    patch_data: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema_version = patch_data.get("schema_version", PATCH_SCHEMA_VERSION)
    if schema_version is None:
        schema_version = PATCH_SCHEMA_VERSION
    if isinstance(schema_version, bool):
        raise ConfigError("patch.schema_version must be an integer.")
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ConfigError("patch.schema_version must be an integer.") from exc
    if schema_version < 1:
        raise ConfigError("patch.schema_version must be >= 1.")

    disabled_raw = patch_data.get("disabled_reactions")
    multipliers_raw = patch_data.get("reaction_multipliers")
    try:
        disabled_entries = normalize_reaction_multipliers(
            {"disabled_reactions": disabled_raw}
        )
        multiplier_entries = normalize_reaction_multipliers(
            {"reaction_multipliers": multipliers_raw}
        )
        combined_entries = normalize_reaction_multipliers(
            {
                "disabled_reactions": disabled_raw,
                "reaction_multipliers": multipliers_raw,
            }
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"patch entries are invalid: {exc}") from exc

    if not combined_entries:
        raise ConfigError("patch must include disabled_reactions or reaction_multipliers.")

    disabled_payload: list[dict[str, Any]] = []
    for entry in disabled_entries:
        if "index" in entry:
            disabled_payload.append({"index": entry["index"]})
        else:
            disabled_payload.append({"reaction_id": entry["reaction_id"]})

    normalized_patch = {
        "schema_version": schema_version,
        "disabled_reactions": disabled_payload,
        "reaction_multipliers": list(multiplier_entries),
    }
    return normalized_patch, combined_entries


def _extract_patch_multipliers(
    patch_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    multipliers_raw = patch_payload.get("reaction_multipliers") or []
    try:
        multipliers = normalize_reaction_multipliers(
            {"reaction_multipliers": multipliers_raw}
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"patch reaction_multipliers are invalid: {exc}") from exc
    filtered: list[dict[str, Any]] = []
    for entry in multipliers:
        try:
            multiplier = float(entry.get("multiplier", 1.0))
        except (TypeError, ValueError):
            multiplier = 0.0
        if multiplier == 0.0:
            continue
        filtered.append(dict(entry))
    return filtered


def _reaction_identifiers(reaction: Any, index: int) -> list[str]:
    identifiers: list[str] = [f"R{index + 1}"]
    if isinstance(reaction, Mapping):
        for key in ("id", "name", "equation", "reaction"):
            value = reaction.get(key)
            if isinstance(value, str) and value.strip():
                identifiers.append(value.strip())
    elif isinstance(reaction, str) and reaction.strip():
        identifiers.append(reaction.strip())
    else:
        raise ConfigError("reaction entries must be mappings or strings.")
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def _reaction_id_index_map(
    reactions: Sequence[Any],
) -> dict[str, list[int]]:
    id_map: dict[str, list[int]] = {}
    for idx, reaction in enumerate(reactions):
        identifiers = _reaction_identifiers(reaction, idx)
        for identifier in identifiers:
            id_map.setdefault(identifier, []).append(idx)
    return id_map


def _resolve_patch_entries(
    entries: Sequence[Mapping[str, Any]],
    reactions: Sequence[Any],
) -> dict[int, dict[str, Any]]:
    if not reactions:
        raise ConfigError("mechanism has no reactions to patch.")
    id_map = _reaction_id_index_map(reactions)
    resolved: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if "index" in entry:
            idx = entry["index"]
            if not isinstance(idx, int):
                raise ConfigError("patch reaction index must be an int.")
            if idx < 0 or idx >= len(reactions):
                raise ConfigError(f"patch reaction index out of range: {idx}.")
        else:
            reaction_id = entry.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id.strip():
                raise ConfigError("patch reaction_id must be a non-empty string.")
            indices = id_map.get(reaction_id)
            if not indices:
                raise ConfigError(f"reaction_id not found in mechanism: {reaction_id!r}.")
            if len(indices) > 1:
                raise ConfigError(
                    f"reaction_id matches multiple reactions: {reaction_id!r}."
                )
            idx = indices[0]
        resolved[idx] = dict(entry)
    return resolved


def _apply_patch_entries(
    mechanism: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], set[int]]:
    reactions = mechanism.get("reactions")
    if reactions is None:
        raise ConfigError("mechanism must define a reactions list.")
    if not isinstance(reactions, Sequence) or isinstance(
        reactions,
        (str, bytes, bytearray),
    ):
        raise ConfigError("mechanism.reactions must be a sequence.")
    reaction_list = list(reactions)
    resolved = _resolve_patch_entries(entries, reaction_list)
    disabled_indices = {
        idx for idx, entry in resolved.items() if float(entry.get("multiplier", 0.0)) == 0.0
    }
    filtered_reactions = [
        reaction
        for idx, reaction in enumerate(reaction_list)
        if idx not in disabled_indices
    ]
    updated = dict(mechanism)
    updated["reactions"] = filtered_reactions
    return updated, disabled_indices


def _write_yaml_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    sort_keys: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=sort_keys,
                ensure_ascii=True,
            )
            handle.write("\n")
            return
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=sort_keys,
        )


def _prepare_reduced_mechanism(
    *,
    mechanism_path: Optional[str],
    reduction_dir: Optional[Path],
    combined_entries: Sequence[Mapping[str, Any]],
    disabled_entries: Sequence[Mapping[str, Any]],
    logger: logging.Logger,
) -> tuple[Optional[str], Optional[tempfile.TemporaryDirectory[str]], bool]:
    if reduction_dir is not None:
        reduced_mechanism = reduction_dir / REDUCED_MECHANISM_FILENAME
        if reduced_mechanism.exists():
            return str(reduced_mechanism), None, False
    if disabled_entries and mechanism_path:
        temp_dir = tempfile.TemporaryDirectory(prefix="rxn_mf_reduction_")
        reduced_mechanism = Path(temp_dir.name) / REDUCED_MECHANISM_FILENAME
        mechanism_payload = load_config(mechanism_path)
        if not isinstance(mechanism_payload, Mapping):
            raise ConfigError("mechanism YAML must be a mapping.")
        reduced_payload, _ = _apply_patch_entries(
            dict(mechanism_payload),
            combined_entries,
        )
        _write_yaml_payload(reduced_mechanism, reduced_payload, sort_keys=False)
        return str(reduced_mechanism), temp_dir, False
    if disabled_entries and not mechanism_path:
        logger.warning(
            "multi_fidelity patch disables reactions but no mechanism was provided; "
            "falling back to multiplier=0 overrides."
        )
        return None, None, True
    return None, None, False


def _apply_patch_multiplier_map(
    base_map: Mapping[tuple[str, Any], float],
    patch_entries: Sequence[Mapping[str, Any]],
    *,
    override: bool,
) -> dict[tuple[str, Any], float]:
    mapping = dict(base_map)
    for entry in patch_entries:
        if "index" in entry:
            key = ("index", entry["index"])
        else:
            key = ("reaction_id", entry["reaction_id"])
        try:
            multiplier = float(entry.get("multiplier", 1.0))
        except (TypeError, ValueError):
            multiplier = 0.0
        if override or key not in mapping:
            mapping[key] = multiplier
    return mapping


def _score_sample(sample: SampleResult, directions: Sequence[str]) -> float:
    if not sample.valid:
        return math.inf
    score = 0.0
    for value, direction in zip(sample.objective_values, directions):
        if direction == "min":
            score += value
        else:
            score -= value
    return score


def _select_high_fidelity_samples(
    samples: Sequence[SampleResult],
    directions: Sequence[str],
    count: int,
) -> list[int]:
    if count <= 0:
        return []
    ranked = sorted(samples, key=lambda sample: _score_sample(sample, directions))
    return [sample.sample_id for sample in ranked[:count]]


def _sim_run_id(sim_cfg: Mapping[str, Any]) -> str:
    manifest_cfg = {"sim": sim_cfg, "inputs": {}, "params": {}}
    return make_run_id(manifest_cfg, exclude_keys=("hydra",))


def _run_observables(
    runner: PipelineRunner,
    run_id: str,
    observables_cfg: Mapping[str, Any],
) -> str:
    pipeline_cfg = {
        "steps": [
            {
                "id": "obs",
                "task": "observables.run",
                "inputs": {"run_id": run_id},
                "params": dict(observables_cfg),
            }
        ]
    }
    results = runner.run(pipeline_cfg)
    return results["obs"]


def _run_sim_and_observables(
    runner: PipelineRunner,
    store: ArtifactStore,
    sim_cfg: Mapping[str, Any],
    observables_cfg: Mapping[str, Any],
) -> tuple[str, str]:
    run_id = _sim_run_id(sim_cfg)
    if store.exists("runs", run_id):
        obs_id = _run_observables(runner, run_id, observables_cfg)
        return run_id, obs_id
    pipeline_cfg = {
        "steps": [
            {"id": "sim", "task": "sim.run", "sim": dict(sim_cfg)},
            {
                "id": "obs",
                "task": "observables.run",
                "inputs": {"run_id": "@sim"},
                "params": dict(observables_cfg),
            },
        ]
    }
    results = runner.run(pipeline_cfg)
    return results["sim"], results["obs"]


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _aggregate_values(values: Sequence[float], method: str) -> float:
    if not values:
        return math.nan
    if method == "first":
        return values[0]
    if method == "mean":
        return sum(values) / float(len(values))
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    if method == "sum":
        return sum(values)
    return math.nan


def _is_valid_objective_vector(values: Sequence[float]) -> bool:
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _dominates(
    candidate: Sequence[float],
    other: Sequence[float],
    directions: Sequence[str],
) -> bool:
    strictly_better = False
    for value, other_value, direction in zip(candidate, other, directions):
        if direction == "min":
            if value > other_value:
                return False
            if value < other_value:
                strictly_better = True
        else:
            if value < other_value:
                return False
            if value > other_value:
                strictly_better = True
    return strictly_better


def _pareto_front_sample_ids(
    samples: Sequence[SampleResult],
    directions: Sequence[str],
) -> list[int]:
    valid_samples = [sample for sample in samples if sample.valid]
    front_ids: list[int] = []
    for candidate in valid_samples:
        dominated = False
        for other in valid_samples:
            if other.sample_id == candidate.sample_id:
                continue
            if _dominates(other.objective_values, candidate.objective_values, directions):
                dominated = True
                break
        if not dominated:
            front_ids.append(candidate.sample_id)
    return front_ids


def _extract_objective_value(
    store: ArtifactStore,
    observable_id: str,
    run_id: str,
    objective: ObjectiveSpec,
) -> tuple[float, dict[str, Any]]:
    obs_dir = store.artifact_dir("observables", observable_id)
    values_path = obs_dir / "values.parquet"
    if not values_path.exists():
        raise ConfigError(f"Observable values not found: {values_path}")
    rows = _read_table_rows(values_path)
    matches = []
    units: list[str] = []
    meta_jsons: list[str] = []
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        if row.get("observable") != objective.target:
            continue
        matches.append(row)
        units.append(str(row.get("unit", "")))
        meta_json = row.get("meta_json")
        if isinstance(meta_json, str):
            meta_jsons.append(meta_json)
    if not matches:
        return math.nan, {
            "status": "missing_target",
            "target": objective.target,
        }
    values: list[float] = []
    invalid = 0
    for row in matches:
        value = row.get("value")
        if value is None:
            invalid += 1
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            invalid += 1
    if not values:
        return math.nan, {
            "status": "non_numeric",
            "target": objective.target,
            "invalid_count": invalid,
        }
    meta: dict[str, Any] = {
        "status": "ok",
        "target": objective.target,
        "aggregate": objective.aggregate,
        "count": len(values),
    }
    if invalid:
        meta["invalid_count"] = invalid
    if units:
        meta["unit"] = units[0]
    if meta_jsons:
        meta["observable_meta_json"] = meta_jsons[0]
    return _aggregate_values(values, objective.aggregate), meta


def _coerce_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    return {"detail": meta}


def _coerce_value(value: Any, meta: dict[str, Any]) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        meta["error"] = "value must be numeric"
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        meta["error"] = f"value not numeric: {value!r}"
        return math.nan


def _build_row(
    sample_id: int,
    run_id: str,
    observable_id: str,
    objective_name: str,
    objective_value: Any,
    direction: str,
    params_payload: Mapping[str, Any],
    meta: Any,
) -> dict[str, Any]:
    params_json = json.dumps(
        params_payload,
        ensure_ascii=True,
        sort_keys=True,
    )
    meta_payload = _coerce_meta(meta)
    objective = _coerce_value(objective_value, meta_payload)
    meta_json = json.dumps(
        meta_payload,
        ensure_ascii=True,
        sort_keys=True,
    )
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "observable_id": observable_id,
        "objective_name": objective_name,
        "objective": objective,
        "direction": direction,
        "params_json": params_json,
        "meta_json": meta_json,
    }


def _collect_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(REQUIRED_COLUMNS)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_history_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = _collect_columns(rows)
    if pd is not None:
        frame = pd.DataFrame(list(rows), columns=columns)
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            pass
    if pa is not None and pq is not None:
        table = pa.Table.from_pylist(list(rows))
        table = table.select(columns)
        pq.write_table(table, path)
        return
    payload = {"columns": columns, "rows": list(rows)}
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger = logging.getLogger("rxn_platform.optimization")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
    )


def _code_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {"version": __version__}
    git_dir = Path.cwd() / ".git"
    if not git_dir.exists():
        return payload
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["git_commit"] = commit
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return payload
    return payload


def _provenance_metadata() -> dict[str, Any]:
    return {"python": platform.python_version()}


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _sample_conditions(
    specs: Sequence[ConditionSpec],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {}
    updates: dict[str, Any] = {}
    for spec in specs:
        if spec.dtype == "int":
            value = rng.randint(int(spec.low), int(spec.high))
        else:
            value = rng.uniform(spec.low, spec.high)
        params[spec.path] = value
        updates[spec.path] = value
    return params, updates


def _sample_multipliers(
    specs: Sequence[MultiplierSpec],
    base_map: Mapping[tuple[str, Any], float],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[tuple[str, Any], float]]:
    mapping = dict(base_map)
    params: list[dict[str, Any]] = []
    for spec in specs:
        value = rng.uniform(spec.low, spec.high)
        key = spec.key()
        mapping[key] = value
        entry: dict[str, Any] = {"multiplier": value}
        if spec.index is not None:
            entry["index"] = spec.index
        else:
            entry["reaction_id"] = spec.reaction_id
        params.append(entry)
    params = sorted(params, key=_multiplier_sort_key) if params else []
    return params, mapping


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run a random search optimization and store history + Pareto archive."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, opt_cfg = _extract_optimization_cfg(resolved_cfg)
    params = _extract_params(opt_cfg)

    base_sim_cfg = _extract_sim_cfg(opt_cfg)
    observables_cfg = _extract_observables_cfg(opt_cfg, params)
    objectives = _extract_objectives(opt_cfg, params)
    sample_count = _extract_sample_count(opt_cfg, params)
    condition_specs, multiplier_specs = _extract_search_space(opt_cfg, params)
    seed = _extract_seed(resolved_cfg)
    rng = random.Random(seed)

    normalized_sim_cfg, base_multiplier_map = _normalize_sim_cfg(base_sim_cfg)

    logger = logging.getLogger("rxn_platform.optimization")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    rows: list[dict[str, Any]] = []
    samples: list[SampleResult] = []
    run_ids: list[str] = []
    observable_ids: list[str] = []

    for sample_id in range(sample_count):
        sim_cfg = copy.deepcopy(normalized_sim_cfg)
        params_payload: dict[str, Any] = {}

        condition_params, condition_updates = _sample_conditions(condition_specs, rng)
        for path, value in condition_updates.items():
            _set_nested_value(sim_cfg, path, value)
        if condition_params:
            params_payload["conditions"] = condition_params

        multiplier_params: list[dict[str, Any]] = []
        if multiplier_specs:
            multiplier_params, multiplier_map = _sample_multipliers(
                multiplier_specs, base_multiplier_map, rng
            )
            multipliers = _rebuild_multipliers(multiplier_map)
            if multipliers:
                sim_cfg["reaction_multipliers"] = multipliers
            else:
                sim_cfg.pop("reaction_multipliers", None)
            sim_cfg.pop("disabled_reactions", None)
            params_payload["multipliers"] = multiplier_params

        run_id, observable_id = _run_sim_and_observables(
            runner,
            store,
            sim_cfg,
            observables_cfg,
        )
        run_ids.append(run_id)
        observable_ids.append(observable_id)

        objective_values: list[float] = []
        objective_meta_list: list[dict[str, Any]] = []
        for obj_index, objective in enumerate(objectives):
            objective_value, objective_meta = _extract_objective_value(
                store,
                observable_id,
                run_id,
                objective,
            )
            objective_values.append(objective_value)
            objective_meta_list.append(objective_meta)
            row_meta = {
                "objective_meta": objective_meta,
                "direction": objective.direction,
                "objective_index": obj_index,
                "objective_count": len(objectives),
            }
            rows.append(
                _build_row(
                    sample_id,
                    run_id,
                    observable_id,
                    objective.target,
                    objective_value,
                    objective.direction,
                    params_payload,
                    row_meta,
                )
            )
        samples.append(
            SampleResult(
                sample_id=sample_id,
                run_id=run_id,
                observable_id=observable_id,
                params_payload=params_payload,
                objective_values=objective_values,
                objective_meta=objective_meta_list,
                valid=_is_valid_objective_vector(objective_values),
            )
        )

    directions = [objective.direction for objective in objectives]
    pareto_sample_ids = set(_pareto_front_sample_ids(samples, directions))
    pareto_rows = [row for row in rows if row.get("sample_id") in pareto_sample_ids]

    inputs_payload = {
        "sample_count": sample_count,
        "seed": seed,
        "objectives": [
            {
                "target": objective.target,
                "direction": objective.direction,
                "aggregate": objective.aggregate,
            }
            for objective in objectives
        ],
        "search_space": {
            "conditions": [
                {
                    "path": spec.path,
                    "low": spec.low,
                    "high": spec.high,
                    "dtype": spec.dtype,
                }
                for spec in condition_specs
            ],
            "multipliers": [
                {
                    "reaction_id": spec.reaction_id,
                    "index": spec.index,
                    "low": spec.low,
                    "high": spec.high,
                }
                for spec in multiplier_specs
            ],
        },
        "run_ids": list(run_ids),
        "observable_ids": list(observable_ids),
    }
    if len(objectives) == 1:
        inputs_payload["objective"] = inputs_payload["objectives"][0]
        inputs_payload.pop("objectives", None)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parents = _dedupe_preserve(run_ids + observable_ids)
    manifest = ArtifactManifest(
        schema_version=1,
        kind="optimization",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_history_table(rows, base_dir / "history.parquet")
        _write_history_table(pareto_rows, base_dir / "pareto.parquet")

    return store.ensure(manifest, writer=_writer)


def run_multi_fidelity(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run a multi-fidelity random search using a reduced mechanism patch."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, opt_cfg = _extract_optimization_cfg(resolved_cfg)
    params = _extract_params(opt_cfg)

    base_sim_cfg = _extract_sim_cfg(opt_cfg)
    observables_cfg = _extract_observables_cfg(opt_cfg, params)
    objectives = _extract_objectives(opt_cfg, params)
    sample_count = _extract_sample_count(opt_cfg, params)
    condition_specs, multiplier_specs = _extract_search_space(opt_cfg, params)
    seed = _extract_seed(resolved_cfg)
    rng = random.Random(seed)

    mf_cfg = _extract_multi_fidelity_cfg(opt_cfg, params)
    if not mf_cfg:
        raise ConfigError("multi_fidelity config must be provided.")
    high_fidelity_samples = _extract_high_fidelity_samples(mf_cfg, sample_count)
    reduction_cfg = _extract_reduction_cfg(mf_cfg)

    reduction_id = reduction_cfg.get("reduction_id") or reduction_cfg.get("id")
    if reduction_id is not None:
        reduction_id = _require_nonempty_str(reduction_id, "multi_fidelity.reduction_id")

    patch_value: Any = None
    for key in ("patch", "patches", "mechanism_patch", "patch_file"):
        if key in reduction_cfg:
            patch_value = reduction_cfg.get(key)
            break
    if patch_value is None and (
        "disabled_reactions" in reduction_cfg or "reaction_multipliers" in reduction_cfg
    ):
        patch_value = reduction_cfg

    if reduction_id is not None and patch_value is not None:
        raise ConfigError("Specify only one of reduction_id or patch for multi_fidelity.")
    if reduction_id is None and patch_value is None:
        raise ConfigError("multi_fidelity requires a reduction patch or reduction_id.")

    logger = logging.getLogger("rxn_platform.optimization")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    reduction_dir: Optional[Path] = None
    patch_payload: dict[str, Any]
    patch_source: Optional[str] = None
    if reduction_id is not None:
        reduction_dir = store.artifact_dir("reduction", reduction_id)
        patch_path = reduction_dir / REDUCTION_PATCH_FILENAME
        if not patch_path.exists():
            raise ConfigError(f"reduction patch not found: {patch_path}")
        patch_payload = load_config(patch_path)
        if not isinstance(patch_payload, Mapping):
            raise ConfigError("reduction patch payload must be a mapping.")
        patch_payload = dict(patch_payload)
    else:
        patch_payload, patch_source = _load_patch_payload(patch_value)

    normalized_patch, combined_entries = _normalize_patch_payload(patch_payload)
    patch_multipliers = _extract_patch_multipliers(normalized_patch)
    disabled_entries = list(normalized_patch.get("disabled_reactions", []))

    mechanism_path = (
        reduction_cfg.get("mechanism")
        or reduction_cfg.get("mechanism_path")
        or reduction_cfg.get("solution")
        or base_sim_cfg.get("mechanism")
    )
    if mechanism_path is not None:
        mechanism_path = _coerce_path(mechanism_path, "multi_fidelity.mechanism")
        if not Path(mechanism_path).exists():
            raise ConfigError(f"mechanism file not found: {mechanism_path}")

    reduced_mechanism_path: Optional[str]
    temp_dir: Optional[tempfile.TemporaryDirectory[str]]
    apply_disabled_as_multipliers: bool
    reduced_mechanism_path, temp_dir, apply_disabled_as_multipliers = (
        _prepare_reduced_mechanism(
            mechanism_path=mechanism_path,
            reduction_dir=reduction_dir,
            combined_entries=combined_entries,
            disabled_entries=disabled_entries,
            logger=logger,
        )
    )

    if apply_disabled_as_multipliers:
        for entry in disabled_entries:
            merged = dict(entry)
            merged["multiplier"] = 0.0
            patch_multipliers.append(merged)

    normalized_sim_cfg, base_multiplier_map = _normalize_sim_cfg(base_sim_cfg)

    rows: list[dict[str, Any]] = []
    low_samples: list[SampleResult] = []
    high_samples: list[SampleResult] = []
    run_ids: list[str] = []
    observable_ids: list[str] = []
    sample_states: list[dict[str, Any]] = []

    try:
        for sample_id in range(sample_count):
            sim_cfg = copy.deepcopy(normalized_sim_cfg)
            params_payload: dict[str, Any] = {}

            condition_params, condition_updates = _sample_conditions(
                condition_specs, rng
            )
            for path, value in condition_updates.items():
                _set_nested_value(sim_cfg, path, value)
            if condition_params:
                params_payload["conditions"] = condition_params

            multiplier_map = dict(base_multiplier_map)
            multiplier_params: list[dict[str, Any]] = []
            if multiplier_specs:
                multiplier_params, multiplier_map = _sample_multipliers(
                    multiplier_specs, base_multiplier_map, rng
                )
                multipliers = _rebuild_multipliers(multiplier_map)
                if multipliers:
                    sim_cfg["reaction_multipliers"] = multipliers
                else:
                    sim_cfg.pop("reaction_multipliers", None)
                sim_cfg.pop("disabled_reactions", None)
                params_payload["multipliers"] = multiplier_params

            sample_states.append(
                {
                    "sample_id": sample_id,
                    "sim_cfg": sim_cfg,
                    "params_payload": params_payload,
                    "multiplier_map": multiplier_map,
                }
            )

            low_sim_cfg = copy.deepcopy(sim_cfg)
            if reduced_mechanism_path is not None:
                low_sim_cfg["mechanism"] = reduced_mechanism_path
            if patch_multipliers:
                patched_map = _apply_patch_multiplier_map(
                    multiplier_map, patch_multipliers, override=True
                )
                multipliers = _rebuild_multipliers(patched_map)
                if multipliers:
                    low_sim_cfg["reaction_multipliers"] = multipliers
                else:
                    low_sim_cfg.pop("reaction_multipliers", None)
                low_sim_cfg.pop("disabled_reactions", None)

            run_id, observable_id = _run_sim_and_observables(
                runner,
                store,
                low_sim_cfg,
                observables_cfg,
            )
            run_ids.append(run_id)
            observable_ids.append(observable_id)

            objective_values: list[float] = []
            objective_meta_list: list[dict[str, Any]] = []
            for obj_index, objective in enumerate(objectives):
                objective_value, objective_meta = _extract_objective_value(
                    store,
                    observable_id,
                    run_id,
                    objective,
                )
                objective_values.append(objective_value)
                objective_meta_list.append(objective_meta)
                row_meta = {
                    "objective_meta": objective_meta,
                    "direction": objective.direction,
                    "objective_index": obj_index,
                    "objective_count": len(objectives),
                }
                row = _build_row(
                    sample_id,
                    run_id,
                    observable_id,
                    objective.target,
                    objective_value,
                    objective.direction,
                    params_payload,
                    row_meta,
                )
                row["fidelity"] = "low"
                rows.append(row)
            low_samples.append(
                SampleResult(
                    sample_id=sample_id,
                    run_id=run_id,
                    observable_id=observable_id,
                    params_payload=params_payload,
                    objective_values=objective_values,
                    objective_meta=objective_meta_list,
                    valid=_is_valid_objective_vector(objective_values),
                )
            )

        directions = [objective.direction for objective in objectives]
        high_sample_ids = _select_high_fidelity_samples(
            low_samples,
            directions,
            high_fidelity_samples,
        )

        for state in sample_states:
            if state["sample_id"] not in high_sample_ids:
                continue
            run_id, observable_id = _run_sim_and_observables(
                runner,
                store,
                state["sim_cfg"],
                observables_cfg,
            )
            run_ids.append(run_id)
            observable_ids.append(observable_id)

            objective_values = []
            objective_meta_list = []
            for obj_index, objective in enumerate(objectives):
                objective_value, objective_meta = _extract_objective_value(
                    store,
                    observable_id,
                    run_id,
                    objective,
                )
                objective_values.append(objective_value)
                objective_meta_list.append(objective_meta)
                row_meta = {
                    "objective_meta": objective_meta,
                    "direction": objective.direction,
                    "objective_index": obj_index,
                    "objective_count": len(objectives),
                }
                row = _build_row(
                    state["sample_id"],
                    run_id,
                    observable_id,
                    objective.target,
                    objective_value,
                    objective.direction,
                    state["params_payload"],
                    row_meta,
                )
                row["fidelity"] = "high"
                rows.append(row)
            high_samples.append(
                SampleResult(
                    sample_id=state["sample_id"],
                    run_id=run_id,
                    observable_id=observable_id,
                    params_payload=state["params_payload"],
                    objective_values=objective_values,
                    objective_meta=objective_meta_list,
                    valid=_is_valid_objective_vector(objective_values),
                )
            )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    pareto_samples = high_samples if high_samples else low_samples
    pareto_fidelity = "high" if high_samples else "low"
    pareto_sample_ids = set(_pareto_front_sample_ids(pareto_samples, directions))
    pareto_rows = [
        row
        for row in rows
        if row.get("sample_id") in pareto_sample_ids
        and row.get("fidelity") == pareto_fidelity
    ]

    fidelity_payload: dict[str, Any] = {
        "low_fidelity_samples": sample_count,
        "high_fidelity_samples": len(high_sample_ids),
        "selection": DEFAULT_FIDELITY_SELECTION,
        "pareto_fidelity": pareto_fidelity,
        "high_fidelity_sample_ids": list(high_sample_ids),
    }
    if reduction_id is not None:
        fidelity_payload["reduction_id"] = reduction_id
    if normalized_patch:
        fidelity_payload["patch"] = normalized_patch
    if mechanism_path is not None:
        fidelity_payload["mechanism"] = mechanism_path
    if patch_source is not None:
        fidelity_payload["patch_source"] = patch_source

    inputs_payload = {
        "mode": "multi_fidelity",
        "sample_count": sample_count,
        "seed": seed,
        "objectives": [
            {
                "target": objective.target,
                "direction": objective.direction,
                "aggregate": objective.aggregate,
            }
            for objective in objectives
        ],
        "search_space": {
            "conditions": [
                {
                    "path": spec.path,
                    "low": spec.low,
                    "high": spec.high,
                    "dtype": spec.dtype,
                }
                for spec in condition_specs
            ],
            "multipliers": [
                {
                    "reaction_id": spec.reaction_id,
                    "index": spec.index,
                    "low": spec.low,
                    "high": spec.high,
                }
                for spec in multiplier_specs
            ],
        },
        "run_ids": list(run_ids),
        "observable_ids": list(observable_ids),
        "fidelity": fidelity_payload,
    }
    if len(objectives) == 1:
        inputs_payload["objective"] = inputs_payload["objectives"][0]
        inputs_payload.pop("objectives", None)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parents = _dedupe_preserve(run_ids + observable_ids)
    if reduction_id is not None:
        parents = _dedupe_preserve(parents + [reduction_id])
    manifest = ArtifactManifest(
        schema_version=1,
        kind="optimization",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_history_table(rows, base_dir / "history.parquet")
        _write_history_table(pareto_rows, base_dir / "pareto.parquet")
        summary_path = base_dir / "fidelity_summary.json"
        summary_path.write_text(
            json.dumps(fidelity_payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


class MultiFidelityOptimizer(Task):
    name = "optimization.multi_fidelity"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run_multi_fidelity(cfg, store=store, registry=registry)


class RandomSearchOptimizer(Task):
    name = "optimization.random_search"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run(cfg, store=store, registry=registry)


register("task", "optimization.random_search", RandomSearchOptimizer())
register("task", "optimization.multi_fidelity", MultiFidelityOptimizer())

__all__ = ["MultiFidelityOptimizer", "RandomSearchOptimizer", "run", "run_multi_fidelity"]
