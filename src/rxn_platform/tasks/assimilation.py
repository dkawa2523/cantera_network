"""Assimilation utilities: parameterization and misfit helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import logging
import math
from pathlib import Path
import random
from typing import Any, Optional
from rxn_platform.core import (
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
    resolve_repo_path,
)
from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import Task
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    read_table_rows as _read_table_rows,
    resolve_cfg as _resolve_cfg,
)

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
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

DEFAULT_AGGREGATE = "mean"
DEFAULT_MISSING_STRATEGY = "nan"
DEFAULT_ENSEMBLE_SIZE = 8
DEFAULT_ITERATIONS = 3
DEFAULT_INFLATION = 1.0
DEFAULT_RIDGE = 1.0e-6
DEFAULT_STEP_SIZE = 1.0
DEFAULT_LAPLACIAN_LAMBDA = 0.0


def _extract_assim_cfg(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "assimilation" in cfg:
        assim_cfg = cfg.get("assimilation")
        if not isinstance(assim_cfg, Mapping):
            raise ConfigError("assimilation config must be a mapping.")
        return dict(cfg), dict(assim_cfg)
    return dict(cfg), dict(cfg)


def _extract_params(assim_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = assim_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("assimilation.params must be a mapping.")
    return dict(params)


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


def _extract_sim_cfg(
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    sim_cfg: Any = None
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "sim" in source:
            sim_cfg = source.get("sim")
            break
        inputs = source.get("inputs")
        if isinstance(inputs, Mapping) and "sim" in inputs:
            sim_cfg = inputs.get("sim")
            break
    if not isinstance(sim_cfg, Mapping):
        raise ConfigError("assimilation sim config must be provided as a mapping.")
    return dict(sim_cfg)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ConfigError(f"observations file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ConfigError(f"observations file is empty: {path}")
    return rows


def _apply_csv_condition(
    sim_cfg: Mapping[str, Any],
    *,
    temperature: Optional[float],
    pressure_atm: Optional[float],
    phi: Optional[float],
    t_end: Optional[float],
    case_id: Optional[str],
) -> dict[str, Any]:
    updated = dict(sim_cfg)
    initial = dict(updated.get("initial", {}))
    if temperature is not None:
        initial["T"] = temperature
    if pressure_atm is not None:
        initial["P"] = pressure_atm * 101325.0
    if phi is not None:
        if phi <= 0.0:
            raise ConfigError("phi must be positive.")
        composition = dict(initial.get("X") or {})
        composition["CH4"] = 1.0
        composition["O2"] = 2.0 / float(phi)
        composition["N2"] = composition["O2"] * 3.76
        initial["X"] = composition
    if initial:
        updated["initial"] = initial

    if t_end is not None:
        if "time_grid" in updated and isinstance(updated.get("time_grid"), Mapping):
            time_grid = dict(updated.get("time_grid") or {})
            time_grid["stop"] = t_end
            updated["time_grid"] = time_grid
        elif "time" in updated and isinstance(updated.get("time"), Mapping):
            time_cfg = dict(updated.get("time") or {})
            time_cfg["stop"] = t_end
            updated["time"] = time_cfg
        else:
            updated["time_grid"] = {"start": 0.0, "stop": t_end, "steps": 4}
    if case_id:
        updated["condition_id"] = case_id
    return updated


def _extract_case_id(
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    sim_cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        for key in ("observed_case_id", "case_id", "condition_id"):
            if key in source and source.get(key) is not None:
                value = source.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError("case_id must be a non-empty string.")
                return value.strip()
    if sim_cfg and isinstance(sim_cfg.get("condition_id"), str):
        value = sim_cfg.get("condition_id")
        if value and value.strip():
            return value.strip()
    return None


def _extract_obs_file(
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[Path]:
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        for key in ("obs_file", "observed_file", "observations_file", "obs_path"):
            if key in source and source.get(key) is not None:
                value = source.get(key)
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ConfigError("obs_file must be a non-empty string or Path.")
                return resolve_repo_path(value)
    return None


def _extract_obs_columns(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, str]:
    columns: Any = None
    for source in (params, assim_cfg):
        if not isinstance(source, Mapping):
            continue
        if "obs_columns" in source:
            columns = source.get("obs_columns")
            break
    if columns is None:
        return {
            "case_id": "case_id",
            "observable": "observable",
            "value": "value",
            "sigma": "sigma",
            "noise": "noise",
            "weight": "weight",
            "aggregate": "aggregate",
        }
    if not isinstance(columns, Mapping):
        raise ConfigError("obs_columns must be a mapping.")
    payload = {
        "case_id": columns.get("case_id", "case_id"),
        "observable": columns.get("observable", "observable"),
        "value": columns.get("value", "value"),
        "sigma": columns.get("sigma", "sigma"),
        "noise": columns.get("noise", "noise"),
        "weight": columns.get("weight", "weight"),
        "aggregate": columns.get("aggregate", "aggregate"),
    }
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"obs_columns.{key} must be a non-empty string.")
        payload[key] = value.strip()
    return payload


def _extract_obs_mapping(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, str]:
    mapping: Any = None
    for source in (params, assim_cfg):
        if not isinstance(source, Mapping):
            continue
        for key in ("obs_mapping", "observable_map", "observed_map"):
            if key in source:
                mapping = source.get(key)
                break
        if mapping is not None:
            break
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ConfigError("obs_mapping must be a mapping.")
    result: dict[str, str] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError("obs_mapping keys must be non-empty strings.")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("obs_mapping values must be non-empty strings.")
        result[key.strip()] = value.strip()
    return result


def _extract_conditions_file_settings(
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    sim_cfg: Mapping[str, Any],
) -> tuple[Optional[Path], Optional[str], Optional[int], str]:
    sources: list[Mapping[str, Any]] = []
    for source in (params, assim_cfg, resolved_cfg, sim_cfg):
        if isinstance(source, Mapping):
            sources.append(source)
        benchmarks = source.get("benchmarks") if isinstance(source, Mapping) else None
        if isinstance(benchmarks, Mapping):
            sources.append(benchmarks)

    conditions_file: Optional[Any] = None
    for source in sources:
        for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
            if key in source and source.get(key) is not None:
                conditions_file = source.get(key)
                break
        if conditions_file is not None:
            break
    if conditions_file is None:
        return None, None, None, "case_id"
    if not isinstance(conditions_file, (str, Path)) or not str(conditions_file).strip():
        raise ConfigError("conditions_file must be a non-empty string or Path.")
    conditions_path = resolve_repo_path(conditions_file)

    case_id: Optional[str] = None
    row_index: Optional[int] = None
    case_col: Optional[str] = None
    for source in sources:
        if case_id is None:
            for key in ("case_id", "condition_id", "case"):
                if key in source and source.get(key) is not None:
                    value = source.get(key)
                    if not isinstance(value, str) or not value.strip():
                        raise ConfigError("case_id must be a non-empty string.")
                    case_id = value.strip()
                    break
        if row_index is None and "row_index" in source:
            value = source.get("row_index")
            if isinstance(value, bool):
                raise ConfigError("row_index must be an integer.")
            try:
                row_index = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError("row_index must be an integer.") from exc
        if case_col is None:
            for key in ("case_column", "case_col", "case_field"):
                if key in source and source.get(key) is not None:
                    case_col = source.get(key)
                    break
        if case_id is not None and row_index is not None and case_col is not None:
            break
    if case_col is None:
        case_col = "case_id"
    if not isinstance(case_col, str) or not case_col.strip():
        raise ConfigError("case_column must be a non-empty string.")
    case_col = case_col.strip()
    return conditions_path, case_id, row_index, case_col


def _select_csv_row(
    rows: list[dict[str, str]],
    *,
    case_id: Optional[str],
    row_index: Optional[int],
    case_col: str,
) -> dict[str, str]:
    if case_id:
        for row in rows:
            if row.get(case_col) == case_id:
                return row
        raise ConfigError(f"case_id {case_id!r} not found in conditions file.")
    if row_index is None:
        row_index = 0
    if row_index < 0 or row_index >= len(rows):
        raise ConfigError("row_index out of range for conditions file.")
    return rows[row_index]


def _apply_conditions_file_to_sim(
    sim_cfg: Mapping[str, Any],
    *,
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    settings = _extract_conditions_file_settings(resolved_cfg, assim_cfg, params, sim_cfg)
    conditions_path, case_id, row_index, case_col = settings
    if conditions_path is None:
        return dict(sim_cfg)
    rows = _load_csv_rows(conditions_path)
    row = _select_csv_row(rows, case_id=case_id, row_index=row_index, case_col=case_col)

    def _pick(keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in row and row.get(key) is not None:
                value = row.get(key)
                if isinstance(value, str) and not value.strip():
                    return None
                return value
        return None

    def _optional_float(value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return _coerce_float(value, label)

    temperature = _optional_float(_pick(("T0", "T", "temperature")), "temperature")
    pressure_atm = _optional_float(
        _pick(("P0_atm", "P_atm", "P0", "pressure_atm", "pressure")),
        "pressure_atm",
    )
    phi = _optional_float(_pick(("phi",)), "phi")
    t_end = _optional_float(_pick(("t_end", "t_end_s", "t_end_seconds")), "t_end")
    row_case_id = row.get(case_col) if case_col in row else None
    if isinstance(row_case_id, str) and not row_case_id.strip():
        row_case_id = None
    if case_id is None and row_case_id is not None:
        case_id = row_case_id

    return _apply_csv_condition(
        sim_cfg,
        temperature=temperature,
        pressure_atm=pressure_atm,
        phi=phi,
        t_end=t_end,
        case_id=case_id,
    )


def _load_observed_rows_from_file(
    obs_path: Path,
    *,
    case_id: Optional[str],
    columns: Mapping[str, str],
    mapping: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = _load_csv_rows(obs_path)
    selected: list[dict[str, Any]] = []
    case_col = columns.get("case_id", "case_id")
    obs_col = columns.get("observable", "observable")
    value_col = columns.get("value", "value")
    sigma_col = columns.get("sigma", "sigma")
    noise_col = columns.get("noise", "noise")
    weight_col = columns.get("weight", "weight")
    aggregate_col = columns.get("aggregate", "aggregate")

    for row in rows:
        if case_id is not None and case_col in row and row.get(case_col) != case_id:
            continue
        target = row.get(obs_col)
        if target is None or (isinstance(target, str) and not target.strip()):
            raise ConfigError("obs_file rows must include observable values.")
        target_name = str(target)
        if target_name in mapping:
            target_name = mapping[target_name]
        value = row.get(value_col)
        if isinstance(value, str) and not value.strip():
            value = None
        sigma = row.get(noise_col)
        if sigma is None:
            sigma = row.get(sigma_col)
        if isinstance(sigma, str) and not sigma.strip():
            sigma = None
        weight = row.get(weight_col)
        if isinstance(weight, str) and not weight.strip():
            weight = None
        aggregate = row.get(aggregate_col)
        payload: dict[str, Any] = {
            "observable": target_name,
            "value": _coerce_float(value, "observed.value"),
        }
        if sigma is not None:
            payload["noise"] = _coerce_optional_float(sigma, "observed.noise", default=1.0)
        if weight is not None:
            payload["weight"] = _coerce_optional_float(weight, "observed.weight", default=1.0)
        if aggregate is not None and str(aggregate).strip():
            payload["aggregate"] = str(aggregate).strip()
        selected.append(payload)

    if not selected:
        raise ConfigError("No observations matched obs_file filters.")
    return selected


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


def _extract_missing_strategy(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    missing_strategy: Any = None
    for source in (params, assim_cfg):
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
    return strategy


def _extract_observables_cfg(
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    missing_strategy: Optional[str],
) -> dict[str, Any]:
    observables_raw: Any = None
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "observables" in source:
            observables_raw = source.get("observables")
            break
        if "observable" in source:
            observables_raw = source.get("observable")
            break
    if observables_raw is None:
        raise ConfigError("assimilation observables must be provided.")
    return _normalize_observables_cfg(
        observables_raw,
        missing_strategy=missing_strategy,
    )


def _extract_default_aggregate(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    aggregate: Any = None
    for source in (params, assim_cfg):
        if "aggregate" in source:
            aggregate = source.get("aggregate")
            break
        if "default_aggregate" in source:
            aggregate = source.get("default_aggregate")
            break
    return _normalize_aggregate(aggregate)


def _extract_ensemble_size(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int:
    value: Any = None
    for source in (params, assim_cfg):
        if "ensemble_size" in source:
            value = source.get("ensemble_size")
            break
    if value is None:
        return DEFAULT_ENSEMBLE_SIZE
    if isinstance(value, bool):
        raise ConfigError("ensemble_size must be an integer.")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("ensemble_size must be an integer.") from exc
    if count <= 0:
        raise ConfigError("ensemble_size must be positive.")
    return count


def _extract_iterations(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int:
    value: Any = None
    for source in (params, assim_cfg):
        if "iterations" in source:
            value = source.get("iterations")
            break
        if "n_iterations" in source:
            value = source.get("n_iterations")
            break
    if value is None:
        return DEFAULT_ITERATIONS
    if isinstance(value, bool):
        raise ConfigError("iterations must be an integer.")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("iterations must be an integer.") from exc
    if count <= 0:
        raise ConfigError("iterations must be positive.")
    return count


def _extract_inflation(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    value: Any = None
    for source in (params, assim_cfg):
        if "inflation" in source:
            value = source.get("inflation")
            break
    if value is None:
        return DEFAULT_INFLATION
    inflation = _coerce_float(value, "inflation")
    if inflation <= 0:
        raise ConfigError("inflation must be > 0.")
    return inflation


def _extract_ridge(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    value: Any = None
    for source in (params, assim_cfg):
        if "ridge" in source:
            value = source.get("ridge")
            break
        if "ridge_reg" in source:
            value = source.get("ridge_reg")
            break
    if value is None:
        return DEFAULT_RIDGE
    ridge = _coerce_float(value, "ridge")
    if ridge < 0:
        raise ConfigError("ridge must be >= 0.")
    return ridge


def _extract_step_size(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    value: Any = None
    for source in (params, assim_cfg):
        if "step_size" in source:
            value = source.get("step_size")
            break
        if "learning_rate" in source:
            value = source.get("learning_rate")
            break
    if value is None:
        return DEFAULT_STEP_SIZE
    step = _coerce_float(value, "step_size")
    if step <= 0:
        raise ConfigError("step_size must be > 0.")
    return step


def _extract_laplacian_cfg(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    config: Any = None
    for source in (params, assim_cfg):
        if not isinstance(source, Mapping):
            continue
        for key in ("laplacian", "laplacian_regularizer", "graph_laplacian"):
            if key in source:
                config = source.get(key)
                break
        if config is not None:
            break
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ConfigError("laplacian config must be a mapping.")
    return dict(config)


def _extract_laplacian_lambda(laplacian_cfg: Mapping[str, Any]) -> float:
    value: Any = None
    for key in ("lambda", "strength", "weight", "penalty"):
        if key in laplacian_cfg:
            value = laplacian_cfg.get(key)
            break
    if value is None:
        return DEFAULT_LAPLACIAN_LAMBDA
    strength = _coerce_float(value, "laplacian.lambda")
    if strength < 0:
        raise ConfigError("laplacian.lambda must be >= 0.")
    return strength


def _extract_alpha_schedule(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    iterations: int,
) -> list[float]:
    value: Any = None
    for source in (params, assim_cfg):
        if "alpha_schedule" in source:
            value = source.get("alpha_schedule")
            break
        if "alphas" in source:
            value = source.get("alphas")
            break
        if "alpha" in source:
            value = source.get("alpha")
            break
        if "esmda_alpha" in source:
            value = source.get("esmda_alpha")
            break
    if value is None:
        return [float(iterations)] * iterations
    if isinstance(value, bool):
        raise ConfigError("alpha_schedule must be a number or sequence of numbers.")
    if isinstance(value, (int, float)):
        alpha = _coerce_float(value, "alpha")
        if alpha <= 0:
            raise ConfigError("alpha must be > 0.")
        return [alpha] * iterations
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        schedule = [
            _coerce_float(entry, f"alpha_schedule[{idx}]")
            for idx, entry in enumerate(value)
        ]
        if len(schedule) != iterations:
            raise ConfigError("alpha_schedule length must match iterations.")
        if any(alpha <= 0 for alpha in schedule):
            raise ConfigError("alpha_schedule values must be > 0.")
        return schedule
    raise ConfigError("alpha_schedule must be a number or sequence of numbers.")


def _extract_columns(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> MisfitColumns:
    columns: Any = None
    for source in (params, assim_cfg):
        if "columns" in source:
            columns = source.get("columns")
            break
        if "misfit_columns" in source:
            columns = source.get("misfit_columns")
            break
    return MisfitColumns.from_config(columns)


def _parse_default_prior(value: Any) -> Optional[PriorDistribution]:
    if value is None:
        return None
    if isinstance(value, PriorDistribution):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError("default_prior must be a mapping.")
    return _parse_prior({"prior": value}, default_prior=None, label="default_prior")


def _extract_parameter_specs(
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    specs: Any = None
    for source in (params, assim_cfg):
        if "parameters" in source:
            specs = source.get("parameters")
            break
        if "parameter_specs" in source:
            specs = source.get("parameter_specs")
            break
    if specs is None:
        raise ConfigError("assimilation parameter specs must be provided.")
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes, bytearray)):
        raise ConfigError("assimilation parameter specs must be a sequence.")
    return list(specs)


def _normalize_initial_ensemble(
    value: Any,
    parameter_vector: ParameterVector,
) -> Optional[list[list[float]]]:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("initial_ensemble must be a sequence of samples.")
    samples: list[list[float]] = []
    names = parameter_vector.names()
    for index, entry in enumerate(value):
        if isinstance(entry, Mapping):
            values: list[float] = []
            for name in names:
                if name not in entry:
                    raise ConfigError(
                        f"initial_ensemble[{index}] missing parameter {name!r}."
                    )
                values.append(_coerce_float(entry[name], f"initial_ensemble[{index}].{name}"))
            samples.append(values)
        elif isinstance(entry, Sequence) and not isinstance(
            entry, (str, bytes, bytearray)
        ):
            values = list(entry)
            if len(values) != len(names):
                raise ConfigError(
                    f"initial_ensemble[{index}] must have {len(names)} values."
                )
            samples.append(
                [
                    _coerce_float(value, f"initial_ensemble[{index}][{idx}]")
                    for idx, value in enumerate(values)
                ]
            )
        else:
            raise ConfigError(
                f"initial_ensemble[{index}] must be a mapping or sequence."
            )
    return samples


def _extract_observed_rows(
    store: ArtifactStore,
    resolved_cfg: Mapping[str, Any],
    assim_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    sim_cfg: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    observed: Any = None
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "observed" in source:
            observed = source.get("observed")
            break
        if "observations" in source:
            observed = source.get("observations")
            break
    if observed is not None:
        return _coerce_rows(observed, "observed")

    obs_path = _extract_obs_file(resolved_cfg, assim_cfg, params)
    if obs_path is not None:
        case_id = _extract_case_id(resolved_cfg, assim_cfg, params, sim_cfg)
        columns = _extract_obs_columns(assim_cfg, params)
        mapping = _extract_obs_mapping(assim_cfg, params)
        return _load_observed_rows_from_file(
            obs_path,
            case_id=case_id,
            columns=columns,
            mapping=mapping,
        )

    observed_id: Any = None
    observed_run_id: Any = None
    for source in (params, assim_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "observed_observable_id" in source:
            observed_id = source.get("observed_observable_id")
        elif "observed_observable" in source:
            observed_id = source.get("observed_observable")
        if "observed_run_id" in source:
            observed_run_id = source.get("observed_run_id")
        if observed_id is not None:
            break

    if observed_id is None:
        raise ConfigError("observed data or observed_observable_id must be provided.")
    observed_id = _require_nonempty_str(observed_id, "observed_observable_id")
    if observed_run_id is not None:
        observed_run_id = _require_nonempty_str(observed_run_id, "observed_run_id")
    return load_observable_rows(store, observed_id, run_id=observed_run_id)

def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number.") from exc


def _coerce_optional_bool(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise ConfigError(f"{label} must be a boolean.")


def _coerce_optional_float(
    value: Any,
    label: str,
    *,
    default: float,
) -> float:
    if value is None:
        return default
    return _coerce_float(value, label)


def _coerce_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer.")
    return value


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


def _normalize_aggregate(value: Any) -> str:
    if value is None:
        return DEFAULT_AGGREGATE
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("aggregate must be a non-empty string.")
    key = value.strip().lower()
    if key in {"first", "mean", "min", "max", "sum"}:
        return key
    raise ConfigError("aggregate must be one of: first, mean, min, max, sum.")


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _coerce_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise ConfigError(f"{label}[{index}] must be a mapping.")
            rows.append(dict(entry))
        return rows
    raise ConfigError(f"{label} must be a mapping or sequence of mappings.")


def _build_multiplier_map(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, Any], float]:
    mapping: dict[tuple[str, Any], float] = {}
    for entry in entries:
        if "index" in entry:
            key = ("index", entry["index"])
        else:
            key = ("reaction_id", entry["reaction_id"])
        mapping[key] = float(entry.get("multiplier", 1.0))
    return mapping


def _multiplier_sort_key(entry: Mapping[str, Any]) -> tuple[int, Any]:
    if "index" in entry:
        return (0, entry["index"])
    return (1, entry["reaction_id"])


def _rebuild_multipliers(mapping: Mapping[tuple[str, Any], float]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, multiplier in mapping.items():
        kind, value = key
        entry: dict[str, Any] = {"multiplier": float(multiplier)}
        if kind == "index":
            entry["index"] = value
        else:
            entry["reaction_id"] = value
        entries.append(entry)
    return sorted(entries, key=_multiplier_sort_key)


class PriorDistribution:
    """Base class for prior distributions."""

    def sample(self, rng: random.Random) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPrior(PriorDistribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        low = _coerce_float(self.low, "uniform.low")
        high = _coerce_float(self.high, "uniform.high")
        if high < low:
            raise ConfigError("uniform prior must satisfy low <= high.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogNormalPrior(PriorDistribution):
    mean: float
    sigma: float
    low: Optional[float] = None
    high: Optional[float] = None
    max_tries: int = 128

    def __post_init__(self) -> None:
        mean = _coerce_float(self.mean, "lognormal.mean")
        sigma = _coerce_float(self.sigma, "lognormal.sigma")
        if sigma < 0:
            raise ConfigError("lognormal.sigma must be >= 0.")
        low = None if self.low is None else _coerce_float(self.low, "lognormal.low")
        high = None if self.high is None else _coerce_float(self.high, "lognormal.high")
        if low is not None and low <= 0:
            raise ConfigError("lognormal.low must be > 0 when provided.")
        if high is not None and low is not None and high < low:
            raise ConfigError("lognormal bounds must satisfy low <= high.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample(self, rng: random.Random) -> float:
        if self.low is None and self.high is None:
            return rng.lognormvariate(self.mean, self.sigma)
        candidate = rng.lognormvariate(self.mean, self.sigma)
        for _ in range(self.max_tries):
            if self.low is not None and candidate < self.low:
                candidate = rng.lognormvariate(self.mean, self.sigma)
                continue
            if self.high is not None and candidate > self.high:
                candidate = rng.lognormvariate(self.mean, self.sigma)
                continue
            return candidate
        if self.low is not None and candidate < self.low:
            return self.low
        if self.high is not None and candidate > self.high:
            return self.high
        return candidate

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "lognormal", "mean": self.mean, "sigma": self.sigma}
        if self.low is not None:
            payload["low"] = self.low
        if self.high is not None:
            payload["high"] = self.high
        return payload


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    key: tuple[str, Any]
    prior: PriorDistribution

    def to_dict(self) -> dict[str, Any]:
        kind, value = self.key
        payload = {"name": self.name, "prior": self.prior.to_dict()}
        if kind == "index":
            payload["index"] = value
        elif kind == "reaction_id":
            payload["reaction_id"] = value
        else:
            payload["key"] = {"kind": kind, "value": value}
        return payload


@dataclass(frozen=True)
class ParameterVector:
    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.specs, tuple):
            object.__setattr__(self, "specs", tuple(self.specs))
        names: set[str] = set()
        keys: set[tuple[str, Any]] = set()
        for spec in self.specs:
            if not isinstance(spec, ParameterSpec):
                raise ConfigError("ParameterVector.specs must contain ParameterSpec entries.")
            if spec.name in names:
                raise ConfigError(f"Duplicate parameter name: {spec.name!r}.")
            if spec.key in keys:
                raise ConfigError(f"Duplicate parameter key: {spec.key!r}.")
            names.add(spec.name)
            keys.add(spec.key)

    def __len__(self) -> int:
        return len(self.specs)

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def sample(self, rng: random.Random) -> list[float]:
        return [spec.prior.sample(rng) for spec in self.specs]

    def sample_dict(self, rng: random.Random) -> dict[str, float]:
        return {spec.name: spec.prior.sample(rng) for spec in self.specs}

    def sample_ensemble(
        self,
        n_samples: int,
        *,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> list[list[float]]:
        if isinstance(n_samples, bool):
            raise ConfigError("n_samples must be an integer.")
        try:
            count = int(n_samples)
        except (TypeError, ValueError) as exc:
            raise ConfigError("n_samples must be an integer.") from exc
        if count <= 0:
            raise ConfigError("n_samples must be positive.")
        if rng is None:
            rng = random.Random(0 if seed is None else seed)
        return [self.sample(rng) for _ in range(count)]

    def sample_ensemble_dicts(
        self,
        n_samples: int,
        *,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> list[dict[str, float]]:
        samples = self.sample_ensemble(n_samples, seed=seed, rng=rng)
        names = self.names()
        return [
            {name: float(value) for name, value in zip(names, values)}
            for values in samples
        ]

    def apply_multiplier_values(
        self,
        base_multipliers: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        values: Sequence[float],
    ) -> list[dict[str, Any]]:
        if len(values) != len(self.specs):
            raise ConfigError("parameter values must match parameter vector length.")
        if isinstance(base_multipliers, Mapping):
            base_entries = normalize_reaction_multipliers(base_multipliers)
        else:
            base_entries = [dict(entry) for entry in base_multipliers]
        mapping = _build_multiplier_map(base_entries)
        for spec, value in zip(self.specs, values):
            mapping[spec.key] = _coerce_float(value, f"value[{spec.name}]")
        return _rebuild_multipliers(mapping)

    def to_payload(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs]


def _extract_prior_params(
    entry: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    *,
    keys: Sequence[str],
    label: str,
) -> Optional[float]:
    for key in keys:
        if prior is not None and key in prior:
            return _coerce_float(prior.get(key), f"{label}.{key}")
        if key in entry:
            return _coerce_float(entry.get(key), f"{label}.{key}")
    return None


def _parse_prior(
    entry: Mapping[str, Any],
    *,
    default_prior: Optional[PriorDistribution],
    label: str,
) -> PriorDistribution:
    prior = entry.get("prior")
    if isinstance(prior, PriorDistribution):
        return prior

    prior_mapping: Optional[Mapping[str, Any]] = None
    prior_type: Optional[str] = None
    if isinstance(prior, Mapping):
        prior_mapping = prior
        prior_type = prior.get("type") or prior.get("kind") or prior.get("distribution")
    elif isinstance(prior, str):
        prior_type = prior

    if prior_type is None and prior_mapping is None:
        if default_prior is not None:
            return default_prior
        inferred = "uniform"
        if any(key in entry for key in ("mean", "mu", "sigma", "std")):
            inferred = "lognormal"
        prior_type = inferred

    if prior_type is None:
        raise ConfigError(f"{label} prior type must be provided.")
    prior_key = prior_type.strip().lower()

    if prior_key in {"uniform", "bounded_uniform", "uniform_bounded"}:
        low = _extract_prior_params(entry, prior_mapping, keys=("low", "min"), label=label)
        high = _extract_prior_params(entry, prior_mapping, keys=("high", "max"), label=label)
        if low is None or high is None:
            raise ConfigError(f"{label} uniform prior requires low/high.")
        return UniformPrior(low=low, high=high)

    if prior_key in {"lognormal", "log-normal", "log_normal"}:
        mean = _extract_prior_params(entry, prior_mapping, keys=("mean", "mu"), label=label)
        sigma = _extract_prior_params(entry, prior_mapping, keys=("sigma", "std"), label=label)
        if mean is None or sigma is None:
            raise ConfigError(f"{label} lognormal prior requires mean/sigma.")
        low = _extract_prior_params(entry, prior_mapping, keys=("low",), label=label)
        high = _extract_prior_params(entry, prior_mapping, keys=("high",), label=label)
        return LogNormalPrior(mean=mean, sigma=sigma, low=low, high=high)

    raise ConfigError(f"{label} prior type {prior_type!r} is not supported.")


def build_reaction_multiplier_parameter_vector(
    specs: Sequence[Mapping[str, Any]],
    *,
    default_prior: Optional[PriorDistribution] = None,
) -> ParameterVector:
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes, bytearray)):
        raise ConfigError("parameter specs must be a sequence of mappings.")
    parsed: list[ParameterSpec] = []
    for index, entry in enumerate(specs):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"specs[{index}] must be a mapping.")
        reaction_id = entry.get("reaction_id") or entry.get("reaction")
        idx = _coerce_optional_int(entry.get("index"), f"specs[{index}].index")
        if reaction_id is None and idx is None:
            raise ConfigError(f"specs[{index}] must include reaction_id or index.")
        if reaction_id is not None and idx is not None:
            raise ConfigError(
                f"specs[{index}] must include only one of reaction_id or index."
            )
        if reaction_id is not None:
            reaction_id = _require_nonempty_str(
                reaction_id, f"specs[{index}].reaction_id"
            )
            key = ("reaction_id", reaction_id)
            default_name = f"reaction_id:{reaction_id}"
        else:
            key = ("index", idx)
            default_name = f"index:{idx}"
        name = entry.get("name") or default_name
        name = _require_nonempty_str(name, f"specs[{index}].name")
        prior = _parse_prior(entry, default_prior=default_prior, label=f"specs[{index}]")
        parsed.append(ParameterSpec(name=name, key=key, prior=prior))
    return ParameterVector(tuple(parsed))


@dataclass(frozen=True)
class LaplacianRegularizer:
    matrix: Any
    scale: float
    param_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    graph_id: str
    normalized: bool
    unmapped: tuple[str, ...]


@dataclass(frozen=True)
class MisfitColumns:
    observed_target: str = "observable"
    observed_value: str = "value"
    observed_weight: str = "weight"
    observed_noise: str = "noise"
    observed_aggregate: str = "aggregate"
    predicted_target: str = "observable"
    predicted_value: str = "value"

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "MisfitColumns":
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ConfigError("misfit columns must be a mapping.")
        target = config.get("target") or config.get("observable")
        observed_target = config.get("observed_target", target or cls.observed_target)
        predicted_target = config.get("predicted_target", target or cls.predicted_target)
        value = config.get("value")
        observed_value = config.get("observed_value", value or cls.observed_value)
        predicted_value = config.get("predicted_value", value or cls.predicted_value)
        observed_weight = config.get("weight", cls.observed_weight)
        observed_noise = config.get("noise", cls.observed_noise)
        observed_aggregate = config.get("aggregate", cls.observed_aggregate)
        return cls(
            observed_target=_require_nonempty_str(observed_target, "columns.observed_target"),
            observed_value=_require_nonempty_str(observed_value, "columns.observed_value"),
            observed_weight=_require_nonempty_str(observed_weight, "columns.observed_weight"),
            observed_noise=_require_nonempty_str(observed_noise, "columns.observed_noise"),
            observed_aggregate=_require_nonempty_str(
                observed_aggregate, "columns.observed_aggregate"
            ),
            predicted_target=_require_nonempty_str(
                predicted_target, "columns.predicted_target"
            ),
            predicted_value=_require_nonempty_str(predicted_value, "columns.predicted_value"),
        )


@dataclass(frozen=True)
class MisfitResult:
    vector: list[float]
    scalar: float
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"vector": self.vector, "scalar": self.scalar, "details": self.details}


def compute_misfit(
    observed_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    predicted_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    columns: MisfitColumns | Mapping[str, Any] | None = None,
    default_aggregate: str = DEFAULT_AGGREGATE,
    missing_strategy: str = DEFAULT_MISSING_STRATEGY,
) -> MisfitResult:
    observed = _coerce_rows(observed_rows, "observed")
    predicted = _coerce_rows(predicted_rows, "predicted")
    if isinstance(columns, MisfitColumns):
        column_spec = columns
    else:
        column_spec = MisfitColumns.from_config(columns)

    missing_key = missing_strategy.strip().lower()
    if missing_key not in {"nan", "skip"}:
        raise ConfigError("missing_strategy must be 'nan' or 'skip'.")

    pred_values: dict[str, list[float]] = {}
    pred_invalid: dict[str, int] = {}
    for row in predicted:
        target = row.get(column_spec.predicted_target)
        if target is None:
            continue
        target_name = str(target)
        value = row.get(column_spec.predicted_value)
        if value is None:
            pred_invalid[target_name] = pred_invalid.get(target_name, 0) + 1
            continue
        try:
            pred_values.setdefault(target_name, []).append(float(value))
        except (TypeError, ValueError):
            pred_invalid[target_name] = pred_invalid.get(target_name, 0) + 1

    vector: list[float] = []
    details: list[dict[str, Any]] = []

    for row in observed:
        target = row.get(column_spec.observed_target)
        target = _require_nonempty_str(target, "observed.target")
        aggregate = _normalize_aggregate(
            row.get(column_spec.observed_aggregate, default_aggregate)
        )
        obs_value_raw = row.get(column_spec.observed_value)
        meta: dict[str, Any] = {"target": target, "aggregate": aggregate}
        try:
            obs_value = _coerce_float(obs_value_raw, "observed.value")
        except ConfigError as exc:
            vector.append(math.nan)
            meta.update({"status": "invalid_observed", "error": str(exc)})
            details.append(meta)
            continue

        weight = _coerce_optional_float(
            row.get(column_spec.observed_weight),
            "observed.weight",
            default=1.0,
        )
        noise = _coerce_optional_float(
            row.get(column_spec.observed_noise),
            "observed.noise",
            default=1.0,
        )
        if noise <= 0:
            vector.append(math.nan)
            meta.update({"status": "invalid_noise", "observed": obs_value, "noise": noise})
            details.append(meta)
            continue

        predicted_values = pred_values.get(target, [])
        predicted_value = _aggregate_values(predicted_values, aggregate)
        invalid_count = pred_invalid.get(target, 0)
        if not predicted_values:
            vector.append(math.nan)
            meta.update(
                {
                    "status": "missing_prediction",
                    "observed": obs_value,
                    "weight": weight,
                    "noise": noise,
                    "invalid_predictions": invalid_count,
                }
            )
            details.append(meta)
            continue

        residual = weight * (predicted_value - obs_value) / noise
        meta.update(
            {
                "status": "ok",
                "observed": obs_value,
                "predicted": predicted_value,
                "weight": weight,
                "noise": noise,
            }
        )
        if invalid_count:
            meta["invalid_predictions"] = invalid_count
        vector.append(float(residual))
        details.append(meta)

    if missing_key == "nan" and any(not _is_finite(value) for value in vector):
        scalar = math.nan
    else:
        finite_values = [value for value in vector if _is_finite(value)]
        if not finite_values:
            scalar = math.nan
        else:
            scalar = sum(value * value for value in finite_values)

    return MisfitResult(vector=vector, scalar=scalar, details=details)


def load_observable_rows(
    store: ArtifactStore,
    observable_id: str,
    *,
    run_id: Optional[str] = None,
    run_id_column: str = "run_id",
) -> list[dict[str, Any]]:
    obs_dir = store.artifact_dir("observables", observable_id)
    values_path = obs_dir / "values.parquet"
    if not values_path.exists():
        raise ConfigError(f"Observable values not found: {values_path}")
    rows = _read_table_rows(values_path)
    if run_id is None:
        return rows
    return [row for row in rows if row.get(run_id_column) == run_id]


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"{label} not found: {path}")
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{label} must contain a JSON object.")
    return dict(payload)


def _extract_graph_nodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    graph_payload: Optional[Mapping[str, Any]] = None
    if "nodes" in payload and ("links" in payload or "edges" in payload):
        graph_payload = payload
    elif "bipartite" in payload and isinstance(payload.get("bipartite"), Mapping):
        data = payload.get("bipartite", {}).get("data")
        if isinstance(data, Mapping):
            graph_payload = data
    if graph_payload is None:
        return []
    nodes_raw = graph_payload.get("nodes") or []
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    return list(nodes_raw)


def _build_reaction_id_map(nodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in nodes:
        if not isinstance(entry, Mapping):
            continue
        node_id = entry.get("id")
        if node_id is None:
            continue
        node_id_str = str(node_id)
        for key in ("reaction_id", "reaction", "label"):
            value = entry.get(key)
            if value is None:
                continue
            reaction_id = str(value)
            mapping.setdefault(reaction_id, node_id_str)
    return mapping


def _extract_param_node_map(
    laplacian_cfg: Mapping[str, Any],
    parameter_vector: ParameterVector,
) -> tuple[Optional[dict[str, str]], bool]:
    names = parameter_vector.names()
    nodes_list = laplacian_cfg.get("parameter_nodes") or laplacian_cfg.get("param_nodes")
    if nodes_list is not None:
        if not isinstance(nodes_list, Sequence) or isinstance(
            nodes_list, (str, bytes, bytearray)
        ):
            raise ConfigError("laplacian.parameter_nodes must be a sequence.")
        if len(nodes_list) != len(names):
            raise ConfigError("laplacian.parameter_nodes must match parameter count.")
        mapping = {}
        for name, node in zip(names, nodes_list):
            mapping[name] = _require_nonempty_str(
                node, f"laplacian.parameter_nodes[{name}]"
            )
        return mapping, True

    raw_map = laplacian_cfg.get("node_map") or laplacian_cfg.get("param_map")
    if raw_map is not None:
        if not isinstance(raw_map, Mapping):
            raise ConfigError("laplacian.node_map must be a mapping.")
        mapping: dict[str, str] = {}
        unknown: list[str] = []
        for param_name, node_id in raw_map.items():
            if param_name not in names:
                unknown.append(str(param_name))
                continue
            mapping[str(param_name)] = _require_nonempty_str(
                node_id, f"laplacian.node_map.{param_name}"
            )
        if unknown:
            raise ConfigError(
                "laplacian.node_map has unknown parameters: "
                + ", ".join(sorted(unknown))
            )
        return mapping, False

    return None, False


def _resolve_laplacian_parameter_nodes(
    parameter_vector: ParameterVector,
    node_index: Mapping[str, int],
    *,
    param_node_map: Optional[Mapping[str, str]],
    reaction_id_map: Optional[Mapping[str, str]],
    require_full: bool,
) -> tuple[list[int], list[str], list[str], list[str]]:
    param_indices: list[int] = []
    node_ids: list[str] = []
    param_names: list[str] = []
    unmapped: list[str] = []
    used_nodes: set[str] = set()
    for idx, spec in enumerate(parameter_vector.specs):
        node_id: Optional[str] = None
        if param_node_map is not None and spec.name in param_node_map:
            node_id = param_node_map.get(spec.name)
        if node_id is None and spec.name in node_index:
            node_id = spec.name
        if node_id is None:
            kind, value = spec.key
            if kind == "reaction_id":
                reaction_id = str(value)
                if reaction_id in node_index:
                    node_id = reaction_id
                elif reaction_id_map is not None:
                    node_id = reaction_id_map.get(reaction_id)
            elif kind == "index":
                if isinstance(value, int):
                    candidate = f"reaction_{value + 1}"
                    if candidate in node_index:
                        node_id = candidate
        if node_id is None:
            unmapped.append(spec.name)
            continue
        if node_id not in node_index:
            raise ConfigError(f"laplacian node {node_id!r} not found in node order.")
        if node_id in used_nodes:
            raise ConfigError(f"laplacian node {node_id!r} mapped multiple times.")
        used_nodes.add(node_id)
        param_indices.append(idx)
        node_ids.append(node_id)
        param_names.append(spec.name)

    if not param_indices:
        raise ConfigError("laplacian regularizer could not map parameters to nodes.")
    if require_full and unmapped:
        raise ConfigError(
            "laplacian regularizer missing node mapping for: "
            + ", ".join(unmapped)
        )
    return param_indices, node_ids, param_names, unmapped


def _load_laplacian_matrix(
    *,
    store: ArtifactStore,
    graph_id: str,
    normalized: bool,
) -> tuple[Any, list[str], Optional[str]]:
    if np is None:
        raise ConfigError("numpy is required to load Laplacian matrices.")
    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    payload = _read_json_mapping(
        graph_dir / "graph.json",
        f"graph.json for graphs/{graph_id}",
    )
    laplacian_meta = payload.get("laplacian")
    if not isinstance(laplacian_meta, Mapping):
        raise ConfigError("laplacian metadata missing from graph.json.")
    nodes_meta = payload.get("nodes")
    if not isinstance(nodes_meta, Mapping):
        raise ConfigError("laplacian nodes metadata missing from graph.json.")
    node_order = nodes_meta.get("order")
    if not isinstance(node_order, Sequence) or isinstance(
        node_order, (str, bytes, bytearray)
    ):
        raise ConfigError("laplacian nodes.order must be a sequence.")
    node_ids = [str(node) for node in node_order]

    path = laplacian_meta.get("path") or "laplacian.npz"
    if not isinstance(path, str) or not path.strip():
        raise ConfigError("laplacian.path must be a non-empty string.")
    laplacian_path = graph_dir / path
    if not laplacian_path.exists():
        raise ConfigError(f"laplacian file not found: {laplacian_path}")

    key: Optional[str]
    if normalized:
        key = laplacian_meta.get("normalized_key")
        if not key:
            raise ConfigError("normalized Laplacian not available in graph.json.")
    else:
        key = laplacian_meta.get("laplacian_key") or "laplacian"

    with np.load(laplacian_path) as data:
        if key not in data:
            raise ConfigError(f"laplacian matrix key {key!r} not found in {path}.")
        matrix = np.asarray(data[key], dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ConfigError("laplacian matrix must be square.")
    if matrix.shape[0] != len(node_ids):
        raise ConfigError("laplacian matrix size does not match node order.")
    if not np.isfinite(matrix).all():
        raise ConfigError("laplacian matrix must be finite.")

    source_graph_id: Optional[str] = None
    source = payload.get("source")
    if isinstance(source, Mapping):
        source_graph_id = source.get("graph_id") or source.get("graph")
        if isinstance(source_graph_id, str) and not source_graph_id.strip():
            source_graph_id = None

    return matrix, node_ids, source_graph_id


def _build_laplacian_regularizer(
    *,
    store: ArtifactStore,
    laplacian_cfg: Mapping[str, Any],
    parameter_vector: ParameterVector,
) -> tuple[Optional[LaplacianRegularizer], Optional[dict[str, Any]]]:
    if not laplacian_cfg:
        return None, None
    strength = _extract_laplacian_lambda(laplacian_cfg)
    if strength <= 0:
        return None, None

    graph_id_raw = (
        laplacian_cfg.get("graph_id")
        or laplacian_cfg.get("graph")
        or laplacian_cfg.get("laplacian_id")
        or laplacian_cfg.get("laplacian_graph_id")
        or laplacian_cfg.get("laplacian_artifact")
    )
    if graph_id_raw is None:
        raise ConfigError("laplacian.graph_id must be provided when lambda > 0.")
    graph_id = _require_nonempty_str(graph_id_raw, "laplacian.graph_id")
    normalized = _coerce_optional_bool(
        laplacian_cfg.get("normalized")
        or laplacian_cfg.get("use_normalized")
        or laplacian_cfg.get("normalized_laplacian"),
        "laplacian.normalized",
        default=False,
    )

    matrix, node_ids, source_graph_id = _load_laplacian_matrix(
        store=store,
        graph_id=graph_id,
        normalized=normalized,
    )
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    param_node_map, require_full = _extract_param_node_map(
        laplacian_cfg, parameter_vector
    )

    reaction_id_map: Optional[dict[str, str]] = None
    if any(spec.key[0] == "reaction_id" for spec in parameter_vector.specs):
        if source_graph_id is not None:
            source_payload = _read_json_mapping(
                store.artifact_dir("graphs", source_graph_id) / "graph.json",
                f"graph.json for graphs/{source_graph_id}",
            )
            nodes = _extract_graph_nodes(source_payload)
            reaction_id_map = _build_reaction_id_map(nodes)

    param_indices, mapped_nodes, param_names, unmapped = _resolve_laplacian_parameter_nodes(
        parameter_vector,
        node_index,
        param_node_map=param_node_map,
        reaction_id_map=reaction_id_map,
        require_full=require_full,
    )
    if unmapped:
        logging.getLogger("rxn_platform.assimilation").warning(
            "Laplacian regularizer missing parameters: %s",
            ", ".join(unmapped),
        )

    node_indices = [node_index[node_id] for node_id in mapped_nodes]
    matrix_sub = matrix[np.ix_(node_indices, node_indices)]
    scale = math.sqrt(strength)
    regularizer = LaplacianRegularizer(
        matrix=matrix_sub,
        scale=scale,
        param_indices=tuple(param_indices),
        node_ids=tuple(mapped_nodes),
        graph_id=graph_id,
        normalized=normalized,
        unmapped=tuple(unmapped),
    )

    inputs_payload: dict[str, Any] = {
        "graph_id": graph_id,
        "lambda": strength,
        "normalized": normalized,
        "parameter_names": list(param_names),
        "parameter_nodes": list(mapped_nodes),
    }
    if unmapped:
        inputs_payload["unmapped_parameters"] = list(unmapped)
    return regularizer, inputs_payload


def _append_laplacian_vectors(
    obs_vectors: list[list[float]],
    pred_vectors: list[list[float]],
    ensemble: Sequence[Sequence[float]],
    regularizer: LaplacianRegularizer,
) -> None:
    if np is None:
        raise ConfigError("numpy is required to apply Laplacian regularization.")
    param_indices = regularizer.param_indices
    if not param_indices:
        return
    matrix = regularizer.matrix
    scale = regularizer.scale
    for sample_id, values in enumerate(ensemble):
        sub_values = [values[idx] for idx in param_indices]
        lp = matrix @ np.asarray(sub_values, dtype=float)
        scaled = (lp * scale).tolist()
        obs_vectors[sample_id].extend([0.0] * len(scaled))
        pred_vectors[sample_id].extend(scaled)


def _normalize_sim_cfg(
    sim_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    return normalized, list(multipliers)


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


def _weighted_vectors_from_details(
    details: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float]]:
    obs: list[float] = []
    pred: list[float] = []
    for detail in details:
        if detail.get("status") != "ok":
            obs.append(math.nan)
            pred.append(math.nan)
            continue
        try:
            obs_value = float(detail.get("observed"))
            pred_value = float(detail.get("predicted"))
            weight = float(detail.get("weight", 1.0))
            noise = float(detail.get("noise", 1.0))
        except (TypeError, ValueError):
            obs.append(math.nan)
            pred.append(math.nan)
            continue
        if noise == 0:
            obs.append(math.nan)
            pred.append(math.nan)
            continue
        obs.append(weight * obs_value / noise)
        pred.append(weight * pred_value / noise)
    return obs, pred


def _valid_indices(
    obs_vectors: Sequence[Sequence[float]],
    pred_vectors: Sequence[Sequence[float]],
) -> list[int]:
    if not obs_vectors:
        return []
    length = len(obs_vectors[0])
    indices: list[int] = []
    for idx in range(length):
        if all(
            _is_finite(obs[idx]) and _is_finite(pred[idx])
            for obs, pred in zip(obs_vectors, pred_vectors)
        ):
            indices.append(idx)
    return indices


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _update_ensemble(
    ensemble: Sequence[Sequence[float]],
    obs_vectors: Sequence[Sequence[float]],
    pred_vectors: Sequence[Sequence[float]],
    *,
    inflation: float,
    ridge: float,
    step_size: float,
    scale: float = 1.0,
) -> tuple[list[list[float]], str, str]:
    updated_ensemble = [list(values) for values in ensemble]
    if len(ensemble) < 2:
        return updated_ensemble, "skipped", "Ensemble size < 2; update skipped."
    if scale <= 0:
        raise ConfigError("scale must be > 0.")

    if scale != 1.0:
        obs_scaled = [[value * scale for value in row] for row in obs_vectors]
        pred_scaled = [[value * scale for value in row] for row in pred_vectors]
    else:
        obs_scaled = obs_vectors
        pred_scaled = pred_vectors

    valid_idx = _valid_indices(obs_scaled, pred_scaled)
    if not valid_idx:
        return updated_ensemble, "skipped", "No valid observations; update skipped."

    y_obs = [obs_scaled[0][idx] for idx in valid_idx]
    y_pred = [[row[idx] for idx in valid_idx] for row in pred_scaled]

    y_mean = _mean_vector(y_pred)
    u_mean = _mean_vector(ensemble)
    u_anom = [
        [value - u_mean[idx] for idx, value in enumerate(values)]
        for values in ensemble
    ]
    if inflation != 1.0:
        u_anom = [[value * inflation for value in row] for row in u_anom]
    y_anom = [
        [value - y_mean[idx] for idx, value in enumerate(values)]
        for values in y_pred
    ]

    obs_dim = len(valid_idx)
    param_dim = len(u_mean)
    denom = float(len(ensemble) - 1)

    c_uy = [[0.0 for _ in range(obs_dim)] for _ in range(param_dim)]
    for row_idx, u_row in enumerate(u_anom):
        y_row = y_anom[row_idx]
        for i in range(param_dim):
            u_val = u_row[i]
            for k in range(obs_dim):
                c_uy[i][k] += u_val * y_row[k]
    for i in range(param_dim):
        for k in range(obs_dim):
            c_uy[i][k] /= denom

    c_yy = [[0.0 for _ in range(obs_dim)] for _ in range(obs_dim)]
    for y_row in y_anom:
        for i in range(obs_dim):
            y_val = y_row[i]
            for j in range(obs_dim):
                c_yy[i][j] += y_val * y_row[j]
    for i in range(obs_dim):
        for j in range(obs_dim):
            c_yy[i][j] /= denom
    for i in range(obs_dim):
        c_yy[i][i] += ridge

    updated: list[list[float]] = []
    for sample_id, values in enumerate(ensemble):
        delta = [y_obs[idx] - y_pred[sample_id][idx] for idx in range(obs_dim)]
        solution = _solve_linear_system(c_yy, delta)
        if solution is None:
            return updated_ensemble, "skipped", "Singular update system; update skipped."
        increments = [
            sum(c_uy[i][k] * solution[k] for k in range(obs_dim))
            for i in range(param_dim)
        ]
        updated.append(
            [value + step_size * increments[idx] for idx, value in enumerate(values)]
        )

    return updated, "analysis", ""


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    count = len(vectors)
    length = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / float(count) for i in range(length)]


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    *,
    tol: float = 1.0e-12,
) -> Optional[list[float]]:
    size = len(rhs)
    if size == 0:
        return []
    if any(len(row) != size for row in matrix):
        raise ConfigError("linear system matrix must be square.")
    augmented = [list(row) + [rhs[idx]] for idx, row in enumerate(matrix)]
    for pivot_idx in range(size):
        pivot_row = max(
            range(pivot_idx, size),
            key=lambda idx: abs(augmented[idx][pivot_idx]),
        )
        if abs(augmented[pivot_row][pivot_idx]) < tol:
            return None
        if pivot_row != pivot_idx:
            augmented[pivot_idx], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_idx],
            )
        pivot_value = augmented[pivot_idx][pivot_idx]
        for col in range(pivot_idx, size + 1):
            augmented[pivot_idx][col] /= pivot_value
        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            if factor == 0:
                continue
            for col in range(pivot_idx, size + 1):
                augmented[row_idx][col] -= factor * augmented[pivot_idx][col]
    return [augmented[idx][size] for idx in range(size)]


def _collect_columns(
    rows: Sequence[Mapping[str, Any]],
    required_columns: Sequence[str],
) -> list[str]:
    columns = list(required_columns)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_table(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    required_columns: Sequence[str],
) -> None:
    columns = _collect_columns(rows, required_columns)
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
    write_json_atomic(path, payload)
    logger = logging.getLogger("rxn_platform.assimilation")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
    )


def _build_posterior_row(
    *,
    iteration: int,
    stage: str,
    sample_id: int,
    run_id: Optional[str],
    observable_id: Optional[str],
    params_payload: Mapping[str, Any],
    misfit_value: Optional[float],
    status: str,
    error: Optional[str],
    meta_payload: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    params_json = json.dumps(params_payload, ensure_ascii=True, sort_keys=True)
    meta = dict(meta_payload or {})
    meta_json = json.dumps(meta, ensure_ascii=True, sort_keys=True)
    row: dict[str, Any] = {
        "iteration": iteration,
        "stage": stage,
        "sample_id": sample_id,
        "run_id": run_id,
        "observable_id": observable_id,
        "params_json": params_json,
        "misfit": float(misfit_value) if misfit_value is not None else math.nan,
        "status": status,
        "error": error or "",
        "meta_json": meta_json,
    }
    return row


def _summarize_misfit(
    iteration: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value = row.get("misfit")
        if isinstance(value, (int, float)) and _is_finite(float(value)):
            values.append(float(value))
    total = len(rows)
    if not values:
        return {
            "iteration": iteration,
            "mean_misfit": math.nan,
            "min_misfit": math.nan,
            "max_misfit": math.nan,
            "valid_count": 0,
            "total_count": total,
            "status": "no_valid",
            "message": "No valid misfit values for iteration.",
        }
    return {
        "iteration": iteration,
        "mean_misfit": sum(values) / float(len(values)),
        "min_misfit": min(values),
        "max_misfit": max(values),
        "valid_count": len(values),
        "total_count": total,
        "status": "ok",
        "message": "",
    }


REQUIRED_POSTERIOR_COLUMNS = (
    "iteration",
    "stage",
    "sample_id",
    "run_id",
    "observable_id",
    "params_json",
    "misfit",
    "status",
    "error",
    "meta_json",
)

REQUIRED_MISFIT_COLUMNS = (
    "iteration",
    "mean_misfit",
    "min_misfit",
    "max_misfit",
    "valid_count",
    "total_count",
    "status",
    "message",
)


def run_eki(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run an Ensemble Kalman Inversion (EKI) update loop."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, assim_cfg = _extract_assim_cfg(resolved_cfg)
    params = _extract_params(assim_cfg)
    seed = _extract_seed(resolved_cfg)

    sim_cfg = _extract_sim_cfg(resolved_cfg, assim_cfg, params)
    base_sim_cfg, base_multipliers = _normalize_sim_cfg(sim_cfg)

    missing_strategy = _extract_missing_strategy(assim_cfg, params)
    observables_cfg = _extract_observables_cfg(
        resolved_cfg,
        assim_cfg,
        params,
        missing_strategy=missing_strategy,
    )

    observed_rows = _extract_observed_rows(
        store,
        resolved_cfg,
        assim_cfg,
        params,
        sim_cfg=sim_cfg,
    )
    columns = _extract_columns(assim_cfg, params)
    default_aggregate = _extract_default_aggregate(assim_cfg, params)

    default_prior: Optional[PriorDistribution] = None
    for source in (params, assim_cfg):
        if "default_prior" in source:
            default_prior = _parse_default_prior(source.get("default_prior"))
            break

    parameter_specs = _extract_parameter_specs(assim_cfg, params)
    parameter_vector = build_reaction_multiplier_parameter_vector(
        parameter_specs,
        default_prior=default_prior,
    )
    names = parameter_vector.names()
    laplacian_cfg = _extract_laplacian_cfg(assim_cfg, params)
    laplacian_regularizer, laplacian_inputs = _build_laplacian_regularizer(
        store=store,
        laplacian_cfg=laplacian_cfg,
        parameter_vector=parameter_vector,
    )

    initial_ensemble: Optional[list[list[float]]] = None
    for source in (params, assim_cfg):
        if "initial_ensemble" in source:
            initial_ensemble = _normalize_initial_ensemble(
                source.get("initial_ensemble"),
                parameter_vector,
            )
            break

    rng = random.Random(seed)
    if initial_ensemble is None:
        ensemble_size = _extract_ensemble_size(assim_cfg, params)
        ensemble = parameter_vector.sample_ensemble(ensemble_size, rng=rng)
    else:
        ensemble = initial_ensemble
        ensemble_size = len(ensemble)

    iterations = _extract_iterations(assim_cfg, params)
    inflation = _extract_inflation(assim_cfg, params)
    ridge = _extract_ridge(assim_cfg, params)
    step_size = _extract_step_size(assim_cfg, params)

    logger = logging.getLogger("rxn_platform.assimilation")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    run_obs_cache: dict[str, str] = {}
    obs_rows_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    posterior_rows: list[dict[str, Any]] = []
    misfit_history_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    observable_ids: list[str] = []
    update_messages: list[str] = []

    for iteration in range(iterations):
        logger.info("EKI iteration %s/%s", iteration + 1, iterations)
        iteration_rows: list[dict[str, Any]] = []
        obs_vectors: list[list[float]] = []
        pred_vectors: list[list[float]] = []

        for sample_id, values in enumerate(ensemble):
            params_payload = {
                name: float(value) for name, value in zip(names, values)
            }
            run_id: Optional[str] = None
            obs_id: Optional[str] = None
            status = "ok"
            error: Optional[str] = None
            misfit_value: Optional[float] = math.nan
            meta_payload: Optional[Mapping[str, Any]] = None
            try:
                multipliers = parameter_vector.apply_multiplier_values(
                    base_multipliers,
                    values,
                )
                sample_sim_cfg = dict(base_sim_cfg)
                if multipliers:
                    sample_sim_cfg["reaction_multipliers"] = multipliers
                else:
                    sample_sim_cfg.pop("reaction_multipliers", None)
                run_id = _sim_run_id(sample_sim_cfg)
                if run_id not in run_obs_cache:
                    run_id, obs_id = _run_sim_and_observables(
                        runner,
                        store,
                        sample_sim_cfg,
                        observables_cfg,
                    )
                    run_obs_cache[run_id] = obs_id
                obs_id = run_obs_cache[run_id]
                if run_id is not None:
                    run_ids.append(run_id)
                if obs_id is not None:
                    observable_ids.append(obs_id)

                cache_key = (obs_id, run_id)
                predicted_rows = obs_rows_cache.get(cache_key)
                if predicted_rows is None:
                    predicted_rows = load_observable_rows(
                        store,
                        obs_id,
                        run_id=run_id,
                    )
                    obs_rows_cache[cache_key] = predicted_rows

                misfit_result = compute_misfit(
                    observed_rows,
                    predicted_rows,
                    columns=columns,
                    default_aggregate=default_aggregate,
                    missing_strategy=missing_strategy,
                )
                obs_vec, pred_vec = _weighted_vectors_from_details(misfit_result.details)
                obs_vectors.append(obs_vec)
                pred_vectors.append(pred_vec)
                misfit_value = misfit_result.scalar
                meta_payload = {
                    "vector": misfit_result.vector,
                    "details": misfit_result.details,
                }
            except Exception as exc:
                status = "error"
                error = str(exc)
                logger.warning(
                    "EKI sample %s failed at iteration %s: %s",
                    sample_id,
                    iteration,
                    exc,
                )
                obs_vectors.append([math.nan] * len(observed_rows))
                pred_vectors.append([math.nan] * len(observed_rows))
                meta_payload = {"error": str(exc)}

            row = _build_posterior_row(
                iteration=iteration,
                stage="forecast",
                sample_id=sample_id,
                run_id=run_id,
                observable_id=obs_id,
                params_payload=params_payload,
                misfit_value=misfit_value,
                status=status,
                error=error,
                meta_payload=meta_payload,
            )
            iteration_rows.append(row)
            posterior_rows.append(row)

        misfit_history_rows.append(_summarize_misfit(iteration, iteration_rows))

        if laplacian_regularizer is not None:
            _append_laplacian_vectors(
                obs_vectors,
                pred_vectors,
                ensemble,
                laplacian_regularizer,
            )

        updated_ensemble, update_status, update_message = _update_ensemble(
            ensemble,
            obs_vectors,
            pred_vectors,
            inflation=inflation,
            ridge=ridge,
            step_size=step_size,
        )

        if update_status != "analysis":
            update_messages.append(update_message)
            logger.warning(update_message)

        for sample_id, values in enumerate(updated_ensemble):
            params_payload = {
                name: float(value) for name, value in zip(names, values)
            }
            run_id: Optional[str] = None
            obs_id: Optional[str] = None
            try:
                multipliers = parameter_vector.apply_multiplier_values(
                    base_multipliers,
                    values,
                )
                sample_sim_cfg = dict(base_sim_cfg)
                if multipliers:
                    sample_sim_cfg["reaction_multipliers"] = multipliers
                else:
                    sample_sim_cfg.pop("reaction_multipliers", None)
                run_id = _sim_run_id(sample_sim_cfg)
                obs_id = run_obs_cache.get(run_id)
            except Exception:
                run_id = None
            posterior_rows.append(
                _build_posterior_row(
                    iteration=iteration,
                    stage="analysis",
                    sample_id=sample_id,
                    run_id=run_id,
                    observable_id=obs_id,
                    params_payload=params_payload,
                    misfit_value=None,
                    status=update_status,
                    error=None,
                    meta_payload={
                        "update_status": update_status,
                        "update_message": update_message,
                    },
                )
            )

        ensemble = updated_ensemble

    inputs_payload: dict[str, Any] = {
        "parameter_vector": parameter_vector.to_payload(),
        "ensemble_size": ensemble_size,
        "iterations": iterations,
        "seed": seed,
    }
    if default_prior is not None:
        inputs_payload["default_prior"] = default_prior.to_dict()
    if initial_ensemble is not None:
        inputs_payload["initial_ensemble"] = [
            {name: float(value) for name, value in zip(names, values)}
            for values in initial_ensemble
        ]
    if inflation != DEFAULT_INFLATION:
        inputs_payload["inflation"] = inflation
    if ridge != DEFAULT_RIDGE:
        inputs_payload["ridge"] = ridge
    if step_size != DEFAULT_STEP_SIZE:
        inputs_payload["step_size"] = step_size
    if missing_strategy != DEFAULT_MISSING_STRATEGY:
        inputs_payload["missing_strategy"] = missing_strategy
    if default_aggregate != DEFAULT_AGGREGATE:
        inputs_payload["default_aggregate"] = default_aggregate
    if laplacian_inputs is not None:
        inputs_payload["laplacian_regularizer"] = laplacian_inputs
    inputs_payload["observed"] = observed_rows
    inputs_payload["run_ids"] = _dedupe_preserve(run_ids)
    inputs_payload["observable_ids"] = _dedupe_preserve(observable_ids)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    notes = ""
    if update_messages:
        notes = " | ".join(update_messages)

    parent_candidates = run_ids + observable_ids
    if laplacian_regularizer is not None:
        parent_candidates.append(laplacian_regularizer.graph_id)
    parents = _dedupe_preserve(parent_candidates)
    manifest = build_manifest(
        kind="assimilation",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
        notes=notes or None,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(
            base_dir / "parameter_vector.json",
            {"parameters": parameter_vector.to_payload()},
        )
        _write_table(posterior_rows, base_dir / "posterior.parquet", REQUIRED_POSTERIOR_COLUMNS)
        _write_table(
            misfit_history_rows,
            base_dir / "misfit_history.parquet",
            REQUIRED_MISFIT_COLUMNS,
        )

    return store.ensure(manifest, writer=_writer)


def run_eki_csv(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run EKI with a sim config resolved from a conditions CSV row."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    resolved_cfg = _resolve_cfg(cfg)
    _, assim_cfg = _extract_assim_cfg(resolved_cfg)
    params = _extract_params(assim_cfg)
    sim_cfg = _extract_sim_cfg(resolved_cfg, assim_cfg, params)
    updated_sim_cfg = _apply_conditions_file_to_sim(
        sim_cfg,
        resolved_cfg=resolved_cfg,
        assim_cfg=assim_cfg,
        params=params,
    )
    updated_cfg = dict(resolved_cfg)
    updated_cfg["sim"] = updated_sim_cfg
    return run_eki(updated_cfg, store=store, registry=registry)


def run_esmda(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run an Ensemble Smoother with Multiple Data Assimilation (ES-MDA) update loop."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, assim_cfg = _extract_assim_cfg(resolved_cfg)
    params = _extract_params(assim_cfg)
    seed = _extract_seed(resolved_cfg)

    sim_cfg = _extract_sim_cfg(resolved_cfg, assim_cfg, params)
    base_sim_cfg, base_multipliers = _normalize_sim_cfg(sim_cfg)

    missing_strategy = _extract_missing_strategy(assim_cfg, params)
    observables_cfg = _extract_observables_cfg(
        resolved_cfg,
        assim_cfg,
        params,
        missing_strategy=missing_strategy,
    )

    observed_rows = _extract_observed_rows(
        store,
        resolved_cfg,
        assim_cfg,
        params,
        sim_cfg=sim_cfg,
    )
    columns = _extract_columns(assim_cfg, params)
    default_aggregate = _extract_default_aggregate(assim_cfg, params)

    default_prior: Optional[PriorDistribution] = None
    for source in (params, assim_cfg):
        if "default_prior" in source:
            default_prior = _parse_default_prior(source.get("default_prior"))
            break

    parameter_specs = _extract_parameter_specs(assim_cfg, params)
    parameter_vector = build_reaction_multiplier_parameter_vector(
        parameter_specs,
        default_prior=default_prior,
    )
    names = parameter_vector.names()
    laplacian_cfg = _extract_laplacian_cfg(assim_cfg, params)
    laplacian_regularizer, laplacian_inputs = _build_laplacian_regularizer(
        store=store,
        laplacian_cfg=laplacian_cfg,
        parameter_vector=parameter_vector,
    )

    initial_ensemble: Optional[list[list[float]]] = None
    for source in (params, assim_cfg):
        if "initial_ensemble" in source:
            initial_ensemble = _normalize_initial_ensemble(
                source.get("initial_ensemble"),
                parameter_vector,
            )
            break

    rng = random.Random(seed)
    if initial_ensemble is None:
        ensemble_size = _extract_ensemble_size(assim_cfg, params)
        ensemble = parameter_vector.sample_ensemble(ensemble_size, rng=rng)
    else:
        ensemble = initial_ensemble
        ensemble_size = len(ensemble)

    iterations = _extract_iterations(assim_cfg, params)
    inflation = _extract_inflation(assim_cfg, params)
    ridge = _extract_ridge(assim_cfg, params)
    step_size = _extract_step_size(assim_cfg, params)
    alpha_schedule = _extract_alpha_schedule(
        assim_cfg,
        params,
        iterations=iterations,
    )

    logger = logging.getLogger("rxn_platform.assimilation")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    alpha_sum = sum(1.0 / alpha for alpha in alpha_schedule)
    if not math.isfinite(alpha_sum) or abs(alpha_sum - 1.0) > 1.0e-6:
        logger.warning(
            "ES-MDA alpha schedule sum(1/alpha)=%.6f (expected 1.0).",
            alpha_sum,
        )

    run_obs_cache: dict[str, str] = {}
    obs_rows_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    posterior_rows: list[dict[str, Any]] = []
    misfit_history_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    observable_ids: list[str] = []
    update_messages: list[str] = []

    for iteration in range(iterations):
        alpha = alpha_schedule[iteration]
        logger.info(
            "ES-MDA iteration %s/%s (alpha=%s)",
            iteration + 1,
            iterations,
            alpha,
        )
        iteration_rows: list[dict[str, Any]] = []
        obs_vectors: list[list[float]] = []
        pred_vectors: list[list[float]] = []

        for sample_id, values in enumerate(ensemble):
            params_payload = {
                name: float(value) for name, value in zip(names, values)
            }
            run_id: Optional[str] = None
            obs_id: Optional[str] = None
            status = "ok"
            error: Optional[str] = None
            misfit_value: Optional[float] = math.nan
            meta_payload: Optional[Mapping[str, Any]] = None
            try:
                multipliers = parameter_vector.apply_multiplier_values(
                    base_multipliers,
                    values,
                )
                sample_sim_cfg = dict(base_sim_cfg)
                if multipliers:
                    sample_sim_cfg["reaction_multipliers"] = multipliers
                else:
                    sample_sim_cfg.pop("reaction_multipliers", None)
                run_id = _sim_run_id(sample_sim_cfg)
                if run_id not in run_obs_cache:
                    run_id, obs_id = _run_sim_and_observables(
                        runner,
                        store,
                        sample_sim_cfg,
                        observables_cfg,
                    )
                    run_obs_cache[run_id] = obs_id
                obs_id = run_obs_cache[run_id]
                if run_id is not None:
                    run_ids.append(run_id)
                if obs_id is not None:
                    observable_ids.append(obs_id)

                cache_key = (obs_id, run_id)
                predicted_rows = obs_rows_cache.get(cache_key)
                if predicted_rows is None:
                    predicted_rows = load_observable_rows(
                        store,
                        obs_id,
                        run_id=run_id,
                    )
                    obs_rows_cache[cache_key] = predicted_rows

                misfit_result = compute_misfit(
                    observed_rows,
                    predicted_rows,
                    columns=columns,
                    default_aggregate=default_aggregate,
                    missing_strategy=missing_strategy,
                )
                obs_vec, pred_vec = _weighted_vectors_from_details(misfit_result.details)
                obs_vectors.append(obs_vec)
                pred_vectors.append(pred_vec)
                misfit_value = misfit_result.scalar
                meta_payload = {
                    "vector": misfit_result.vector,
                    "details": misfit_result.details,
                }
            except Exception as exc:
                status = "error"
                error = str(exc)
                logger.warning(
                    "ES-MDA sample %s failed at iteration %s: %s",
                    sample_id,
                    iteration,
                    exc,
                )
                obs_vectors.append([math.nan] * len(observed_rows))
                pred_vectors.append([math.nan] * len(observed_rows))
                meta_payload = {"error": str(exc)}

            row = _build_posterior_row(
                iteration=iteration,
                stage="forecast",
                sample_id=sample_id,
                run_id=run_id,
                observable_id=obs_id,
                params_payload=params_payload,
                misfit_value=misfit_value,
                status=status,
                error=error,
                meta_payload=meta_payload,
            )
            iteration_rows.append(row)
            posterior_rows.append(row)

        misfit_history_rows.append(_summarize_misfit(iteration, iteration_rows))

        if laplacian_regularizer is not None:
            _append_laplacian_vectors(
                obs_vectors,
                pred_vectors,
                ensemble,
                laplacian_regularizer,
            )

        scale = 1.0 / math.sqrt(alpha)
        updated_ensemble, update_status, update_message = _update_ensemble(
            ensemble,
            obs_vectors,
            pred_vectors,
            inflation=inflation,
            ridge=ridge,
            step_size=step_size,
            scale=scale,
        )

        if update_status != "analysis":
            if update_message:
                update_message = f"alpha={alpha}: {update_message}"
            update_messages.append(update_message)
            logger.warning(update_message)

        for sample_id, values in enumerate(updated_ensemble):
            params_payload = {
                name: float(value) for name, value in zip(names, values)
            }
            run_id: Optional[str] = None
            obs_id: Optional[str] = None
            try:
                multipliers = parameter_vector.apply_multiplier_values(
                    base_multipliers,
                    values,
                )
                sample_sim_cfg = dict(base_sim_cfg)
                if multipliers:
                    sample_sim_cfg["reaction_multipliers"] = multipliers
                else:
                    sample_sim_cfg.pop("reaction_multipliers", None)
                run_id = _sim_run_id(sample_sim_cfg)
                obs_id = run_obs_cache.get(run_id)
            except Exception:
                run_id = None
            posterior_rows.append(
                _build_posterior_row(
                    iteration=iteration,
                    stage="analysis",
                    sample_id=sample_id,
                    run_id=run_id,
                    observable_id=obs_id,
                    params_payload=params_payload,
                    misfit_value=None,
                    status=update_status,
                    error=None,
                    meta_payload={
                        "update_status": update_status,
                        "update_message": update_message,
                        "alpha": alpha,
                    },
                )
            )

        ensemble = updated_ensemble

    inputs_payload: dict[str, Any] = {
        "parameter_vector": parameter_vector.to_payload(),
        "ensemble_size": ensemble_size,
        "iterations": iterations,
        "seed": seed,
        "alpha_schedule": [float(alpha) for alpha in alpha_schedule],
    }
    if default_prior is not None:
        inputs_payload["default_prior"] = default_prior.to_dict()
    if initial_ensemble is not None:
        inputs_payload["initial_ensemble"] = [
            {name: float(value) for name, value in zip(names, values)}
            for values in initial_ensemble
        ]
    if inflation != DEFAULT_INFLATION:
        inputs_payload["inflation"] = inflation
    if ridge != DEFAULT_RIDGE:
        inputs_payload["ridge"] = ridge
    if step_size != DEFAULT_STEP_SIZE:
        inputs_payload["step_size"] = step_size
    if missing_strategy != DEFAULT_MISSING_STRATEGY:
        inputs_payload["missing_strategy"] = missing_strategy
    if default_aggregate != DEFAULT_AGGREGATE:
        inputs_payload["default_aggregate"] = default_aggregate
    if laplacian_inputs is not None:
        inputs_payload["laplacian_regularizer"] = laplacian_inputs
    inputs_payload["observed"] = observed_rows
    inputs_payload["run_ids"] = _dedupe_preserve(run_ids)
    inputs_payload["observable_ids"] = _dedupe_preserve(observable_ids)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    notes = ""
    if update_messages:
        notes = " | ".join(update_messages)

    parent_candidates = run_ids + observable_ids
    if laplacian_regularizer is not None:
        parent_candidates.append(laplacian_regularizer.graph_id)
    parents = _dedupe_preserve(parent_candidates)
    manifest = build_manifest(
        kind="assimilation",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
        notes=notes or None,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(
            base_dir / "parameter_vector.json",
            {"parameters": parameter_vector.to_payload()},
        )
        _write_table(posterior_rows, base_dir / "posterior.parquet", REQUIRED_POSTERIOR_COLUMNS)
        _write_table(
            misfit_history_rows,
            base_dir / "misfit_history.parquet",
            REQUIRED_MISFIT_COLUMNS,
        )

    return store.ensure(manifest, writer=_writer)


def run_esmda_csv(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run ES-MDA with a sim config resolved from a conditions CSV row."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    resolved_cfg = _resolve_cfg(cfg)
    _, assim_cfg = _extract_assim_cfg(resolved_cfg)
    params = _extract_params(assim_cfg)
    sim_cfg = _extract_sim_cfg(resolved_cfg, assim_cfg, params)
    updated_sim_cfg = _apply_conditions_file_to_sim(
        sim_cfg,
        resolved_cfg=resolved_cfg,
        assim_cfg=assim_cfg,
        params=params,
    )
    updated_cfg = dict(resolved_cfg)
    updated_cfg["sim"] = updated_sim_cfg
    return run_esmda(updated_cfg, store=store, registry=registry)


class EkiTask(Task):
    name = "assimilation.eki"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run_eki(cfg, store=store, registry=registry)


register("task", "assimilation.eki", EkiTask())


class EkiCsvTask(Task):
    name = "assimilation.eki_csv"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run_eki_csv(cfg, store=store, registry=registry)


register("task", "assimilation.eki_csv", EkiCsvTask())


class EsmdaTask(Task):
    name = "assimilation.esmda"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run_esmda(cfg, store=store, registry=registry)


register("task", "assimilation.esmda", EsmdaTask())


class EsmdaCsvTask(Task):
    name = "assimilation.esmda_csv"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run_esmda_csv(cfg, store=store, registry=registry)


register("task", "assimilation.esmda_csv", EsmdaCsvTask())


def write_assimilation_artifact(
    *,
    store: ArtifactStore,
    manifest_cfg: Mapping[str, Any],
    parameter_vector: ParameterVector,
    samples: Optional[Sequence[Sequence[float]]] = None,
    misfit: Optional[MisfitResult] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    parents: Optional[Sequence[str]] = None,
    notes: Optional[str] = None,
) -> ArtifactCacheResult:
    inputs_payload = dict(inputs or {})
    inputs_payload.setdefault("parameter_vector", parameter_vector.to_payload())
    sample_list: Optional[list[Sequence[float]]] = None
    if samples is not None:
        sample_list = [list(values) for values in samples]
        inputs_payload.setdefault("sample_count", len(sample_list))

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="assimilation",
        artifact_id=artifact_id,
        parents=list(parents or []),
        inputs=inputs_payload,
        config=manifest_cfg,
        notes=notes,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(
            base_dir / "parameter_vector.json",
            {"parameters": parameter_vector.to_payload()},
        )
        if sample_list is not None:
            rows = []
            names = parameter_vector.names()
            for idx, values in enumerate(sample_list):
                rows.append(
                    {
                        "sample_id": idx,
                        "params": {name: float(value) for name, value in zip(names, values)},
                    }
                )
            write_json_atomic(base_dir / "prior_samples.json", {"samples": rows})
        if misfit is not None:
            write_json_atomic(base_dir / "misfit.json", misfit.to_dict())
        logger = logging.getLogger("rxn_platform.assimilation")
        if samples is not None and (pd is None or pq is None):
            logger.debug("Stored assimilation samples as JSON (parquet unavailable).")

    return store.ensure(manifest, writer=_writer)


__all__ = [
    "EkiTask",
    "EsmdaTask",
    "LogNormalPrior",
    "MisfitColumns",
    "MisfitResult",
    "ParameterSpec",
    "ParameterVector",
    "PriorDistribution",
    "UniformPrior",
    "build_reaction_multiplier_parameter_vector",
    "compute_misfit",
    "load_observable_rows",
    "run_eki",
    "run_esmda",
    "write_assimilation_artifact",
]
