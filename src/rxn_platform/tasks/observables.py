"""Observable framework and task entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Optional
from rxn_platform.core import make_artifact_id
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.io_utils import write_json_atomic
from rxn_platform.registry import Registry, register
from rxn_platform.run_store import resolve_run_dataset_dir
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    load_run_dataset_payload,
    load_run_ids_from_run_set,
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
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None

DEFAULT_MISSING_STRATEGY = "nan"
REQUIRED_COLUMNS = ("run_id", "observable", "value", "unit", "meta_json")
_GAS_STATS = ("last", "mean", "max", "integral")
_GAS_PREFIX = "gas"
_COVERAGE_PREFIX = "cov"
_FILM_THICKNESS_OUTPUT = "film.thickness"
_FILM_RATE_VARS = ("deposition_rate", "film_growth_flux")
_ELEMENT_PREFIX = "gas.element"
_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class RunDatasetView:
    coords: Mapping[str, Any]
    data_vars: Mapping[str, Any]
    attrs: Mapping[str, Any]
    raw: Mapping[str, Any]


class Observable:
    """Base class for observable plugins."""

    name: str
    requires: Sequence[str] = ()
    requires_coords: Sequence[str] = ()
    requires_attrs: Sequence[str] = ()

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError("Observable implementations must override compute().")


class _LazyDatasetMapping(Mapping[str, Any]):
    def __init__(self, dataset: Any, kind: str) -> None:
        self._dataset = dataset
        self._kind = kind
        self._cache: dict[str, dict[str, Any]] = {}

    def _source(self) -> Mapping[str, Any]:
        if self._kind == "coords":
            return self._dataset.coords
        return self._dataset.data_vars

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key in self._cache:
            return self._cache[key]
        source = self._source()
        if key not in source:
            raise KeyError(key)
        array = source[key]
        payload = {
            "dims": list(getattr(array, "dims", (key,))),
            "data": array.values.tolist(),
        }
        self._cache[key] = payload
        return payload

    def __iter__(self):
        return iter(self._source())

    def __len__(self) -> int:
        return len(self._source())

    def __contains__(self, key: object) -> bool:
        return key in self._source()


class GasCompositionObservable(Observable):
    name = "gas_composition"
    requires = ("X",)
    requires_coords = ("time", "species")

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("gas_composition params must be a mapping.")

        stats = _coerce_gas_stats(cfg.get("stats"))
        rank_by = _coerce_rank_by(cfg.get("rank_by"))
        top_n = _coerce_top_n(cfg.get("top_n"))
        species_filter = _coerce_str_sequence(cfg.get("species"), "species")
        aggregate_elements = bool(cfg.get("aggregate_elements", False))
        elements_filter = _coerce_str_sequence(cfg.get("elements"), "elements")

        time_values = _coerce_float_sequence(
            _extract_coord_data(run_dataset, "time"),
            "coords.time",
        )
        species_names = _coerce_str_sequence(
            _extract_coord_data(run_dataset, "species"),
            "coords.species",
        )
        matrix = _extract_time_species_matrix(
            run_dataset.data_vars.get("X"),
            time_values,
            species_names,
        )

        stats_to_compute = set(stats)
        stats_to_compute.add(rank_by)
        stats_by_species = _compute_stats_by_species(
            species_names,
            matrix,
            time_values,
            stats_to_compute,
        )

        selected_species = _select_species(
            species_filter,
            top_n,
            rank_by,
            stats_by_species,
        )

        units = run_dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")
        base_unit = "" if units.get("X") is None else str(units.get("X"))
        time_unit = "" if units.get("time") is None else str(units.get("time"))
        integral_unit = _integral_unit(base_unit, time_unit)

        rows: list[dict[str, Any]] = []
        for species in selected_species:
            values = stats_by_species[species]
            for stat in stats:
                unit = base_unit if stat != "integral" else integral_unit
                rows.append(
                    {
                        "observable": f"{_GAS_PREFIX}.{species}.{stat}",
                        "value": values[stat],
                        "unit": unit,
                    }
                )

        if aggregate_elements:
            rows.extend(
                _build_element_rows(
                    species_names,
                    matrix,
                    time_values,
                    stats,
                    stats_to_compute,
                    base_unit,
                    integral_unit,
                    elements_filter,
                    cfg,
                    run_dataset.attrs,
                )
            )

        return rows


class IgnitionDelayObservable(Observable):
    """Compute ignition delay as the time of maximum dT/dt.

    This matches the definition used in benchmarks/scripts/netbench_generate_obs_cantera.py
    (finite difference vs previous point; take the timestamp of the max slope).
    """

    name = "ignition_delay"
    requires = ("T",)
    requires_coords = ("time",)

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("ignition_delay params must be a mapping.")

        time_values = _coerce_float_sequence(
            _extract_coord_data(run_dataset, "time"),
            "coords.time",
        )

        t_payload = run_dataset.data_vars.get("T")
        if not isinstance(t_payload, Mapping):
            raise ConfigError("data_vars.T must be a mapping.")
        dims = t_payload.get("dims")
        data = t_payload.get("data")
        if isinstance(dims, str) or not isinstance(dims, Sequence):
            raise ConfigError("data_vars.T.dims must be a sequence.")
        dims_list = list(dims)
        if dims_list != ["time"]:
            raise ConfigError("data_vars.T.dims must be ['time'].")
        temperatures = _coerce_float_sequence(data, "data_vars.T.data")
        if len(temperatures) != len(time_values):
            raise ConfigError("T series length must match time coordinates.")

        time_unit = ""
        units = run_dataset.attrs.get("units")
        if units is not None:
            if not isinstance(units, Mapping):
                raise ConfigError("attrs.units must be a mapping when provided.")
            raw_time = units.get("time")
            if raw_time is not None:
                time_unit = str(raw_time)

        if len(time_values) < 2:
            return {
                "observable": "ignition_delay",
                "value": math.nan,
                "unit": time_unit,
                "meta": {"status": "insufficient_points", "n": len(time_values)},
            }

        best_idx = 1
        best_value = -math.inf
        for idx in range(1, len(time_values)):
            dt = float(time_values[idx]) - float(time_values[idx - 1])
            if dt <= 0.0:
                dt = 1.0e-12
            dTdt = (float(temperatures[idx]) - float(temperatures[idx - 1])) / dt
            if dTdt > best_value:
                best_value = dTdt
                best_idx = idx

        return {
            "observable": "ignition_delay",
            "value": float(time_values[best_idx]),
            "unit": time_unit,
            # Keep meta stable across baseline/reduced runs so validation can
            # compare values without mismatching on debug-only fields.
            "meta": {"method": "max_dTdt_prev_point"},
        }


class CoverageObservable(Observable):
    name = "coverage_summary"

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("coverage_summary params must be a mapping.")

        stats = _coerce_gas_stats(cfg.get("stats"))
        species_filter = _coerce_str_sequence(
            cfg.get("surface_species", cfg.get("species")),
            "surface_species",
        )

        coverage_payload = run_dataset.data_vars.get("coverage")

        units = run_dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")
        base_unit = "" if units.get("coverage") is None else str(units.get("coverage"))
        time_unit = "" if units.get("time") is None else str(units.get("time"))
        integral_unit = _integral_unit(base_unit, time_unit)

        if coverage_payload is None:
            surface_species: list[str] = []
            coord_payload = run_dataset.coords.get("surface_species")
            if isinstance(coord_payload, Mapping):
                try:
                    surface_species = _coerce_str_sequence(
                        coord_payload.get("data"),
                        "coords.surface_species",
                    )
                except ConfigError:
                    surface_species = []
            if not surface_species:
                surface_species = ["missing"]
            meta = {"status": "missing_coverage", "missing": ["data_vars.coverage"]}
            rows: list[dict[str, Any]] = []
            for name in surface_species:
                for stat in stats:
                    unit = base_unit if stat != "integral" else integral_unit
                    rows.append(
                        {
                            "observable": f"{_COVERAGE_PREFIX}.{name}.{stat}",
                            "value": math.nan,
                            "unit": unit,
                            "meta": meta,
                        }
                    )
            return rows

        time_values = _coerce_float_sequence(
            _extract_coord_data(run_dataset, "time"),
            "coords.time",
        )
        surface_species = _coerce_str_sequence(
            _extract_coord_data(run_dataset, "surface_species"),
            "coords.surface_species",
        )
        matrix = _extract_time_surface_matrix(
            coverage_payload,
            time_values,
            surface_species,
        )

        stats_to_compute = set(stats)
        stats_by_species = _compute_stats_by_species(
            surface_species,
            matrix,
            time_values,
            stats_to_compute,
        )
        if species_filter:
            unknown = [name for name in species_filter if name not in stats_by_species]
            if unknown:
                missing = ", ".join(unknown)
                raise ConfigError(f"Unknown surface species requested: {missing}.")
            selected_species = list(species_filter)
        else:
            selected_species = list(surface_species)

        rows: list[dict[str, Any]] = []
        for name in selected_species:
            values = stats_by_species[name]
            for stat in stats:
                unit = base_unit if stat != "integral" else integral_unit
                rows.append(
                    {
                        "observable": f"{_COVERAGE_PREFIX}.{name}.{stat}",
                        "value": values[stat],
                        "unit": unit,
                    }
                )
        return rows


class FilmThicknessObservable(Observable):
    name = "film_thickness"
    requires_coords = ("time",)

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("film_thickness params must be a mapping.")

        time_values = _coerce_float_sequence(
            _extract_coord_data(run_dataset, "time"),
            "coords.time",
        )

        units = run_dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")
        time_unit = "" if units.get("time") is None else str(units.get("time"))

        unit_override = cfg.get("unit")
        if unit_override is not None:
            unit_override = str(unit_override)

        scale = _coerce_optional_float(cfg.get("scale"), "scale", default=1.0)

        rate_vars = cfg.get("rate_vars")
        if rate_vars is None:
            rate_vars = cfg.get("rate_var")
        if rate_vars is None:
            rate_candidates = list(_FILM_RATE_VARS)
        else:
            rate_candidates = _coerce_str_sequence(rate_vars, "rate_vars")

        missing: list[str] = []
        for name in rate_candidates:
            if name not in run_dataset.data_vars:
                missing.append(f"data_vars.{name}")
                continue
            try:
                series = _extract_time_series(
                    run_dataset.data_vars.get(name),
                    time_values,
                    name,
                )
            except ConfigError as exc:
                meta = {
                    "status": "invalid_rate",
                    "source": f"data_vars.{name}",
                    "error": str(exc),
                }
                return {
                    "name": _FILM_THICKNESS_OUTPUT,
                    "value": math.nan,
                    "unit": unit_override or "",
                    "meta": meta,
                }

            base_unit = "" if units.get(name) is None else str(units.get(name))
            if scale != 1.0:
                series = [value * scale for value in series]
            thickness = _integrate(series, time_values)
            unit = (
                unit_override
                if unit_override is not None
                else _integral_unit(base_unit, time_unit)
            )
            meta = {
                "status": "ok",
                "source": f"data_vars.{name}",
                "method": "integrate",
            }
            if scale != 1.0:
                meta["scale"] = scale
            return {
                "name": _FILM_THICKNESS_OUTPUT,
                "value": thickness,
                "unit": unit,
                "meta": meta,
            }

        proxy_cfg = cfg.get("proxy")
        if proxy_cfg is None:
            meta = {"status": "missing_rate", "missing": missing}
            if rate_candidates:
                meta["candidates"] = list(rate_candidates)
            return {
                "name": _FILM_THICKNESS_OUTPUT,
                "value": math.nan,
                "unit": unit_override or "",
                "meta": meta,
            }
        if not isinstance(proxy_cfg, Mapping):
            raise ConfigError("film_thickness proxy must be a mapping.")

        proxy_var = proxy_cfg.get("data_var")
        if proxy_var is None:
            proxy_var = proxy_cfg.get("var")
        if proxy_var is None:
            proxy_var = proxy_cfg.get("name")
        proxy_var = _require_nonempty_str(proxy_var, "proxy.data_var")

        if proxy_var not in run_dataset.data_vars:
            meta = {
                "status": "missing_proxy",
                "missing": [f"data_vars.{proxy_var}"],
                "source": f"data_vars.{proxy_var}",
            }
            if missing:
                meta["fallback_from"] = missing
            return {
                "name": _FILM_THICKNESS_OUTPUT,
                "value": math.nan,
                "unit": unit_override or "",
                "meta": meta,
            }

        try:
            series, proxy_meta = _resolve_proxy_series(
                run_dataset,
                time_values,
                proxy_var,
                proxy_cfg,
            )
        except ConfigError as exc:
            meta = {
                "status": "invalid_proxy",
                "source": f"data_vars.{proxy_var}",
                "error": str(exc),
            }
            return {
                "name": _FILM_THICKNESS_OUTPUT,
                "value": math.nan,
                "unit": unit_override or "",
                "meta": meta,
            }

        sign = _coerce_optional_float(proxy_cfg.get("sign"), "proxy.sign", default=1.0)
        proxy_scale = _coerce_optional_float(
            proxy_cfg.get("scale"),
            "proxy.scale",
            default=scale,
        )
        if sign != 1.0 or proxy_scale != 1.0:
            series = [value * sign * proxy_scale for value in series]

        base_unit = "" if units.get(proxy_var) is None else str(units.get(proxy_var))
        thickness = _integrate(series, time_values)
        unit = (
            unit_override
            if unit_override is not None
            else _integral_unit(base_unit, time_unit)
        )
        meta = {
            "status": "proxy",
            "source": f"data_vars.{proxy_var}",
            "method": "integrate",
        }
        meta.update(proxy_meta)
        if sign != 1.0:
            meta["sign"] = sign
        if proxy_scale != 1.0:
            meta["scale"] = proxy_scale
        if missing:
            meta["fallback_from"] = missing
        return {
            "name": _FILM_THICKNESS_OUTPUT,
            "value": thickness,
            "unit": unit,
            "meta": meta,
        }


def _coerce_gas_stats(value: Any) -> list[str]:
    if value is None:
        return list(_GAS_STATS)
    raw = _coerce_str_sequence(value, "stats")
    if not raw:
        raise ConfigError("stats must contain at least one entry.")
    stats: list[str] = []
    for entry in raw:
        key = entry.strip().lower()
        if key not in _GAS_STATS:
            allowed = ", ".join(_GAS_STATS)
            raise ConfigError(f"stats entries must be one of: {allowed}.")
        if key not in stats:
            stats.append(key)
    return stats


def _coerce_rank_by(value: Any) -> str:
    if value is None:
        return "mean"
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("rank_by must be a non-empty string.")
    key = value.strip().lower()
    if key not in _GAS_STATS:
        allowed = ", ".join(_GAS_STATS)
        raise ConfigError(f"rank_by must be one of: {allowed}.")
    return key


def _coerce_top_n(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("top_n must be an integer.")
    if value <= 0:
        raise ConfigError("top_n must be a positive integer.")
    return value


def _coerce_optional_float(value: Any, label: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a float.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a float.") from exc


def _coerce_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer.")
    if value < 0:
        raise ConfigError(f"{label} must be a non-negative integer.")
    return value


def _extract_coord_data(run_dataset: RunDatasetView, name: str) -> Any:
    payload = run_dataset.coords.get(name)
    if not isinstance(payload, Mapping):
        raise ConfigError(f"coords.{name} must be a mapping.")
    if "data" not in payload:
        raise ConfigError(f"coords.{name} is missing data.")
    return payload.get("data")


def _coerce_float_sequence(value: Any, label: str) -> list[float]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{label} must be a sequence of floats.")
    values: list[float] = []
    for entry in value:
        try:
            values.append(float(entry))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label} entries must be numeric.") from exc
    if not values:
        raise ConfigError(f"{label} must contain at least one entry.")
    return values


def _coerce_matrix(value: Any, label: str) -> list[list[float]]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{label} must be a 2D sequence.")
    matrix: list[list[float]] = []
    for row in value:
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise ConfigError(f"{label} rows must be sequences.")
        row_values: list[float] = []
        for entry in row:
            try:
                row_values.append(float(entry))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{label} entries must be numeric.") from exc
        matrix.append(row_values)
    if not matrix:
        raise ConfigError(f"{label} must contain at least one row.")
    return matrix


def _validate_matrix_shape(
    matrix: Sequence[Sequence[float]],
    rows: int,
    cols: int,
    label: str,
) -> None:
    if len(matrix) != rows:
        raise ConfigError(
            f"{label} rows mismatch: expected {rows}, got {len(matrix)}."
        )
    for row in matrix:
        if len(row) != cols:
            raise ConfigError(
                f"{label} columns mismatch: expected {cols}, got {len(row)}."
            )


def _transpose_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _extract_time_species_matrix(
    payload: Any,
    time_values: Sequence[float],
    species_names: Sequence[str],
) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise ConfigError("X data_vars entry must be a mapping.")
    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError("X dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time", "species"]:
        matrix = _coerce_matrix(data, "X")
        _validate_matrix_shape(
            matrix,
            len(time_values),
            len(species_names),
            "X",
        )
        return matrix
    if dims_list == ["species", "time"]:
        matrix = _coerce_matrix(data, "X")
        _validate_matrix_shape(
            matrix,
            len(species_names),
            len(time_values),
            "X",
        )
        return _transpose_matrix(matrix)
    raise ConfigError("X dims must be [time, species] or [species, time].")


def _extract_time_surface_matrix(
    payload: Any,
    time_values: Sequence[float],
    surface_species: Sequence[str],
) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise ConfigError("coverage data_vars entry must be a mapping.")
    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError("coverage dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time", "surface_species"]:
        matrix = _coerce_matrix(data, "coverage")
        _validate_matrix_shape(
            matrix,
            len(time_values),
            len(surface_species),
            "coverage",
        )
        return matrix
    if dims_list == ["surface_species", "time"]:
        matrix = _coerce_matrix(data, "coverage")
        _validate_matrix_shape(
            matrix,
            len(surface_species),
            len(time_values),
            "coverage",
        )
        return _transpose_matrix(matrix)
    raise ConfigError(
        "coverage dims must be [time, surface_species] or [surface_species, time]."
    )


def _extract_time_series(
    payload: Any,
    time_values: Sequence[float],
    label: str,
) -> list[float]:
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{label} data_vars entry must be a mapping.")
    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError(f"{label} dims must be a sequence.")
    dims_list = list(dims)
    if dims_list != ["time"]:
        raise ConfigError(f"{label} dims must be [time].")
    series = _coerce_float_sequence(data, label)
    if len(series) != len(time_values):
        raise ConfigError(f"{label} length must match time dimension.")
    return series


def _resolve_proxy_index(
    names: Sequence[str],
    proxy_cfg: Mapping[str, Any],
    axis: str,
) -> tuple[int, dict[str, Any]]:
    if not names:
        raise ConfigError("proxy species list is empty.")
    species = proxy_cfg.get("species")
    if species is not None:
        species_name = _require_nonempty_str(species, "proxy.species")
        if species_name not in names:
            raise ConfigError(f"proxy species {species_name!r} not found.")
        index = names.index(species_name)
        return index, {"proxy_axis": axis, "proxy_species": species_name}
    index = _coerce_optional_int(proxy_cfg.get("index"), "proxy.index")
    if index is not None:
        if index >= len(names):
            raise ConfigError("proxy.index is out of range.")
        meta = {"proxy_axis": axis, "proxy_index": index, "proxy_species": names[index]}
        return index, meta
    if len(names) == 1:
        return 0, {"proxy_axis": axis, "proxy_species": names[0], "proxy_index": 0}
    raise ConfigError(
        "proxy.species or proxy.index is required when multiple entries exist."
    )


def _resolve_proxy_series(
    run_dataset: RunDatasetView,
    time_values: Sequence[float],
    proxy_var: str,
    proxy_cfg: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    payload = run_dataset.data_vars.get(proxy_var)
    if not isinstance(payload, Mapping):
        raise ConfigError("proxy data_vars entry must be a mapping.")
    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError("proxy dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time"]:
        if proxy_cfg.get("species") is not None or proxy_cfg.get("index") is not None:
            raise ConfigError("proxy selector requires species dimension.")
        series = _coerce_float_sequence(data, proxy_var)
        if len(series) != len(time_values):
            raise ConfigError("proxy length must match time dimension.")
        return series, {"proxy_axis": "time"}
    if "time" not in dims_list:
        raise ConfigError("proxy dims must include time.")
    if "species" in dims_list:
        axis = "species"
        species_names = _coerce_str_sequence(
            _extract_coord_data(run_dataset, "species"),
            "coords.species",
        )
        matrix = _extract_time_species_matrix(payload, time_values, species_names)
    elif "surface_species" in dims_list:
        axis = "surface_species"
        species_names = _coerce_str_sequence(
            _extract_coord_data(run_dataset, "surface_species"),
            "coords.surface_species",
        )
        matrix = _extract_time_surface_matrix(payload, time_values, species_names)
    else:
        raise ConfigError("proxy dims must include species or surface_species.")
    index, meta = _resolve_proxy_index(species_names, proxy_cfg, axis)
    series = [row[index] for row in matrix]
    return series, meta


def _series_stats(
    values: Sequence[float],
    time_values: Sequence[float],
) -> dict[str, float]:
    if len(values) != len(time_values):
        raise ConfigError("X data length must match time dimension.")
    if not values:
        raise ConfigError("X data must contain at least one entry.")
    last = values[-1]
    mean = sum(values) / float(len(values))
    max_value = max(values)
    integral = _integrate(values, time_values)
    return {
        "last": last,
        "mean": mean,
        "max": max_value,
        "integral": integral,
    }


def _integrate(values: Sequence[float], time_values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(values)):
        dt = time_values[index] - time_values[index - 1]
        total += 0.5 * (values[index] + values[index - 1]) * dt
    return total


def _compute_stats_by_species(
    species_names: Sequence[str],
    matrix: Sequence[Sequence[float]],
    time_values: Sequence[float],
    stats_to_compute: set[str],
) -> dict[str, dict[str, float]]:
    stats_by_species: dict[str, dict[str, float]] = {}
    for index, name in enumerate(species_names):
        series = [row[index] for row in matrix]
        all_stats = _series_stats(series, time_values)
        stats_by_species[name] = {
            key: value for key, value in all_stats.items() if key in stats_to_compute
        }
    return stats_by_species


def _select_species(
    species_filter: Sequence[str],
    top_n: Optional[int],
    rank_by: str,
    stats_by_species: Mapping[str, Mapping[str, float]],
) -> list[str]:
    if species_filter:
        unknown = [name for name in species_filter if name not in stats_by_species]
        if unknown:
            missing = ", ".join(unknown)
            raise ConfigError(f"Unknown species requested: {missing}.")
        return list(species_filter)
    if top_n is None:
        return list(stats_by_species.keys())
    if top_n >= len(stats_by_species):
        return list(stats_by_species.keys())
    ranked = sorted(
        stats_by_species.items(),
        key=lambda item: _rank_key(item[0], item[1], rank_by),
    )
    selected = [name for name, _ in ranked[:top_n]]
    return sorted(selected)


def _rank_key(
    name: str,
    stats: Mapping[str, float],
    rank_by: str,
) -> tuple[float, str]:
    value = stats.get(rank_by, math.nan)
    if isinstance(value, float) and math.isnan(value):
        value = -math.inf
    return (-float(value), name)


def _integral_unit(base_unit: str, time_unit: str) -> str:
    if base_unit and time_unit:
        return f"{base_unit}*{time_unit}"
    if base_unit:
        return base_unit
    if time_unit:
        return time_unit
    return ""


def _parse_elements_from_name(species_name: str) -> list[str]:
    matches = _ELEMENT_PATTERN.findall(species_name)
    if not matches:
        return []
    elements: list[str] = []
    for symbol, _ in matches:
        if symbol not in elements:
            elements.append(symbol)
    return elements


def _coerce_elements_entry(entry: Any, label: str) -> list[str]:
    if isinstance(entry, Mapping):
        return [_require_nonempty_str(key, label) for key in entry.keys()]
    return _coerce_str_sequence(entry, label)


def _resolve_species_elements(
    species_names: Sequence[str],
    cfg: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> dict[str, list[str]]:
    mapping = cfg.get("species_elements")
    if mapping is None:
        mapping = attrs.get("species_elements")
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise ConfigError("species_elements must be a mapping.")
        resolved: dict[str, list[str]] = {}
        for name in species_names:
            entry = mapping.get(name)
            if entry is None:
                continue
            elements = _coerce_elements_entry(entry, f"species_elements[{name}]")
            if elements:
                resolved[name] = list(dict.fromkeys(elements))
        if resolved:
            return resolved
    resolved = {}
    for name in species_names:
        elements = _parse_elements_from_name(name)
        if elements:
            resolved[name] = elements
    return resolved


def _build_element_rows(
    species_names: Sequence[str],
    matrix: Sequence[Sequence[float]],
    time_values: Sequence[float],
    stats: Sequence[str],
    stats_to_compute: set[str],
    base_unit: str,
    integral_unit: str,
    elements_filter: Sequence[str],
    cfg: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    species_elements = _resolve_species_elements(species_names, cfg, attrs)
    if not species_elements:
        return []
    element_to_indices: dict[str, list[int]] = {}
    for index, name in enumerate(species_names):
        for element in species_elements.get(name, []):
            element_to_indices.setdefault(element, []).append(index)
    if elements_filter:
        allowed = set(elements_filter)
        element_to_indices = {
            element: indices
            for element, indices in element_to_indices.items()
            if element in allowed
        }
    if not element_to_indices:
        return []
    rows: list[dict[str, Any]] = []
    for element in sorted(element_to_indices.keys()):
        indices = element_to_indices[element]
        series = [
            sum(row[index] for index in indices) for row in matrix
        ]
        all_stats = _series_stats(series, time_values)
        for stat in stats:
            if stat not in stats_to_compute:
                continue
            unit = base_unit if stat != "integral" else integral_unit
            rows.append(
                {
                    "observable": f"{_ELEMENT_PREFIX}.{element}.{stat}",
                    "value": all_stats[stat],
                    "unit": unit,
                }
            )
    return rows


def _extract_observables_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "observables" in cfg and isinstance(cfg.get("observables"), Mapping):
        obs_cfg = cfg.get("observables")
        if not isinstance(obs_cfg, Mapping):
            raise ConfigError("observables config must be a mapping.")
        return dict(cfg), dict(obs_cfg)
    if "observable" in cfg and isinstance(cfg.get("observable"), Mapping):
        obs_cfg = cfg.get("observable")
        if not isinstance(obs_cfg, Mapping):
            raise ConfigError("observable config must be a mapping.")
        return dict(cfg), dict(obs_cfg)
    return dict(cfg), dict(cfg)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_str_sequence(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_require_nonempty_str(value, label)]
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items: list[str] = []
        for entry in value:
            items.append(_require_nonempty_str(entry, label))
        return items
    raise ConfigError(f"{label} must be a string or sequence of strings.")


def _coerce_run_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_require_nonempty_str(value, "run_id")]
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        run_ids: list[str] = []
        for entry in value:
            run_ids.append(_require_nonempty_str(entry, "run_id"))
        return run_ids
    raise ConfigError("run_id(s) must be a string or sequence of strings.")


def _extract_run_ids(
    obs_cfg: Mapping[str, Any],
    *,
    store: Optional[ArtifactStore] = None,
) -> list[str]:
    inputs = obs_cfg.get("inputs")
    run_set_id: Any = None
    run_ids: Any = None
    if inputs is None:
        run_ids = None
    elif not isinstance(inputs, Mapping):
        raise ConfigError("observables.inputs must be a mapping.")
    else:
        if "run_set_id" in inputs:
            run_set_id = inputs.get("run_set_id")
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in inputs:
                run_ids = inputs.get(key)
                break
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is None and "run_set_id" in obs_cfg:
        run_set_id = obs_cfg.get("run_set_id")
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is not None:
        run_set_id = _require_nonempty_str(run_set_id, "run_set_id")
        if store is None:
            raise ConfigError("run_set_id requires a store to be provided.")
        return load_run_ids_from_run_set(store, run_set_id)
    if run_ids is None:
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in obs_cfg:
                run_ids = obs_cfg.get(key)
                break
    run_id_list = _coerce_run_ids(run_ids)
    if not run_id_list:
        raise ConfigError("observables run_id is required.")
    return run_id_list


def _extract_params(obs_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = obs_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("observables.params must be a mapping.")
    return dict(params)


def _normalize_observable_spec(entry: Mapping[str, Any], label: str) -> ObservableSpec:
    name = entry.get("name") or entry.get("observable")
    name = _require_nonempty_str(name, f"{label}.name")
    params = entry.get("params")
    if params is None:
        params = entry.get("config")
    if params is None:
        params = {
            key: value
            for key, value in entry.items()
            if key not in ("name", "observable")
        }
    if not isinstance(params, Mapping):
        raise ConfigError(f"{label}.params must be a mapping.")
    return ObservableSpec(name=name, params=dict(params))


def _normalize_observable_specs(raw: Any) -> list[ObservableSpec]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "name" in raw or "observable" in raw:
            return [_normalize_observable_spec(raw, "observables")]
        specs: list[ObservableSpec] = []
        for key, value in raw.items():
            name = _require_nonempty_str(key, "observables")
            if value is None:
                params = {}
            elif not isinstance(value, Mapping):
                raise ConfigError("observables mapping values must be mappings.")
            else:
                params = dict(value)
            specs.append(ObservableSpec(name=name, params=params))
        return specs
    if isinstance(raw, str):
        return [ObservableSpec(name=_require_nonempty_str(raw, "observables"), params={})]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        specs: list[ObservableSpec] = []
        for index, entry in enumerate(raw):
            if isinstance(entry, str):
                specs.append(
                    ObservableSpec(
                        name=_require_nonempty_str(entry, "observables"),
                        params={},
                    )
                )
                continue
            if isinstance(entry, Mapping):
                specs.append(
                    _normalize_observable_spec(entry, f"observables[{index}]")
                )
                continue
            raise ConfigError(
                "observables entries must be strings or mappings."
            )
        return specs
    raise ConfigError("observables must be a mapping, list, or string.")


def _normalize_missing_strategy(value: Any) -> str:
    if value is None:
        return DEFAULT_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in {"nan", "skip"}:
        raise ConfigError("missing_strategy must be 'nan' or 'skip'.")
    return strategy


def _load_run_dataset_payload(run_dir: Path) -> dict[str, Any]:
    dataset_dir = resolve_run_dataset_dir(run_dir)
    if dataset_dir is None:
        raise ArtifactError(
            "Run dataset not found; expected sim/timeseries.zarr or state.zarr."
        )
    dataset_path = dataset_dir / "dataset.json"
    if dataset_path.exists():
        return load_run_dataset_payload(run_dir, dataset_dir=dataset_dir)
    if xr is None:
        raise ArtifactError(
            "Run dataset not found; install xarray to load timeseries.zarr."
        )
    dataset = xr.open_zarr(dataset_dir)
    coords = _LazyDatasetMapping(dataset, "coords")
    data_vars = _LazyDatasetMapping(dataset, "data_vars")
    return {"coords": coords, "data_vars": data_vars, "attrs": dict(dataset.attrs)}


def _load_run_dataset_view(run_dir: Path) -> RunDatasetView:
    payload = _load_run_dataset_payload(run_dir)
    coords = payload.get("coords", {})
    data_vars = payload.get("data_vars", {})
    attrs = payload.get("attrs", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    if not isinstance(data_vars, Mapping):
        raise ArtifactError("Run dataset data_vars must be a mapping.")
    if not isinstance(attrs, Mapping):
        raise ArtifactError("Run dataset attrs must be a mapping.")
    return RunDatasetView(
        coords=coords,
        data_vars=data_vars,
        attrs=attrs,
        raw=dict(payload),
    )


def _observable_requirements(obs: Any) -> tuple[list[str], list[str], list[str]]:
    requires = _coerce_str_sequence(getattr(obs, "requires", None), "requires")
    requires_coords = _coerce_str_sequence(
        getattr(obs, "requires_coords", None),
        "requires_coords",
    )
    requires_attrs = _coerce_str_sequence(
        getattr(obs, "requires_attrs", None),
        "requires_attrs",
    )
    return requires, requires_coords, requires_attrs


def _missing_inputs(
    run_dataset: RunDatasetView,
    *,
    requires: Sequence[str],
    requires_coords: Sequence[str],
    requires_attrs: Sequence[str],
) -> list[str]:
    missing: list[str] = []
    for name in requires:
        if name not in run_dataset.data_vars:
            missing.append(f"data_vars.{name}")
    for name in requires_coords:
        if name not in run_dataset.coords:
            missing.append(f"coords.{name}")
    for name in requires_attrs:
        if name not in run_dataset.attrs:
            missing.append(f"attrs.{name}")
    return missing


def _resolve_observable(
    name: str,
    *,
    registry: Optional[Registry],
) -> Any:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("observable name must be a non-empty string.")
    if registry is None:
        try:
            from rxn_platform.registry import get as registry_get

            return registry_get("observable", name)
        except KeyError as exc:
            from rxn_platform.registry import list as registry_list

            available = ", ".join(sorted(registry_list("observable"))) or "<none>"
            raise ConfigError(
                f"Observable {name!r} is not registered. Available: {available}."
            ) from exc
    try:
        return registry.get("observable", name)
    except KeyError as exc:
        available = ", ".join(sorted(registry.list("observable"))) or "<none>"
        raise ConfigError(
            f"Observable {name!r} is not registered. Available: {available}."
        ) from exc


def _call_observable(
    obs: Any,
    run_dataset: RunDatasetView,
    params: Mapping[str, Any],
) -> Any:
    if hasattr(obs, "compute"):
        func = obs.compute
    else:
        func = obs
    if not callable(func):
        raise ConfigError("Observable entry is not callable.")
    return func(run_dataset, params)


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


def _compose_observable_name(base: str, suffix: Any) -> str:
    suffix_str = str(suffix)
    if suffix_str == base or suffix_str.startswith(f"{base}."):
        return suffix_str
    return f"{base}.{suffix_str}"


def _build_row(
    run_id: str,
    observable: str,
    value: Any,
    unit: Any,
    meta: Any,
) -> dict[str, Any]:
    observable_name = _require_nonempty_str(observable, "observable")
    unit_str = "" if unit is None else str(unit)
    meta_payload = _coerce_meta(meta)
    value_float = _coerce_value(value, meta_payload)
    try:
        meta_json = json.dumps(
            meta_payload,
            ensure_ascii=True,
            sort_keys=True,
        )
    except TypeError:
        meta_json = json.dumps(
            {"detail": str(meta_payload)},
            ensure_ascii=True,
            sort_keys=True,
        )
    return {
        "run_id": run_id,
        "observable": observable_name,
        "value": value_float,
        "unit": unit_str,
        "meta_json": meta_json,
    }


def _rows_from_values(
    run_id: str,
    base_name: str,
    values: Any,
    unit: Any,
    meta: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if values is None:
        rows.append(_build_row(run_id, base_name, None, unit, meta))
        return rows
    if isinstance(values, Mapping):
        for key, entry in values.items():
            row_unit = unit
            row_meta = meta
            row_value = entry
            if isinstance(entry, Mapping) and (
                "value" in entry or "unit" in entry or "meta" in entry
            ):
                row_value = entry.get("value")
                row_unit = entry.get("unit", unit)
                row_meta = entry.get("meta", meta)
            rows.append(
                _build_row(
                    run_id,
                    _compose_observable_name(base_name, key),
                    row_value,
                    row_unit,
                    row_meta,
                )
            )
        return rows
    if isinstance(values, Sequence) and not isinstance(
        values,
        (str, bytes, bytearray),
    ):
        for index, entry in enumerate(values):
            rows.append(
                _build_row(
                    run_id,
                    _compose_observable_name(base_name, index),
                    entry,
                    unit,
                    meta,
                )
            )
        return rows
    rows.append(_build_row(run_id, base_name, values, unit, meta))
    return rows


def _normalize_mapping_output(
    run_id: str,
    base_name: str,
    output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if {"value", "values", "unit", "meta", "meta_json", "name", "observable"} & set(
        output.keys()
    ):
        if "values" in output:
            return _rows_from_values(
                run_id,
                base_name,
                output.get("values"),
                output.get("unit", ""),
                output.get("meta", output.get("meta_json")),
            )
        if "value" in output:
            name = output.get("name") or output.get("observable") or base_name
            return [
                _build_row(
                    run_id,
                    str(name),
                    output.get("value"),
                    output.get("unit", ""),
                    output.get("meta", output.get("meta_json")),
                )
            ]
        raise ConfigError("Observable output is missing 'value' or 'values'.")
    return _rows_from_values(run_id, base_name, output, "", {})


def _normalize_output(
    run_id: str,
    base_name: str,
    output: Any,
) -> list[dict[str, Any]]:
    if output is None:
        return []
    if pd is not None and isinstance(output, pd.DataFrame):
        rows: list[dict[str, Any]] = []
        for _, row in output.iterrows():
            row_dict = row.to_dict()
            if not isinstance(row_dict, Mapping):
                continue
            rows.extend(_normalize_mapping_output(run_id, base_name, row_dict))
        return rows
    if isinstance(output, Mapping):
        return _normalize_mapping_output(run_id, base_name, output)
    if isinstance(output, Sequence) and not isinstance(
        output, (str, bytes, bytearray)
    ):
        rows: list[dict[str, Any]] = []
        for entry in output:
            if isinstance(entry, Mapping):
                rows.extend(_normalize_mapping_output(run_id, base_name, entry))
            else:
                rows.append(_build_row(run_id, base_name, entry, "", {}))
        return rows
    return [_build_row(run_id, base_name, output, "", {})]


def _write_values_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if pd is not None:
        frame = pd.DataFrame(list(rows), columns=REQUIRED_COLUMNS)
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            pass
    if pa is not None and pq is not None:
        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("observable", pa.string()),
                ("value", pa.float64()),
                ("unit", pa.string()),
                ("meta_json", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(table, path)
        return
    payload = {
        "columns": list(REQUIRED_COLUMNS),
        "rows": list(rows),
    }
    write_json_atomic(path, payload)
    json_path = path.with_suffix(".json")
    if json_path != path:
        write_json_atomic(json_path, payload)
    logger = logging.getLogger("rxn_platform.observables")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s and %s.",
        path,
        json_path,
    )


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Compute observables for one or more run artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, obs_cfg = _extract_observables_cfg(resolved_cfg)
    params = _extract_params(obs_cfg)
    run_ids = _extract_run_ids(obs_cfg, store=store)

    observables_raw = params.get("observables", obs_cfg.get("observables"))
    specs = _normalize_observable_specs(observables_raw)
    if not specs:
        raise ConfigError("observables list must not be empty.")
    missing_strategy = _normalize_missing_strategy(
        params.get("missing_strategy", obs_cfg.get("missing_strategy"))
    )

    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        run_dataset = _load_run_dataset_view(run_dir)
        for spec in specs:
            obs = _resolve_observable(spec.name, registry=registry)
            requires, requires_coords, requires_attrs = _observable_requirements(obs)
            missing = _missing_inputs(
                run_dataset,
                requires=requires,
                requires_coords=requires_coords,
                requires_attrs=requires_attrs,
            )
            if missing:
                status = "skipped" if missing_strategy == "skip" else "missing_input"
                meta = {"status": status, "missing": missing}
                rows.append(
                    _build_row(run_id, spec.name, math.nan, "", meta)
                )
                continue
            output = _call_observable(obs, run_dataset, spec.params)
            rows.extend(_normalize_output(run_id, spec.name, output))

    inputs_payload = {
        "runs": run_ids,
        "observables": [spec.name for spec in specs],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="observables",
        artifact_id=artifact_id,
        parents=run_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_values_table(rows, base_dir / "values.parquet")

    return store.ensure(manifest, writer=_writer)


register("task", "observables.run", run)
register("task", "observables.compute", run)
register("observable", "gas_composition", GasCompositionObservable())
register("observable", "ignition_delay", IgnitionDelayObservable())
register("observable", "coverage_summary", CoverageObservable())
register("observable", "film_thickness", FilmThicknessObservable())

__all__ = [
    "FilmThicknessObservable",
    "GasCompositionObservable",
    "IgnitionDelayObservable",
    "CoverageObservable",
    "Observable",
    "ObservableSpec",
    "RunDatasetView",
    "run",
]
