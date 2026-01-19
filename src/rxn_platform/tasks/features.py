"""Feature extraction framework and time-series summary implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
import logging
import math
import platform
from pathlib import Path
import subprocess
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import ArtifactManifest, make_artifact_id
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore

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

try:  # Optional dependency.
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None

DEFAULT_MISSING_STRATEGY = "nan"
DEFAULT_STATS = ("mean", "max", "min", "last", "integral")
DEFAULT_ROP_STATS = ("integral", "max")
DEFAULT_ROP_TOP_N = 5
DEFAULT_ROP_RANK_BY = "integral"
DEFAULT_ROP_RANK_ABS = True
DEFAULT_NETWORK_METRICS = (
    "degree",
    "in_degree",
    "out_degree",
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
)
DEFAULT_NETWORK_TOP_N = 25
DEFAULT_NETWORK_STABILITY_TOP_N = 10
REQUIRED_COLUMNS = ("run_id", "feature", "value", "unit", "meta_json")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class VariableSpec:
    name: str
    stats: list[str]
    species: list[str]
    top_n: Optional[int]
    rank_by: str
    axis: Optional[str]
    base_name: str


@dataclass(frozen=True)
class RopWdotSpec:
    name: str
    axis: str
    id_label: str
    base_name: str
    stats: list[str]
    top_n: Optional[int]
    rank_by: str
    rank_abs: bool


@dataclass(frozen=True)
class RunDatasetView:
    coords: Mapping[str, Any]
    data_vars: Mapping[str, Any]
    attrs: Mapping[str, Any]
    raw: Mapping[str, Any]


class FeatureExtractor:
    """Base class for feature extraction plugins."""

    name: str
    requires: Sequence[str] = ()
    requires_coords: Sequence[str] = ()
    requires_attrs: Sequence[str] = ()

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError("FeatureExtractor implementations must override compute().")


class TimeseriesSummaryFeature(FeatureExtractor):
    name = "timeseries_summary"

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("timeseries_summary params must be a mapping.")

        stats = _normalize_stats(cfg.get("stats"))
        variables_raw = cfg.get("variables", cfg.get("data_vars", cfg.get("vars")))
        var_specs = _normalize_variable_specs(variables_raw, stats)
        if not var_specs:
            raise ConfigError("timeseries_summary variables must be provided.")

        time_values: Optional[list[float]]
        time_error: Optional[str] = None
        try:
            time_values = _extract_time_values(run_dataset)
        except ConfigError as exc:
            time_values = None
            time_error = str(exc)

        units = run_dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")

        rows: list[dict[str, Any]] = []
        for spec in var_specs:
            rows.extend(
                _summarize_variable(
                    spec,
                    run_dataset,
                    time_values,
                    time_error,
                    units,
                )
            )
        return rows


class RopWdotFeature(FeatureExtractor):
    name = "rop_wdot_summary"

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("rop_wdot_summary params must be a mapping.")

        stats_default = (
            _normalize_stats(cfg.get("stats"))
            if cfg.get("stats") is not None
            else list(DEFAULT_ROP_STATS)
        )
        if "top_n" in cfg:
            top_n_default = _coerce_optional_int(cfg.get("top_n"), "top_n")
        else:
            top_n_default = DEFAULT_ROP_TOP_N
        rank_by_default = _normalize_rank_by(cfg.get("rank_by") or DEFAULT_ROP_RANK_BY)
        rank_abs_default = _normalize_rank_abs(cfg.get("rank_abs"), DEFAULT_ROP_RANK_ABS)

        rop_spec = _normalize_rop_wdot_spec(
            cfg.get("rop"),
            "rop",
            default_name="rop_net",
            axis="reaction",
            id_label="reaction_id",
            default_stats=stats_default,
            default_top_n=top_n_default,
            default_rank_by=rank_by_default,
            default_rank_abs=rank_abs_default,
        )
        wdot_spec = _normalize_rop_wdot_spec(
            cfg.get("wdot"),
            "wdot",
            default_name="net_production_rates",
            axis="species",
            id_label="species",
            default_stats=stats_default,
            default_top_n=top_n_default,
            default_rank_by=rank_by_default,
            default_rank_abs=rank_abs_default,
        )
        if rop_spec is None and wdot_spec is None:
            raise ConfigError("rop_wdot_summary requires rop or wdot configuration.")

        time_values: Optional[list[float]]
        time_error: Optional[str] = None
        try:
            time_values = _extract_time_values(run_dataset)
        except ConfigError as exc:
            time_values = None
            time_error = str(exc)

        units = run_dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")

        rows: list[dict[str, Any]] = []
        for spec in (rop_spec, wdot_spec):
            if spec is None:
                continue
            rows.extend(
                _summarize_ranked_variable(
                    spec,
                    run_dataset,
                    time_values,
                    time_error,
                    units,
                )
            )
        return rows


class NetworkMetricFeature(FeatureExtractor):
    name = "network_metrics"

    def compute(
        self,
        run_dataset: RunDatasetView,
        cfg: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(cfg, Mapping):
            raise ConfigError("network_metrics params must be a mapping.")

        graph_payload = cfg.get("graph_payload")
        graph_id = _extract_graph_id_from_params(cfg)
        if graph_payload is None:
            return _network_missing_rows(
                graph_id,
                "graph_payload is missing; supply graph_id in features params.",
            )
        if not isinstance(graph_payload, Mapping):
            raise ConfigError("graph_payload must be a mapping.")

        metrics = _normalize_network_metrics(cfg.get("metrics"))
        if "top_n" in cfg:
            top_n = _coerce_optional_int(cfg.get("top_n"), "top_n")
        else:
            top_n = DEFAULT_NETWORK_TOP_N
        node_kinds = _coerce_str_sequence(cfg.get("node_kinds") or cfg.get("kinds"), "node_kinds")

        return _network_metric_rows(
            graph_payload,
            graph_id=graph_id,
            metrics=metrics,
            top_n=top_n,
            node_kinds=node_kinds,
            direction_mode=cfg.get("direction_mode"),
            directed_override=cfg.get("directed"),
            betweenness_max_nodes=cfg.get("betweenness_max_nodes"),
        )


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


def _extract_features_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "features" in cfg and isinstance(cfg.get("features"), Mapping):
        feat_cfg = cfg.get("features")
        if not isinstance(feat_cfg, Mapping):
            raise ConfigError("features config must be a mapping.")
        return dict(cfg), dict(feat_cfg)
    if "feature" in cfg and isinstance(cfg.get("feature"), Mapping):
        feat_cfg = cfg.get("feature")
        if not isinstance(feat_cfg, Mapping):
            raise ConfigError("feature config must be a mapping.")
        return dict(cfg), dict(feat_cfg)
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


def _coerce_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer.")
    if value <= 0:
        raise ConfigError(f"{label} must be a positive integer.")
    return value


def _normalize_stats(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_STATS)
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = list(value)
    else:
        raise ConfigError("stats must be a string or list of strings.")
    stats: list[str] = []
    for entry in raw:
        key = _require_nonempty_str(entry, "stats").lower()
        if key not in DEFAULT_STATS:
            allowed = ", ".join(DEFAULT_STATS)
            raise ConfigError(f"stats entries must be one of: {allowed}.")
        if key not in stats:
            stats.append(key)
    if not stats:
        raise ConfigError("stats must include at least one entry.")
    return stats


def _normalize_rank_by(value: Any) -> str:
    if value is None:
        return "mean"
    key = _require_nonempty_str(value, "rank_by").lower()
    if key not in DEFAULT_STATS:
        allowed = ", ".join(DEFAULT_STATS)
        raise ConfigError(f"rank_by must be one of: {allowed}.")
    return key


def _normalize_axis(value: Any) -> Optional[str]:
    if value is None:
        return None
    axis = _require_nonempty_str(value, "axis").lower()
    if axis not in {"species", "surface_species"}:
        raise ConfigError("axis must be 'species' or 'surface_species'.")
    return axis


def _normalize_rank_abs(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ConfigError("rank_abs must be a boolean.")


def _normalize_rop_wdot_spec(
    entry: Any,
    label: str,
    *,
    default_name: str,
    axis: str,
    id_label: str,
    default_stats: Sequence[str],
    default_top_n: Optional[int],
    default_rank_by: str,
    default_rank_abs: bool,
) -> Optional[RopWdotSpec]:
    if entry is False:
        return None
    if entry is None:
        payload: dict[str, Any] = {}
    elif isinstance(entry, Mapping):
        payload = dict(entry)
    else:
        raise ConfigError(f"{label} must be a mapping or false.")

    name = payload.get("name") or payload.get("var") or payload.get("variable")
    if name is None:
        name = default_name
    name = _require_nonempty_str(name, f"{label}.name")

    base_name = payload.get("feature") or payload.get("label") or payload.get("prefix")
    if base_name is None:
        base_name = name
    base_name = _require_nonempty_str(base_name, f"{label}.feature")

    stats_raw = payload.get("stats")
    stats = _normalize_stats(stats_raw) if stats_raw is not None else list(default_stats)

    if "top_n" in payload:
        top_n = _coerce_optional_int(payload.get("top_n"), f"{label}.top_n")
    else:
        top_n = default_top_n

    rank_by = _normalize_rank_by(payload.get("rank_by") or default_rank_by)
    rank_abs = _normalize_rank_abs(payload.get("rank_abs"), default_rank_abs)

    return RopWdotSpec(
        name=name,
        axis=axis,
        id_label=id_label,
        base_name=base_name,
        stats=stats,
        top_n=top_n,
        rank_by=rank_by,
        rank_abs=rank_abs,
    )


def _normalize_variable_spec(
    entry: Mapping[str, Any],
    label: str,
    default_stats: Sequence[str],
) -> VariableSpec:
    name = entry.get("name") or entry.get("var") or entry.get("variable")
    name = _require_nonempty_str(name, f"{label}.name")

    stats_raw = entry.get("stats")
    stats = _normalize_stats(stats_raw) if stats_raw is not None else list(default_stats)

    species = _coerce_str_sequence(entry.get("species"), f"{label}.species")
    surface_species = _coerce_str_sequence(
        entry.get("surface_species"), f"{label}.surface_species"
    )
    axis = _normalize_axis(entry.get("axis") or entry.get("dim"))
    if surface_species:
        axis = "surface_species"
        species = surface_species
    elif axis is None and species:
        axis = "species"

    top_n = _coerce_optional_int(entry.get("top_n"), f"{label}.top_n")
    rank_by = _normalize_rank_by(entry.get("rank_by"))

    base_name = entry.get("feature") or entry.get("label") or entry.get("prefix")
    if base_name is None:
        base_name = name
    base_name = _require_nonempty_str(base_name, f"{label}.feature")

    return VariableSpec(
        name=name,
        stats=stats,
        species=species,
        top_n=top_n,
        rank_by=rank_by,
        axis=axis,
        base_name=base_name,
    )


def _normalize_variable_specs(
    raw: Any,
    default_stats: Sequence[str],
) -> list[VariableSpec]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "name" in raw or "var" in raw or "variable" in raw:
            return [_normalize_variable_spec(raw, "variables", default_stats)]
        specs: list[VariableSpec] = []
        for key, value in raw.items():
            name = _require_nonempty_str(key, "variables")
            params: dict[str, Any] = {}
            if value is None:
                params = {}
            elif isinstance(value, Mapping):
                params = dict(value)
            else:
                raise ConfigError("variables mapping values must be mappings.")
            params["name"] = name
            specs.append(
                _normalize_variable_spec(params, f"variables[{name}]", default_stats)
            )
        return specs
    if isinstance(raw, str):
        return [
            _normalize_variable_spec({"name": raw}, "variables", default_stats)
        ]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        specs: list[VariableSpec] = []
        for index, entry in enumerate(raw):
            if isinstance(entry, str):
                specs.append(
                    _normalize_variable_spec(
                        {"name": entry}, f"variables[{index}]", default_stats
                    )
                )
                continue
            if isinstance(entry, Mapping):
                specs.append(
                    _normalize_variable_spec(entry, f"variables[{index}]", default_stats)
                )
                continue
            raise ConfigError("variables entries must be strings or mappings.")
        return specs
    raise ConfigError("variables must be a mapping, list, or string.")


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


def _extract_run_ids(feat_cfg: Mapping[str, Any]) -> list[str]:
    inputs = feat_cfg.get("inputs")
    run_ids: Any = None
    if inputs is None:
        run_ids = None
    elif not isinstance(inputs, Mapping):
        raise ConfigError("features.inputs must be a mapping.")
    else:
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in inputs:
                run_ids = inputs.get(key)
                break
    if run_ids is None:
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in feat_cfg:
                run_ids = feat_cfg.get(key)
                break
    run_id_list = _coerce_run_ids(run_ids)
    if not run_id_list:
        raise ConfigError("features run_id is required.")
    return run_id_list


def _extract_params(feat_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = feat_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("features.params must be a mapping.")
    return dict(params)


def _normalize_missing_strategy(value: Any) -> str:
    if value is None:
        return DEFAULT_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in {"nan", "skip"}:
        raise ConfigError("missing_strategy must be 'nan' or 'skip'.")
    return strategy


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


def _load_run_dataset_payload(run_dir: Path) -> dict[str, Any]:
    dataset_path = run_dir / "state.zarr" / "dataset.json"
    if dataset_path.exists():
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Run dataset JSON is invalid: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("Run dataset JSON must be a mapping.")
        return dict(payload)
    if xr is None:
        raise ArtifactError(
            "Run dataset not found; install xarray to load state.zarr."
        )
    dataset = xr.open_zarr(run_dir / "state.zarr")
    coords = {
        name: {"dims": [name], "data": dataset.coords[name].values.tolist()}
        for name in dataset.coords
    }
    data_vars = {
        name: {"dims": list(dataset[name].dims), "data": dataset[name].values.tolist()}
        for name in dataset.data_vars
    }
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
        coords=dict(coords),
        data_vars=dict(data_vars),
        attrs=dict(attrs),
        raw=dict(payload),
    )


def _feature_requirements(feature: Any) -> tuple[list[str], list[str], list[str]]:
    requires = _coerce_str_sequence(getattr(feature, "requires", None), "requires")
    requires_coords = _coerce_str_sequence(
        getattr(feature, "requires_coords", None),
        "requires_coords",
    )
    requires_attrs = _coerce_str_sequence(
        getattr(feature, "requires_attrs", None),
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


def _resolve_feature(
    name: str,
    *,
    registry: Optional[Registry],
) -> Any:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("feature name must be a non-empty string.")
    if registry is None:
        try:
            from rxn_platform.registry import get as registry_get

            return registry_get("feature", name)
        except KeyError as exc:
            from rxn_platform.registry import list as registry_list

            available = ", ".join(sorted(registry_list("feature"))) or "<none>"
            raise ConfigError(
                f"Feature {name!r} is not registered. Available: {available}."
            ) from exc
    try:
        return registry.get("feature", name)
    except KeyError as exc:
        available = ", ".join(sorted(registry.list("feature"))) or "<none>"
        raise ConfigError(
            f"Feature {name!r} is not registered. Available: {available}."
        ) from exc


def _call_feature(
    feature: Any,
    run_dataset: RunDatasetView,
    params: Mapping[str, Any],
) -> Any:
    if hasattr(feature, "compute"):
        func = feature.compute
    else:
        func = feature
    if not callable(func):
        raise ConfigError("Feature entry is not callable.")
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


def _compose_feature_name(base: str, suffix: Any) -> str:
    suffix_str = str(suffix)
    if suffix_str == base or suffix_str.startswith(f"{base}."):
        return suffix_str
    return f"{base}.{suffix_str}"


def _build_row(
    run_id: str,
    feature: str,
    value: Any,
    unit: Any,
    meta: Any,
) -> dict[str, Any]:
    feature_name = _require_nonempty_str(feature, "feature")
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
        "feature": feature_name,
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
                    _compose_feature_name(base_name, key),
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
                    _compose_feature_name(base_name, index),
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
    if {"value", "values", "unit", "meta", "meta_json", "name", "feature"} & set(
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
            name = output.get("name") or output.get("feature") or base_name
            return [
                _build_row(
                    run_id,
                    str(name),
                    output.get("value"),
                    output.get("unit", ""),
                    output.get("meta", output.get("meta_json")),
                )
            ]
        raise ConfigError("Feature output is missing 'value' or 'values'.")
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


def _write_features_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
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
                ("feature", pa.string()),
                ("value", pa.float64()),
                ("unit", pa.string()),
                ("meta_json", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(table, path)
        return
    payload = {"columns": list(REQUIRED_COLUMNS), "rows": list(rows)}
    payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
    path.write_text(payload_text, encoding="utf-8")
    json_path = path.with_suffix(".json")
    if json_path != path:
        json_path.write_text(payload_text, encoding="utf-8")
    logger = logging.getLogger("rxn_platform.features")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s and %s.",
        path,
        json_path,
    )


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


def _transpose_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _extract_time_matrix(
    payload: Any,
    time_values: Optional[Sequence[float]],
    axis: str,
    axis_names: Sequence[str],
) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise ConfigError("data_vars entry must be a mapping.")
    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError("data_vars dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time", axis]:
        matrix = _coerce_matrix(data, axis)
        if time_values is not None and len(matrix) != len(time_values):
            raise ConfigError(
                f"{axis} rows mismatch: expected {len(time_values)}, got {len(matrix)}."
            )
        for row in matrix:
            if len(row) != len(axis_names):
                raise ConfigError(
                    f"{axis} columns mismatch: expected {len(axis_names)}, got {len(row)}."
                )
        return matrix
    if dims_list == [axis, "time"]:
        matrix = _coerce_matrix(data, axis)
        if len(matrix) != len(axis_names):
            raise ConfigError(
                f"{axis} rows mismatch: expected {len(axis_names)}, got {len(matrix)}."
            )
        if time_values is not None:
            for row in matrix:
                if len(row) != len(time_values):
                    raise ConfigError(
                        f"{axis} columns mismatch: expected {len(time_values)}, got {len(row)}."
                    )
        return _transpose_matrix(matrix)
    raise ConfigError(f"data_vars dims must be [time, {axis}] or [{axis}, time].")


def _extract_time_values(run_dataset: RunDatasetView) -> list[float]:
    return _coerce_float_sequence(
        _extract_coord_data(run_dataset, "time"),
        "coords.time",
    )


def _integral_unit(base_unit: str, time_unit: str) -> str:
    if base_unit and time_unit:
        return f"{base_unit}*{time_unit}"
    if base_unit:
        return base_unit
    if time_unit:
        return time_unit
    return ""


def _integrate(values: Sequence[float], time_values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(values)):
        dt = time_values[index] - time_values[index - 1]
        total += 0.5 * (values[index] + values[index - 1]) * dt
    return total


def _series_stats(
    values: Sequence[float],
    time_values: Optional[Sequence[float]],
) -> dict[str, float]:
    if not values:
        raise ConfigError("data_vars entries must contain at least one entry.")
    last = values[-1]
    mean = sum(values) / float(len(values))
    max_value = max(values)
    min_value = min(values)
    if time_values is None:
        integral = math.nan
    else:
        if len(values) != len(time_values):
            raise ConfigError("data_vars length must match time dimension.")
        integral = _integrate(values, time_values)
    return {
        "last": last,
        "mean": mean,
        "max": max_value,
        "min": min_value,
        "integral": integral,
    }


def _compute_stats_by_species(
    axis_names: Sequence[str],
    matrix: Sequence[Sequence[float]],
    time_values: Optional[Sequence[float]],
) -> dict[str, dict[str, float]]:
    stats_by_species: dict[str, dict[str, float]] = {}
    for index, name in enumerate(axis_names):
        series = [row[index] for row in matrix]
        stats_by_species[name] = _series_stats(series, time_values)
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


def _rank_value(
    stats: Mapping[str, float],
    rank_by: str,
    *,
    rank_abs: bool,
) -> float:
    value = stats.get(rank_by, math.nan)
    if isinstance(value, float) and math.isnan(value):
        return -math.inf
    value = float(value)
    if rank_abs and math.isfinite(value):
        return abs(value)
    return value


def _select_top_entities(
    stats_by_entity: Mapping[str, Mapping[str, float]],
    top_n: Optional[int],
    rank_by: str,
    *,
    rank_abs: bool,
) -> list[str]:
    ranked = sorted(
        stats_by_entity.items(),
        key=lambda item: (-_rank_value(item[1], rank_by, rank_abs=rank_abs), item[0]),
    )
    if top_n is None or top_n >= len(ranked):
        return [name for name, _ in ranked]
    return [name for name, _ in ranked[:top_n]]


def _compose_feature_label(base_name: str, stat: str, species: Optional[str]) -> str:
    if species is None:
        return f"{base_name}.{stat}"
    return f"{base_name}.{species}.{stat}"


def _nan_rows_for_stats(
    base_name: str,
    stats: Sequence[str],
    unit: str,
    integral_unit: str,
    meta: Mapping[str, Any],
    *,
    species_names: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = species_names or [None]
    for stat in stats:
        unit_name = integral_unit if stat == "integral" else unit
        for species in targets:
            meta_payload = dict(meta)
            meta_payload["stat"] = stat
            if species is not None:
                meta_payload["species"] = species
            rows.append(
                {
                    "feature": _compose_feature_label(base_name, stat, species),
                    "value": math.nan,
                    "unit": unit_name,
                    "meta": meta_payload,
                }
            )
    return rows


def _nan_rows_for_ranked(
    spec: RopWdotSpec,
    unit: str,
    integral_unit: str,
    meta: Mapping[str, Any],
    axis_names: Optional[Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets: list[Optional[str]]
    if axis_names:
        targets = list(axis_names)
        if spec.top_n is not None:
            targets = targets[: spec.top_n]
    else:
        targets = [None]
    for stat in spec.stats:
        unit_name = integral_unit if stat == "integral" else unit
        for target in targets:
            meta_payload = dict(meta)
            meta_payload["stat"] = stat
            if target is not None:
                meta_payload[spec.id_label] = target
            rows.append(
                {
                    "feature": _compose_feature_label(spec.base_name, stat, target),
                    "value": math.nan,
                    "unit": unit_name,
                    "meta": meta_payload,
                }
            )
    return rows


def _summarize_variable(
    spec: VariableSpec,
    run_dataset: RunDatasetView,
    time_values: Optional[Sequence[float]],
    time_error: Optional[str],
    units: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base_unit = "" if units.get(spec.name) is None else str(units.get(spec.name))
    time_unit = "" if units.get("time") is None else str(units.get("time"))
    integral_unit = _integral_unit(base_unit, time_unit)

    payload = run_dataset.data_vars.get(spec.name)
    if payload is None:
        axis_names = None
        if spec.axis is not None:
            try:
                axis_names = _coerce_str_sequence(
                    _extract_coord_data(run_dataset, spec.axis),
                    f"coords.{spec.axis}",
                )
            except ConfigError:
                axis_names = None
        if not axis_names and spec.species:
            axis_names = list(spec.species)
        if not axis_names and spec.axis is not None:
            axis_names = ["missing"]
        meta = {
            "status": "missing_variable",
            "missing": [f"data_vars.{spec.name}"],
        }
        if spec.axis is not None:
            meta["axis"] = spec.axis
        if time_error:
            meta["time_error"] = time_error
        return _nan_rows_for_stats(
            spec.base_name,
            spec.stats,
            base_unit,
            integral_unit,
            meta,
            species_names=axis_names,
        )

    if not isinstance(payload, Mapping):
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": "data_vars entry must be a mapping.",
        }
        return _nan_rows_for_stats(
            spec.base_name,
            spec.stats,
            base_unit,
            integral_unit,
            meta,
        )

    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": "data_vars dims must be a sequence.",
        }
        return _nan_rows_for_stats(
            spec.base_name,
            spec.stats,
            base_unit,
            integral_unit,
            meta,
        )
    dims_list = list(dims)

    if dims_list == ["time"]:
        try:
            series = _coerce_float_sequence(data, spec.name)
            stats = _series_stats(series, time_values)
        except ConfigError as exc:
            meta = {
                "status": "invalid_variable",
                "source": f"data_vars.{spec.name}",
                "error": str(exc),
            }
            return _nan_rows_for_stats(
                spec.base_name,
                spec.stats,
                base_unit,
                integral_unit,
                meta,
            )
        rows: list[dict[str, Any]] = []
        for stat in spec.stats:
            value = stats.get(stat, math.nan)
            unit = integral_unit if stat == "integral" else base_unit
            meta = {
                "status": "ok",
                "source": f"data_vars.{spec.name}",
                "stat": stat,
            }
            if time_values is None and stat == "integral":
                meta["status"] = "missing_time"
                if time_error:
                    meta["time_error"] = time_error
            rows.append(
                {
                    "feature": _compose_feature_label(spec.base_name, stat, None),
                    "value": value,
                    "unit": unit,
                    "meta": meta,
                }
            )
        return rows

    axis = spec.axis
    if axis is None:
        if "species" in dims_list:
            axis = "species"
        elif "surface_species" in dims_list:
            axis = "surface_species"
    if axis is None or axis not in dims_list or "time" not in dims_list:
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": "data_vars dims must include time with species or surface_species.",
        }
        return _nan_rows_for_stats(
            spec.base_name,
            spec.stats,
            base_unit,
            integral_unit,
            meta,
        )

    try:
        axis_names = _coerce_str_sequence(
            _extract_coord_data(run_dataset, axis),
            f"coords.{axis}",
        )
        matrix = _extract_time_matrix(payload, time_values, axis, axis_names)
        stats_by_species = _compute_stats_by_species(
            axis_names,
            matrix,
            time_values,
        )
        selected = _select_species(
            spec.species,
            spec.top_n,
            spec.rank_by,
            stats_by_species,
        )
    except ConfigError as exc:
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": str(exc),
            "axis": axis,
        }
        return _nan_rows_for_stats(
            spec.base_name,
            spec.stats,
            base_unit,
            integral_unit,
            meta,
        )

    rows: list[dict[str, Any]] = []
    for name in selected:
        stats = stats_by_species.get(name, {})
        for stat in spec.stats:
            value = stats.get(stat, math.nan)
            unit = integral_unit if stat == "integral" else base_unit
            meta = {
                "status": "ok",
                "source": f"data_vars.{spec.name}",
                "stat": stat,
                "species": name,
                "axis": axis,
            }
            if time_values is None and stat == "integral":
                meta["status"] = "missing_time"
                if time_error:
                    meta["time_error"] = time_error
            rows.append(
                {
                    "feature": _compose_feature_label(spec.base_name, stat, name),
                    "value": value,
                    "unit": unit,
                    "meta": meta,
                }
            )
    return rows


def _summarize_ranked_variable(
    spec: RopWdotSpec,
    run_dataset: RunDatasetView,
    time_values: Optional[Sequence[float]],
    time_error: Optional[str],
    units: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base_unit = "" if units.get(spec.name) is None else str(units.get(spec.name))
    time_unit = "" if units.get("time") is None else str(units.get("time"))
    integral_unit = _integral_unit(base_unit, time_unit)

    payload = run_dataset.data_vars.get(spec.name)
    axis_names: Optional[list[str]] = None
    axis_error: Optional[str] = None
    try:
        axis_names = _coerce_str_sequence(
            _extract_coord_data(run_dataset, spec.axis),
            f"coords.{spec.axis}",
        )
    except ConfigError as exc:
        axis_error = str(exc)
        axis_names = None

    if payload is None:
        meta = {
            "status": "missing_variable",
            "missing": [f"data_vars.{spec.name}"],
            "axis": spec.axis,
        }
        if axis_error:
            meta["axis_error"] = axis_error
        if time_error:
            meta["time_error"] = time_error
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    if not isinstance(payload, Mapping):
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": "data_vars entry must be a mapping.",
            "axis": spec.axis,
        }
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    dims = payload.get("dims")
    data = payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": "data_vars dims must be a sequence.",
            "axis": spec.axis,
        }
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    dims_list = list(dims)
    if spec.axis not in dims_list or "time" not in dims_list:
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": f"data_vars dims must include time and {spec.axis}.",
            "axis": spec.axis,
        }
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    if axis_names is None:
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": f"coords.{spec.axis} is missing.",
            "axis": spec.axis,
        }
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    try:
        matrix = _extract_time_matrix(payload, time_values, spec.axis, axis_names)
        stats_by_id = _compute_stats_by_species(axis_names, matrix, time_values)
    except ConfigError as exc:
        meta = {
            "status": "invalid_variable",
            "source": f"data_vars.{spec.name}",
            "error": str(exc),
            "axis": spec.axis,
        }
        return _nan_rows_for_ranked(
            spec,
            base_unit,
            integral_unit,
            meta,
            axis_names,
        )

    selected = _select_top_entities(
        stats_by_id,
        spec.top_n,
        spec.rank_by,
        rank_abs=spec.rank_abs,
    )
    rows: list[dict[str, Any]] = []
    for name in selected:
        stats = stats_by_id.get(name, {})
        for stat in spec.stats:
            value = stats.get(stat, math.nan)
            unit = integral_unit if stat == "integral" else base_unit
            meta = {
                "status": "ok",
                "source": f"data_vars.{spec.name}",
                "stat": stat,
                spec.id_label: name,
                "axis": spec.axis,
                "rank_by": spec.rank_by,
                "rank_abs": spec.rank_abs,
            }
            if time_values is None and stat == "integral":
                meta["status"] = "missing_time"
                if time_error:
                    meta["time_error"] = time_error
            rows.append(
                {
                    "feature": _compose_feature_label(spec.base_name, stat, name),
                    "value": value,
                    "unit": unit,
                    "meta": meta,
                }
            )
    return rows


def _normalize_feature_spec(entry: Mapping[str, Any], label: str) -> FeatureSpec:
    name = entry.get("name") or entry.get("feature")
    name = _require_nonempty_str(name, f"{label}.name")
    params = entry.get("params")
    if params is None:
        params = entry.get("config")
    if params is None:
        params = {key: value for key, value in entry.items() if key not in ("name", "feature")}
    if not isinstance(params, Mapping):
        raise ConfigError(f"{label}.params must be a mapping.")
    return FeatureSpec(name=name, params=dict(params))


def _normalize_feature_specs(raw: Any) -> list[FeatureSpec]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "name" in raw or "feature" in raw:
            return [_normalize_feature_spec(raw, "features")]
        specs: list[FeatureSpec] = []
        for key, value in raw.items():
            name = _require_nonempty_str(key, "features")
            if value is None:
                params = {}
            elif not isinstance(value, Mapping):
                raise ConfigError("features mapping values must be mappings.")
            else:
                params = dict(value)
            specs.append(FeatureSpec(name=name, params=params))
        return specs
    if isinstance(raw, str):
        return [FeatureSpec(name=_require_nonempty_str(raw, "features"), params={})]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        specs: list[FeatureSpec] = []
        for index, entry in enumerate(raw):
            if isinstance(entry, str):
                specs.append(
                    FeatureSpec(
                        name=_require_nonempty_str(entry, "features"),
                        params={},
                    )
                )
                continue
            if isinstance(entry, Mapping):
                specs.append(_normalize_feature_spec(entry, f"features[{index}]"))
                continue
            raise ConfigError("features entries must be strings or mappings.")
        return specs
    raise ConfigError("features must be a mapping, list, or string.")


def _normalize_network_metrics(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_NETWORK_METRICS)
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = list(value)
    else:
        raise ConfigError("metrics must be a string or list of strings.")

    aliases = {
        "degree": "degree",
        "in_degree": "in_degree",
        "out_degree": "out_degree",
        "degree_centrality": "degree_centrality",
        "in_degree_centrality": "in_degree_centrality",
        "out_degree_centrality": "out_degree_centrality",
        "betweenness": "betweenness_centrality",
        "betweenness_centrality": "betweenness_centrality",
    }
    metrics: list[str] = []
    for entry in raw:
        key = _require_nonempty_str(entry, "metrics").lower()
        metric = aliases.get(key)
        if metric is None:
            allowed = ", ".join(sorted(set(aliases.values())))
            raise ConfigError(f"metrics entries must be one of: {allowed}.")
        if metric not in metrics:
            metrics.append(metric)
    if not metrics:
        raise ConfigError("metrics must include at least one entry.")
    return metrics


def _extract_graph_id_from_params(params: Mapping[str, Any]) -> str:
    graph_id = params.get("graph_id")
    if graph_id is None:
        graph_section = params.get("graph")
        if isinstance(graph_section, Mapping):
            graph_id = graph_section.get("id") or graph_section.get("graph_id")
        else:
            graph_id = graph_section
    if graph_id is None:
        graph_id = params.get("id")
    return _require_nonempty_str(graph_id, "graph_id")


def _load_graph_payload(path: Path) -> dict[str, Any]:
    graph_path = path / "graph.json"
    if not graph_path.exists():
        raise ArtifactError(f"graph.json not found in {path}.")
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("graph.json must contain a JSON object.")
    return dict(payload)


def _extract_graph_payload(
    payload: Mapping[str, Any],
) -> tuple[Optional[dict[str, Any]], bool, Optional[dict[str, Any]]]:
    graph_data: Optional[dict[str, Any]] = None
    is_bipartite = False
    if "bipartite" in payload and isinstance(payload.get("bipartite"), Mapping):
        bipartite = payload.get("bipartite")
        data = bipartite.get("data") if isinstance(bipartite, Mapping) else None
        if isinstance(data, Mapping):
            graph_data = dict(data)
            is_bipartite = True
    if graph_data is None and "nodes" in payload and (
        "links" in payload or "edges" in payload
    ):
        graph_data = dict(payload)
        graph_meta = payload.get("graph")
        if isinstance(graph_meta, Mapping) and graph_meta.get("bipartite"):
            is_bipartite = True
    analysis = payload.get("analysis")
    if not isinstance(analysis, Mapping):
        analysis = None
    return graph_data, is_bipartite, analysis


def _coerce_optional_bool(value: Any, label: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{label} must be a boolean.")


def _node_id_from_entry(entry: Any) -> str:
    if isinstance(entry, Mapping):
        for key in ("id", "name", "key"):
            if key in entry and entry.get(key) is not None:
                return str(entry.get(key))
    if isinstance(entry, str):
        return entry
    if entry is None:
        raise ConfigError("graph node id must not be null.")
    return str(entry)


def _normalize_graph_nodes(
    nodes_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    node_map: dict[str, dict[str, Any]] = {}
    for entry in nodes_raw:
        node_id = _node_id_from_entry(entry)
        if isinstance(entry, Mapping):
            node = dict(entry)
        else:
            node = {}
        node["id"] = node_id
        node_map[node_id] = node
    return list(node_map.values()), node_map


def _coerce_node_ref(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("id") or value.get("name") or value.get("key")
    if value is None:
        return None
    return str(value)


def _normalize_graph_links(links_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(links_raw, Sequence) or isinstance(
        links_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph links must be a sequence.")
    links: list[dict[str, Any]] = []
    for entry in links_raw:
        if not isinstance(entry, Mapping):
            continue
        source = _coerce_node_ref(entry.get("source"))
        target = _coerce_node_ref(entry.get("target"))
        if source is None or target is None:
            continue
        link = dict(entry)
        link["source"] = source
        link["target"] = target
        links.append(link)
    return links


def _normalize_direction_mode(value: Any, *, is_bipartite: bool) -> str:
    if value is None:
        return "role" if is_bipartite else "as_is"
    if not isinstance(value, str):
        raise ConfigError("direction_mode must be a string.")
    normalized = value.strip().lower()
    if normalized in ("as_is", "asis"):
        return "as_is"
    if normalized in ("role", "bipartite_role", "stoich_role"):
        return "role"
    raise ConfigError("direction_mode must be one of: as_is, role.")


def _infer_link_role(link: Mapping[str, Any]) -> Optional[str]:
    role = link.get("role")
    if isinstance(role, str):
        role_lower = role.strip().lower()
        if role_lower in ("reactant", "product"):
            return role_lower
    stoich = link.get("stoich")
    if stoich is None:
        return None
    try:
        value = float(stoich)
    except (TypeError, ValueError):
        return None
    if value > 0:
        return "product"
    if value < 0:
        return "reactant"
    return None


def _apply_role_direction(
    source: str,
    target: str,
    *,
    link: Mapping[str, Any],
    node_kind: Mapping[str, str],
) -> tuple[str, str]:
    role = _infer_link_role(link)
    if role is None:
        return source, target
    if role == "reactant":
        desired_source = "species"
        desired_target = "reaction"
    else:
        desired_source = "reaction"
        desired_target = "species"
    source_kind = node_kind.get(source)
    target_kind = node_kind.get(target)
    if source_kind == desired_source and target_kind == desired_target:
        return source, target
    if source_kind == desired_target and target_kind == desired_source:
        return target, source
    if role == "product":
        return target, source
    return source, target


def _build_directed_edges(
    links: Sequence[Mapping[str, Any]],
    *,
    node_kind: Mapping[str, str],
    direction_mode: str,
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is None or target is None:
            continue
        if direction_mode == "role":
            source, target = _apply_role_direction(
                source,
                target,
                link=link,
                node_kind=node_kind,
            )
        edges.append((source, target))
    return edges


def _build_graph_adjacency(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    adjacency = {node: set() for node in nodes}
    reverse = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        reverse.setdefault(target, set()).add(source)
        adjacency.setdefault(target, set())
        reverse.setdefault(source, set())
    return adjacency, reverse


def _network_missing_rows(graph_id: str, reason: str) -> list[dict[str, Any]]:
    meta = {
        "status": "missing_graph",
        "reason": reason,
        "feature_kind": "network_metrics",
        "graph_id": graph_id,
    }
    return [
        {
            "feature": "network_metrics",
            "value": math.nan,
            "unit": "",
            "meta": meta,
        }
    ]


def _network_metric_rows(
    graph_payload: Mapping[str, Any],
    *,
    graph_id: str,
    metrics: Sequence[str],
    top_n: Optional[int],
    node_kinds: Sequence[str],
    direction_mode: Any,
    directed_override: Any,
    betweenness_max_nodes: Any,
) -> list[dict[str, Any]]:
    graph_data, is_bipartite, analysis = _extract_graph_payload(graph_payload)
    if graph_data is None and analysis is None:
        return _network_missing_rows(
            graph_id,
            "graph payload has no node-link data or analysis.",
        )

    if graph_data is None:
        return _network_metric_rows_from_analysis(
            analysis,
            graph_id=graph_id,
            metrics=metrics,
            top_n=top_n,
            node_kinds=node_kinds,
        )

    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    try:
        nodes, node_map = _normalize_graph_nodes(nodes_raw)
        links = _normalize_graph_links(links_raw)
    except ConfigError as exc:
        return _network_missing_rows(graph_id, str(exc))

    node_ids = [node["id"] for node in nodes]
    node_kind = {
        node_id: str(node.get("kind", "unknown")) for node_id, node in node_map.items()
    }
    direction_mode = _normalize_direction_mode(
        direction_mode, is_bipartite=is_bipartite
    )
    edges = _build_directed_edges(
        links,
        node_kind=node_kind,
        direction_mode=direction_mode,
    )

    directed = _coerce_optional_bool(directed_override, "directed")
    if directed is None:
        directed = bool(graph_data.get("directed", True))
    if not directed:
        undirected_edges: list[tuple[str, str]] = []
        for source, target in edges:
            undirected_edges.append((source, target))
            undirected_edges.append((target, source))
        edges = undirected_edges

    adjacency, reverse = _build_graph_adjacency(node_ids, edges)
    metric_values, metric_status = _compute_network_metric_values(
        metrics,
        node_ids,
        adjacency,
        reverse,
        directed=directed,
        edges=edges,
        betweenness_max_nodes=betweenness_max_nodes,
    )

    if node_kinds:
        allowed = set(node_kinds)
        metric_values = {
            metric: {
                node_id: value
                for node_id, value in values.items()
                if node_kind.get(node_id, "unknown") in allowed
            }
            for metric, values in metric_values.items()
        }

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = metric_values.get(metric, {})
        status = metric_status.get(metric, "missing_metric")
        selected = _select_top_nodes(values, top_n)
        if not selected:
            meta = {
                "status": status,
                "feature_kind": "network_metrics",
                "metric": metric,
                "graph_id": graph_id,
                "directed": directed,
                "direction_mode": direction_mode,
                "top_n": top_n,
                "source": "graph",
            }
            rows.append(
                {
                    "feature": f"network.{metric}",
                    "value": math.nan,
                    "unit": "",
                    "meta": meta,
                }
            )
            continue
        for node_id in selected:
            meta = {
                "status": status,
                "feature_kind": "network_metrics",
                "metric": metric,
                "node_id": node_id,
                "node_kind": node_kind.get(node_id, "unknown"),
                "graph_id": graph_id,
                "directed": directed,
                "direction_mode": direction_mode,
                "top_n": top_n,
                "source": "graph",
            }
            rows.append(
                {
                    "feature": f"network.{metric}.{node_id}",
                    "value": values.get(node_id, math.nan),
                    "unit": "",
                    "meta": meta,
                }
            )
    return rows


def _network_metric_rows_from_analysis(
    analysis: Optional[Mapping[str, Any]],
    *,
    graph_id: str,
    metrics: Sequence[str],
    top_n: Optional[int],
    node_kinds: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(analysis, Mapping):
        return _network_missing_rows(graph_id, "graph analysis payload is missing.")

    centrality = analysis.get("centrality")
    if not isinstance(centrality, Mapping):
        return _network_missing_rows(graph_id, "analysis.centrality is missing.")

    summary = analysis.get("summary")
    node_count: Optional[int] = None
    if isinstance(summary, Mapping):
        node_count = summary.get("node_count")
        if isinstance(node_count, bool) or not isinstance(node_count, int):
            node_count = None

    denom = float(node_count - 1) if node_count and node_count > 1 else 0.0
    node_kind_map: dict[str, str] = {}
    metric_values = {metric: {} for metric in metrics}
    metric_status = {metric: "missing_metric" for metric in metrics}

    degree_info = centrality.get("degree")
    if isinstance(degree_info, Mapping):
        ranking = degree_info.get("ranking")
        if isinstance(ranking, Sequence):
            for entry in ranking:
                if not isinstance(entry, Mapping):
                    continue
                node_id = entry.get("node_id")
                if node_id is None:
                    continue
                node_id = str(node_id)
                node_kind_map[node_id] = str(entry.get("kind", "unknown"))
                if "degree" in metric_values:
                    metric_values["degree"][node_id] = entry.get("degree")
                    metric_status["degree"] = "ok"
                if "in_degree" in metric_values:
                    metric_values["in_degree"][node_id] = entry.get("in_degree")
                    metric_status["in_degree"] = "ok"
                if "out_degree" in metric_values:
                    metric_values["out_degree"][node_id] = entry.get("out_degree")
                    metric_status["out_degree"] = "ok"
                if "degree_centrality" in metric_values:
                    metric_values["degree_centrality"][node_id] = entry.get("score")
                    metric_status["degree_centrality"] = "ok"
                if "in_degree_centrality" in metric_values:
                    in_value = entry.get("in_degree")
                    metric_values["in_degree_centrality"][node_id] = (
                        float(in_value) / denom if denom and in_value is not None else math.nan
                    )
                    metric_status["in_degree_centrality"] = "ok"
                if "out_degree_centrality" in metric_values:
                    out_value = entry.get("out_degree")
                    metric_values["out_degree_centrality"][node_id] = (
                        float(out_value) / denom if denom and out_value is not None else math.nan
                    )
                    metric_status["out_degree_centrality"] = "ok"

    betweenness_info = centrality.get("betweenness")
    if isinstance(betweenness_info, Mapping):
        ranking = betweenness_info.get("ranking")
        if isinstance(ranking, Sequence):
            for entry in ranking:
                if not isinstance(entry, Mapping):
                    continue
                node_id = entry.get("node_id")
                if node_id is None:
                    continue
                node_id = str(node_id)
                node_kind_map[node_id] = str(entry.get("kind", "unknown"))
                if "betweenness_centrality" in metric_values:
                    metric_values["betweenness_centrality"][node_id] = entry.get("score")
                    metric_status["betweenness_centrality"] = "ok"

    if node_kinds:
        allowed = set(node_kinds)
        metric_values = {
            metric: {
                node_id: value
                for node_id, value in values.items()
                if node_kind_map.get(node_id, "unknown") in allowed
            }
            for metric, values in metric_values.items()
        }

    rows: list[dict[str, Any]] = []
    directed = True
    summary = analysis.get("summary")
    if isinstance(summary, Mapping):
        directed = bool(summary.get("directed", True))
    for metric in metrics:
        values = metric_values.get(metric, {})
        status = metric_status.get(metric, "missing_metric")
        selected = _select_top_nodes(values, top_n)
        if not selected:
            meta = {
                "status": status,
                "feature_kind": "network_metrics",
                "metric": metric,
                "graph_id": graph_id,
                "directed": directed,
                "top_n": top_n,
                "source": "analysis",
            }
            rows.append(
                {
                    "feature": f"network.{metric}",
                    "value": math.nan,
                    "unit": "",
                    "meta": meta,
                }
            )
            continue
        for node_id in selected:
            meta = {
                "status": status,
                "feature_kind": "network_metrics",
                "metric": metric,
                "node_id": node_id,
                "node_kind": node_kind_map.get(node_id, "unknown"),
                "graph_id": graph_id,
                "directed": directed,
                "top_n": top_n,
                "source": "analysis",
            }
            rows.append(
                {
                    "feature": f"network.{metric}.{node_id}",
                    "value": values.get(node_id, math.nan),
                    "unit": "",
                    "meta": meta,
                }
            )
    return rows


def _safe_metric_value(value: Any) -> float:
    if value is None:
        return -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    if math.isnan(number):
        return -math.inf
    return number


def _select_top_nodes(values: Mapping[str, Any], top_n: Optional[int]) -> list[str]:
    ranked = sorted(
        values.items(),
        key=lambda item: (-_safe_metric_value(item[1]), item[0]),
    )
    if top_n is None or top_n >= len(ranked):
        return [node_id for node_id, _ in ranked]
    return [node_id for node_id, _ in ranked[:top_n]]


def _compute_network_metric_values(
    metrics: Sequence[str],
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
    *,
    directed: bool,
    edges: Sequence[tuple[str, str]],
    betweenness_max_nodes: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    metric_values: dict[str, dict[str, float]] = {metric: {} for metric in metrics}
    metric_status: dict[str, str] = {metric: "ok" for metric in metrics}

    node_count = len(nodes)
    denom = float(node_count - 1) if node_count > 1 else 0.0
    for node_id in nodes:
        if directed:
            in_degree = len(reverse.get(node_id, set()))
            out_degree = len(adjacency.get(node_id, set()))
            degree = in_degree + out_degree
        else:
            degree = len(adjacency.get(node_id, set()))
            in_degree = degree
            out_degree = degree

        if "degree" in metric_values:
            metric_values["degree"][node_id] = float(degree)
        if "in_degree" in metric_values:
            metric_values["in_degree"][node_id] = float(in_degree)
        if "out_degree" in metric_values:
            metric_values["out_degree"][node_id] = float(out_degree)
        if "degree_centrality" in metric_values:
            metric_values["degree_centrality"][node_id] = (
                float(degree) / denom if denom else 0.0
            )
        if "in_degree_centrality" in metric_values:
            metric_values["in_degree_centrality"][node_id] = (
                float(in_degree) / denom if denom else 0.0
            )
        if "out_degree_centrality" in metric_values:
            metric_values["out_degree_centrality"][node_id] = (
                float(out_degree) / denom if denom else 0.0
            )

    if "betweenness_centrality" in metric_values:
        if nx is None:
            metric_status["betweenness_centrality"] = "missing_dependency"
            for node_id in nodes:
                metric_values["betweenness_centrality"][node_id] = math.nan
        else:
            max_nodes = _coerce_optional_int(
                betweenness_max_nodes,
                "betweenness_max_nodes",
            )
            if max_nodes is not None and node_count > max_nodes:
                metric_status["betweenness_centrality"] = "node_limit"
                for node_id in nodes:
                    metric_values["betweenness_centrality"][node_id] = math.nan
            else:
                graph = nx.DiGraph() if directed else nx.Graph()
                graph.add_nodes_from(nodes)
                graph.add_edges_from(edges)
                scores = nx.betweenness_centrality(graph)
                for node_id in nodes:
                    metric_values["betweenness_centrality"][node_id] = float(
                        scores.get(node_id, 0.0)
                    )
    return metric_values, metric_status


def _filter_finite(values: Mapping[str, Any]) -> dict[str, float]:
    filtered: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            filtered[key] = number
    return filtered


def _rank_nodes(values: Mapping[str, float]) -> dict[str, float]:
    items = sorted(
        values.items(),
        key=lambda item: (-item[1], item[0]),
    )
    ranks: dict[str, float] = {}
    index = 0
    count = len(items)
    while index < count:
        value = items[index][1]
        start = index
        while index < count and items[index][1] == value:
            index += 1
        end = index
        rank_value = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[items[pos][0]] = rank_value
    return ranks


def _pearson_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return math.nan
    mean_a = sum(values_a) / float(len(values_a))
    mean_b = sum(values_b) / float(len(values_b))
    var_a = 0.0
    var_b = 0.0
    cov = 0.0
    for a, b in zip(values_a, values_b):
        da = a - mean_a
        db = b - mean_b
        var_a += da * da
        var_b += db * db
        cov += da * db
    if var_a <= 0.0 or var_b <= 0.0:
        return math.nan
    return cov / math.sqrt(var_a * var_b)


def _mean_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / float(len(values))


def _min_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return min(values)


def _max_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return max(values)


def _default_top_k(values_by_run: Mapping[str, Mapping[str, float]]) -> Optional[int]:
    min_nodes = min((len(values) for values in values_by_run.values()), default=0)
    if min_nodes <= 0:
        return None
    return min(DEFAULT_NETWORK_STABILITY_TOP_N, min_nodes)


def _compute_rank_stability(
    values_by_run: Mapping[str, Mapping[str, Any]],
    *,
    top_n: Optional[int],
) -> dict[str, Any]:
    run_ids = sorted(values_by_run.keys())
    stability: dict[str, Any] = {
        "run_count": len(run_ids),
        "pair_count": 0,
    }
    if len(run_ids) < 2:
        stability["status"] = "insufficient_runs"
        return stability

    filtered_values = {rid: _filter_finite(values) for rid, values in values_by_run.items()}
    ranks_by_run = {rid: _rank_nodes(values) for rid, values in filtered_values.items()}

    spearman_values: list[float] = []
    for run_a, run_b in combinations(run_ids, 2):
        common = set(ranks_by_run[run_a]).intersection(ranks_by_run[run_b])
        if len(common) < 2:
            continue
        scores_a = [ranks_by_run[run_a][node] for node in common]
        scores_b = [ranks_by_run[run_b][node] for node in common]
        corr = _pearson_corr(scores_a, scores_b)
        if math.isfinite(corr):
            spearman_values.append(corr)
    stability["pair_count"] = len(run_ids) * (len(run_ids) - 1) // 2
    stability["spearman_mean"] = _mean_value(spearman_values)
    stability["spearman_min"] = _min_value(spearman_values)
    stability["spearman_max"] = _max_value(spearman_values)

    top_k = top_n if top_n is not None else _default_top_k(filtered_values)
    stability["top_k"] = top_k
    jaccard_values: list[float] = []
    if top_k is not None and top_k > 0:
        top_sets: dict[str, set[str]] = {}
        for run_id, values in filtered_values.items():
            top_nodes = _select_top_nodes(values, top_k)
            top_sets[run_id] = set(top_nodes)
        for run_a, run_b in combinations(run_ids, 2):
            union = top_sets[run_a].union(top_sets[run_b])
            if not union:
                continue
            intersect = top_sets[run_a].intersection(top_sets[run_b])
            jaccard_values.append(len(intersect) / float(len(union)))
    stability["top_k_jaccard_mean"] = _mean_value(jaccard_values)
    stability["top_k_jaccard_min"] = _min_value(jaccard_values)
    stability["top_k_jaccard_max"] = _max_value(jaccard_values)

    stability["status"] = "computed" if spearman_values or jaccard_values else "insufficient_overlap"
    return stability


def _apply_network_stability(rows: list[dict[str, Any]], run_ids: Sequence[str]) -> None:
    if len(run_ids) < 2:
        return
    metric_values: dict[str, dict[str, dict[str, Any]]] = {}
    metric_rows: dict[str, list[int]] = {}
    metric_top_n: dict[str, int] = {}
    for index, row in enumerate(rows):
        meta_json = row.get("meta_json")
        if not isinstance(meta_json, str):
            continue
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError:
            continue
        if meta.get("feature_kind") != "network_metrics":
            continue
        metric = meta.get("metric")
        node_id = meta.get("node_id")
        if not metric or not node_id:
            continue
        run_id = row.get("run_id")
        if run_id not in run_ids:
            continue
        metric_values.setdefault(metric, {}).setdefault(run_id, {})[node_id] = row.get("value")
        metric_rows.setdefault(metric, []).append(index)
        top_n = meta.get("top_n")
        if isinstance(top_n, int) and top_n > 0 and metric not in metric_top_n:
            metric_top_n[metric] = top_n

    for metric, values_by_run in metric_values.items():
        stability = _compute_rank_stability(
            values_by_run,
            top_n=metric_top_n.get(metric),
        )
        for index in metric_rows.get(metric, []):
            meta_json = rows[index].get("meta_json")
            if not isinstance(meta_json, str):
                continue
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                continue
            meta["rank_stability"] = stability
            rows[index]["meta_json"] = json.dumps(
                meta,
                ensure_ascii=True,
                sort_keys=True,
            )


def _prepare_network_params(
    params: Mapping[str, Any],
    *,
    store: ArtifactStore,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = params.get("graph_payload")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ConfigError("graph_payload must be a mapping.")
        graph_id = _extract_graph_id_from_params(params)
        prepared = dict(params)
        prepared["graph_id"] = graph_id
        prepared["graph_payload"] = dict(payload)
        return prepared

    graph_id = _extract_graph_id_from_params(params)
    payload = cache.get(graph_id)
    if payload is None:
        store.read_manifest("graphs", graph_id)
        graph_dir = store.artifact_dir("graphs", graph_id)
        payload = _load_graph_payload(graph_dir)
        cache[graph_id] = payload
    prepared = dict(params)
    prepared["graph_id"] = graph_id
    prepared["graph_payload"] = payload
    return prepared


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Compute features for one or more run artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = _extract_params(feat_cfg)
    run_ids = _extract_run_ids(feat_cfg)

    features_raw = params.get("features", feat_cfg.get("features"))
    specs = _normalize_feature_specs(features_raw)
    if not specs:
        raise ConfigError("features list must not be empty.")
    missing_strategy = _normalize_missing_strategy(
        params.get("missing_strategy", feat_cfg.get("missing_strategy"))
    )

    rows: list[dict[str, Any]] = []
    graph_payload_cache: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        run_dataset = _load_run_dataset_view(run_dir)
        for spec in specs:
            feature_params = dict(spec.params)
            if spec.name == "network_metrics":
                feature_params = _prepare_network_params(
                    feature_params,
                    store=store,
                    cache=graph_payload_cache,
                )
            feat = _resolve_feature(spec.name, registry=registry)
            requires, requires_coords, requires_attrs = _feature_requirements(feat)
            missing = _missing_inputs(
                run_dataset,
                requires=requires,
                requires_coords=requires_coords,
                requires_attrs=requires_attrs,
            )
            if missing:
                status = "skipped" if missing_strategy == "skip" else "missing_input"
                meta = {"status": status, "missing": missing}
                rows.append(_build_row(run_id, spec.name, math.nan, "", meta))
                continue
            output = _call_feature(feat, run_dataset, feature_params)
            rows.extend(_normalize_output(run_id, spec.name, output))

    _apply_network_stability(rows, run_ids)

    inputs_payload = {"runs": run_ids, "features": [spec.name for spec in specs]}
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="features",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=run_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


register("task", "features.run", run)
register("task", "features.compute", run)
register("feature", "timeseries_summary", TimeseriesSummaryFeature())
register("feature", "rop_wdot_summary", RopWdotFeature())
register("feature", "network_metrics", NetworkMetricFeature())

__all__ = [
    "FeatureExtractor",
    "FeatureSpec",
    "RunDatasetView",
    "TimeseriesSummaryFeature",
    "RopWdotFeature",
    "NetworkMetricFeature",
    "run",
]
