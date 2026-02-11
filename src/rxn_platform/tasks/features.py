"""Feature extraction framework and time-series summary implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional
from rxn_platform.core import make_artifact_id, normalize_reaction_multipliers, resolve_repo_path
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.mechanism import read_yaml_payload
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
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

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
DEFAULT_STABLE_STATES_PATH = Path("reduction/learnck/stable_states.yaml")
DEFAULT_STABLE_STATS = ("last",)
DEFAULT_STABLE_SITE_TOL = 1.0e-6


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


def _extract_run_ids(
    feat_cfg: Mapping[str, Any],
    *,
    store: Optional[ArtifactStore] = None,
) -> list[str]:
    inputs = feat_cfg.get("inputs")
    run_set_id: Any = None
    run_ids: Any = None
    if inputs is None:
        run_ids = None
    elif not isinstance(inputs, Mapping):
        raise ConfigError("features.inputs must be a mapping.")
    else:
        if "run_set_id" in inputs:
            run_set_id = inputs.get("run_set_id")
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in inputs:
                run_ids = inputs.get(key)
                break
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is None and "run_set_id" in feat_cfg:
        run_set_id = feat_cfg.get("run_set_id")
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is not None:
        run_set_id = _require_nonempty_str(run_set_id, "run_set_id")
        if store is None:
            raise ConfigError("run_set_id requires a store to be provided.")
        return load_run_ids_from_run_set(store, run_set_id)
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


def _coerce_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number.") from exc


def _load_stable_states_payload(
    value: Any,
) -> tuple[dict[str, Any], Optional[str]]:
    if value is None:
        path = resolve_repo_path(DEFAULT_STABLE_STATES_PATH)
        payload = read_yaml_payload(path)
        if not isinstance(payload, Mapping):
            raise ConfigError("stable_states config must be a mapping.")
        return dict(payload), str(path)
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        path = resolve_repo_path(value)
        if not path.exists():
            raise ConfigError(f"stable_states file not found: {value}")
        payload = read_yaml_payload(path)
        if not isinstance(payload, Mapping):
            raise ConfigError("stable_states file must contain a mapping.")
        return dict(payload), str(path)
    raise ConfigError("stable_states must be a mapping or path to YAML/JSON.")


def _normalize_phase(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("stable_states.phase must be a string when provided.")
    text = value.strip().lower()
    if not text:
        return None
    if "surf" in text or "surface" in text:
        return "surface"
    if "gas" in text:
        return "gas"
    return text


def _normalize_stable_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries_raw: Any = None
    for key in ("stable_species", "stable_states", "states", "species"):
        if key in payload:
            entries_raw = payload.get(key)
            break
    if entries_raw is None:
        raise ConfigError("stable_states must define stable_species.")
    if isinstance(entries_raw, str):
        entries_raw = [entries_raw]
    if not isinstance(entries_raw, Sequence) or isinstance(
        entries_raw,
        (bytes, bytearray),
    ):
        raise ConfigError("stable_species must be a list of entries.")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries_raw):
        if isinstance(entry, str):
            name = _require_nonempty_str(entry, f"stable_species[{index}]")
            entries.append({"name": name, "phase": None})
            continue
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("species") or entry.get("id")
            name = _require_nonempty_str(name, f"stable_species[{index}].name")
            phase = _normalize_phase(entry.get("phase") or entry.get("kind"))
            entries.append({"name": name, "phase": phase})
            continue
        raise ConfigError("stable_species entries must be strings or mappings.")
    return entries


def _stable_site_total(
    payload: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[float]:
    for key in ("site_total", "sites_total", "total_sites", "coverage_total"):
        if key in params:
            return _coerce_optional_float(params.get(key), key)
    for key in ("site_total", "sites_total", "total_sites"):
        if key in payload:
            return _coerce_optional_float(payload.get(key), key)
    return None


def _extract_series(
    dataset: Any,
    *,
    var_name: str,
    coord_name: str,
    species_name: str,
) -> Optional[Any]:
    if var_name not in dataset.data_vars:
        raise ConfigError(f"Run dataset missing variable: {var_name}")
    data = dataset[var_name]
    if coord_name not in data.dims:
        raise ConfigError(f"{var_name} is missing coord {coord_name}.")
    try:
        series = data.sel({coord_name: species_name})
    except KeyError:
        return None
    if "time" not in series.dims:
        raise ConfigError(f"{var_name} must include time dimension.")
    return series.transpose("time").values


def _compute_stat(
    series: Any,
    time_values: Optional[Any],
    stat: str,
) -> float:
    if np is None:
        raise ConfigError("numpy is required to compute stable projection stats.")
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return math.nan
    if stat == "last":
        return float(values[-1])
    if stat == "mean":
        return float(np.nanmean(values))
    if stat == "max":
        return float(np.nanmax(values))
    if stat == "min":
        return float(np.nanmin(values))
    if stat == "integral":
        if time_values is None:
            raise ConfigError("integral stat requires time coordinates.")
        return float(np.trapz(values, x=np.asarray(time_values, dtype=float)))
    raise ConfigError(f"Unsupported stat: {stat}")


def _load_run_dataset_view(run_dir: Path) -> RunDatasetView:
    payload = load_run_dataset_payload(run_dir)
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


def _load_gnn_dataset_payload(
    store: ArtifactStore,
    dataset_id: str,
) -> tuple[dict[str, Any], Path]:
    store.read_manifest("gnn_datasets", dataset_id)
    dataset_dir = store.artifact_dir("gnn_datasets", dataset_id)
    payload_path = dataset_dir / "dataset.json"
    if not payload_path.exists():
        raise ConfigError(f"gnn dataset metadata not found: {payload_path}")
    try:
        payload = read_json(payload_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"gnn dataset dataset.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("gnn dataset dataset.json must be a JSON object.")
    return dict(payload), dataset_dir


def _load_gnn_dataset_items(
    payload: Mapping[str, Any],
    *,
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        files = {}
    dataset_root = payload.get("dataset_root")
    if isinstance(dataset_root, str) and dataset_root:
        data_root = Path(dataset_root)
    else:
        data_root = dataset_dir
    data_json = files.get("data_json") or "data.json"
    data_pt = files.get("data_pt") or "data.pt"

    data_path = data_root / str(data_json)
    if data_path.exists():
        try:
            data_payload = read_json(data_path)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"gnn dataset data.json is not valid JSON: {exc}") from exc
        if not isinstance(data_payload, Mapping):
            raise ConfigError("gnn dataset data.json must be a JSON object.")
        items = data_payload.get("items")
        if not isinstance(items, Sequence) or isinstance(
            items, (str, bytes, bytearray)
        ):
            raise ConfigError("gnn dataset items must be a list.")
        return [dict(item) for item in items if isinstance(item, Mapping)]

    data_path = data_root / str(data_pt)
    if data_path.exists():
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "torch is required to load pyg datasets; "
                "rerun gnn_dataset with json output or install rxn-platform[gnn]."
            ) from exc
        try:
            data_payload = torch.load(data_path, weights_only=False)
        except TypeError:
            data_payload = torch.load(data_path)
        if not isinstance(data_payload, Mapping):
            raise ConfigError("gnn dataset data.pt must contain a mapping.")
        raw_items = data_payload.get("data_list")
        if not isinstance(raw_items, Sequence):
            raise ConfigError("gnn dataset data.pt missing data_list.")
        items: list[dict[str, Any]] = []
        for data in raw_items:
            window_id = getattr(data, "window_id", None)
            features = getattr(data, "x", None)
            edge_index = getattr(data, "edge_index", None)
            edge_attr = getattr(data, "edge_attr", None)
            if window_id is None or features is None:
                continue
            try:
                features_np = features.detach().cpu().numpy().tolist()
                edge_index_np = (
                    edge_index.detach().cpu().numpy().tolist()
                    if edge_index is not None
                    else [[], []]
                )
                edge_attr_np = (
                    edge_attr.detach().cpu().numpy().tolist()
                    if edge_attr is not None
                    else []
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ConfigError("gnn dataset features could not be converted.") from exc
            if isinstance(edge_attr_np, Sequence) and edge_attr_np:
                if isinstance(edge_attr_np[0], Sequence):
                    edge_attr_np = [row[0] for row in edge_attr_np]
            items.append(
                {
                    "window_id": int(window_id),
                    "x": features_np,
                    "edge_index": edge_index_np,
                    "edge_attr": edge_attr_np,
                    "window": {
                        "start_time": getattr(data, "window_start", None),
                        "end_time": getattr(data, "window_end", None),
                        "start_idx": getattr(data, "window_index", None),
                    },
                    "run_id": getattr(data, "run_id", None),
                    "case_id": getattr(data, "case_id", None),
                }
            )
        return items

    raise ConfigError(
        "gnn dataset data.json not found; rerun gnn_dataset with json output."
    )


def _load_run_dataset_xr(run_dir: Path) -> Any:
    if xr is None:
        raise ArtifactError("xarray is required to load timeseries.zarr.")
    dataset_dir = resolve_run_dataset_dir(run_dir)
    if dataset_dir is None:
        raise ArtifactError(
            "Run dataset not found; expected sim/timeseries.zarr or state.zarr."
        )
    json_path = dataset_dir / "dataset.json"
    if json_path.exists():
        try:
            payload = read_json(json_path)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Run dataset JSON is invalid: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("Run dataset JSON must be a mapping.")
        coords_payload = payload.get("coords", {})
        vars_payload = payload.get("data_vars", {})
        attrs_payload = payload.get("attrs", {})
        if not isinstance(coords_payload, Mapping) or not isinstance(
            vars_payload, Mapping
        ):
            raise ArtifactError("Run dataset JSON is missing coords/data_vars.")
        coords: dict[str, Any] = {}
        for name, entry in coords_payload.items():
            if not isinstance(entry, Mapping):
                continue
            dims = entry.get("dims") or [name]
            data = entry.get("data")
            coords[name] = (list(dims), data)
        data_vars: dict[str, Any] = {}
        for name, entry in vars_payload.items():
            if not isinstance(entry, Mapping):
                continue
            dims = entry.get("dims")
            data = entry.get("data")
            if dims is None:
                continue
            data_vars[name] = (list(dims), data)
        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs_payload)
        return dataset
    return xr.open_zarr(dataset_dir)


def _extract_mapping_id(feat_cfg: Mapping[str, Any]) -> str:
    inputs = feat_cfg.get("inputs")
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("features.inputs must be a mapping.")
    for key in ("mapping_id", "reduction_id", "mapping"):
        if key in inputs:
            return _require_nonempty_str(inputs.get(key), "mapping_id")
    for key in ("mapping_id", "reduction_id", "mapping"):
        if key in feat_cfg:
            return _require_nonempty_str(feat_cfg.get(key), "mapping_id")
    raise ConfigError("mapping_id is required for superstate timeseries.")


def _load_mapping_payload(store: ArtifactStore, mapping_id: str) -> dict[str, Any]:
    store.read_manifest("reduction", mapping_id)
    mapping_path = store.artifact_dir("reduction", mapping_id) / "mapping.json"
    if not mapping_path.exists():
        raise ConfigError(f"mapping.json not found for reduction/{mapping_id}")
    try:
        payload = read_json(mapping_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"mapping.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("mapping.json must contain a JSON object.")
    return dict(payload)


def _load_superstate_payload(store: ArtifactStore, mapping_id: str) -> dict[str, Any]:
    """Load a superstate mapping payload from a reduction artifact.

    Supported files:
    - mapping.json (CNR/GNN mapping output)
    - node_lumping.json (stoichiometric node lumping output)
    """
    store.read_manifest("reduction", mapping_id)
    base_dir = store.artifact_dir("reduction", mapping_id)
    mapping_path = base_dir / "mapping.json"
    node_lumping_path = base_dir / "node_lumping.json"
    payload_path = None
    if mapping_path.exists():
        payload_path = mapping_path
    elif node_lumping_path.exists():
        payload_path = node_lumping_path
    else:
        raise ConfigError(
            f"Expected mapping.json or node_lumping.json for reduction/{mapping_id}."
        )
    try:
        payload = read_json(payload_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{payload_path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{payload_path.name} must contain a JSON object.")
    return dict(payload)


def _resolve_superstate_mapping_any(
    payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, int], dict[int, list[str]]]:
    """Resolve superstate mapping from either mapping.json or node_lumping.json."""
    mapping = payload.get("mapping")
    if not isinstance(mapping, Sequence) or isinstance(mapping, (str, bytes, bytearray)):
        raise ConfigError("superstate mapping payload must include a mapping list.")
    first = None
    for entry in mapping:
        if isinstance(entry, Mapping):
            first = entry
            break
    if first is None:
        raise ConfigError("superstate mapping payload has no mapping entries.")
    if "superstate_id" in first:
        return _resolve_superstate_mapping(payload)
    if "cluster_id" in first:
        clusters = payload.get("clusters")
        if not isinstance(clusters, Sequence) or isinstance(
            clusters, (str, bytes, bytearray)
        ):
            raise ConfigError("node_lumping clusters must be a list.")
        name_by_id: dict[int, str] = {}
        members_by_id: dict[int, list[str]] = {}
        for entry in clusters:
            if not isinstance(entry, Mapping):
                continue
            cid = entry.get("cluster_id")
            rep = entry.get("representative")
            if not isinstance(cid, int):
                continue
            if isinstance(rep, str) and rep.strip():
                name_by_id[int(cid)] = rep.strip()
            members = entry.get("members") or []
            if isinstance(members, Sequence) and not isinstance(
                members, (str, bytes, bytearray)
            ):
                members_by_id[int(cid)] = [str(item) for item in members]
            else:
                members_by_id[int(cid)] = []
        mapping_by_name: dict[str, int] = {}
        for entry in mapping:
            if not isinstance(entry, Mapping):
                continue
            species = entry.get("species")
            cid = entry.get("cluster_id")
            if isinstance(species, str) and isinstance(cid, int):
                mapping_by_name[species] = int(cid)
        if not mapping_by_name:
            raise ConfigError("node_lumping has no species mapping.")
        max_id = max(mapping_by_name.values())
        superstate_names: list[str] = []
        for idx in range(max_id + 1):
            name = name_by_id.get(idx) or f"S{idx:03d}"
            superstate_names.append(name)
        return superstate_names, mapping_by_name, members_by_id
    raise ConfigError("Unsupported mapping entries: expected superstate_id or cluster_id.")


def _resolve_superstate_mapping(
    payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, int], dict[int, list[str]]]:
    clusters = payload.get("clusters")
    mapping = payload.get("mapping")
    if not isinstance(mapping, Sequence) or isinstance(
        mapping, (str, bytes, bytearray)
    ):
        raise ConfigError("mapping.json mapping entries must be a list.")
    # "clusters" is optional for some mapping producers (e.g., node_lumping emits only
    # the per-species mapping). When missing, we synthesize members_by_id from mapping.
    if clusters is None:
        clusters_seq: Sequence[Any] = []
    elif not isinstance(clusters, Sequence) or isinstance(
        clusters, (str, bytes, bytearray)
    ):
        raise ConfigError("mapping.json clusters must be a list when provided.")
    else:
        clusters_seq = clusters
    name_by_id: dict[int, str] = {}
    members_by_id: dict[int, list[str]] = {}
    for entry in clusters_seq:
        if not isinstance(entry, Mapping):
            continue
        sid = entry.get("superstate_id")
        name = entry.get("name")
        if isinstance(sid, int) and isinstance(name, str) and name.strip():
            name_by_id[int(sid)] = name
            members = entry.get("members")
            if isinstance(members, Sequence) and not isinstance(
                members, (str, bytes, bytearray)
            ):
                members_by_id[int(sid)] = [str(item) for item in members]
            else:
                members_by_id[int(sid)] = []
    mapping_by_name: dict[str, int] = {}
    for entry in mapping:
        if not isinstance(entry, Mapping):
            continue
        species = entry.get("species")
        super_id = entry.get("superstate_id")
        if isinstance(species, str) and isinstance(super_id, int):
            mapping_by_name[species] = int(super_id)
    if not mapping_by_name:
        raise ConfigError("mapping.json has no species mapping.")
    # Fill members_by_id using mapping entries, but de-duplicate any overlap with the
    # optional clusters[*].members lists (some producers emit both).
    for species, sid in mapping_by_name.items():
        members_by_id.setdefault(int(sid), []).append(species)
    for sid, members in list(members_by_id.items()):
        seen: set[str] = set()
        unique: list[str] = []
        for item in members:
            name = str(item)
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
        members_by_id[int(sid)] = unique
    max_id = max(mapping_by_name.values())
    superstate_names: list[str] = []
    for idx in range(max_id + 1):
        name = name_by_id.get(idx)
        if not name:
            name = f"S{idx:03d}"
        superstate_names.append(name)
    return superstate_names, mapping_by_name, members_by_id


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
    write_json_atomic(path, payload)
    json_path = path.with_suffix(".json")
    if json_path != path:
        write_json_atomic(json_path, payload)
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
    index_map: dict[str, int] = {}
    if spec.axis == "reaction":
        index_map = {name: idx for idx, name in enumerate(axis_names)}
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
            if spec.axis == "reaction" and name in index_map:
                meta["reaction_index"] = index_map[name]
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
        payload = read_json(graph_path)
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
            node_meta = node_map.get(node_id, {})
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
            reaction_id = node_meta.get("reaction_id")
            if isinstance(reaction_id, str) and reaction_id.strip():
                meta["reaction_id"] = reaction_id.strip()
            reaction_index = node_meta.get("reaction_index")
            if reaction_index is not None and not isinstance(reaction_index, bool):
                try:
                    meta["reaction_index"] = int(reaction_index)
                except (TypeError, ValueError):
                    pass
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
        max_nodes = _coerce_optional_int(
            betweenness_max_nodes,
            "betweenness_max_nodes",
        )
        if max_nodes is not None and node_count > max_nodes:
            metric_status["betweenness_centrality"] = "node_limit"
            for node_id in nodes:
                metric_values["betweenness_centrality"][node_id] = math.nan
        elif nx is not None:
            graph = nx.DiGraph() if directed else nx.Graph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            scores = nx.betweenness_centrality(graph)
            metric_status["betweenness_centrality"] = "ok"
            for node_id in nodes:
                metric_values["betweenness_centrality"][node_id] = float(
                    scores.get(node_id, 0.0)
                )
        else:
            # Optional dependency fallback: Brandes betweenness centrality for
            # unweighted graphs (sufficient for ranking/pruning).
            from collections import deque

            index = {node_id: idx for idx, node_id in enumerate(nodes)}
            neighbors: list[list[int]] = []
            for node_id in nodes:
                nbrs = []
                for target in adjacency.get(node_id, set()):
                    idx = index.get(target)
                    if idx is not None:
                        nbrs.append(idx)
                neighbors.append(nbrs)

            n = len(nodes)
            betweenness = [0.0] * n
            for s in range(n):
                stack: list[int] = []
                preds: list[list[int]] = [[] for _ in range(n)]
                sigma = [0.0] * n
                dist = [-1] * n
                sigma[s] = 1.0
                dist[s] = 0
                queue: deque[int] = deque([s])
                while queue:
                    v = queue.popleft()
                    stack.append(v)
                    for w in neighbors[v]:
                        if dist[w] < 0:
                            dist[w] = dist[v] + 1
                            queue.append(w)
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            preds[w].append(v)
                delta = [0.0] * n
                while stack:
                    w = stack.pop()
                    for v in preds[w]:
                        if sigma[w]:
                            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    if w != s:
                        betweenness[w] += delta[w]

            metric_status["betweenness_centrality"] = "ok"
            for node_id, score in zip(nodes, betweenness):
                metric_values["betweenness_centrality"][node_id] = float(score)
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


def stable_projection_timeseries(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Project run time series onto a stable-species subset."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if xr is None:
        raise ConfigError("xarray is required for stable projection.")
    if np is None:
        raise ConfigError("numpy is required for stable projection.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = _extract_params(feat_cfg)
    run_ids = _extract_run_ids(feat_cfg, store=store)

    stable_value = (
        params.get("stable_states")
        or feat_cfg.get("stable_states")
        or feat_cfg.get("stable")
    )
    stable_payload, stable_path = _load_stable_states_payload(stable_value)
    stable_entries = _normalize_stable_entries(stable_payload)
    if not stable_entries:
        raise ConfigError("stable_states produced no stable species entries.")

    stats = (
        _normalize_stats(params.get("stats"))
        if params.get("stats") is not None
        else list(DEFAULT_STABLE_STATS)
    )
    x_var = params.get("x_var") or feat_cfg.get("x_var") or "X"
    if not isinstance(x_var, str) or not x_var.strip():
        raise ConfigError("x_var must be a non-empty string.")
    coverage_var = params.get("coverage_var") or feat_cfg.get("coverage_var") or "coverage"
    if not isinstance(coverage_var, str) or not coverage_var.strip():
        raise ConfigError("coverage_var must be a non-empty string.")
    missing_strategy = _normalize_missing_strategy(
        params.get("missing_strategy", feat_cfg.get("missing_strategy"))
    )
    site_total = _stable_site_total(stable_payload, params)
    if site_total is None:
        site_total = 1.0
    site_tol = _coerce_optional_float(params.get("site_tolerance"), "site_tolerance")
    if site_tol is None:
        site_tol = DEFAULT_STABLE_SITE_TOL

    phase_assignment: dict[str, Optional[str]] = {
        entry["name"]: entry.get("phase") for entry in stable_entries
    }
    gas_names: list[str] = []
    surface_names: list[str] = []
    time_values = None
    gas_series_list: list[Any] = []
    surface_series_list: list[Any] = []
    rows: list[dict[str, Any]] = []

    for run_index, run_id in enumerate(run_ids):
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        dataset = _load_run_dataset_xr(run_dir)

        if "time" not in dataset.coords:
            raise ConfigError("Run dataset must include time coord.")
        current_time = np.asarray(dataset.coords["time"].values)
        if time_values is None:
            time_values = current_time
        else:
            if len(current_time) != len(time_values) or np.any(
                np.asarray(current_time) != np.asarray(time_values)
            ):
                raise ConfigError("Time grid mismatch across runs.")

        gas_coord = []
        if "species" in dataset.coords:
            gas_coord = [str(name) for name in dataset.coords["species"].values]
        surface_coord = []
        if "surface_species" in dataset.coords:
            surface_coord = [str(name) for name in dataset.coords["surface_species"].values]

        if run_index == 0:
            for name, phase in phase_assignment.items():
                if phase is None:
                    if name in gas_coord:
                        phase_assignment[name] = "gas"
                    elif name in surface_coord:
                        phase_assignment[name] = "surface"

            gas_names = [name for name, phase in phase_assignment.items() if phase == "gas"]
            surface_names = [
                name for name, phase in phase_assignment.items() if phase == "surface"
            ]

        missing_names = [
            name for name, phase in phase_assignment.items() if phase is None
        ]
        for name in missing_names:
            if missing_strategy == "skip":
                continue
            for stat in stats:
                rows.append(
                    _build_row(
                        run_id,
                        f"stable_projection.unknown.{name}.{stat}",
                        math.nan,
                        "",
                        {
                            "status": "missing_species",
                            "stable_states_path": stable_path,
                        },
                    )
                )

        if surface_coord and coverage_var in dataset.data_vars:
            coverage = dataset[coverage_var].transpose("time", "surface_species").values
            if np.any(coverage < -site_tol) or np.any(coverage > 1.0 + site_tol):
                raise ConfigError("surface coverage must be within [0, 1].")
            total_sites = np.sum(coverage, axis=1)
            if site_total is not None and np.any(
                np.abs(total_sites - site_total) > site_tol
            ):
                raise ConfigError("surface coverage total must be constant.")

        gas_values = None
        if gas_names:
            gas_values = np.full((len(current_time), len(gas_names)), np.nan)
            for idx, name in enumerate(gas_names):
                series = _extract_series(
                    dataset,
                    var_name=x_var,
                    coord_name="species",
                    species_name=name,
                )
                if series is not None:
                    gas_values[:, idx] = np.asarray(series, dtype=float)
                elif missing_strategy != "skip":
                    for stat in stats:
                        rows.append(
                            _build_row(
                                run_id,
                                f"stable_projection.gas.{name}.{stat}",
                                math.nan,
                                "",
                                {
                                    "status": "missing_species",
                                    "variable": x_var,
                                    "stable_states_path": stable_path,
                                },
                            )
                        )

        surface_values = None
        if surface_names:
            if coverage_var not in dataset.data_vars:
                raise ConfigError(f"Run dataset missing variable: {coverage_var}")
            surface_values = np.full((len(current_time), len(surface_names)), np.nan)
            for idx, name in enumerate(surface_names):
                series = _extract_series(
                    dataset,
                    var_name=coverage_var,
                    coord_name="surface_species",
                    species_name=name,
                )
                if series is not None:
                    surface_values[:, idx] = np.asarray(series, dtype=float)
                elif missing_strategy != "skip":
                    for stat in stats:
                        rows.append(
                            _build_row(
                                run_id,
                                f"stable_projection.surface.{name}.{stat}",
                                math.nan,
                                "",
                                {
                                    "status": "missing_species",
                                    "variable": coverage_var,
                                    "stable_states_path": stable_path,
                                },
                            )
                        )

        units = dataset.attrs.get("units", {})
        if units is None:
            units = {}
        if not isinstance(units, Mapping):
            raise ConfigError("attrs.units must be a mapping when provided.")

        if gas_values is not None:
            unit = units.get(x_var, "")
            for idx, name in enumerate(gas_names):
                series = gas_values[:, idx]
                if np.isnan(series).all():
                    continue
                for stat in stats:
                    value = _compute_stat(series, time_values, stat)
                    rows.append(
                        _build_row(
                            run_id,
                            f"stable_projection.gas.{name}.{stat}",
                            value,
                            unit,
                            {
                                "phase": "gas",
                                "species": name,
                                "stat": stat,
                                "variable": x_var,
                                "stable_states_path": stable_path,
                            },
                        )
                    )

        if surface_values is not None:
            unit = units.get(coverage_var, "")
            for idx, name in enumerate(surface_names):
                series = surface_values[:, idx]
                if np.isnan(series).all():
                    continue
                for stat in stats:
                    value = _compute_stat(series, time_values, stat)
                    rows.append(
                        _build_row(
                            run_id,
                            f"stable_projection.surface.{name}.{stat}",
                            value,
                            unit,
                            {
                                "phase": "surface",
                                "species": name,
                                "stat": stat,
                                "variable": coverage_var,
                                "stable_states_path": stable_path,
                            },
                        )
                    )

        if gas_values is not None:
            gas_series_list.append(gas_values)
        if surface_values is not None:
            surface_series_list.append(surface_values)

    data_vars: dict[str, Any] = {}
    coords: dict[str, Any] = {"time": time_values}
    if gas_names and gas_series_list:
        if len(run_ids) > 1:
            data = np.stack(gas_series_list, axis=0)
            data_vars["X_stable"] = (("run", "time", "stable_species"), data)
            coords["run"] = run_ids
        else:
            data_vars["X_stable"] = (("time", "stable_species"), gas_series_list[0])
        coords["stable_species"] = gas_names
    if surface_names and surface_series_list:
        if len(run_ids) > 1:
            data = np.stack(surface_series_list, axis=0)
            data_vars["coverage_stable"] = (
                ("run", "time", "stable_surface_species"),
                data,
            )
            coords["run"] = run_ids
        else:
            data_vars["coverage_stable"] = (
                ("time", "stable_surface_species"),
                surface_series_list[0],
            )
        coords["stable_surface_species"] = surface_names

    if not data_vars:
        raise ConfigError("stable projection produced no time series data.")

    output = xr.Dataset(data_vars=data_vars, coords=coords)
    output.attrs["stable_states_path"] = stable_path
    output.attrs["stable_species"] = json.dumps(
        [entry["name"] for entry in stable_entries],
        ensure_ascii=True,
        sort_keys=True,
    )

    inputs_payload = {
        "runs": run_ids,
        "stable_states_path": stable_path,
        "stable_species": [entry["name"] for entry in stable_entries],
        "variables": {"x_var": x_var, "coverage_var": coverage_var},
        "stats": stats,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=run_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        output.to_zarr(base_dir / "stable_projection_timeseries.zarr", mode="w")
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


def superstate_timeseries(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Project species time series onto CNR-Coarse superstates."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for superstate timeseries.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = feat_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    params = dict(params)

    run_ids = _extract_run_ids(feat_cfg, store=store)
    if not run_ids:
        raise ConfigError("superstate timeseries requires run_id.")

    mapping_id = _extract_mapping_id(feat_cfg)
    mapping_payload = _load_mapping_payload(store, mapping_id)
    superstate_names, mapping_by_name, members_by_id = _resolve_superstate_mapping(
        mapping_payload
    )

    x_var = params.get("x_var") or feat_cfg.get("x_var") or "X"
    if not isinstance(x_var, str) or not x_var.strip():
        raise ConfigError("x_var must be a non-empty string.")
    wdot_var = params.get("wdot_var") or feat_cfg.get("wdot_var") or "net_production_rates"
    if not isinstance(wdot_var, str) or not wdot_var.strip():
        raise ConfigError("wdot_var must be a non-empty string.")
    include_wdot = params.get("include_wdot")
    if include_wdot is None:
        include_wdot = True
    if not isinstance(include_wdot, bool):
        raise ConfigError("include_wdot must be a boolean.")

    species_coord = params.get("species_coord") or feat_cfg.get("species_coord")
    if species_coord is not None and (
        not isinstance(species_coord, str) or not species_coord.strip()
    ):
        raise ConfigError("species_coord must be a non-empty string.")

    time_values = None
    series_list: list[Any] = []
    wdot_list: list[Any] = []
    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        if xr is not None:
            dataset = _load_run_dataset_xr(run_dir)

            coord_name = species_coord
            if coord_name is None:
                coord_name = "species" if "species" in dataset.coords else None
            if coord_name is None and "surface_species" in dataset.coords:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")

            if "time" not in dataset.coords:
                raise ConfigError("Run dataset must include time coord.")
            current_time = dataset.coords["time"].values
            if time_values is None:
                time_values = current_time
            else:
                if len(current_time) != len(time_values) or np.any(
                    np.asarray(current_time) != np.asarray(time_values)
                ):
                    raise ConfigError("Time grid mismatch across runs.")

            species_names = [str(name) for name in dataset.coords[coord_name].values]
            if x_var not in dataset.data_vars:
                raise ConfigError(f"Run dataset missing variable: {x_var}")
            x_values = dataset[x_var].transpose("time", coord_name).values
            if include_wdot:
                if wdot_var not in dataset.data_vars:
                    raise ConfigError(f"Run dataset missing variable: {wdot_var}")
                wdot_values = dataset[wdot_var].transpose("time", coord_name).values
            else:
                wdot_values = None
        else:
            payload = load_run_dataset_payload(run_dir)
            coords = payload.get("coords", {})
            if not isinstance(coords, Mapping):
                raise ConfigError("Run dataset coords must be a mapping.")
            coord_name = species_coord
            if coord_name is None:
                coord_name = "species" if "species" in coords else None
            if coord_name is None and "surface_species" in coords:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")

            time_payload = coords.get("time")
            if not isinstance(time_payload, Mapping):
                raise ConfigError("Run dataset must include time coord.")
            current_time = time_payload.get("data")
            if not isinstance(current_time, Sequence):
                raise ConfigError("Run dataset time coord is invalid.")
            current_time = np.asarray(current_time, dtype=float)
            if time_values is None:
                time_values = current_time
            else:
                if len(current_time) != len(time_values) or np.any(
                    np.asarray(current_time) != np.asarray(time_values)
                ):
                    raise ConfigError("Time grid mismatch across runs.")

            coord_payload = coords.get(coord_name)
            if not isinstance(coord_payload, Mapping):
                raise ConfigError("Run dataset species coord is missing.")
            species_data = coord_payload.get("data")
            if not isinstance(species_data, Sequence):
                raise ConfigError("Run dataset species coord is invalid.")
            species_names = [str(name) for name in species_data]

            data_vars = payload.get("data_vars", {})
            if not isinstance(data_vars, Mapping):
                raise ConfigError("Run dataset data_vars must be a mapping.")

            def _var_to_time_species(var_name: str) -> Any:
                entry = data_vars.get(var_name)
                if not isinstance(entry, Mapping):
                    raise ConfigError(f"Run dataset missing variable: {var_name}")
                dims = entry.get("dims")
                data = entry.get("data")
                if not isinstance(dims, Sequence) or not isinstance(data, Sequence):
                    raise ConfigError(f"Run dataset variable {var_name} is invalid.")
                dims_list = [str(d) for d in dims]
                if "time" not in dims_list or coord_name not in dims_list:
                    raise ConfigError(
                        f"Run dataset variable {var_name} missing time/species dims."
                    )
                arr = np.asarray(data, dtype=float)
                time_axis = dims_list.index("time")
                species_axis = dims_list.index(coord_name)
                arr = np.moveaxis(arr, [time_axis, species_axis], [0, 1])
                return arr

            x_values = _var_to_time_species(x_var)
            wdot_values = _var_to_time_species(wdot_var) if include_wdot else None
        missing = [name for name in species_names if name not in mapping_by_name]
        if missing:
            raise ConfigError(
                f"Missing species in mapping: {', '.join(missing[:5])}"
            )

        n_super = len(superstate_names)
        indicator = np.zeros((len(species_names), n_super), dtype=float)
        for idx, name in enumerate(species_names):
            indicator[idx, mapping_by_name[name]] = 1.0

        x_super = np.matmul(x_values, indicator)
        series_list.append(x_super)

        if include_wdot:
            wdot_super = np.matmul(wdot_values, indicator)
            wdot_list.append(wdot_super)

    dims = ("time", "superstate")
    if len(run_ids) > 1:
        dims = ("run", "time", "superstate")
        x_data = np.stack(series_list, axis=0)
        wdot_data = np.stack(wdot_list, axis=0) if wdot_list else None
    else:
        x_data = series_list[0]
        wdot_data = wdot_list[0] if wdot_list else None

    coords: dict[str, Any] = {
        "time": time_values,
        "superstate": superstate_names,
    }
    if len(run_ids) > 1:
        coords["run"] = run_ids

    data_vars: dict[str, Any] = {
        "X_super": (dims, x_data),
    }
    if include_wdot and wdot_data is not None:
        data_vars["wdot_super"] = (dims, wdot_data)

    attrs = {
        "mapping_id": mapping_id,
        "superstate_members": json.dumps(
            {str(key): value for key, value in members_by_id.items()},
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    output = None
    if xr is not None:
        output = xr.Dataset(data_vars=data_vars, coords=coords)
        output.attrs.update(attrs)

    inputs_payload = {
        "runs": run_ids,
        "mapping_id": mapping_id,
        "variables": {"x_var": x_var, "wdot_var": wdot_var, "include_wdot": include_wdot},
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=list(dict.fromkeys(run_ids + [mapping_id])),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        target = base_dir / "superstate_timeseries.zarr"
        target.mkdir(parents=True, exist_ok=True)
        if output is not None:
            output.to_zarr(target, mode="w")
            return
        payload = {
            "coords": {
                name: {"dims": [name], "data": np.asarray(values).tolist()}
                for name, values in coords.items()
            },
            "data_vars": {
                name: {"dims": list(dims), "data": np.asarray(values).tolist()}
                for name, (dims, values) in data_vars.items()
            },
            "attrs": attrs,
        }
        write_json_atomic(target / "dataset.json", payload)

    return store.ensure(manifest, writer=_writer)


def superstate_qoi(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Compute QoIs on superstates defined by a mapping (mapping.json or node_lumping.json).

    This is used to keep QoI comparisons valid even when species are merged/removed
    by a reduction method.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for superstate QoI.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = feat_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    params = dict(params)

    run_ids = _extract_run_ids(feat_cfg, store=store)
    if not run_ids:
        raise ConfigError("superstate QoI requires run_id.")
    mapping_id = _extract_mapping_id(feat_cfg)
    mapping_payload = _load_superstate_payload(store, mapping_id)
    superstate_names, mapping_by_name, _members_by_id = _resolve_superstate_mapping_any(
        mapping_payload
    )

    x_var = params.get("x_var") or feat_cfg.get("x_var") or "X"
    if not isinstance(x_var, str) or not x_var.strip():
        raise ConfigError("x_var must be a non-empty string.")
    t_var = params.get("t_var") or feat_cfg.get("t_var") or "T"
    if not isinstance(t_var, str) or not t_var.strip():
        raise ConfigError("t_var must be a non-empty string.")

    include_temperature_qoi = params.get("include_temperature_qoi")
    if include_temperature_qoi is None:
        include_temperature_qoi = True
    if not isinstance(include_temperature_qoi, bool):
        raise ConfigError("include_temperature_qoi must be a boolean.")
    include_ignition_delay = params.get("include_ignition_delay")
    if include_ignition_delay is None:
        include_ignition_delay = True
    if not isinstance(include_ignition_delay, bool):
        raise ConfigError("include_ignition_delay must be a boolean.")

    species_coord = params.get("species_coord") or feat_cfg.get("species_coord")
    if species_coord is not None and (
        not isinstance(species_coord, str) or not species_coord.strip()
    ):
        raise ConfigError("species_coord must be a non-empty string.")

    raw_targets = params.get("targets") or feat_cfg.get("targets")
    if raw_targets is None:
        raw_targets = [
            {"name": "CO_final_super", "member_species": "CO", "stat": "last"},
            {"name": "CO2_final_super", "member_species": "CO2", "stat": "last"},
        ]
    if not isinstance(raw_targets, Sequence) or isinstance(
        raw_targets, (str, bytes, bytearray)
    ):
        raise ConfigError("targets must be a list of mappings.")
    targets: list[dict[str, str]] = []
    for idx, entry in enumerate(raw_targets):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"targets[{idx}] must be a mapping.")
        name = entry.get("name") or entry.get("qoi") or entry.get("feature")
        member = entry.get("member_species") or entry.get("species") or entry.get("member")
        stat = entry.get("stat") or entry.get("agg") or "last"
        name = _require_nonempty_str(name, f"targets[{idx}].name")
        member = _require_nonempty_str(member, f"targets[{idx}].member_species")
        stat = _require_nonempty_str(stat, f"targets[{idx}].stat").lower()
        if stat not in DEFAULT_STATS:
            allowed = ", ".join(DEFAULT_STATS)
            raise ConfigError(f"targets[{idx}].stat must be one of: {allowed}.")
        targets.append({"name": name, "member_species": member, "stat": stat})
    if not targets:
        raise ConfigError("targets must include at least one entry.")

    def _as_qoi_name(name: str) -> str:
        cleaned = name.strip()
        if cleaned.startswith("qoi."):
            return cleaned
        return f"qoi.{cleaned}"

    def _ignition_delay(time_values: Sequence[float], temperatures: Sequence[float]) -> float:
        if len(time_values) != len(temperatures):
            raise ConfigError("T series length must match time coordinates.")
        if len(time_values) < 2:
            return math.nan
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
        return float(time_values[best_idx])

    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        species_names: list[str] = []
        time_values: list[float] = []
        x_values: Any = None
        t_values: Optional[Any] = None
        units: Mapping[str, Any] = {}
        coord_name = species_coord

        if xr is not None:
            dataset = _load_run_dataset_xr(run_dir)

            if coord_name is None:
                coord_name = "species" if "species" in dataset.coords else None
            if coord_name is None and "surface_species" in dataset.coords:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")
            if "time" not in dataset.coords:
                raise ConfigError("Run dataset must include time coord.")

            time_values = _coerce_float_sequence(
                dataset.coords["time"].values.tolist(),
                "coords.time",
            )
            species_names = [str(name) for name in dataset.coords[coord_name].values]
            if not species_names:
                raise ConfigError("Run dataset species coord is empty.")
            if x_var not in dataset.data_vars:
                raise ConfigError(f"Run dataset missing variable: {x_var}")
            x_values = np.asarray(
                dataset[x_var].transpose("time", coord_name).values, dtype=float
            )
            if x_values.shape[0] != len(time_values) or x_values.shape[1] != len(
                species_names
            ):
                raise ConfigError("X dims must match time/species coordinates.")

            if include_temperature_qoi or include_ignition_delay:
                if t_var not in dataset.data_vars:
                    raise ConfigError(f"Run dataset missing variable: {t_var}")
                t_values = np.asarray(
                    dataset[t_var].transpose("time").values, dtype=float
                )
                if t_values.shape[0] != len(time_values):
                    raise ConfigError("T dims must match time coordinates.")

            units = dataset.attrs.get("units", {})
            if units is None:
                units = {}
            if not isinstance(units, Mapping):
                raise ConfigError("attrs.units must be a mapping when provided.")
        else:
            payload = load_run_dataset_payload(run_dir)
            coords_payload = payload.get("coords", {})
            if not isinstance(coords_payload, Mapping):
                raise ConfigError("Run dataset coords must be a mapping.")
            data_vars_payload = payload.get("data_vars", {})
            if not isinstance(data_vars_payload, Mapping):
                raise ConfigError("Run dataset data_vars must be a mapping.")

            if coord_name is None:
                coord_name = "species" if "species" in coords_payload else None
            if coord_name is None and "surface_species" in coords_payload:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")

            time_entry = coords_payload.get("time")
            if not isinstance(time_entry, Mapping):
                raise ConfigError("Run dataset must include time coord.")
            time_values = _coerce_float_sequence(time_entry.get("data"), "coords.time")

            species_entry = coords_payload.get(coord_name)
            if not isinstance(species_entry, Mapping):
                raise ConfigError("Run dataset species coord is missing.")
            species_names = _coerce_str_sequence(species_entry.get("data"), "coords.species")

            def _var_to_time_species(var_name: str) -> Any:
                entry = data_vars_payload.get(var_name)
                if not isinstance(entry, Mapping):
                    raise ConfigError(f"Run dataset missing variable: {var_name}")
                dims = entry.get("dims")
                data = entry.get("data")
                if not isinstance(dims, Sequence) or not isinstance(data, Sequence):
                    raise ConfigError(f"Run dataset variable {var_name} is invalid.")
                dims_list = [str(d) for d in dims]
                if "time" not in dims_list or coord_name not in dims_list:
                    raise ConfigError(
                        f"Run dataset variable {var_name} missing time/species dims."
                    )
                arr = np.asarray(data, dtype=float)
                time_axis = dims_list.index("time")
                species_axis = dims_list.index(coord_name)
                arr = np.moveaxis(arr, [time_axis, species_axis], [0, 1])
                return arr

            def _var_to_time(var_name: str) -> Any:
                entry = data_vars_payload.get(var_name)
                if not isinstance(entry, Mapping):
                    raise ConfigError(f"Run dataset missing variable: {var_name}")
                dims = entry.get("dims")
                data = entry.get("data")
                if not isinstance(dims, Sequence) or not isinstance(data, Sequence):
                    raise ConfigError(f"Run dataset variable {var_name} is invalid.")
                dims_list = [str(d) for d in dims]
                if dims_list != ["time"]:
                    raise ConfigError(f"Run dataset variable {var_name} must be time-series.")
                return np.asarray(data, dtype=float)

            x_values = _var_to_time_species(x_var)
            if include_temperature_qoi or include_ignition_delay:
                t_values = _var_to_time(t_var)

            attrs_payload = payload.get("attrs", {})
            if not isinstance(attrs_payload, Mapping):
                attrs_payload = {}
            units = attrs_payload.get("units", {})
            if units is None:
                units = {}
            if not isinstance(units, Mapping):
                raise ConfigError("attrs.units must be a mapping when provided.")

        if t_values is not None and len(t_values) != len(time_values):
            raise ConfigError("T series length must match time coordinates.")

        missing_map = [name for name in species_names if name not in mapping_by_name]
        if missing_map:
            raise ConfigError(f"Missing species in mapping: {', '.join(missing_map[:5])}")

        x_unit = "" if units.get(x_var) is None else str(units.get(x_var))
        t_unit = "" if units.get(t_var) is None else str(units.get(t_var))
        time_unit = "" if units.get("time") is None else str(units.get("time"))

        for target in targets:
            member = target["member_species"]
            if member not in mapping_by_name:
                raise ConfigError(f"member_species not found in mapping: {member}")
            sid = int(mapping_by_name[member])
            sname = (
                superstate_names[sid]
                if 0 <= sid < len(superstate_names)
                else f"S{sid:03d}"
            )
            indices = [
                idx for idx, name in enumerate(species_names) if mapping_by_name[name] == sid
            ]
            if not indices:
                raise ConfigError(
                    f"No species from superstate {sname} (id={sid}) are present in run dataset."
                )
            series = np.sum(x_values[:, indices], axis=1)
            value = _compute_stat(series, time_values, target["stat"])
            rows.append(
                {
                    "run_id": run_id,
                    "feature": _as_qoi_name(target["name"]),
                    "value": float(value),
                    "unit": x_unit,
                    "meta_json": json.dumps(
                        {
                            "mapping_id": mapping_id,
                            "member_species": member,
                            "superstate_id": sid,
                            "superstate_name": sname,
                            "stat": target["stat"],
                            "variable": x_var,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                }
            )

        if include_temperature_qoi and t_values is not None:
            rows.append(
                {
                    "run_id": run_id,
                    "feature": "qoi.T_peak",
                    "value": float(np.nanmax(t_values)),
                    "unit": t_unit,
                    "meta_json": json.dumps(
                        {"method": "max", "variable": t_var},
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                }
            )
        if include_ignition_delay and t_values is not None:
            tau = _ignition_delay(time_values, np.asarray(t_values, dtype=float).tolist())
            rows.append(
                {
                    "run_id": run_id,
                    "feature": "qoi.ignition_delay",
                    "value": float(tau),
                    "unit": time_unit,
                    "meta_json": json.dumps(
                        {"method": "max_dTdt_prev_point", "variable": t_var},
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                }
            )

    inputs_payload = {
        "runs": run_ids,
        "mapping_id": mapping_id,
        "targets": list(targets),
        "variables": {"x_var": x_var, "t_var": t_var, "species_coord": species_coord},
        "include_temperature_qoi": include_temperature_qoi,
        "include_ignition_delay": include_ignition_delay,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=list(dict.fromkeys(run_ids + [mapping_id])),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


def superstate_merge_quality(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Evaluate a superstate mapping against a RunArtifact (projection-only).

    This task does not change the simulation; it reports information-loss / plausibility
    indicators for a given mapping when projected onto a specific run dataset.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for superstate merge quality.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = feat_cfg.get("params") or {}
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    params = dict(params)

    run_ids = _extract_run_ids(feat_cfg, store=store)
    if not run_ids:
        raise ConfigError("superstate_merge_quality requires run_id.")
    mapping_id = _extract_mapping_id(feat_cfg)
    mapping_payload = _load_superstate_payload(store, mapping_id)
    superstate_names, mapping_by_name, members_by_id = _resolve_superstate_mapping_any(
        mapping_payload
    )

    x_var = params.get("x_var") or feat_cfg.get("x_var") or "X"
    if not isinstance(x_var, str) or not x_var.strip():
        raise ConfigError("x_var must be a non-empty string.")
    species_coord = params.get("species_coord") or feat_cfg.get("species_coord")
    if species_coord is not None and (
        not isinstance(species_coord, str) or not species_coord.strip()
    ):
        raise ConfigError("species_coord must be a non-empty string.")

    kind_by_species: dict[str, str] = {}
    elements_by_species: dict[str, set[str]] = {}
    rep_by_super: dict[int, str] = {}
    # Optional extra metadata emitted by reduction.superstate_mapping.
    if isinstance(mapping_payload.get("composition_meta"), Sequence) and not isinstance(
        mapping_payload.get("composition_meta"),
        (str, bytes, bytearray),
    ):
        for entry in mapping_payload.get("composition_meta") or []:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("species")
            if not isinstance(name, str) or not name.strip():
                continue
            kind = entry.get("kind")
            if isinstance(kind, str) and kind.strip():
                kind_by_species[name.strip()] = kind.strip()
            elements = entry.get("elements")
            if isinstance(elements, Mapping):
                elements_by_species[name.strip()] = {
                    str(k) for k, v in elements.items() if v is not None and float(v) != 0.0
                }
    clusters = mapping_payload.get("clusters") or mapping_payload.get("superstates") or []
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes, bytearray)):
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            sid = cluster.get("superstate_id")
            rep = cluster.get("representative")
            if isinstance(sid, int) and isinstance(rep, str) and rep.strip():
                rep_by_super[int(sid)] = rep.strip()

    def _integrate_series(time_values: Sequence[float], matrix: Any) -> Any:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ConfigError("X variable must be a 2D time x species array.")
        if len(time_values) != arr.shape[0]:
            raise ConfigError("time coord length must match X time dimension.")
        if arr.shape[0] < 2:
            return np.zeros((arr.shape[1],), dtype=float)
        times = np.asarray(list(time_values), dtype=float)
        dt = np.diff(times)
        dt = np.where(dt > 0.0, dt, 1.0e-12)
        avg = 0.5 * (arr[1:, :] + arr[:-1, :])
        return np.sum(avg * dt[:, None], axis=0)

    def _entropy_effective(p: Any) -> float:
        probs = np.asarray(p, dtype=float)
        probs = probs[probs > 0.0]
        if probs.size == 0:
            return 0.0
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(math.exp(entropy))

    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        units: Mapping[str, Any] = {}

        coord_name = species_coord
        if xr is not None:
            dataset = _load_run_dataset_xr(run_dir)
            if coord_name is None:
                coord_name = "species" if "species" in dataset.coords else None
            if coord_name is None and "surface_species" in dataset.coords:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")
            if "time" not in dataset.coords:
                raise ConfigError("Run dataset must include time coord.")
            time_values = _coerce_float_sequence(
                dataset.coords["time"].values.tolist(),
                "coords.time",
            )
            species_names = [str(name) for name in dataset.coords[coord_name].values]
            if x_var not in dataset.data_vars:
                raise ConfigError(f"Run dataset missing variable: {x_var}")
            x_values = np.asarray(
                dataset[x_var].transpose("time", coord_name).values, dtype=float
            )
            units = dataset.attrs.get("units", {}) or {}
        else:
            payload = load_run_dataset_payload(run_dir)
            coords_payload = payload.get("coords", {})
            if not isinstance(coords_payload, Mapping):
                raise ConfigError("Run dataset coords must be a mapping.")
            data_vars_payload = payload.get("data_vars", {})
            if not isinstance(data_vars_payload, Mapping):
                raise ConfigError("Run dataset data_vars must be a mapping.")
            if coord_name is None:
                coord_name = "species" if "species" in coords_payload else None
            if coord_name is None and "surface_species" in coords_payload:
                coord_name = "surface_species"
            if coord_name is None:
                raise ConfigError("Run dataset must include species coords.")
            time_entry = coords_payload.get("time")
            if not isinstance(time_entry, Mapping):
                raise ConfigError("Run dataset must include time coord.")
            time_values = _coerce_float_sequence(time_entry.get("data"), "coords.time")
            species_entry = coords_payload.get(coord_name)
            if not isinstance(species_entry, Mapping):
                raise ConfigError("Run dataset species coord is missing.")
            species_names = _coerce_str_sequence(species_entry.get("data"), "coords.species")
            var_entry = data_vars_payload.get(x_var)
            if not isinstance(var_entry, Mapping):
                raise ConfigError(f"Run dataset missing variable: {x_var}")
            dims = var_entry.get("dims")
            data = var_entry.get("data")
            if not isinstance(dims, Sequence) or not isinstance(data, Sequence):
                raise ConfigError(f"Run dataset variable {x_var} is invalid.")
            dims_list = [str(d) for d in dims]
            if "time" not in dims_list or coord_name not in dims_list:
                raise ConfigError(f"Run dataset variable {x_var} missing time/species dims.")
            arr = np.asarray(data, dtype=float)
            time_axis = dims_list.index("time")
            species_axis = dims_list.index(coord_name)
            x_values = np.moveaxis(arr, [time_axis, species_axis], [0, 1])
            attrs_payload = payload.get("attrs", {}) or {}
            if isinstance(attrs_payload, Mapping):
                units = attrs_payload.get("units", {}) or {}
            else:
                units = {}

        missing_map = [name for name in species_names if name not in mapping_by_name]
        if missing_map:
            raise ConfigError(f"Missing species in mapping: {', '.join(missing_map[:5])}")

        integrals = _integrate_series(time_values, x_values)
        species_integral = {name: float(integrals[idx]) for idx, name in enumerate(species_names)}

        purity_values: list[float] = []
        effective_values: list[float] = []
        per_super: list[dict[str, Any]] = []
        kind_mismatch = 0
        element_overlap_violations = 0

        for sid in range(len(superstate_names)):
            members = members_by_id.get(sid, [])
            if not members:
                continue
            member_ints = np.asarray([species_integral.get(name, 0.0) for name in members], dtype=float)
            total = float(np.sum(member_ints))
            if total <= 0.0:
                purity = 1.0
                effective = 0.0
            else:
                purity = float(np.max(member_ints) / total)
                effective = _entropy_effective(member_ints / total)
            purity_values.append(purity)
            effective_values.append(effective)
            per_super.append(
                {
                    "superstate_id": int(sid),
                    "superstate": (
                        superstate_names[sid]
                        if 0 <= sid < len(superstate_names)
                        else f"S{sid:03d}"
                    ),
                    "members": list(members),
                    "member_count": int(len(members)),
                    "purity": float(purity),
                    "effective_species": float(effective),
                    "integral_total": float(total),
                }
            )

            # Optional guard-violation checks when metadata exists.
            if kind_by_species:
                kinds = {kind_by_species.get(name, "unknown") for name in members}
                if len(kinds) > 1:
                    kind_mismatch += 1
            if elements_by_species and rep_by_super:
                rep = rep_by_super.get(sid)
                if rep is not None:
                    rep_set = elements_by_species.get(rep, set())
                    if rep_set:
                        for name in members:
                            if name == rep:
                                continue
                            if not (rep_set & elements_by_species.get(name, set())):
                                element_overlap_violations += 1

        purity_mean = float(sum(purity_values) / len(purity_values)) if purity_values else math.nan
        purity_max = float(max(purity_values)) if purity_values else math.nan
        eff_mean = float(sum(effective_values) / len(effective_values)) if effective_values else math.nan
        eff_max = float(max(effective_values)) if effective_values else math.nan

        super_count = int(len(superstate_names))
        sp_count = int(len(species_names))
        coverage = 1.0
        compression_ratio = float(super_count) / float(sp_count) if sp_count else math.nan

        meta_common = {"mapping_id": mapping_id, "x_var": x_var, "species_coord": coord_name}

        def _row(
            feature: str,
            value: float,
            *,
            unit: str = "",
            extra: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            meta = dict(meta_common)
            if extra:
                meta.update(extra)
            return {
                "run_id": run_id,
                "feature": feature,
                "value": float(value),
                "unit": unit,
                "meta_json": json.dumps(meta, ensure_ascii=True, sort_keys=True),
            }

        rows.append(_row("merge.coverage", coverage))
        rows.append(_row("merge.superstate_count", float(super_count), unit="count", extra={"stat": "count"}))
        rows.append(_row("merge.compression_ratio", compression_ratio))
        rows.append(_row("merge.cluster_purity.mean", purity_mean))
        rows.append(_row("merge.cluster_purity.max", purity_max))
        rows.append(_row("merge.cluster_effective_species.mean", eff_mean))
        rows.append(_row("merge.cluster_effective_species.max", eff_max))
        rows.append(_row("merge.kind_mismatch_count", float(kind_mismatch), unit="count"))
        rows.append(_row("merge.element_overlap_violations", float(element_overlap_violations), unit="count"))

        # Per-superstate details for downstream "repair" steps.
        for entry in per_super:
            sid = entry["superstate_id"]
            sname = entry["superstate"]
            member_count = entry["member_count"]
            meta = {
                "mapping_id": mapping_id,
                "x_var": x_var,
                "species_coord": coord_name,
                "superstate_id": sid,
                "superstate": sname,
                "member_count": member_count,
            }
            rows.append(
                {
                    "run_id": run_id,
                    "feature": "merge.superstate_purity",
                    "value": float(entry["purity"]),
                    "unit": "",
                    "meta_json": json.dumps(meta, ensure_ascii=True, sort_keys=True),
                }
            )
            rows.append(
                {
                    "run_id": run_id,
                    "feature": "merge.superstate_effective_species",
                    "value": float(entry["effective_species"]),
                    "unit": "count",
                    "meta_json": json.dumps(meta, ensure_ascii=True, sort_keys=True),
                }
            )
            rows.append(
                {
                    "run_id": run_id,
                    "feature": "merge.superstate_integral_total",
                    "value": float(entry["integral_total"]),
                    "unit": "",
                    "meta_json": json.dumps(meta, ensure_ascii=True, sort_keys=True),
                }
            )

    inputs_payload = {
        "runs": run_ids,
        "mapping_id": mapping_id,
        "x_var": x_var,
        "species_coord": species_coord,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=list(dict.fromkeys(run_ids + [mapping_id])),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


def reduction_cheap_metrics(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Compute reduction "cheap" metrics from TemporalFluxGraph reaction activity.

    This task does not re-simulate. It provides fast, method-agnostic proxies such as:
      - cheap.flux_coverage_retained (activity-weighted retention under multipliers)
      - cheap.top_path_jaccard (Top-K reaction set overlap proxy)

    Inputs:
      - graph_flux_id: GraphArtifact id (graphs.temporal_flux output)
      - patches: list of reduction artifact ids (each should include mechanism_patch.yaml)
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for reduction cheap metrics.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    inputs = feat_cfg.get("inputs") or {}
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("inputs must be a mapping when provided.")
    inputs = dict(inputs)
    params = feat_cfg.get("params") or {}
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    params = dict(params)

    graph_flux_id = (
        inputs.get("graph_flux_id")
        or inputs.get("graph_id")
        or inputs.get("graph")
        or feat_cfg.get("graph_flux_id")
        or feat_cfg.get("graph_id")
    )
    graph_flux_id = _require_nonempty_str(graph_flux_id, "graph_flux_id")
    store.read_manifest("graphs", graph_flux_id)
    graph_dir = store.artifact_dir("graphs", graph_flux_id)
    graph_payload = read_json(graph_dir / "graph.json")
    if not isinstance(graph_payload, Mapping):
        raise ConfigError("TemporalFluxGraph graph.json must be a mapping.")

    reactions_section = graph_payload.get("reactions") or {}
    reaction_order: list[str] = []
    if isinstance(reactions_section, Mapping):
        order = reactions_section.get("order")
        if isinstance(order, Sequence) and not isinstance(order, (str, bytes, bytearray)):
            reaction_order = [str(item) for item in order]

    activity_values: list[float] = []
    activity_section = (
        (graph_payload.get("reaction_stats") or {}).get("activity") if isinstance(graph_payload.get("reaction_stats"), Mapping) else None
    )
    if isinstance(activity_section, Mapping):
        values = activity_section.get("values")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            try:
                activity_values = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise ConfigError("reaction_stats.activity.values must be numeric.") from exc
    if not activity_values:
        raise ConfigError("TemporalFluxGraph is missing reaction_stats.activity.values.")

    n_reactions = len(activity_values)
    if reaction_order and len(reaction_order) != n_reactions:
        raise ConfigError("TemporalFluxGraph reactions.order length mismatch.")

    total_activity = float(sum(activity_values))
    if total_activity <= 0.0:
        total_activity = 0.0

    patches_raw = inputs.get("patches") or inputs.get("patch_ids") or inputs.get("reductions") or []
    patch_ids: list[str] = []
    if isinstance(patches_raw, str):
        patch_ids = [_require_nonempty_str(patches_raw, "patches")]
    elif isinstance(patches_raw, Sequence) and not isinstance(
        patches_raw, (str, bytes, bytearray)
    ):
        patch_ids = [
            _require_nonempty_str(item, "patches")
            for item in patches_raw
            if item is not None
        ]
    else:
        raise ConfigError("patches must be a string or list of strings.")
    patch_ids = [pid for pid in patch_ids if pid]
    if not patch_ids:
        raise ConfigError("patches must include at least one reduction id.")

    top_k_raw = params.get("top_k", params.get("top_n", 50))
    if isinstance(top_k_raw, bool):
        raise ConfigError("top_k must be an integer.")
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("top_k must be an integer.") from exc
    if top_k <= 0:
        raise ConfigError("top_k must be positive.")
    top_k = min(top_k, n_reactions) if n_reactions else top_k

    # Baseline Top-K reactions by activity.
    base_scores = np.asarray(activity_values, dtype=float)
    base_order = sorted(
        range(n_reactions),
        key=lambda idx: (-float(base_scores[idx]), int(idx)),
    )
    base_top = set(base_order[:top_k])

    reaction_index_by_id: dict[str, int] = {}
    if reaction_order:
        for idx, rid in enumerate(reaction_order):
            if rid not in reaction_index_by_id:
                reaction_index_by_id[rid] = int(idx)

    def _effective_multiplier(raw: float) -> float:
        if not math.isfinite(raw):
            return 0.0
        value = abs(float(raw))
        if value <= 0.0:
            return 0.0
        # Treat this as a "retention" proxy, not amplification.
        return min(1.0, value)

    source_runs = graph_payload.get("source")
    run_id = "unknown"
    if isinstance(source_runs, Mapping):
        raw = source_runs.get("run_ids") or []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and raw:
            first = raw[0]
            if isinstance(first, str) and first.strip():
                run_id = first.strip()

    rows: list[dict[str, Any]] = []
    for patch_id in patch_ids:
        reduction_id = str(patch_id)
        patch_path = store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
        patch_payload: Mapping[str, Any] = {}
        status = "ok"
        if patch_path.exists():
            patch_raw = read_yaml_payload(patch_path)
            if isinstance(patch_raw, Mapping):
                patch_payload = dict(patch_raw)
            else:
                status = "invalid_patch"
        else:
            status = "missing_patch"

        multipliers_vec = [1.0 for _ in range(n_reactions)]
        if status == "ok":
            try:
                specs = normalize_reaction_multipliers(dict(patch_payload))
            except Exception:
                specs = []
                status = "invalid_patch"
            for entry in specs:
                if not isinstance(entry, Mapping):
                    continue
                multiplier_raw = entry.get("multiplier", 1.0)
                try:
                    multiplier = float(multiplier_raw)
                except (TypeError, ValueError):
                    continue
                if "index" in entry:
                    idx = entry.get("index")
                    if isinstance(idx, bool) or idx is None:
                        continue
                    try:
                        idx_int = int(idx)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx_int < n_reactions:
                        multipliers_vec[idx_int] = float(multiplier)
                else:
                    rid = entry.get("reaction_id")
                    if not isinstance(rid, str) or not rid.strip():
                        continue
                    mapped = reaction_index_by_id.get(rid.strip())
                    if mapped is None:
                        continue
                    multipliers_vec[mapped] = float(multiplier)

        if status == "ok":
            eff = np.asarray([_effective_multiplier(v) for v in multipliers_vec], dtype=float)
            weighted = base_scores * eff

            retained = float(np.sum(weighted)) / total_activity if total_activity > 0.0 else math.nan
            reduced_order = sorted(
                range(n_reactions),
                key=lambda idx: (-float(weighted[idx]), int(idx)),
            )
            reduced_top = set(reduced_order[:top_k])
            union = base_top | reduced_top
            jaccard = (len(base_top & reduced_top) / float(len(union))) if union else math.nan
        else:
            retained = math.nan
            jaccard = math.nan

        meta_common = {
            "graph_flux_id": graph_flux_id,
            "reduction_id": reduction_id,
            "top_k": int(top_k),
            "status": status,
        }

        def _emit(feature: str, value: float, unit: str = "") -> None:
            rows.append(
                {
                    "run_id": run_id,
                    "feature": feature,
                    "value": float(value),
                    "unit": unit,
                    "meta_json": json.dumps(meta_common, ensure_ascii=True, sort_keys=True),
                }
            )

        _emit("cheap.flux_coverage_retained", retained)
        _emit("cheap.top_path_jaccard", jaccard)

    inputs_payload = {
        "graph_flux_id": graph_flux_id,
        "patches": list(patch_ids),
        "top_k": int(top_k),
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=list(dict.fromkeys([graph_flux_id] + list(patch_ids))),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


def _normalize_feature_norm(value: Any) -> str:
    if value is None:
        return "mean_abs"
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("gnn.feature_norm must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized in {"mean_abs", "mean"}:
        return "mean_abs"
    if normalized in {"l1", "sum_abs"}:
        return "l1"
    if normalized in {"l2", "norm"}:
        return "l2"
    raise ConfigError("gnn.feature_norm must be one of: mean_abs, l1, l2.")


def _gnn_base_scores(
    features: Sequence[Sequence[Any]],
    *,
    norm: str,
) -> list[float]:
    scores: list[float] = []
    for row in features:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            scores.append(0.0)
            continue
        values: list[float] = []
        for entry in row:
            try:
                values.append(float(entry))
            except (TypeError, ValueError):
                values.append(0.0)
        if not values:
            scores.append(0.0)
            continue
        if norm == "l1":
            scores.append(sum(abs(v) for v in values))
        elif norm == "l2":
            scores.append(math.sqrt(sum(v * v for v in values)))
        else:
            scores.append(sum(abs(v) for v in values) / float(len(values)))
    return scores


def _gnn_message_passing(
    base_scores: Sequence[float],
    *,
    edges: Sequence[tuple[int, int, float]],
    steps: int,
    alpha: float,
    normalize: bool,
    undirected: bool,
) -> list[float]:
    node_count = len(base_scores)
    if node_count == 0:
        return []
    row_sum = [0.0 for _ in range(node_count)]
    if normalize:
        for src, dst, weight in edges:
            if 0 <= src < node_count and 0 <= dst < node_count:
                row_sum[src] += abs(weight)
                if undirected and src != dst:
                    row_sum[dst] += abs(weight)
        row_sum = [value if value > 0.0 else 1.0 for value in row_sum]
    scores = [float(value) for value in base_scores]
    for _ in range(max(steps, 0)):
        neighbor = [0.0 for _ in range(node_count)]
        for src, dst, weight in edges:
            if 0 <= src < node_count and 0 <= dst < node_count:
                w = abs(weight)
                neighbor[src] += w * scores[dst]
                if undirected and src != dst:
                    neighbor[dst] += w * scores[src]
        if normalize:
            neighbor = [value / row_sum[idx] for idx, value in enumerate(neighbor)]
        scores = [
            float(base_scores[idx]) + alpha * neighbor[idx]
            for idx in range(node_count)
        ]
    return scores


def _extract_window_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    windows = payload.get("windows")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
        raise ConfigError("gnn dataset windows must be a list.")
    entries: list[dict[str, Any]] = []
    for entry in windows:
        if not isinstance(entry, Mapping):
            continue
        window_meta = entry.get("window") or {}
        if window_meta is None:
            window_meta = {}
        if not isinstance(window_meta, Mapping):
            raise ConfigError("gnn dataset window metadata must be a mapping.")
        window_id = entry.get("index", entry.get("window_id"))
        if window_id is None:
            window_id = len(entries)
        try:
            window_id = int(window_id)
        except (TypeError, ValueError) as exc:
            raise ConfigError("gnn dataset window index must be an integer.") from exc
        start_idx = int(window_meta.get("start_idx", 0))
        end_idx = int(window_meta.get("end_idx", start_idx))
        entries.append(
            {
                "window_id": window_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window": dict(window_meta),
            }
        )
    if not entries:
        raise ConfigError("gnn dataset windows are empty.")
    return entries


def _integrate_window_matrix(
    time_values: Sequence[float],
    matrix: Sequence[Sequence[float]],
    *,
    start_idx: int,
    end_idx: int,
    use_abs: bool,
) -> list[float]:
    if end_idx <= start_idx:
        return [0.0 for _ in matrix[0]]
    start_idx = max(0, start_idx)
    end_idx = min(len(time_values) - 1, end_idx)
    values = matrix[start_idx : end_idx + 1]
    times = time_values[start_idx : end_idx + 1]
    if len(values) < 2 or len(times) < 2:
        return [0.0 for _ in matrix[0]]
    totals = [0.0 for _ in values[0]]
    for idx in range(1, len(times)):
        dt = times[idx] - times[idx - 1]
        row = values[idx]
        prev = values[idx - 1]
        for col in range(len(totals)):
            v0 = prev[col]
            v1 = row[col]
            if use_abs:
                v0 = abs(v0)
                v1 = abs(v1)
            totals[col] += 0.5 * (v0 + v1) * dt
    return totals


def gnn_importance(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Estimate reaction/species importance using temporal GNN inputs."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, feat_cfg = _extract_features_cfg(resolved_cfg)
    params = feat_cfg.get("params") or {}
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    params = dict(params)
    inputs = feat_cfg.get("inputs") or {}
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("inputs must be a mapping when provided.")

    dataset_id = None
    for key in ("dataset", "dataset_id", "gnn_dataset", "gnn_dataset_id"):
        if key in inputs:
            dataset_id = inputs.get(key)
            break
    if dataset_id is None:
        for key in ("dataset", "dataset_id", "gnn_dataset", "gnn_dataset_id"):
            if key in params:
                dataset_id = params.get(key)
                break
    dataset_id = _require_nonempty_str(dataset_id, "gnn_dataset_id")

    dataset_payload, dataset_dir = _load_gnn_dataset_payload(store, dataset_id)
    items = _load_gnn_dataset_items(dataset_payload, dataset_dir=dataset_dir)
    if not items:
        raise ConfigError("gnn dataset contains no items.")

    source_meta = dataset_payload.get("source") or {}
    if source_meta is None:
        source_meta = {}
    if not isinstance(source_meta, Mapping):
        raise ConfigError("gnn dataset source must be a mapping.")
    graph_id = inputs.get("graph_id") or source_meta.get("graph_id")

    run_set_id = inputs.get("run_set_id")
    inputs_run_ids_raw = inputs.get("run_ids") or inputs.get("runs")
    source_run_ids_raw = source_meta.get("run_ids") or source_meta.get("runs")
    if run_set_id is not None and inputs_run_ids_raw is not None:
        raise ConfigError("Specify only one of run_set_id or run_id(s) for gnn_importance.")
    if run_set_id is not None:
        run_set_id = _require_nonempty_str(run_set_id, "run_set_id")
        run_ids = load_run_ids_from_run_set(store, run_set_id)
    else:
        run_ids = _coerce_str_sequence(inputs_run_ids_raw or source_run_ids_raw, "run_ids")
        if not run_ids:
            raise ConfigError("gnn_importance requires run_ids.")

    windows = _extract_window_entries(dataset_payload)

    gnn_cfg = params.get("gnn") or {}
    if gnn_cfg is None:
        gnn_cfg = {}
    if not isinstance(gnn_cfg, Mapping):
        raise ConfigError("gnn params must be a mapping.")
    gnn_cfg = dict(gnn_cfg)

    alpha = gnn_cfg.get("alpha", 0.6)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ConfigError("gnn.alpha must be numeric.") from exc
    steps = gnn_cfg.get("steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ConfigError("gnn.steps must be an integer.")
    if steps < 0:
        raise ConfigError("gnn.steps must be >= 0.")
    normalize = gnn_cfg.get("normalize")
    if normalize is None:
        normalize = True
    if not isinstance(normalize, bool):
        raise ConfigError("gnn.normalize must be a boolean.")
    undirected = gnn_cfg.get("undirected")
    if undirected is None:
        undirected = True
    if not isinstance(undirected, bool):
        raise ConfigError("gnn.undirected must be a boolean.")
    feature_norm = _normalize_feature_norm(gnn_cfg.get("feature_norm"))

    output_cfg = params.get("output") or {}
    if output_cfg is None:
        output_cfg = {}
    if not isinstance(output_cfg, Mapping):
        raise ConfigError("output params must be a mapping.")
    output_cfg = dict(output_cfg)
    reaction_feature = output_cfg.get("reaction_feature") or "gnn_reaction_importance"
    species_feature = output_cfg.get("species_feature") or "gnn_species_importance"
    include_reactions = output_cfg.get("include_reactions")
    if include_reactions is None:
        include_reactions = True
    if not isinstance(include_reactions, bool):
        raise ConfigError("output.include_reactions must be a boolean.")
    include_species = output_cfg.get("include_species")
    if include_species is None:
        include_species = True
    if not isinstance(include_species, bool):
        raise ConfigError("output.include_species must be a boolean.")

    rop_var = params.get("rop_var") or params.get("rop") or "rop_net"
    if not isinstance(rop_var, str) or not rop_var.strip():
        raise ConfigError("rop_var must be a non-empty string.")
    rop_use_abs = params.get("rop_use_abs")
    if rop_use_abs is None:
        rop_use_abs = True
    if not isinstance(rop_use_abs, bool):
        raise ConfigError("rop_use_abs must be a boolean.")

    node_meta = []
    node_section = dataset_payload.get("nodes") or {}
    if isinstance(node_section, Mapping):
        node_meta = node_section.get("meta") or []
    if not isinstance(node_meta, Sequence) or isinstance(
        node_meta, (str, bytes, bytearray)
    ):
        node_meta = []
    node_meta_list = [dict(entry) for entry in node_meta if isinstance(entry, Mapping)]
    node_order = []
    if isinstance(node_section, Mapping):
        order = node_section.get("order")
        if isinstance(order, Sequence) and not isinstance(
            order, (str, bytes, bytearray)
        ):
            node_order = [str(name) for name in order]

    rows: list[dict[str, Any]] = []

    if include_species:
        for item in items:
            run_id = item.get("run_id") or item.get("case_id")
            if run_id is None or not isinstance(run_id, str):
                continue
            if run_id not in run_ids:
                run_ids.append(run_id)
            window_id = item.get("window_id")
            if window_id is None:
                continue
            try:
                window_id = int(window_id)
            except (TypeError, ValueError):
                continue
            features = item.get("x") or item.get("features")
            if not isinstance(features, Sequence) or isinstance(
                features, (str, bytes, bytearray)
            ):
                continue
            base_scores = _gnn_base_scores(features, norm=feature_norm)
            edge_index = item.get("edge_index") or []
            edge_attr = item.get("edge_attr") or []
            edges: list[tuple[int, int, float]] = []
            if (
                isinstance(edge_index, Sequence)
                and len(edge_index) == 2
                and isinstance(edge_index[0], Sequence)
                and isinstance(edge_index[1], Sequence)
            ):
                rows_idx = list(edge_index[0])
                cols_idx = list(edge_index[1])
                for idx, (src, dst) in enumerate(zip(rows_idx, cols_idx)):
                    try:
                        weight = float(edge_attr[idx]) if idx < len(edge_attr) else 1.0
                    except (TypeError, ValueError):
                        weight = 1.0
                    try:
                        src_i = int(src)
                        dst_i = int(dst)
                    except (TypeError, ValueError):
                        continue
                    edges.append((src_i, dst_i, weight))
            scores = _gnn_message_passing(
                base_scores,
                edges=edges,
                steps=steps,
                alpha=alpha,
                normalize=normalize,
                undirected=undirected,
            )
            window_info = item.get("window") or {}
            if window_info is None:
                window_info = {}
            for idx, score in enumerate(scores):
                meta = (
                    node_meta_list[idx] if idx < len(node_meta_list) else {}
                )
                species = meta.get("species")
                if species is None:
                    continue
                meta_payload = {
                    "species": species,
                    "species_index": meta.get("species_index"),
                    "node_id": node_order[idx] if idx < len(node_order) else meta.get("id"),
                    "window_id": window_id,
                    "window": window_info,
                    "case_id": run_id,
                    "source": "gnn",
                }
                rows.append(
                    _build_row(
                        run_id,
                        species_feature,
                        float(score),
                        "",
                        meta_payload,
                    )
                )

    run_ids = list(dict.fromkeys(run_ids))

    if include_reactions:
        for run_id in run_ids:
            store.read_manifest("runs", run_id)
            run_dir = store.artifact_dir("runs", run_id)
            run_dataset = _load_run_dataset_view(run_dir)
            time_values = _extract_time_values(run_dataset)
            reaction_names = _extract_coord_data(run_dataset, "reaction")
            if not isinstance(reaction_names, Sequence) or isinstance(
                reaction_names, (str, bytes, bytearray)
            ):
                raise ConfigError("coords.reaction must be a sequence.")
            reaction_names = [str(name) for name in reaction_names]
            rop_payload = run_dataset.data_vars.get(rop_var)
            if rop_payload is None:
                raise ConfigError(f"Run dataset missing variable: {rop_var}")
            matrix = _extract_time_matrix(
                rop_payload,
                time_values,
                axis="reaction",
                axis_names=reaction_names,
            )
            units = run_dataset.attrs.get("units", {})
            if units is None:
                units = {}
            if not isinstance(units, Mapping):
                raise ConfigError("attrs.units must be a mapping when provided.")
            time_unit = units.get("time", "")
            unit = _integral_unit(str(units.get(rop_var, "")), str(time_unit))
            for window in windows:
                values = _integrate_window_matrix(
                    time_values,
                    matrix,
                    start_idx=window["start_idx"],
                    end_idx=window["end_idx"],
                    use_abs=rop_use_abs,
                )
                for r_idx, value in enumerate(values):
                    meta_payload = {
                        "reaction_id": reaction_names[r_idx],
                        "reaction_index": r_idx,
                        "window_id": window["window_id"],
                        "window": window["window"],
                        "case_id": run_id,
                        "source": "rop",
                    }
                    rows.append(
                        _build_row(
                            run_id,
                            reaction_feature,
                            float(value),
                            unit,
                            meta_payload,
                        )
                    )

    if not rows:
        raise ConfigError("gnn_importance produced no feature rows.")

    inputs_payload = {
        "runs": run_ids,
        "dataset_id": dataset_id,
        "graph_id": graph_id,
        "features": [reaction_feature, species_feature],
        "gnn": {
            "alpha": alpha,
            "steps": steps,
            "normalize": normalize,
            "undirected": undirected,
            "feature_norm": feature_norm,
        },
        "rop_var": rop_var,
        "rop_use_abs": rop_use_abs,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = list(dict.fromkeys(run_ids + [dataset_id]))
    if graph_id:
        parents.append(str(graph_id))
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")
        summary = {
            "rows": len(rows),
            "run_ids": run_ids,
            "dataset_id": dataset_id,
            "graph_id": graph_id,
            "features": [reaction_feature, species_feature],
        }
        write_json_atomic(base_dir / "importance_summary.json", summary)

    return store.ensure(manifest, writer=_writer)


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
    run_ids = _extract_run_ids(feat_cfg, store=store)
    inputs = feat_cfg.get("inputs")

    graph_id_override: Optional[str] = None
    if isinstance(inputs, Mapping):
        raw_graph = None
        for key in ("graph", "graphs", "graph_id", "graph_ids"):
            if key in inputs:
                raw_graph = inputs.get(key)
                break
        if raw_graph is not None:
            if isinstance(raw_graph, str):
                graph_id_override = _require_nonempty_str(raw_graph, "graph_id")
            elif isinstance(raw_graph, Sequence) and not isinstance(
                raw_graph, (str, bytes, bytearray)
            ):
                items = [item for item in raw_graph if item is not None]
                if len(items) != 1:
                    raise ConfigError("graph_id must include exactly one entry.")
                graph_id_override = _require_nonempty_str(items[0], "graph_id")
            else:
                raise ConfigError("graph_id must be a string or single-item sequence.")

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
                if graph_id_override and "graph_id" not in feature_params:
                    feature_params["graph_id"] = graph_id_override
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
    manifest = build_manifest(
        kind="features",
        artifact_id=artifact_id,
        parents=run_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_features_table(rows, base_dir / "features.parquet")

    return store.ensure(manifest, writer=_writer)


register("task", "features.run", run)
register("task", "features.compute", run)
register("task", "features.stable_projection", stable_projection_timeseries)
register("task", "features.superstate_timeseries", superstate_timeseries)
register("task", "features.superstate_qoi", superstate_qoi)
register("task", "features.superstate_merge_quality", superstate_merge_quality)
register("task", "features.reduction_cheap_metrics", reduction_cheap_metrics)
register("task", "features.gnn_importance", gnn_importance)
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
    "stable_projection_timeseries",
    "superstate_timeseries",
    "superstate_qoi",
    "superstate_merge_quality",
    "gnn_importance",
]
