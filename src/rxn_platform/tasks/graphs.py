"""Graph tasks for stoichiometric matrix artifacts."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Optional
from rxn_platform.core import make_artifact_id, stable_hash
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.mechanism import read_yaml_payload as _read_mech_yaml_payload
from rxn_platform.graphs.temporal_flux_graph import build_temporal_flux_graph
from rxn_platform.registry import Registry, register
from rxn_platform.run_store import (
    resolve_run_root_from_store,
    sync_temporal_graph_from_artifact,
)
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    load_run_dataset_payload,
    load_run_ids_from_run_set,
    resolve_cfg as _resolve_cfg,
)

try:  # Optional dependency.
    import cantera as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

logger = logging.getLogger(__name__)

try:  # Optional dependency.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

try:  # Optional dependency.
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None


@dataclass(frozen=True)
class StoichResult:
    matrix: Any
    species: list[str]
    reaction_ids: list[str]
    reaction_equations: list[str]
    format: str


@dataclass(frozen=True)
class LaplacianResult:
    laplacian: Any
    degree: Any
    nodes: list[str]
    normalized_laplacian: Optional[Any]
    format: str
    normalized_format: Optional[str]


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float


def _extract_graph_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "graphs" in cfg and isinstance(cfg.get("graphs"), Mapping):
        graph_cfg = cfg.get("graphs")
        if not isinstance(graph_cfg, Mapping):
            raise ConfigError("graphs config must be a mapping.")
        return dict(cfg), dict(graph_cfg)
    if "graph" in cfg and isinstance(cfg.get("graph"), Mapping):
        graph_cfg = cfg.get("graph")
        if not isinstance(graph_cfg, Mapping):
            raise ConfigError("graph config must be a mapping.")
        return dict(cfg), dict(graph_cfg)
    return dict(cfg), dict(cfg)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_optional_str(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_str(value, label)


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


def _extract_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = cfg.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ConfigError("graph.params must be a mapping.")
    return dict(params)


def _find_value(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            if key in source and source.get(key) is not None:
                return source.get(key)
    return None


def _normalize_mechanism(value: Any) -> str:
    mech = _require_nonempty_str(value, "mechanism")
    mech_path = Path(mech)
    if mech_path.is_absolute() or len(mech_path.parts) > 1:
        if not mech_path.exists():
            raise ConfigError(f"mechanism file not found: {mech}")
    return mech


def _extract_mechanism(graph_cfg: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    mechanism: Any = None
    phase: Any = None
    inputs = graph_cfg.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise ConfigError("graphs.inputs must be a mapping.")
        for key in ("mechanism", "solution", "mechanism_path", "source"):
            if key in inputs:
                mechanism = inputs.get(key)
                break
        if "phase" in inputs:
            phase = inputs.get("phase")
    if mechanism is None:
        for key in ("mechanism", "solution", "mechanism_path", "source"):
            if key in graph_cfg:
                mechanism = graph_cfg.get(key)
                break
    if phase is None and "phase" in graph_cfg:
        phase = graph_cfg.get("phase")

    mechanism = _normalize_mechanism(mechanism)

    if phase is not None:
        phase = _require_nonempty_str(phase, "phase")
    return mechanism, phase


def _extract_optional_mechanism(
    graph_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    mechanism: Any = None
    phase: Any = None
    inputs = graph_cfg.get("inputs")
    if inputs is not None and not isinstance(inputs, Mapping):
        raise ConfigError("graphs.inputs must be a mapping.")
    sources: list[Mapping[str, Any]] = []
    if isinstance(inputs, Mapping):
        sources.append(inputs)
    sources.append(params)
    sources.append(graph_cfg)

    mechanism = _find_value(
        sources,
        ("mechanism", "solution", "mechanism_path", "source"),
    )
    phase = _find_value(sources, ("phase",))

    if mechanism is None:
        return None, None
    mechanism = _normalize_mechanism(mechanism)
    if phase is not None:
        phase = _require_nonempty_str(phase, "phase")
    return mechanism, phase


def _coerce_single_run_id(value: Any) -> str:
    if value is None:
        raise ConfigError("run_id is required for run-based graphs.")
    if isinstance(value, str):
        return _require_nonempty_str(value, "run_id")
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items = [item for item in value if item is not None]
        if len(items) != 1:
            raise ConfigError("run_id must include exactly one entry.")
        return _require_nonempty_str(items[0], "run_id")
    raise ConfigError("run_id must be a string or single-item sequence.")


def _extract_run_id(graph_cfg: Mapping[str, Any]) -> str:
    run_id: Any = None
    inputs = graph_cfg.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise ConfigError("graphs.inputs must be a mapping.")
        for key in ("run_id", "run", "runs", "run_ids"):
            if key in inputs:
                run_id = inputs.get(key)
                break
    if run_id is None:
        for key in ("run_id", "run", "runs", "run_ids"):
            if key in graph_cfg:
                run_id = graph_cfg.get(key)
                break
    return _coerce_single_run_id(run_id)


def _extract_run_ids(
    graph_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    store: Optional[ArtifactStore] = None,
) -> list[str]:
    inputs = graph_cfg.get("inputs")
    if inputs is not None and not isinstance(inputs, Mapping):
        raise ConfigError("graphs.inputs must be a mapping.")
    run_set_id: Any = None
    explicit_runs_present = False
    if isinstance(inputs, Mapping):
        if "run_set_id" in inputs:
            run_set_id = inputs.get("run_set_id")
        for key in ("run_ids", "runs", "run_id", "run"):
            if key in inputs:
                explicit_runs_present = True
                break
    if run_set_id is None and "run_set_id" in graph_cfg:
        run_set_id = graph_cfg.get("run_set_id")
        explicit_runs_present = explicit_runs_present or any(
            key in graph_cfg for key in ("run_ids", "runs", "run_id", "run")
        )
    if run_set_id is not None:
        if explicit_runs_present:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
        run_set_id = _require_nonempty_str(run_set_id, "run_set_id")
        if store is None:
            raise ConfigError("run_set_id requires a store to be provided.")
        run_ids = load_run_ids_from_run_set(store, run_set_id)
        if not run_ids:
            raise ConfigError("run_set_id resolved to an empty run_ids list.")
        return run_ids
    sources: list[Mapping[str, Any]] = []
    if isinstance(inputs, Mapping):
        sources.append(inputs)
    sources.append(params)
    sources.append(graph_cfg)

    run_ids_value = _find_value(
        sources,
        (
            "run_ids",
            "runs",
            "run_id",
            "run",
            "condition_ids",
            "conditions",
        ),
    )
    if run_ids_value is None:
        raise ConfigError("run_id or run_ids is required for temporal graphs.")
    if isinstance(run_ids_value, str):
        return [_require_nonempty_str(run_ids_value, "run_id")]
    run_ids = _coerce_str_sequence(run_ids_value, "run_ids")
    if not run_ids:
        raise ConfigError("run_ids must include at least one entry.")
    return run_ids


def _extract_coord_names(payload: Mapping[str, Any], coord: str) -> list[str]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    return _coerce_str_sequence(coord_payload.get("data"), f"coords.{coord}.data")


def _extract_coord_values(payload: Mapping[str, Any], coord: str) -> list[Any]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    data = coord_payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ArtifactError(f"coords.{coord}.data must be a sequence.")
    return list(data)


def _extract_time_values(payload: Mapping[str, Any]) -> list[float]:
    values = _extract_coord_values(payload, "time")
    if not values:
        raise ArtifactError("Run dataset missing coords.time.")
    time_values: list[float] = []
    for entry in values:
        try:
            time_values.append(float(entry))
        except (TypeError, ValueError) as exc:
            raise ArtifactError("coords.time entries must be numeric.") from exc
    return time_values


def _extract_time_matrix(
    payload: Mapping[str, Any],
    *,
    var_name: str,
    axis: str,
    axis_names: Sequence[str],
) -> list[list[float]]:
    data_vars = payload.get("data_vars", {})
    if not isinstance(data_vars, Mapping):
        raise ArtifactError("Run dataset data_vars must be a mapping.")
    var_payload = data_vars.get(var_name)
    if not isinstance(var_payload, Mapping):
        raise ArtifactError(f"data_vars.{var_name} must be a mapping.")
    dims = var_payload.get("dims")
    data = var_payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ArtifactError(f"data_vars.{var_name}.dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time", axis]:
        matrix = _coerce_matrix(data, axis)
        for row in matrix:
            if len(row) != len(axis_names):
                raise ArtifactError(
                    f"{var_name} columns mismatch: expected {len(axis_names)}, got {len(row)}."
                )
        return matrix
    if dims_list == [axis, "time"]:
        matrix = _coerce_matrix(data, axis)
        if len(matrix) != len(axis_names):
            raise ArtifactError(
                f"{var_name} rows mismatch: expected {len(axis_names)}, got {len(matrix)}."
            )
        return _transpose_matrix(matrix)
    raise ArtifactError(
        f"data_vars.{var_name}.dims must be [time, {axis}] or [{axis}, time]."
    )


def _coerce_matrix(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactError(f"{label} data must be a sequence of rows.")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(
            row,
            (str, bytes, bytearray),
        ):
            raise ArtifactError(f"{label} rows must be sequences.")
        row_values: list[float] = []
        for entry in row:
            try:
                row_values.append(float(entry))
            except (TypeError, ValueError) as exc:
                raise ArtifactError(f"{label} entries must be numeric.") from exc
        matrix.append(row_values)
    if not matrix:
        raise ArtifactError(f"{label} must contain at least one row.")
    return matrix


def _transpose_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _build_run_bipartite_graph(
    *,
    gas_species: Sequence[str],
    surface_species: Sequence[str],
    reactions: Sequence[str],
) -> dict[str, Any]:
    species_nodes: list[dict[str, Any]] = []
    reaction_nodes: list[dict[str, Any]] = []
    species_node_ids: list[str] = []
    reaction_node_ids: list[str] = []

    for idx, name in enumerate(gas_species):
        node_id = f"species_{name}"
        species_node_ids.append(node_id)
        species_nodes.append(
            {
                "id": node_id,
                "kind": "species",
                "label": name,
                "species": name,
                "phase": "gas",
                "species_index": idx,
            }
        )

    for idx, name in enumerate(surface_species):
        node_id = f"surface_{name}"
        species_node_ids.append(node_id)
        species_nodes.append(
            {
                "id": node_id,
                "kind": "species",
                "label": name,
                "species": name,
                "phase": "surface",
                "species_index": idx,
            }
        )

    for idx, reaction_id in enumerate(reactions):
        node_id = f"reaction_{idx + 1}"
        reaction_node_ids.append(node_id)
        reaction_nodes.append(
            {
                "id": node_id,
                "kind": "reaction",
                "label": reaction_id,
                "reaction_id": reaction_id,
                "reaction_index": idx,
            }
        )

    links: list[dict[str, Any]] = []
    if reaction_node_ids and species_node_ids:
        for r_index, reaction_node_id in enumerate(reaction_node_ids):
            for s_index, species_node_id in enumerate(species_node_ids):
                value = -1.0 if (r_index + s_index) % 2 == 0 else 1.0
                links.append(
                    {
                        "source": species_node_id,
                        "target": reaction_node_id,
                        "stoich": value,
                        "role": "reactant" if value < 0 else "product",
                    }
                )

    graph_attrs = {
        "bipartite": "species-reaction",
        "edge_direction": "species_to_reaction",
        "stoich_sign": "synthetic",
    }
    return {
        "directed": True,
        "multigraph": False,
        "graph": graph_attrs,
        "nodes": species_nodes + reaction_nodes,
        "links": links,
    }


def _sort_time_series(
    time_values: Sequence[float],
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    order = sorted(range(len(time_values)), key=lambda idx: time_values[idx])
    if order == list(range(len(time_values))):
        return list(time_values), [list(row) for row in matrix]
    sorted_time = [float(time_values[idx]) for idx in order]
    sorted_matrix = [list(matrix[idx]) for idx in order]
    return sorted_time, sorted_matrix


def _build_fixed_windows(
    time_values: Sequence[float],
    n_windows: int,
) -> list[TimeWindow]:
    if not time_values:
        raise ConfigError("time values are required for windowing.")
    n_points = len(time_values)
    n_windows = min(max(int(n_windows), 1), n_points)
    if n_windows == 1:
        return [
            TimeWindow(
                index=0,
                start_idx=0,
                end_idx=n_points - 1,
                start_time=float(time_values[0]),
                end_time=float(time_values[-1]),
            )
        ]
    windows: list[TimeWindow] = []
    for idx in range(n_windows):
        start_idx = int(round(idx * (n_points - 1) / n_windows))
        end_idx = int(round((idx + 1) * (n_points - 1) / n_windows))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, n_points - 1)
        windows.append(
            TimeWindow(
                index=idx,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=float(time_values[start_idx]),
                end_time=float(time_values[end_idx]),
            )
        )
    return windows


def _window_indices_from_boundaries(
    time_values: Sequence[float],
    boundaries: Sequence[float],
) -> list[int]:
    n_points = len(time_values)
    indices = [0]
    for boundary in boundaries[1:-1]:
        idx = bisect_left(time_values, boundary)
        idx = max(0, min(idx, n_points - 1))
        indices.append(idx)
    indices.append(n_points - 1)
    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            indices[i] = min(indices[i - 1] + 1, n_points - 1)
    return indices


def _build_windows_from_boundaries(
    time_values: Sequence[float],
    boundaries: Sequence[float],
) -> list[TimeWindow]:
    indices = _window_indices_from_boundaries(time_values, boundaries)
    windows: list[TimeWindow] = []
    for idx in range(len(indices) - 1):
        start_idx = indices[idx]
        end_idx = indices[idx + 1]
        if end_idx < start_idx:
            end_idx = start_idx
        windows.append(
            TimeWindow(
                index=idx,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=float(time_values[start_idx]),
                end_time=float(time_values[end_idx]),
            )
        )
    return windows


def _build_log_time_windows(
    time_values: Sequence[float],
    *,
    n_windows: int,
    base: float,
) -> list[TimeWindow]:
    if np is None:
        raise ConfigError("numpy is required for log_time windowing.")
    if not time_values:
        raise ConfigError("time values are required for windowing.")
    n_points = len(time_values)
    n_windows = min(max(int(n_windows), 1), n_points)
    if n_windows == 1:
        return _build_fixed_windows(time_values, n_windows=1)
    t_min = float(time_values[0])
    t_max = float(time_values[-1])
    if t_max <= t_min:
        return _build_fixed_windows(time_values, n_windows=n_windows)
    positive_times = [t for t in time_values if t > 0.0]
    if not positive_times:
        return _build_fixed_windows(time_values, n_windows=n_windows)
    t_start = min(positive_times)
    if t_start <= 0.0 or t_start >= t_max:
        return _build_fixed_windows(time_values, n_windows=n_windows)
    if base <= 0.0:
        raise ConfigError("windowing.base must be positive.")
    log_start = math.log(t_start, base)
    log_end = math.log(t_max, base)
    if log_end <= log_start:
        return _build_fixed_windows(time_values, n_windows=n_windows)
    boundaries = np.logspace(log_start, log_end, n_windows + 1, base=base).tolist()
    boundaries[0] = t_min
    boundaries[-1] = t_max
    return _build_windows_from_boundaries(time_values, boundaries)


def _build_event_windows(
    time_values: Sequence[float],
    activity: Sequence[float],
    *,
    n_windows: int,
) -> list[TimeWindow]:
    if np is None:
        raise ConfigError("numpy is required for event_based windowing.")
    if not time_values:
        raise ConfigError("time values are required for windowing.")
    n_points = len(time_values)
    n_windows = min(max(int(n_windows), 1), n_points)
    if len(activity) != n_points:
        raise ConfigError("activity length must match time values.")
    if n_windows == 1:
        return _build_fixed_windows(time_values, n_windows=1)
    cumulative = [0.0] * n_points
    for idx in range(1, n_points):
        dt = float(time_values[idx]) - float(time_values[idx - 1])
        cumulative[idx] = cumulative[idx - 1] + 0.5 * (
            float(activity[idx]) + float(activity[idx - 1])
        ) * dt
    total = cumulative[-1]
    if total <= 0.0:
        return _build_fixed_windows(time_values, n_windows=n_windows)
    boundaries = [float(time_values[0])]
    for window_idx in range(1, n_windows):
        target = total * window_idx / float(n_windows)
        idx = bisect_left(cumulative, target)
        idx = max(0, min(idx, n_points - 1))
        boundaries.append(float(time_values[idx]))
    boundaries.append(float(time_values[-1]))
    return _build_windows_from_boundaries(time_values, boundaries)


def _normalize_windowing_cfg(
    params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    windowing = _find_value(
        [params, graph_cfg],
        ("windowing", "windows", "window"),
    )
    if windowing is None:
        return {"type": "log_time", "count": 4, "base": 10.0}
    if isinstance(windowing, str):
        return {"type": windowing, "count": 4, "base": 10.0}
    if not isinstance(windowing, Mapping):
        raise ConfigError("windowing must be a mapping or string.")
    windowing_cfg = dict(windowing)
    window_type = windowing_cfg.get("type") or windowing_cfg.get("mode") or windowing_cfg.get(
        "kind"
    )
    if window_type is None:
        window_type = "log_time"
    if not isinstance(window_type, str) or not window_type.strip():
        raise ConfigError("windowing.type must be a non-empty string.")
    count = windowing_cfg.get("count")
    if count is None:
        count = windowing_cfg.get("windows")
    if count is None:
        count = windowing_cfg.get("n_windows")
    if count is None:
        count = windowing_cfg.get("n")
    count = _coerce_positive_int(count, "windowing.count", default=4)
    base = windowing_cfg.get("base", 10.0)
    base = _coerce_float(base, "windowing.base")
    return {"type": window_type.strip(), "count": count, "base": base, "raw": windowing_cfg}


def _normalize_aggregation_cfg(
    params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
) -> dict[str, float]:
    agg_cfg = _find_value(
        [params, graph_cfg],
        ("aggregation", "aggregate", "condition_aggregate"),
    )
    if agg_cfg is None:
        agg_cfg = {}
    if not isinstance(agg_cfg, Mapping):
        raise ConfigError("aggregation must be a mapping.")
    agg_cfg = dict(agg_cfg)
    mix_cfg = agg_cfg.get("mix") or agg_cfg.get("weights") or agg_cfg
    if not isinstance(mix_cfg, Mapping):
        raise ConfigError("aggregation.mix must be a mapping when provided.")
    mean_weight = _coerce_optional_float(
        mix_cfg.get("mean", mix_cfg.get("mean_weight")), "aggregation.mean"
    )
    p95_weight = _coerce_optional_float(
        mix_cfg.get("p95", mix_cfg.get("p95_weight")), "aggregation.p95"
    )
    if mean_weight is None and p95_weight is None:
        mean_weight = 0.5
        p95_weight = 0.5
    if mean_weight is None:
        mean_weight = 0.0
    if p95_weight is None:
        p95_weight = 0.0
    total = mean_weight + p95_weight
    if total <= 0.0:
        raise ConfigError("aggregation weights must sum to a positive value.")
    return {"mean": mean_weight / total, "p95": p95_weight / total}


def _normalize_sparsify_cfg(
    params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
) -> dict[str, Optional[float]]:
    sparsify_cfg = _find_value(
        [params, graph_cfg],
        ("sparsify", "sparsification", "sparse"),
    )
    if sparsify_cfg is None:
        return {"quantile": None, "min_value": None}
    if not isinstance(sparsify_cfg, Mapping):
        raise ConfigError("sparsify must be a mapping.")
    sparsify_cfg = dict(sparsify_cfg)
    quantile = _coerce_quantile(
        sparsify_cfg.get("quantile", sparsify_cfg.get("q")),
        "sparsify.quantile",
    )
    min_value = _coerce_optional_float(
        sparsify_cfg.get("min_value", sparsify_cfg.get("threshold")),
        "sparsify.min_value",
    )
    return {"quantile": quantile, "min_value": min_value}


def _normalize_rop_cfg(
    params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    rop_cfg = _find_value([params, graph_cfg], ("rop", "rop_cfg", "rop_var"))
    if rop_cfg is None:
        rop_cfg = {}
    if isinstance(rop_cfg, str):
        return {"var": rop_cfg, "use_abs": True}
    if not isinstance(rop_cfg, Mapping):
        raise ConfigError("rop config must be a mapping or string.")
    rop_cfg = dict(rop_cfg)
    var_name = (
        rop_cfg.get("var")
        or rop_cfg.get("name")
        or rop_cfg.get("data_var")
        or rop_cfg.get("rop_var")
        or "rop_net"
    )
    if not isinstance(var_name, str) or not var_name.strip():
        raise ConfigError("rop.var must be a non-empty string.")
    use_abs = rop_cfg.get("use_abs")
    if use_abs is None:
        use_abs = True
    if not isinstance(use_abs, bool):
        raise ConfigError("rop.use_abs must be a boolean.")
    return {"var": var_name.strip(), "use_abs": use_abs}


def _extract_reaction_map_cfg(
    params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    map_cfg = _find_value(
        [params, graph_cfg],
        ("reaction_species_map", "reaction_map", "stoich_map"),
    )
    if map_cfg is None:
        return None
    if not isinstance(map_cfg, Mapping):
        raise ConfigError("reaction_species_map must be a mapping.")
    return dict(map_cfg)


def _build_reaction_map_from_cfg(
    reaction_ids: Sequence[str],
    species: Sequence[str],
    map_cfg: Mapping[str, Any],
) -> list[list[tuple[int, float]]]:
    reaction_index = {rid: idx for idx, rid in enumerate(reaction_ids)}
    species_index = {name: idx for idx, name in enumerate(species)}
    reaction_map: list[list[tuple[int, float]]] = [
        [] for _ in range(len(reaction_ids))
    ]

    for key, entry in map_cfg.items():
        r_idx: Optional[int] = None
        if isinstance(key, int):
            r_idx = key
        elif isinstance(key, str):
            if key in reaction_index:
                r_idx = reaction_index[key]
            elif key.isdigit():
                r_idx = int(key)
        if r_idx is None or r_idx < 0 or r_idx >= len(reaction_ids):
            raise ConfigError(f"Unknown reaction identifier in reaction_map: {key!r}.")
        if isinstance(entry, Mapping):
            species_entries = entry.items()
        elif isinstance(entry, Sequence) and not isinstance(
            entry, (str, bytes, bytearray)
        ):
            species_entries = [(name, 1.0) for name in entry]
        else:
            raise ConfigError(
                f"reaction_map entry for {key!r} must be a mapping or sequence."
            )
        for species_name, coeff in species_entries:
            if species_name not in species_index:
                raise ConfigError(
                    f"reaction_map species {species_name!r} not found in species list."
                )
            coeff_value = _coerce_float(coeff, f"reaction_map[{key}]")
            reaction_map[r_idx].append(
                (species_index[species_name], abs(coeff_value))
            )
    return reaction_map


def _build_reaction_map_from_stoich(
    stoich: StoichResult,
    reaction_ids: Sequence[str],
    species: Sequence[str],
) -> list[list[tuple[int, float]]]:
    reaction_index = {rid: idx for idx, rid in enumerate(reaction_ids)}
    species_index = {name: idx for idx, name in enumerate(species)}
    reaction_map: list[list[tuple[int, float]]] = [
        [] for _ in range(len(reaction_ids))
    ]
    for s_idx, r_idx, coeff in _iter_stoich_entries(stoich):
        if r_idx < 0 or r_idx >= len(stoich.reaction_ids):
            continue
        reaction_id = stoich.reaction_ids[r_idx]
        if reaction_id not in reaction_index:
            continue
        target_idx = reaction_index[reaction_id]
        species_name = stoich.species[s_idx]
        if species_name not in species_index:
            raise ConfigError(
                f"Species {species_name!r} not found in run species list."
            )
        reaction_map[target_idx].append(
            (species_index[species_name], abs(float(coeff)))
        )
    return reaction_map


def _fallback_reaction_map(
    reaction_ids: Sequence[str],
    species: Sequence[str],
) -> list[list[tuple[int, float]]]:
    species_indices = [(idx, 1.0) for idx in range(len(species))]
    return [list(species_indices) for _ in reaction_ids]


def _precompute_reaction_pairs(
    reaction_map: Sequence[Sequence[tuple[int, float]]],
) -> list[list[tuple[int, int, float]]]:
    pairs: list[list[tuple[int, int, float]]] = []
    for entries in reaction_map:
        pair_list: list[tuple[int, int, float]] = []
        if len(entries) >= 2:
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    idx_i, coeff_i = entries[i]
                    idx_j, coeff_j = entries[j]
                    pair_list.append((idx_i, idx_j, coeff_i * coeff_j))
        pairs.append(pair_list)
    return pairs


def _integrate_window_fluxes(
    time_values: Sequence[float],
    rop_matrix: Sequence[Sequence[float]],
    window: TimeWindow,
) -> list[float]:
    if np is None:
        raise ConfigError("numpy is required to integrate fluxes.")
    if window.end_idx <= window.start_idx:
        return [0.0 for _ in rop_matrix[0]]
    times = np.asarray(time_values[window.start_idx : window.end_idx + 1], dtype=float)
    values = np.asarray(
        rop_matrix[window.start_idx : window.end_idx + 1], dtype=float
    )
    if values.ndim != 2:
        raise ConfigError("ROP matrix must be 2D.")
    dt = np.diff(times)
    avg = 0.5 * (values[1:, :] + values[:-1, :])
    return np.sum(avg * dt[:, None], axis=0).tolist()


def _build_species_adjacency(
    fluxes: Sequence[float],
    reaction_pairs: Sequence[Sequence[tuple[int, int, float]]],
    *,
    n_species: int,
    use_abs: bool,
) -> Any:
    if np is None:
        raise ConfigError("numpy is required to build adjacency.")
    matrix = np.zeros((n_species, n_species), dtype=float)
    for r_idx, pairs in enumerate(reaction_pairs):
        if not pairs:
            continue
        flux = float(fluxes[r_idx])
        if use_abs:
            flux = abs(flux)
        if flux == 0.0:
            continue
        for i_idx, j_idx, weight in pairs:
            value = flux * weight
            matrix[i_idx, j_idx] += value
            matrix[j_idx, i_idx] += value
    return matrix


def _aggregate_condition_matrices(
    matrices: Sequence[Any],
    *,
    mean_weight: float,
    p95_weight: float,
) -> Any:
    if np is None:
        raise ConfigError("numpy is required for aggregation.")
    if not matrices:
        raise ConfigError("No matrices to aggregate.")
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=float)
    stacked = np.stack(matrices, axis=0).astype(float)
    mean = np.mean(stacked, axis=0)
    p95 = np.percentile(stacked, 95, axis=0)
    return mean_weight * mean + p95_weight * p95


def _sparsify_matrix(
    matrix: Any,
    *,
    quantile: Optional[float],
    min_value: Optional[float],
) -> Any:
    if np is None:
        raise ConfigError("numpy is required for sparsification.")
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return values
    threshold = None
    if quantile is not None:
        nonzero = np.abs(values[values != 0.0])
        if nonzero.size > 0:
            threshold = float(np.quantile(nonzero, quantile))
    if min_value is not None:
        min_value = abs(float(min_value))
        threshold = min_value if threshold is None else max(threshold, min_value)
    if threshold is None:
        return values
    mask = np.abs(values) >= threshold
    return values * mask


def _dense_to_csr(matrix: Any) -> tuple[Any, Any, Any, tuple[int, int]]:
    if np is None:
        raise ConfigError("numpy is required to build CSR matrices.")
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ConfigError("Adjacency matrix must be 2D.")
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for row in array:
        for col_idx, value in enumerate(row):
            if value != 0.0:
                data.append(float(value))
                indices.append(col_idx)
        indptr.append(len(data))
    return (
        np.asarray(data, dtype=float),
        np.asarray(indices, dtype=int),
        np.asarray(indptr, dtype=int),
        (array.shape[0], array.shape[1]),
    )


def _write_csr_npz(path: Path, matrix: Any) -> None:
    if sp is not None:
        csr = sp.csr_matrix(matrix)
        sp.save_npz(path, csr)
        return
    if np is None:
        raise ConfigError("numpy is required to save CSR matrices.")
    data, indices, indptr, shape = _dense_to_csr(matrix)
    np.savez(
        path,
        data=data,
        indices=indices,
        indptr=indptr,
        shape=np.asarray(shape, dtype=int),
        format="csr",
    )



def _reaction_equations(solution: Any, count: int) -> list[str]:
    if count <= 0:
        return []
    try:
        equations = list(solution.reaction_equations())
        if len(equations) == count:
            return [str(entry) for entry in equations]
        logger.warning(
            "Reaction equations length mismatch: expected %d, got %d.",
            count,
            len(equations),
        )
    except Exception as exc:
        logger.warning("Reaction equations lookup failed: %s", exc)
    labels: list[str] = []
    had_failure = False
    for idx in range(count):
        try:
            labels.append(str(solution.reaction_equation(idx)))
        except Exception:
            had_failure = True
            labels.append(f"R{idx + 1}")
    if had_failure:
        logger.warning("Reaction equation lookup failed; using fallback labels.")
    return labels


def _reaction_ids(solution: Any, count: int) -> list[str]:
    ids: list[str] = []
    for idx in range(count):
        reaction_id: Optional[str] = None
        try:
            reaction = solution.reaction(idx)
            reaction_id = getattr(reaction, "id", None)
        except Exception:
            reaction = None
        if reaction_id:
            ids.append(str(reaction_id))
        else:
            ids.append(f"R{idx + 1}")
    return ids


def _normalize_phase_label(value: str) -> str:
    label = value.strip().lower()
    if not label:
        return "unknown"
    if "gas" in label:
        return "gas"
    if "surf" in label or "surface" in label or "interface" in label:
        return "surface"
    if "solid" in label or "bulk" in label:
        return "solid"
    return "unknown"


def _infer_phase_label(phase: Optional[str], solution: Any) -> tuple[str, bool]:
    if phase:
        normalized = _normalize_phase_label(str(phase))
        if normalized != "unknown":
            return normalized, False
    n_sites = getattr(solution, "n_sites", None)
    if isinstance(n_sites, (int, float)) and n_sites:
        return "surface", True
    for attr in ("phase_name", "name", "thermo_model", "transport_model"):
        value = getattr(solution, attr, None)
        if value:
            normalized = _normalize_phase_label(str(value))
            if normalized != "unknown":
                return normalized, True
    return "unknown", True


def _coerce_element_counts(
    composition: Any,
) -> tuple[dict[str, float], bool]:
    if not isinstance(composition, Mapping):
        return {}, True
    elements: dict[str, float] = {}
    inferred = False
    for element, count in composition.items():
        if element is None:
            inferred = True
            continue
        try:
            value = float(count)
        except (TypeError, ValueError):
            inferred = True
            continue
        if value == 0.0:
            continue
        elements[str(element)] = value
    if not elements:
        inferred = True
    return elements, inferred


def _formula_element_order(elements: Mapping[str, float]) -> list[str]:
    names = sorted(elements.keys())
    if "C" in elements:
        order = ["C"]
        if "H" in elements:
            order.append("H")
        for name in names:
            if name not in ("C", "H"):
                order.append(name)
        return order
    return names


def _format_count(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-8):
        int_value = int(round(value))
        return "" if int_value == 1 else str(int_value)
    return "" if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-8) else f"{value:g}"


def _format_formula(elements: Mapping[str, float]) -> str:
    if not elements:
        return ""
    parts: list[str] = []
    for element in _formula_element_order(elements):
        count = elements[element]
        parts.append(f"{element}{_format_count(count)}")
    return "".join(parts)


def _elements_vector(elements: Mapping[str, float]) -> list[list[Any]]:
    return [[element, elements[element]] for element in _formula_element_order(elements)]


def _parse_charge_from_name(name: str) -> Optional[int]:
    match = re.search(r"([+-]+)$", name)
    if match:
        signs = match.group(1)
        return signs.count("+") - signs.count("-")
    match = re.search(r"([+-])(\d+)$", name)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        return sign * int(match.group(2))
    return None


def _extract_charge(species: Any, name: str) -> tuple[Optional[float], bool]:
    charge_value: Optional[Any] = None
    if species is not None:
        value = getattr(species, "charge", None)
        if callable(value):
            try:
                charge_value = value()
            except Exception:
                charge_value = None
        else:
            charge_value = value
    if charge_value is not None:
        try:
            charge = float(charge_value)
        except (TypeError, ValueError):
            charge = None
        if charge is not None:
            if math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
                charge = 0.0
            return charge, False
    parsed = _parse_charge_from_name(name)
    if parsed is not None:
        return float(parsed), True
    return None, True


def _radical_heuristic(name: str) -> Optional[bool]:
    if name.endswith(".") or "RAD" in name or "rad" in name:
        return True
    return None


def _classify_state(name: str, charge: Optional[float]) -> tuple[str, bool]:
    if charge is None:
        return "unknown", True
    if not math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "ion", False
    if _radical_heuristic(name):
        return "radical", True
    return "neutral", True


def annotate_species(
    solution: Any,
    *,
    phase: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Annotate species with formula/elements/charge/state/phase metadata."""
    species_names = list(getattr(solution, "species_names", []))
    if not species_names:
        raise ConfigError("mechanism has no species.")
    phase_label, phase_inferred = _infer_phase_label(phase, solution)

    annotations: dict[str, dict[str, Any]] = {}
    for name in species_names:
        species_obj = None
        species_attr = getattr(solution, "species", None)
        if callable(species_attr):
            try:
                species_obj = species_attr(name)
            except Exception:
                species_obj = None
        composition = getattr(species_obj, "composition", None)
        elements, elements_inferred = _coerce_element_counts(composition)
        formula = _format_formula(elements)
        charge, charge_inferred = _extract_charge(species_obj, name)
        state, state_inferred = _classify_state(name, charge)

        inferred_fields: list[str] = []
        if elements_inferred:
            inferred_fields.extend(["elements", "formula"])
        if charge_inferred:
            inferred_fields.append("charge")
        if phase_inferred:
            inferred_fields.append("phase")
        if state_inferred:
            inferred_fields.append("state")

        annotations[name] = {
            "formula": formula or None,
            "elements": elements,
            "elements_vector": _elements_vector(elements),
            "charge": charge,
            "phase": phase_label,
            "state": state,
            "is_inferred": bool(inferred_fields),
            "inferred_fields": sorted(set(inferred_fields)),
        }
    return annotations


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(value)
    return text if text else None


def _normalize_reaction_type(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return "unknown"
    lowered = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    if not lowered:
        return "unknown"
    if "plog" in lowered:
        return "plog"
    if "chebyshev" in lowered:
        return "chebyshev"
    if "falloff" in lowered or "lindemann" in lowered or "troe" in lowered:
        return "falloff"
    if "chem" in lowered and "activ" in lowered:
        return "chemically-activated"
    if "three" in lowered and "body" in lowered:
        return "three-body"
    if "arrhenius" in lowered or "blowers" in lowered:
        return "elementary"
    if "pressure" in lowered and ("dep" in lowered or "depend" in lowered):
        return "pressure-dependent"
    if "pdep" in lowered:
        return "pressure-dependent"
    if "electro" in lowered:
        return "electrochemical"
    if "adsorption" in lowered:
        return "adsorption"
    if "desorption" in lowered:
        return "desorption"
    if "surface" in lowered:
        return "surface"
    if "interface" in lowered:
        return "interface"
    if "elementary" in lowered:
        return "elementary"
    return "unknown"


def _reaction_type_from_attr(reaction: Any) -> tuple[Optional[str], Optional[str]]:
    for attr in ("reaction_type", "type"):
        value = getattr(reaction, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        text = _as_text(value)
        if text:
            return text, f"attr:{attr}"
    return None, None


def _has_third_body(reaction: Any) -> bool:
    if getattr(reaction, "third_body", None) is not None:
        return True
    efficiencies = getattr(reaction, "efficiencies", None)
    if isinstance(efficiencies, Mapping) and efficiencies:
        return True
    if getattr(reaction, "default_efficiency", None) is not None:
        return True
    return False


def _has_falloff(reaction: Any) -> bool:
    for attr in ("falloff", "falloff_params", "falloff_parameters", "falloff_coeffs"):
        if getattr(reaction, attr, None) is not None:
            return True
    return False


def _classify_reaction_type(
    reaction: Any,
    *,
    phase_label: Optional[str],
) -> tuple[str, str, bool]:
    explicit, source = _reaction_type_from_attr(reaction)
    fallback_reason: Optional[str] = None
    if explicit:
        normalized = _normalize_reaction_type(explicit)
        if normalized != "unknown":
            return normalized, f"{source}:{explicit}", False
        fallback_reason = f"{source}:{explicit}"
    class_name = type(reaction).__name__
    normalized = _normalize_reaction_type(class_name)
    if normalized != "unknown":
        return normalized, f"class_name:{class_name}", True
    rate = getattr(reaction, "rate", None)
    if rate is not None:
        rate_class = type(rate).__name__
        normalized = _normalize_reaction_type(rate_class)
        if normalized != "unknown":
            return normalized, f"rate_class:{rate_class}", True
        for attr in ("type", "rate_type"):
            value = getattr(rate, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            normalized = _normalize_reaction_type(value)
            if normalized != "unknown":
                return normalized, f"rate_attr:{attr}:{value}", True
    if _has_falloff(reaction):
        return "falloff", "heuristic:falloff", True
    if _has_third_body(reaction):
        return "three-body", "heuristic:third_body", True
    if phase_label == "surface":
        return "surface", "heuristic:phase", True
    if fallback_reason:
        return "unknown", f"unknown: {fallback_reason}", True
    return "unknown", "unknown: no reaction type match", True


def _extract_bool_flag(reaction: Any, attr: str) -> tuple[Optional[bool], bool]:
    value = getattr(reaction, attr, None)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    if value is None:
        return None, True
    if isinstance(value, bool):
        return value, False
    if isinstance(value, (int, float)):
        return bool(value), True
    return None, True


def _estimate_reaction_order(reaction: Any) -> tuple[Optional[float], str, bool]:
    reactants = getattr(reaction, "reactants", None)
    if not isinstance(reactants, Mapping):
        return None, "missing_reactants", True
    if not reactants:
        return None, "empty_reactants", True
    total = 0.0
    valid = 0
    invalid = False
    for coeff in reactants.values():
        try:
            value = float(coeff)
        except (TypeError, ValueError):
            invalid = True
            continue
        total += value
        valid += 1
    if valid == 0:
        return None, "non_numeric_reactants", True
    if invalid:
        return total, "partial_stoich_sum", True
    return total, "stoich_sum", True


def annotate_reactions(
    solution: Any,
    *,
    phase: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Annotate reactions with type/order/reversible/duplicate metadata."""
    n_reactions = int(getattr(solution, "n_reactions", 0) or 0)
    if n_reactions <= 0:
        raise ConfigError("mechanism has no reactions.")
    phase_label, _ = _infer_phase_label(phase, solution)
    reaction_ids = _reaction_ids(solution, n_reactions)
    annotations: dict[str, dict[str, Any]] = {}
    for idx in range(n_reactions):
        reaction = solution.reaction(idx)
        reaction_id = reaction_ids[idx]
        reaction_type, type_reason, type_inferred = _classify_reaction_type(
            reaction,
            phase_label=phase_label,
        )
        order, order_source, order_inferred = _estimate_reaction_order(reaction)
        reversible, reversible_inferred = _extract_bool_flag(reaction, "reversible")
        duplicate, duplicate_inferred = _extract_bool_flag(reaction, "duplicate")

        inferred_fields: list[str] = []
        if type_inferred:
            inferred_fields.append("reaction_type")
        if order_inferred:
            inferred_fields.append("order")
        if reversible_inferred:
            inferred_fields.append("reversible")
        if duplicate_inferred:
            inferred_fields.append("duplicate")

        annotations[reaction_id] = {
            "reaction_index": idx,
            "reaction_type": reaction_type,
            "reaction_type_reason": type_reason,
            "order": order,
            "order_source": order_source,
            "reversible": reversible,
            "duplicate": duplicate,
            "is_inferred": bool(inferred_fields),
            "inferred_fields": sorted(set(inferred_fields)),
        }
    return annotations


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a float, got {value!r}.") from exc


def _stoich_entries(
    solution: Any,
    species_index: Mapping[str, int],
    n_reactions: int,
) -> dict[tuple[int, int], float]:
    entries: dict[tuple[int, int], float] = {}
    for rxn_idx in range(n_reactions):
        reaction = solution.reaction(rxn_idx)
        reactants = getattr(reaction, "reactants", None)
        products = getattr(reaction, "products", None)
        if not isinstance(reactants, Mapping) or not isinstance(products, Mapping):
            raise ConfigError("Reaction stoichiometry must be mappings.")
        for name, coeff in reactants.items():
            if name not in species_index:
                raise ConfigError(f"Unknown reactant species: {name}")
            value = -_as_float(coeff, f"reactants[{name}]")
            key = (species_index[name], rxn_idx)
            entries[key] = entries.get(key, 0.0) + value
        for name, coeff in products.items():
            if name not in species_index:
                raise ConfigError(f"Unknown product species: {name}")
            value = _as_float(coeff, f"products[{name}]")
            key = (species_index[name], rxn_idx)
            entries[key] = entries.get(key, 0.0) + value
    return entries


def build_stoich(solution: Any) -> StoichResult:
    """Build a species x reaction stoichiometric matrix from a Cantera Solution."""
    species_names = list(getattr(solution, "species_names", []))
    if not species_names:
        raise ConfigError("mechanism has no species.")
    n_reactions = int(getattr(solution, "n_reactions", 0) or 0)
    if n_reactions <= 0:
        raise ConfigError("mechanism has no reactions.")

    reaction_equations = _reaction_equations(solution, n_reactions)
    reaction_ids = _reaction_ids(solution, n_reactions)

    species_index = {name: idx for idx, name in enumerate(species_names)}
    entries = _stoich_entries(solution, species_index, n_reactions)
    rows = [key[0] for key in entries]
    cols = [key[1] for key in entries]
    data = [entries[key] for key in entries]

    if sp is not None:
        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(species_names), n_reactions),
            dtype=float,
        )
        fmt = "scipy.sparse.coo"
    else:
        if np is None:
            raise ConfigError("numpy is required to build dense stoichiometry.")
        matrix = np.zeros((len(species_names), n_reactions), dtype=float)
        for (row, col), value in entries.items():
            matrix[row, col] = value
        fmt = "numpy.ndarray"

    return StoichResult(
        matrix=matrix,
        species=species_names,
        reaction_ids=reaction_ids,
        reaction_equations=reaction_equations,
        format=fmt,
    )


def _iter_stoich_entries(result: StoichResult) -> list[tuple[int, int, float]]:
    matrix = result.matrix
    entries: list[tuple[int, int, float]] = []
    if sp is not None and hasattr(matrix, "tocoo"):
        coo = matrix.tocoo()
        for row, col, value in zip(coo.row, coo.col, coo.data):
            if value != 0:
                entries.append((int(row), int(col), float(value)))
        return entries
    if np is not None and isinstance(matrix, np.ndarray):
        rows, cols = np.nonzero(matrix)
        for row, col in zip(rows, cols):
            value = float(matrix[row, col])
            if value != 0.0:
                entries.append((int(row), int(col), value))
        return entries
    try:
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                if value:
                    entries.append((row_idx, col_idx, float(value)))
    except TypeError as exc:
        raise ConfigError(
            "Unsupported stoichiometric matrix type for bipartite graph."
        ) from exc
    return entries


def build_bipartite_graph(
    result: StoichResult,
    *,
    species_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    reaction_annotations: Optional[Mapping[Any, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a node-link bipartite graph from the stoichiometric matrix."""
    species_nodes: list[dict[str, Any]] = []
    reaction_nodes: list[dict[str, Any]] = []
    species_node_ids: list[str] = []
    reaction_node_ids: list[str] = []

    for idx, name in enumerate(result.species):
        node_id = f"species_{name}"
        species_node_ids.append(node_id)
        node = {
            "id": node_id,
            "kind": "species",
            "label": name,
            "species_index": idx,
        }
        if species_annotations and name in species_annotations:
            annotation = species_annotations.get(name)
            if isinstance(annotation, Mapping):
                node.update(annotation)
        species_nodes.append(node)

    for idx, reaction_id in enumerate(result.reaction_ids):
        node_id = f"reaction_{idx + 1}"
        reaction_node_ids.append(node_id)
        node = {
            "id": node_id,
            "kind": "reaction",
            "reaction_id": reaction_id,
            "reaction_index": idx,
            "reaction_equation": result.reaction_equations[idx],
        }
        if reaction_annotations:
            annotation = None
            if reaction_id in reaction_annotations:
                annotation = reaction_annotations.get(reaction_id)
            elif idx in reaction_annotations:
                annotation = reaction_annotations.get(idx)
            elif str(idx) in reaction_annotations:
                annotation = reaction_annotations.get(str(idx))
            if isinstance(annotation, Mapping):
                node.update(annotation)
        reaction_nodes.append(node)

    links: list[dict[str, Any]] = []
    for row, col, value in _iter_stoich_entries(result):
        role = "reactant" if value < 0 else "product"
        links.append(
            {
                "source": species_node_ids[row],
                "target": reaction_node_ids[col],
                "stoich": value,
                "role": role,
            }
        )

    graph_attrs = {
        "bipartite": "species-reaction",
        "edge_direction": "species_to_reaction",
        "stoich_sign": "reactants_negative_products_positive",
    }

    nodes = species_nodes + reaction_nodes
    if nx is not None:
        graph = nx.DiGraph()
        graph.graph.update(graph_attrs)
        for node in nodes:
            node_id = node["id"]
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for link in links:
            source = link["source"]
            target = link["target"]
            attrs = {
                key: value
                for key, value in link.items()
                if key not in {"source", "target"}
            }
            graph.add_edge(source, target, **attrs)
        data = nx.readwrite.json_graph.node_link_data(graph)
        if "links" not in data and "edges" in data:
            data["links"] = data["edges"]
        return data

    return {
        "directed": True,
        "multigraph": False,
        "graph": graph_attrs,
        "nodes": nodes,
        "links": links,
    }


def _edge_weight(
    link: Mapping[str, Any],
    *,
    weight_key: Optional[str],
    use_abs_weights: bool,
) -> float:
    value: Any = None
    if weight_key:
        value = link.get(weight_key)
    else:
        for key in ("weight", "value", "stoich"):
            if key in link:
                value = link.get(key)
                break
    if value is None:
        weight = 1.0
    else:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 1.0
    if not math.isfinite(weight):
        raise ConfigError("edge weight must be finite.")
    if use_abs_weights:
        weight = abs(weight)
    if weight < 0.0:
        raise ConfigError("edge weight must be non-negative.")
    return weight


def build_laplacian(
    graph_payload: Mapping[str, Any],
    *,
    normalized: bool = False,
    weight_key: Optional[str] = None,
    use_abs_weights: bool = True,
    symmetrize: bool = True,
) -> LaplacianResult:
    """Build degree and Laplacian matrices from a node-link graph payload."""
    if np is None:
        raise ConfigError("numpy is required to build Laplacian matrices.")
    if not isinstance(graph_payload, Mapping):
        raise ConfigError("graph payload must be a mapping.")

    if "nodes" in graph_payload and ("links" in graph_payload or "edges" in graph_payload):
        graph_data = dict(graph_payload)
    else:
        graph_data, _ = _extract_node_link_payload(graph_payload)

    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    nodes, node_map = _normalize_nodes(nodes_raw)
    links = _normalize_links(links_raw)

    node_ids = [node["id"] for node in nodes]
    edge_nodes = set()
    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is not None:
            edge_nodes.add(source)
        if target is not None:
            edge_nodes.add(target)
    missing_nodes = sorted(edge_nodes.difference(node_map))
    for node_id in missing_nodes:
        node_map[node_id] = {"id": node_id}
        node_ids.append(node_id)

    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    size = len(node_ids)
    adjacency = np.zeros((size, size), dtype=float)

    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is None or target is None:
            continue
        weight = _edge_weight(
            link,
            weight_key=weight_key,
            use_abs_weights=use_abs_weights,
        )
        if weight == 0.0:
            continue
        row = index.get(source)
        col = index.get(target)
        if row is None or col is None:
            continue
        adjacency[row, col] += weight
        if symmetrize and row != col:
            adjacency[col, row] += weight

    degree = adjacency.sum(axis=1)
    laplacian = -adjacency
    np.fill_diagonal(laplacian, degree)

    normalized_laplacian: Optional[Any] = None
    normalized_format: Optional[str] = None
    if normalized:
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0.0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        scaled = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
        norm = -scaled
        np.fill_diagonal(norm, 1.0)
        if np.any(~nonzero):
            zero_idx = np.where(~nonzero)[0]
            for idx in zero_idx:
                norm[idx, idx] = 0.0
        normalized_laplacian = norm
        normalized_format = "numpy.ndarray"

    return LaplacianResult(
        laplacian=laplacian,
        degree=degree,
        nodes=node_ids,
        normalized_laplacian=normalized_laplacian,
        format="numpy.ndarray",
        normalized_format=normalized_format,
    )


def _load_solution(mechanism: str, phase: Optional[str]) -> Any:
    if ct is None:
        raise ConfigError("Cantera is required to build stoichiometric graphs.")
    if phase is None:
        return ct.Solution(mechanism)
    return ct.Solution(mechanism, phase)


def _write_stoich_npz(path: Path, result: StoichResult) -> None:
    if sp is not None and hasattr(result.matrix, "tocsr"):
        sp.save_npz(path, result.matrix.tocsr())
        return
    if np is None:
        raise ConfigError("numpy is required to write stoich.npz without scipy.")
    np.savez_compressed(path, stoich=result.matrix)


def _write_laplacian_npz(path: Path, result: LaplacianResult) -> None:
    if np is None:
        raise ConfigError("numpy is required to write laplacian.npz.")
    payload: dict[str, Any] = {
        "laplacian": np.asarray(result.laplacian, dtype=float),
        "degree": np.asarray(result.degree, dtype=float),
    }
    if result.normalized_laplacian is not None:
        payload["laplacian_norm"] = np.asarray(
            result.normalized_laplacian, dtype=float
        )
    np.savez_compressed(path, **payload)


def _graph_metadata(
    result: StoichResult,
    *,
    mechanism: str,
    phase: Optional[str],
    bipartite_graph: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    reactions = [
        {"id": rid, "equation": eq}
        for rid, eq in zip(result.reaction_ids, result.reaction_equations)
    ]
    metadata: dict[str, Any] = {
        "kind": "stoichiometric_matrix",
        "shape": [len(result.species), len(result.reaction_ids)],
        "species": list(result.species),
        "reactions": reactions,
        "stoich": {"path": "stoich.npz", "format": result.format},
        "source": {"mechanism": mechanism},
    }
    if phase is not None:
        metadata["source"]["phase"] = phase
    if bipartite_graph is not None:
        links_key = "links" if "links" in bipartite_graph else "edges"
        metadata["bipartite"] = {
            "format": "node_link",
            "node_count": len(bipartite_graph.get("nodes", [])),
            "edge_count": len(bipartite_graph.get(links_key, [])),
            "data": bipartite_graph,
        }
    return metadata


def _extract_analysis_cfg(graph_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("analysis", "analytics", "graph_analysis"):
        if key in graph_cfg:
            value = graph_cfg.get(key)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(graph_cfg)


def _extract_laplacian_cfg(graph_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("laplacian", "graph_laplacian"):
        if key in graph_cfg:
            value = graph_cfg.get(key)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(graph_cfg)


def _extract_graph_id(analysis_cfg: Mapping[str, Any]) -> str:
    graph_id: Any = None
    inputs = analysis_cfg.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("graph_id", "graph", "id", "source"):
            if key in inputs:
                graph_id = inputs.get(key)
                break
    if graph_id is None:
        for key in ("graph_id", "graph", "id", "source"):
            if key in analysis_cfg:
                graph_id = analysis_cfg.get(key)
                break
    if graph_id is None:
        graph_section = analysis_cfg.get("graph")
        if isinstance(graph_section, Mapping):
            graph_id = graph_section.get("id") or graph_section.get("graph_id")
    return _require_nonempty_str(graph_id, "graph_id")


def _coerce_positive_int(
    value: Any,
    label: str,
    *,
    default: int,
) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer.") from exc
    if number <= 0:
        raise ConfigError(f"{label} must be a positive integer.")
    return number


def _coerce_bool(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise ConfigError(f"{label} must be a boolean.")


def _coerce_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a float.") from exc


def _coerce_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _coerce_float(value, label)


def _coerce_quantile(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    quantile = _coerce_float(value, label)
    if quantile < 0.0 or quantile > 1.0:
        raise ConfigError(f"{label} must be between 0 and 1.")
    return quantile


def _load_graph_payload(path: Path) -> dict[str, Any]:
    graph_path = path / "graph.json"
    if not graph_path.exists():
        raise ConfigError(f"graph.json not found in {path}.")
    try:
        payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")
    return dict(payload)


def _extract_node_link_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if "bipartite" in payload and isinstance(payload.get("bipartite"), Mapping):
        bipartite = payload.get("bipartite")
        data = bipartite.get("data") if isinstance(bipartite, Mapping) else None
        if isinstance(data, Mapping):
            return dict(data), True
    if "nodes" in payload and ("links" in payload or "edges" in payload):
        graph_meta = payload.get("graph")
        is_bipartite = False
        if isinstance(graph_meta, Mapping) and graph_meta.get("bipartite"):
            is_bipartite = True
        return dict(payload), is_bipartite
    raise ConfigError("graph.json has no node-link graph data.")


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


def _normalize_nodes(
    nodes_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    nodes: list[dict[str, Any]] = []
    node_map: dict[str, dict[str, Any]] = {}
    for entry in nodes_raw:
        node_id = _node_id_from_entry(entry)
        if isinstance(entry, Mapping):
            node = dict(entry)
        else:
            node = {}
        node["id"] = node_id
        node_map[node_id] = node
    nodes = list(node_map.values())
    return nodes, node_map


def _coerce_node_ref(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("id") or value.get("name") or value.get("key")
    if value is None:
        return None
    return str(value)


def _normalize_links(links_raw: Any) -> list[dict[str, Any]]:
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


def _build_adjacency(
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


def _finish_order(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []
    for node in nodes:
        if node in visited:
            continue
        stack: list[tuple[str, bool]] = [(node, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            if current in visited:
                continue
            visited.add(current)
            stack.append((current, True))
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    stack.append((neighbor, False))
    return order


def _collect_component(
    start: str,
    adjacency: Mapping[str, set[str]],
    visited: set[str],
) -> list[str]:
    component: list[str] = []
    stack = [start]
    visited.add(start)
    while stack:
        node = stack.pop()
        component.append(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return component


def _strongly_connected_components(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
) -> list[list[str]]:
    order = _finish_order(nodes, adjacency)
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in reversed(order):
        if node in visited:
            continue
        component = _collect_component(node, reverse, visited)
        components.append(component)
    return components


def _undirected_components(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> list[list[str]]:
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in visited:
            continue
        component = _collect_component(node, adjacency, visited)
        components.append(component)
    return components


def _summarize_components(
    components: Sequence[Sequence[str]],
    *,
    max_components: int,
    max_component_nodes: int,
) -> dict[str, Any]:
    summarized: list[dict[str, Any]] = []
    sorted_components = sorted(
        components,
        key=lambda comp: (-len(comp), sorted(comp)[0] if comp else ""),
    )
    for comp in sorted_components[:max_components]:
        nodes_sorted = sorted(comp)
        truncated = len(nodes_sorted) > max_component_nodes
        summarized.append(
            {
                "size": len(comp),
                "nodes": nodes_sorted[:max_component_nodes],
                "truncated": truncated,
            }
        )
    largest = summarized[0] if summarized else {"size": 0, "nodes": [], "truncated": False}
    return {
        "count": len(components),
        "components": summarized,
        "largest": largest,
    }


def _node_label(node: Mapping[str, Any], node_id: str) -> str:
    for key in ("label", "name", "reaction_id", "species"):
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return node_id


def _degree_centrality(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
    *,
    node_meta: Mapping[str, Mapping[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    node_count = len(nodes)
    ranking: list[dict[str, Any]] = []
    denom = float(node_count - 1) if node_count > 1 else 0.0
    for node_id in nodes:
        out_degree = len(adjacency.get(node_id, set()))
        in_degree = len(reverse.get(node_id, set()))
        degree = in_degree + out_degree
        score = degree / denom if denom else 0.0
        meta = node_meta.get(node_id, {})
        ranking.append(
            {
                "node_id": node_id,
                "label": _node_label(meta, node_id),
                "kind": meta.get("kind", "unknown"),
                "degree": degree,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "score": score,
            }
        )
    ranking.sort(key=lambda item: (-item["score"], item["node_id"]))
    if top_n < len(ranking):
        ranking = ranking[:top_n]
    return {"count": node_count, "top_n": top_n, "ranking": ranking}


def _betweenness_centrality(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    directed: bool,
    node_meta: Mapping[str, Mapping[str, Any]],
    top_n: int,
    max_nodes: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "skipped", "reason": "disabled", "ranking": []}
    if nx is None:
        return {"status": "skipped", "reason": "networkx_unavailable", "ranking": []}
    if len(nodes) > max_nodes:
        return {
            "status": "skipped",
            "reason": "node_limit",
            "max_nodes": max_nodes,
            "ranking": [],
        }
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    scores = nx.betweenness_centrality(graph)
    ranking = []
    for node_id, score in scores.items():
        meta = node_meta.get(node_id, {})
        ranking.append(
            {
                "node_id": node_id,
                "label": _node_label(meta, node_id),
                "kind": meta.get("kind", "unknown"),
                "score": float(score),
            }
        )
    ranking.sort(key=lambda item: (-item["score"], item["node_id"]))
    if top_n < len(ranking):
        ranking = ranking[:top_n]
    return {"status": "computed", "count": len(nodes), "top_n": top_n, "ranking": ranking}


def analyze_graph(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Analyze a stored GraphArtifact and emit graph analytics."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    analysis_cfg = _extract_analysis_cfg(graph_cfg)
    graph_id = _extract_graph_id(analysis_cfg)

    top_n = _coerce_positive_int(analysis_cfg.get("top_n"), "top_n", default=25)
    max_components = _coerce_positive_int(
        analysis_cfg.get("max_components"),
        "max_components",
        default=25,
    )
    max_component_nodes = _coerce_positive_int(
        analysis_cfg.get("max_component_nodes"),
        "max_component_nodes",
        default=200,
    )
    betweenness_enabled = _coerce_bool(
        analysis_cfg.get("compute_betweenness"),
        "compute_betweenness",
        default=False,
    )
    betweenness_max_nodes = _coerce_positive_int(
        analysis_cfg.get("betweenness_max_nodes"),
        "betweenness_max_nodes",
        default=2000,
    )

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    payload = _load_graph_payload(graph_dir)
    graph_data, is_bipartite = _extract_node_link_payload(payload)
    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    nodes, node_map = _normalize_nodes(nodes_raw)
    links = _normalize_links(links_raw)

    node_ids = [node["id"] for node in nodes]
    node_kind = {
        node_id: str(node.get("kind", "unknown"))
        for node_id, node in node_map.items()
    }
    direction_mode = _normalize_direction_mode(
        analysis_cfg.get("direction_mode"),
        is_bipartite=is_bipartite,
    )
    edges = _build_directed_edges(
        links,
        node_kind=node_kind,
        direction_mode=direction_mode,
    )
    edge_nodes = {node_id for edge in edges for node_id in edge}
    missing_nodes = sorted(edge_nodes.difference(node_map))
    for node_id in missing_nodes:
        node_map[node_id] = {"id": node_id}
    node_ids = list(node_map.keys())
    adjacency, reverse = _build_adjacency(node_ids, edges)

    scc_components = _strongly_connected_components(node_ids, adjacency, reverse)
    scc_summary = _summarize_components(
        scc_components,
        max_components=max_components,
        max_component_nodes=max_component_nodes,
    )

    undirected_adjacency: dict[str, set[str]] = {
        node: set(neighbors) for node, neighbors in adjacency.items()
    }
    for source, target in edges:
        undirected_adjacency.setdefault(source, set()).add(target)
        undirected_adjacency.setdefault(target, set()).add(source)
    communities = _undirected_components(node_ids, undirected_adjacency)
    community_summary = _summarize_components(
        communities,
        max_components=max_components,
        max_component_nodes=max_component_nodes,
    )

    degree_summary = _degree_centrality(
        node_ids,
        adjacency,
        reverse,
        node_meta=node_map,
        top_n=top_n,
    )
    betweenness_summary = _betweenness_centrality(
        node_ids,
        edges,
        directed=bool(graph_data.get("directed", True)),
        node_meta=node_map,
        top_n=top_n,
        max_nodes=betweenness_max_nodes,
        enabled=betweenness_enabled,
    )

    edge_count = sum(len(neighbors) for neighbors in adjacency.values())
    analysis_payload = {
        "source": {"graph_id": graph_id},
        "summary": {
            "node_count": len(node_ids),
            "edge_count": edge_count,
            "directed": bool(graph_data.get("directed", True)),
            "bipartite": is_bipartite,
        },
        "analysis": {
            "direction_mode": direction_mode,
            "limits": {
                "top_n": top_n,
                "max_components": max_components,
                "max_component_nodes": max_component_nodes,
                "betweenness_max_nodes": betweenness_max_nodes,
            },
            "scc": scc_summary,
            "communities": community_summary,
            "centrality": {
                "degree": degree_summary,
                "betweenness": betweenness_summary,
            },
        },
    }

    inputs_payload = {"graph_id": graph_id}
    code_meta = _code_metadata()
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "graph.json", analysis_payload)

    result = store.ensure(manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_temporal_graph_from_artifact(result.path, run_root)
    return result


def run_laplacian(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build and store a Laplacian matrix derived from a GraphArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    laplacian_cfg = _extract_laplacian_cfg(graph_cfg)
    graph_id = _extract_graph_id(laplacian_cfg)

    normalized = _coerce_bool(
        laplacian_cfg.get("normalized"),
        "normalized",
        default=False,
    )
    symmetrize = _coerce_bool(
        laplacian_cfg.get("symmetrize"),
        "symmetrize",
        default=True,
    )
    use_abs_weights = _coerce_bool(
        laplacian_cfg.get("use_abs_weights"),
        "use_abs_weights",
        default=True,
    )
    weight_key = _coerce_optional_str(
        laplacian_cfg.get("weight_key") or laplacian_cfg.get("weight_field"),
        "weight_key",
    )

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_payload = _load_graph_payload(graph_dir)
    result = build_laplacian(
        graph_payload,
        normalized=normalized,
        weight_key=weight_key,
        use_abs_weights=use_abs_weights,
        symmetrize=symmetrize,
    )

    inputs_payload = {"graph_id": graph_id}
    code_meta = _code_metadata()
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
    )

    laplacian_meta: dict[str, Any] = {
        "path": "laplacian.npz",
        "format": result.format,
        "shape": [len(result.nodes), len(result.nodes)],
        "degree_format": "vector",
        "laplacian_key": "laplacian",
        "degree_key": "degree",
        "normalized": normalized,
        "weight_key": weight_key or "auto",
        "use_abs_weights": use_abs_weights,
        "symmetrize": symmetrize,
    }
    if result.normalized_laplacian is not None:
        laplacian_meta["normalized_key"] = "laplacian_norm"

    laplacian_payload = {
        "kind": "laplacian",
        "source": {"graph_id": graph_id},
        "nodes": {"count": len(result.nodes), "order": list(result.nodes)},
        "laplacian": laplacian_meta,
    }

    def _writer(base_dir: Path) -> None:
        _write_laplacian_npz(base_dir / "laplacian.npz", result)
        write_json_atomic(base_dir / "graph.json", laplacian_payload)

    result = store.ensure(manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_temporal_graph_from_artifact(result.path, run_root)
    return result


def run_from_run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build a minimal bipartite GraphArtifact from a RunArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    run_id = _extract_run_id(graph_cfg)

    store.read_manifest("runs", run_id)
    run_dir = store.artifact_dir("runs", run_id)
    payload = load_run_dataset_payload(run_dir)
    gas_species = _extract_coord_names(payload, "species")
    surface_species = _extract_coord_names(payload, "surface_species")
    if not gas_species and not surface_species:
        raise ConfigError("run dataset must include species or surface_species coords.")
    reactions = _extract_coord_names(payload, "reaction")

    graph_data = _build_run_bipartite_graph(
        gas_species=gas_species,
        surface_species=surface_species,
        reactions=reactions,
    )
    links_key = "links" if "links" in graph_data else "edges"
    graph_payload = {
        "kind": "run_bipartite",
        "source": {"run_id": run_id},
        "bipartite": {
            "format": "node_link",
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get(links_key, [])),
            "data": graph_data,
        },
    }

    inputs_payload: dict[str, Any] = {"run_id": run_id}
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        parents=[run_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "graph.json", graph_payload)

    result = store.ensure(manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_temporal_graph_from_artifact(result.path, run_root)
    return result


def run_temporal_flux(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build a temporal species graph from windowed ROP integrals."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required to build temporal flux graphs.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    params = _extract_params(graph_cfg)
    run_ids = _extract_run_ids(graph_cfg, params, store=store)
    run_ids = sorted({str(run_id) for run_id in run_ids})
    mechanism, phase = _extract_optional_mechanism(graph_cfg, params)
    windowing_cfg = _normalize_windowing_cfg(params, graph_cfg)
    aggregation_cfg = _normalize_aggregation_cfg(params, graph_cfg)
    sparsify_cfg = _normalize_sparsify_cfg(params, graph_cfg)
    rop_cfg = _normalize_rop_cfg(params, graph_cfg)
    reaction_map_cfg = _extract_reaction_map_cfg(params, graph_cfg)
    cache_bust = params.get("cache_bust") or params.get("cache_key") or graph_cfg.get(
        "cache_bust"
    )

    if mechanism is None:
        first_payload = load_run_dataset_payload(
            store.artifact_dir("runs", run_ids[0])
        )
        attrs = first_payload.get("attrs", {})
        if isinstance(attrs, Mapping):
            mech_value = attrs.get("mechanism")
            if isinstance(mech_value, str) and mech_value.strip():
                mechanism = mech_value.strip()
            phase_value = attrs.get("phase")
            if isinstance(phase_value, str) and phase_value.strip():
                phase = phase_value.strip()

    inputs_payload: dict[str, Any] = {"run_ids": list(run_ids)}
    if mechanism is not None:
        inputs_payload["mechanism"] = mechanism
    if phase is not None:
        inputs_payload["phase"] = phase
    inputs_payload["windowing"] = {
        "type": windowing_cfg["type"],
        "count": windowing_cfg["count"],
        "base": windowing_cfg["base"],
    }
    inputs_payload["rop"] = dict(rop_cfg)
    inputs_payload["aggregation"] = dict(aggregation_cfg)
    inputs_payload["sparsify"] = dict(sparsify_cfg)
    if reaction_map_cfg is not None:
        inputs_payload["reaction_map"] = dict(reaction_map_cfg)
    if cache_bust is not None:
        inputs_payload["cache_bust"] = str(cache_bust)

    code_meta = _code_metadata()
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        parents=list(run_ids),
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
    )

    def _writer(base_dir: Path) -> None:
        run_payloads: list[dict[str, Any]] = []
        for run_id in run_ids:
            store.read_manifest("runs", run_id)
            run_dir = store.artifact_dir("runs", run_id)
            run_payloads.append(load_run_dataset_payload(run_dir))

        build_temporal_flux_graph(
            run_payloads=run_payloads,
            run_ids=run_ids,
            mechanism=mechanism,
            phase=phase,
            windowing_cfg=windowing_cfg,
            aggregation_cfg=aggregation_cfg,
            sparsify_cfg=sparsify_cfg,
            rop_cfg=rop_cfg,
            reaction_map_cfg=reaction_map_cfg,
            base_dir=base_dir,
        )

    result = store.ensure(manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_temporal_graph_from_artifact(result.path, run_root)
    return result


def superstate_reaction_merge_batch(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Aggregate reactions into "superreactions" under a superstate mapping.

    The output is a GraphArtifact (graph.json) that reports, for the baseline mechanism
    graph and for each provided reduction patch:
      - reaction_count
      - superreaction_exact_count (exact super-stoichiometry match)
      - traceability mappings (reaction_index -> superreaction_id, member reactions)

    This is an analysis/projection step; it does not synthesize a new kinetic model.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    params = _extract_params(graph_cfg)

    inputs = graph_cfg.get("inputs") or {}
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("graphs.inputs must be a mapping when provided.")
    inputs = dict(inputs)

    graph_id = inputs.get("graph_id") or inputs.get("graph") or graph_cfg.get("graph_id")
    graph_id = _require_nonempty_str(graph_id, "inputs.graph_id")
    mapping_id = (
        inputs.get("mapping_id")
        or inputs.get("mapping")
        or inputs.get("reduction_id")
        or graph_cfg.get("mapping_id")
    )
    mapping_id = _require_nonempty_str(mapping_id, "inputs.mapping_id")

    patches_raw = inputs.get("patches") or inputs.get("reductions") or inputs.get("patch_ids")
    if patches_raw is None:
        patches_raw = []
    if isinstance(patches_raw, str):
        patches = [_require_nonempty_str(patches_raw, "inputs.patches")]
    elif isinstance(patches_raw, Sequence) and not isinstance(
        patches_raw, (str, bytes, bytearray)
    ):
        patches = [
            _require_nonempty_str(item, "inputs.patches")
            for item in patches_raw
            if item is not None
        ]
    else:
        raise ConfigError("inputs.patches must be a string or list of strings.")
    patches = [pid for pid in patches if pid]

    drop_disabled = params.get("drop_disabled")
    if drop_disabled is None:
        drop_disabled = True
    if not isinstance(drop_disabled, bool):
        raise ConfigError("params.drop_disabled must be a boolean.")

    allow_mixed_reaction_type = params.get("allow_mixed_reaction_type")
    if allow_mixed_reaction_type is None:
        allow_mixed_reaction_type = False
    if not isinstance(allow_mixed_reaction_type, bool):
        raise ConfigError("params.allow_mixed_reaction_type must be a boolean.")

    use_reduced_mechanism = params.get("use_reduced_mechanism") or "auto"
    if not isinstance(use_reduced_mechanism, str) or not use_reduced_mechanism.strip():
        raise ConfigError("params.use_reduced_mechanism must be a string.")
    use_reduced_mechanism = use_reduced_mechanism.strip().lower()
    if use_reduced_mechanism not in {"auto", "always", "never"}:
        raise ConfigError("params.use_reduced_mechanism must be one of: auto, always, never.")

    include_kinetics_dispersion = params.get("include_kinetics_dispersion")
    if include_kinetics_dispersion is None:
        include_kinetics_dispersion = False
    if not isinstance(include_kinetics_dispersion, bool):
        raise ConfigError("params.include_kinetics_dispersion must be a boolean.")
    if include_kinetics_dispersion and ct is None:  # pragma: no cover - optional dependency
        raise ConfigError("include_kinetics_dispersion requires cantera to be installed.")

    def _load_mapping() -> dict[str, int]:
        store.read_manifest("reduction", mapping_id)
        base_dir = store.artifact_dir("reduction", mapping_id)
        path = base_dir / "mapping.json"
        if not path.exists():
            path = base_dir / "node_lumping.json"
        if not path.exists():
            raise ConfigError(f"Expected mapping.json or node_lumping.json for reduction/{mapping_id}.")
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ConfigError("mapping payload must be a JSON object.")
        mapping_raw = payload.get("mapping") or []
        if not isinstance(mapping_raw, Sequence) or isinstance(mapping_raw, (str, bytes, bytearray)):
            raise ConfigError("mapping payload must include a mapping list.")
        mapping_by_name: dict[str, int] = {}
        for entry in mapping_raw:
            if not isinstance(entry, Mapping):
                continue
            species = entry.get("species")
            if not isinstance(species, str) or not species.strip():
                continue
            if "superstate_id" in entry:
                sid = entry.get("superstate_id")
            else:
                sid = entry.get("cluster_id")
            try:
                sid_int = int(sid)
            except (TypeError, ValueError):
                continue
            mapping_by_name[species.strip()] = sid_int
        if not mapping_by_name:
            raise ConfigError("mapping payload produced no species->superstate entries.")
        return mapping_by_name

    mapping_by_name = _load_mapping()

    def _disabled_indices_from_patch(patch: Mapping[str, Any]) -> set[int]:
        disabled: set[int] = set()
        disabled_raw = patch.get("disabled_reactions") or []
        if isinstance(disabled_raw, Mapping):
            disabled_raw = [disabled_raw]
        if isinstance(disabled_raw, Sequence) and not isinstance(
            disabled_raw, (str, bytes, bytearray)
        ):
            for entry in disabled_raw:
                if not isinstance(entry, Mapping):
                    continue
                idx = entry.get("index")
                if isinstance(idx, bool) or idx is None:
                    continue
                try:
                    disabled.add(int(idx))
                except (TypeError, ValueError):
                    continue
        multipliers_raw = patch.get("reaction_multipliers") or []
        if isinstance(multipliers_raw, Mapping):
            multipliers_raw = [multipliers_raw]
        if isinstance(multipliers_raw, Sequence) and not isinstance(
            multipliers_raw, (str, bytes, bytearray)
        ):
            for entry in multipliers_raw:
                if not isinstance(entry, Mapping):
                    continue
                idx = entry.get("index")
                if isinstance(idx, bool) or idx is None:
                    continue
                try:
                    multiplier = float(entry.get("multiplier", 1.0))
                except (TypeError, ValueError):
                    continue
                if multiplier == 0.0:
                    try:
                        disabled.add(int(idx))
                    except (TypeError, ValueError):
                        continue
        return disabled

    def _species_name(node: Mapping[str, Any]) -> str:
        for key in ("label", "species", "name"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id.startswith("species_"):
            trimmed = node_id[len("species_") :]
            return trimmed if trimmed else node_id
        return str(node.get("id", "unknown"))

    def _canonical_coeff(value: float) -> float:
        if abs(value) < 1e-15:
            return 0.0
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-12:
            return float(int(rounded))
        return float(round(float(value), 12))

    def _signature_id(signature: Mapping[str, Any]) -> str:
        return stable_hash(signature, length=16)

    def _group_from_stoich_entries(
        *,
        species_names: Sequence[str],
        reaction_count: int,
        entries: Sequence[tuple[int, int, float]],
        reaction_types: Optional[Mapping[int, str]] = None,
        reaction_equations: Optional[Sequence[str]] = None,
        reaction_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        stoich_by_rxn: list[dict[int, float]] = [dict() for _ in range(reaction_count)]
        for s_idx, r_idx, value in entries:
            if r_idx < 0 or r_idx >= reaction_count:
                continue
            if s_idx < 0 or s_idx >= len(species_names):
                continue
            species = str(species_names[s_idx])
            sid = mapping_by_name.get(species)
            if sid is None:
                raise ConfigError(f"Species {species!r} missing from mapping_id={mapping_id}.")
            cur = stoich_by_rxn[r_idx].get(int(sid), 0.0)
            stoich_by_rxn[r_idx][int(sid)] = cur + float(value)

        reaction_to_superreaction: list[str] = []
        superreactions: dict[str, dict[str, Any]] = {}
        for r_idx in range(reaction_count):
            vec = stoich_by_rxn[r_idx]
            items = [
                (int(sid), _canonical_coeff(coeff))
                for sid, coeff in vec.items()
                if _canonical_coeff(coeff) != 0.0
            ]
            items.sort(key=lambda pair: pair[0])
            rtype = None
            if not allow_mixed_reaction_type and reaction_types is not None:
                rtype = reaction_types.get(r_idx) or "unknown"
            signature_obj: dict[str, Any] = {"stoich": items}
            if rtype is not None:
                signature_obj["reaction_type"] = str(rtype)
            srid = _signature_id(signature_obj)
            reaction_to_superreaction.append(srid)
            info = superreactions.get(srid)
            if info is None:
                info = {
                    "superreaction_id": srid,
                    "signature": signature_obj,
                    "members": [],
                }
                superreactions[srid] = info
            member = {
                "reaction_index": int(r_idx),
            }
            if reaction_ids is not None and r_idx < len(reaction_ids):
                member["reaction_id"] = str(reaction_ids[r_idx])
            if reaction_equations is not None and r_idx < len(reaction_equations):
                member["reaction_equation"] = str(reaction_equations[r_idx])
            if reaction_types is not None and r_idx in reaction_types:
                member["reaction_type"] = str(reaction_types[r_idx])
            info["members"].append(member)

        group_sizes = [len(info.get("members") or []) for info in superreactions.values()]
        group_sizes.sort()
        return {
            "reaction_indices": [int(idx) for idx in range(reaction_count)],
            "reaction_count": int(reaction_count),
            "superreaction_exact_count": int(len(superreactions)),
            "reaction_to_superreaction": reaction_to_superreaction,
            "superreactions": list(superreactions.values()),
            "group_size_summary": {
                "count": int(len(group_sizes)),
                "min": int(group_sizes[0]) if group_sizes else 0,
                "max": int(group_sizes[-1]) if group_sizes else 0,
                "mean": float(sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
                "p50": int(group_sizes[len(group_sizes) // 2]) if group_sizes else 0,
            },
        }

    # Baseline: use the provided stoichiometric graph artifact.
    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    payload = _load_graph_payload(graph_dir)
    graph_data, _ = _extract_node_link_payload(payload)
    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    nodes, node_map = _normalize_nodes(nodes_raw)
    links = _normalize_links(links_raw)

    species_node_to_name: dict[str, str] = {}
    reaction_node_to_index: dict[str, int] = {}
    reaction_type_by_index: dict[int, str] = {}
    reaction_equation_by_index: dict[int, str] = {}
    reaction_id_by_index: dict[int, str] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        kind = node.get("kind")
        if kind == "species":
            species_node_to_name[node_id] = _species_name(node)
        elif kind == "reaction":
            idx = node.get("reaction_index")
            if isinstance(idx, bool) or idx is None:
                continue
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            reaction_node_to_index[node_id] = idx_int
            rtype = node.get("reaction_type") or node.get("type") or "unknown"
            reaction_type_by_index[idx_int] = str(rtype)
            eq = node.get("reaction_equation") or node.get("equation")
            if isinstance(eq, str) and eq.strip():
                reaction_equation_by_index[idx_int] = eq.strip()
            rid = node.get("reaction_id") or node.get("reaction")
            if isinstance(rid, str) and rid.strip():
                reaction_id_by_index[idx_int] = rid.strip()

    if not reaction_type_by_index:
        raise ConfigError("Baseline graph has no reaction_index metadata.")
    baseline_reaction_count = max(reaction_type_by_index.keys()) + 1
    baseline_species_names: list[str] = []
    # Preserve stable ordering by species_index if present.
    species_entries = [
        (node.get("species_index"), node_id, name)
        for node_id, name in species_node_to_name.items()
        for node in [node_map.get(node_id, {})]
    ]
    species_entries.sort(key=lambda item: (item[0] if isinstance(item[0], int) else 1_000_000, str(item[2]).lower()))
    baseline_species_names = [name for _, _, name in species_entries]
    species_index_by_node_id = {node_id: idx for idx, (_, node_id, _) in enumerate(species_entries)}

    # Convert bipartite links into (species_index, reaction_index, stoich) entries.
    stoich_entries: list[tuple[int, int, float]] = []
    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is None or target is None:
            continue
        s_node = None
        r_node = None
        if source in species_node_to_name and target in reaction_node_to_index:
            s_node = source
            r_node = target
        elif target in species_node_to_name and source in reaction_node_to_index:
            s_node = target
            r_node = source
        if s_node is None or r_node is None:
            continue
        s_idx = species_index_by_node_id.get(s_node)
        r_idx = reaction_node_to_index.get(r_node)
        if s_idx is None or r_idx is None:
            continue
        stoich = link.get("stoich")
        if stoich is None:
            continue
        try:
            value = float(stoich)
        except (TypeError, ValueError):
            continue
        if value == 0.0:
            continue
        stoich_entries.append((int(s_idx), int(r_idx), float(value)))

    baseline_reaction_ids = [reaction_id_by_index.get(i, f"R{i+1}") for i in range(baseline_reaction_count)]
    baseline_equations = [reaction_equation_by_index.get(i, "") for i in range(baseline_reaction_count)]
    baseline_group = _group_from_stoich_entries(
        species_names=baseline_species_names,
        reaction_count=baseline_reaction_count,
        entries=stoich_entries,
        reaction_types=reaction_type_by_index,
        reaction_equations=baseline_equations,
        reaction_ids=baseline_reaction_ids,
    )
    baseline_group["graph_id"] = graph_id

    patches_out: list[dict[str, Any]] = []
    for reduction_id in patches:
        store.read_manifest("reduction", reduction_id)
        red_dir = store.artifact_dir("reduction", reduction_id)
        patch_path = red_dir / "mechanism_patch.yaml"
        patch_payload: Optional[dict[str, Any]] = None
        if patch_path.exists():
            raw = _read_mech_yaml_payload(patch_path)
            if isinstance(raw, Mapping):
                patch_payload = dict(raw)
        disabled_indices = _disabled_indices_from_patch(patch_payload or {})
        is_state_merge = bool(isinstance(patch_payload, Mapping) and patch_payload.get("state_merge"))

        mechanism_path = red_dir / "mechanism.yaml"
        use_reduced = False
        if use_reduced_mechanism == "always":
            use_reduced = mechanism_path.exists() and ct is not None
        elif use_reduced_mechanism == "never":
            use_reduced = False
        else:
            # auto: use reduced mechanism when a state-merge patch changes stoichiometry.
            use_reduced = bool(is_state_merge and mechanism_path.exists() and ct is not None)

        if use_reduced:
            solution = ct.Solution(str(mechanism_path))
            stoich = build_stoich(solution)
            reaction_annotations = annotate_reactions(solution)
            types_by_index = {
                int(meta.get("reaction_index")): str(meta.get("reaction_type", "unknown"))
                for meta in reaction_annotations.values()
                if isinstance(meta, Mapping) and meta.get("reaction_index") is not None
            }
            entries = _iter_stoich_entries(stoich)
            grouped = _group_from_stoich_entries(
                species_names=stoich.species,
                reaction_count=len(stoich.reaction_ids),
                entries=entries,
                reaction_types=types_by_index,
                reaction_equations=stoich.reaction_equations,
                reaction_ids=stoich.reaction_ids,
            )
            mode = "reduced_mechanism"
            reaction_count_after = int(grouped["reaction_count"])
        else:
            # Fast path: filter baseline reactions by disabled indices.
            remaining = [
                idx for idx in range(baseline_reaction_count) if idx not in disabled_indices
            ] if drop_disabled else list(range(baseline_reaction_count))

            # Build superreaction groups by reusing baseline reaction_to_superreaction.
            baseline_map = baseline_group.get("reaction_to_superreaction") or []
            kept_super: dict[str, list[int]] = {}
            for idx in remaining:
                if idx < 0 or idx >= len(baseline_map):
                    continue
                srid = baseline_map[idx]
                kept_super.setdefault(str(srid), []).append(int(idx))
            superreactions = []
            for srid, members in kept_super.items():
                superreactions.append(
                    {
                        "superreaction_id": srid,
                        "members": [{"reaction_index": int(m)} for m in members],
                    }
                )
            grouped = {
                "reaction_indices": [int(idx) for idx in remaining],
                "reaction_count": int(len(remaining)),
                "superreaction_exact_count": int(len(kept_super)),
                "reaction_to_superreaction": [
                    str(baseline_map[idx]) for idx in remaining if idx < len(baseline_map)
                ],
                "superreactions": superreactions,
                "group_size_summary": {
                    "count": int(len(kept_super)),
                    "min": int(min((len(v) for v in kept_super.values()), default=0)),
                    "max": int(max((len(v) for v in kept_super.values()), default=0)),
                    "mean": float(
                        sum(len(v) for v in kept_super.values()) / len(kept_super)
                    ) if kept_super else 0.0,
                    "p50": 0,
                },
            }
            mode = "drop_disabled" if drop_disabled else "baseline"
            reaction_count_after = int(grouped["reaction_count"])

        patch_entry: dict[str, Any] = {
            "reduction_id": reduction_id,
            "mode": mode,
            "disabled_reactions": int(len(disabled_indices)),
            "reactions_after": reaction_count_after,
            "superreaction_exact_count": int(grouped.get("superreaction_exact_count") or 0),
            "group_size_summary": grouped.get("group_size_summary") or {},
        }
        # Traceability: always keep a reaction->superreaction mapping and the member lists.
        patch_entry["reaction_indices"] = grouped.get("reaction_indices")
        patch_entry["reaction_to_superreaction"] = grouped.get("reaction_to_superreaction") or []
        patch_entry["superreactions"] = grouped.get("superreactions") or []
        patches_out.append(patch_entry)

    output_payload = {
        "schema_version": 1,
        "kind": "superstate_reaction_merge_batch",
        "source": {"graph_id": graph_id, "mapping_id": mapping_id},
        "params": {
            "drop_disabled": drop_disabled,
            "allow_mixed_reaction_type": allow_mixed_reaction_type,
            "use_reduced_mechanism": use_reduced_mechanism,
            "include_kinetics_dispersion": include_kinetics_dispersion,
        },
        "baseline": {
            "reaction_count": baseline_group["reaction_count"],
            "superreaction_exact_count": baseline_group["superreaction_exact_count"],
            "group_size_summary": baseline_group.get("group_size_summary"),
            "reaction_to_superreaction": baseline_group.get("reaction_to_superreaction") or [],
            "superreactions": baseline_group.get("superreactions") or [],
        },
        "patches": patches_out,
    }

    inputs_payload = {
        "graph_id": graph_id,
        "mapping_id": mapping_id,
        "patches": patches,
        "params": output_payload["params"],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = [graph_id, mapping_id] + list(patches)
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        parents=list(dict.fromkeys(parents)),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "graph.json", output_payload)

    return store.ensure(manifest, writer=_writer)


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build and store a stoichiometric matrix GraphArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    mechanism, phase = _extract_mechanism(graph_cfg)

    solution = _load_solution(mechanism, phase)
    result = build_stoich(solution)
    species_annotations = annotate_species(solution, phase=phase)
    reaction_annotations = annotate_reactions(solution, phase=phase)
    bipartite_graph = build_bipartite_graph(
        result,
        species_annotations=species_annotations,
        reaction_annotations=reaction_annotations,
    )

    inputs_payload: dict[str, Any] = {"mechanism": mechanism}
    if phase is not None:
        inputs_payload["phase"] = phase

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="graphs",
        artifact_id=artifact_id,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    metadata = _graph_metadata(
        result,
        mechanism=mechanism,
        phase=phase,
        bipartite_graph=bipartite_graph,
    )

    def _writer(base_dir: Path) -> None:
        _write_stoich_npz(base_dir / "stoich.npz", result)
        write_json_atomic(base_dir / "graph.json", metadata)

    return store.ensure(manifest, writer=_writer)


register("task", "graphs.stoich", run)
register("task", "graphs.analyze", analyze_graph)
register("task", "graphs.analytics", analyze_graph)
register("task", "graphs.laplacian", run_laplacian)
register("task", "graphs.from_run", run_from_run)
register("task", "graphs.temporal_flux", run_temporal_flux)
register("task", "graphs.superstate_reaction_merge_batch", superstate_reaction_merge_batch)

__all__ = [
    "LaplacianResult",
    "StoichResult",
    "annotate_reactions",
    "annotate_species",
    "analyze_graph",
    "build_stoich",
    "build_bipartite_graph",
    "build_laplacian",
    "run_from_run",
    "run_laplacian",
    "run_temporal_flux",
    "superstate_reaction_merge_batch",
    "run",
]
