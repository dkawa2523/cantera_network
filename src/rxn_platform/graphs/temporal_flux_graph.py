"""Temporal flux graph generation utilities."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional

from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import write_json_atomic

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:  # Optional dependency.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float


def _extract_coord_names(payload: Mapping[str, Any], coord: str) -> list[str]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ConfigError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    data = coord_payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ConfigError(f"coords.{coord}.data must be a sequence.")
    return [str(entry) for entry in data]


def _extract_coord_values(payload: Mapping[str, Any], coord: str) -> list[Any]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ConfigError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    data = coord_payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ConfigError(f"coords.{coord}.data must be a sequence.")
    return list(data)


def _extract_time_values(payload: Mapping[str, Any]) -> list[float]:
    values = _extract_coord_values(payload, "time")
    if not values:
        raise ConfigError("Run dataset missing coords.time.")
    time_values: list[float] = []
    for entry in values:
        try:
            time_values.append(float(entry))
        except (TypeError, ValueError) as exc:
            raise ConfigError("coords.time entries must be numeric.") from exc
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
        raise ConfigError("Run dataset data_vars must be a mapping.")
    var_payload = data_vars.get(var_name)
    if not isinstance(var_payload, Mapping):
        raise ConfigError(f"data_vars.{var_name} must be a mapping.")
    dims = var_payload.get("dims")
    data = var_payload.get("data")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise ConfigError(f"data_vars.{var_name}.dims must be a sequence.")
    dims_list = list(dims)
    if dims_list == ["time", axis]:
        matrix = _coerce_matrix(data, axis)
        for row in matrix:
            if len(row) != len(axis_names):
                raise ConfigError(
                    f"{var_name} columns mismatch: expected {len(axis_names)}, got {len(row)}."
                )
        return matrix
    if dims_list == [axis, "time"]:
        matrix = _coerce_matrix(data, axis)
        if len(matrix) != len(axis_names):
            raise ConfigError(
                f"{var_name} rows mismatch: expected {len(axis_names)}, got {len(matrix)}."
            )
        return _transpose_matrix(matrix)
    raise ConfigError(
        f"data_vars.{var_name}.dims must be [time, {axis}] or [{axis}, time]."
    )


def _coerce_matrix(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{label} data must be a sequence of rows.")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(
            row,
            (str, bytes, bytearray),
        ):
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
            coeff_value = float(coeff)
            reaction_map[r_idx].append(
                (species_index[species_name], abs(coeff_value))
            )
    return reaction_map


def _iter_stoich_entries(stoich: Any) -> list[tuple[int, int, float]]:
    matrix = getattr(stoich, "matrix", None)
    if matrix is None:
        return []
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
            "Unsupported stoichiometric matrix type for temporal flux graph."
        ) from exc
    return entries


def _build_reaction_map_from_stoich(
    stoich: Any,
    reaction_ids: Sequence[str],
    species: Sequence[str],
) -> tuple[list[list[tuple[int, float]]], dict[str, Any]]:
    """Build reaction->species incidence weights from a stoichiometric matrix.

    Important: RunArtifacts coming from Cantera typically label the `reaction` coord
    using reaction equations, while the stoich artifact may use synthetic IDs like
    `R1..`. For GRI-Mech style workflows the reaction order is stable, so we prefer
    index-aligned mapping when the reaction counts match.
    """
    reaction_index = {rid: idx for idx, rid in enumerate(reaction_ids)}
    species_index = {name: idx for idx, name in enumerate(species)}
    reaction_map: list[list[tuple[int, float]]] = [
        [] for _ in range(len(reaction_ids))
    ]
    meta: dict[str, Any] = {
        "method": "label",
        "stoich_reaction_count": len(getattr(stoich, "reaction_ids", []) or []),
        "run_reaction_count": len(reaction_ids),
    }

    # Prefer index alignment when the reaction axis lengths match. This avoids the
    # common mismatch between synthetic stoich IDs and Cantera equation labels.
    stoich_reaction_ids = list(getattr(stoich, "reaction_ids", []) or [])
    stoich_reaction_equations = list(getattr(stoich, "reaction_equations", []) or [])
    stoich_reaction_count = len(stoich_reaction_ids) or len(stoich_reaction_equations)
    if stoich_reaction_count == len(reaction_ids) and stoich_reaction_count > 0:
        meta["method"] = "index"
        meta["note"] = "Mapped stoich reaction columns to run reaction axis by index."
        for s_idx, r_idx, coeff in _iter_stoich_entries(stoich):
            if r_idx < 0 or r_idx >= len(reaction_ids):
                continue
            target_idx = int(r_idx)
            # Species names come from the stoich artifact; map into run species list.
            stoich_species = getattr(stoich, "species", [])
            if s_idx < 0 or s_idx >= len(stoich_species):
                continue
            species_name = stoich_species[s_idx]
            if species_name not in species_index:
                raise ConfigError(
                    f"Species {species_name!r} not found in run species list."
                )
            reaction_map[target_idx].append(
                (species_index[species_name], abs(float(coeff)))
            )
        return reaction_map, meta

    for s_idx, r_idx, coeff in _iter_stoich_entries(stoich):
        reaction_id: Optional[str] = None
        if 0 <= r_idx < len(stoich_reaction_ids):
            reaction_id = stoich_reaction_ids[r_idx]
        if (reaction_id is None or reaction_id not in reaction_index) and 0 <= r_idx < len(
            stoich_reaction_equations
        ):
            # Best-effort fallback: match by equation label (common for Cantera runs).
            reaction_id = stoich_reaction_equations[r_idx]
            if reaction_id in reaction_index:
                meta["method"] = "equation"
        if reaction_id is None or reaction_id not in reaction_index:
            continue
        target_idx = reaction_index[reaction_id]
        stoich_species = getattr(stoich, "species", [])
        if s_idx < 0 or s_idx >= len(stoich_species):
            continue
        species_name = stoich_species[s_idx]
        if species_name not in species_index:
            raise ConfigError(
                f"Species {species_name!r} not found in run species list."
            )
        reaction_map[target_idx].append(
            (species_index[species_name], abs(float(coeff)))
        )
    return reaction_map, meta


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
    )


def build_temporal_flux_graph(
    *,
    run_payloads: Sequence[Mapping[str, Any]],
    run_ids: Sequence[str],
    mechanism: Optional[str],
    phase: Optional[str],
    windowing_cfg: Mapping[str, Any],
    aggregation_cfg: Mapping[str, float],
    sparsify_cfg: Mapping[str, Optional[float]],
    rop_cfg: Mapping[str, Any],
    reaction_map_cfg: Optional[Mapping[str, Any]],
    base_dir: Path,
) -> dict[str, Any]:
    if np is None:
        raise ConfigError("numpy is required to build temporal flux graphs.")
    if not run_payloads:
        raise ConfigError("No run datasets were loaded.")

    base_species: Optional[list[str]] = None
    base_reactions: Optional[list[str]] = None
    base_time: Optional[list[float]] = None
    rop_matrices: list[Any] = []
    activity_sum: Optional[Any] = None
    reaction_activity_sum: Optional[Any] = None

    for payload in run_payloads:
        species = _extract_coord_names(payload, "species")
        if not species:
            species = _extract_coord_names(payload, "surface_species")
        if not species:
            raise ConfigError("Run dataset must include species coords.")
        reactions = _extract_coord_names(payload, "reaction")
        if not reactions:
            raise ConfigError("Run dataset must include reaction coords.")
        time_values = _extract_time_values(payload)

        rop_matrix = _extract_time_matrix(
            payload,
            var_name=str(rop_cfg["var"]),
            axis="reaction",
            axis_names=reactions,
        )
        time_values, rop_matrix = _sort_time_series(time_values, rop_matrix)

        if base_species is None:
            base_species = species
        elif species != base_species:
            raise ConfigError("Species ordering must be consistent across runs.")
        if base_reactions is None:
            base_reactions = reactions
        elif reactions != base_reactions:
            raise ConfigError("Reaction ordering must be consistent across runs.")
        if base_time is None:
            base_time = time_values
        else:
            if len(time_values) != len(base_time):
                raise ConfigError("Time grid length mismatch across runs.")
            for idx, (a, b) in enumerate(zip(time_values, base_time)):
                if abs(a - b) > 1.0e-9:
                    raise ConfigError(
                        f"Time grid mismatch at index {idx} ({a} vs {b})."
                    )

        rop_arr = np.asarray(rop_matrix, dtype=float)
        rop_matrices.append(rop_arr)
        activity = np.sum(np.abs(rop_arr), axis=1)
        activity_sum = activity if activity_sum is None else activity_sum + activity
        reaction_activity = np.sum(np.abs(rop_arr), axis=0)
        reaction_activity_sum = (
            reaction_activity
            if reaction_activity_sum is None
            else reaction_activity_sum + reaction_activity
        )

    if base_species is None or base_reactions is None or base_time is None:
        raise ConfigError("No run datasets were loaded.")

    activity_avg = (
        activity_sum / float(len(rop_matrices)) if activity_sum is not None else None
    )
    reaction_activity_avg = (
        reaction_activity_sum / float(len(rop_matrices))
        if reaction_activity_sum is not None
        else None
    )

    window_type = str(windowing_cfg["type"])
    if window_type == "log_time":
        windows = _build_log_time_windows(
            base_time,
            n_windows=int(windowing_cfg["count"]),
            base=float(windowing_cfg["base"]),
        )
    elif window_type == "event_based":
        if activity_avg is None:
            raise ConfigError("event_based windowing requires ROP activity data.")
        windows = _build_event_windows(
            base_time,
            activity_avg.tolist(),
            n_windows=int(windowing_cfg["count"]),
        )
    elif window_type == "fixed":
        windows = _build_fixed_windows(base_time, n_windows=int(windowing_cfg["count"]))
    else:
        raise ConfigError(
            "windowing.type must be one of log_time, event_based, fixed."
        )

    stoich_source = "fallback_full"
    solution_for_meta: Optional[Any] = None
    if reaction_map_cfg is not None:
        reaction_map = _build_reaction_map_from_cfg(
            base_reactions, base_species, reaction_map_cfg
        )
        stoich_source = "reaction_map"
    elif mechanism is not None:
        from rxn_platform.tasks import graphs as graph_tasks

        if graph_tasks.ct is None:
            reaction_map = _fallback_reaction_map(base_reactions, base_species)
        else:
            solution_for_meta = graph_tasks._load_solution(mechanism, phase)
            stoich = graph_tasks.build_stoich(solution_for_meta)
            reaction_map, reaction_map_meta = _build_reaction_map_from_stoich(
                stoich, base_reactions, base_species
            )
            stoich_source = "mechanism"
    else:
        reaction_map = _fallback_reaction_map(base_reactions, base_species)

    reaction_pairs = _precompute_reaction_pairs(reaction_map)

    # Optional node metadata: keep TemporalFluxGraph self-contained even when
    # downstream tasks do not have access to the original stoich graph.
    species_annotations: dict[str, dict[str, Any]] = {}
    molecular_weights: dict[str, float] = {}
    if solution_for_meta is not None:
        try:
            from rxn_platform.tasks import graphs as graph_tasks

            annotations = graph_tasks.annotate_species(solution_for_meta, phase=phase)
            if isinstance(annotations, Mapping):
                species_annotations = {str(k): dict(v) for k, v in annotations.items() if isinstance(v, Mapping)}
        except Exception:
            species_annotations = {}
        try:
            mw_values = getattr(solution_for_meta, "molecular_weights", None)
            species_names = list(getattr(solution_for_meta, "species_names", []) or [])
            if isinstance(mw_values, Sequence) and species_names:
                for idx, name in enumerate(species_names):
                    if idx >= len(mw_values):
                        break
                    try:
                        molecular_weights[str(name)] = float(mw_values[idx])
                    except (TypeError, ValueError):
                        continue
        except Exception:
            molecular_weights = {}

    def _heavy_elements(elements: Any) -> list[str]:
        if not isinstance(elements, Mapping):
            return []
        heavy: list[str] = []
        for key, value in elements.items():
            if key is None:
                continue
            element = str(key)
            if element == "H":
                continue
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count == 0.0:
                continue
            heavy.append(element)
        return sorted(set(heavy))

    def _infer_kind(elements: Any, charge: Any, state: Any) -> str:
        try:
            charge_value = float(charge)
        except (TypeError, ValueError):
            charge_value = 0.0
        if abs(charge_value) > 1.0e-12:
            return "ion"
        if isinstance(elements, Mapping):
            total_atoms = 0.0
            nonzero = 0
            for value in elements.values():
                try:
                    total_atoms += float(value)
                except (TypeError, ValueError):
                    continue
                if float(value) != 0.0:
                    nonzero += 1
            if nonzero == 1 and abs(total_atoms - 1.0) < 1.0e-12:
                return "atom"
        if isinstance(state, str) and state.strip().lower() == "radical":
            return "radical"
        return "molecule"

    condition_matrices: list[list[Any]] = []
    for rop_arr in rop_matrices:
        window_mats: list[Any] = []
        for window in windows:
            fluxes = _integrate_window_fluxes(base_time, rop_arr, window)
            adj = _build_species_adjacency(
                fluxes,
                reaction_pairs,
                n_species=len(base_species),
                use_abs=bool(rop_cfg.get("use_abs", True)),
            )
            window_mats.append(adj)
        condition_matrices.append(window_mats)

    layer_dir = base_dir / "species_graph"
    layer_dir.mkdir(parents=True, exist_ok=True)
    layers_meta: list[dict[str, Any]] = []

    for idx, window in enumerate(windows):
        window_mats = [cond[idx] for cond in condition_matrices]
        aggregated = _aggregate_condition_matrices(
            window_mats,
            mean_weight=float(aggregation_cfg["mean"]),
            p95_weight=float(aggregation_cfg["p95"]),
        )
        np.fill_diagonal(aggregated, 0.0)
        aggregated = _sparsify_matrix(
            aggregated,
            quantile=sparsify_cfg.get("quantile"),
            min_value=sparsify_cfg.get("min_value"),
        )
        filename = f"layer_{idx:03d}.npz"
        _write_csr_npz(layer_dir / filename, aggregated)

        nnz = int(np.count_nonzero(aggregated))
        density = float(nnz) / float(aggregated.shape[0] * aggregated.shape[1])
        layers_meta.append(
            {
                "index": idx,
                "path": f"species_graph/{filename}",
                "nnz": nnz,
                "density": density,
                "window": {
                    "start_idx": window.start_idx,
                    "end_idx": window.end_idx,
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                },
            }
        )

    graph_payload = {
        "kind": "temporal_flux",
        "source": {
            "run_ids": list(run_ids),
            "mechanism": mechanism,
            "phase": phase,
        },
        "species": {"count": len(base_species), "order": list(base_species)},
        "reactions": {"count": len(base_reactions), "order": list(base_reactions)},
        "windowing": {
            "type": window_type,
            "count": int(windowing_cfg["count"]),
            "base": float(windowing_cfg["base"]),
        },
        "aggregation": dict(aggregation_cfg),
        "rop": dict(rop_cfg),
        "sparsify": dict(sparsify_cfg),
        "stoich_source": stoich_source,
        "reaction_map": reaction_map_meta if "reaction_map_meta" in locals() else {},
        "reaction_stats": {
            "activity": {
                "type": "abs_rop_sum",
                "values": reaction_activity_avg.tolist()
                if reaction_activity_avg is not None
                else [],
                "total": float(np.sum(reaction_activity_avg))
                if reaction_activity_avg is not None
                else 0.0,
            }
        },
        "species_graph": {
            "format": "csr",
            "path": "species_graph",
            "layers": layers_meta,
        },
        "nodes": [
            {
                "id": f"species_{name}",
                "kind": "species",
                "label": name,
                "species_index": idx,
                "formula": (species_annotations.get(name) or {}).get("formula"),
                "elements": (species_annotations.get(name) or {}).get("elements") or {},
                "charge": (species_annotations.get(name) or {}).get("charge"),
                "phase": (species_annotations.get(name) or {}).get("phase") or phase,
                "state": (species_annotations.get(name) or {}).get("state"),
                "kind_class": _infer_kind(
                    (species_annotations.get(name) or {}).get("elements") or {},
                    (species_annotations.get(name) or {}).get("charge"),
                    (species_annotations.get(name) or {}).get("state"),
                ),
                "mw": molecular_weights.get(name),
                "heavy_elements_set": _heavy_elements(
                    (species_annotations.get(name) or {}).get("elements") or {}
                ),
                "heavy_elements_signature": ",".join(
                    _heavy_elements(
                        (species_annotations.get(name) or {}).get("elements") or {}
                    )
                )
                if _heavy_elements((species_annotations.get(name) or {}).get("elements") or {})
                else "none",
            }
            for idx, name in enumerate(base_species)
        ],
    }

    write_json_atomic(base_dir / "graph.json", graph_payload)
    # Contract: meta.json is a compatibility copy of graph.json.
    write_json_atomic(base_dir / "meta.json", graph_payload)
    return graph_payload


__all__ = ["TimeWindow", "build_temporal_flux_graph"]
