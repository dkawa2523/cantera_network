"""GNN dataset export helpers for dynamic graph node features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
import random
from pathlib import Path
from typing import Any, Optional
from rxn_platform.core import make_artifact_id
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.registry import Registry, register
from rxn_platform.run_store import utc_now_iso
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    load_run_dataset_payload,
    load_run_ids_from_run_set,
    resolve_cfg as _resolve_cfg,
)

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

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
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

DEFAULT_NODE_KINDS = ("species",)
DEFAULT_MISSING_STRATEGY = "nan"
TABLE_COLUMNS = (
    "run_id",
    "time_index",
    "time",
    "node_id",
    "feature",
    "value",
    "meta_json",
)


@dataclass(frozen=True)
class NodeFeatureSpec:
    name: str
    data_var: str
    coord: Optional[str]


def _extract_gnn_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ("gnn_dataset", "gnn_datasets", "gnn"):
        if key in cfg and isinstance(cfg.get(key), Mapping):
            gnn_cfg = cfg.get(key)
            if not isinstance(gnn_cfg, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(cfg), dict(gnn_cfg)
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
    cfg: Mapping[str, Any],
    *,
    store: Optional[ArtifactStore] = None,
) -> list[str]:
    inputs = cfg.get("inputs")
    run_set_id: Any = None
    run_ids: Any = None
    if inputs is None:
        run_ids = None
    elif not isinstance(inputs, Mapping):
        raise ConfigError("gnn_dataset.inputs must be a mapping.")
    else:
        if "run_set_id" in inputs:
            run_set_id = inputs.get("run_set_id")
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in inputs:
                run_ids = inputs.get(key)
                break
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is None and "run_set_id" in cfg:
        run_set_id = cfg.get("run_set_id")
        if run_set_id is not None and run_ids is not None:
            raise ConfigError("Specify only one of run_set_id or run_id(s).")
    if run_set_id is not None:
        run_set_id = _require_nonempty_str(run_set_id, "run_set_id")
        if store is None:
            raise ConfigError("run_set_id requires a store to be provided.")
        return load_run_ids_from_run_set(store, run_set_id)
    if run_ids is None:
        for key in ("runs", "run_ids", "run_id", "run"):
            if key in cfg:
                run_ids = cfg.get(key)
                break
    run_id_list = _coerce_run_ids(run_ids)
    if not run_id_list:
        raise ConfigError("gnn_dataset run_id is required.")
    return run_id_list


def _extract_graph_id(cfg: Mapping[str, Any]) -> str:
    inputs = cfg.get("inputs")
    graph_id: Any = None
    if inputs is None:
        graph_id = None
    elif not isinstance(inputs, Mapping):
        raise ConfigError("gnn_dataset.inputs must be a mapping.")
    else:
        for key in ("graph_id", "graph"):
            if key in inputs:
                graph_id = inputs.get(key)
                break
    if graph_id is None:
        for key in ("graph_id", "graph"):
            if key in cfg:
                graph_id = cfg.get(key)
                break
    if graph_id is None:
        raise ConfigError("gnn_dataset graph_id is required.")
    return _require_nonempty_str(graph_id, "graph_id")


def _extract_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("gnn_dataset.params must be a mapping.")
    return dict(params)


def _normalize_missing_strategy(value: Any) -> str:
    if value is None:
        return DEFAULT_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in {"nan", "skip", "error"}:
        raise ConfigError("missing_strategy must be 'nan', 'skip', or 'error'.")
    return strategy


def _normalize_feature_specs(value: Any) -> list[NodeFeatureSpec]:
    if value is None:
        return []
    entries: list[Any]
    if isinstance(value, Mapping):
        entries = [value]
    elif isinstance(value, str):
        entries = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        entries = list(value)
    else:
        raise ConfigError("node_features must be a string, mapping, or sequence.")

    specs: list[NodeFeatureSpec] = []
    for idx, entry in enumerate(entries, start=1):
        label = f"node_features[{idx}]"
        if isinstance(entry, str):
            name = _require_nonempty_str(entry, f"{label}.name")
            specs.append(NodeFeatureSpec(name=name, data_var=name, coord=None))
            continue
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{label} must be a mapping or string.")
        raw_data_var = entry.get("data_var") or entry.get("var") or entry.get("source")
        name = entry.get("name") or entry.get("feature") or raw_data_var
        if raw_data_var is None:
            raw_data_var = entry.get("name") or entry.get("feature")
        data_var = _require_nonempty_str(raw_data_var, f"{label}.data_var")
        name = _require_nonempty_str(name, f"{label}.name")
        coord = entry.get("coord") or entry.get("axis")
        if coord is not None:
            coord = _require_nonempty_str(coord, f"{label}.coord").strip()
        specs.append(NodeFeatureSpec(name=name, data_var=data_var, coord=coord))
    return specs


def _default_feature_specs(data_vars: Mapping[str, Any]) -> list[NodeFeatureSpec]:
    specs: list[NodeFeatureSpec] = []
    if "X" in data_vars:
        specs.append(NodeFeatureSpec(name="X", data_var="X", coord=None))
    if "net_production_rates" in data_vars:
        specs.append(
            NodeFeatureSpec(
                name="wdot",
                data_var="net_production_rates",
                coord=None,
            )
        )
    if not specs:
        raise ConfigError(
            "node_features must be provided when X or net_production_rates are missing."
        )
    return specs


def _extract_time_values(payload: Mapping[str, Any]) -> list[float]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    time_payload = coords.get("time")
    if not isinstance(time_payload, Mapping):
        raise ArtifactError("Run dataset coords.time is missing.")
    time_data = time_payload.get("data")
    if not isinstance(time_data, Sequence) or isinstance(
        time_data, (str, bytes, bytearray)
    ):
        raise ArtifactError("Run dataset coords.time.data must be a sequence.")
    values: list[float] = []
    for entry in time_data:
        try:
            values.append(float(entry))
        except (TypeError, ValueError) as exc:
            raise ArtifactError("Run dataset time values must be numeric.") from exc
    return values


def _extract_coord_values(payload: Mapping[str, Any], coord: str) -> list[str]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    return _coerce_str_sequence(
        coord_payload.get("data"), f"coords.{coord}.data"
    )


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


def _extract_node_link_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "bipartite" in payload and isinstance(payload.get("bipartite"), Mapping):
        bipartite = payload.get("bipartite")
        data = bipartite.get("data") if isinstance(bipartite, Mapping) else None
        if isinstance(data, Mapping):
            return dict(data)
    if "nodes" in payload and ("links" in payload or "edges" in payload):
        return dict(payload)
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


def _normalize_nodes(nodes_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    nodes: list[dict[str, Any]] = []
    for entry in nodes_raw:
        node_id = _node_id_from_entry(entry)
        node = dict(entry) if isinstance(entry, Mapping) else {}
        node["id"] = node_id
        nodes.append(node)
    return nodes


def _infer_node_kind(node: Mapping[str, Any]) -> str:
    kind = node.get("kind")
    if kind is not None:
        return str(kind)
    node_id = str(node.get("id", ""))
    if node_id.startswith(("species_", "surface_")):
        return "species"
    if node_id.startswith("reaction_"):
        return "reaction"
    return "unknown"


def _infer_node_phase(node: Mapping[str, Any], kind: str) -> str:
    phase = node.get("phase")
    if phase is not None:
        return str(phase)
    node_id = str(node.get("id", ""))
    if node_id.startswith("surface_"):
        return "surface"
    if kind == "species":
        return "gas"
    return "unknown"


def _species_name_from_node(node: Mapping[str, Any]) -> Optional[str]:
    for key in ("species", "label", "name"):
        value = node.get(key)
        if value:
            return str(value)
    node_id = str(node.get("id", ""))
    for prefix in ("species_", "surface_"):
        if node_id.startswith(prefix):
            return node_id[len(prefix) :]
    return None


def _prepare_graph_nodes(
    nodes: Sequence[Mapping[str, Any]],
    node_kinds: Sequence[str],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    kind_filter = set(node_kinds)
    for entry in nodes:
        node = dict(entry)
        kind = _infer_node_kind(node)
        if kind_filter and kind not in kind_filter:
            continue
        phase = _infer_node_phase(node, kind)
        species = _species_name_from_node(node) if kind == "species" else None
        coord = "surface_species" if phase == "surface" else "species"
        prepared.append(
            {
                "id": node.get("id"),
                "kind": kind,
                "phase": phase,
                "species": species,
                "coord": coord,
                "label": node.get("label") or node.get("name"),
            }
        )
    return prepared


def _build_node_meta(
    nodes: Sequence[Mapping[str, Any]],
    gas_species: Sequence[str],
    surface_species: Sequence[str],
) -> list[dict[str, Any]]:
    gas_map = {name: idx for idx, name in enumerate(gas_species)}
    surface_map = {name: idx for idx, name in enumerate(surface_species)}
    meta: list[dict[str, Any]] = []
    for node in nodes:
        coord = node.get("coord")
        species = node.get("species")
        if coord == "surface_species":
            species_index = surface_map.get(species)
        else:
            species_index = gas_map.get(species)
        node_meta: dict[str, Any] = {
            "id": node.get("id"),
            "kind": node.get("kind"),
            "phase": node.get("phase"),
            "coord": coord,
            "species": species,
            "species_index": species_index,
        }
        label = node.get("label")
        if label is not None:
            node_meta["label"] = label
        meta.append(node_meta)
    return meta


def _axis_indices(
    nodes: Sequence[Mapping[str, Any]],
    axis: str,
    gas_species: Sequence[str],
    surface_species: Sequence[str],
) -> list[Optional[int]]:
    gas_map = {name: idx for idx, name in enumerate(gas_species)}
    surface_map = {name: idx for idx, name in enumerate(surface_species)}
    indices: list[Optional[int]] = []
    for node in nodes:
        coord = node.get("coord")
        species = node.get("species")
        if coord != axis or species is None:
            indices.append(None)
            continue
        if axis == "surface_species":
            indices.append(surface_map.get(species))
        else:
            indices.append(gas_map.get(species))
    return indices


def _infer_axis_from_dims(dims: Sequence[Any]) -> str:
    if "species" in dims:
        return "species"
    if "surface_species" in dims:
        return "surface_species"
    raise ArtifactError("Run dataset variable must include species axis.")


def _coerce_matrix(data: Any, label: str) -> list[list[Any]]:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ArtifactError(f"Run dataset {label} must be a matrix.")
    matrix: list[list[Any]] = []
    for entry in data:
        if not isinstance(entry, Sequence) or isinstance(
            entry, (str, bytes, bytearray)
        ):
            raise ArtifactError(f"Run dataset {label} must be a matrix.")
        matrix.append(list(entry))
    return matrix


def _align_matrix(
    matrix: list[list[Any]],
    dims: Sequence[Any],
    axis: str,
    time_values: Sequence[float],
    axis_len: int,
    label: str,
) -> list[list[Any]]:
    if len(dims) != 2:
        raise ArtifactError(f"Run dataset {label} must be 2D with time and {axis}.")

    time_len = len(time_values)
    if dims[0] == "time" and dims[1] == axis:
        if time_len and len(matrix) != time_len:
            raise ArtifactError(f"Run dataset {label} time length mismatch.")
        if axis_len == 0 and matrix:
            axis_len = len(matrix[0])
        if axis_len:
            for row in matrix:
                if len(row) != axis_len:
                    raise ArtifactError(f"Run dataset {label} axis length mismatch.")
        return matrix

    if dims[1] == "time" and dims[0] == axis:
        if axis_len == 0:
            axis_len = len(matrix)
        if axis_len and len(matrix) != axis_len:
            raise ArtifactError(f"Run dataset {label} axis length mismatch.")
        if time_len and matrix and len(matrix[0]) != time_len:
            raise ArtifactError(f"Run dataset {label} time length mismatch.")
        return [list(row) for row in zip(*matrix)] if matrix else []

    raise ArtifactError(f"Run dataset {label} must include time and {axis} dims.")


def _extract_feature_matrix(
    data_vars: Mapping[str, Any],
    spec: NodeFeatureSpec,
    time_values: Sequence[float],
    gas_species: Sequence[str],
    surface_species: Sequence[str],
    missing_strategy: str,
) -> tuple[Optional[list[list[Any]]], Optional[str], bool]:
    entry = data_vars.get(spec.data_var)
    if entry is None:
        if missing_strategy == "error":
            raise ConfigError(f"Run dataset is missing data_vars.{spec.data_var}.")
        return None, None, True
    if not isinstance(entry, Mapping):
        raise ArtifactError(f"Run dataset data_vars.{spec.data_var} is invalid.")
    dims = entry.get("dims")
    data = entry.get("data")
    if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes, bytearray)):
        raise ArtifactError(f"Run dataset data_vars.{spec.data_var}.dims is invalid.")
    axis = spec.coord or _infer_axis_from_dims(dims)
    if axis not in dims:
        raise ArtifactError(f"Run dataset {spec.data_var} must include {axis} dims.")
    axis_len = len(gas_species) if axis == "species" else len(surface_species)
    matrix = _coerce_matrix(data, spec.data_var)
    matrix = _align_matrix(
        matrix,
        dims,
        axis,
        time_values,
        axis_len,
        spec.data_var,
    )
    return matrix, axis, False


def _flatten_rows(
    run_id: str,
    time_values: Sequence[float],
    node_order: Sequence[str],
    feature_specs: Sequence[NodeFeatureSpec],
    values: Sequence[Sequence[Sequence[float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(time_values):
        for node_index, node_id in enumerate(node_order):
            for feat_index, spec in enumerate(feature_specs):
                value = values[time_index][node_index][feat_index]
                rows.append(
                    {
                        "run_id": run_id,
                        "time_index": time_index,
                        "time": time_value,
                        "node_id": node_id,
                        "feature": spec.name,
                        "value": value,
                        "meta_json": "{}",
                    }
                )
    return rows


def _write_node_features_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if pd is not None:
        frame = pd.DataFrame(list(rows), columns=TABLE_COLUMNS)
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            pass
    if pa is not None and pq is not None:
        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("time_index", pa.int64()),
                ("time", pa.float64()),
                ("node_id", pa.string()),
                ("feature", pa.string()),
                ("value", pa.float64()),
                ("meta_json", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(table, path)
        return
    payload = {"columns": list(TABLE_COLUMNS), "rows": list(rows)}
    write_json_atomic(path, payload)
    logger = logging.getLogger("rxn_platform.gnn_dataset")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
    )


def _write_node_features_npz(path: Path, payload: Mapping[str, Any]) -> bool:
    if np is None:
        return False
    np.savez_compressed(path, **payload)
    return True


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Export node feature time series for GNN training datasets."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, gnn_cfg = _extract_gnn_cfg(resolved_cfg)
    params = _extract_params(gnn_cfg)
    run_ids = _extract_run_ids(gnn_cfg, store=store)
    graph_id = _extract_graph_id(gnn_cfg)

    node_kinds = _coerce_str_sequence(
        params.get("node_kinds") or params.get("kinds"),
        "node_kinds",
    )
    if not node_kinds:
        node_kinds = list(DEFAULT_NODE_KINDS)
    missing_strategy = _normalize_missing_strategy(params.get("missing_strategy"))
    feature_specs = _normalize_feature_specs(
        params.get("node_features") or params.get("features")
    )

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_payload = _load_graph_payload(graph_dir)
    graph_data = _extract_node_link_payload(graph_payload)
    nodes_raw = graph_data.get("nodes")
    if nodes_raw is None:
        raise ConfigError("graph.json nodes are missing.")
    nodes = _normalize_nodes(nodes_raw)
    graph_nodes = _prepare_graph_nodes(nodes, node_kinds=node_kinds)
    if not graph_nodes:
        raise ConfigError("graph.json has no nodes matching requested kinds.")
    node_order = [node.get("id") for node in graph_nodes]

    rows: list[dict[str, Any]] = []
    time_by_run: dict[str, dict[str, Any]] = {}
    missing_features: dict[str, list[str]] = {}
    node_meta: Optional[list[dict[str, Any]]] = None
    node_axis_indices: dict[str, list[Optional[int]]] = {}
    species_ref: Optional[list[str]] = None
    surface_ref: Optional[list[str]] = None

    npz_payload: dict[str, Any] = {}

    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        run_payload = load_run_dataset_payload(run_dir)
        time_values = _extract_time_values(run_payload)
        gas_species = _extract_coord_values(run_payload, "species")
        surface_species = _extract_coord_values(run_payload, "surface_species")

        if species_ref is None:
            species_ref = gas_species
        elif gas_species != species_ref:
            raise ConfigError("gnn_dataset requires consistent species coords.")
        if surface_ref is None:
            surface_ref = surface_species
        elif surface_species != surface_ref:
            raise ConfigError("gnn_dataset requires consistent surface species coords.")

        data_vars = run_payload.get("data_vars", {})
        if not isinstance(data_vars, Mapping):
            raise ArtifactError("Run dataset data_vars must be a mapping.")
        if not feature_specs:
            feature_specs = _default_feature_specs(data_vars)

        if node_meta is None:
            node_meta = _build_node_meta(graph_nodes, gas_species, surface_species)
            node_axis_indices["species"] = _axis_indices(
                graph_nodes,
                "species",
                gas_species,
                surface_species,
            )
            node_axis_indices["surface_species"] = _axis_indices(
                graph_nodes,
                "surface_species",
                gas_species,
                surface_species,
            )

        node_count = len(node_order)
        feature_count = len(feature_specs)
        time_len = len(time_values)
        values = [
            [
                [math.nan for _ in range(feature_count)]
                for _ in range(node_count)
            ]
            for _ in range(time_len)
        ]

        for feat_index, spec in enumerate(feature_specs):
            matrix, axis, missing = _extract_feature_matrix(
                data_vars,
                spec,
                time_values,
                gas_species,
                surface_species,
                missing_strategy,
            )
            if missing:
                missing_features.setdefault(run_id, []).append(spec.name)
                continue
            axis_indices = node_axis_indices.get(axis or "species", [])
            for t_index in range(time_len):
                row = matrix[t_index]
                for node_index, axis_index in enumerate(axis_indices):
                    if axis_index is None:
                        continue
                    try:
                        values[t_index][node_index][feat_index] = float(
                            row[axis_index]
                        )
                    except (TypeError, ValueError, IndexError) as exc:
                        raise ArtifactError(
                            f"Run dataset {spec.data_var} contains non-numeric values."
                        ) from exc

        rows.extend(
            _flatten_rows(
                run_id,
                time_values,
                node_order,
                feature_specs,
                values,
            )
        )

        time_by_run[run_id] = {
            "count": time_len,
            "values": list(time_values),
        }
        if np is not None:
            npz_payload[f"features_{run_id}"] = np.asarray(values, dtype=float)
            npz_payload[f"time_{run_id}"] = np.asarray(time_values, dtype=float)

    if node_meta is None:
        raise ConfigError("No runs provided for gnn_dataset export.")

    if np is not None:
        npz_payload["node_ids"] = np.asarray(node_order, dtype=str)
        npz_payload["feature_names"] = np.asarray(
            [spec.name for spec in feature_specs],
            dtype=str,
        )
        npz_payload["run_ids"] = np.asarray(run_ids, dtype=str)

    inputs_payload = {
        "run_ids": list(run_ids),
        "graph_id": graph_id,
        "node_kinds": list(node_kinds),
        "features": [spec.name for spec in feature_specs],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = list(dict.fromkeys(list(run_ids) + [graph_id]))
    manifest = build_manifest(
        kind="gnn_datasets",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    dataset_meta: dict[str, Any] = {
        "kind": "gnn_dataset",
        "source": {"graph_id": graph_id, "run_ids": list(run_ids)},
        "nodes": {
            "count": len(node_order),
            "order": node_order,
            "meta": node_meta,
        },
        "features": {
            "count": len(feature_specs),
            "names": [spec.name for spec in feature_specs],
            "sources": {
                spec.name: {
                    "data_var": spec.data_var,
                    "coord": spec.coord or "auto",
                }
                for spec in feature_specs
            },
        },
        "time": {"coord": "time", "by_run": time_by_run},
        "coords": {
            "species": species_ref or [],
            "surface_species": surface_ref or [],
        },
        "node_kinds": list(node_kinds),
        "missing_strategy": missing_strategy,
    }
    if missing_features:
        dataset_meta["missing_features"] = missing_features

    def _writer(base_dir: Path) -> None:
        table_path = base_dir / "node_features.parquet"
        _write_node_features_table(rows, table_path)
        files_meta = {"node_features_table": table_path.name}
        if npz_payload:
            npz_path = base_dir / "node_features.npz"
            if _write_node_features_npz(npz_path, npz_payload):
                files_meta["node_features_npz"] = npz_path.name
        dataset_meta["files"] = files_meta
        write_json_atomic(base_dir / "dataset.json", dataset_meta)

    return store.ensure(manifest, writer=_writer)


def _resolve_dataset_root(store_root: Path, dataset_name: str) -> Path:
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ConfigError("dataset_name must be a non-empty string.")
    dataset_path = Path(dataset_name)
    if dataset_path.is_absolute() or ".." in dataset_path.parts:
        raise ConfigError("dataset_name must be a relative path without '..'.")
    if len(dataset_path.parts) > 1:
        raise ConfigError("dataset_name must be a single path component.")
    run_root = store_root.parent if store_root.name == "artifacts" else store_root
    return run_root / "datasets" / dataset_path.name


def _normalize_split_cfg(
    params: Mapping[str, Any],
    resolved_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    split_cfg = params.get("split") or params.get("splits") or {}
    if split_cfg is None:
        split_cfg = {}
    if not isinstance(split_cfg, Mapping):
        raise ConfigError("split must be a mapping.")
    try:
        train_ratio = float(split_cfg.get("train_ratio", split_cfg.get("train", 0.8)))
    except (TypeError, ValueError) as exc:
        raise ConfigError("split.train_ratio must be numeric.") from exc
    val_ratio_raw = split_cfg.get("val_ratio", split_cfg.get("val"))
    if val_ratio_raw is None:
        val_ratio = max(0.0, 1.0 - train_ratio)
    else:
        try:
            val_ratio = float(val_ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("split.val_ratio must be numeric.") from exc

    if train_ratio < 0.0 or val_ratio < 0.0:
        raise ConfigError("split ratios must be non-negative.")
    if train_ratio + val_ratio > 1.0 + 1e-8:
        raise ConfigError("split ratios must sum to <= 1.0.")

    seed = split_cfg.get("seed", params.get("seed"))
    if seed is None and isinstance(resolved_cfg.get("common"), Mapping):
        seed = resolved_cfg["common"].get("seed")
    if seed is None:
        seed = 0
    try:
        seed_value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError("split.seed must be an integer.") from exc
    shuffle = split_cfg.get("shuffle", True)
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed_value,
        "shuffle": bool(shuffle),
    }


def _split_indices(
    total: int,
    split_cfg: Mapping[str, Any],
) -> dict[str, list[int]]:
    indices = list(range(total))
    if split_cfg.get("shuffle", True):
        rng = random.Random(split_cfg.get("seed", 0))
        rng.shuffle(indices)
    train_count = int(total * float(split_cfg.get("train_ratio", 0.8)))
    val_count = int(total * float(split_cfg.get("val_ratio", 0.2)))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _normalize_constraint_groups(
    value: Any,
    node_order: Sequence[str],
) -> dict[str, Any]:
    if value is None:
        return {"group_ids": [], "source": None}
    if isinstance(value, Mapping):
        group_ids = [value.get(node_id) for node_id in node_order]
        return {"group_ids": group_ids, "source": "mapping"}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        group_ids = list(value)
        if len(group_ids) != len(node_order):
            raise ConfigError("constraint_groups length must match node count.")
        return {"group_ids": group_ids, "source": "sequence"}
    raise ConfigError("constraint_groups must be a mapping or sequence.")


def _load_csr_payload(path: Path) -> dict[str, Any]:
    if np is None:
        raise ConfigError("numpy is required to load sparse matrices.")
    if not path.exists():
        raise ConfigError(f"Missing sparse matrix file: {path}")
    if sp is not None:
        try:
            return {"matrix": sp.load_npz(path)}
        except Exception as exc:
            raise ConfigError(f"Invalid sparse matrix file: {path}") from exc
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ConfigError(f"Invalid sparse matrix file: {path}") from exc
    required = {"data", "indices", "indptr", "shape"}
    if not required.issubset(payload.files):
        raise ConfigError(f"Invalid sparse matrix payload: {path}")
    return {
        "data": payload["data"],
        "indices": payload["indices"],
        "indptr": payload["indptr"],
        "shape": payload["shape"],
    }


def _csr_to_edges(payload: Mapping[str, Any]) -> tuple[list[int], list[int], list[float]]:
    if "matrix" in payload:
        matrix = payload["matrix"]
        coo = matrix.tocoo()
        rows = [int(value) for value in coo.row.tolist()]
        cols = [int(value) for value in coo.col.tolist()]
        vals = [float(value) for value in coo.data.tolist()]
        return rows, cols, vals

    data = payload.get("data")
    indices = payload.get("indices")
    indptr = payload.get("indptr")
    shape = payload.get("shape")
    if data is None or indices is None or indptr is None or shape is None:
        raise ConfigError("Sparse payload is missing required arrays.")
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_rows = int(shape[0])
    for row in range(n_rows):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        for idx in range(start, end):
            rows.append(row)
            cols.append(int(indices[idx]))
            vals.append(float(data[idx]))
    return rows, cols, vals


def _aggregate_window_vector(
    matrix: Sequence[Sequence[Any]],
    start_idx: int,
    end_idx: int,
    method: str,
) -> list[float]:
    if start_idx < 0 or end_idx < start_idx:
        raise ConfigError("Invalid window indices for feature aggregation.")
    if not matrix:
        return []
    if end_idx >= len(matrix):
        raise ConfigError("Window index exceeds time series length.")
    window = matrix[start_idx : end_idx + 1]
    if not window:
        return [math.nan for _ in range(len(matrix[0]))]
    if np is not None:
        arr = np.asarray(window, dtype=float)
        if method == "mean":
            return arr.mean(axis=0).tolist()
        if method == "sum":
            return arr.sum(axis=0).tolist()
        raise ConfigError("feature_aggregation must be 'mean' or 'sum'.")

    if method == "mean":
        total = [0.0 for _ in range(len(window[0]))]
        for row in window:
            for idx, value in enumerate(row):
                total[idx] += float(value)
        return [value / float(len(window)) for value in total]
    if method == "sum":
        total = [0.0 for _ in range(len(window[0]))]
        for row in window:
            for idx, value in enumerate(row):
                total[idx] += float(value)
        return total
    raise ConfigError("feature_aggregation must be 'mean' or 'sum'.")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:  # pragma: no cover - optional dependency
        return False


def _pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:  # pragma: no cover - optional dependency
        return False


def run_temporal_graph_pyg(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build a temporal PyG dataset from a TemporalFluxGraph artifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, gnn_cfg = _extract_gnn_cfg(resolved_cfg)
    params = _extract_params(gnn_cfg)
    run_ids = _extract_run_ids(gnn_cfg, store=store)
    graph_id = _extract_graph_id(gnn_cfg)

    if np is None:
        raise ConfigError("numpy is required to build PyG datasets.")

    feature_specs = _normalize_feature_specs(
        params.get("node_features") or params.get("features")
    )
    node_kinds = _coerce_str_sequence(
        params.get("node_kinds") or params.get("kinds"),
        "node_kinds",
    )
    if not node_kinds:
        node_kinds = list(DEFAULT_NODE_KINDS)
    if any(kind != "species" for kind in node_kinds):
        raise ConfigError("temporal_graph_pyg currently supports species nodes only.")

    missing_strategy = _normalize_missing_strategy(params.get("missing_strategy"))
    feature_agg = params.get("feature_aggregation", "mean")
    if not isinstance(feature_agg, str):
        raise ConfigError("feature_aggregation must be a string.")
    feature_agg = feature_agg.strip().lower()

    split_cfg = _normalize_split_cfg(params, resolved_cfg)
    sort_cases = params.get("sort_cases", True)

    dataset_name = params.get("dataset_name", "temporal_graph_pyg")
    dataset_root = _resolve_dataset_root(store.root, str(dataset_name))

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_payload = _load_graph_payload(graph_dir)

    nodes_raw = graph_payload.get("nodes")
    if nodes_raw is None:
        raise ConfigError("graph.json nodes are missing.")
    nodes = _normalize_nodes(nodes_raw)
    graph_nodes = _prepare_graph_nodes(nodes, node_kinds=node_kinds)
    if not graph_nodes:
        raise ConfigError("graph.json has no nodes matching requested kinds.")
    node_order = [node.get("id") for node in graph_nodes]

    species_order: list[str] = []
    species_payload = graph_payload.get("species")
    if isinstance(species_payload, Mapping):
        order = species_payload.get("order")
        if isinstance(order, Sequence) and not isinstance(
            order,
            (str, bytes, bytearray),
        ):
            species_order = [str(name) for name in order]

    graph_layers_payload = None
    species_graph = graph_payload.get("species_graph")
    if isinstance(species_graph, Mapping):
        graph_layers_payload = species_graph.get("layers")
    if not isinstance(graph_layers_payload, Sequence):
        raise ConfigError("graph.json species_graph.layers are missing.")

    layers_meta = [dict(layer) for layer in graph_layers_payload]
    if not layers_meta:
        raise ConfigError("graph.json species_graph has no layers.")

    if sort_cases:
        run_ids = sorted(run_ids)

    node_meta = None
    node_axis_indices: list[Optional[int]] = []
    window_meta: list[dict[str, Any]] = []
    edge_sets: dict[int, tuple[list[int], list[int], list[float]]] = {}

    for layer in layers_meta:
        path_value = layer.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ConfigError("graph layer path is missing.")
        layer_path = graph_dir / path_value
        payload = _load_csr_payload(layer_path)
        layer_index = int(layer.get("index", len(window_meta)))
        edge_sets[layer_index] = _csr_to_edges(payload)
        window_meta.append(
            {
                "index": layer_index,
                "window": layer.get("window", {}),
                "path": path_value,
                "nnz": layer.get("nnz"),
            }
        )

    case_items: list[dict[str, Any]] = []
    feature_names: list[str] = []
    missing_features: dict[str, list[str]] = {}
    time_by_run: dict[str, dict[str, Any]] = {}

    for run_id in run_ids:
        store.read_manifest("runs", run_id)
        run_dir = store.artifact_dir("runs", run_id)
        run_payload = load_run_dataset_payload(run_dir)
        time_values = _extract_time_values(run_payload)
        gas_species = _extract_coord_values(run_payload, "species")
        surface_species = _extract_coord_values(run_payload, "surface_species")

        if species_order and gas_species and gas_species != species_order:
            raise ConfigError("temporal_graph_pyg requires consistent species ordering.")
        if not species_order and gas_species:
            species_order = list(gas_species)

        data_vars = run_payload.get("data_vars", {})
        if not isinstance(data_vars, Mapping):
            raise ArtifactError("Run dataset data_vars must be a mapping.")
        if not feature_specs:
            feature_specs = _default_feature_specs(data_vars)
        if not feature_names:
            feature_names = [spec.name for spec in feature_specs]

        if node_meta is None:
            node_meta = _build_node_meta(graph_nodes, gas_species, surface_species)
            node_axis_indices = _axis_indices(
                graph_nodes,
                "species",
                gas_species,
                surface_species,
            )

        feature_matrices: list[Optional[list[list[Any]]]] = []
        for spec in feature_specs:
            matrix, axis, missing = _extract_feature_matrix(
                data_vars,
                spec,
                time_values,
                gas_species,
                surface_species,
                missing_strategy,
            )
            if missing:
                missing_features.setdefault(run_id, []).append(spec.name)
                feature_matrices.append(None)
                continue
            if axis != "species":
                raise ConfigError("temporal_graph_pyg only supports species features.")
            feature_matrices.append(matrix)

        for window in window_meta:
            window_info = window.get("window", {})
            start_idx = int(window_info.get("start_idx", 0))
            end_idx = int(window_info.get("end_idx", 0))
            node_count = len(node_order)
            feature_count = len(feature_specs)
            node_features = [
                [math.nan for _ in range(feature_count)]
                for _ in range(node_count)
            ]
            for feat_index, matrix in enumerate(feature_matrices):
                if matrix is None:
                    continue
                agg_values = _aggregate_window_vector(
                    matrix,
                    start_idx,
                    end_idx,
                    feature_agg,
                )
                for node_index, axis_index in enumerate(node_axis_indices):
                    if axis_index is None:
                        continue
                    try:
                        node_features[node_index][feat_index] = float(
                            agg_values[axis_index]
                        )
                    except (IndexError, TypeError, ValueError) as exc:
                        raise ArtifactError(
                            "Aggregated feature values are invalid."
                        ) from exc

            case_items.append(
                {
                    "run_id": run_id,
                    "case_id": run_id,
                    "window_id": int(window.get("index", 0)),
                    "window": window_info,
                    "features": node_features,
                }
            )

        time_by_run[run_id] = {
            "count": len(time_values),
            "values": list(time_values),
        }

    if node_meta is None:
        raise ConfigError("No runs provided for temporal_graph_pyg.")

    if not case_items:
        raise ConfigError("No dataset items produced for temporal_graph_pyg.")
    split_indices = _split_indices(len(case_items), split_cfg)
    constraint_meta = _normalize_constraint_groups(
        params.get("constraint_groups"), node_order
    )

    inputs_payload = {
        "run_ids": list(run_ids),
        "graph_id": graph_id,
        "node_kinds": list(node_kinds),
        "features": feature_names,
        "feature_aggregation": feature_agg,
        "split": dict(split_cfg),
        "dataset_name": str(dataset_name),
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = list(dict.fromkeys(list(run_ids) + [graph_id]))
    manifest = build_manifest(
        kind="gnn_datasets",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    dataset_dir = dataset_root / artifact_id
    dataset_meta: dict[str, Any] = {
        "schema_version": 1,
        "kind": "temporal_graph_pyg",
        "id": artifact_id,
        "created_at": utc_now_iso(),
        "source": {"graph_id": graph_id, "run_ids": list(run_ids)},
        "nodes": {"count": len(node_order), "order": node_order, "meta": node_meta},
        "species": {"order": species_order or []},
        "features": {"count": len(feature_names), "names": feature_names},
        "constraints": constraint_meta,
        "windows": window_meta,
        "time": {"coord": "time", "by_run": time_by_run},
        "splits": split_indices,
        "split_config": dict(split_cfg),
        "keys": [
            {"case_id": item["case_id"], "window_id": item["window_id"]}
            for item in case_items
        ],
        "dataset_root": str(dataset_dir),
    }
    if missing_features:
        dataset_meta["missing_features"] = missing_features

    def _writer(base_dir: Path) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        files_meta: dict[str, Any] = {}
        if _torch_available() and _pyg_available():
            import torch
            from torch_geometric.data import Data

            data_list = []
            for item in case_items:
                window_id = item["window_id"]
                if window_id not in edge_sets:
                    raise ConfigError(
                        f"Missing edge data for window_id={window_id}."
                    )
                rows, cols, vals = edge_sets[window_id]
                if rows:
                    edge_index = torch.tensor([rows, cols], dtype=torch.long)
                    edge_attr = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
                    edge_weight = edge_attr.view(-1)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, 1), dtype=torch.float32)
                    edge_weight = torch.empty((0,), dtype=torch.float32)
                x = torch.tensor(item["features"], dtype=torch.float32)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data.edge_weight = edge_weight
                data.case_id = item["case_id"]
                data.run_id = item["run_id"]
                data.window_id = window_id
                window_info = item.get("window", {})
                data.window_start = float(window_info.get("start_time", 0.0))
                data.window_end = float(window_info.get("end_time", 0.0))
                data.window_index = int(window_info.get("start_idx", 0))
                data_list.append(data)

            data_path = dataset_dir / "data.pt"
            torch.save(
                {
                    "data_list": data_list,
                    "splits": split_indices,
                    "keys": dataset_meta["keys"],
                },
                data_path,
            )
            files_meta["data_pt"] = data_path.name
            dataset_meta["format"] = "pyg"
        else:
            data_path = dataset_dir / "data.json"
            payload_items: list[dict[str, Any]] = []
            for item in case_items:
                window_id = item["window_id"]
                if window_id not in edge_sets:
                    raise ConfigError(
                        f"Missing edge data for window_id={window_id}."
                    )
                rows, cols, vals = edge_sets[window_id]
                payload_items.append(
                    {
                        "case_id": item["case_id"],
                        "run_id": item["run_id"],
                        "window_id": window_id,
                        "window": item.get("window", {}),
                        "x": item["features"],
                        "edge_index": [rows, cols],
                        "edge_attr": vals,
                    }
                )
            write_json_atomic(
                data_path,
                {"items": payload_items, "splits": split_indices},
            )
            files_meta["data_json"] = data_path.name
            dataset_meta["format"] = "json"
            dataset_meta["note"] = (
                "Install rxn-platform[gnn] for torch/torch_geometric output."
            )

        dataset_meta["files"] = files_meta
        write_json_atomic(dataset_dir / "dataset.json", dataset_meta)
        write_json_atomic(base_dir / "dataset.json", dataset_meta)

    return store.ensure(manifest, writer=_writer)


register("task", "gnn_dataset.temporal_graph_pyg", run_temporal_graph_pyg)
register("task", "gnn_dataset.export", run)
register("task", "gnn_dataset.run", run)

__all__ = ["NodeFeatureSpec", "run", "run_temporal_graph_pyg"]
