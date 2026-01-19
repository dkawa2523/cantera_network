"""Visualization task entrypoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import html
import json
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
from rxn_platform.reporting import render_report_html
from rxn_platform.store import ArtifactCacheResult, ArtifactStore


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


def _extract_viz_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "viz" in cfg:
        viz_cfg = cfg.get("viz")
        if not isinstance(viz_cfg, Mapping):
            raise ConfigError("viz config must be a mapping.")
        return dict(cfg), dict(viz_cfg)
    if "name" in cfg:
        viz_cfg = dict(cfg)
        return {"viz": viz_cfg}, viz_cfg
    raise ConfigError("viz config is missing.")


def _normalize_inputs(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    inputs: list[dict[str, str]] = []
    if isinstance(raw, Mapping):
        for kind, value in raw.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                items: Sequence[Any] = list(value)
            else:
                items = [value]
            for item in items:
                if not isinstance(item, str) or not item.strip():
                    raise ConfigError("viz.inputs values must be non-empty strings.")
                kind_str = str(kind)
                if not kind_str.strip():
                    raise ConfigError("viz.inputs kind keys must be non-empty strings.")
                inputs.append({"kind": kind_str, "id": item})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, entry in enumerate(raw):
            if not isinstance(entry, Mapping):
                raise ConfigError(
                    "viz.inputs list entries must be mappings with kind/id."
                )
            kind = entry.get("kind")
            artifact_id = entry.get("id") or entry.get("artifact_id")
            if not isinstance(kind, str) or not kind.strip():
                raise ConfigError(
                    f"viz.inputs[{index}].kind must be a non-empty string."
                )
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                raise ConfigError(
                    f"viz.inputs[{index}].id must be a non-empty string."
                )
            inputs.append({"kind": kind, "id": artifact_id})
    else:
        raise ConfigError("viz.inputs must be a mapping or list of mappings.")

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in inputs:
        key = (item["kind"], item["id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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


def _normalize_name_list(raw: Any, field_name: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            raise ConfigError(f"{field_name} must be a non-empty string.")
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        names: list[str] = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError(
                    f"{field_name}[{index}] must be a non-empty string."
                )
            names.append(entry)
        return names
    raise ConfigError(f"{field_name} must be a string or list of strings.")


def _normalize_field_specs(raw: Any, field_name: str) -> list[dict[str, str]]:
    if raw is None:
        return []
    specs: list[dict[str, str]] = []
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("path", "key", "field")):
            path = raw.get("path") or raw.get("key") or raw.get("field")
            label = raw.get("label") or path
            if not isinstance(path, str) or not path.strip():
                raise ConfigError(f"{field_name}.path must be a non-empty string.")
            if not isinstance(label, str) or not label.strip():
                raise ConfigError(f"{field_name}.label must be a non-empty string.")
            specs.append({"path": path, "label": label})
        else:
            for key, label in raw.items():
                if not isinstance(key, str) or not key.strip():
                    raise ConfigError(f"{field_name} keys must be non-empty strings.")
                if label is None:
                    label = key
                if not isinstance(label, str) or not label.strip():
                    raise ConfigError(f"{field_name} labels must be non-empty strings.")
                specs.append({"path": key, "label": str(label)})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, entry in enumerate(raw):
            if isinstance(entry, str):
                if not entry.strip():
                    raise ConfigError(
                        f"{field_name}[{index}] must be a non-empty string."
                    )
                specs.append({"path": entry, "label": entry})
                continue
            if isinstance(entry, Mapping):
                path = entry.get("path") or entry.get("key") or entry.get("field")
                label = entry.get("label") or path
                if not isinstance(path, str) or not path.strip():
                    raise ConfigError(
                        f"{field_name}[{index}].path must be a non-empty string."
                    )
                if not isinstance(label, str) or not label.strip():
                    raise ConfigError(
                        f"{field_name}[{index}].label must be a non-empty string."
                    )
                specs.append({"path": path, "label": label})
                continue
            raise ConfigError(
                f"{field_name}[{index}] must be a string or mapping with path/label."
            )
    else:
        raise ConfigError(f"{field_name} must be a mapping or list.")

    label_counts: dict[str, int] = {}
    for spec in specs:
        label = spec["label"]
        if label in label_counts:
            label_counts[label] += 1
            spec["label"] = f"{label} ({label_counts[label]})"
        else:
            label_counts[label] = 1
    return specs


def _infer_condition_fields(
    run_manifests: Sequence[ArtifactManifest],
    limit: int,
) -> list[dict[str, str]]:
    if not run_manifests or limit <= 0:
        return []
    config = run_manifests[0].config
    if not isinstance(config, Mapping):
        return []
    paths: list[str] = []

    def _walk(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value.keys(), key=str):
                child = value[key]
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child, next_prefix)
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if prefix:
                paths.append(prefix)

    _walk(config, "")
    return [{"path": path, "label": path} for path in paths[:limit]]


def _extract_manifest_value(manifest: ArtifactManifest, path: str) -> Any:
    if path == "id":
        return manifest.id
    payload: Any = manifest.config
    remaining = path
    if path.startswith("config."):
        remaining = path[len("config.") :]
        payload = manifest.config
    elif path.startswith("inputs."):
        remaining = path[len("inputs.") :]
        payload = manifest.inputs
    if not remaining:
        return None
    for part in remaining.split("."):
        if isinstance(payload, Mapping) and part in payload:
            payload = payload[part]
        else:
            return None
    return payload


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _coerce_positive_int(value: Any, label: str, *, default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer.") from exc
    if number <= 0:
        raise ConfigError(f"{label} must be a positive integer.")
    return number


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("rows", [])
    elif isinstance(payload, Sequence):
        rows = payload
    else:
        rows = []
    if not isinstance(rows, Sequence):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


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
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:
        raise ArtifactError(
            "Run dataset not found; install xarray to load state.zarr."
        ) from exc
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


def _extract_coord_values(payload: Mapping[str, Any], name: str) -> list[Any]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    entry = coords.get(name)
    if not isinstance(entry, Mapping):
        raise ArtifactError(f"Run dataset coords.{name} is missing.")
    data = entry.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ArtifactError(f"Run dataset coords.{name}.data must be a sequence.")
    return list(data)


def _extract_data_var(payload: Mapping[str, Any], name: str) -> tuple[list[str], Any]:
    data_vars = payload.get("data_vars", {})
    if not isinstance(data_vars, Mapping):
        raise ArtifactError("Run dataset data_vars must be a mapping.")
    entry = data_vars.get(name)
    if not isinstance(entry, Mapping):
        raise ArtifactError(f"Run dataset data_vars.{name} is missing.")
    dims = entry.get("dims")
    data = entry.get("data")
    if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes, bytearray)):
        raise ArtifactError(f"Run dataset data_vars.{name}.dims must be a sequence.")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ArtifactError(f"Run dataset data_vars.{name}.data must be a sequence.")
    return [str(dim) for dim in dims], data


def _coerce_float_list(values: Sequence[Any], label: str) -> list[float]:
    numbers: list[float] = []
    for index, value in enumerate(values):
        number = _coerce_float(value)
        if number is None:
            raise ArtifactError(f"{label}[{index}] must be numeric.")
        numbers.append(number)
    return numbers


def _normalize_time_matrix(
    dims: Sequence[str],
    data: Any,
    axis: str,
) -> list[list[float]]:
    if len(dims) != 2:
        raise ArtifactError("Run dataset time series must be 2D.")
    if "time" not in dims or axis not in dims:
        raise ArtifactError(f"Run dataset dims must include time and {axis}.")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise ArtifactError("Run dataset matrix must be a sequence.")
    if dims == ["time", axis]:
        matrix_raw = data
    elif dims == [axis, "time"]:
        matrix_raw = list(zip(*data))
    else:
        raise ArtifactError(
            f"Run dataset dims ordering not supported: {dims!r}."
        )
    matrix: list[list[float]] = []
    for row_index, row in enumerate(matrix_raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ArtifactError(
                f"Run dataset matrix row {row_index} must be a sequence."
            )
        matrix.append(
            _coerce_float_list(row, f"run data row {row_index}")
        )
    return matrix


def _prepare_timeseries(
    payload: Mapping[str, Any],
    var_name: str,
    axis: str,
) -> tuple[
    Optional[list[float]],
    Optional[list[str]],
    Optional[list[list[float]]],
    Optional[str],
]:
    try:
        time_values = _coerce_float_list(
            _extract_coord_values(payload, "time"),
            "coords.time",
        )
        axis_values = [
            str(value) for value in _extract_coord_values(payload, axis)
        ]
        dims, data = _extract_data_var(payload, var_name)
        matrix = _normalize_time_matrix(dims, data, axis)
        if len(matrix) != len(time_values):
            raise ArtifactError("Run dataset time dimension length mismatch.")
        return time_values, axis_values, matrix, None
    except Exception as exc:
        return None, None, None, str(exc)


def _downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= max_points:
        return list(range(length))
    step = max(1, length // max_points)
    indices = list(range(0, length, step))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def _series_stat(
    series: Sequence[float],
    time_values: Optional[Sequence[float]],
    stat: str,
) -> float:
    if not series:
        return math.nan
    if stat == "last":
        return series[-1]
    if stat == "mean":
        return sum(series) / float(len(series))
    if stat == "max":
        return max(series)
    if stat == "min":
        return min(series)
    if stat == "integral":
        if time_values is None or len(time_values) != len(series):
            return sum(series)
        integral = 0.0
        for index in range(1, len(series)):
            dt = time_values[index] - time_values[index - 1]
            integral += 0.5 * (series[index] + series[index - 1]) * dt
        return integral
    raise ConfigError(f"Unsupported stat: {stat!r}.")


def _rank_entities(
    axis_names: Sequence[str],
    matrix: Sequence[Sequence[float]],
    time_values: Optional[Sequence[float]],
    *,
    stat: str,
    rank_abs: bool,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for idx, name in enumerate(axis_names):
        series = []
        for row in matrix:
            if idx >= len(row):
                continue
            series.append(row[idx])
        value = _series_stat(series, time_values, stat)
        ranked.append((name, value))
    ranked.sort(key=lambda item: abs(item[1]) if rank_abs else item[1], reverse=True)
    return ranked


def _parse_meta_json(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def _parse_feature_label(
    feature: Any,
    prefix: str,
) -> Optional[tuple[str, str]]:
    if not isinstance(feature, str):
        return None
    needle = f"{prefix}."
    if not feature.startswith(needle):
        return None
    remainder = feature[len(needle) :]
    parts = remainder.split(".")
    if len(parts) < 2:
        return None
    stat = parts[-1]
    entity = ".".join(parts[:-1])
    return entity, stat


def _rank_from_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    data_var: str,
    id_label: str,
    stat: str,
    top_n: int,
    rank_abs: bool,
    run_id: Optional[str],
) -> list[tuple[str, float]]:
    values_by_id: dict[str, list[float]] = {}
    source_match = f"data_vars.{data_var}" if data_var else ""
    for row in rows:
        if run_id is not None and row.get("run_id") != run_id:
            continue
        meta = _parse_meta_json(row.get("meta_json"))
        feature_name = row.get("feature")
        parsed = _parse_feature_label(feature_name, prefix)
        if source_match:
            if meta.get("source") != source_match and parsed is None:
                continue
        if parsed is None and not meta:
            continue
        row_stat = meta.get("stat")
        entity = meta.get(id_label)
        if parsed is not None:
            if row_stat is None:
                row_stat = parsed[1]
            if entity is None:
                entity = parsed[0]
        if row_stat != stat:
            continue
        if not isinstance(entity, str) or not entity.strip():
            continue
        value = _coerce_float(row.get("value"))
        if value is None:
            continue
        values_by_id.setdefault(entity, []).append(value)

    ranked: list[tuple[str, float]] = []
    for entity, values in values_by_id.items():
        if not values:
            continue
        mean_value = sum(values) / float(len(values))
        ranked.append((entity, mean_value))
    ranked.sort(key=lambda item: abs(item[1]) if rank_abs else item[1], reverse=True)
    return ranked[:top_n]


def _rank_from_run_payload(
    payload: Mapping[str, Any],
    *,
    var_name: str,
    axis: str,
    stat: str,
    top_n: int,
    rank_abs: bool,
) -> tuple[list[tuple[str, float]], Optional[str]]:
    time_values, axis_values, matrix, error = _prepare_timeseries(
        payload, var_name, axis
    )
    if error is not None or time_values is None or axis_values is None or matrix is None:
        return [], error or "time series data missing"
    ranked = _rank_entities(
        axis_values,
        matrix,
        time_values,
        stat=stat,
        rank_abs=rank_abs,
    )
    return ranked[:top_n], None


def _plotly_context() -> Optional[tuple[Any, Any]]:
    try:
        import plotly.graph_objects as go  # type: ignore
        import plotly.io as pio  # type: ignore
    except Exception:
        return None
    return go, pio


def _plotly_html(fig: Any, pio: Any, state: dict[str, bool]) -> str:
    include_js = state.get("include_js", True)
    state["include_js"] = False
    include_plotlyjs: Any = "inline" if include_js else False
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displayModeBar": False},
    )


def _wrap_card(contents: str) -> str:
    return (
        "<div style=\"border:1px solid var(--border);border-radius:16px;padding:12px;background:#fbfaf7;\">"
        + contents
        + "</div>"
    )


def _panel(title: str, contents: str) -> str:
    return (
        "<section class=\"panel\">"
        f"<h2>{html.escape(title)}</h2>"
        f"{contents}"
        "</section>"
    )


def _render_message_list(lines: Sequence[str], empty_text: str) -> str:
    if not lines:
        return f"<p class=\"muted\">{html.escape(empty_text)}</p>"
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return "<ul>" + items + "</ul>"


def _summarize_values(values: Sequence[Any]) -> str:
    numeric = [number for value in values if (number := _coerce_float(value)) is not None]
    if numeric:
        mean = sum(numeric) / len(numeric)
        return (
            f"n={len(numeric)}, "
            f"min={min(numeric):.4g}, "
            f"max={max(numeric):.4g}, "
            f"mean={mean:.4g}"
        )
    normalized = [str(value) for value in values if value is not None]
    if not normalized:
        return "no data"
    counts = Counter(normalized)
    most_common = ", ".join(
        f"{label} ({count})" for label, count in counts.most_common(3)
    )
    return f"n={len(normalized)}, unique={len(counts)}, top={most_common}"


def _merge_inputs(*input_groups: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, str]] = []
    for group in input_groups:
        for entry in group:
            kind = entry.get("kind")
            artifact_id = entry.get("id")
            if not isinstance(kind, str) or not isinstance(artifact_id, str):
                continue
            key = (kind, artifact_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"kind": kind, "id": artifact_id})
    return merged


def _normalize_group_inputs(raw: Any, field_name: str) -> dict[str, list[dict[str, str]]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{field_name} must be a mapping.")
    groups: dict[str, list[dict[str, str]]] = {}
    for group_name, group_cfg in raw.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ConfigError(f"{field_name} keys must be non-empty strings.")
        groups[group_name] = _normalize_inputs(group_cfg)
    return groups


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    empty_text: str,
) -> str:
    if not rows:
        return f"<p class=\"muted\">{html.escape(empty_text)}</p>"
    header_html = "".join(
        f"<th style=\"text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);\">"
        f"{html.escape(header)}</th>"
        for header in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td style=\"padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top;\">"
            f"{html.escape(str(cell))}</td>"
            for cell in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table style=\"width:100%;border-collapse:collapse;\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _read_patch_payload(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, f"{path.name} not found"
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            if yaml is None:
                payload = json.load(handle)
            else:
                payload = yaml.safe_load(handle)
    except Exception as exc:
        return None, f"Failed to read {path.name}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"{path.name} must contain a mapping."
    return dict(payload), None


def _extract_reaction_count_from_graph_payload(
    payload: Mapping[str, Any],
) -> Optional[int]:
    reactions = payload.get("reactions")
    if isinstance(reactions, Sequence) and not isinstance(
        reactions, (str, bytes, bytearray)
    ):
        return len(reactions)
    graph_payload: Mapping[str, Any] = payload
    bipartite = payload.get("bipartite")
    if isinstance(bipartite, Mapping):
        data = bipartite.get("data")
        if isinstance(data, Mapping):
            graph_payload = data
    nodes = graph_payload.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes, bytearray)):
        count = sum(
            1
            for node in nodes
            if isinstance(node, Mapping) and node.get("kind") == "reaction"
        )
        if count:
            return count
    return None


def _infer_reaction_count(
    viz_cfg: Mapping[str, Any],
    graph_manifests: Sequence[ArtifactManifest],
    store: ArtifactStore,
    notices: list[str],
) -> Optional[int]:
    reduction_cfg = viz_cfg.get("reduction")
    if reduction_cfg is None:
        reduction_cfg = {}
    if not isinstance(reduction_cfg, Mapping):
        raise ConfigError("viz.reduction must be a mapping when provided.")
    for source in (reduction_cfg, viz_cfg):
        for key in ("baseline_reactions", "reaction_count", "total_reactions"):
            if key in source:
                value = _coerce_optional_int(source.get(key))
                if value is None or value <= 0:
                    notices.append(f"{key} must be a positive integer.")
                else:
                    return value
    if not graph_manifests:
        return None
    if len(graph_manifests) > 1:
        notices.append("Multiple graphs provided; using the first for reaction count.")
    graph_manifest = graph_manifests[0]
    graph_path = store.artifact_dir("graphs", graph_manifest.id) / "graph.json"
    if not graph_path.exists():
        notices.append(f"graph.json missing for graphs/{graph_manifest.id}.")
        return None
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        notices.append(f"graph.json invalid for graphs/{graph_manifest.id}: {exc}")
        return None
    if not isinstance(payload, Mapping):
        notices.append(f"graph.json payload invalid for graphs/{graph_manifest.id}.")
        return None
    count = _extract_reaction_count_from_graph_payload(payload)
    if count is None:
        notices.append(f"No reaction count found in graphs/{graph_manifest.id}.")
    return count


def _collect_condition_values(
    run_manifests: Sequence[ArtifactManifest],
    field_specs: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    values_by_label: dict[str, list[Any]] = {}
    by_run_by_label: dict[str, dict[str, Any]] = {}
    for spec in field_specs:
        label = str(spec["label"])
        path = str(spec["path"])
        values: list[Any] = []
        by_run: dict[str, Any] = {}
        for manifest in run_manifests:
            value = _extract_manifest_value(manifest, path)
            normalized = _normalize_scalar(value)
            if normalized is None:
                continue
            values.append(normalized)
            by_run[manifest.id] = normalized
        values_by_label[label] = values
        by_run_by_label[label] = by_run
    return values_by_label, by_run_by_label


def _reaction_label(row: Mapping[str, Any]) -> str:
    reaction_id = row.get("reaction_id")
    if isinstance(reaction_id, str) and reaction_id.strip():
        return reaction_id
    reaction_index = row.get("reaction_index")
    if reaction_index is not None:
        return f"r{reaction_index}"
    return "unknown"


def _inject_section(html_doc: str, section_html: str) -> str:
    marker = "\n  <script type=\"application/json\" id=\"report-config\">"
    if marker in html_doc:
        return html_doc.replace(marker, section_html + marker, 1)
    return html_doc + section_html


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Render a base report from existing artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, viz_cfg = _extract_viz_cfg(resolved_cfg)
    raw_inputs = viz_cfg.get("inputs", viz_cfg.get("artifacts"))
    input_specs = _normalize_inputs(raw_inputs)

    input_manifests = []
    for entry in input_specs:
        manifest = store.read_manifest(entry["kind"], entry["id"])
        input_manifests.append(manifest)

    inputs_payload = {"artifacts": input_specs}
    parent_ids = [entry["id"] for entry in input_specs]
    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    report_manifest = ArtifactManifest(
        schema_version=1,
        kind="reports",
        id=report_id,
        created_at=_utc_now_iso(),
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    title = viz_cfg.get("title") or "rxn_platform Report"
    if not isinstance(title, str):
        raise ConfigError("viz.title must be a string.")
    dashboard = viz_cfg.get("dashboard") or "base"
    if not isinstance(dashboard, str):
        raise ConfigError("viz.dashboard must be a string.")

    placeholders = viz_cfg.get("placeholders")
    if placeholders is None:
        placeholders = ("DS dashboard", "Chem dashboard", "Notes")
    if not isinstance(placeholders, Sequence) or isinstance(
        placeholders, (str, bytes, bytearray)
    ):
        raise ConfigError("viz.placeholders must be a list of strings.")
    placeholder_labels: list[str] = []
    for label in placeholders:
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("viz.placeholders entries must be non-empty strings.")
        placeholder_labels.append(label)

    html = render_report_html(
        title=title,
        dashboard=dashboard,
        created_at=report_manifest.created_at,
        manifest=report_manifest,
        inputs=input_specs,
        config=manifest_cfg,
        placeholders=placeholder_labels,
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / "index.html").write_text(html, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


register("task", "viz.base", run)


def ds_dashboard(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Render a DS dashboard report from observables and sensitivity artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, viz_cfg = _extract_viz_cfg(resolved_cfg)

    raw_inputs = viz_cfg.get("inputs", viz_cfg.get("artifacts"))
    if raw_inputs is None and any(
        key in viz_cfg for key in ("runs", "observables", "sensitivity")
    ):
        raw_inputs = {
            "runs": viz_cfg.get("runs"),
            "observables": viz_cfg.get("observables"),
            "sensitivity": viz_cfg.get("sensitivity"),
        }
    input_specs = _normalize_inputs(raw_inputs)

    notices: list[str] = []
    missing_inputs: list[str] = []
    input_manifests: list[ArtifactManifest] = []
    for entry in input_specs:
        try:
            manifest = store.read_manifest(entry["kind"], entry["id"])
        except ArtifactError as exc:
            missing_inputs.append(f"{entry['kind']}/{entry['id']}: {exc}")
            continue
        input_manifests.append(manifest)

    inputs_payload = {"artifacts": input_specs}
    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parent_ids = [manifest.id for manifest in input_manifests]
    notes = None
    if missing_inputs:
        notes = "Missing inputs: " + "; ".join(missing_inputs)

    report_manifest = ArtifactManifest(
        schema_version=1,
        kind="reports",
        id=report_id,
        created_at=_utc_now_iso(),
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
        notes=notes,
    )

    title = viz_cfg.get("title") or "DS Dashboard"
    if not isinstance(title, str):
        raise ConfigError("viz.title must be a string.")
    dashboard = viz_cfg.get("dashboard") or "ds"
    if not isinstance(dashboard, str):
        raise ConfigError("viz.dashboard must be a string.")

    placeholders = viz_cfg.get("placeholders")
    if placeholders is None:
        placeholders = ("Convergence",)
    if not isinstance(placeholders, Sequence) or isinstance(
        placeholders, (str, bytes, bytearray)
    ):
        raise ConfigError("viz.placeholders must be a list of strings.")
    placeholder_labels: list[str] = []
    for label in placeholders:
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("viz.placeholders entries must be non-empty strings.")
        placeholder_labels.append(label)

    run_manifests = [m for m in input_manifests if m.kind == "runs"]
    observables_manifests = [m for m in input_manifests if m.kind == "observables"]
    sensitivity_manifests = [m for m in input_manifests if m.kind == "sensitivity"]

    condition_fields = _normalize_field_specs(
        viz_cfg.get("condition_fields", viz_cfg.get("conditions")),
        "viz.condition_fields",
    )
    if not condition_fields:
        auto_limit = viz_cfg.get("condition_auto_limit", 4)
        if not isinstance(auto_limit, int) or auto_limit <= 0:
            raise ConfigError("viz.condition_auto_limit must be a positive int.")
        condition_fields = _infer_condition_fields(run_manifests, auto_limit)
        if condition_fields:
            notices.append("Condition fields inferred from run config.")
    condition_values, condition_by_run = _collect_condition_values(
        run_manifests,
        condition_fields,
    )

    plotly_ctx = _plotly_context()
    plotly_state = {"include_js": True}
    if plotly_ctx is None:
        notices.append("Plotly not available; using summaries instead of charts.")
    else:
        go, pio = plotly_ctx

    condition_cards: list[str] = []
    if condition_fields and run_manifests:
        for spec in condition_fields:
            label = str(spec["label"])
            values = condition_values.get(label, [])
            if not values:
                card_html = f"<p class=\"muted\">No values for {html.escape(label)}.</p>"
                condition_cards.append(_wrap_card(card_html))
                continue
            if plotly_ctx is None:
                summary = _summarize_values(values)
                condition_cards.append(
                    _wrap_card(
                        f"<strong>{html.escape(label)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                    )
                )
                continue
            fig = go.Figure()
            fig.add_histogram(x=values, nbinsx=20)
            fig.update_layout(
                title=label,
                template="plotly_white",
                height=260,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            condition_cards.append(_wrap_card(_plotly_html(fig, pio, plotly_state)))
    condition_body = (
        "<div class=\"grid\">" + "".join(condition_cards) + "</div>"
        if condition_cards
        else "<p class=\"muted\">No condition data available.</p>"
    )
    condition_section = _panel("Condition Distribution", condition_body)

    objective_cfg = viz_cfg.get("objective", {})
    if objective_cfg is None:
        objective_cfg = {}
    if not isinstance(objective_cfg, Mapping):
        raise ConfigError("viz.objective must be a mapping when provided.")
    objective_names = _normalize_name_list(
        objective_cfg.get("observables", objective_cfg.get("observable")),
        "viz.objective.observables",
    )
    max_objectives = objective_cfg.get("max_series", 3)
    if not isinstance(max_objectives, int) or max_objectives <= 0:
        raise ConfigError("viz.objective.max_series must be a positive int.")

    observable_rows: list[dict[str, Any]] = []
    for manifest in observables_manifests:
        table_path = store.artifact_dir("observables", manifest.id) / "values.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read observables/{manifest.id}: {exc}")
            continue
        observable_rows.extend(rows)
    if not objective_names:
        unique_names = sorted(
            {
                str(row.get("observable"))
                for row in observable_rows
                if isinstance(row.get("observable"), str)
                and str(row.get("observable")).strip()
            }
        )
        objective_names = unique_names[:max_objectives]

    scatter_field = objective_cfg.get("condition_field")
    scatter_label = None
    scatter_by_run: dict[str, Any] = {}
    if scatter_field:
        if not isinstance(scatter_field, str) or not scatter_field.strip():
            raise ConfigError("viz.objective.condition_field must be a non-empty string.")
        for spec in condition_fields:
            if scatter_field in (spec["label"], spec["path"]):
                scatter_label = str(spec["label"])
                scatter_by_run = condition_by_run.get(scatter_label, {})
                break
        if scatter_label is None:
            scatter_label = scatter_field
            scatter_by_run = _collect_condition_values(
                run_manifests,
                [{"label": scatter_label, "path": scatter_field}],
            )[1].get(scatter_label, {})
    elif condition_fields:
        scatter_label = str(condition_fields[0]["label"])
        scatter_by_run = condition_by_run.get(scatter_label, {})

    objective_cards: list[str] = []
    if objective_names and observable_rows:
        for name in objective_names:
            values: list[float] = []
            run_entries: list[tuple[str, float]] = []
            units: list[str] = []
            for row in observable_rows:
                if row.get("observable") != name:
                    continue
                value = _coerce_float(row.get("value"))
                if value is None:
                    continue
                run_id = row.get("run_id")
                if isinstance(run_id, str) and run_id.strip():
                    run_entries.append((run_id, value))
                values.append(value)
                unit = row.get("unit")
                if isinstance(unit, str) and unit.strip():
                    units.append(unit)
            if not values:
                objective_cards.append(
                    _wrap_card(
                        f"<p class=\"muted\">No objective values for {html.escape(name)}.</p>"
                    )
                )
                continue
            unit_label = ""
            if units:
                unit_label = Counter(units).most_common(1)[0][0]
            if plotly_ctx is None:
                summary = _summarize_values(values)
                objective_cards.append(
                    _wrap_card(
                        f"<strong>{html.escape(name)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                    )
                )
                continue
            hist_title = name if not unit_label else f"{name} ({unit_label})"
            fig = go.Figure()
            fig.add_histogram(x=values, nbinsx=20)
            fig.update_layout(
                title=f"Objective Distribution: {hist_title}",
                template="plotly_white",
                height=260,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            objective_cards.append(_wrap_card(_plotly_html(fig, pio, plotly_state)))

            if scatter_label and scatter_by_run:
                scatter_x: list[Any] = []
                scatter_y: list[float] = []
                for run_id, value in run_entries:
                    x_val = scatter_by_run.get(run_id)
                    if x_val is None:
                        continue
                    scatter_x.append(x_val)
                    scatter_y.append(value)
                if scatter_x and scatter_y:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_x,
                            y=scatter_y,
                            mode="markers",
                            marker=dict(size=8, color="#0f6f68", opacity=0.75),
                        )
                    )
                    fig.update_layout(
                        title=f"{name} vs {scatter_label}",
                        xaxis_title=scatter_label,
                        yaxis_title=hist_title,
                        template="plotly_white",
                        height=260,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    objective_cards.append(
                        _wrap_card(_plotly_html(fig, pio, plotly_state))
                    )
    objective_body = (
        "<div class=\"grid\">" + "".join(objective_cards) + "</div>"
        if objective_cards
        else "<p class=\"muted\">No objective data available.</p>"
    )
    objective_section = _panel("Objective Overview", objective_body)

    sensitivity_cfg = viz_cfg.get("sensitivity", {})
    if sensitivity_cfg is None:
        sensitivity_cfg = {}
    if not isinstance(sensitivity_cfg, Mapping):
        raise ConfigError("viz.sensitivity must be a mapping when provided.")
    sensitivity_targets = _normalize_name_list(
        sensitivity_cfg.get("targets", sensitivity_cfg.get("target")),
        "viz.sensitivity.targets",
    )
    sensitivity_top_n = sensitivity_cfg.get("top_n", 12)
    if not isinstance(sensitivity_top_n, int) or sensitivity_top_n <= 0:
        raise ConfigError("viz.sensitivity.top_n must be a positive int.")
    max_targets = sensitivity_cfg.get("max_targets", 4)
    if not isinstance(max_targets, int) or max_targets <= 0:
        raise ConfigError("viz.sensitivity.max_targets must be a positive int.")
    rank_by = sensitivity_cfg.get("rank_by", "abs")
    if rank_by not in ("abs", "value"):
        raise ConfigError("viz.sensitivity.rank_by must be 'abs' or 'value'.")

    sensitivity_rows: list[dict[str, Any]] = []
    for manifest in sensitivity_manifests:
        table_path = store.artifact_dir("sensitivity", manifest.id) / "sensitivity.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read sensitivity/{manifest.id}: {exc}")
            continue
        sensitivity_rows.extend(rows)

    if not sensitivity_targets:
        unique_targets = sorted(
            {
                str(row.get("target"))
                for row in sensitivity_rows
                if isinstance(row.get("target"), str)
                and str(row.get("target")).strip()
            }
        )
        sensitivity_targets = unique_targets[:max_targets]

    condition_id = sensitivity_cfg.get("condition_id")
    if condition_id is not None and (
        not isinstance(condition_id, str) or not condition_id.strip()
    ):
        raise ConfigError("viz.sensitivity.condition_id must be a non-empty string.")
    if condition_id is None:
        condition_ids = sorted(
            {
                str(row.get("condition_id"))
                for row in sensitivity_rows
                if row.get("condition_id") is not None
            }
        )
        if len(condition_ids) > 1:
            condition_id = condition_ids[0]
            notices.append(
                f"Multiple condition_id values found; using {condition_id}."
            )

    if condition_id:
        sensitivity_rows = [
            row
            for row in sensitivity_rows
            if str(row.get("condition_id", "")) == condition_id
        ]

    sensitivity_body = "<p class=\"muted\">No sensitivity data available.</p>"
    if sensitivity_rows and sensitivity_targets:
        values_by_key: dict[tuple[str, str], list[float]] = {}
        for row in sensitivity_rows:
            target = row.get("target")
            if target not in sensitivity_targets:
                continue
            value = _coerce_float(row.get("value"))
            if value is None:
                continue
            reaction = _reaction_label(row)
            key = (reaction, str(target))
            values_by_key.setdefault(key, []).append(value)

        values_by_target: dict[str, list[tuple[str, float]]] = {
            target: [] for target in sensitivity_targets
        }
        for (reaction, target), values in values_by_key.items():
            if not values:
                continue
            mean_value = sum(values) / len(values)
            values_by_target[target].append((reaction, mean_value))

        reactions: set[str] = set()
        for target, items in values_by_target.items():
            items_sorted = sorted(
                items,
                key=lambda item: abs(item[1]) if rank_by == "abs" else item[1],
                reverse=True,
            )
            reactions.update([reaction for reaction, _ in items_sorted[:sensitivity_top_n]])

        if reactions:
            reaction_scores: dict[str, float] = {}
            for reaction in reactions:
                values = [
                    abs(value)
                    for (r_label, target), values in values_by_key.items()
                    for value in values
                    if r_label == reaction and target in sensitivity_targets
                ]
                reaction_scores[reaction] = max(values) if values else 0.0
            reaction_order = sorted(
                reaction_scores.keys(),
                key=lambda item: reaction_scores[item],
                reverse=True,
            )
            z_values: list[list[Optional[float]]] = []
            for reaction in reaction_order:
                row_values: list[Optional[float]] = []
                for target in sensitivity_targets:
                    values = values_by_key.get((reaction, target), [])
                    if values:
                        row_values.append(sum(values) / len(values))
                    else:
                        row_values.append(None)
                z_values.append(row_values)
            if plotly_ctx is None:
                summary_lines = [
                    f"{target}: {len(values_by_target.get(target, []))} reactions"
                    for target in sensitivity_targets
                ]
                sensitivity_body = _render_message_list(
                    summary_lines,
                    "No sensitivity summary available.",
                )
            else:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z_values,
                        x=sensitivity_targets,
                        y=reaction_order,
                        colorbar=dict(title="Sensitivity"),
                        colorscale="RdBu",
                        zmid=0.0,
                    )
                )
                title_suffix = f" (condition_id={condition_id})" if condition_id else ""
                fig.update_layout(
                    title=f"Sensitivity Heatmap{title_suffix}",
                    template="plotly_white",
                    height=320 + 12 * len(reaction_order),
                    margin=dict(l=80, r=20, t=60, b=40),
                )
                sensitivity_body = _wrap_card(_plotly_html(fig, pio, plotly_state))
        else:
            sensitivity_body = "<p class=\"muted\">No ranked sensitivity values found.</p>"

    sensitivity_section = _panel("Sensitivity Heatmap", sensitivity_body)

    availability_lines: list[str] = []
    availability_lines.extend(missing_inputs)
    availability_lines.extend(notices)
    availability_section = ""
    if availability_lines:
        availability_section = _panel(
            "Data Availability",
            _render_message_list(availability_lines, "All inputs available."),
        )

    html_doc = render_report_html(
        title=title,
        dashboard=dashboard,
        created_at=report_manifest.created_at,
        manifest=report_manifest,
        inputs=input_specs,
        config=manifest_cfg,
        placeholders=placeholder_labels,
    )
    sections_html = (
        condition_section + objective_section + sensitivity_section + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


def chem_dashboard(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Render a chemistry dashboard from run, feature, and graph artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, viz_cfg = _extract_viz_cfg(resolved_cfg)

    raw_inputs = viz_cfg.get("inputs", viz_cfg.get("artifacts"))
    if raw_inputs is None and any(
        key in viz_cfg for key in ("runs", "features", "graphs", "reduction")
    ):
        raw_inputs = {
            "runs": viz_cfg.get("runs"),
            "features": viz_cfg.get("features"),
            "graphs": viz_cfg.get("graphs"),
            "reduction": viz_cfg.get("reduction"),
        }
    input_specs = _normalize_inputs(raw_inputs)

    notices: list[str] = []
    missing_inputs: list[str] = []
    input_manifests: list[ArtifactManifest] = []
    for entry in input_specs:
        try:
            manifest = store.read_manifest(entry["kind"], entry["id"])
        except ArtifactError as exc:
            missing_inputs.append(f"{entry['kind']}/{entry['id']}: {exc}")
            continue
        input_manifests.append(manifest)

    inputs_payload = {"artifacts": input_specs}
    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parent_ids = [manifest.id for manifest in input_manifests]
    notes = None
    if missing_inputs:
        notes = "Missing inputs: " + "; ".join(missing_inputs)

    report_manifest = ArtifactManifest(
        schema_version=1,
        kind="reports",
        id=report_id,
        created_at=_utc_now_iso(),
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
        notes=notes,
    )

    title = viz_cfg.get("title") or "Chem Dashboard"
    if not isinstance(title, str):
        raise ConfigError("viz.title must be a string.")
    dashboard = viz_cfg.get("dashboard") or "chem"
    if not isinstance(dashboard, str):
        raise ConfigError("viz.dashboard must be a string.")

    placeholders = viz_cfg.get("placeholders")
    if placeholders is None:
        placeholders = ("Reduction Diff",)
    if not isinstance(placeholders, Sequence) or isinstance(
        placeholders, (str, bytes, bytearray)
    ):
        raise ConfigError("viz.placeholders must be a list of strings.")
    placeholder_labels: list[str] = []
    for label in placeholders:
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("viz.placeholders entries must be non-empty strings.")
        placeholder_labels.append(label)

    run_manifests = [m for m in input_manifests if m.kind == "runs"]
    feature_manifests = [m for m in input_manifests if m.kind == "features"]
    graph_manifests = [m for m in input_manifests if m.kind == "graphs"]
    reduction_inputs = [item for item in input_specs if item["kind"] == "reduction"]

    plotly_ctx = _plotly_context()
    plotly_state = {"include_js": True}
    if plotly_ctx is None:
        notices.append("Plotly not available; using summaries instead of charts.")
    else:
        go, pio = plotly_ctx

    selected_run: Optional[ArtifactManifest] = None
    run_id = viz_cfg.get("run_id") or viz_cfg.get("run")
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ConfigError("viz.run_id must be a non-empty string.")
        for manifest in run_manifests:
            if manifest.id == run_id:
                selected_run = manifest
                break
        if selected_run is None:
            notices.append(f"Requested run_id {run_id} not found in inputs.")
    if selected_run is None and run_manifests:
        if len(run_manifests) > 1:
            notices.append("Multiple runs provided; using the first run for plots.")
        selected_run = run_manifests[0]

    run_payload: Optional[dict[str, Any]] = None
    if selected_run is not None:
        run_dir = store.artifact_dir("runs", selected_run.id)
        try:
            run_payload = _load_run_dataset_payload(run_dir)
        except Exception as exc:
            notices.append(f"Failed to load run dataset {selected_run.id}: {exc}")
            run_payload = None

    species_cfg = viz_cfg.get("species") or {}
    if not isinstance(species_cfg, Mapping):
        raise ConfigError("viz.species must be a mapping when provided.")
    series_vars = _normalize_name_list(
        species_cfg.get("variables", species_cfg.get("var")),
        "viz.species.variables",
    )
    if not series_vars:
        series_vars = ["X", "C", "Y"]
    species_names = _normalize_name_list(
        species_cfg.get("species", species_cfg.get("names")),
        "viz.species.species",
    )
    species_top_n = _coerce_positive_int(
        species_cfg.get("top_n"), "viz.species.top_n", default=5
    )
    species_rank_by = species_cfg.get("rank_by", "max")
    if species_rank_by not in ("max", "mean", "last", "integral", "min"):
        raise ConfigError("viz.species.rank_by must be a supported stat.")
    species_rank_abs = species_cfg.get("rank_abs")
    if species_rank_abs is None:
        species_rank_abs = False
    if not isinstance(species_rank_abs, bool):
        raise ConfigError("viz.species.rank_abs must be a boolean.")
    species_max_points = _coerce_positive_int(
        species_cfg.get("max_points"), "viz.species.max_points", default=200
    )
    axis_override = species_cfg.get("axis")
    if axis_override is not None and (
        not isinstance(axis_override, str) or not axis_override.strip()
    ):
        raise ConfigError("viz.species.axis must be a non-empty string.")

    selected_species: list[str] = []
    species_body = "<p class=\"muted\">No run data available for species plots.</p>"
    if run_payload is not None:
        data_vars = run_payload.get("data_vars", {})
        if not isinstance(data_vars, Mapping):
            notices.append("Run dataset data_vars missing for species plots.")
        else:
            series_var = None
            for candidate in series_vars:
                if candidate in data_vars:
                    series_var = candidate
                    break
            if series_var is None:
                species_body = (
                    "<p class=\"muted\">No species concentration data found.</p>"
                )
            else:
                try:
                    dims, _ = _extract_data_var(run_payload, series_var)
                except ArtifactError as exc:
                    species_body = f"<p class=\"muted\">{html.escape(str(exc))}</p>"
                    dims = []
                if dims:
                    axis_name = axis_override
                    if axis_name is None:
                        if "species" in dims:
                            axis_name = "species"
                        elif "surface_species" in dims:
                            axis_name = "surface_species"
                    if axis_name is None:
                        species_body = (
                            "<p class=\"muted\">Species axis not found in run data.</p>"
                        )
                    else:
                        (
                            time_values,
                            axis_values,
                            matrix,
                            error,
                        ) = _prepare_timeseries(run_payload, series_var, axis_name)
                        if error is not None or time_values is None or axis_values is None:
                            species_body = (
                                f"<p class=\"muted\">{html.escape(str(error))}</p>"
                            )
                        else:
                            units = run_payload.get("attrs", {}).get("units", {})
                            unit_label = ""
                            time_unit = ""
                            if isinstance(units, Mapping):
                                unit_label = str(units.get(series_var, "")) if units else ""
                                time_unit = str(units.get("time", "")) if units else ""
                            axis_index = {
                                name: idx for idx, name in enumerate(axis_values)
                            }
                            missing_species = [
                                name for name in species_names if name not in axis_index
                            ]
                            if missing_species:
                                notices.append(
                                    "Species not found in run data: "
                                    + ", ".join(missing_species)
                                )
                            if species_names:
                                selected_species = [
                                    name for name in species_names if name in axis_index
                                ]
                            else:
                                ranked = _rank_entities(
                                    axis_values,
                                    matrix,
                                    time_values,
                                    stat=species_rank_by,
                                    rank_abs=species_rank_abs,
                                )
                                selected_species = [
                                    name for name, _ in ranked[:species_top_n]
                                ]
                            if not selected_species:
                                species_body = (
                                    "<p class=\"muted\">No species selected for plots.</p>"
                                )
                            elif plotly_ctx is None:
                                summary = []
                                ranked = _rank_entities(
                                    axis_values,
                                    matrix,
                                    time_values,
                                    stat=species_rank_by,
                                    rank_abs=species_rank_abs,
                                )
                                rank_map = {name: value for name, value in ranked}
                                for name in selected_species:
                                    value = rank_map.get(name, math.nan)
                                    summary.append(
                                        f"{name}: {species_rank_by}={value:.4g}"
                                    )
                                species_body = _render_message_list(
                                    summary,
                                    "No species summary available.",
                                )
                            else:
                                indices = _downsample_indices(
                                    len(time_values),
                                    species_max_points,
                                )
                                fig = go.Figure()
                                for name in selected_species:
                                    idx = axis_index.get(name)
                                    if idx is None:
                                        continue
                                    series = [matrix[i][idx] for i in indices]
                                    times = [time_values[i] for i in indices]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=times,
                                            y=series,
                                            mode="lines",
                                            name=name,
                                        )
                                    )
                                title_parts = ["Species Time Series"]
                                if unit_label:
                                    title_parts.append(f"({unit_label})")
                                fig.update_layout(
                                    title=" ".join(title_parts),
                                    xaxis_title="Time"
                                    + (f" ({time_unit})" if time_unit else ""),
                                    yaxis_title=unit_label or series_var,
                                    template="plotly_white",
                                    height=320,
                                    margin=dict(l=50, r=20, t=50, b=40),
                                )
                                species_body = _wrap_card(
                                    _plotly_html(fig, pio, plotly_state)
                                )

    species_section = _panel("Species Time Series", species_body)

    rate_cfg = viz_cfg.get("rates", viz_cfg.get("production", {}))
    if rate_cfg is None:
        rate_cfg = {}
    if not isinstance(rate_cfg, Mapping):
        raise ConfigError("viz.rates must be a mapping when provided.")
    rate_vars = _normalize_name_list(
        rate_cfg.get("variables", rate_cfg.get("var")),
        "viz.rates.variables",
    )
    if not rate_vars:
        rate_vars = ["net_production_rates"]
    rate_top_n = _coerce_positive_int(
        rate_cfg.get("top_n"), "viz.rates.top_n", default=5
    )
    rate_rank_by = rate_cfg.get("rank_by", "max")
    if rate_rank_by not in ("max", "mean", "last", "integral", "min"):
        raise ConfigError("viz.rates.rank_by must be a supported stat.")
    rate_rank_abs = rate_cfg.get("rank_abs")
    if rate_rank_abs is None:
        rate_rank_abs = True
    if not isinstance(rate_rank_abs, bool):
        raise ConfigError("viz.rates.rank_abs must be a boolean.")
    rate_max_points = _coerce_positive_int(
        rate_cfg.get("max_points"), "viz.rates.max_points", default=species_max_points
    )

    wdot_ranked: list[tuple[str, float]] = []
    rate_body = "<p class=\"muted\">No production rate data available.</p>"
    if run_payload is not None:
        data_vars = run_payload.get("data_vars", {})
        if isinstance(data_vars, Mapping):
            rate_var = None
            for candidate in rate_vars:
                if candidate in data_vars:
                    rate_var = candidate
                    break
            if rate_var is not None:
                (
                    time_values,
                    axis_values,
                    matrix,
                    error,
                ) = _prepare_timeseries(run_payload, rate_var, "species")
                if error is None and time_values is not None and axis_values is not None:
                    wdot_ranked = _rank_entities(
                        axis_values,
                        matrix,
                        time_values,
                        stat=rate_rank_by,
                        rank_abs=rate_rank_abs,
                    )
                    selected_rate_species = [
                        name for name, _ in wdot_ranked[:rate_top_n]
                    ]
                    if plotly_ctx is None:
                        summary = [
                            f"{name}: {rate_rank_by}={value:.4g}"
                            for name, value in wdot_ranked[:rate_top_n]
                        ]
                        rate_body = _render_message_list(
                            summary,
                            "No production rate summary available.",
                        )
                    else:
                        axis_index = {
                            name: idx for idx, name in enumerate(axis_values)
                        }
                        indices = _downsample_indices(
                            len(time_values),
                            rate_max_points,
                        )
                        fig = go.Figure()
                        for name in selected_rate_species:
                            idx = axis_index.get(name)
                            if idx is None:
                                continue
                            series = [matrix[i][idx] for i in indices]
                            times = [time_values[i] for i in indices]
                            fig.add_trace(
                                go.Scatter(
                                    x=times,
                                    y=series,
                                    mode="lines",
                                    name=name,
                                )
                            )
                        units = run_payload.get("attrs", {}).get("units", {})
                        unit_label = ""
                        time_unit = ""
                        if isinstance(units, Mapping):
                            unit_label = str(units.get(rate_var, "")) if units else ""
                            time_unit = str(units.get("time", "")) if units else ""
                        title_parts = ["Net Production Rates"]
                        if unit_label:
                            title_parts.append(f"({unit_label})")
                        fig.update_layout(
                            title=" ".join(title_parts),
                            xaxis_title="Time" + (f" ({time_unit})" if time_unit else ""),
                            yaxis_title=unit_label or rate_var,
                            template="plotly_white",
                            height=320,
                            margin=dict(l=50, r=20, t=50, b=40),
                        )
                        rate_body = _wrap_card(
                            _plotly_html(fig, pio, plotly_state)
                        )

    rate_section = _panel("Net Production Rates", rate_body)

    feature_rows: list[dict[str, Any]] = []
    for manifest in feature_manifests:
        table_path = store.artifact_dir("features", manifest.id) / "features.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read features/{manifest.id}: {exc}")
            continue
        feature_rows.extend(rows)

    rop_cfg = viz_cfg.get("rop", {})
    if rop_cfg is None:
        rop_cfg = {}
    if not isinstance(rop_cfg, Mapping):
        raise ConfigError("viz.rop must be a mapping when provided.")
    rop_prefix = rop_cfg.get("feature_prefix") or rop_cfg.get("feature") or rop_cfg.get("prefix") or "rop_net"
    if not isinstance(rop_prefix, str) or not rop_prefix.strip():
        raise ConfigError("viz.rop.feature_prefix must be a non-empty string.")
    rop_var = rop_cfg.get("data_var") or rop_cfg.get("name") or "rop_net"
    if not isinstance(rop_var, str) or not rop_var.strip():
        raise ConfigError("viz.rop.data_var must be a non-empty string.")
    rop_stat = rop_cfg.get("stat", "integral")
    if rop_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.rop.stat must be a supported stat.")
    rop_top_n = _coerce_positive_int(
        rop_cfg.get("top_n"), "viz.rop.top_n", default=8
    )
    rop_rank_abs = rop_cfg.get("rank_abs")
    if rop_rank_abs is None:
        rop_rank_abs = True
    if not isinstance(rop_rank_abs, bool):
        raise ConfigError("viz.rop.rank_abs must be a boolean.")
    rop_run_id = rop_cfg.get("run_id")
    if rop_run_id is None and selected_run is not None:
        rop_run_id = selected_run.id
    if rop_run_id is not None and (
        not isinstance(rop_run_id, str) or not rop_run_id.strip()
    ):
        raise ConfigError("viz.rop.run_id must be a non-empty string.")

    rop_ranked: list[tuple[str, float]] = []
    if feature_rows:
        rop_ranked = _rank_from_feature_rows(
            feature_rows,
            prefix=rop_prefix,
            data_var=rop_var,
            id_label="reaction_id",
            stat=rop_stat,
            top_n=rop_top_n,
            rank_abs=rop_rank_abs,
            run_id=rop_run_id,
        )

    if not rop_ranked and run_payload is not None:
        rop_ranked, error = _rank_from_run_payload(
            run_payload,
            var_name=rop_var,
            axis=rop_cfg.get("axis", "reaction"),
            stat=rop_stat,
            top_n=rop_top_n,
            rank_abs=rop_rank_abs,
        )
        if error is not None:
            notices.append(f"ROP ranking unavailable: {error}")

    wdot_cfg = viz_cfg.get("wdot", {})
    if wdot_cfg is None:
        wdot_cfg = {}
    if not isinstance(wdot_cfg, Mapping):
        raise ConfigError("viz.wdot must be a mapping when provided.")
    wdot_prefix = wdot_cfg.get("feature_prefix") or wdot_cfg.get("feature") or wdot_cfg.get("prefix") or "net_production_rates"
    if not isinstance(wdot_prefix, str) or not wdot_prefix.strip():
        raise ConfigError("viz.wdot.feature_prefix must be a non-empty string.")
    wdot_var = wdot_cfg.get("data_var") or wdot_cfg.get("name") or "net_production_rates"
    if not isinstance(wdot_var, str) or not wdot_var.strip():
        raise ConfigError("viz.wdot.data_var must be a non-empty string.")
    wdot_stat = wdot_cfg.get("stat", "integral")
    if wdot_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.wdot.stat must be a supported stat.")
    wdot_top_n = _coerce_positive_int(
        wdot_cfg.get("top_n"), "viz.wdot.top_n", default=8
    )
    wdot_rank_abs = wdot_cfg.get("rank_abs")
    if wdot_rank_abs is None:
        wdot_rank_abs = True
    if not isinstance(wdot_rank_abs, bool):
        raise ConfigError("viz.wdot.rank_abs must be a boolean.")
    wdot_run_id = wdot_cfg.get("run_id")
    if wdot_run_id is None and selected_run is not None:
        wdot_run_id = selected_run.id
    if wdot_run_id is not None and (
        not isinstance(wdot_run_id, str) or not wdot_run_id.strip()
    ):
        raise ConfigError("viz.wdot.run_id must be a non-empty string.")

    wdot_ranked_from_features: list[tuple[str, float]] = []
    if feature_rows:
        wdot_ranked_from_features = _rank_from_feature_rows(
            feature_rows,
            prefix=wdot_prefix,
            data_var=wdot_var,
            id_label="species",
            stat=wdot_stat,
            top_n=wdot_top_n,
            rank_abs=wdot_rank_abs,
            run_id=wdot_run_id,
        )
    if not wdot_ranked_from_features and run_payload is not None:
        wdot_ranked_from_features, error = _rank_from_run_payload(
            run_payload,
            var_name=wdot_var,
            axis=wdot_cfg.get("axis", "species"),
            stat=wdot_stat,
            top_n=wdot_top_n,
            rank_abs=wdot_rank_abs,
        )
        if error is not None:
            notices.append(f"WDOT ranking unavailable: {error}")

    rop_card = "<p class=\"muted\">No ROP data available.</p>"
    if rop_ranked:
        if plotly_ctx is None:
            summary = [
                f"{name}: {value:.4g}" for name, value in rop_ranked
            ]
            rop_card = _render_message_list(
                summary,
                "No ROP summary available.",
            )
        else:
            fig = go.Figure(
                data=go.Bar(
                    x=[value for _, value in rop_ranked],
                    y=[name for name, _ in rop_ranked],
                    orientation="h",
                    marker=dict(color="#0f6f68"),
                )
            )
            fig.update_layout(
                title="Top ROP Reactions",
                template="plotly_white",
                height=260 + 14 * len(rop_ranked),
                margin=dict(l=120, r=20, t=50, b=40),
            )
            rop_card = _plotly_html(fig, pio, plotly_state)

    wdot_card = "<p class=\"muted\">No wdot data available.</p>"
    if wdot_ranked_from_features:
        if plotly_ctx is None:
            summary = [
                f"{name}: {value:.4g}" for name, value in wdot_ranked_from_features
            ]
            wdot_card = _render_message_list(
                summary,
                "No wdot summary available.",
            )
        else:
            fig = go.Figure(
                data=go.Bar(
                    x=[value for _, value in wdot_ranked_from_features],
                    y=[name for name, _ in wdot_ranked_from_features],
                    orientation="h",
                    marker=dict(color="#7a3e2f"),
                )
            )
            fig.update_layout(
                title="Top Net Production Species",
                template="plotly_white",
                height=260 + 14 * len(wdot_ranked_from_features),
                margin=dict(l=120, r=20, t=50, b=40),
            )
            wdot_card = _plotly_html(fig, pio, plotly_state)

    rop_section = _panel(
        "Rate-of-Production Ranking",
        "<div class=\"grid\">"
        + _wrap_card(rop_card)
        + _wrap_card(wdot_card)
        + "</div>",
    )

    network_cfg = viz_cfg.get("network", {})
    if network_cfg is None:
        network_cfg = {}
    if not isinstance(network_cfg, Mapping):
        raise ConfigError("viz.network must be a mapping when provided.")
    network_max_nodes = _coerce_positive_int(
        network_cfg.get("max_nodes"), "viz.network.max_nodes", default=18
    )
    network_max_edges = _coerce_positive_int(
        network_cfg.get("max_edges"), "viz.network.max_edges", default=40
    )
    include_neighbors = network_cfg.get("include_neighbors", True)
    if not isinstance(include_neighbors, bool):
        raise ConfigError("viz.network.include_neighbors must be a boolean.")
    focus_nodes = _normalize_name_list(
        network_cfg.get("focus_nodes"), "viz.network.focus_nodes"
    )

    network_body = "<p class=\"muted\">No graph data available.</p>"
    graph_payload: Optional[Mapping[str, Any]] = None
    if graph_manifests:
        graph_id = network_cfg.get("graph_id")
        if graph_id is not None and (
            not isinstance(graph_id, str) or not graph_id.strip()
        ):
            raise ConfigError("viz.network.graph_id must be a non-empty string.")
        selected_graph = None
        if graph_id:
            for manifest in graph_manifests:
                if manifest.id == graph_id:
                    selected_graph = manifest
                    break
            if selected_graph is None:
                notices.append(f"Requested graph_id {graph_id} not found in inputs.")
        if selected_graph is None:
            if len(graph_manifests) > 1:
                notices.append("Multiple graphs provided; using the first graph.")
            selected_graph = graph_manifests[0]
        graph_path = store.artifact_dir("graphs", selected_graph.id) / "graph.json"
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                graph_payload = payload
        except Exception as exc:
            notices.append(f"Failed to read graph/{selected_graph.id}: {exc}")
            graph_payload = None

    if graph_payload is not None:
        graph_data: Optional[Mapping[str, Any]] = None
        if isinstance(graph_payload.get("bipartite"), Mapping):
            data = graph_payload.get("bipartite", {}).get("data")
            if isinstance(data, Mapping):
                graph_data = data
        if graph_data is None and "nodes" in graph_payload:
            graph_data = graph_payload

        if graph_data is None:
            network_body = "<p class=\"muted\">Graph payload has no node data.</p>"
        else:
            nodes_raw = graph_data.get("nodes", [])
            links_raw = graph_data.get("links") or graph_data.get("edges") or []
            if not isinstance(nodes_raw, Sequence) or isinstance(
                nodes_raw, (str, bytes, bytearray)
            ):
                network_body = "<p class=\"muted\">Graph nodes not available.</p>"
            elif not isinstance(links_raw, Sequence) or isinstance(
                links_raw, (str, bytes, bytearray)
            ):
                network_body = "<p class=\"muted\">Graph links not available.</p>"
            else:
                node_map: dict[str, dict[str, Any]] = {}
                for entry in nodes_raw:
                    if not isinstance(entry, Mapping):
                        continue
                    node_id = entry.get("id") or entry.get("name") or entry.get("key")
                    if node_id is None:
                        continue
                    node_map[str(node_id)] = dict(entry)
                    node_map[str(node_id)]["id"] = str(node_id)
                edge_list: list[tuple[str, str]] = []
                for entry in links_raw:
                    if not isinstance(entry, Mapping):
                        continue
                    source = entry.get("source")
                    target = entry.get("target")
                    if isinstance(source, Mapping):
                        source = source.get("id") or source.get("name") or source.get("key")
                    if isinstance(target, Mapping):
                        target = target.get("id") or target.get("name") or target.get("key")
                    if source is None or target is None:
                        continue
                    source_id = str(source)
                    target_id = str(target)
                    if source_id not in node_map or target_id not in node_map:
                        continue
                    edge_list.append((source_id, target_id))

                if not node_map:
                    network_body = "<p class=\"muted\">Graph nodes missing.</p>"
                else:
                    label_index: dict[str, str] = {}
                    for node_id, node in node_map.items():
                        for key in (
                            "id",
                            "label",
                            "reaction_id",
                            "species",
                            "name",
                            "reaction_equation",
                        ):
                            value = node.get(key)
                            if value is None:
                                continue
                            label_index[str(value)] = node_id
                    focus_labels: list[str] = list(focus_nodes)
                    focus_labels.extend(selected_species)
                    focus_labels.extend([name for name, _ in rop_ranked])
                    focus_labels.extend(
                        [name for name, _ in wdot_ranked_from_features]
                    )
                    selected_ids = []
                    for label in focus_labels:
                        node_id = label_index.get(label)
                        if node_id:
                            selected_ids.append(node_id)

                    if not selected_ids:
                        degree_counts: dict[str, int] = {node_id: 0 for node_id in node_map}
                        for source_id, target_id in edge_list:
                            degree_counts[source_id] += 1
                            degree_counts[target_id] += 1
                        sorted_nodes = sorted(
                            degree_counts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        selected_ids = [node_id for node_id, _ in sorted_nodes[:network_max_nodes]]
                    else:
                        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_map}
                        for source_id, target_id in edge_list:
                            adjacency.setdefault(source_id, set()).add(target_id)
                            adjacency.setdefault(target_id, set()).add(source_id)
                        selected_set = set(selected_ids)
                        if include_neighbors:
                            for node_id in list(selected_set):
                                selected_set.update(adjacency.get(node_id, set()))
                        if len(selected_set) > network_max_nodes:
                            trimmed = []
                            for node_id in sorted(selected_set):
                                trimmed.append((node_id, len(adjacency.get(node_id, set()))))
                            trimmed.sort(key=lambda item: (-item[1], item[0]))
                            selected_ids = [
                                node_id for node_id, _ in trimmed[:network_max_nodes]
                            ]
                        else:
                            selected_ids = sorted(selected_set)

                    edge_list = [
                        (source_id, target_id)
                        for source_id, target_id in edge_list
                        if source_id in selected_ids and target_id in selected_ids
                    ]
                    if len(edge_list) > network_max_edges:
                        edge_list = edge_list[:network_max_edges]

                    if plotly_ctx is None:
                        summary = [
                            f"nodes={len(selected_ids)}, edges={len(edge_list)}"
                        ]
                        network_body = _render_message_list(
                            summary,
                            "No network summary available.",
                        )
                    else:
                        species_nodes = [
                            node_id
                            for node_id in selected_ids
                            if node_map.get(node_id, {}).get("kind") == "species"
                        ]
                        reaction_nodes = [
                            node_id
                            for node_id in selected_ids
                            if node_map.get(node_id, {}).get("kind") == "reaction"
                        ]
                        other_nodes = [
                            node_id
                            for node_id in selected_ids
                            if node_id not in species_nodes
                            and node_id not in reaction_nodes
                        ]
                        positions: dict[str, tuple[float, float]] = {}
                        if species_nodes and reaction_nodes:
                            for group, x_value in ((species_nodes, 0.0), (reaction_nodes, 1.0)):
                                for idx, node_id in enumerate(group):
                                    y_value = idx / max(len(group) - 1, 1)
                                    positions[node_id] = (x_value, y_value)
                            for idx, node_id in enumerate(other_nodes):
                                y_value = idx / max(len(other_nodes) - 1, 1)
                                positions[node_id] = (0.5, y_value)
                        else:
                            total = len(selected_ids)
                            for idx, node_id in enumerate(selected_ids):
                                angle = 2.0 * math.pi * idx / max(total, 1)
                                positions[node_id] = (math.cos(angle), math.sin(angle))

                        edge_x: list[float] = []
                        edge_y: list[float] = []
                        for source_id, target_id in edge_list:
                            x0, y0 = positions.get(source_id, (0.0, 0.0))
                            x1, y1 = positions.get(target_id, (0.0, 0.0))
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                        node_x: list[float] = []
                        node_y: list[float] = []
                        node_text: list[str] = []
                        for node_id in selected_ids:
                            x_val, y_val = positions.get(node_id, (0.0, 0.0))
                            node_x.append(x_val)
                            node_y.append(y_val)
                            node = node_map.get(node_id, {})
                            label = (
                                node.get("label")
                                or node.get("reaction_id")
                                or node.get("species")
                                or node.get("id")
                                or node_id
                            )
                            node_text.append(str(label))

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=edge_x,
                                y=edge_y,
                                mode="lines",
                                line=dict(color="#9bb3b0", width=1),
                                hoverinfo="none",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=node_x,
                                y=node_y,
                                mode="markers+text",
                                text=node_text,
                                textposition="top center",
                                marker=dict(
                                    size=10,
                                    color="#0f6f68",
                                    line=dict(color="#ffffff", width=1),
                                ),
                                hoverinfo="text",
                            )
                        )
                        fig.update_layout(
                            title="Reaction Network Subgraph",
                            template="plotly_white",
                            height=360,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                        )
                        network_body = _wrap_card(
                            _plotly_html(fig, pio, plotly_state)
                        )

    network_section = _panel("Reaction Network Subgraph", network_body)

    reduction_lines: list[str] = []
    if reduction_inputs:
        reduction_lines.extend(
            [f"reduction/{item['id']}" for item in reduction_inputs]
        )
    if reduction_lines:
        reduction_body = _render_message_list(
            reduction_lines,
            "Reduction diff placeholder.",
        )
    else:
        reduction_body = "<p class=\"muted\">Reduction diff placeholder.</p>"
    reduction_section = _panel("Reduction Delta", reduction_body)

    availability_lines: list[str] = []
    availability_lines.extend(missing_inputs)
    availability_lines.extend(notices)
    availability_section = ""
    if availability_lines:
        availability_section = _panel(
            "Data Availability",
            _render_message_list(availability_lines, "All inputs available."),
        )

    html_doc = render_report_html(
        title=title,
        dashboard=dashboard,
        created_at=report_manifest.created_at,
        manifest=report_manifest,
        inputs=input_specs,
        config=manifest_cfg,
        placeholders=placeholder_labels,
    )
    sections_html = (
        species_section
        + rate_section
        + rop_section
        + network_section
        + reduction_section
        + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


def benchmark_report(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Render a benchmark comparison report from optimization/assimilation/validation artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, viz_cfg = _extract_viz_cfg(resolved_cfg)

    raw_inputs = viz_cfg.get("inputs", viz_cfg.get("artifacts"))
    input_specs = _normalize_inputs(raw_inputs) if raw_inputs is not None else []

    raw_groups = viz_cfg.get("groups")
    if raw_groups is None:
        benchmark_cfg = viz_cfg.get("benchmark")
        if isinstance(benchmark_cfg, Mapping):
            if isinstance(benchmark_cfg.get("groups"), Mapping):
                raw_groups = benchmark_cfg.get("groups")
            else:
                raw_groups = benchmark_cfg
    group_inputs = _normalize_group_inputs(raw_groups, "viz.groups")

    group_entries = [
        entry for entries in group_inputs.values() for entry in entries
    ]
    all_inputs = _merge_inputs(input_specs, group_entries)

    notices: list[str] = []
    missing_inputs: list[str] = []
    input_manifests: list[ArtifactManifest] = []
    manifest_by_key: dict[tuple[str, str], ArtifactManifest] = {}
    for entry in all_inputs:
        try:
            manifest = store.read_manifest(entry["kind"], entry["id"])
        except ArtifactError as exc:
            missing_inputs.append(f"{entry['kind']}/{entry['id']}: {exc}")
            continue
        input_manifests.append(manifest)
        manifest_by_key[(entry["kind"], entry["id"])] = manifest

    inputs_payload: dict[str, Any] = {"artifacts": all_inputs}
    if group_inputs:
        inputs_payload["groups"] = group_inputs

    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parent_ids = [manifest.id for manifest in input_manifests]
    notes = None
    if missing_inputs:
        notes = "Missing inputs: " + "; ".join(missing_inputs)

    report_manifest = ArtifactManifest(
        schema_version=1,
        kind="reports",
        id=report_id,
        created_at=_utc_now_iso(),
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
        notes=notes,
    )

    title = viz_cfg.get("title") or "Benchmark Report"
    if not isinstance(title, str):
        raise ConfigError("viz.title must be a string.")
    dashboard = viz_cfg.get("dashboard") or "benchmark"
    if not isinstance(dashboard, str):
        raise ConfigError("viz.dashboard must be a string.")

    placeholders = viz_cfg.get("placeholders")
    if placeholders is None:
        placeholders = ("Notes",)
    if not isinstance(placeholders, Sequence) or isinstance(
        placeholders, (str, bytes, bytearray)
    ):
        raise ConfigError("viz.placeholders must be a list of strings.")
    placeholder_labels: list[str] = []
    for label in placeholders:
        if not isinstance(label, str) or not label.strip():
            raise ConfigError("viz.placeholders entries must be non-empty strings.")
        placeholder_labels.append(label)

    graph_manifests = [m for m in input_manifests if m.kind == "graphs"]
    reaction_count = _infer_reaction_count(viz_cfg, graph_manifests, store, notices)

    group_by_key: dict[tuple[str, str], list[str]] = {}
    for group_name, entries in group_inputs.items():
        for entry in entries:
            key = (entry["kind"], entry["id"])
            group_by_key.setdefault(key, []).append(group_name)

    def _group_label(kind: str, artifact_id: str) -> str:
        labels = group_by_key.get((kind, artifact_id), [])
        if labels:
            return ", ".join(labels)
        if group_inputs:
            return "unassigned"
        return "all"

    def _format_objectives(best_map: Mapping[str, tuple[float, str]]) -> str:
        if not best_map:
            return "n/a"
        parts = []
        for name in sorted(best_map.keys()):
            value, direction = best_map[name]
            parts.append(f"{name}: {value:.4g} ({direction})")
        return "; ".join(parts)

    def _update_best_objectives(
        best_map: dict[str, tuple[float, str]],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        for row in rows:
            name = row.get("objective_name")
            if not isinstance(name, str) or not name.strip():
                continue
            direction = str(row.get("direction", "min")).lower()
            if direction not in ("min", "max"):
                direction = "min"
            value = _coerce_float(row.get("objective"))
            if value is None:
                continue
            current = best_map.get(name)
            if current is None:
                best_map[name] = (value, direction)
                continue
            current_value, current_direction = current
            if current_direction != direction:
                notices.append(
                    f"Objective direction mismatch for {name}; using {current_direction}."
                )
                direction = current_direction
            if direction == "min" and value < current_value:
                best_map[name] = (value, direction)
            if direction == "max" and value > current_value:
                best_map[name] = (value, direction)

    def _summarize_optimization(
        manifest: ArtifactManifest,
    ) -> tuple[dict[str, tuple[float, str]], Optional[int], Optional[int]]:
        table_path = store.artifact_dir("optimization", manifest.id) / "history.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read optimization/{manifest.id}: {exc}")
            return {}, None, None
        best_map: dict[str, tuple[float, str]] = {}
        _update_best_objectives(best_map, rows)
        sample_count = _coerce_optional_int(manifest.inputs.get("sample_count"))
        if sample_count is None:
            sample_ids = {
                row.get("sample_id") for row in rows if row.get("sample_id") is not None
            }
            sample_count = len(sample_ids) if sample_ids else len(rows)
        run_ids = manifest.inputs.get("run_ids")
        run_count = None
        if isinstance(run_ids, Sequence) and not isinstance(
            run_ids, (str, bytes, bytearray)
        ):
            run_count = len(run_ids)
        else:
            run_ids = {
                row.get("run_id") for row in rows if row.get("run_id") is not None
            }
            run_count = len(run_ids) if run_ids else None
        return best_map, sample_count, run_count

    def _select_final_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if not rows:
            return None
        iterations = [
            _coerce_optional_int(row.get("iteration")) for row in rows
        ]
        valid_iterations = [value for value in iterations if value is not None]
        if not valid_iterations:
            return rows[-1]
        final_iteration = max(valid_iterations)
        for row in reversed(rows):
            if _coerce_optional_int(row.get("iteration")) == final_iteration:
                return row
        return rows[-1]

    def _summarize_assimilation(
        manifest: ArtifactManifest,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[int]]:
        table_path = store.artifact_dir("assimilation", manifest.id) / "misfit_history.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read assimilation/{manifest.id}: {exc}")
            return None, None, None, None, None, None
        final_row = _select_final_row(rows)
        mean_misfit = _coerce_float(final_row.get("mean_misfit")) if final_row else None
        min_misfit = _coerce_float(final_row.get("min_misfit")) if final_row else None
        max_misfit = _coerce_float(final_row.get("max_misfit")) if final_row else None
        iterations = _coerce_optional_int(manifest.inputs.get("iterations"))
        if iterations is None:
            iterations = len(
                {
                    _coerce_optional_int(row.get("iteration"))
                    for row in rows
                    if _coerce_optional_int(row.get("iteration")) is not None
                }
            ) or None
        ensemble_size = _coerce_optional_int(manifest.inputs.get("ensemble_size"))
        eval_count = (
            iterations * ensemble_size
            if iterations is not None and ensemble_size is not None
            else None
        )
        return mean_misfit, min_misfit, max_misfit, iterations, ensemble_size, eval_count

    def _summarize_validation(
        manifest: ArtifactManifest,
    ) -> tuple[Optional[float], Optional[int], Optional[int]]:
        table_path = store.artifact_dir("validation", manifest.id) / "metrics.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read validation/{manifest.id}: {exc}")
            return None, None, None
        selected_patch = manifest.inputs.get("selected_patch")
        patch_index = None
        reduction_id = None
        if isinstance(selected_patch, Mapping):
            patch_index = _coerce_optional_int(selected_patch.get("patch_index"))
            reduction_id = selected_patch.get("reduction_id")
        if patch_index is not None:
            rows = [
                row
                for row in rows
                if _coerce_optional_int(row.get("patch_index")) == patch_index
            ]
        if reduction_id is None:
            patches = manifest.inputs.get("patches")
            if isinstance(patches, Sequence) and not isinstance(
                patches, (str, bytes, bytearray)
            ):
                for entry in patches:
                    if not isinstance(entry, Mapping):
                        continue
                    entry_index = _coerce_optional_int(entry.get("patch_index"))
                    if patch_index is None or entry_index == patch_index:
                        reduction_id = entry.get("reduction_id")
                        break
        pass_values = [
            bool(row.get("passed"))
            for row in rows
            if isinstance(row.get("passed"), bool)
        ]
        pass_rate = sum(pass_values) / len(pass_values) if pass_values else None
        disabled_count = None
        if isinstance(reduction_id, str) and reduction_id.strip():
            patch_path = (
                store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
            )
            patch_payload, error = _read_patch_payload(patch_path)
            if error is not None:
                notices.append(f"Reduction patch {reduction_id}: {error}")
            elif patch_payload is not None:
                disabled = patch_payload.get("disabled_reactions", [])
                if isinstance(disabled, Sequence) and not isinstance(
                    disabled, (str, bytes, bytearray)
                ):
                    disabled_count = len(disabled)
                else:
                    disabled_count = 0
        metric_count = len(rows) if rows else None
        return pass_rate, disabled_count, metric_count

    summary_rows: list[list[str]] = []
    optimization_rows: list[list[str]] = []
    assimilation_rows: list[list[str]] = []
    validation_rows: list[list[str]] = []

    group_specs = (
        group_inputs
        if group_inputs
        else ({"all": all_inputs} if all_inputs else {})
    )
    for group_name, entries in group_specs.items():
        group_manifests: list[ArtifactManifest] = []
        for entry in entries:
            manifest = manifest_by_key.get((entry["kind"], entry["id"]))
            if manifest is not None:
                group_manifests.append(manifest)
        opt_manifests = [m for m in group_manifests if m.kind == "optimization"]
        assim_manifests = [m for m in group_manifests if m.kind == "assimilation"]
        val_manifests = [m for m in group_manifests if m.kind == "validation"]

        group_best: dict[str, tuple[float, str]] = {}
        opt_samples_total = 0
        opt_samples_known = False
        opt_runs_total = 0
        opt_runs_known = False
        for manifest in opt_manifests:
            best_map, sample_count, run_count = _summarize_optimization(manifest)
            _update_best_objectives(group_best, [
                {"objective_name": key, "objective": value, "direction": direction}
                for key, (value, direction) in best_map.items()
            ])
            if sample_count is not None:
                opt_samples_total += sample_count
                opt_samples_known = True
            if run_count is not None:
                opt_runs_total += run_count
                opt_runs_known = True

        assim_means: list[float] = []
        assim_eval_total = 0
        assim_eval_known = False
        for manifest in assim_manifests:
            mean_misfit, _, _, _, _, eval_count = _summarize_assimilation(manifest)
            if mean_misfit is not None:
                assim_means.append(mean_misfit)
            if eval_count is not None:
                assim_eval_total += eval_count
                assim_eval_known = True

        val_pass_rates: list[float] = []
        val_disabled_counts: list[int] = []
        val_metric_total = 0
        val_metric_known = False
        for manifest in val_manifests:
            pass_rate, disabled_count, metric_count = _summarize_validation(manifest)
            if pass_rate is not None:
                val_pass_rates.append(pass_rate)
            if disabled_count is not None:
                val_disabled_counts.append(disabled_count)
            if metric_count is not None:
                val_metric_total += metric_count
                val_metric_known = True

        objective_text = _format_objectives(group_best)
        error_text = (
            f"mean={_format_number(min(assim_means))}"
            if assim_means
            else "n/a"
        )
        disabled_text = "n/a"
        if val_disabled_counts:
            disabled_value = max(val_disabled_counts)
            if reaction_count:
                reduction_rate = disabled_value / float(reaction_count)
                disabled_text = f"disabled={disabled_value} ({_format_percent(reduction_rate)})"
            else:
                disabled_text = f"disabled={disabled_value}"
        compute_parts: list[str] = []
        if opt_samples_known:
            compute_parts.append(f"opt samples={opt_samples_total}")
        if assim_eval_known:
            compute_parts.append(f"assim eval={assim_eval_total}")
        if val_metric_known:
            compute_parts.append(f"val metrics={val_metric_total}")
        compute_text = "; ".join(compute_parts) if compute_parts else "n/a"
        pass_rate_text = _format_percent(
            sum(val_pass_rates) / len(val_pass_rates)
        ) if val_pass_rates else "n/a"

        summary_rows.append(
            [
                group_name,
                objective_text,
                error_text,
                f"{disabled_text}, pass={pass_rate_text}",
                compute_text,
            ]
        )

    for manifest in input_manifests:
        if manifest.kind == "optimization":
            best_map, sample_count, run_count = _summarize_optimization(manifest)
            optimization_rows.append(
                [
                    _group_label("optimization", manifest.id),
                    manifest.id,
                    _format_objectives(best_map),
                    str(sample_count) if sample_count is not None else "n/a",
                    str(run_count) if run_count is not None else "n/a",
                ]
            )
        elif manifest.kind == "assimilation":
            mean_misfit, min_misfit, max_misfit, iterations, ensemble, eval_count = (
                _summarize_assimilation(manifest)
            )
            assimilation_rows.append(
                [
                    _group_label("assimilation", manifest.id),
                    manifest.id,
                    _format_number(mean_misfit),
                    _format_number(min_misfit),
                    _format_number(max_misfit),
                    str(iterations) if iterations is not None else "n/a",
                    str(ensemble) if ensemble is not None else "n/a",
                    str(eval_count) if eval_count is not None else "n/a",
                ]
            )
        elif manifest.kind == "validation":
            pass_rate, disabled_count, metric_count = _summarize_validation(manifest)
            reduction_rate = (
                (disabled_count / float(reaction_count))
                if disabled_count is not None and reaction_count
                else None
            )
            validation_rows.append(
                [
                    _group_label("validation", manifest.id),
                    manifest.id,
                    _format_percent(pass_rate),
                    str(disabled_count) if disabled_count is not None else "n/a",
                    _format_percent(reduction_rate),
                    str(metric_count) if metric_count is not None else "n/a",
                ]
            )

    summary_table = _render_table(
        ["Group", "Objective", "Error", "Reduction", "Compute"],
        summary_rows,
        "No benchmark data available.",
    )
    summary_section = _panel(
        "Benchmark Summary",
        _wrap_card(summary_table),
    )

    optimization_section = _panel(
        "Optimization Artifacts",
        _wrap_card(
            _render_table(
                ["Group", "Artifact", "Objectives", "Samples", "Runs"],
                optimization_rows,
                "No optimization artifacts available.",
            )
        ),
    )

    assimilation_section = _panel(
        "Assimilation Artifacts",
        _wrap_card(
            _render_table(
                [
                    "Group",
                    "Artifact",
                    "Mean misfit",
                    "Min misfit",
                    "Max misfit",
                    "Iterations",
                    "Ensemble",
                    "Eval count",
                ],
                assimilation_rows,
                "No assimilation artifacts available.",
            )
        ),
    )

    validation_section = _panel(
        "Validation Artifacts",
        _wrap_card(
            _render_table(
                [
                    "Group",
                    "Artifact",
                    "Pass rate",
                    "Disabled",
                    "Reduction rate",
                    "Metrics",
                ],
                validation_rows,
                "No validation artifacts available.",
            )
        ),
    )

    availability_lines: list[str] = []
    availability_lines.extend(missing_inputs)
    availability_lines.extend(notices)
    availability_section = ""
    if availability_lines:
        availability_section = _panel(
            "Data Availability",
            _render_message_list(availability_lines, "All inputs available."),
        )

    html_doc = render_report_html(
        title=title,
        dashboard=dashboard,
        created_at=report_manifest.created_at,
        manifest=report_manifest,
        inputs=all_inputs,
        config=manifest_cfg,
        placeholders=placeholder_labels,
    )
    sections_html = (
        summary_section
        + optimization_section
        + assimilation_section
        + validation_section
        + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


register("task", "viz.ds_dashboard", ds_dashboard)
register("task", "viz.chem_dashboard", chem_dashboard)
register("task", "viz.benchmark_report", benchmark_report)

__all__ = ["benchmark_report", "chem_dashboard", "ds_dashboard", "run"]
