"""Visualization task entrypoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import base64
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional

from rxn_platform.core import make_artifact_id
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.io_utils import read_yaml_payload as _read_yaml_payload
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.registry import Registry, register
from rxn_platform.run_store import (
    resolve_run_root_from_store,
    sync_report_from_artifact,
)
from rxn_platform.reporting import render_report_html
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    load_run_dataset_payload,
    read_table_rows as _read_table_rows,
    resolve_cfg as _resolve_cfg,
)

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:  # Optional dependency.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None


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


def _matplotlib_context() -> Optional[Any]:
    try:
        import matplotlib  # type: ignore
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def _resolve_chart_backend(
    viz_cfg: Mapping[str, Any],
    notices: list[str],
) -> tuple[str, Optional[tuple[Any, Any]], Optional[Any]]:
    backend = viz_cfg.get("chart_backend", "plotly")
    if not isinstance(backend, str) or not backend.strip():
        raise ConfigError("viz.chart_backend must be a non-empty string.")
    backend = backend.strip().lower()
    if backend not in ("plotly", "matplotlib", "svg"):
        raise ConfigError("viz.chart_backend must be plotly, matplotlib, or svg.")
    plotly_ctx: Optional[tuple[Any, Any]] = None
    mpl_ctx: Optional[Any] = None
    if backend == "plotly":
        plotly_ctx = _plotly_context()
        if plotly_ctx is None:
            notices.append("Plotly not available; falling back to SVG charts.")
            backend = "svg"
    elif backend == "matplotlib":
        mpl_ctx = _matplotlib_context()
        if mpl_ctx is None:
            notices.append("Matplotlib not available; falling back to SVG charts.")
            backend = "svg"
    return backend, plotly_ctx, mpl_ctx


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


def _normalize_image_formats(value: Any) -> list[str]:
    if value is None:
        return ["png"]
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        raw = list(value)
    else:
        raise ConfigError("viz.image_formats must be a string or list of strings.")
    formats: list[str] = []
    allowed = {"png", "jpg", "jpeg", "svg"}
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError("viz.image_formats entries must be non-empty strings.")
        fmt = entry.strip().lower()
        if fmt not in allowed:
            raise ConfigError(f"viz.image_formats must be one of: {', '.join(sorted(allowed))}.")
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in formats:
            formats.append(fmt)
    return formats


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "figure"


def _init_export_state(
    viz_cfg: Mapping[str, Any],
    plotly_ctx: Optional[tuple[Any, Any]],
    notices: list[str],
) -> Optional[dict[str, Any]]:
    export_cfg = viz_cfg.get("export_images", True)
    enabled = True
    formats: list[str] = []
    scale = viz_cfg.get("image_scale", 2)
    dir_name = viz_cfg.get("image_dir", "images")
    if isinstance(export_cfg, Mapping):
        if export_cfg.get("enabled") is not None:
            enabled = bool(export_cfg.get("enabled"))
        formats = _normalize_image_formats(
            export_cfg.get("formats", export_cfg.get("image_formats"))
        )
        if export_cfg.get("scale") is not None:
            scale = export_cfg.get("scale")
        if export_cfg.get("dir") is not None:
            dir_name = export_cfg.get("dir")
    else:
        enabled = bool(export_cfg)
        formats = _normalize_image_formats(viz_cfg.get("image_formats"))
    try:
        scale_value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ConfigError("viz.image_scale must be a positive number.") from exc
    if scale_value <= 0:
        raise ConfigError("viz.image_scale must be a positive number.")
    if not isinstance(dir_name, str) or not dir_name.strip():
        raise ConfigError("viz.image_dir must be a non-empty string.")
    if plotly_ctx is None or not enabled:
        if enabled and plotly_ctx is None:
            notices.append("Image export skipped: plotly not available.")
        return None
    try:
        import kaleido  # noqa: F401
    except Exception:
        notices.append("Image export skipped: kaleido not installed.")
        return None
    return {
        "enabled": True,
        "formats": formats,
        "scale": scale_value,
        "dir_name": dir_name,
        "queue": [],
        "planned": [],
        "used": set(),
        "pio": plotly_ctx[1],
    }


def _init_matplotlib_export_state(
    viz_cfg: Mapping[str, Any],
    notices: list[str],
) -> Optional[dict[str, Any]]:
    export_cfg = viz_cfg.get("export_images", True)
    enabled = True
    formats: list[str] = []
    scale = viz_cfg.get("image_scale", 2)
    dir_name = viz_cfg.get("image_dir", "images")
    if isinstance(export_cfg, Mapping):
        if export_cfg.get("enabled") is not None:
            enabled = bool(export_cfg.get("enabled"))
        formats = _normalize_image_formats(
            export_cfg.get("formats", export_cfg.get("image_formats"))
        )
        if export_cfg.get("scale") is not None:
            scale = export_cfg.get("scale")
        if export_cfg.get("dir") is not None:
            dir_name = export_cfg.get("dir")
    else:
        enabled = bool(export_cfg)
        formats = _normalize_image_formats(viz_cfg.get("image_formats"))
    try:
        scale_value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ConfigError("viz.image_scale must be a positive number.") from exc
    if scale_value <= 0:
        raise ConfigError("viz.image_scale must be a positive number.")
    if not isinstance(dir_name, str) or not dir_name.strip():
        raise ConfigError("viz.image_dir must be a non-empty string.")
    if not enabled:
        return None
    return {
        "enabled": True,
        "formats": formats,
        "scale": scale_value,
        "dir_name": dir_name,
        "queue": [],
        "planned": [],
        "used": set(),
        "base_dpi": 120,
    }


def _init_svg_export_state(
    viz_cfg: Mapping[str, Any],
    notices: list[str],
) -> Optional[dict[str, Any]]:
    export_cfg = viz_cfg.get("export_images", True)
    enabled = True
    formats: list[str] = []
    dir_name = viz_cfg.get("image_dir", "images")
    if isinstance(export_cfg, Mapping):
        if export_cfg.get("enabled") is not None:
            enabled = bool(export_cfg.get("enabled"))
        formats = _normalize_image_formats(
            export_cfg.get("formats", export_cfg.get("image_formats"))
        )
        if export_cfg.get("dir") is not None:
            dir_name = export_cfg.get("dir")
    else:
        enabled = bool(export_cfg)
        formats = _normalize_image_formats(viz_cfg.get("image_formats"))
    if not isinstance(dir_name, str) or not dir_name.strip():
        raise ConfigError("viz.image_dir must be a non-empty string.")
    if not enabled:
        return None
    if "svg" not in formats:
        notices.append("SVG backend only supports svg; requested formats ignored.")
    elif len(formats) > 1:
        notices.append("SVG backend ignores non-svg formats.")
    return {
        "enabled": True,
        "formats": ["svg"],
        "dir_name": dir_name,
        "queue": [],
        "planned": [],
        "used": set(),
    }


def _reserve_export_name(state: Mapping[str, Any], name: str) -> str:
    base = _slugify(name)
    used: set[str] = state["used"]
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _plotly_html_with_export(
    fig: Any,
    pio: Any,
    state: dict[str, bool],
    export_state: Optional[dict[str, Any]],
    name: str,
) -> str:
    html_snippet = _plotly_html(fig, pio, state)
    if export_state is None:
        return html_snippet
    export_name = _reserve_export_name(export_state, name)
    export_state["queue"].append({"name": export_name, "fig": fig})
    for fmt in export_state["formats"]:
        export_state["planned"].append(
            f"{export_state['dir_name']}/{export_name}.{fmt}"
        )
    return html_snippet


def _matplotlib_fig_bytes(
    fig: Any,
    fmt: str,
    *,
    scale: float,
    base_dpi: int,
) -> Optional[bytes]:
    buffer = io.BytesIO()
    try:
        fig.savefig(
            buffer,
            format=fmt,
            dpi=base_dpi * scale,
            bbox_inches="tight",
        )
    except Exception:
        return None
    return buffer.getvalue()


def _matplotlib_html_with_export(
    fig: Any,
    export_state: Optional[dict[str, Any]],
    name: str,
    *,
    alt: str,
    notices: list[str],
) -> str:
    scale = export_state.get("scale", 1.0) if export_state else 1.0
    base_dpi = export_state.get("base_dpi", 120) if export_state else 120
    png_bytes = _matplotlib_fig_bytes(fig, "png", scale=scale, base_dpi=base_dpi)
    if png_bytes is None:
        notices.append(f"Matplotlib render failed for {name}.")
        return "<p class=\"muted\">Chart rendering failed.</p>"
    data_uri = base64.b64encode(png_bytes).decode("ascii")
    html_snippet = (
        "<div style=\"display:flex;justify-content:center;\">"
        f"<img src=\"data:image/png;base64,{data_uri}\" "
        f"alt=\"{html.escape(alt)}\" style=\"max-width:100%;height:auto;\" />"
        "</div>"
    )
    if export_state is not None:
        export_name = _reserve_export_name(export_state, name)
        images: dict[str, bytes] = {}
        for fmt in export_state["formats"]:
            image_bytes = _matplotlib_fig_bytes(
                fig,
                fmt,
                scale=scale,
                base_dpi=base_dpi,
            )
            if image_bytes is None:
                notices.append(f"Matplotlib export failed ({export_name}.{fmt}).")
                continue
            images[fmt] = image_bytes
            export_state["planned"].append(
                f"{export_state['dir_name']}/{export_name}.{fmt}"
            )
        export_state["queue"].append({"name": export_name, "images": images})
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.close(fig)
    except Exception:
        pass
    return html_snippet


def _svg_html_with_export(
    chart_html: str,
    export_state: Optional[dict[str, Any]],
    name: str,
    notices: list[str],
) -> str:
    if export_state is None:
        return chart_html
    svg = _extract_svg_fragment(chart_html)
    if svg is None:
        notices.append(f"SVG export skipped for {name}; svg markup not found.")
        return chart_html
    export_name = _reserve_export_name(export_state, name)
    export_state["planned"].append(
        f"{export_state['dir_name']}/{export_name}.svg"
    )
    export_state["queue"].append({"name": export_name, "svg": svg})
    return chart_html


def _graphviz_available(engine: str) -> bool:
    if not isinstance(engine, str) or not engine.strip():
        return False
    return shutil.which(engine) is not None


def _render_graphviz_svg(
    dot_source: str,
    *,
    engine: str,
    timeout_s: float = 10.0,
) -> tuple[Optional[str], Optional[str]]:
    if not _graphviz_available(engine):
        return None, f"Graphviz engine '{engine}' not available."
    try:
        proc = subprocess.run(
            [engine, "-Tsvg"],
            input=dot_source.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="ignore").strip()
        return None, error or f"Graphviz {engine} failed."
    return proc.stdout.decode("utf-8", errors="ignore"), None


def _graphviz_html_with_export(
    dot_source: str,
    export_state: Optional[dict[str, Any]],
    name: str,
    notices: list[str],
    *,
    engine: str,
    timeout_s: float = 10.0,
) -> str:
    svg, error = _render_graphviz_svg(
        dot_source, engine=engine, timeout_s=timeout_s
    )
    if svg is None:
        if error:
            notices.append(f"Graphviz render failed ({name}): {error}")
        escaped = html.escape(dot_source)
        return (
            "<p class=\"muted\">Graphviz render unavailable. DOT source:</p>"
            "<pre style=\"white-space:pre-wrap;background:#f6f4ef;border-radius:12px;"
            "padding:12px;font-size:11px;max-height:360px;overflow:auto;\">"
            f"{escaped}</pre>"
        )
    return _svg_html_with_export(svg, export_state, name, notices)


def _export_queued_plotly(
    export_state: Optional[dict[str, Any]],
    base_dir: Path,
    notices: list[str],
) -> list[str]:
    if export_state is None:
        return []
    queue = export_state.get("queue") or []
    if not queue:
        return []
    export_dir = base_dir / export_state["dir_name"]
    export_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    pio = export_state["pio"]
    for item in queue:
        name = item["name"]
        fig = item["fig"]
        for fmt in export_state["formats"]:
            target = export_dir / f"{name}.{fmt}"
            try:
                image_bytes = pio.to_image(
                    fig,
                    format=fmt,
                    scale=export_state["scale"],
                )
                target.write_bytes(image_bytes)
                exported.append(f"{export_state['dir_name']}/{target.name}")
            except Exception as exc:
                notices.append(f"Image export failed ({target.name}): {exc}")
    if exported:
        index_payload = {"files": exported}
        write_json_atomic(export_dir / "index.json", index_payload)
    return exported


def _export_queued_matplotlib(
    export_state: Optional[dict[str, Any]],
    base_dir: Path,
    notices: list[str],
) -> list[str]:
    if export_state is None:
        return []
    queue = export_state.get("queue") or []
    if not queue:
        return []
    export_dir = base_dir / export_state["dir_name"]
    export_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for item in queue:
        name = item["name"]
        images = item.get("images") or {}
        for fmt, payload in images.items():
            target = export_dir / f"{name}.{fmt}"
            try:
                target.write_bytes(payload)
                exported.append(f"{export_state['dir_name']}/{target.name}")
            except Exception as exc:
                notices.append(f"Image export failed ({target.name}): {exc}")
    if exported:
        index_payload = {"files": exported}
        write_json_atomic(export_dir / "index.json", index_payload)
    return exported


def _export_queued_svg(
    export_state: Optional[dict[str, Any]],
    base_dir: Path,
    notices: list[str],
) -> list[str]:
    if export_state is None:
        return []
    queue = export_state.get("queue") or []
    if not queue:
        return []
    export_dir = base_dir / export_state["dir_name"]
    export_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for item in queue:
        name = item["name"]
        svg = item.get("svg")
        if not svg:
            continue
        target = export_dir / f"{name}.svg"
        try:
            target.write_text(svg, encoding="utf-8")
            exported.append(f"{export_state['dir_name']}/{target.name}")
        except Exception as exc:
            notices.append(f"Image export failed ({target.name}): {exc}")
    if exported:
        index_payload = {"files": exported}
        write_json_atomic(export_dir / "index.json", index_payload)
    return exported


def _extract_svg_fragment(chart_html: str) -> Optional[str]:
    start = chart_html.find("<svg")
    end = chart_html.rfind("</svg>")
    if start == -1 or end == -1:
        return None
    return chart_html[start : end + len("</svg>")]


def _build_svg_line_chart(
    *,
    title: str,
    times: Sequence[float],
    series: Mapping[str, Sequence[float]],
    unit: Optional[str] = None,
) -> str:
    if not times or not series:
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 200
    margin_left = 44
    margin_right = 12
    margin_top = 18
    margin_bottom = 28

    all_values: list[float] = []
    for values in series.values():
        all_values.extend(values)
    if not all_values:
        return "<p class=\"muted\">No data available.</p>"

    min_x = min(times)
    max_x = max(times)
    min_y = min(all_values)
    max_y = max(all_values)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    x_span = max_x - min_x
    y_span = max_y - min_y
    x_scale = (width - margin_left - margin_right) / x_span
    y_scale = (height - margin_top - margin_bottom) / y_span

    def _point_pair(x_val: float, y_val: float) -> str:
        x = margin_left + (x_val - min_x) * x_scale
        y = height - margin_bottom - (y_val - min_y) * y_scale
        return f"{x:.1f},{y:.1f}"

    palette = ["#0f6f68", "#d17b0f", "#3a6ea5", "#8b4a2a", "#5c8f3a"]
    lines = []
    legend_items = []
    for idx, (name, values) in enumerate(series.items()):
        if len(values) != len(times):
            continue
        points = " ".join(
            _point_pair(x_val, y_val) for x_val, y_val in zip(times, values)
        )
        color = palette[idx % len(palette)]
        lines.append(
            f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{points}\" />"
        )
        legend_items.append(
            "<div style=\"display:flex;align-items:center;gap:6px;\">"
            f"<span style=\"width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;\"></span>"
            f"<span>{html.escape(name)}</span>"
            "</div>"
        )

    unit_text = f" ({html.escape(unit)})" if unit else ""
    legend_html = "".join(legend_items)
    if legend_html:
        legend_html = (
            "<div style=\"display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted);\">"
            + legend_html
            + "</div>"
        )

    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}{unit_text}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        f"<line x1=\"{margin_left}\" y1=\"{height - margin_bottom}\" "
        f"x2=\"{width - margin_right}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" "
        f"x2=\"{margin_left}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        + "".join(lines)
        + "</svg>"
        + legend_html
        + "</div>"
    )


def _build_svg_bar_chart(
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    unit: Optional[str] = None,
) -> str:
    if not labels or not values or len(labels) != len(values):
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 220
    margin_left = 40
    margin_right = 14
    margin_top = 20
    margin_bottom = 44

    max_val = max(values) if values else 0.0
    if max_val <= 0:
        max_val = 1.0
    bar_area_width = width - margin_left - margin_right
    bar_area_height = height - margin_top - margin_bottom
    bar_width = bar_area_width / max(len(values), 1)

    bars = []
    label_items = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + idx * bar_width + bar_width * 0.1
        w = bar_width * 0.8
        h = (value / max_val) * bar_area_height
        y = margin_top + (bar_area_height - h)
        bars.append(
            f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{w:.1f}\" height=\"{h:.1f}\" "
            "rx=\"3\" fill=\"#0f6f68\" opacity=\"0.85\" />"
        )
        if label:
            label_items.append(
                f"<text x=\"{x + w / 2:.1f}\" y=\"{height - 16}\" "
                "text-anchor=\"middle\" font-size=\"10\" fill=\"#5f6c77\">"
                f"{html.escape(str(label))}</text>"
            )

    unit_text = f" ({html.escape(unit)})" if unit else ""
    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}{unit_text}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" "
        f"x2=\"{margin_left}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        f"<line x1=\"{margin_left}\" y1=\"{height - margin_bottom}\" "
        f"x2=\"{width - margin_right}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        + "".join(bars)
        + "".join(label_items)
        + "</svg>"
        "</div>"
    )


def _build_svg_barh_chart(
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    unit: Optional[str] = None,
) -> str:
    if not labels or not values or len(labels) != len(values):
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 220 + 14 * max(len(labels) - 6, 0)
    margin_left = 140
    margin_right = 16
    margin_top = 20
    margin_bottom = 24

    max_val = max(values) if values else 0.0
    if max_val <= 0:
        max_val = 1.0
    bar_area_width = width - margin_left - margin_right
    bar_area_height = height - margin_top - margin_bottom
    bar_height = bar_area_height / max(len(values), 1)

    bars = []
    label_items = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * bar_height + bar_height * 0.1
        h = bar_height * 0.8
        w = (value / max_val) * bar_area_width
        x = margin_left
        bars.append(
            f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{w:.1f}\" height=\"{h:.1f}\" "
            "rx=\"3\" fill=\"#0f6f68\" opacity=\"0.85\" />"
        )
        label_items.append(
            f"<text x=\"{margin_left - 6}\" y=\"{y + h * 0.7:.1f}\" "
            "text-anchor=\"end\" font-size=\"10\" fill=\"#5f6c77\">"
            f"{html.escape(str(label))}</text>"
        )

    unit_text = f" ({html.escape(unit)})" if unit else ""
    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}{unit_text}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" "
        f"x2=\"{margin_left}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        + "".join(bars)
        + "".join(label_items)
        + "</svg>"
        "</div>"
    )


def _build_svg_histogram_chart(
    *,
    title: str,
    values: Sequence[float],
    bins: int = 20,
    unit: Optional[str] = None,
) -> str:
    if not values:
        return "<p class=\"muted\">No data available.</p>"
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0
    span = max_val - min_val
    bin_width = span / max(bins, 1)
    counts = [0] * max(bins, 1)
    for value in values:
        try:
            idx = int((value - min_val) / bin_width)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            idx = 0
        if idx >= len(counts):
            idx = len(counts) - 1
        counts[idx] += 1
    labels = []
    for idx in range(len(counts)):
        if len(counts) <= 8 or idx % max(len(counts) // 6, 1) == 0:
            center = min_val + (idx + 0.5) * bin_width
            labels.append(f"{center:.2g}")
        else:
            labels.append("")
    return _build_svg_bar_chart(
        title=title,
        labels=labels,
        values=counts,
        unit=unit,
    )


def _build_svg_scatter_chart(
    *,
    title: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    if not x_values or not y_values or len(x_values) != len(y_values):
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 220
    margin_left = 44
    margin_right = 12
    margin_top = 18
    margin_bottom = 36

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    x_span = max_x - min_x
    y_span = max_y - min_y
    x_scale = (width - margin_left - margin_right) / x_span
    y_scale = (height - margin_top - margin_bottom) / y_span

    points = []
    for x_val, y_val in zip(x_values, y_values):
        x = margin_left + (x_val - min_x) * x_scale
        y = height - margin_bottom - (y_val - min_y) * y_scale
        points.append(
            f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"3\" fill=\"#0f6f68\" opacity=\"0.75\" />"
        )
    axis_label = ""
    if x_label or y_label:
        axis_label = f"{x_label or ''} vs {y_label or ''}".strip()
    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        f"<line x1=\"{margin_left}\" y1=\"{height - margin_bottom}\" "
        f"x2=\"{width - margin_right}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" "
        f"x2=\"{margin_left}\" y2=\"{height - margin_bottom}\" "
        "stroke=\"#c8d0d4\" stroke-width=\"1\" />"
        + "".join(points)
        + "</svg>"
        + (f"<div class=\"muted\" style=\"font-size:12px;\">{html.escape(axis_label)}</div>" if axis_label else "")
        + "</div>"
    )


def _heat_color(value: Optional[float], max_abs: float) -> str:
    if value is None:
        return "#efefef"
    if max_abs <= 0:
        return "#f5f5f5"
    ratio = max(min(value / max_abs, 1.0), -1.0)
    if ratio >= 0:
        base = (240, 240, 240)
        target = (197, 74, 74)
        t = ratio
    else:
        base = (240, 240, 240)
        target = (64, 120, 198)
        t = abs(ratio)
    r = int(base[0] + (target[0] - base[0]) * t)
    g = int(base[1] + (target[1] - base[1]) * t)
    b = int(base[2] + (target[2] - base[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_svg_heatmap_chart(
    *,
    title: str,
    z_values: Sequence[Sequence[Optional[float]]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
) -> str:
    if not z_values or not x_labels or not y_labels:
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 240 + 12 * max(len(y_labels) - 6, 0)
    margin_left = 120
    margin_right = 12
    margin_top = 26
    margin_bottom = 40

    flat_values = [
        abs(value)
        for row in z_values
        for value in row
        if value is not None
    ]
    max_abs = max(flat_values) if flat_values else 0.0
    grid_width = width - margin_left - margin_right
    grid_height = height - margin_top - margin_bottom
    cell_w = grid_width / max(len(x_labels), 1)
    cell_h = grid_height / max(len(y_labels), 1)

    rects = []
    for row_idx, row in enumerate(z_values):
        for col_idx, value in enumerate(row):
            x = margin_left + col_idx * cell_w
            y = margin_top + row_idx * cell_h
            color = _heat_color(value, max_abs)
            rects.append(
                f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{cell_w:.1f}\" height=\"{cell_h:.1f}\" "
                f"fill=\"{color}\" stroke=\"#ffffff\" stroke-width=\"0.5\" />"
            )

    x_text = []
    for idx, label in enumerate(x_labels):
        x = margin_left + idx * cell_w + cell_w / 2
        x_text.append(
            f"<text x=\"{x:.1f}\" y=\"{height - 16}\" text-anchor=\"middle\" "
            "font-size=\"9\" fill=\"#5f6c77\">"
            f"{html.escape(label)}</text>"
        )
    y_text = []
    for idx, label in enumerate(y_labels):
        y = margin_top + idx * cell_h + cell_h / 2 + 4
        y_text.append(
            f"<text x=\"{margin_left - 6}\" y=\"{y:.1f}\" text-anchor=\"end\" "
            "font-size=\"9\" fill=\"#5f6c77\">"
            f"{html.escape(label)}</text>"
        )

    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        + "".join(rects)
        + "".join(x_text)
        + "".join(y_text)
        + "</svg>"
        "</div>"
    )


def _build_svg_network_chart(
    *,
    title: str,
    positions: Mapping[str, tuple[float, float]],
    edges: Sequence[tuple[str, str]],
    labels: Mapping[str, str],
) -> str:
    if not positions:
        return "<p class=\"muted\">No data available.</p>"
    width = 520
    height = 320
    margin = 20

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    x_span = max_x - min_x
    y_span = max_y - min_y
    x_scale = (width - 2 * margin) / x_span
    y_scale = (height - 2 * margin) / y_span

    def _scale_point(point: tuple[float, float]) -> tuple[float, float]:
        x_val, y_val = point
        x = margin + (x_val - min_x) * x_scale
        y = height - margin - (y_val - min_y) * y_scale
        return x, y

    edge_lines = []
    for source_id, target_id in edges:
        if source_id not in positions or target_id not in positions:
            continue
        x0, y0 = _scale_point(positions[source_id])
        x1, y1 = _scale_point(positions[target_id])
        edge_lines.append(
            f"<line x1=\"{x0:.1f}\" y1=\"{y0:.1f}\" x2=\"{x1:.1f}\" y2=\"{y1:.1f}\" "
            "stroke=\"#9bb3b0\" stroke-width=\"1\" />"
        )

    nodes = []
    text_items = []
    for node_id, point in positions.items():
        x, y = _scale_point(point)
        label = labels.get(node_id, node_id)
        nodes.append(
            f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"5\" fill=\"#0f6f68\" stroke=\"#ffffff\" stroke-width=\"1\" />"
        )
        text_items.append(
            f"<text x=\"{x:.1f}\" y=\"{y - 8:.1f}\" text-anchor=\"middle\" "
            "font-size=\"9\" fill=\"#5f6c77\">"
            f"{html.escape(label)}</text>"
        )

    return (
        "<div>"
        f"<div style=\"font-size:12px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);\">"
        f"{html.escape(title)}</div>"
        f"<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
        + "".join(edge_lines)
        + "".join(nodes)
        + "".join(text_items)
        + "</svg>"
        "</div>"
    )


def _network_output_dir(run_root: Path) -> Path:
    target = run_root / "viz" / "network"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _timeseries_output_dir(run_root: Path) -> Path:
    target = run_root / "viz" / "timeseries"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _reduction_output_dir(run_root: Path) -> Path:
    target = run_root / "viz" / "reduction"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _placeholder_svg(message: str) -> str:
    safe = html.escape(message)
    return (
        "<svg width=\"520\" height=\"180\" viewBox=\"0 0 520 180\" "
        "xmlns=\"http://www.w3.org/2000/svg\">"
        "<rect width=\"520\" height=\"180\" fill=\"#fbfaf7\" stroke=\"#dcd6cb\" />"
        f"<text x=\"20\" y=\"90\" fill=\"#5f6c77\" font-size=\"12\">{safe}</text>"
        "</svg>"
    )


def _write_svg(path: Path, svg: str) -> None:
    if "<svg" not in svg:
        svg = _placeholder_svg(svg)
    path.write_text(svg, encoding="utf-8")


def _load_csr_edges(path: Path) -> tuple[list[tuple[int, int, float]], tuple[int, int]]:
    if not path.exists():
        raise ConfigError(f"species graph layer not found: {path}")
    if sp is not None:
        matrix = sp.load_npz(path)
        coo = matrix.tocoo()
        edges = [
            (int(r), int(c), float(v))
            for r, c, v in zip(coo.row, coo.col, coo.data)
            if v != 0
        ]
        return edges, matrix.shape
    if np is None:
        raise ConfigError("numpy is required to load species graph layers.")
    payload = np.load(path, allow_pickle=False)
    if "data" not in payload:
        raise ConfigError("species graph layer missing CSR data.")
    data = payload["data"]
    indices = payload["indices"]
    indptr = payload["indptr"]
    shape = tuple(int(value) for value in payload["shape"])
    edges: list[tuple[int, int, float]] = []
    for row in range(shape[0]):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        for idx in range(start, end):
            col = int(indices[idx])
            value = float(data[idx])
            if value != 0.0:
                edges.append((row, col, value))
    return edges, shape


def _select_top_nodes_edges(
    edges: Sequence[tuple[int, int, float]],
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    if not edges:
        return [], []
    scored: dict[int, float] = {}
    for src, tgt, weight in edges:
        scored[src] = scored.get(src, 0.0) + abs(weight)
        scored[tgt] = scored.get(tgt, 0.0) + abs(weight)
    top_nodes = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    node_ids = [node_id for node_id, _ in top_nodes[:max_nodes]]
    node_set = set(node_ids)
    filtered_edges = [
        (src, tgt, weight)
        for src, tgt, weight in edges
        if src in node_set and tgt in node_set
    ]
    filtered_edges.sort(key=lambda item: abs(item[2]), reverse=True)
    return node_ids, filtered_edges[:max_edges]


def _circle_layout(node_ids: Sequence[str]) -> dict[str, tuple[float, float]]:
    if np is None:
        return {node_id: (float(idx), 0.0) for idx, node_id in enumerate(node_ids)}
    count = max(len(node_ids), 1)
    positions: dict[str, tuple[float, float]] = {}
    for idx, node_id in enumerate(node_ids):
        angle = 2.0 * math.pi * idx / float(count)
        positions[node_id] = (math.cos(angle), math.sin(angle))
    return positions


def _emit_network_exports(
    *,
    run_root: Path,
    run_id: str,
    store: ArtifactStore,
    run_payload: Optional[Mapping[str, Any]],
    bipartite_manifest: Optional[ArtifactManifest],
    bipartite_payload: Optional[Mapping[str, Any]],
    bipartite_data: Optional[Mapping[str, Any]],
    flux_manifest: Optional[ArtifactManifest],
    flux_payload: Optional[Mapping[str, Any]],
    reduction_ids: Sequence[str],
    graphviz_cfg: Mapping[str, Any],
    notices: list[str],
) -> None:
    target_dir = _network_output_dir(run_root)

    export_dot = bool(graphviz_cfg.get("export_dot", True))
    export_svg = bool(graphviz_cfg.get("export_svg", True))
    show_edge_labels = bool(graphviz_cfg.get("show_edge_labels", False))
    emit_all_patches = bool(
        graphviz_cfg.get("emit_all_patches", graphviz_cfg.get("patch_all", False))
    )

    top_n = _coerce_positive_int(
        graphviz_cfg.get("top_n"), "viz.graphviz.top_n", default=8
    )
    max_nodes = _coerce_positive_int(
        graphviz_cfg.get("max_nodes"), "viz.graphviz.max_nodes", default=80
    )
    max_edges = _coerce_positive_int(
        graphviz_cfg.get("max_edges"), "viz.graphviz.max_edges", default=160
    )

    engine_bipartite = graphviz_cfg.get("engine_bipartite") or graphviz_cfg.get("engine") or "dot"
    if not isinstance(engine_bipartite, str) or not engine_bipartite.strip():
        raise ConfigError("viz.graphviz.engine_bipartite must be a non-empty string.")
    engine_bipartite = engine_bipartite.strip()

    engine_flux = graphviz_cfg.get("engine_flux") or "sfdp"
    if not isinstance(engine_flux, str) or not engine_flux.strip():
        raise ConfigError("viz.graphviz.engine_flux must be a non-empty string.")
    engine_flux = engine_flux.strip()

    flux_layer_raw = graphviz_cfg.get("flux_layer", 0)
    if isinstance(flux_layer_raw, bool):
        raise ConfigError("viz.graphviz.flux_layer must be an integer.")
    try:
        flux_layer = int(flux_layer_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("viz.graphviz.flux_layer must be an integer.") from exc
    if flux_layer < 0:
        raise ConfigError("viz.graphviz.flux_layer must be >= 0.")

    plots_raw = graphviz_cfg.get("plots")
    if plots_raw is None:
        plots = [
            "top_rop",
            "top_wdot",
            "patch",
            "top_rop_reduced",
            "flux_species",
        ]
    else:
        if isinstance(plots_raw, str):
            raw_list = [plots_raw]
        elif isinstance(plots_raw, Sequence) and not isinstance(plots_raw, (bytes, bytearray)):
            raw_list = list(plots_raw)
        else:
            raise ConfigError("viz.graphviz.plots must be a string or list of strings.")
        plots = []
        for entry in raw_list:
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError("viz.graphviz.plots entries must be non-empty strings.")
            plots.append(entry.strip().lower().replace("-", "_"))

    allowed_plots = {
        "top_rop",
        "top_wdot",
        "patch",
        "top_rop_reduced",
        "flux_species",
        "rxn_proj",
        "diff_disabled",
    }
    for entry in plots:
        if entry not in allowed_plots:
            raise ConfigError(
                "viz.graphviz.plots contains unsupported entry: " + str(entry)
            )

    rop_stat = graphviz_cfg.get("rop_stat") or "integral"
    wdot_stat = graphviz_cfg.get("wdot_stat") or "integral"
    if rop_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.graphviz.rop_stat must be a supported stat.")
    if wdot_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.graphviz.wdot_stat must be a supported stat.")
    rop_var = graphviz_cfg.get("rop_var") or "rop_net"
    wdot_var = graphviz_cfg.get("wdot_var") or "net_production_rates"

    if export_svg and engine_flux and not _graphviz_available(engine_flux):
        if _graphviz_available(engine_bipartite):
            notices.append(
                f"Graphviz engine '{engine_flux}' not available; falling back to '{engine_bipartite}' for flux graphs."
            )
            engine_flux = engine_bipartite

    index_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "inputs": {
            "bipartite_graph_id": bipartite_manifest.id if bipartite_manifest is not None else None,
            "flux_graph_id": flux_manifest.id if flux_manifest is not None else None,
            "reduction_ids": list(reduction_ids),
        },
        "params": {
            "top_n": top_n,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "plots": list(plots),
            "engine_bipartite": engine_bipartite,
            "engine_flux": engine_flux,
            "flux_layer": flux_layer,
            "export_dot": export_dot,
            "export_svg": export_svg,
            "show_edge_labels": show_edge_labels,
            "emit_all_patches": emit_all_patches,
            "rop_stat": rop_stat,
            "wdot_stat": wdot_stat,
        },
        "plots": [],
    }

    def _emit_dot_svg(
        *,
        plot: str,
        basename: str,
        dot_source: str,
        engine: str,
        inputs: Mapping[str, Any],
        note: Optional[str] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "plot": plot,
            "basename": basename,
            "inputs": dict(inputs),
        }
        if note:
            entry["note"] = note
        if export_dot:
            dot_path = target_dir / f"{basename}.dot"
            try:
                dot_path.write_text(dot_source, encoding="utf-8")
                entry["dot"] = dot_path.name
            except Exception as exc:
                notices.append(f"DOT export failed ({dot_path.name}): {exc}")
        if export_svg:
            svg, error = _render_graphviz_svg(dot_source, engine=engine)
            if svg is not None:
                svg_path = target_dir / f"{basename}.svg"
                try:
                    svg_path.write_text(svg, encoding="utf-8")
                    entry["svg"] = svg_path.name
                except Exception as exc:
                    notices.append(f"SVG export failed ({svg_path.name}): {exc}")
            else:
                entry["svg_error"] = error or "render failed"
        index_payload["plots"].append(entry)

    bip_nodes: list[dict[str, Any]] = []
    bip_links: list[dict[str, Any]] = []
    if bipartite_data is not None:
        nodes_raw = bipartite_data.get("nodes")
        links_raw = bipartite_data.get("links")
        if isinstance(nodes_raw, Sequence) and not isinstance(nodes_raw, (str, bytes, bytearray)):
            bip_nodes = [dict(n) for n in nodes_raw if isinstance(n, Mapping)]
        if isinstance(links_raw, Sequence) and not isinstance(links_raw, (str, bytes, bytearray)):
            bip_links = [dict(l) for l in links_raw if isinstance(l, Mapping)]

    rop_ranked: list[tuple[str, float]] = []
    wdot_ranked: list[tuple[str, float]] = []
    if run_payload is not None:
        rop_ranked, rop_err = _rank_from_run_payload(
            run_payload,
            var_name=str(rop_var),
            axis="reaction",
            stat=str(rop_stat),
            top_n=top_n,
            rank_abs=True,
        )
        if rop_err is not None:
            notices.append(f"Top ROP ranking unavailable: {rop_err}")
        wdot_ranked, wdot_err = _rank_from_run_payload(
            run_payload,
            var_name=str(wdot_var),
            axis="species",
            stat=str(wdot_stat),
            top_n=top_n,
            rank_abs=True,
        )
        if wdot_err is not None:
            notices.append(f"Top WDOT ranking unavailable: {wdot_err}")

    primary_reduction_id: Optional[str] = None
    if reduction_ids:
        first = reduction_ids[0]
        primary_reduction_id = first if isinstance(first, str) and first.strip() else None

    patch_payload: Optional[dict[str, Any]] = None
    disabled_ids: set[str] = set()
    disabled_indices: set[int] = set()
    if primary_reduction_id is not None:
        patch_path = store.artifact_dir("reduction", primary_reduction_id) / "mechanism_patch.yaml"
        payload, error = _read_patch_payload(patch_path)
        if error is not None:
            notices.append(f"Reduction patch {primary_reduction_id}: {error}")
        else:
            patch_payload = payload

    def _extract_disabled(patch: Mapping[str, Any]) -> None:
        for key in ("disabled_reactions", "reaction_multipliers"):
            entries = patch.get(key)
            if isinstance(entries, Mapping):
                entries = [entries]
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                reaction_id = entry.get("reaction_id") or entry.get("reaction")
                if isinstance(reaction_id, str) and reaction_id.strip():
                    disabled_ids.add(reaction_id.strip())
                idx = _coerce_optional_int(entry.get("index"))
                if idx is not None:
                    disabled_indices.add(int(idx))

    if patch_payload is not None:
        _extract_disabled(patch_payload)

    def _disabled_node_ids(nodes: Sequence[Mapping[str, Any]]) -> set[str]:
        disabled_node_ids: set[str] = set()
        for node in nodes:
            if node.get("kind") != "reaction":
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            identifiers: list[str] = []
            for key in ("reaction_equation", "equation", "reaction_id", "label"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    identifiers.append(value.strip())
            reaction_index = _coerce_optional_int(node.get("reaction_index"))
            if any(identifier in disabled_ids for identifier in identifiers) or (
                reaction_index is not None and reaction_index in disabled_indices
            ):
                disabled_node_ids.add(node_id)
        return disabled_node_ids

    def _filter_disabled(
        nodes: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
        disabled_node_ids = _disabled_node_ids(nodes)
        if not disabled_node_ids:
            return [dict(n) for n in nodes], [dict(l) for l in links], set()
        filtered_nodes = [
            dict(node)
            for node in nodes
            if isinstance(node, Mapping)
            and isinstance(node.get("id"), str)
            and node.get("id") not in disabled_node_ids
        ]
        filtered_links = [
            dict(link)
            for link in links
            if isinstance(link, Mapping)
            and link.get("source") not in disabled_node_ids
            and link.get("target") not in disabled_node_ids
        ]
        return filtered_nodes, filtered_links, disabled_node_ids

    def _combine_notes(*parts: Optional[str]) -> Optional[str]:
        cleaned = [str(part).strip() for part in parts if part and str(part).strip()]
        return " ".join(cleaned) if cleaned else None

    def _fallback_top_reactions(*, count: int) -> tuple[list[str], list[int]]:
        if count <= 0:
            return [], []
        node_by_id = {
            node.get("id"): node
            for node in bip_nodes
            if isinstance(node.get("id"), str) and node.get("id")
        }
        reaction_degrees: Counter[str] = Counter()
        for link in bip_links:
            target = link.get("target")
            if isinstance(target, str) and target in node_by_id:
                reaction_degrees[target] += 1
        reaction_ids: list[str] = []
        reaction_indices: list[int] = []
        for node_id, _deg in reaction_degrees.most_common():
            node = node_by_id.get(node_id)
            if not isinstance(node, Mapping) or node.get("kind") != "reaction":
                continue
            idx = _coerce_optional_int(node.get("reaction_index"))
            if idx is not None:
                reaction_indices.append(int(idx))
            else:
                identifier = (
                    node.get("reaction_equation")
                    or node.get("equation")
                    or node.get("reaction_id")
                    or node.get("label")
                    or node_id
                )
                reaction_ids.append(str(identifier))
            if len(reaction_indices) + len(reaction_ids) >= count:
                break
        return reaction_ids[:count], reaction_indices[:count]

    def _fallback_top_species(*, count: int) -> list[str]:
        if count <= 0:
            return []
        node_by_id = {
            node.get("id"): node
            for node in bip_nodes
            if isinstance(node.get("id"), str) and node.get("id")
        }
        species_degrees: Counter[str] = Counter()
        for link in bip_links:
            source = link.get("source")
            if isinstance(source, str) and source in node_by_id:
                species_degrees[source] += 1
        species_names: list[str] = []
        for node_id, _deg in species_degrees.most_common():
            node = node_by_id.get(node_id)
            if not isinstance(node, Mapping) or node.get("kind") != "species":
                continue
            identifier = node.get("label") or node.get("species") or node.get("name") or node_id
            species_names.append(str(identifier))
            if len(species_names) >= count:
                break
        return species_names[:count]

    # 1) Top ROP Reaction Network (bipartite)
    if "top_rop" in plots and bip_nodes and bip_links:
        fallback_note = None
        reaction_ids = [name for name, _ in rop_ranked] if rop_ranked else []
        reaction_indices: list[int] = []
        if not reaction_ids:
            fallback_note = "ROP ranking unavailable; seeded by node degree."
            reaction_ids, reaction_indices = _fallback_top_reactions(count=top_n)
        nodes, links, note = _select_bipartite_subgraph(
            bip_nodes,
            bip_links,
            reaction_ids=reaction_ids or None,
            reaction_indices=reaction_indices or None,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not nodes or not links:
            notices.append("Top ROP network skipped: no bipartite subgraph found.")
        else:
            seed_set = set(reaction_ids)
            seed_indices = set(reaction_indices)
            highlight = {
                node.get("id")
                for node in nodes
                if node.get("kind") == "reaction"
                and isinstance(node.get("id"), str)
                and (
                    (_coerce_optional_int(node.get("reaction_index")) in seed_indices)
                    or (
                        isinstance(node.get("reaction_equation"), str)
                        and node.get("reaction_equation") in seed_set
                    )
                    or (
                        isinstance(node.get("reaction_id"), str)
                        and node.get("reaction_id") in seed_set
                    )
                    or (isinstance(node.get("label"), str) and node.get("label") in seed_set)
                )
            }
            note_text = _combine_notes(note, fallback_note)
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Top ROP Reaction Network",
                highlight_reactions=highlight,
                note=note_text,
                show_edge_labels=show_edge_labels,
            )
            _emit_dot_svg(
                plot="top_rop",
                basename=f"top_rop__{run_id}__bipartite",
                dot_source=dot_source,
                engine=engine_bipartite,
                inputs={
                    "run_id": run_id,
                    "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                },
                note=note_text,
            )

    # 2) Top WDOT Species Network (bipartite)
    if "top_wdot" in plots and bip_nodes and bip_links:
        fallback_note = None
        species_names = [name for name, _ in wdot_ranked] if wdot_ranked else []
        if not species_names:
            fallback_note = "WDOT ranking unavailable; seeded by node degree."
            species_names = _fallback_top_species(count=top_n)
        nodes, links, note = _select_bipartite_subgraph(
            bip_nodes,
            bip_links,
            species_names=species_names or None,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not nodes or not links:
            notices.append("Top WDOT network skipped: no bipartite subgraph found.")
        else:
            seed_set = set(species_names)
            highlight = {
                node.get("id")
                for node in nodes
                if node.get("kind") == "species"
                and isinstance(node.get("id"), str)
                and (
                    (isinstance(node.get("label"), str) and node.get("label") in seed_set)
                    or (
                        isinstance(node.get("species"), str)
                        and node.get("species") in seed_set
                    )
                )
            }
            note_text = _combine_notes(note, fallback_note)
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Top WDOT Species Network",
                highlight_species=highlight,
                note=note_text,
                show_edge_labels=show_edge_labels,
            )
            _emit_dot_svg(
                plot="top_wdot",
                basename=f"top_wdot__{run_id}__bipartite",
                dot_source=dot_source,
                engine=engine_bipartite,
                inputs={
                    "run_id": run_id,
                    "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                },
                note=note_text,
            )

    # Optional P1: Reaction–Reaction Projection (top-ROP seed, shared-species weight)
    if "rxn_proj" in plots and bip_nodes and bip_links:
        fallback_note = None
        reaction_ids = [name for name, _ in rop_ranked] if rop_ranked else []
        reaction_indices: list[int] = []
        if not reaction_ids:
            fallback_note = "ROP ranking unavailable; seeded by node degree."
            reaction_ids, reaction_indices = _fallback_top_reactions(count=top_n)
        nodes, links, note = _select_bipartite_subgraph(
            bip_nodes,
            bip_links,
            reaction_ids=reaction_ids or None,
            reaction_indices=reaction_indices or None,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not nodes:
            notices.append("Reaction projection skipped: no bipartite subgraph found.")
        else:
            reaction_nodes = [
                dict(node)
                for node in nodes
                if isinstance(node, Mapping) and node.get("kind") == "reaction"
            ]
            reaction_node_ids = [
                node.get("id")
                for node in reaction_nodes
                if isinstance(node.get("id"), str) and node.get("id")
            ]
            neighbor_map: dict[str, set[str]] = {node_id: set() for node_id in reaction_node_ids}
            for link in links:
                if not isinstance(link, Mapping):
                    continue
                source = link.get("source")
                target = link.get("target")
                if isinstance(target, str) and target in neighbor_map and isinstance(source, str):
                    neighbor_map[target].add(source)

            edges_proj: list[tuple[str, str, float]] = []
            for idx, src_id in enumerate(reaction_node_ids):
                src_neighbors = neighbor_map.get(src_id, set())
                for tgt_id in reaction_node_ids[idx + 1 :]:
                    shared = src_neighbors.intersection(neighbor_map.get(tgt_id, set()))
                    if shared:
                        edges_proj.append((src_id, tgt_id, float(len(shared))))
            edges_proj.sort(key=lambda item: item[2], reverse=True)
            trimmed_note = None
            if len(edges_proj) > max_edges:
                edges_proj = edges_proj[:max_edges]
                trimmed_note = "Graph trimmed for readability."

            note_text = _combine_notes(note, fallback_note, trimmed_note)
            dot_source = _build_graphviz_reaction_projection_dot(
                reaction_nodes,
                edges_proj,
                title="Reaction Projection (Top ROP)",
                note=note_text,
                show_edge_labels=show_edge_labels,
            )
            _emit_dot_svg(
                plot="rxn_proj",
                basename=f"rxn_proj__{run_id}__toprop",
                dot_source=dot_source,
                engine=engine_flux,
                inputs={
                    "run_id": run_id,
                    "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                },
                note=note_text,
            )

    # 3) Reduction Patch Neighborhood (bipartite)
    if "patch" in plots and bip_nodes and bip_links:
        patch_targets: list[tuple[str, dict[str, Any]]] = []
        if emit_all_patches and reduction_ids:
            for reduction_id in reduction_ids:
                if not isinstance(reduction_id, str) or not reduction_id.strip():
                    continue
                patch_path = (
                    store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
                )
                payload, error = _read_patch_payload(patch_path)
                if error is not None:
                    notices.append(f"Reduction patch {reduction_id}: {error}")
                    continue
                patch_targets.append((reduction_id, payload))
        elif patch_payload is not None and primary_reduction_id is not None:
            patch_targets.append((primary_reduction_id, patch_payload))

        for reduction_id, patch in patch_targets:
            reaction_ids: list[str] = []
            reaction_indices: list[int] = []
            for key in ("disabled_reactions", "reaction_multipliers"):
                entries = patch.get(key)
                if isinstance(entries, Mapping):
                    entries = [entries]
                if not isinstance(entries, Sequence) or isinstance(
                    entries, (str, bytes, bytearray)
                ):
                    continue
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    reaction_id = entry.get("reaction_id") or entry.get("reaction")
                    if isinstance(reaction_id, str) and reaction_id.strip():
                        reaction_ids.append(reaction_id.strip())
                    idx = _coerce_optional_int(entry.get("index"))
                    if idx is not None:
                        reaction_indices.append(int(idx))
            nodes, links, note = _select_bipartite_subgraph(
                bip_nodes,
                bip_links,
                reaction_ids=reaction_ids or None,
                reaction_indices=reaction_indices or None,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
            if not nodes or not links:
                notices.append(
                    f"Patch network skipped for {reduction_id}: no bipartite subgraph found."
                )
                continue
            disabled_node_ids = _disabled_node_ids(nodes)
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Reduction Patch Neighborhood",
                highlight_reactions=disabled_node_ids,
                disabled_reactions=disabled_node_ids,
                note=note,
                show_edge_labels=show_edge_labels,
            )
            _emit_dot_svg(
                plot="patch",
                basename=f"patch__{run_id}__{reduction_id}",
                dot_source=dot_source,
                engine=engine_bipartite,
                inputs={
                    "run_id": run_id,
                    "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                    "reduction_id": reduction_id,
                },
                note=note,
            )

    # 4) Reduced Top ROP Network (bipartite)
    if (
        "top_rop_reduced" in plots
        and bip_nodes
        and bip_links
        and (disabled_ids or disabled_indices)
    ):
        fallback_note = None
        reduced_nodes, reduced_links, removed = _filter_disabled(bip_nodes, bip_links)
        reaction_ids = [name for name, _ in rop_ranked] if rop_ranked else []
        reaction_indices: list[int] = []
        if not reaction_ids:
            fallback_note = "ROP ranking unavailable; seeded by node degree."
            reaction_ids, reaction_indices = _fallback_top_reactions(count=top_n)
        nodes, links, note = _select_bipartite_subgraph(
            reduced_nodes,
            reduced_links,
            reaction_ids=reaction_ids or None,
            reaction_indices=reaction_indices or None,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not nodes or not links:
            notices.append("Reduced Top ROP network skipped: no bipartite subgraph found.")
        else:
            seed_set = set(reaction_ids)
            seed_indices = set(reaction_indices)
            highlight = {
                node.get("id")
                for node in nodes
                if node.get("kind") == "reaction"
                and isinstance(node.get("id"), str)
                and (
                    (_coerce_optional_int(node.get("reaction_index")) in seed_indices)
                    or (
                        isinstance(node.get("reaction_equation"), str)
                        and node.get("reaction_equation") in seed_set
                    )
                    or (
                        isinstance(node.get("reaction_id"), str)
                        and node.get("reaction_id") in seed_set
                    )
                    or (isinstance(node.get("label"), str) and node.get("label") in seed_set)
                )
            }
            note_parts: list[str] = []
            if note:
                note_parts.append(note)
            if fallback_note:
                note_parts.append(fallback_note)
            if removed:
                note_parts.append(f"Removed {removed} disabled reactions.")
            note_text = " ".join(note_parts) if note_parts else None
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Top ROP Reaction Network (Reduced)",
                highlight_reactions=highlight,
                note=note_text,
                show_edge_labels=show_edge_labels,
            )
            _emit_dot_svg(
                plot="top_rop_reduced",
                basename=f"top_rop__{run_id}__reduced",
                dot_source=dot_source,
                engine=engine_bipartite,
                inputs={
                    "run_id": run_id,
                    "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                    "reduction_id": primary_reduction_id,
                },
                note=note_text,
            )

    # 5) Temporal Flux Species Graph (species graph)
    if "flux_species" in plots and flux_manifest is not None and flux_payload is not None:
        species_order = flux_payload.get("species", {}).get("order")
        if not isinstance(species_order, Sequence) or isinstance(species_order, (str, bytes, bytearray)):
            notices.append("Temporal flux graph missing species order.")
        else:
            layer_path = (
                store.artifact_dir("graphs", flux_manifest.id)
                / "species_graph"
                / f"layer_{flux_layer:03d}.npz"
            )
            try:
                edges, _shape = _load_csr_edges(layer_path)
                node_indices, edge_list = _select_top_nodes_edges(
                    edges, max_nodes=max_nodes, max_edges=max_edges
                )
                node_ids = [
                    str(species_order[idx])
                    for idx in node_indices
                    if 0 <= idx < len(species_order)
                ]
                named_edges = [
                    (str(species_order[src]), str(species_order[tgt]), float(weight))
                    for src, tgt, weight in edge_list
                    if 0 <= src < len(species_order) and 0 <= tgt < len(species_order)
                ]
                dot_source = _build_graphviz_species_graph_dot(
                    node_ids,
                    named_edges,
                    title="Temporal Flux Species Graph",
                    note=None,
                    show_edge_labels=show_edge_labels,
                )
                _emit_dot_svg(
                    plot="flux_species",
                    basename=f"flux_species__{run_id}__layer{flux_layer:03d}",
                    dot_source=dot_source,
                    engine=engine_flux,
                    inputs={
                        "run_id": run_id,
                        "graphs": flux_manifest.id,
                        "layer": flux_layer,
                    },
                    note=None,
                )
            except Exception as exc:
                notices.append(f"Flux species export failed: {exc}")

    # Optional P1: diff highlighting disabled reactions.
    if (
        "diff_disabled" in plots
        and bip_nodes
        and bip_links
        and (disabled_ids or disabled_indices)
    ):
        seed_reaction_ids = [name for name, _ in rop_ranked] if rop_ranked else []
        combined_ids = list(dict.fromkeys(seed_reaction_ids + sorted(disabled_ids)))
        nodes, links, note = _select_bipartite_subgraph(
            bip_nodes,
            bip_links,
            reaction_ids=combined_ids or None,
            reaction_indices=sorted(disabled_indices) or None,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        disabled_node_ids = _disabled_node_ids(nodes)
        dot_source = _build_graphviz_bipartite_dot(
            nodes,
            links,
            title="Disabled Reactions (Diff)",
            highlight_reactions=disabled_node_ids,
            disabled_reactions=disabled_node_ids,
            note=note,
            show_edge_labels=show_edge_labels,
        )
        _emit_dot_svg(
            plot="diff_disabled",
            basename=f"diff_disabled__{run_id}__bipartite",
            dot_source=dot_source,
            engine=engine_bipartite,
            inputs={
                "run_id": run_id,
                "graphs": bipartite_manifest.id if bipartite_manifest is not None else None,
                "reduction_id": primary_reduction_id,
            },
            note=note,
        )

    write_json_atomic(target_dir / "index.json", index_payload)


def _emit_timeseries_exports(
    *,
    run_root: Path,
    run_id: str,
    run_payload: Optional[Mapping[str, Any]],
    species_var: Optional[str],
    species_axis: Optional[str],
    species_names: Sequence[str],
    rate_var: Optional[str],
    rate_axis: Optional[str],
    rate_names: Sequence[str],
    max_points: int,
    notices: list[str],
) -> None:
    timeseries_dir = _timeseries_output_dir(run_root)
    reduction_dir = _reduction_output_dir(run_root)

    def _export_series(
        *,
        var_name: str,
        axis_name: str,
        names: Sequence[str],
        filename: str,
        title: str,
    ) -> None:
        if run_payload is None:
            _write_svg(timeseries_dir / filename, _placeholder_svg("run data unavailable"))
            return
        time_values, axis_values, matrix, error = _prepare_timeseries(
            run_payload,
            var_name,
            axis_name,
        )
        if error is not None or time_values is None or axis_values is None:
            _write_svg(timeseries_dir / filename, _placeholder_svg(str(error)))
            return
        axis_index = {name: idx for idx, name in enumerate(axis_values)}
        selected = [name for name in names if name in axis_index]
        if not selected:
            _write_svg(timeseries_dir / filename, _placeholder_svg("no series selected"))
            return
        indices = _downsample_indices(len(time_values), max_points)
        times = [time_values[i] for i in indices]
        series_map: dict[str, list[float]] = {}
        for name in selected:
            idx = axis_index[name]
            series_map[name] = [matrix[i][idx] for i in indices]
        chart_html = _build_svg_line_chart(
            title=title,
            times=times,
            series=series_map,
            unit=None,
        )
        _write_svg(timeseries_dir / filename, chart_html)

    if species_var and species_axis and species_names:
        _export_series(
            var_name=species_var,
            axis_name=species_axis,
            names=species_names,
            filename=f"species__{run_id}__{species_var}.svg",
            title="Species Time Series",
        )
    else:
        _write_svg(
            timeseries_dir / f"species__{run_id}__none.svg",
            _placeholder_svg("species series unavailable"),
        )

    if rate_var and rate_axis and rate_names:
        _export_series(
            var_name=rate_var,
            axis_name=rate_axis,
            names=rate_names,
            filename=f"rates__{run_id}__{rate_var}.svg",
            title="Rate Time Series",
        )
    else:
        _write_svg(
            timeseries_dir / f"rates__{run_id}__none.svg",
            _placeholder_svg("rate series unavailable"),
        )

    _write_svg(
        reduction_dir / f"qoi_compare__{run_id}__scatter.svg",
        _placeholder_svg("reduction comparison unavailable"),
    )


def _emit_validation_exports(
    *,
    run_root: Path,
    run_id: str,
    store: ArtifactStore,
    validation_manifests: Sequence[ArtifactManifest],
    superstate_mapping_id: Optional[str] = None,
    superreaction_graph_id: Optional[str] = None,
    notices: list[str],
) -> None:
    """Export reduction/validation summary charts into RunStore.

    These exports are deliberately lightweight (pure-SVG, no extra deps) so that
    reduction effect checks can be done quickly from `runs/.../viz/reduction/`.
    """

    if not validation_manifests:
        return

    reduction_dir = _reduction_output_dir(run_root)

    def _disabled_count_from_patch(patch_payload: Mapping[str, Any]) -> int:
        disabled_raw = patch_payload.get("disabled_reactions") or []
        count = 0
        if isinstance(disabled_raw, Mapping):
            count += len(disabled_raw)
        elif isinstance(disabled_raw, Sequence) and not isinstance(
            disabled_raw, (str, bytes, bytearray)
        ):
            count += len(disabled_raw)

        multipliers_raw = patch_payload.get("reaction_multipliers") or []
        if isinstance(multipliers_raw, Mapping):
            multipliers_raw = [multipliers_raw]
        if isinstance(multipliers_raw, Sequence) and not isinstance(
            multipliers_raw, (str, bytes, bytearray)
        ):
            for entry in multipliers_raw:
                if not isinstance(entry, Mapping):
                    continue
                try:
                    multiplier = float(entry.get("multiplier", 1.0))
                except (TypeError, ValueError):
                    continue
                if multiplier == 0.0:
                    count += 1
        return int(count)

    exported: list[str] = []
    for manifest in validation_manifests:
        table_path = store.artifact_dir("validation", manifest.id) / "metrics.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read validation/{manifest.id} metrics: {exc}")
            continue

        patch_specs = manifest.inputs.get("patches")
        patch_index_to_id: dict[int, str] = {}
        if isinstance(patch_specs, Sequence) and not isinstance(
            patch_specs, (str, bytes, bytearray)
        ):
            for entry in patch_specs:
                if not isinstance(entry, Mapping):
                    continue
                patch_index = _coerce_optional_int(entry.get("patch_index"))
                reduction_id = entry.get("reduction_id")
                if patch_index is None:
                    continue
                if isinstance(reduction_id, str) and reduction_id.strip():
                    patch_index_to_id[int(patch_index)] = reduction_id.strip()

        # Aggregate per patch.
        pass_by_patch: dict[int, list[bool]] = {}
        abs_by_patch: dict[int, list[float]] = {}
        for row in rows:
            patch_index = _coerce_optional_int(row.get("patch_index"))
            if patch_index is None:
                continue
            passed = row.get("passed")
            if isinstance(passed, bool):
                pass_by_patch.setdefault(int(patch_index), []).append(passed)
            abs_diff = _coerce_float(row.get("abs_diff"))
            if abs_diff is not None and not math.isnan(abs_diff) and not math.isinf(abs_diff):
                abs_by_patch.setdefault(int(patch_index), []).append(abs_diff)

        if not pass_by_patch:
            continue

        patch_indices = sorted(pass_by_patch.keys())
        labels: list[str] = []
        pass_rates: list[float] = []
        disabled_counts: list[float] = []
        mean_abs_diffs: list[float] = []

        for idx in patch_indices:
            reduction_id = patch_index_to_id.get(idx)
            label = f"patch_{idx}"
            disabled_count = math.nan
            mean_abs = math.nan
            if reduction_id is not None:
                # Prefer threshold labels (k=...) when available.
                try:
                    red_manifest = store.read_manifest("reduction", reduction_id)
                    threshold_cfg = (
                        red_manifest.inputs.get("threshold")
                        if isinstance(red_manifest.inputs, Mapping)
                        else None
                    )
                    if isinstance(threshold_cfg, Mapping):
                        top_k = threshold_cfg.get("top_k")
                        if top_k is not None:
                            try:
                                label = f"k{int(top_k)}"
                            except (TypeError, ValueError):
                                pass
                except Exception:
                    pass

                patch_path = store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
                payload, error = _read_patch_payload(patch_path)
                if error is None and payload is not None:
                    disabled_count = float(_disabled_count_from_patch(payload))

            pass_values = pass_by_patch.get(idx, [])
            pass_rate = sum(pass_values) / len(pass_values) if pass_values else 0.0
            abs_values = abs_by_patch.get(idx, [])
            if abs_values:
                mean_abs = sum(abs_values) / len(abs_values)

            labels.append(label)
            pass_rates.append(float(pass_rate))
            disabled_counts.append(float(disabled_count) if disabled_count == disabled_count else 0.0)
            mean_abs_diffs.append(float(mean_abs) if mean_abs == mean_abs else 0.0)

        prefix = f"{run_id}__{manifest.id}"
        pass_svg = _build_svg_bar_chart(
            title=f"Validation Pass Rate ({manifest.id})",
            labels=labels,
            values=pass_rates,
            unit="fraction",
        )
        pass_name = f"pass_rate__{prefix}.svg"
        _write_svg(reduction_dir / pass_name, _extract_svg_fragment(pass_svg) or pass_svg)
        exported.append(pass_name)

        disabled_svg = _build_svg_bar_chart(
            title=f"Disabled Reactions ({manifest.id})",
            labels=labels,
            values=disabled_counts,
            unit="count",
        )
        disabled_name = f"disabled__{prefix}.svg"
        _write_svg(
            reduction_dir / disabled_name,
            _extract_svg_fragment(disabled_svg) or disabled_svg,
        )
        exported.append(disabled_name)

        mean_abs_svg = _build_svg_bar_chart(
            title=f"Mean Abs Diff ({manifest.id})",
            labels=labels,
            values=mean_abs_diffs,
            unit="abs",
        )
        mean_abs_name = f"mean_abs_diff__{prefix}.svg"
        _write_svg(
            reduction_dir / mean_abs_name,
            _extract_svg_fragment(mean_abs_svg) or mean_abs_svg,
        )
        exported.append(mean_abs_name)

    # Build a single, human-readable comparison table across validation groups.
    compare_rows: list[dict[str, Any]] = []
    label_by_validation: dict[str, str] = {}
    base_species = None
    base_reactions = None

    superstate_count: Optional[int] = None
    if isinstance(superstate_mapping_id, str) and superstate_mapping_id.strip():
        mapping_path = (
            store.artifact_dir("reduction", superstate_mapping_id.strip()) / "mapping.json"
        )
        if mapping_path.exists():
            try:
                mapping_payload = read_json(mapping_path)
            except Exception as exc:
                notices.append(f"Failed to read reduction/{superstate_mapping_id}/mapping.json: {exc}")
                mapping_payload = None
            if isinstance(mapping_payload, Mapping):
                clusters = (
                    mapping_payload.get("clusters")
                    or mapping_payload.get("superstates")
                    or mapping_payload.get("clusters_payload")
                    or []
                )
                if isinstance(clusters, Sequence) and not isinstance(
                    clusters, (str, bytes, bytearray)
                ):
                    superstate_count = len(clusters)

    superreaction_baseline: Optional[int] = None
    superreaction_by_reduction: dict[str, int] = {}
    if isinstance(superreaction_graph_id, str) and superreaction_graph_id.strip():
        graph_path = (
            store.artifact_dir("graphs", superreaction_graph_id.strip()) / "graph.json"
        )
        if graph_path.exists():
            try:
                super_payload = read_json(graph_path)
            except Exception as exc:
                notices.append(f"Failed to read graphs/{superreaction_graph_id}/graph.json: {exc}")
                super_payload = None
            if isinstance(super_payload, Mapping) and super_payload.get("kind") == "superstate_reaction_merge_batch":
                baseline_section = super_payload.get("baseline")
                if isinstance(baseline_section, Mapping):
                    superreaction_baseline = _coerce_optional_int(
                        baseline_section.get("superreaction_exact_count")
                    )
                patches_section = super_payload.get("patches") or []
                if isinstance(patches_section, Sequence) and not isinstance(
                    patches_section, (str, bytes, bytearray)
                ):
                    for entry in patches_section:
                        if not isinstance(entry, Mapping):
                            continue
                        rid = entry.get("reduction_id")
                        count = entry.get("superreaction_exact_count")
                        if isinstance(rid, str) and rid.strip():
                            value = _coerce_optional_int(count)
                            if value is not None:
                                superreaction_by_reduction[rid.strip()] = int(value)

    summary_path = run_root / "summary.json"
    if summary_path.exists():
        try:
            summary_payload = read_json(summary_path)
        except Exception:
            summary_payload = None
        if isinstance(summary_payload, Mapping):
            results = summary_payload.get("results")
            if isinstance(results, Mapping):
                val_ids = {m.id for m in validation_manifests}
                for key, value in results.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        continue
                    if value in val_ids:
                        label_by_validation[value] = key
                graph_mech_id = results.get("graph_mech")
                if isinstance(graph_mech_id, str) and graph_mech_id.strip():
                    try:
                        graph_payload = read_json(
                            store.artifact_dir("graphs", graph_mech_id.strip()) / "graph.json"
                        )
                    except Exception:
                        graph_payload = None
                    if isinstance(graph_payload, Mapping):
                        species_section = graph_payload.get("species")
                        reactions_section = graph_payload.get("reactions")
                        if isinstance(species_section, Mapping):
                            base_species = _coerce_optional_int(species_section.get("count"))
                        elif isinstance(species_section, Sequence) and not isinstance(
                            species_section, (str, bytes, bytearray)
                        ):
                            base_species = len(species_section)
                        if isinstance(reactions_section, Mapping):
                            base_reactions = _coerce_optional_int(reactions_section.get("count"))
                        elif isinstance(reactions_section, Sequence) and not isinstance(
                            reactions_section, (str, bytes, bytearray)
                        ):
                            base_reactions = len(reactions_section)

    def _load_reduction_metrics(reduction_id: str) -> dict[str, Any]:
        try:
            payload = read_json(store.artifact_dir("reduction", reduction_id) / "metrics.json")
        except Exception:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    def _counts_from_reduction(reduction_id: str) -> dict[str, Any]:
        metrics = _load_reduction_metrics(reduction_id)
        counts = metrics.get("counts")
        if isinstance(counts, Mapping):
            return dict(counts)
        # Fallback: derive from patch file.
        patch_path = store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
        patch_payload, error = _read_patch_payload(patch_path)
        disabled_count = None
        merged_species = None
        if error is None and isinstance(patch_payload, Mapping):
            disabled_count = _coerce_optional_int(
                metrics.get("disabled_count")
                if isinstance(metrics.get("disabled_count"), (int, float))
                else None
            )
            if disabled_count is None:
                disabled_count = 0
                disabled_raw = patch_payload.get("disabled_reactions") or []
                if isinstance(disabled_raw, Mapping):
                    disabled_count += len(disabled_raw)
                elif isinstance(disabled_raw, Sequence) and not isinstance(
                    disabled_raw, (str, bytes, bytearray)
                ):
                    disabled_count += len(disabled_raw)
            state_merge = patch_payload.get("state_merge")
            if isinstance(state_merge, Mapping):
                mapping = state_merge.get("species_to_representative")
                if isinstance(mapping, Mapping):
                    merged_species = len(mapping)
        reactions_before = base_reactions
        species_before = base_species
        reactions_after = (
            reactions_before - disabled_count
            if isinstance(reactions_before, int) and isinstance(disabled_count, int)
            else None
        )
        species_after = (
            species_before - merged_species
            if isinstance(species_before, int) and isinstance(merged_species, int)
            else species_before
        )
        return {
            "species_before": species_before,
            "species_after": species_after,
            "merged_species": merged_species or 0,
            "reactions_before": reactions_before,
            "reactions_after": reactions_after,
            "disabled_reactions": disabled_count or 0,
            "merged_reactions": 0,
        }

    for manifest in validation_manifests:
        table_path = store.artifact_dir("validation", manifest.id) / "metrics.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception:
            continue

        patch_specs = manifest.inputs.get("patches")
        patch_index_to_id: dict[int, str] = {}
        if isinstance(patch_specs, Sequence) and not isinstance(
            patch_specs, (str, bytes, bytearray)
        ):
            for entry in patch_specs:
                if not isinstance(entry, Mapping):
                    continue
                patch_index = _coerce_optional_int(entry.get("patch_index"))
                reduction_id = entry.get("reduction_id")
                if patch_index is None:
                    continue
                if isinstance(reduction_id, str) and reduction_id.strip():
                    patch_index_to_id[int(patch_index)] = reduction_id.strip()

        per_patch: dict[int, dict[str, Any]] = {}
        for row in rows:
            patch_index = _coerce_optional_int(row.get("patch_index"))
            if patch_index is None:
                continue
            stats = per_patch.setdefault(
                int(patch_index),
                {"passed": [], "rel": [], "abs": []},
            )
            passed = row.get("passed")
            if isinstance(passed, bool):
                stats["passed"].append(passed)
            rel = _coerce_float(row.get("rel_diff"))
            if rel is not None and math.isfinite(rel):
                stats["rel"].append(float(rel))
            abs_diff = _coerce_float(row.get("abs_diff"))
            if abs_diff is not None and math.isfinite(abs_diff):
                stats["abs"].append(float(abs_diff))

        if not per_patch:
            continue

        # Select best patch: prefer all-pass, then maximize disabled reactions.
        candidates: list[tuple[tuple[int, int, float], int]] = []
        for idx, stats in per_patch.items():
            reduction_id = patch_index_to_id.get(idx)
            if reduction_id is None:
                continue
            passed_list = stats.get("passed") or []
            pass_rate = (sum(passed_list) / len(passed_list)) if passed_list else 0.0
            all_pass = 1 if pass_rate >= 1.0 - 1e-12 else 0
            counts = _counts_from_reduction(reduction_id)
            disabled = _coerce_optional_int(counts.get("disabled_reactions")) or 0
            rel_vals = stats.get("rel") or []
            mean_rel = (sum(rel_vals) / len(rel_vals)) if rel_vals else math.inf
            # sort key: all_pass desc, disabled desc, mean_rel asc
            candidates.append(((-all_pass, -disabled, float(mean_rel)), idx))

        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        best_idx = candidates[0][1]
        best_reduction = patch_index_to_id.get(best_idx)
        if best_reduction is None:
            continue
        best_stats = per_patch.get(best_idx, {})
        passed_list = best_stats.get("passed") or []
        pass_rate = (sum(passed_list) / len(passed_list)) if passed_list else 0.0
        rel_vals = best_stats.get("rel") or []
        mean_rel = (sum(rel_vals) / len(rel_vals)) if rel_vals else None
        max_rel = max(rel_vals) if rel_vals else None

        # Prefer explicit labels stored in the validation manifest config. This makes
        # compare.md stable even when run_root/summary.json doesn't exist yet (viz runs
        # before the runner writes it).
        method = None
        cfg = getattr(manifest, "config", None)
        if isinstance(cfg, Mapping):
            vcfg = cfg.get("validation")
            if isinstance(vcfg, Mapping):
                raw = vcfg.get("label")
                if isinstance(raw, str) and raw.strip():
                    method = raw.strip()
        if method is None:
            label = label_by_validation.get(manifest.id, manifest.id)
            method = label
            if isinstance(label, str) and label.startswith("val_"):
                method = label[len("val_") :]

        counts = _counts_from_reduction(best_reduction)
        superreaction_after = (
            superreaction_by_reduction.get(best_reduction)
            if isinstance(best_reduction, str)
            else None
        )
        if superreaction_after is None:
            superreaction_after = superreaction_baseline
        superreaction_ratio = None
        reactions_after = (
            _coerce_optional_int(counts.get("reactions_after"))
            if isinstance(counts, Mapping)
            else None
        )
        if (
            isinstance(superreaction_after, int)
            and isinstance(reactions_after, int)
            and reactions_after > 0
        ):
            superreaction_ratio = float(superreaction_after) / float(reactions_after)
        compare_rows.append(
            {
                "method": method,
                "validation_id": manifest.id,
                "selected_reduction_id": best_reduction,
                "pass_rate": float(pass_rate),
                "mean_rel_diff": float(mean_rel) if mean_rel is not None and math.isfinite(mean_rel) else None,
                "max_rel_diff": float(max_rel) if max_rel is not None and math.isfinite(max_rel) else None,
                "counts": counts,
                "superstates_after": superstate_count,
                "superreactions_after": superreaction_after,
                "superreaction_ratio": superreaction_ratio,
            }
        )

    if compare_rows:
        compare_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "superstate_mapping_id": superstate_mapping_id,
            "superreaction_graph_id": superreaction_graph_id,
            "rows": compare_rows,
        }
        write_json_atomic(reduction_dir / "compare.json", compare_payload)

        # Markdown table for quick reading.
        header = (
            "| method | reduction_id | pass_rate | mean_rel | max_rel | "
            "species(before->after) | reactions(before->after) | disabled | merged_species | "
            "superstates | superreactions | sr/reactions |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        lines = [header]
        for row in sorted(compare_rows, key=lambda r: str(r.get("method"))):
            counts = row.get("counts") if isinstance(row.get("counts"), Mapping) else {}
            sb = counts.get("species_before")
            sa = counts.get("species_after")
            rb = counts.get("reactions_before")
            ra = counts.get("reactions_after")
            disabled = counts.get("disabled_reactions")
            merged_sp = counts.get("merged_species")
            superstates_after = row.get("superstates_after")
            superreactions_after = row.get("superreactions_after")
            sr_ratio = row.get("superreaction_ratio")
            lines.append(
                "| {method} | {rid} | {pass_rate:.3f} | {mean_rel} | {max_rel} | {sb}->{sa} | {rb}->{ra} | {disabled} | {merged_sp} | {ss} | {sr} | {ratio} |\n".format(
                    method=row.get("method"),
                    rid=row.get("selected_reduction_id"),
                    pass_rate=float(row.get("pass_rate") or 0.0),
                    mean_rel=(
                        f"{row.get('mean_rel_diff'):.3g}"
                        if isinstance(row.get("mean_rel_diff"), (int, float))
                        else "n/a"
                    ),
                    max_rel=(
                        f"{row.get('max_rel_diff'):.3g}"
                        if isinstance(row.get("max_rel_diff"), (int, float))
                        else "n/a"
                    ),
                    sb=sb if sb is not None else "n/a",
                    sa=sa if sa is not None else "n/a",
                    rb=rb if rb is not None else "n/a",
                    ra=ra if ra is not None else "n/a",
                    disabled=disabled if disabled is not None else "n/a",
                    merged_sp=merged_sp if merged_sp is not None else "n/a",
                    ss=superstates_after if superstates_after is not None else "n/a",
                    sr=superreactions_after if superreactions_after is not None else "n/a",
                    ratio=(
                        f"{sr_ratio:.3f}"
                        if isinstance(sr_ratio, (int, float))
                        else "n/a"
                    ),
                )
            )
        (reduction_dir / "compare.md").write_text("".join(lines), encoding="utf-8")
        exported.extend(["compare.json", "compare.md"])

    if exported:
        write_json_atomic(reduction_dir / "index.json", {"files": sorted(set(exported))})

def _dot_quote(value: str) -> str:
    return "\"" + value.replace("\"", "\\\"") + "\""


def _truncate_label(text: str, max_len: int = 36) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _extract_bipartite_payload(
    payload: Mapping[str, Any],
    notices: list[str],
) -> Optional[dict[str, Any]]:
    graph_payload: Mapping[str, Any] = payload
    bipartite = payload.get("bipartite")
    if isinstance(bipartite, Mapping):
        data = bipartite.get("data")
        if isinstance(data, Mapping):
            graph_payload = data
    nodes = graph_payload.get("nodes")
    links = graph_payload.get("links") or graph_payload.get("edges")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        notices.append("Graph payload missing nodes for graphviz.")
        return None
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes, bytearray)):
        notices.append("Graph payload missing links for graphviz.")
        return None
    return {"nodes": list(nodes), "links": list(links)}


def _select_bipartite_subgraph(
    nodes: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    *,
    reaction_ids: Optional[Sequence[str]] = None,
    reaction_indices: Optional[Sequence[int]] = None,
    species_names: Optional[Sequence[str]] = None,
    max_nodes: int = 80,
    max_edges: int = 160,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    node_by_id: dict[str, dict[str, Any]] = {}
    reaction_by_id: dict[str, str] = {}
    reaction_by_index: dict[int, str] = {}
    species_by_name: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        node_dict = dict(node)
        node_by_id[node_id] = node_dict
        kind = node_dict.get("kind")
        if kind == "reaction":
            identifiers: list[str] = []
            for key in ("reaction_id", "reaction_equation", "equation", "label", "id"):
                value = node_dict.get(key)
                if isinstance(value, str) and value.strip():
                    identifiers.append(value.strip())
            for identifier in identifiers:
                reaction_by_id.setdefault(identifier, node_id)
            reaction_index = node_dict.get("reaction_index")
            if isinstance(reaction_index, int):
                reaction_by_index[reaction_index] = node_id
        elif kind == "species":
            identifiers = []
            for key in ("label", "species", "name", "id"):
                value = node_dict.get(key)
                if isinstance(value, str) and value.strip():
                    identifiers.append(value.strip())
            for identifier in identifiers:
                species_by_name.setdefault(identifier, node_id)

    selected_ids: set[str] = set()
    if reaction_ids:
        for rid in reaction_ids:
            if not isinstance(rid, str):
                continue
            node_id = reaction_by_id.get(rid)
            if node_id is not None:
                selected_ids.add(node_id)
    if reaction_indices:
        for idx in reaction_indices:
            if not isinstance(idx, int):
                continue
            node_id = reaction_by_index.get(idx)
            if node_id is not None:
                selected_ids.add(node_id)
    if species_names:
        for name in species_names:
            if not isinstance(name, str):
                continue
            node_id = species_by_name.get(name)
            if node_id is not None:
                selected_ids.add(node_id)

    if not selected_ids:
        return [], [], None

    selected_nodes: set[str] = set(selected_ids)
    selected_links: list[dict[str, Any]] = []
    for link in links:
        if not isinstance(link, Mapping):
            continue
        source = link.get("source")
        target = link.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source in selected_ids or target in selected_ids:
            if source in node_by_id and target in node_by_id:
                selected_nodes.add(source)
                selected_nodes.add(target)
                selected_links.append(dict(link))

    if not selected_links:
        return [], [], None

    if len(selected_nodes) > max_nodes:
        degrees: Counter[str] = Counter()
        for link in selected_links:
            source = link.get("source")
            target = link.get("target")
            if isinstance(source, str):
                degrees[source] += 1
            if isinstance(target, str):
                degrees[target] += 1
        reaction_nodes = [
            node_id
            for node_id in selected_nodes
            if node_by_id.get(node_id, {}).get("kind") == "reaction"
        ]
        species_nodes = [
            node_id
            for node_id in selected_nodes
            if node_by_id.get(node_id, {}).get("kind") == "species"
        ]

        def _rank(node_id: str) -> tuple[int, str]:
            return (int(degrees.get(node_id, 0)), node_id)

        reaction_nodes.sort(key=_rank, reverse=True)
        species_nodes.sort(key=_rank, reverse=True)

        if reaction_nodes and species_nodes and max_nodes >= 2:
            # Keep a mix of reaction + species nodes so bipartite edges remain.
            reaction_budget = max(1, int(round(max_nodes * 0.6)))
            reaction_budget = min(reaction_budget, max_nodes - 1, len(reaction_nodes))
            species_budget = max_nodes - reaction_budget
            if species_budget <= 0:
                reaction_budget = min(len(reaction_nodes), max_nodes - 1)
                species_budget = max_nodes - reaction_budget
            kept = reaction_nodes[:reaction_budget] + species_nodes[:species_budget]
        else:
            ranked = list(selected_nodes)
            ranked.sort(key=_rank, reverse=True)
            kept = ranked[:max_nodes]

        selected_nodes = set(kept)
        selected_links = [
            link
            for link in selected_links
            if link.get("source") in selected_nodes
            and link.get("target") in selected_nodes
        ]

    note = None
    if len(selected_links) > max_edges:
        selected_links.sort(
            key=lambda link: abs(_coerce_float(link.get("stoich")) or 0.0),
            reverse=True,
        )
        selected_links = selected_links[:max_edges]
        note = "Graph trimmed for readability."

    selected_node_list = [node_by_id[node_id] for node_id in selected_nodes if node_id in node_by_id]
    return selected_node_list, selected_links, note


def _build_graphviz_bipartite_dot(
    nodes: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    *,
    title: str,
    highlight_reactions: Optional[set[str]] = None,
    highlight_species: Optional[set[str]] = None,
    disabled_reactions: Optional[set[str]] = None,
    note: Optional[str] = None,
    show_edge_labels: bool = False,
) -> str:
    highlight_reactions = highlight_reactions or set()
    highlight_species = highlight_species or set()
    disabled_reactions = disabled_reactions or set()
    label_text = title if not note else f"{title} ({note})"
    dot_lines = [
        "digraph mechanism {",
        "  rankdir=LR;",
        "  splines=true;",
        "  overlap=false;",
        "  nodesep=0.24;",
        "  ranksep=0.42;",
        "  fontname=\"Helvetica\";",
        "  graph [fontsize=10, labelloc=\"t\", labeljust=\"l\"];",
        f"  label={_dot_quote(label_text)};",
        "  node [style=filled, fontname=\"Helvetica\", fontsize=9, penwidth=1.0];",
        "  edge [color=\"#8a8a8a\", arrowsize=0.6, penwidth=0.9];",
        "  subgraph cluster_species {",
        "    label=\"Species\";",
        "    color=\"#dcd6cb\";",
        "    style=\"rounded\";",
        "    rank=source;",
    ]
    for node in nodes:
        if node.get("kind") != "species":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        label = node.get("label") or node_id
        label_str = _truncate_label(str(label))
        fill = "#e8f1ff" if node_id not in highlight_species else "#cfe1ff"
        dot_lines.append(
            f"    {_dot_quote(node_id)} [shape=ellipse, fillcolor=\"{fill}\", label={_dot_quote(label_str)}];"
        )
    dot_lines.extend([
        "  }",
        "  subgraph cluster_reactions {",
        "    label=\"Reactions\";",
        "    color=\"#dcd6cb\";",
        "    style=\"rounded\";",
        "    rank=sink;",
    ])
    for node in nodes:
        if node.get("kind") != "reaction":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        label = (
            node.get("reaction_equation")
            or node.get("equation")
            or node.get("label")
            or node.get("reaction_id")
            or node_id
        )
        label_str = _truncate_label(str(label))
        fill = "#fff1d6" if node_id not in highlight_reactions else "#f8d7da"
        dot_lines.append(
            f"    {_dot_quote(node_id)} [shape=box, fillcolor=\"{fill}\", label={_dot_quote(label_str)}];"
        )
    dot_lines.append("  }")

    for link in links:
        source = link.get("source")
        target = link.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        role = link.get("role")
        color = "#8a8a8a"
        style = None
        if role == "reactant":
            color = "#c1121f"
            arrow_from = source
            arrow_to = target
        elif role == "product":
            color = "#1a7f37"
            arrow_from = target
            arrow_to = source
        else:
            arrow_from = source
            arrow_to = target
        if source in disabled_reactions or target in disabled_reactions:
            style = "dashed"
            color = "#b91c1c"
        attrs = [f"color=\"{color}\""]
        if style:
            attrs.append(f"style=\"{style}\"")
        if show_edge_labels:
            stoich = _coerce_float(link.get("stoich"))
            if stoich is not None:
                coeff = abs(float(stoich))
                if abs(coeff - round(coeff)) < 1.0e-9:
                    coeff_label = str(int(round(coeff)))
                else:
                    coeff_label = f"{coeff:.3g}"
                attrs.append(f"label={_dot_quote(coeff_label)}")
                attrs.append("fontsize=8")
        dot_lines.append(
            f"  {_dot_quote(arrow_from)} -> {_dot_quote(arrow_to)} [{', '.join(attrs)}];"
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


def _build_graphviz_species_graph_dot(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    *,
    title: str,
    note: Optional[str] = None,
    show_edge_labels: bool = False,
) -> str:
    label_text = title if not note else f"{title} ({note})"
    dot_lines = [
        "digraph flux_species {",
        "  overlap=false;",
        "  splines=true;",
        "  nodesep=0.28;",
        "  ranksep=0.36;",
        "  fontname=\"Helvetica\";",
        "  graph [fontsize=10, labelloc=\"t\", labeljust=\"l\"];",
        f"  label={_dot_quote(label_text)};",
        "  node [shape=ellipse, style=filled, fillcolor=\"#e8f1ff\", fontname=\"Helvetica\", fontsize=9, penwidth=1.0];",
        "  edge [color=\"#64748b\", arrowsize=0.55, penwidth=0.9];",
    ]

    for node_id in node_ids:
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        label = _truncate_label(node_id)
        dot_lines.append(
            f"  {_dot_quote(node_id)} [label={_dot_quote(label)}];"
        )

    weights = [abs(float(w)) for _src, _tgt, w in edges if w is not None]
    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 0.0
    span = w_max - w_min
    if span <= 0.0:
        span = 1.0

    for src, tgt, weight in edges:
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        w_abs = abs(float(weight))
        penwidth = 0.8 + 2.8 * (w_abs - w_min) / span
        attrs = [f"penwidth={penwidth:.2f}"]
        if show_edge_labels:
            attrs.append(f"label={_dot_quote(f'{w_abs:.3g}')}")
            attrs.append("fontsize=8")
        dot_lines.append(
            f"  {_dot_quote(src)} -> {_dot_quote(tgt)} [{', '.join(attrs)}];"
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


def _build_graphviz_reaction_projection_dot(
    reaction_nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[tuple[str, str, float]],
    *,
    title: str,
    note: Optional[str] = None,
    show_edge_labels: bool = False,
) -> str:
    label_text = title if not note else f"{title} ({note})"
    dot_lines = [
        "graph reaction_projection {",
        "  overlap=false;",
        "  splines=true;",
        "  nodesep=0.28;",
        "  ranksep=0.34;",
        "  fontname=\"Helvetica\";",
        "  graph [fontsize=10, labelloc=\"t\", labeljust=\"l\"];",
        f"  label={_dot_quote(label_text)};",
        "  node [shape=box, style=filled, fillcolor=\"#fff1d6\", fontname=\"Helvetica\", fontsize=9, penwidth=1.0];",
        "  edge [color=\"#64748b\", penwidth=0.9];",
    ]

    node_ids: list[str] = []
    for node in reaction_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        node_ids.append(node_id)
        label = (
            node.get("reaction_equation")
            or node.get("equation")
            or node.get("label")
            or node.get("reaction_id")
            or node_id
        )
        label_str = _truncate_label(str(label))
        dot_lines.append(f"  {_dot_quote(node_id)} [label={_dot_quote(label_str)}];")

    weights = [abs(float(w)) for _src, _tgt, w in edges if w is not None]
    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 0.0
    span = w_max - w_min
    if span <= 0.0:
        span = 1.0

    node_set = set(node_ids)
    for src, tgt, weight in edges:
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if src not in node_set or tgt not in node_set:
            continue
        w_abs = abs(float(weight))
        penwidth = 0.8 + 2.8 * (w_abs - w_min) / span
        attrs = [f"penwidth={penwidth:.2f}"]
        if show_edge_labels:
            if abs(w_abs - round(w_abs)) < 1.0e-9:
                label_value = str(int(round(w_abs)))
            else:
                label_value = f"{w_abs:.3g}"
            attrs.append(f"label={_dot_quote(label_value)}")
            attrs.append("fontsize=8")
        dot_lines.append(
            f"  {_dot_quote(src)} -- {_dot_quote(tgt)} [{', '.join(attrs)}];"
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


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
        payload = _read_yaml_payload(path)
    except (OSError, ValueError) as exc:
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
        payload = read_json(graph_path)
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


def _load_graph_payload_for_manifests(
    graph_manifests: Sequence[ArtifactManifest],
    store: ArtifactStore,
    notices: list[str],
) -> tuple[
    Optional[ArtifactManifest],
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
    Optional[ArtifactManifest],
    Optional[dict[str, Any]],
]:
    """Load graph.json payloads and select bipartite + temporal-flux inputs for viz.

    Returns:
      - selected bipartite graph manifest + full payload + extracted {nodes, links}
      - selected temporal flux graph manifest + full payload
    """
    if not graph_manifests:
        return None, None, None, None, None

    loaded: list[tuple[ArtifactManifest, dict[str, Any]]] = []
    for manifest in graph_manifests:
        graph_path = store.artifact_dir("graphs", manifest.id) / "graph.json"
        if not graph_path.exists():
            notices.append(f"graph.json missing for graphs/{manifest.id}.")
            continue
        try:
            payload = read_json(graph_path)
        except Exception as exc:
            notices.append(f"graph.json invalid for graphs/{manifest.id}: {exc}")
            continue
        if not isinstance(payload, Mapping):
            notices.append(f"graph.json payload invalid for graphs/{manifest.id}.")
            continue
        loaded.append((manifest, dict(payload)))

    if not loaded:
        return None, None, None, None, None

    def _has_bipartite(payload: Mapping[str, Any]) -> bool:
        bipartite = payload.get("bipartite")
        if isinstance(bipartite, Mapping) and isinstance(bipartite.get("data"), Mapping):
            return True
        return bool(
            "nodes" in payload
            and ("links" in payload or "edges" in payload)
        )

    selected_bip_manifest: Optional[ArtifactManifest] = None
    selected_bip_payload: Optional[dict[str, Any]] = None
    selected_bip_data: Optional[dict[str, Any]] = None

    selected_flux_manifest: Optional[ArtifactManifest] = None
    selected_flux_payload: Optional[dict[str, Any]] = None

    for manifest, payload in loaded:
        kind = payload.get("kind")
        if (
            selected_flux_manifest is None
            and isinstance(kind, str)
            and kind == "temporal_flux"
        ):
            selected_flux_manifest = manifest
            selected_flux_payload = payload
        if (
            selected_bip_manifest is None
            and isinstance(kind, str)
            and kind == "stoichiometric_matrix"
            and _has_bipartite(payload)
        ):
            selected_bip_manifest = manifest
            selected_bip_payload = payload
            selected_bip_data = _extract_bipartite_payload(payload, notices)

    if selected_bip_manifest is None:
        for manifest, payload in loaded:
            if not _has_bipartite(payload):
                continue
            extracted = _extract_bipartite_payload(payload, notices)
            if extracted is None:
                continue
            selected_bip_manifest = manifest
            selected_bip_payload = payload
            selected_bip_data = extracted
            break

    if selected_bip_manifest is None and graph_manifests:
        notices.append("No bipartite graph payload available for networks.")

    return (
        selected_bip_manifest,
        selected_bip_payload,
        selected_bip_data,
        selected_flux_manifest,
        selected_flux_payload,
    )


def _select_run_and_payload(
    run_manifests: Sequence[ArtifactManifest],
    store: ArtifactStore,
    notices: list[str],
    *,
    run_id: Optional[Any] = None,
    multiple_notice: Optional[str] = None,
) -> tuple[Optional[ArtifactManifest], Optional[dict[str, Any]]]:
    selected_run: Optional[ArtifactManifest] = None
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
        if len(run_manifests) > 1 and multiple_notice:
            notices.append(multiple_notice)
        selected_run = run_manifests[0]

    run_payload: Optional[dict[str, Any]] = None
    if selected_run is not None:
        run_dir = store.artifact_dir("runs", selected_run.id)
        try:
            run_payload = load_run_dataset_payload(run_dir)
        except Exception as exc:
            notices.append(f"Failed to load run dataset {selected_run.id}: {exc}")
            run_payload = None
    return selected_run, run_payload


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

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
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

    result = store.ensure(report_manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_report_from_artifact(result.path, run_root)
    return result


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

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
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
    graph_manifests = [m for m in input_manifests if m.kind == "graphs"]

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

    chart_backend, plotly_ctx, mpl_ctx = _resolve_chart_backend(viz_cfg, notices)
    plotly_state = {"include_js": True}
    use_plotly = chart_backend == "plotly" and plotly_ctx is not None
    use_mpl = chart_backend == "matplotlib" and mpl_ctx is not None
    use_svg = chart_backend == "svg"
    if use_plotly:
        go, pio = plotly_ctx
        export_state = _init_export_state(viz_cfg, plotly_ctx, notices)
    elif use_mpl:
        export_state = _init_matplotlib_export_state(viz_cfg, notices)
    elif use_svg:
        export_state = _init_svg_export_state(viz_cfg, notices)
    else:
        export_state = None

    selected_run, run_payload = _select_run_and_payload(
        run_manifests,
        store,
        notices,
        multiple_notice="Multiple runs provided; using the first for charts.",
    )

    species_series_var: Optional[str] = None
    species_axis_name: Optional[str] = None
    rate_series_var: Optional[str] = None
    rate_axis_name: Optional[str] = None
    rate_selected_names: list[str] = []

    (
        _selected_graph_manifest,
        graph_payload,
        bipartite_payload,
        _selected_flux_manifest,
        _flux_payload,
    ) = _load_graph_payload_for_manifests(graph_manifests, store, notices)

    # Reuse the selected run/payload for ranking charts.

    condition_cards: list[str] = []
    if condition_fields and run_manifests:
        for spec in condition_fields:
            label = str(spec["label"])
            values = condition_values.get(label, [])
            if not values:
                card_html = f"<p class=\"muted\">No values for {html.escape(label)}.</p>"
                condition_cards.append(_wrap_card(card_html))
                continue
            numeric_values = [
                number
                for value in values
                if (number := _coerce_float(value)) is not None
            ]
            if use_plotly:
                fig = go.Figure()
                fig.add_histogram(x=values, nbinsx=20)
                fig.update_layout(
                    title=label,
                    template="plotly_white",
                    height=260,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                condition_cards.append(
                    _wrap_card(
                        _plotly_html_with_export(
                            fig,
                            pio,
                            plotly_state,
                            export_state,
                            f"condition_{label}",
                        )
                    )
                )
            elif use_mpl:
                if not numeric_values:
                    summary = _summarize_values(values)
                    condition_cards.append(
                        _wrap_card(
                            f"<strong>{html.escape(label)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                        )
                    )
                    continue
                fig, ax = mpl_ctx.subplots(figsize=(5.2, 3.0))
                ax.hist(numeric_values, bins=20, color="#0f6f68", alpha=0.85)
                ax.set_title(label)
                ax.set_xlabel(label)
                ax.set_ylabel("count")
                fig.tight_layout()
                condition_cards.append(
                    _wrap_card(
                        _matplotlib_html_with_export(
                            fig,
                            export_state,
                            f"condition_{label}",
                            alt=label,
                            notices=notices,
                        )
                    )
                )
            elif use_svg:
                if not numeric_values:
                    summary = _summarize_values(values)
                    condition_cards.append(
                        _wrap_card(
                            f"<strong>{html.escape(label)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                        )
                    )
                    continue
                chart_html = _build_svg_histogram_chart(
                    title=label,
                    values=numeric_values,
                )
                condition_cards.append(
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            f"condition_{label}",
                            notices,
                        )
                    )
                )
            else:
                summary = _summarize_values(values)
                condition_cards.append(
                    _wrap_card(
                        f"<strong>{html.escape(label)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                    )
                )
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
            hist_title = name if not unit_label else f"{name} ({unit_label})"
            if use_plotly:
                fig = go.Figure()
                fig.add_histogram(x=values, nbinsx=20)
                fig.update_layout(
                    title=f"Objective Distribution: {hist_title}",
                    template="plotly_white",
                    height=260,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                objective_cards.append(
                    _wrap_card(
                        _plotly_html_with_export(
                            fig,
                            pio,
                            plotly_state,
                            export_state,
                            f"objective_dist_{name}",
                        )
                    )
                )
            elif use_mpl:
                fig, ax = mpl_ctx.subplots(figsize=(5.2, 3.0))
                ax.hist(values, bins=20, color="#0f6f68", alpha=0.85)
                ax.set_title(f"Objective Distribution: {hist_title}")
                ax.set_xlabel(hist_title)
                ax.set_ylabel("count")
                fig.tight_layout()
                objective_cards.append(
                    _wrap_card(
                        _matplotlib_html_with_export(
                            fig,
                            export_state,
                            f"objective_dist_{name}",
                            alt=f"Objective Distribution: {hist_title}",
                            notices=notices,
                        )
                    )
                )
            elif use_svg:
                chart_html = _build_svg_histogram_chart(
                    title=f"Objective Distribution: {hist_title}",
                    values=values,
                )
                objective_cards.append(
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            f"objective_dist_{name}",
                            notices,
                        )
                    )
                )
            else:
                summary = _summarize_values(values)
                objective_cards.append(
                    _wrap_card(
                        f"<strong>{html.escape(name)}</strong><p class=\"muted\">{html.escape(summary)}</p>"
                    )
                )
                continue

            if scatter_label and scatter_by_run:
                scatter_x: list[float] = []
                scatter_y: list[float] = []
                for run_id, value in run_entries:
                    x_val = scatter_by_run.get(run_id)
                    x_num = _coerce_float(x_val)
                    if x_num is None:
                        continue
                    scatter_x.append(x_num)
                    scatter_y.append(value)
                if scatter_x and scatter_y:
                    if use_plotly:
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
                            _wrap_card(
                                _plotly_html_with_export(
                                    fig,
                                    pio,
                                    plotly_state,
                                    export_state,
                                    f"objective_scatter_{name}",
                                )
                            )
                        )
                    elif use_mpl:
                        fig, ax = mpl_ctx.subplots(figsize=(5.2, 3.0))
                        ax.scatter(scatter_x, scatter_y, s=28, color="#0f6f68", alpha=0.75)
                        ax.set_title(f"{name} vs {scatter_label}")
                        ax.set_xlabel(scatter_label)
                        ax.set_ylabel(hist_title)
                        fig.tight_layout()
                        objective_cards.append(
                            _wrap_card(
                                _matplotlib_html_with_export(
                                    fig,
                                    export_state,
                                    f"objective_scatter_{name}",
                                    alt=f"{name} vs {scatter_label}",
                                    notices=notices,
                                )
                            )
                        )
                    elif use_svg:
                        chart_html = _build_svg_scatter_chart(
                            title=f"{name} vs {scatter_label}",
                            x_values=scatter_x,
                            y_values=scatter_y,
                            x_label=scatter_label,
                            y_label=hist_title,
                        )
                        objective_cards.append(
                            _wrap_card(
                                _svg_html_with_export(
                                    chart_html,
                                    export_state,
                                    f"objective_scatter_{name}",
                                    notices,
                                )
                            )
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
            if use_plotly:
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
                sensitivity_body = _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        "sensitivity_heatmap",
                    )
                )
            elif use_mpl:
                title_suffix = f" (condition_id={condition_id})" if condition_id else ""
                flat_values = [
                    abs(value)
                    for row in z_values
                    for value in row
                    if value is not None
                ]
                max_abs = max(flat_values) if flat_values else 1.0
                display_values = [
                    [value if value is not None else 0.0 for value in row]
                    for row in z_values
                ]
                fig, ax = mpl_ctx.subplots(
                    figsize=(6.0, 2.6 + 0.2 * len(reaction_order))
                )
                image = ax.imshow(
                    display_values,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-max_abs,
                    vmax=max_abs,
                )
                ax.set_title(f"Sensitivity Heatmap{title_suffix}")
                ax.set_xticks(list(range(len(sensitivity_targets))))
                ax.set_xticklabels(sensitivity_targets, rotation=45, ha="right")
                ax.set_yticks(list(range(len(reaction_order))))
                ax.set_yticklabels(reaction_order)
                fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
                fig.tight_layout()
                sensitivity_body = _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        "sensitivity_heatmap",
                        alt="Sensitivity Heatmap",
                        notices=notices,
                    )
                )
            elif use_svg:
                title_suffix = f" (condition_id={condition_id})" if condition_id else ""
                chart_html = _build_svg_heatmap_chart(
                    title=f"Sensitivity Heatmap{title_suffix}",
                    z_values=z_values,
                    x_labels=sensitivity_targets,
                    y_labels=reaction_order,
                )
                sensitivity_body = _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        "sensitivity_heatmap",
                        notices,
                    )
                )
            else:
                summary_lines = [
                    f"{target}: {len(values_by_target.get(target, []))} reactions"
                    for target in sensitivity_targets
                ]
                sensitivity_body = _render_message_list(
                    summary_lines,
                    "No sensitivity summary available.",
                )
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

    image_export_section = ""
    if export_state is not None and export_state["planned"]:
        image_export_section = _panel(
            "Image Exports",
            _render_message_list(
                export_state["planned"],
                "No image exports available.",
            ),
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
        condition_section
        + objective_section
        + sensitivity_section
        + image_export_section
        + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        if use_plotly:
            _export_queued_plotly(export_state, base_dir, notices)
        elif use_mpl:
            _export_queued_matplotlib(export_state, base_dir, notices)
        elif use_svg:
            _export_queued_svg(export_state, base_dir, notices)
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")
    result = store.ensure(report_manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_report_from_artifact(result.path, run_root)
    return result


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

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
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
    reaction_count = _infer_reaction_count(viz_cfg, graph_manifests, store, notices)

    chart_backend, plotly_ctx, mpl_ctx = _resolve_chart_backend(viz_cfg, notices)
    plotly_state = {"include_js": True}
    use_plotly = chart_backend == "plotly" and plotly_ctx is not None
    use_mpl = chart_backend == "matplotlib" and mpl_ctx is not None
    use_svg = chart_backend == "svg"
    if use_plotly:
        go, pio = plotly_ctx
        export_state = _init_export_state(viz_cfg, plotly_ctx, notices)
    elif use_mpl:
        export_state = _init_matplotlib_export_state(viz_cfg, notices)
    elif use_svg:
        export_state = _init_svg_export_state(viz_cfg, notices)
    else:
        export_state = None

    selected_run, run_payload = _select_run_and_payload(
        run_manifests,
        store,
        notices,
        run_id=viz_cfg.get("run_id") or viz_cfg.get("run"),
        multiple_notice="Multiple runs provided; using the first run for plots.",
    )

    species_series_var: Optional[str] = None
    species_axis_name: Optional[str] = None
    rate_series_var: Optional[str] = None
    rate_axis_name: Optional[str] = None
    rate_selected_names: list[str] = []

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
                        species_series_var = series_var
                        species_axis_name = axis_name
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
                            elif use_plotly:
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
                                    _plotly_html_with_export(
                                        fig,
                                        pio,
                                        plotly_state,
                                        export_state,
                                        f"species_timeseries_{series_var}",
                                    )
                                )
                            elif use_mpl:
                                indices = _downsample_indices(
                                    len(time_values),
                                    species_max_points,
                                )
                                times = [time_values[i] for i in indices]
                                series_map: dict[str, list[float]] = {}
                                for name in selected_species:
                                    idx = axis_index.get(name)
                                    if idx is None:
                                        continue
                                    series_map[name] = [matrix[i][idx] for i in indices]
                                fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.2))
                                for name, series in series_map.items():
                                    ax.plot(times, series, label=name)
                                ax.set_title("Species Time Series")
                                ax.set_xlabel("Time" + (f" ({time_unit})" if time_unit else ""))
                                ax.set_ylabel(unit_label or series_var)
                                ax.legend(fontsize=8, ncol=2)
                                fig.tight_layout()
                                species_body = _wrap_card(
                                    _matplotlib_html_with_export(
                                        fig,
                                        export_state,
                                        f"species_timeseries_{series_var}",
                                        alt="Species Time Series",
                                        notices=notices,
                                    )
                                )
                            elif use_svg:
                                indices = _downsample_indices(
                                    len(time_values),
                                    species_max_points,
                                )
                                times = [time_values[i] for i in indices]
                                series_map: dict[str, list[float]] = {}
                                for name in selected_species:
                                    idx = axis_index.get(name)
                                    if idx is None:
                                        continue
                                    series_map[name] = [matrix[i][idx] for i in indices]
                                chart_html = _build_svg_line_chart(
                                    title="Species Time Series",
                                    times=times,
                                    series=series_map,
                                    unit=unit_label or None,
                                )
                                species_body = _wrap_card(
                                    _svg_html_with_export(
                                        chart_html,
                                        export_state,
                                        f"species_timeseries_{series_var}",
                                        notices,
                                    )
                                )
                            else:
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
                    rate_selected_names = list(selected_rate_species)
                    rate_series_var = rate_var
                    rate_axis_name = "species"
                    if use_plotly:
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
                            _plotly_html_with_export(
                                fig,
                                pio,
                                plotly_state,
                                export_state,
                                f"net_production_{rate_var}",
                            )
                        )
                    elif use_mpl:
                        axis_index = {
                            name: idx for idx, name in enumerate(axis_values)
                        }
                        indices = _downsample_indices(
                            len(time_values),
                            rate_max_points,
                        )
                        times = [time_values[i] for i in indices]
                        units = run_payload.get("attrs", {}).get("units", {})
                        unit_label = ""
                        time_unit = ""
                        if isinstance(units, Mapping):
                            unit_label = str(units.get(rate_var, "")) if units else ""
                            time_unit = str(units.get("time", "")) if units else ""
                        fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.2))
                        for name in selected_rate_species:
                            idx = axis_index.get(name)
                            if idx is None:
                                continue
                            series = [matrix[i][idx] for i in indices]
                            ax.plot(times, series, label=name)
                        ax.set_title("Net Production Rates")
                        ax.set_xlabel("Time" + (f" ({time_unit})" if time_unit else ""))
                        ax.set_ylabel(unit_label or rate_var)
                        ax.legend(fontsize=8, ncol=2)
                        fig.tight_layout()
                        rate_body = _wrap_card(
                            _matplotlib_html_with_export(
                                fig,
                                export_state,
                                f"net_production_{rate_var}",
                                alt="Net Production Rates",
                                notices=notices,
                            )
                        )
                    elif use_svg:
                        axis_index = {
                            name: idx for idx, name in enumerate(axis_values)
                        }
                        indices = _downsample_indices(
                            len(time_values),
                            rate_max_points,
                        )
                        times = [time_values[i] for i in indices]
                        units = run_payload.get("attrs", {}).get("units", {})
                        unit_label = ""
                        if isinstance(units, Mapping):
                            unit_label = str(units.get(rate_var, "")) if units else ""
                        series_map: dict[str, list[float]] = {}
                        for name in selected_rate_species:
                            idx = axis_index.get(name)
                            if idx is None:
                                continue
                            series_map[name] = [matrix[i][idx] for i in indices]
                        chart_html = _build_svg_line_chart(
                            title="Net Production Rates",
                            times=times,
                            series=series_map,
                            unit=unit_label or None,
                        )
                        rate_body = _wrap_card(
                            _svg_html_with_export(
                                chart_html,
                                export_state,
                                f"net_production_{rate_var}",
                                notices,
                            )
                        )
                    else:
                        summary = [
                            f"{name}: {rate_rank_by}={value:.4g}"
                            for name, value in wdot_ranked[:rate_top_n]
                        ]
                        rate_body = _render_message_list(
                            summary,
                            "No production rate summary available.",
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
        if use_plotly:
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
            rop_card = _plotly_html_with_export(
                fig,
                pio,
                plotly_state,
                export_state,
                "rop_ranking",
            )
        elif use_mpl:
            labels = [name for name, _ in rop_ranked]
            values = [value for _, value in rop_ranked]
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 2.4 + 0.12 * len(labels)))
            ax.barh(labels, values, color="#0f6f68")
            ax.set_title("Top ROP Reactions")
            ax.set_xlabel("value")
            ax.invert_yaxis()
            fig.tight_layout()
            rop_card = _matplotlib_html_with_export(
                fig,
                export_state,
                "rop_ranking",
                alt="Top ROP Reactions",
                notices=notices,
            )
        elif use_svg:
            labels = [name for name, _ in rop_ranked]
            values = [value for _, value in rop_ranked]
            chart_html = _build_svg_barh_chart(
                title="Top ROP Reactions",
                labels=labels,
                values=values,
            )
            rop_card = _svg_html_with_export(
                chart_html,
                export_state,
                "rop_ranking",
                notices,
            )
        else:
            summary = [
                f"{name}: {value:.4g}" for name, value in rop_ranked
            ]
            rop_card = _render_message_list(
                summary,
                "No ROP summary available.",
            )

    wdot_card = "<p class=\"muted\">No wdot data available.</p>"
    if wdot_ranked_from_features:
        if use_plotly:
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
            wdot_card = _plotly_html_with_export(
                fig,
                pio,
                plotly_state,
                export_state,
                "wdot_ranking",
            )
        elif use_mpl:
            labels = [name for name, _ in wdot_ranked_from_features]
            values = [value for _, value in wdot_ranked_from_features]
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 2.4 + 0.12 * len(labels)))
            ax.barh(labels, values, color="#7a3e2f")
            ax.set_title("Top Net Production Species")
            ax.set_xlabel("value")
            ax.invert_yaxis()
            fig.tight_layout()
            wdot_card = _matplotlib_html_with_export(
                fig,
                export_state,
                "wdot_ranking",
                alt="Top Net Production Species",
                notices=notices,
            )
        elif use_svg:
            labels = [name for name, _ in wdot_ranked_from_features]
            values = [value for _, value in wdot_ranked_from_features]
            chart_html = _build_svg_barh_chart(
                title="Top Net Production Species",
                labels=labels,
                values=values,
            )
            wdot_card = _svg_html_with_export(
                chart_html,
                export_state,
                "wdot_ranking",
                notices,
            )
        else:
            summary = [
                f"{name}: {value:.4g}" for name, value in wdot_ranked_from_features
            ]
            wdot_card = _render_message_list(
                summary,
                "No wdot summary available.",
            )

    rop_section = _panel(
        "Rate-of-Production Ranking",
        "<div class=\"grid\">"
        + _wrap_card(rop_card)
        + _wrap_card(wdot_card)
        + "</div>",
    )

    feature_overview_cfg = viz_cfg.get("feature_overview", {})
    if feature_overview_cfg is None:
        feature_overview_cfg = {}
    if not isinstance(feature_overview_cfg, Mapping):
        raise ConfigError("viz.feature_overview must be a mapping when provided.")
    feature_top_n = _coerce_positive_int(
        feature_overview_cfg.get("top_n"),
        "viz.feature_overview.top_n",
        default=12,
    )
    feature_body = "<p class=\"muted\">No feature data available.</p>"
    if feature_rows:
        values_by_feature: dict[str, list[float]] = {}
        units_by_feature: dict[str, list[str]] = {}
        for row in feature_rows:
            name = row.get("feature")
            if not isinstance(name, str) or not name.strip():
                continue
            value = _coerce_float(row.get("value"))
            if value is None:
                continue
            values_by_feature.setdefault(name, []).append(value)
            unit = row.get("unit")
            if isinstance(unit, str) and unit.strip():
                units_by_feature.setdefault(name, []).append(unit)
        ranked = []
        for name, values in values_by_feature.items():
            mean_value = sum(values) / len(values)
            ranked.append((name, mean_value, len(values)))
        ranked.sort(key=lambda item: abs(item[1]), reverse=True)
        ranked = ranked[:feature_top_n]
        if ranked:
            if use_plotly:
                fig = go.Figure(
                    data=go.Bar(
                        x=[value for _, value, _ in ranked],
                        y=[name for name, _, _ in ranked],
                        orientation="h",
                        marker=dict(color="#0f6f68"),
                    )
                )
                fig.update_layout(
                    title="Feature Mean Values (Top)",
                    template="plotly_white",
                    height=260 + 14 * len(ranked),
                    margin=dict(l=80, r=20, t=50, b=40),
                )
                feature_body = _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        "feature_overview",
                    )
                )
            elif use_mpl:
                labels = [name for name, _, _ in ranked]
                values = [value for _, value, _ in ranked]
                fig, ax = mpl_ctx.subplots(figsize=(6.0, 2.4 + 0.12 * len(labels)))
                ax.barh(labels, values, color="#0f6f68")
                ax.set_title("Feature Mean Values (Top)")
                ax.set_xlabel("mean value")
                ax.invert_yaxis()
                fig.tight_layout()
                feature_body = _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        "feature_overview",
                        alt="Feature Mean Values (Top)",
                        notices=notices,
                    )
                )
            elif use_svg:
                labels = [name for name, _, _ in ranked]
                values = [value for _, value, _ in ranked]
                chart_html = _build_svg_barh_chart(
                    title="Feature Mean Values (Top)",
                    labels=labels,
                    values=values,
                )
                feature_body = _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        "feature_overview",
                        notices,
                    )
                )
            else:
                feature_lines = [
                    f"{name}: {value:.4g} (n={count})"
                    for name, value, count in ranked
                ]
                feature_body = _render_message_list(
                    feature_lines,
                    "No feature summary available.",
                )
    feature_section = _panel("Feature Overview", feature_body)

    network_cfg = viz_cfg.get("network", {})
    if network_cfg is None:
        network_cfg = {}
    if not isinstance(network_cfg, Mapping):
        raise ConfigError("viz.network must be a mapping when provided.")
    network_enabled = network_cfg.get("enabled")
    if network_enabled is None:
        network_enabled = False
    if not isinstance(network_enabled, bool):
        raise ConfigError("viz.network.enabled must be a boolean.")
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
    selected_graph_manifest: Optional[ArtifactManifest] = None
    bipartite_payload: Optional[dict[str, Any]] = None
    selected_flux_manifest: Optional[ArtifactManifest] = None
    flux_payload: Optional[dict[str, Any]] = None

    (
        selected_graph_manifest,
        selected_graph_payload,
        selected_bipartite_payload,
        selected_flux_manifest,
        flux_payload,
    ) = _load_graph_payload_for_manifests(graph_manifests, store, notices)
    graph_payload = selected_graph_payload
    bipartite_payload = selected_bipartite_payload

    graph_id = network_cfg.get("graph_id")
    if graph_id is not None:
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise ConfigError("viz.network.graph_id must be a non-empty string.")
        requested = None
        for manifest in graph_manifests:
            if manifest.id == graph_id:
                requested = manifest
                break
        if requested is None:
            notices.append(f"Requested graph_id {graph_id} not found in inputs.")
        else:
            graph_path = store.artifact_dir("graphs", requested.id) / "graph.json"
            try:
                payload = read_json(graph_path)
                if isinstance(payload, Mapping):
                    graph_payload = payload
                    bipartite_payload = _extract_bipartite_payload(payload, notices)
                    selected_graph_manifest = requested
            except Exception as exc:
                notices.append(f"Failed to read graph/{requested.id}: {exc}")
                graph_payload = None
                bipartite_payload = None

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

                    if not (use_plotly or use_mpl or use_svg):
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

                        node_labels: dict[str, str] = {}
                        for node_id in selected_ids:
                            node = node_map.get(node_id, {})
                            label = (
                                node.get("label")
                                or node.get("reaction_id")
                                or node.get("species")
                                or node.get("id")
                                or node_id
                            )
                            node_labels[node_id] = str(label)

                        if use_plotly:
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
                                node_text.append(node_labels.get(node_id, node_id))

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
                                _plotly_html_with_export(
                                    fig,
                                    pio,
                                    plotly_state,
                                    export_state,
                                    "network_subgraph",
                                )
                            )
                        elif use_mpl:
                            fig, ax = mpl_ctx.subplots(figsize=(5.6, 3.6))
                            for source_id, target_id in edge_list:
                                if source_id not in positions or target_id not in positions:
                                    continue
                                x0, y0 = positions[source_id]
                                x1, y1 = positions[target_id]
                                ax.plot([x0, x1], [y0, y1], color="#9bb3b0", linewidth=1)
                            node_x = [positions[node_id][0] for node_id in selected_ids]
                            node_y = [positions[node_id][1] for node_id in selected_ids]
                            ax.scatter(node_x, node_y, s=60, color="#0f6f68", edgecolors="#ffffff")
                            for node_id in selected_ids:
                                x_val, y_val = positions[node_id]
                                ax.text(
                                    x_val,
                                    y_val + 0.04,
                                    node_labels.get(node_id, node_id),
                                    fontsize=8,
                                    ha="center",
                                    color="#5f6c77",
                                )
                            ax.set_title("Reaction Network Subgraph")
                            ax.axis("off")
                            fig.tight_layout()
                            network_body = _wrap_card(
                                _matplotlib_html_with_export(
                                    fig,
                                    export_state,
                                    "network_subgraph",
                                    alt="Reaction Network Subgraph",
                                    notices=notices,
                                )
                            )
                        else:
                            chart_html = _build_svg_network_chart(
                                title="Reaction Network Subgraph",
                                positions=positions,
                                edges=edge_list,
                                labels=node_labels,
                            )
                            network_body = _wrap_card(
                                _svg_html_with_export(
                                    chart_html,
                                    export_state,
                                    "network_subgraph",
                                    notices,
                                )
                            )

    network_section = (
        _panel("Reaction Network Subgraph", network_body) if network_enabled else ""
    )

    graphviz_cards: list[str] = []
    graphviz_cfg = viz_cfg.get("graphviz", {})
    if graphviz_cfg is None:
        graphviz_cfg = {}
    if not isinstance(graphviz_cfg, Mapping):
        raise ConfigError("viz.graphviz must be a mapping when provided.")
    gv_top_n = _coerce_positive_int(
        graphviz_cfg.get("top_n"), "viz.graphviz.top_n", default=8
    )
    gv_max_nodes = _coerce_positive_int(
        graphviz_cfg.get("max_nodes"), "viz.graphviz.max_nodes", default=80
    )
    gv_max_edges = _coerce_positive_int(
        graphviz_cfg.get("max_edges"), "viz.graphviz.max_edges", default=160
    )
    gv_engine = graphviz_cfg.get("engine", "dot")
    if not isinstance(gv_engine, str) or not gv_engine.strip():
        raise ConfigError("viz.graphviz.engine must be a non-empty string.")
    gv_engine = gv_engine.strip()

    bipartite_payload = None
    if graph_payload is not None:
        bipartite_payload = _extract_bipartite_payload(graph_payload, notices)
    if bipartite_payload is not None:
        graph_nodes = bipartite_payload.get("nodes", [])
        graph_links = bipartite_payload.get("links", [])
        if rop_ranked:
            reaction_ids = [name for name, _ in rop_ranked[:gv_top_n]]
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                reaction_ids=reaction_ids,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Top ROP Reaction Network",
                highlight_reactions={
                    node.get("id")
                    for node in nodes
                    if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
                },
                note=note,
            )
            graphviz_cards.append(
                _wrap_card(
                    _graphviz_html_with_export(
                        dot_source,
                        export_state,
                        "chem_graphviz_rop",
                        notices,
                        engine=gv_engine,
                    )
                )
            )
        if wdot_ranked_from_features:
            species_names = [name for name, _ in wdot_ranked_from_features[:gv_top_n]]
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                species_names=species_names,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            dot_source = _build_graphviz_bipartite_dot(
                nodes,
                links,
                title="Top WDOT Species Network",
                highlight_species={
                    node.get("id")
                    for node in nodes
                    if node.get("kind") == "species" and isinstance(node.get("id"), str)
                },
                note=note,
            )
            graphviz_cards.append(
                _wrap_card(
                    _graphviz_html_with_export(
                        dot_source,
                        export_state,
                        "chem_graphviz_wdot",
                        notices,
                        engine=gv_engine,
                    )
                )
            )
        if reduction_inputs:
            for entry in reduction_inputs[:2]:
                reduction_id = entry["id"]
                patch_path = (
                    store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
                )
                patch_payload, error = _read_patch_payload(patch_path)
                if error is not None or patch_payload is None:
                    notices.append(f"Reduction patch {reduction_id}: {error}")
                    continue
                reaction_ids: list[str] = []
                reaction_indices: list[int] = []
                for key in ("disabled_reactions", "reaction_multipliers"):
                    entries = patch_payload.get(key)
                    if isinstance(entries, Mapping):
                        entries = [entries]
                    if not isinstance(entries, Sequence) or isinstance(
                        entries, (str, bytes, bytearray)
                    ):
                        continue
                    for item in entries:
                        if not isinstance(item, Mapping):
                            continue
                        reaction_id = item.get("reaction_id") or item.get("reaction")
                        if isinstance(reaction_id, str) and reaction_id.strip():
                            reaction_ids.append(reaction_id)
                            continue
                        idx = _coerce_optional_int(item.get("index"))
                        if idx is not None:
                            reaction_indices.append(idx)
                nodes, links, note = _select_bipartite_subgraph(
                    graph_nodes,
                    graph_links,
                    reaction_ids=reaction_ids,
                    reaction_indices=reaction_indices,
                    max_nodes=gv_max_nodes,
                    max_edges=gv_max_edges,
                )
                dot_source = _build_graphviz_bipartite_dot(
                    nodes,
                    links,
                    title=f"Reduction Patch Network ({reduction_id})",
                    highlight_reactions={
                        node.get("id")
                        for node in nodes
                        if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
                    },
                    note=note,
                )
                graphviz_cards.append(
                    _wrap_card(
                        _graphviz_html_with_export(
                            dot_source,
                            export_state,
                            f"chem_graphviz_reduction_{reduction_id}",
                            notices,
                            engine=gv_engine,
                        )
                    )
                )

    graphviz_section = _panel(
        "Mechanism Networks (Graphviz)",
        "<div class=\"grid\">" + "".join(graphviz_cards) + "</div>"
        if graphviz_cards
        else "<p class=\"muted\">No graphviz networks available.</p>",
    )

    reduction_body = "<p class=\"muted\">No reduction data available.</p>"
    if reduction_inputs:
        reduction_rows: list[list[str]] = []
        reduction_counts: list[tuple[str, int]] = []
        for entry in reduction_inputs:
            reduction_id = entry["id"]
            patch_path = (
                store.artifact_dir("reduction", reduction_id) / "mechanism_patch.yaml"
            )
            patch_payload, error = _read_patch_payload(patch_path)
            if error is not None or patch_payload is None:
                notices.append(f"Reduction patch {reduction_id}: {error}")
                reduction_rows.append([reduction_id, "n/a", "n/a", "error"])
                continue
            disabled = patch_payload.get("disabled_reactions", [])
            multipliers = patch_payload.get("reaction_multipliers", [])
            if isinstance(disabled, Mapping):
                disabled_count = len(disabled)
            elif isinstance(disabled, Sequence) and not isinstance(
                disabled, (str, bytes, bytearray)
            ):
                disabled_count = len(disabled)
            else:
                disabled_count = 0
            if isinstance(multipliers, Mapping):
                multiplier_count = len(multipliers)
            elif isinstance(multipliers, Sequence) and not isinstance(
                multipliers, (str, bytes, bytearray)
            ):
                multiplier_count = len(multipliers)
            else:
                multiplier_count = 0
            reduction_counts.append((reduction_id, disabled_count))
            rate_text = "n/a"
            if reaction_count:
                rate_text = _format_percent(disabled_count / float(reaction_count))
            reduction_rows.append(
                [
                    reduction_id,
                    str(disabled_count),
                    str(multiplier_count),
                    rate_text,
                ]
            )
        reduction_table = _render_table(
            ["Artifact", "Disabled", "Multipliers", "Reduction rate"],
            reduction_rows,
            "No reduction summaries available.",
        )
        reduction_body = _wrap_card(reduction_table)
        if reduction_counts:
            if use_plotly:
                fig = go.Figure(
                    data=go.Bar(
                        x=[count for _, count in reduction_counts],
                        y=[rid for rid, _ in reduction_counts],
                        orientation="h",
                        marker=dict(color="#d17b0f"),
                    )
                )
                fig.update_layout(
                    title="Disabled Reactions by Reduction",
                    template="plotly_white",
                    height=220 + 14 * len(reduction_counts),
                    margin=dict(l=80, r=20, t=50, b=40),
                )
                reduction_body = (
                    _wrap_card(
                        _plotly_html_with_export(
                            fig,
                            pio,
                            plotly_state,
                            export_state,
                            "reduction_disabled",
                        )
                    )
                    + reduction_body
                )
            elif use_mpl:
                labels = [rid for rid, _ in reduction_counts]
                values = [count for _, count in reduction_counts]
                fig, ax = mpl_ctx.subplots(figsize=(6.0, 2.0 + 0.12 * len(labels)))
                ax.barh(labels, values, color="#d17b0f")
                ax.set_title("Disabled Reactions by Reduction")
                ax.set_xlabel("count")
                ax.invert_yaxis()
                fig.tight_layout()
                reduction_body = (
                    _wrap_card(
                        _matplotlib_html_with_export(
                            fig,
                            export_state,
                            "reduction_disabled",
                            alt="Disabled Reactions by Reduction",
                            notices=notices,
                        )
                    )
                    + reduction_body
                )
            elif use_svg:
                labels = [rid for rid, _ in reduction_counts]
                values = [count for _, count in reduction_counts]
                chart_html = _build_svg_barh_chart(
                    title="Disabled Reactions by Reduction",
                    labels=labels,
                    values=values,
                )
                reduction_body = (
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            "reduction_disabled",
                            notices,
                        )
                    )
                    + reduction_body
                )
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

    image_export_section = ""
    if export_state is not None and export_state["planned"]:
        image_export_section = _panel(
            "Image Exports",
            _render_message_list(
                export_state["planned"],
                "No image exports available.",
            ),
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
        + feature_section
        + network_section
        + graphviz_section
        + reduction_section
        + image_export_section
        + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        if use_plotly:
            _export_queued_plotly(export_state, base_dir, notices)
        elif use_mpl:
            _export_queued_matplotlib(export_state, base_dir, notices)
        elif use_svg:
            _export_queued_svg(export_state, base_dir, notices)
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    result = store.ensure(report_manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_report_from_artifact(result.path, run_root)
        run_id_for_export = run_root.name
        _emit_network_exports(
            run_root=run_root,
            run_id=run_id_for_export,
            store=store,
            run_payload=run_payload,
            bipartite_manifest=selected_graph_manifest,
            bipartite_payload=graph_payload,
            bipartite_data=bipartite_payload,
            flux_manifest=selected_flux_manifest,
            flux_payload=flux_payload,
            reduction_ids=[entry["id"] for entry in reduction_inputs],
            graphviz_cfg=graphviz_cfg,
            notices=notices,
        )
        _emit_timeseries_exports(
            run_root=run_root,
            run_id=run_id_for_export,
            run_payload=run_payload,
            species_var=species_series_var,
            species_axis=species_axis_name,
            species_names=selected_species,
            rate_var=rate_series_var,
            rate_axis=rate_axis_name,
            rate_names=rate_selected_names,
            max_points=max(species_max_points, rate_max_points),
            notices=notices,
        )
    return result


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

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        parents=parent_ids,
        inputs=inputs_payload,
        config=manifest_cfg,
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

    chart_backend, plotly_ctx, mpl_ctx = _resolve_chart_backend(viz_cfg, notices)
    plotly_state = {"include_js": True}
    use_plotly = chart_backend == "plotly" and plotly_ctx is not None
    use_mpl = chart_backend == "matplotlib" and mpl_ctx is not None
    use_svg = chart_backend == "svg"
    if use_plotly:
        go, pio = plotly_ctx
        export_state = _init_export_state(viz_cfg, plotly_ctx, notices)
    elif use_mpl:
        export_state = _init_matplotlib_export_state(viz_cfg, notices)
    elif use_svg:
        export_state = _init_svg_export_state(viz_cfg, notices)
    else:
        export_state = None

    graph_manifests = [m for m in input_manifests if m.kind == "graphs"]
    run_manifests = [m for m in input_manifests if m.kind == "runs"]
    reduction_manifests = [m for m in input_manifests if m.kind == "reduction"]
    reaction_count = _infer_reaction_count(viz_cfg, graph_manifests, store, notices)

    selected_run, run_payload = _select_run_and_payload(
        run_manifests,
        store,
        notices,
        multiple_notice="Multiple runs provided; using the first for graphviz.",
    )

    (
        selected_graph_manifest,
        graph_payload,
        bipartite_payload,
        selected_flux_manifest,
        flux_payload,
    ) = _load_graph_payload_for_manifests(graph_manifests, store, notices)

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

    def _mean(values: Sequence[Optional[float]]) -> float:
        cleaned = [value for value in values if value is not None and math.isfinite(value)]
        if not cleaned:
            return math.nan
        return sum(cleaned) / len(cleaned)

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

    def _extract_disabled_reactions(
        patch_payload: Mapping[str, Any],
    ) -> tuple[set[str], set[int]]:
        reaction_ids: set[str] = set()
        reaction_indices: set[int] = set()
        for key in ("disabled_reactions", "reaction_multipliers"):
            entries = patch_payload.get(key)
            if isinstance(entries, Mapping):
                entries = [entries]
            if not isinstance(entries, Sequence) or isinstance(
                entries, (str, bytes, bytearray)
            ):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                reaction_id = entry.get("reaction_id") or entry.get("reaction")
                if isinstance(reaction_id, str) and reaction_id.strip():
                    reaction_ids.add(reaction_id)
                    continue
                idx = _coerce_optional_int(entry.get("index"))
                if idx is not None:
                    reaction_indices.add(idx)
        return reaction_ids, reaction_indices

    def _filter_bipartite_by_disabled(
        nodes: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
        *,
        disabled_ids: set[str],
        disabled_indices: set[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        if not disabled_ids and not disabled_indices:
            return list(nodes), list(links), 0
        disabled_node_ids: set[str] = set()
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if node.get("kind") != "reaction":
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            identifiers: list[str] = []
            for key in ("reaction_equation", "equation", "reaction_id", "label", "id"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    identifiers.append(value.strip())
            reaction_index = _coerce_optional_int(node.get("reaction_index"))
            if (
                any(identifier in disabled_ids for identifier in identifiers)
            ) or (reaction_index is not None and reaction_index in disabled_indices):
                disabled_node_ids.add(node_id)
        if not disabled_node_ids:
            return list(nodes), list(links), 0
        filtered_nodes = [
            dict(node)
            for node in nodes
            if isinstance(node, Mapping)
            and isinstance(node.get("id"), str)
            and node.get("id") not in disabled_node_ids
        ]
        filtered_links = [
            dict(link)
            for link in links
            if isinstance(link, Mapping)
            and link.get("source") not in disabled_node_ids
            and link.get("target") not in disabled_node_ids
        ]
        return filtered_nodes, filtered_links, len(disabled_node_ids)

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

    comparison_cards: list[str] = []

    # Optimization objective comparison
    opt_objectives: dict[str, dict[str, float]] = {}
    opt_labels: list[str] = []
    for manifest in input_manifests:
        if manifest.kind != "optimization":
            continue
        best_map, _, _ = _summarize_optimization(manifest)
        if not best_map:
            continue
        label = f"{_group_label('optimization', manifest.id)}:{manifest.id[:6]}"
        opt_labels.append(label)
        for name, (value, _) in best_map.items():
            opt_objectives.setdefault(name, {})[label] = value
    if opt_objectives and opt_labels:
        if use_plotly:
            fig = go.Figure()
            for name, series in opt_objectives.items():
                fig.add_trace(
                    go.Bar(
                        x=opt_labels,
                        y=[series.get(label) for label in opt_labels],
                        name=name,
                    )
                )
            fig.update_layout(
                title="Optimization Objective Comparison",
                barmode="group",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=60, b=60),
            )
            comparison_cards.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        "opt_objective_comparison",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.2))
            width = 0.8 / max(len(opt_objectives), 1)
            x = list(range(len(opt_labels)))
            for idx, (name, series) in enumerate(opt_objectives.items()):
                values = [series.get(label, math.nan) for label in opt_labels]
                ax.bar([v + idx * width for v in x], values, width=width, label=name)
            ax.set_title("Optimization Objective Comparison")
            ax.set_xticks([v + width * (len(opt_objectives) - 1) / 2 for v in x])
            ax.set_xticklabels(opt_labels, rotation=30, ha="right")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            comparison_cards.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        "opt_objective_comparison",
                        alt="Optimization Objective Comparison",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            for name, series in opt_objectives.items():
                chart_html = _build_svg_bar_chart(
                    title=f"Optimization Objective ({name})",
                    labels=opt_labels,
                    values=[series.get(label, 0.0) for label in opt_labels],
                )
                comparison_cards.append(
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            f"opt_objective_{name}",
                            notices,
                        )
                    )
                )

    # Assimilation final misfit comparison
    assim_labels: list[str] = []
    assim_series: dict[str, list[float]] = {"mean": [], "min": [], "max": []}
    for manifest in input_manifests:
        if manifest.kind != "assimilation":
            continue
        mean_misfit, min_misfit, max_misfit, _, _, _ = _summarize_assimilation(
            manifest
        )
        if mean_misfit is None and min_misfit is None and max_misfit is None:
            continue
        label = f"{_group_label('assimilation', manifest.id)}:{manifest.id[:6]}"
        assim_labels.append(label)
        assim_series["mean"].append(mean_misfit if mean_misfit is not None else math.nan)
        assim_series["min"].append(min_misfit if min_misfit is not None else math.nan)
        assim_series["max"].append(max_misfit if max_misfit is not None else math.nan)
    if assim_labels:
        if use_plotly:
            fig = go.Figure()
            for name, values in assim_series.items():
                fig.add_trace(go.Bar(x=assim_labels, y=values, name=name))
            fig.update_layout(
                title="Assimilation Final Misfit Comparison",
                barmode="group",
                template="plotly_white",
                height=320,
                margin=dict(l=50, r=20, t=60, b=60),
            )
            comparison_cards.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        "assim_final_misfit",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.2))
            width = 0.8 / 3
            x = list(range(len(assim_labels)))
            for idx, (name, values) in enumerate(assim_series.items()):
                ax.bar([v + idx * width for v in x], values, width=width, label=name)
            ax.set_title("Assimilation Final Misfit Comparison")
            ax.set_xticks([v + width for v in x])
            ax.set_xticklabels(assim_labels, rotation=30, ha="right")
            ax.legend(fontsize=8, ncol=3)
            fig.tight_layout()
            comparison_cards.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        "assim_final_misfit",
                        alt="Assimilation Final Misfit Comparison",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            chart_html = _build_svg_bar_chart(
                title="Assimilation Final Misfit (mean)",
                labels=assim_labels,
                values=assim_series["mean"],
            )
            comparison_cards.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        "assim_final_misfit_mean",
                        notices,
                    )
                )
            )

    # Validation pass rate comparison
    val_labels: list[str] = []
    val_rates: list[float] = []
    val_disabled: list[float] = []
    for manifest in input_manifests:
        if manifest.kind != "validation":
            continue
        pass_rate, disabled_count, _ = _summarize_validation(manifest)
        if pass_rate is None and disabled_count is None:
            continue
        label = f"{_group_label('validation', manifest.id)}:{manifest.id[:6]}"
        val_labels.append(label)
        val_rates.append(pass_rate if pass_rate is not None else math.nan)
        val_disabled.append(
            float(disabled_count) if disabled_count is not None else math.nan
        )
    if val_labels:
        if use_plotly:
            fig = go.Figure(
                data=go.Bar(
                    x=val_labels,
                    y=val_rates,
                    marker=dict(color="#3a6ea5"),
                )
            )
            fig.update_layout(
                title="Validation Pass Rate Comparison",
                yaxis=dict(tickformat=".0%"),
                template="plotly_white",
                height=300,
                margin=dict(l=50, r=20, t=60, b=60),
            )
            comparison_cards.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        "validation_pass_comparison",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(5.8, 2.8))
            ax.bar(val_labels, val_rates, color="#3a6ea5")
            ax.set_title("Validation Pass Rate Comparison")
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            comparison_cards.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        "validation_pass_comparison",
                        alt="Validation Pass Rate Comparison",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            chart_html = _build_svg_bar_chart(
                title="Validation Pass Rate Comparison",
                labels=val_labels,
                values=val_rates,
            )
            comparison_cards.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        "validation_pass_comparison",
                        notices,
                    )
                )
            )

    # Reduction impact scatter
    if val_labels and any(not math.isnan(value) for value in val_disabled):
        points = [
            (x, y)
            for x, y in zip(val_disabled, val_rates)
            if not math.isnan(x) and not math.isnan(y)
        ]
        labels = [
            label
            for label, x, y in zip(val_labels, val_disabled, val_rates)
            if not math.isnan(x) and not math.isnan(y)
        ]
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            if use_plotly:
                fig = go.Figure(
                    data=go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        marker=dict(size=8, color="#9c6644"),
                    )
                )
                fig.update_layout(
                    title="Reduction Size vs Pass Rate",
                    xaxis_title="disabled reactions",
                    yaxis_title="pass rate",
                    yaxis=dict(tickformat=".0%"),
                    template="plotly_white",
                    height=300,
                    margin=dict(l=50, r=20, t=60, b=40),
                )
                comparison_cards.append(
                    _wrap_card(
                        _plotly_html_with_export(
                            fig,
                            pio,
                            plotly_state,
                            export_state,
                            "reduction_vs_pass",
                        )
                    )
                )
            elif use_mpl:
                fig, ax = mpl_ctx.subplots(figsize=(5.6, 2.8))
                ax.scatter(xs, ys, color="#9c6644")
                for label, x_val, y_val in zip(labels, xs, ys):
                    ax.annotate(label, (x_val, y_val), fontsize=7)
                ax.set_title("Reduction Size vs Pass Rate")
                ax.set_xlabel("disabled reactions")
                ax.set_ylabel("pass rate")
                ax.set_ylim(0.0, 1.05)
                fig.tight_layout()
                comparison_cards.append(
                    _wrap_card(
                        _matplotlib_html_with_export(
                            fig,
                            export_state,
                            "reduction_vs_pass",
                            alt="Reduction Size vs Pass Rate",
                            notices=notices,
                        )
                    )
                )
            elif use_svg:
                chart_html = _build_svg_scatter_chart(
                    title="Reduction Size vs Pass Rate",
                    x_values=xs,
                    y_values=ys,
                )
                comparison_cards.append(
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            "reduction_vs_pass",
                            notices,
                        )
                    )
                )

    comparison_section = _panel(
        "Benchmark Comparisons",
        "<div class=\"grid\">" + "".join(comparison_cards) + "</div>"
        if comparison_cards
        else "<p class=\"muted\">No benchmark comparison charts available.</p>",
    )

    optimization_charts: list[str] = []
    for manifest in input_manifests:
        if manifest.kind != "optimization":
            continue
        table_path = store.artifact_dir("optimization", manifest.id) / "history.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read optimization/{manifest.id}: {exc}")
            continue
        objective_groups: dict[str, list[tuple[int, float, str]]] = {}
        for index, row in enumerate(rows):
            name = row.get("objective_name")
            if not isinstance(name, str) or not name.strip():
                continue
            value = _coerce_float(row.get("objective"))
            if value is None:
                continue
            direction = str(row.get("direction", "min")).lower()
            if direction not in ("min", "max"):
                direction = "min"
            sample_id = _coerce_optional_int(row.get("sample_id"))
            if sample_id is None:
                sample_id = index
            objective_groups.setdefault(name, []).append((sample_id, value, direction))
        if not objective_groups:
            optimization_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">No objective history.</p>"
                )
            )
            continue
        series_map: dict[str, tuple[list[int], list[float]]] = {}
        for name, entries in objective_groups.items():
            entries.sort(key=lambda item: item[0])
            best_values: list[float] = []
            x_values: list[int] = []
            current_best: Optional[float] = None
            direction = entries[0][2] if entries else "min"
            for sample_id, value, _ in entries:
                if current_best is None:
                    current_best = value
                elif direction == "min":
                    current_best = min(current_best, value)
                else:
                    current_best = max(current_best, value)
                x_values.append(sample_id)
                best_values.append(current_best)
            series_map[name] = (x_values, best_values)
        if use_plotly:
            fig = go.Figure()
            for name, (x_values, best_values) in series_map.items():
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=best_values,
                        mode="lines+markers",
                        name=name,
                    )
                )
            fig.update_layout(
                title=f"Optimization Convergence ({manifest.id})",
                xaxis_title="sample_id",
                yaxis_title="best objective",
                template="plotly_white",
                height=300,
                margin=dict(l=50, r=20, t=60, b=40),
            )
            optimization_charts.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        f"optimization_convergence_{manifest.id}",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.0))
            for name, (x_values, best_values) in series_map.items():
                ax.plot(x_values, best_values, marker="o", label=name)
            ax.set_title(f"Optimization Convergence ({manifest.id})")
            ax.set_xlabel("sample_id")
            ax.set_ylabel("best objective")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            optimization_charts.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        f"optimization_convergence_{manifest.id}",
                        alt="Optimization Convergence",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            max_len = max((len(values) for _, values in series_map.values()), default=0)
            times = list(range(max_len))
            aligned: dict[str, list[float]] = {}
            for name, (_, values) in series_map.items():
                if not values:
                    continue
                if len(values) < max_len:
                    padded = values + [values[-1]] * (max_len - len(values))
                else:
                    padded = values
                aligned[name] = padded
            chart_html = _build_svg_line_chart(
                title=f"Optimization Convergence ({manifest.id})",
                times=times,
                series=aligned,
            )
            optimization_charts.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        f"optimization_convergence_{manifest.id}",
                        notices,
                    )
                )
            )
        else:
            optimization_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">Optimization chart unavailable.</p>"
                )
            )

    optimization_chart_section = _panel(
        "Optimization Convergence",
        "<div class=\"grid\">" + "".join(optimization_charts) + "</div>"
        if optimization_charts
        else "<p class=\"muted\">No optimization charts available.</p>",
    )

    assimilation_charts: list[str] = []
    for manifest in input_manifests:
        if manifest.kind != "assimilation":
            continue
        table_path = store.artifact_dir("assimilation", manifest.id) / "misfit_history.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read assimilation/{manifest.id}: {exc}")
            continue
        rows_sorted = sorted(
            rows,
            key=lambda row: _coerce_optional_int(row.get("iteration")) or 0,
        )
        iterations = [
            _coerce_optional_int(row.get("iteration")) or 0 for row in rows_sorted
        ]
        mean_vals = [_coerce_float(row.get("mean_misfit")) for row in rows_sorted]
        min_vals = [_coerce_float(row.get("min_misfit")) for row in rows_sorted]
        max_vals = [_coerce_float(row.get("max_misfit")) for row in rows_sorted]
        if not iterations:
            assimilation_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">No misfit history.</p>"
                )
            )
            continue
        if use_plotly:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iterations, y=mean_vals, mode="lines+markers", name="mean")
            )
            fig.add_trace(
                go.Scatter(x=iterations, y=min_vals, mode="lines+markers", name="min")
            )
            fig.add_trace(
                go.Scatter(x=iterations, y=max_vals, mode="lines+markers", name="max")
            )
            fig.update_layout(
                title=f"Assimilation Misfit ({manifest.id})",
                xaxis_title="iteration",
                yaxis_title="misfit",
                template="plotly_white",
                height=300,
                margin=dict(l=50, r=20, t=60, b=40),
            )
            assimilation_charts.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        f"assimilation_misfit_{manifest.id}",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.0))
            ax.plot(iterations, mean_vals, marker="o", label="mean")
            ax.plot(iterations, min_vals, marker="o", label="min")
            ax.plot(iterations, max_vals, marker="o", label="max")
            ax.set_title(f"Assimilation Misfit ({manifest.id})")
            ax.set_xlabel("iteration")
            ax.set_ylabel("misfit")
            ax.legend(fontsize=8, ncol=3)
            fig.tight_layout()
            assimilation_charts.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        f"assimilation_misfit_{manifest.id}",
                        alt="Assimilation Misfit",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            def _safe_series(values: Sequence[Optional[float]]) -> list[float]:
                return [value if value is not None else 0.0 for value in values]

            series_map = {
                "mean": _safe_series(mean_vals),
                "min": _safe_series(min_vals),
                "max": _safe_series(max_vals),
            }
            chart_html = _build_svg_line_chart(
                title=f"Assimilation Misfit ({manifest.id})",
                times=iterations,
                series=series_map,
            )
            assimilation_charts.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        f"assimilation_misfit_{manifest.id}",
                        notices,
                    )
                )
            )
        else:
            assimilation_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">Assimilation chart unavailable.</p>"
                )
            )

    assimilation_chart_section = _panel(
        "Assimilation Misfit History",
        "<div class=\"grid\">" + "".join(assimilation_charts) + "</div>"
        if assimilation_charts
        else "<p class=\"muted\">No assimilation charts available.</p>",
    )

    validation_charts: list[str] = []
    for manifest in input_manifests:
        if manifest.kind != "validation":
            continue
        table_path = store.artifact_dir("validation", manifest.id) / "metrics.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read validation/{manifest.id}: {exc}")
            continue
        pass_by_patch: dict[str, list[bool]] = {}
        for row in rows:
            patch_index = _coerce_optional_int(row.get("patch_index"))
            key = f"patch_{patch_index}" if patch_index is not None else "all"
            passed = row.get("passed")
            if isinstance(passed, bool):
                pass_by_patch.setdefault(key, []).append(passed)
        if not pass_by_patch:
            validation_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">No validation metrics.</p>"
                )
            )
            continue
        labels: list[str] = []
        rates: list[float] = []
        for key, values in pass_by_patch.items():
            labels.append(key)
            rates.append(sum(values) / len(values) if values else 0.0)
        if use_plotly:
            fig = go.Figure(
                data=go.Bar(
                    x=labels,
                    y=rates,
                    marker=dict(color="#3a6ea5"),
                )
            )
            fig.update_layout(
                title=f"Validation Pass Rate ({manifest.id})",
                yaxis=dict(tickformat=".0%"),
                template="plotly_white",
                height=280,
                margin=dict(l=50, r=20, t=60, b=40),
            )
            validation_charts.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        f"validation_pass_{manifest.id}",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(5.8, 2.6))
            ax.bar(labels, rates, color="#3a6ea5")
            ax.set_title(f"Validation Pass Rate ({manifest.id})")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("pass rate")
            fig.tight_layout()
            validation_charts.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        f"validation_pass_{manifest.id}",
                        alt="Validation Pass Rate",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            chart_html = _build_svg_bar_chart(
                title=f"Validation Pass Rate ({manifest.id})",
                labels=labels,
                values=rates,
            )
            validation_charts.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        f"validation_pass_{manifest.id}",
                        notices,
                    )
                )
            )
        else:
            validation_charts.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">Validation chart unavailable.</p>"
                )
            )

    validation_chart_section = _panel(
        "Validation Pass Rate",
        "<div class=\"grid\">" + "".join(validation_charts) + "</div>"
        if validation_charts
        else "<p class=\"muted\">No validation charts available.</p>",
    )

    reduction_effect_cfg = viz_cfg.get("reduction_effect", {})
    if reduction_effect_cfg is None:
        reduction_effect_cfg = {}
    if not isinstance(reduction_effect_cfg, Mapping):
        raise ConfigError("viz.reduction_effect must be a mapping when provided.")
    reduction_top_n = _coerce_positive_int(
        reduction_effect_cfg.get("top_n"), "viz.reduction_effect.top_n", default=12
    )

    reduction_effect_cards: list[str] = []
    for manifest in input_manifests:
        if manifest.kind != "validation":
            continue
        table_path = store.artifact_dir("validation", manifest.id) / "metrics.parquet"
        try:
            rows = _read_table_rows(table_path)
        except Exception as exc:
            notices.append(f"Failed to read validation/{manifest.id} metrics: {exc}")
            continue
        diff_rows: list[dict[str, Any]] = []
        for row in rows:
            status = row.get("status")
            if status in ("missing", "invalid", "skipped"):
                continue
            baseline_value = _coerce_float(row.get("baseline_value"))
            reduced_value = _coerce_float(row.get("reduced_value"))
            abs_diff = _coerce_float(row.get("abs_diff"))
            rel_diff = _coerce_float(row.get("rel_diff"))
            if baseline_value is None or reduced_value is None:
                continue
            if abs_diff is None:
                abs_diff = abs(baseline_value - reduced_value)
            if rel_diff is None:
                rel_diff = math.nan
            patch_index = _coerce_optional_int(row.get("patch_index"))
            diff_rows.append(
                {
                    "patch_index": patch_index,
                    "baseline_value": baseline_value,
                    "reduced_value": reduced_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
        if not diff_rows:
            reduction_effect_cards.append(
                _wrap_card(
                    f"<strong>{manifest.id}</strong><p class=\"muted\">No reduction diffs available.</p>"
                )
            )
            continue

        patch_stats: dict[str, dict[str, Any]] = {}
        for row in diff_rows:
            patch_index = row.get("patch_index")
            key = f"patch_{patch_index}" if patch_index is not None else "all"
            entry = patch_stats.setdefault(
                key,
                {"abs_values": [], "rel_values": []},
            )
            entry["abs_values"].append(row["abs_diff"])
            entry["rel_values"].append(row["rel_diff"])

        labels = sorted(patch_stats.keys())
        mean_abs_values = [
            _mean([value for value in patch_stats[label]["abs_values"] if value is not None])
            for label in labels
        ]
        mean_rel_values = [
            _mean([value for value in patch_stats[label]["rel_values"] if value is not None])
            for label in labels
        ]

        if use_plotly:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=mean_abs_values, name="mean abs"))
            fig.add_trace(go.Bar(x=labels, y=mean_rel_values, name="mean rel"))
            fig.update_layout(
                title=f"Validation Diff Summary ({manifest.id})",
                barmode="group",
                template="plotly_white",
                height=300,
                margin=dict(l=50, r=20, t=60, b=40),
            )
            reduction_effect_cards.append(
                _wrap_card(
                    _plotly_html_with_export(
                        fig,
                        pio,
                        plotly_state,
                        export_state,
                        f"validation_diff_mean_{manifest.id}",
                    )
                )
            )
        elif use_mpl:
            fig, ax = mpl_ctx.subplots(figsize=(6.0, 3.0))
            x = list(range(len(labels)))
            width = 0.4
            ax.bar([v - width / 2 for v in x], mean_abs_values, width=width, label="mean abs")
            ax.bar([v + width / 2 for v in x], mean_rel_values, width=width, label="mean rel")
            ax.set_title(f"Validation Diff Summary ({manifest.id})")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            reduction_effect_cards.append(
                _wrap_card(
                    _matplotlib_html_with_export(
                        fig,
                        export_state,
                        f"validation_diff_mean_{manifest.id}",
                        alt="Validation Diff Summary",
                        notices=notices,
                    )
                )
            )
        elif use_svg:
            chart_html = _build_svg_bar_chart(
                title=f"Validation Mean Abs Diff ({manifest.id})",
                labels=labels,
                values=mean_abs_values,
            )
            reduction_effect_cards.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        f"validation_mean_abs_{manifest.id}",
                        notices,
                    )
                )
            )
            chart_html = _build_svg_bar_chart(
                title=f"Validation Mean Rel Diff ({manifest.id})",
                labels=labels,
                values=mean_rel_values,
            )
            reduction_effect_cards.append(
                _wrap_card(
                    _svg_html_with_export(
                        chart_html,
                        export_state,
                        f"validation_mean_rel_{manifest.id}",
                        notices,
                    )
                )
            )

        sorted_rows = sorted(
            diff_rows,
            key=lambda row: row.get("abs_diff") if row.get("abs_diff") is not None else 0.0,
            reverse=True,
        )
        sample_rows = sorted_rows[:reduction_top_n]
        x_vals = [row["baseline_value"] for row in sample_rows]
        y_vals = [row["reduced_value"] for row in sample_rows]
        if x_vals and y_vals:
            if use_plotly:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        marker=dict(size=7, color="#2a9d8f"),
                    )
                )
                min_val = min(min(x_vals), min(y_vals))
                max_val = max(max(x_vals), max(y_vals))
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="#94a3b8", dash="dash"),
                        showlegend=False,
                    )
                )
                fig.update_layout(
                    title=f"Baseline vs Reduced (top {reduction_top_n}) ({manifest.id})",
                    xaxis_title="baseline",
                    yaxis_title="reduced",
                    template="plotly_white",
                    height=300,
                    margin=dict(l=50, r=20, t=60, b=40),
                )
                reduction_effect_cards.append(
                    _wrap_card(
                        _plotly_html_with_export(
                            fig,
                            pio,
                            plotly_state,
                            export_state,
                            f"validation_scatter_{manifest.id}",
                        )
                    )
                )
            elif use_mpl:
                fig, ax = mpl_ctx.subplots(figsize=(5.6, 3.0))
                ax.scatter(x_vals, y_vals, color="#2a9d8f")
                min_val = min(min(x_vals), min(y_vals))
                max_val = max(max(x_vals), max(y_vals))
                ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#94a3b8")
                ax.set_title(f"Baseline vs Reduced (top {reduction_top_n})")
                ax.set_xlabel("baseline")
                ax.set_ylabel("reduced")
                fig.tight_layout()
                reduction_effect_cards.append(
                    _wrap_card(
                        _matplotlib_html_with_export(
                            fig,
                            export_state,
                            f"validation_scatter_{manifest.id}",
                            alt="Baseline vs Reduced",
                            notices=notices,
                        )
                    )
                )
            elif use_svg:
                chart_html = _build_svg_scatter_chart(
                    title=f"Baseline vs Reduced (top {reduction_top_n}) ({manifest.id})",
                    x_values=x_vals,
                    y_values=y_vals,
                )
                reduction_effect_cards.append(
                    _wrap_card(
                        _svg_html_with_export(
                            chart_html,
                            export_state,
                            f"validation_scatter_{manifest.id}",
                            notices,
                        )
                    )
                )

    graphviz_cards: list[str] = []
    graphviz_cfg = viz_cfg.get("graphviz", {})
    if graphviz_cfg is None:
        graphviz_cfg = {}
    if not isinstance(graphviz_cfg, Mapping):
        raise ConfigError("viz.graphviz must be a mapping when provided.")
    gv_top_n = _coerce_positive_int(
        graphviz_cfg.get("top_n"), "viz.graphviz.top_n", default=8
    )
    gv_max_nodes = _coerce_positive_int(
        graphviz_cfg.get("max_nodes"), "viz.graphviz.max_nodes", default=80
    )
    gv_max_edges = _coerce_positive_int(
        graphviz_cfg.get("max_edges"), "viz.graphviz.max_edges", default=160
    )
    gv_engine = graphviz_cfg.get("engine", "dot")
    if not isinstance(gv_engine, str) or not gv_engine.strip():
        raise ConfigError("viz.graphviz.engine must be a non-empty string.")
    gv_engine = gv_engine.strip()
    rop_stat = graphviz_cfg.get("rop_stat") or "integral"
    wdot_stat = graphviz_cfg.get("wdot_stat") or "integral"
    if rop_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.graphviz.rop_stat must be a supported stat.")
    if wdot_stat not in ("integral", "max", "mean", "last", "min"):
        raise ConfigError("viz.graphviz.wdot_stat must be a supported stat.")

    def _add_graphviz_card(
        title: str,
        nodes: list[dict[str, Any]],
        links: list[dict[str, Any]],
        *,
        highlight_reactions: Optional[set[str]] = None,
        highlight_species: Optional[set[str]] = None,
        note: Optional[str] = None,
        name: str,
    ) -> None:
        if not nodes or not links:
            graphviz_cards.append(
                _wrap_card(
                    f"<strong>{html.escape(title)}</strong>"
                    "<p class=\"muted\">No graph data available.</p>"
                )
            )
            return
        dot_source = _build_graphviz_bipartite_dot(
            nodes,
            links,
            title=title,
            highlight_reactions=highlight_reactions,
            highlight_species=highlight_species,
            note=note,
        )
        graphviz_cards.append(
            _wrap_card(
                _graphviz_html_with_export(
                    dot_source,
                    export_state,
                    name,
                    notices,
                    engine=gv_engine,
                )
            )
        )

    primary_reduction_payload: Optional[Mapping[str, Any]] = None
    primary_reduction_id: Optional[str] = None
    for manifest in input_manifests:
        if manifest.kind != "validation":
            continue
        selected_patch = manifest.inputs.get("selected_patch")
        if isinstance(selected_patch, Mapping):
            reduction_id = selected_patch.get("reduction_id")
            if isinstance(reduction_id, str) and reduction_id.strip():
                primary_reduction_id = reduction_id
                break
    if primary_reduction_id is None and reduction_manifests:
        primary_reduction_id = reduction_manifests[0].id
    if primary_reduction_id is not None:
        patch_path = (
            store.artifact_dir("reduction", primary_reduction_id) / "mechanism_patch.yaml"
        )
        patch_payload, error = _read_patch_payload(patch_path)
        if error is not None:
            notices.append(f"Reduction patch {primary_reduction_id}: {error}")
        elif patch_payload is not None:
            primary_reduction_payload = patch_payload

    if bipartite_payload is not None:
        graph_nodes = bipartite_payload.get("nodes", [])
        graph_links = bipartite_payload.get("links", [])
        # Top ROP reactions
        rop_ranked, rop_error = ([], "run data missing")
        if run_payload is not None:
            rop_ranked, rop_error = _rank_from_run_payload(
                run_payload,
                var_name=graphviz_cfg.get("rop_var") or "rop_net",
                axis="reaction",
                stat=rop_stat,
                top_n=gv_top_n,
                rank_abs=True,
            )
        if rop_error is None:
            reaction_ids = [name for name, _ in rop_ranked]
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                reaction_ids=reaction_ids,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            _add_graphviz_card(
                "Top ROP Reaction Network",
                nodes,
                links,
                highlight_reactions={
                    node.get("id")
                    for node in nodes
                    if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
                },
                note=note,
                name="graphviz_top_rop",
            )
            if primary_reduction_payload is not None:
                disabled_ids, disabled_indices = _extract_disabled_reactions(
                    primary_reduction_payload
                )
                if disabled_ids or disabled_indices:
                    reduced_nodes, reduced_links, removed_count = _filter_bipartite_by_disabled(
                        graph_nodes,
                        graph_links,
                        disabled_ids=disabled_ids,
                        disabled_indices=disabled_indices,
                    )
                    reduced_subset_nodes, reduced_subset_links, reduced_note = _select_bipartite_subgraph(
                        reduced_nodes,
                        reduced_links,
                        reaction_ids=reaction_ids,
                        max_nodes=gv_max_nodes,
                        max_edges=gv_max_edges,
                    )
                    note_suffix = []
                    if reduced_note:
                        note_suffix.append(reduced_note)
                    if removed_count:
                        note_suffix.append(f"Removed {removed_count} disabled reactions.")
                    note_text = " ".join(note_suffix) if note_suffix else None
                    _add_graphviz_card(
                        "Top ROP Reaction Network (Reduced)",
                        reduced_subset_nodes,
                        reduced_subset_links,
                        highlight_reactions={
                            node.get("id")
                            for node in reduced_subset_nodes
                            if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
                        },
                        note=note_text,
                        name="graphviz_top_rop_reduced",
                    )
        elif rop_error and run_payload is not None:
            notices.append(f"Graphviz ROP network skipped: {rop_error}")

        # Top WDOT species
        wdot_ranked, wdot_error = ([], "run data missing")
        if run_payload is not None:
            wdot_ranked, wdot_error = _rank_from_run_payload(
                run_payload,
                var_name=graphviz_cfg.get("wdot_var") or "net_production_rates",
                axis="species",
                stat=wdot_stat,
                top_n=gv_top_n,
                rank_abs=True,
            )
        if wdot_error is None:
            species_names = [name for name, _ in wdot_ranked]
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                species_names=species_names,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            _add_graphviz_card(
                "Top WDOT Species Network",
                nodes,
                links,
                highlight_species={
                    node.get("id")
                    for node in nodes
                    if node.get("kind") == "species" and isinstance(node.get("id"), str)
                },
                note=note,
                name="graphviz_top_wdot",
            )
        elif wdot_error and run_payload is not None:
            notices.append(f"Graphviz WDOT network skipped: {wdot_error}")

        # Reduction patches
        for manifest in reduction_manifests[:3]:
            patch_path = (
                store.artifact_dir("reduction", manifest.id) / "mechanism_patch.yaml"
            )
            patch_payload, error = _read_patch_payload(patch_path)
            if error is not None or patch_payload is None:
                notices.append(f"Reduction patch {manifest.id}: {error}")
                continue
            reaction_ids: list[str] = []
            reaction_indices: list[int] = []
            for key in ("disabled_reactions", "reaction_multipliers"):
                entries = patch_payload.get(key)
                if isinstance(entries, Mapping):
                    entries = [entries]
                if not isinstance(entries, Sequence) or isinstance(
                    entries, (str, bytes, bytearray)
                ):
                    continue
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    reaction_id = entry.get("reaction_id") or entry.get("reaction")
                    if isinstance(reaction_id, str) and reaction_id.strip():
                        reaction_ids.append(reaction_id)
                        continue
                    idx = _coerce_optional_int(entry.get("index"))
                    if idx is not None:
                        reaction_indices.append(idx)
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                reaction_ids=reaction_ids,
                reaction_indices=reaction_indices,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            highlight = {
                node.get("id")
                for node in nodes
                if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
            }
            _add_graphviz_card(
                f"Reduction Patch Network ({manifest.id})",
                nodes,
                links,
                highlight_reactions=highlight,
                note=note,
                name=f"graphviz_reduction_{manifest.id}",
            )

        # Assimilation parameter network
        for manifest in [m for m in input_manifests if m.kind == "assimilation"][:3]:
            param_path = (
                store.artifact_dir("assimilation", manifest.id) / "parameter_vector.json"
            )
            if not param_path.exists():
                continue
            try:
                param_payload = read_json(param_path)
            except Exception as exc:
                notices.append(f"parameter_vector invalid for assimilation/{manifest.id}: {exc}")
                continue
            params = param_payload.get("parameters")
            if not isinstance(params, Sequence) or isinstance(
                params, (str, bytes, bytearray)
            ):
                continue
            reaction_ids = []
            reaction_indices = []
            for entry in params:
                if not isinstance(entry, Mapping):
                    continue
                reaction_id = entry.get("reaction_id")
                if isinstance(reaction_id, str) and reaction_id.strip():
                    reaction_ids.append(reaction_id)
                    continue
                idx = _coerce_optional_int(entry.get("index"))
                if idx is not None:
                    reaction_indices.append(idx)
            nodes, links, note = _select_bipartite_subgraph(
                graph_nodes,
                graph_links,
                reaction_ids=reaction_ids,
                reaction_indices=reaction_indices,
                max_nodes=gv_max_nodes,
                max_edges=gv_max_edges,
            )
            highlight = {
                node.get("id")
                for node in nodes
                if node.get("kind") == "reaction" and isinstance(node.get("id"), str)
            }
            _add_graphviz_card(
                f"Assimilation Parameter Network ({manifest.id})",
                nodes,
                links,
                highlight_reactions=highlight,
                note=note,
                name=f"graphviz_assim_{manifest.id}",
            )

    graphviz_section = _panel(
        "Mechanism Networks (Graphviz)",
        "<div class=\"grid\">" + "".join(graphviz_cards) + "</div>"
        if graphviz_cards
        else "<p class=\"muted\">No graphviz networks available.</p>",
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

    reduction_effect_section = _panel(
        "Reduction Effect Details",
        "<div class=\"grid\">" + "".join(reduction_effect_cards) + "</div>"
        if reduction_effect_cards
        else "<p class=\"muted\">No reduction effect charts available.</p>",
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

    image_export_section = ""
    if export_state is not None and export_state["planned"]:
        image_export_section = _panel(
            "Image Exports",
            _render_message_list(
                export_state["planned"],
                "No image exports available.",
            ),
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
        + comparison_section
        + graphviz_section
        + optimization_chart_section
        + optimization_section
        + assimilation_chart_section
        + assimilation_section
        + validation_chart_section
        + reduction_effect_section
        + validation_section
        + image_export_section
        + availability_section
    )
    html_doc = _inject_section(html_doc, sections_html)

    def _writer(base_dir: Path) -> None:
        if use_plotly:
            _export_queued_plotly(export_state, base_dir, notices)
        elif use_mpl:
            _export_queued_matplotlib(export_state, base_dir, notices)
        elif use_svg:
            _export_queued_svg(export_state, base_dir, notices)
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    result = store.ensure(report_manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_report_from_artifact(result.path, run_root)
        run_id_for_export = run_root.name

        reduction_ids_for_export: list[str] = []
        validation_manifests = [m for m in input_manifests if m.kind == "validation"]
        for manifest in validation_manifests:
            selected_patch = manifest.inputs.get("selected_patch")
            if isinstance(selected_patch, Mapping):
                rid = selected_patch.get("reduction_id")
                if isinstance(rid, str) and rid.strip() and rid.strip() not in reduction_ids_for_export:
                    reduction_ids_for_export.append(rid.strip())
        for manifest in reduction_manifests:
            if manifest.id not in reduction_ids_for_export:
                reduction_ids_for_export.append(manifest.id)

        # Mapping-only reductions (e.g., superstate_mapping) do not have patches and
        # should not be passed into network patch exports.
        reduction_ids_for_export = [
            rid
            for rid in reduction_ids_for_export
            if (store.artifact_dir("reduction", rid) / "mechanism_patch.yaml").exists()
        ]

        superstate_mapping_id: Optional[str] = None
        for manifest in reduction_manifests:
            mode = manifest.inputs.get("mode") if isinstance(manifest.inputs, Mapping) else None
            if isinstance(mode, str) and mode.strip() == "superstate_mapping":
                superstate_mapping_id = manifest.id
                break

        superreaction_graph_id: Optional[str] = None
        for manifest in graph_manifests:
            graph_path = store.artifact_dir("graphs", manifest.id) / "graph.json"
            if not graph_path.exists():
                continue
            try:
                payload = read_json(graph_path)
            except Exception:
                continue
            if isinstance(payload, Mapping) and payload.get("kind") == "superstate_reaction_merge_batch":
                superreaction_graph_id = manifest.id
                break

        _emit_network_exports(
            run_root=run_root,
            run_id=run_id_for_export,
            store=store,
            run_payload=run_payload,
            bipartite_manifest=selected_graph_manifest,
            bipartite_payload=graph_payload,
            bipartite_data=bipartite_payload,
            flux_manifest=selected_flux_manifest,
            flux_payload=flux_payload,
            reduction_ids=reduction_ids_for_export,
            graphviz_cfg=graphviz_cfg,
            notices=notices,
        )
        _emit_validation_exports(
            run_root=run_root,
            run_id=run_id_for_export,
            store=store,
            validation_manifests=validation_manifests,
            superstate_mapping_id=superstate_mapping_id,
            superreaction_graph_id=superreaction_graph_id,
            notices=notices,
        )
    return result


register("task", "viz.ds_dashboard", ds_dashboard)
register("task", "viz.chem_dashboard", chem_dashboard)
register("task", "viz.benchmark_report", benchmark_report)

__all__ = ["benchmark_report", "chem_dashboard", "ds_dashboard", "run"]
