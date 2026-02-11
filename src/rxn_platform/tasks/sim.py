"""Simulation task entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import hashlib
import html
import math
from pathlib import Path
from typing import Any, Optional
from rxn_platform.backends.base import dump_run_dataset
from rxn_platform.core import (
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
    resolve_repo_path,
)
from rxn_platform.errors import ArtifactError, BackendError, ConfigError
from rxn_platform.io_utils import write_json_atomic
from rxn_platform.registry import Registry, register, resolve_backend
from rxn_platform.reporting import render_report_html
from rxn_platform.run_store import (
    resolve_run_root_from_store,
    sync_timeseries_from_artifact,
)
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    load_run_dataset_payload,
    resolve_cfg as _resolve_cfg,
)

RUN_STATE_DIRNAME = "state.zarr"
DEFAULT_TOP_SPECIES = 3
DEFAULT_MAX_POINTS = 200


def _ensure_backends_registered() -> None:
    import rxn_platform.backends.dummy  # noqa: F401
    import rxn_platform.backends.cantera  # noqa: F401


def _extract_sim_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "sim" in cfg:
        sim_cfg = cfg.get("sim")
        if not isinstance(sim_cfg, Mapping):
            raise ConfigError("sim config must be a mapping.")
        return dict(cfg), dict(sim_cfg)
    if "name" in cfg:
        sim_cfg = dict(cfg)
        return {"sim": sim_cfg}, sim_cfg
    raise ConfigError("sim config is missing.")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ConfigError(f"conditions_file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ConfigError(f"conditions_file is empty: {path}")
    return rows


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


def _coerce_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be numeric.") from exc


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
        composition = dict(initial.get("X") or {})
        if phi <= 0.0:
            raise ConfigError("phi must be positive.")
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


def _extract_run_csv_settings(
    resolved_cfg: Mapping[str, Any],
    sim_cfg: Mapping[str, Any],
) -> tuple[Path, Optional[str], Optional[int], str]:
    sources: list[Mapping[str, Any]] = []
    for key in ("params", "inputs", "benchmarks"):
        value = resolved_cfg.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    sources.append(resolved_cfg)
    sources.append(sim_cfg)

    conditions_file: Optional[Any] = None
    for source in sources:
        for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
            if key in source:
                conditions_file = source.get(key)
                break
        if conditions_file is not None:
            break
    if conditions_file is None:
        raise ConfigError("conditions_file is required for sim.run_csv.")
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
                    case_id = source.get(key)
                    break
        if row_index is None and "row_index" in source:
            row_index = source.get("row_index")
        if case_col is None:
            for key in ("case_column", "case_col", "case_field"):
                if key in source and source.get(key) is not None:
                    case_col = source.get(key)
                    break
        if case_id is not None and row_index is not None and case_col is not None:
            break
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ConfigError("case_id must be a non-empty string.")
        case_id = case_id.strip()

    if row_index is not None:
        if isinstance(row_index, bool):
            raise ConfigError("row_index must be an integer.")
        try:
            row_index = int(row_index)
        except (TypeError, ValueError) as exc:
            raise ConfigError("row_index must be an integer.") from exc

    if case_col is None:
        case_col = "case_id"
    if not isinstance(case_col, str) or not case_col.strip():
        raise ConfigError("case_column must be a non-empty string.")
    case_col = case_col.strip()

    return conditions_path, case_id, row_index, case_col


def _backend_name(sim_cfg: Mapping[str, Any]) -> str:
    name = sim_cfg.get("name") or sim_cfg.get("backend")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("sim.name must be a non-empty string.")
    return name


def _as_float(value: Any, label: str, errors: list[str]) -> None:
    if value is None:
        return
    try:
        float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a float.")


def _as_int(value: Any, label: str, errors: list[str]) -> None:
    if value is None:
        return
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an int.")
        return
    if value_int <= 0:
        errors.append(f"{label} must be a positive int.")


def _validate_dummy_config(sim_cfg: Mapping[str, Any], errors: list[str]) -> None:
    time_cfg = sim_cfg.get("time")
    if time_cfg is not None and not isinstance(time_cfg, Mapping):
        errors.append("sim.time must be a mapping.")
    if isinstance(time_cfg, Mapping):
        _as_float(time_cfg.get("start"), "sim.time.start", errors)
        _as_float(time_cfg.get("stop"), "sim.time.stop", errors)
        _as_int(time_cfg.get("steps"), "sim.time.steps", errors)

    species = sim_cfg.get("species")
    if species is not None:
        if isinstance(species, str) or not isinstance(species, Sequence):
            errors.append("sim.species must be a sequence of strings.")
        else:
            species_list = [item for item in species if isinstance(item, str) and item]
            if len(species_list) != len(list(species)):
                errors.append("sim.species entries must be non-empty strings.")

    initial = sim_cfg.get("initial")
    if initial is not None and not isinstance(initial, Mapping):
        errors.append("sim.initial must be a mapping.")
    if isinstance(initial, Mapping):
        _as_float(initial.get("T"), "sim.initial.T", errors)
        _as_float(initial.get("P"), "sim.initial.P", errors)

    ramp = sim_cfg.get("ramp")
    if ramp is not None and not isinstance(ramp, Mapping):
        errors.append("sim.ramp must be a mapping.")
    if isinstance(ramp, Mapping):
        _as_float(ramp.get("T"), "sim.ramp.T", errors)
        _as_float(ramp.get("P"), "sim.ramp.P", errors)


def _validate_cantera_config(sim_cfg: Mapping[str, Any], errors: list[str]) -> None:
    mechanism = sim_cfg.get("mechanism") or sim_cfg.get("solution")
    if not isinstance(mechanism, str) or not mechanism.strip():
        errors.append("sim.mechanism must be a non-empty string for cantera.")
    else:
        mech_path = Path(mechanism)
        if mech_path.is_absolute() or len(mech_path.parts) > 1:
            resolved = resolve_repo_path(mechanism)
            if not resolved.exists():
                if resolved != mech_path:
                    errors.append(
                        f"sim.mechanism file not found: {mechanism} (resolved to {resolved})"
                    )
                else:
                    errors.append(f"sim.mechanism file not found: {mechanism}")

    initial = sim_cfg.get("initial")
    if not isinstance(initial, Mapping):
        errors.append("sim.initial must be a mapping for cantera.")
    else:
        _as_float(initial.get("T"), "sim.initial.T", errors)
        _as_float(initial.get("P"), "sim.initial.P", errors)
        composition = initial.get("X")
        if composition is None:
            errors.append("sim.initial.X is required for cantera.")
        elif isinstance(composition, str):
            if not composition.strip():
                errors.append("sim.initial.X must be a non-empty string.")
        elif isinstance(composition, Mapping):
            if not composition:
                errors.append("sim.initial.X must include at least one species.")
            for key, value in composition.items():
                if not isinstance(key, str) or not key.strip():
                    errors.append("sim.initial.X keys must be non-empty strings.")
                    break
                _as_float(value, f"sim.initial.X[{key}]", errors)
        else:
            errors.append("sim.initial.X must be a string or mapping.")

    time_cfg = sim_cfg.get("time_grid", sim_cfg.get("time"))
    if time_cfg is None:
        errors.append("sim.time_grid is required for cantera.")
    elif isinstance(time_cfg, Sequence) and not isinstance(
        time_cfg, (str, bytes, bytearray)
    ):
        if not time_cfg:
            errors.append("sim.time_grid must contain at least one entry.")
        for entry in time_cfg:
            _as_float(entry, "sim.time_grid entry", errors)
    elif isinstance(time_cfg, Mapping):
        if time_cfg.get("points") is not None:
            points = time_cfg.get("points")
            if not isinstance(points, Sequence) or isinstance(
                points, (str, bytes, bytearray)
            ):
                errors.append("sim.time_grid.points must be a sequence of floats.")
            else:
                if not points:
                    errors.append("sim.time_grid.points must not be empty.")
                for entry in points:
                    _as_float(entry, "sim.time_grid.points entry", errors)
        else:
            _as_float(time_cfg.get("start"), "sim.time_grid.start", errors)
            _as_float(time_cfg.get("stop"), "sim.time_grid.stop", errors)
            _as_float(time_cfg.get("dt"), "sim.time_grid.dt", errors)
            _as_int(time_cfg.get("steps"), "sim.time_grid.steps", errors)
    else:
        errors.append("sim.time_grid must be a mapping or sequence of floats.")


def validate_config(
    cfg: Mapping[str, Any],
    *,
    registry: Optional[Registry] = None,
) -> list[str]:
    """Validate simulation config and return a list of issues."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    _ensure_backends_registered()
    resolved_cfg = _resolve_cfg(cfg)
    _, sim_cfg = _extract_sim_cfg(resolved_cfg)
    backend_name = _backend_name(sim_cfg)
    errors: list[str] = []
    try:
        resolve_backend(backend_name, registry=registry)
    except BackendError as exc:
        errors.append(str(exc))

    if backend_name == "dummy":
        _validate_dummy_config(sim_cfg, errors)
    elif backend_name == "cantera":
        _validate_cantera_config(sim_cfg, errors)
    elif backend_name:
        errors.append(f"Unsupported sim backend: {backend_name}")

    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise ConfigError(
            "Sim config validation failed.",
            user_message=f"Sim config validation failed:\n{details}",
        )
    return errors


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


def _extract_run_id(viz_cfg: Mapping[str, Any]) -> str:
    run_id = viz_cfg.get("run_id") or viz_cfg.get("run") or viz_cfg.get("artifact_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ConfigError("viz.run_id must be a non-empty string.")
    return run_id


def _coerce_numeric(values: Sequence[Any]) -> Optional[list[float]]:
    numbers: list[float] = []
    for entry in values:
        try:
            numbers.append(float(entry))
        except (TypeError, ValueError):
            return None
    return numbers


def _downsample_indices(count: int, max_points: int) -> list[int]:
    if max_points <= 0 or count <= max_points:
        return list(range(count))
    step = int(math.ceil(count / float(max_points)))
    return list(range(0, count, step))


def _downsample_series(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices if index < len(values)]


def _select_top_species(
    species: Sequence[Any],
    series: Sequence[Sequence[Any]],
    top_n: int,
) -> list[tuple[str, list[float]]]:
    if not species or not series:
        return []
    species_names = [str(item) for item in species]
    if top_n <= 0:
        return []
    maxima: list[tuple[float, str, list[float]]] = []
    for idx, name in enumerate(species_names):
        values = []
        for row in series:
            if idx >= len(row):
                continue
            try:
                values.append(float(row[idx]))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        maxima.append((max(values), name, values))
    maxima.sort(key=lambda item: item[0], reverse=True)
    return [(name, values) for _, name, values in maxima[:top_n]]


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
    height = 180
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
    height = 200
    margin_left = 40
    margin_right = 14
    margin_top = 20
    margin_bottom = 36

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


def _extract_svg_fragment(chart_html: str) -> Optional[str]:
    start = chart_html.find("<svg")
    end = chart_html.rfind("</svg>")
    if start == -1 or end == -1:
        return None
    return chart_html[start : end + len("</svg>")]


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


def _inject_section(html_doc: str, section_html: str) -> str:
    marker = "\n  <script type=\"application/json\" id=\"report-config\">"
    if marker in html_doc:
        return html_doc.replace(marker, section_html + marker, 1)
    return html_doc + section_html


def run_csv(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
    cache_bust: Optional[str] = None,
) -> ArtifactCacheResult:
    """Run a simulation using a single row from a conditions CSV."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    _ensure_backends_registered()
    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, sim_cfg = _extract_sim_cfg(resolved_cfg)

    conditions_path, case_id, row_index, case_col = _extract_run_csv_settings(
        resolved_cfg, sim_cfg
    )
    rows = _load_csv_rows(conditions_path)
    row = _select_csv_row(rows, case_id=case_id, row_index=row_index, case_col=case_col)

    def _pick_value(keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in row and row.get(key) is not None:
                value = row.get(key)
                if isinstance(value, str) and not value.strip():
                    return None
                return value
        return None

    temperature = _coerce_optional_float(
        _pick_value(("T0", "T", "temperature")), "temperature"
    )
    pressure_atm = _coerce_optional_float(
        _pick_value(("P0_atm", "P_atm", "P0", "pressure_atm", "pressure")), "pressure_atm"
    )
    phi = _coerce_optional_float(_pick_value(("phi",)), "phi")
    t_end = _coerce_optional_float(
        _pick_value(("t_end", "t_end_s", "t_end_seconds")), "t_end"
    )
    row_case_id = row.get(case_col) if case_col in row else None
    if isinstance(row_case_id, str) and not row_case_id.strip():
        row_case_id = None
    if case_id is None and row_case_id is not None:
        case_id = row_case_id

    updated_sim_cfg = _apply_csv_condition(
        sim_cfg,
        temperature=temperature,
        pressure_atm=pressure_atm,
        phi=phi,
        t_end=t_end,
        case_id=case_id,
    )

    updated_cfg = dict(resolved_cfg)
    updated_cfg["sim"] = updated_sim_cfg
    return run(updated_cfg, store=store, registry=registry, cache_bust=cache_bust)


def _hash_file_16(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigError(f"Failed to read conditions_file for hashing: {path}") from exc
    return digest[:16]


def _coerce_case_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _extract_sweep_csv_settings(
    resolved_cfg: Mapping[str, Any],
    sim_cfg: Mapping[str, Any],
) -> tuple[Path, str, str, list[str], str]:
    """Return (conditions_path, case_col, case_mode, case_ids, time_grid_policy)."""
    sources: list[Mapping[str, Any]] = []
    for key in ("params", "inputs", "benchmarks"):
        value = resolved_cfg.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    sources.append(resolved_cfg)
    sources.append(sim_cfg)

    conditions_file: Optional[Any] = None
    for source in sources:
        for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
            if key in source and source.get(key) is not None:
                conditions_file = source.get(key)
                break
        if conditions_file is not None:
            break
    if conditions_file is None:
        raise ConfigError("conditions_file is required for sim.sweep_csv.")
    if not isinstance(conditions_file, (str, Path)) or not str(conditions_file).strip():
        raise ConfigError("conditions_file must be a non-empty string or Path.")
    conditions_path = resolve_repo_path(conditions_file)

    case_col: Optional[Any] = None
    for source in sources:
        for key in ("case_column", "case_col", "case_field"):
            if key in source and source.get(key) is not None:
                case_col = source.get(key)
                break
        if case_col is not None:
            break
    if case_col is None:
        case_col = "case_id"
    if not isinstance(case_col, str) or not case_col.strip():
        raise ConfigError("case_col must be a non-empty string.")
    case_col = case_col.strip()

    case_mode: Optional[Any] = None
    case_ids_raw: Any = None
    time_grid_policy: Optional[Any] = None
    for source in sources:
        if case_mode is None and "case_mode" in source and source.get("case_mode") is not None:
            case_mode = source.get("case_mode")
        if case_ids_raw is None:
            for key in ("case_ids", "cases", "case_id_list"):
                if key in source and source.get(key) is not None:
                    case_ids_raw = source.get(key)
                    break
        if time_grid_policy is None and source.get("time_grid_policy") is not None:
            time_grid_policy = source.get("time_grid_policy")

    if case_mode is None:
        case_mode = "all"
    if not isinstance(case_mode, str) or not case_mode.strip():
        raise ConfigError("case_mode must be a string.")
    case_mode = case_mode.strip().lower()
    if case_mode not in {"all", "list"}:
        raise ConfigError("case_mode must be 'all' or 'list'.")

    selected_case_ids: list[str] = []
    if case_mode == "list":
        if case_ids_raw is None:
            raise ConfigError("case_ids is required when case_mode='list'.")
        if isinstance(case_ids_raw, str):
            selected_case_ids = [_require_nonempty_str(case_ids_raw, "case_ids")]
        elif isinstance(case_ids_raw, Sequence) and not isinstance(
            case_ids_raw, (str, bytes, bytearray)
        ):
            selected_case_ids = [
                _require_nonempty_str(entry, "case_ids")
                for entry in case_ids_raw
                if entry is not None
            ]
        else:
            raise ConfigError("case_ids must be a string or list of strings.")
        selected_case_ids = sorted({cid.strip() for cid in selected_case_ids if cid.strip()})
        if not selected_case_ids:
            raise ConfigError("case_ids must not be empty.")

    if time_grid_policy is None:
        time_grid_policy = "row"
    if not isinstance(time_grid_policy, str) or not time_grid_policy.strip():
        raise ConfigError("time_grid_policy must be a string.")
    time_grid_policy = time_grid_policy.strip().lower()
    if time_grid_policy not in {"row", "fixed", "max"}:
        raise ConfigError("time_grid_policy must be one of: row, fixed, max.")

    return conditions_path, case_col, case_mode, selected_case_ids, time_grid_policy


def sweep_csv(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run a simulation for multiple rows from a conditions CSV and return a RunSet artifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    _ensure_backends_registered()
    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, sim_cfg = _extract_sim_cfg(resolved_cfg)

    conditions_path, case_col, case_mode, selected_case_ids, time_grid_policy = (
        _extract_sweep_csv_settings(resolved_cfg, sim_cfg)
    )
    rows = _load_csv_rows(conditions_path)

    # Normalize case ids from rows (fill blanks as row_0000).
    normalized_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_map = dict(row)
        row_case_id = _coerce_case_id(row_map.get(case_col))
        if row_case_id is None:
            row_case_id = f"row_{idx:04d}"
            row_map[case_col] = row_case_id
        normalized_rows.append(row_map)

    row_by_case: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(normalized_rows):
        cid = _coerce_case_id(row.get(case_col)) or f"row_{idx:04d}"
        if cid in row_by_case:
            raise ConfigError(f"Duplicate case_id {cid!r} in conditions file.")
        row_by_case[cid] = row

    if case_mode == "all":
        case_ids = sorted(row_by_case.keys())
    else:
        missing = [cid for cid in selected_case_ids if cid not in row_by_case]
        if missing:
            raise ConfigError(f"case_ids not found in conditions file: {', '.join(missing)}")
        case_ids = list(selected_case_ids)

    def _pick_value(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in row and row.get(key) is not None:
                value = row.get(key)
                if isinstance(value, str) and not value.strip():
                    return None
                return str(value)
        return None

    # Determine the applied t_end for time_grid_policy=max.
    max_t_end: Optional[float] = None
    if time_grid_policy == "max":
        for cid in case_ids:
            row = row_by_case[cid]
            t_end_row = _coerce_optional_float(
                _pick_value(row, ("t_end", "t_end_s", "t_end_seconds")),
                "t_end",
            )
            if t_end_row is None:
                continue
            if not math.isfinite(t_end_row):
                continue
            if max_t_end is None or t_end_row > max_t_end:
                max_t_end = float(t_end_row)

    # Pass 1: compute per-case sim configs + run_ids without executing the backend.
    backend_name = _backend_name(sim_cfg)
    base_time_cfg = sim_cfg.get("time_grid", sim_cfg.get("time"))
    time_grid_applied: Optional[dict[str, Any]] = None
    if time_grid_policy in {"fixed", "max"} and isinstance(base_time_cfg, Mapping):
        start = _coerce_optional_float(base_time_cfg.get("start"), "time.start")
        stop = _coerce_optional_float(base_time_cfg.get("stop"), "time.stop")
        steps = base_time_cfg.get("steps")
        steps_int: Optional[int] = None
        if steps is not None:
            if isinstance(steps, bool):
                raise ConfigError("time.steps must be an integer.")
            try:
                steps_int = int(steps)
            except (TypeError, ValueError) as exc:
                raise ConfigError("time.steps must be an integer.") from exc
        # stop may be overridden below for max.
        if time_grid_policy == "max" and max_t_end is not None:
            stop = float(max_t_end)
        time_grid_applied = {
            "start": float(start) if start is not None else None,
            "stop": float(stop) if stop is not None else None,
            "steps": int(steps_int) if steps_int is not None else None,
        }

    per_case: list[dict[str, Any]] = []
    for cid in case_ids:
        row = row_by_case[cid]
        temperature = _coerce_optional_float(
            _pick_value(row, ("T0", "T", "temperature")), "temperature"
        )
        pressure_atm = _coerce_optional_float(
            _pick_value(row, ("P0_atm", "P_atm", "P0", "pressure_atm", "pressure")),
            "pressure_atm",
        )
        phi = _coerce_optional_float(_pick_value(row, ("phi",)), "phi")
        t_end_row = _coerce_optional_float(
            _pick_value(row, ("t_end", "t_end_s", "t_end_seconds")),
            "t_end",
        )
        if time_grid_policy == "row":
            t_end_applied = t_end_row
        elif time_grid_policy == "fixed":
            t_end_applied = None
        else:
            t_end_applied = max_t_end

        updated_sim_cfg = _apply_csv_condition(
            sim_cfg,
            temperature=temperature,
            pressure_atm=pressure_atm,
            phi=phi,
            t_end=t_end_applied,
            case_id=cid,
        )

        # Match sim.run normalization: normalize multipliers and drop disabled_reactions.
        try:
            normalized_multipliers = normalize_reaction_multipliers(updated_sim_cfg)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"reaction multipliers are invalid: {exc}") from exc
        sim_cfg_norm = dict(updated_sim_cfg)
        if normalized_multipliers:
            sim_cfg_norm["reaction_multipliers"] = normalized_multipliers
        else:
            sim_cfg_norm.pop("reaction_multipliers", None)
        sim_cfg_norm.pop("disabled_reactions", None)

        # Match sim.run_csv manifest identity: keep per-case config minimal so that
        # the resulting run_id does not depend on sweep-level settings.
        run_manifest_cfg: dict[str, Any] = {
            "sim": dict(sim_cfg_norm),
            "inputs": {},
            "params": {
                "conditions_file": str(conditions_path),
                "case_id": cid,
                "case_column": case_col,
            },
        }

        run_id = make_run_id(run_manifest_cfg, exclude_keys=("hydra",))
        per_case.append(
            {
                "case_id": cid,
                "row": row,
                "temperature": temperature,
                "pressure_atm": pressure_atm,
                "phi": phi,
                "t_end_row": t_end_row,
                "t_end_applied": t_end_applied,
                "sim_cfg": sim_cfg_norm,
                "run_manifest_cfg": run_manifest_cfg,
                "run_id": run_id,
                "normalized_multipliers": normalized_multipliers,
                "backend_name": backend_name,
            }
        )

    run_ids = [entry["run_id"] for entry in per_case]
    case_to_run = {entry["case_id"]: entry["run_id"] for entry in per_case}
    conditions_hash = _hash_file_16(conditions_path)

    inputs_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "run_set",
        "conditions_file": str(conditions_path),
        "conditions_hash": conditions_hash,
        "case_col": case_col,
        "case_mode": case_mode,
        "case_ids": list(case_ids),
        "run_ids": list(run_ids),
        "case_to_run": dict(case_to_run),
        "time_grid_policy": time_grid_policy,
        "time_grid_applied": time_grid_applied,
    }

    run_set_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    run_set_manifest = build_manifest(
        kind="run_sets",
        artifact_id=run_set_id,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    # Fast path: if the run_set already exists, reuse it (no backend calls).
    if store.exists("run_sets", run_set_id):
        return store.ensure(run_set_manifest)

    # Pass 2: ensure each run exists (execute backend only for missing runs).
    backend = resolve_backend(backend_name, registry=registry)
    run_root = resolve_run_root_from_store(store.root)
    for entry in per_case:
        run_id = entry["run_id"]
        run_manifest_cfg = entry["run_manifest_cfg"]
        sim_cfg_norm = entry["sim_cfg"]
        normalized_multipliers = entry["normalized_multipliers"]

        inputs: dict[str, Any] = {}
        if normalized_multipliers:
            inputs["reaction_multipliers"] = normalized_multipliers
        run_manifest = build_manifest(
            kind="runs",
            artifact_id=run_id,
            inputs=inputs,
            config=run_manifest_cfg,
        )

        if store.exists("runs", run_id):
            store.ensure(run_manifest)
            continue

        dataset = backend.run(sim_cfg_norm)

        def _writer(base_dir: Path, _dataset=dataset) -> None:
            dump_run_dataset(_dataset, base_dir / RUN_STATE_DIRNAME)

        result = store.ensure(run_manifest, writer=_writer)
        if run_root is not None:
            sync_timeseries_from_artifact(result.path, run_root)

    def _writer(base_dir: Path) -> None:
        runs_json = dict(inputs_payload)
        runs_json["case_meta"] = [
            {
                "case_id": entry["case_id"],
                "row_index": int(index),
                "T0": entry["temperature"],
                "P0_atm": entry["pressure_atm"],
                "phi": entry["phi"],
                "t_end_row": entry["t_end_row"],
                "t_end_applied": entry["t_end_applied"],
            }
            for index, entry in enumerate(per_case)
        ]
        write_json_atomic(base_dir / "runs.json", runs_json)

    return store.ensure(run_set_manifest, writer=_writer)


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
    cache_bust: Optional[str] = None,
) -> ArtifactCacheResult:
    """Run a simulation backend and store the run artifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    _ensure_backends_registered()
    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, sim_cfg = _extract_sim_cfg(resolved_cfg)
    try:
        normalized_multipliers = normalize_reaction_multipliers(sim_cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"reaction multipliers are invalid: {exc}"
        ) from exc

    sim_cfg = dict(sim_cfg)
    if normalized_multipliers:
        sim_cfg["reaction_multipliers"] = normalized_multipliers
    else:
        sim_cfg.pop("reaction_multipliers", None)
    sim_cfg.pop("disabled_reactions", None)

    manifest_cfg = dict(manifest_cfg)
    manifest_sim_cfg = dict(sim_cfg)
    if cache_bust:
        manifest_sim_cfg["cache_bust"] = cache_bust
    manifest_cfg["sim"] = manifest_sim_cfg

    backend_name = _backend_name(sim_cfg)
    backend = resolve_backend(backend_name, registry=registry)

    dataset = backend.run(sim_cfg)
    run_id = make_run_id(manifest_cfg, exclude_keys=("hydra",))
    inputs: dict[str, Any] = {}
    applied_multipliers = normalized_multipliers
    dataset_attrs = getattr(dataset, "attrs", None)
    if isinstance(dataset_attrs, Mapping) and "reaction_multipliers" in dataset_attrs:
        applied = dataset_attrs.get("reaction_multipliers")
        if applied:
            applied_multipliers = applied
    if applied_multipliers:
        inputs["reaction_multipliers"] = applied_multipliers
    manifest = build_manifest(
        kind="runs",
        artifact_id=run_id,
        inputs=inputs,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        dump_run_dataset(dataset, base_dir / RUN_STATE_DIRNAME)

    result = store.ensure(manifest, writer=_writer)
    run_root = resolve_run_root_from_store(store.root)
    if run_root is not None:
        sync_timeseries_from_artifact(result.path, run_root)
    return result


register("task", "sim.run", run)
register("task", "sim.run_csv", run_csv)
register("task", "sim.sweep_csv", sweep_csv)

def viz(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Render a quick report from a run artifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, viz_cfg = _extract_viz_cfg(resolved_cfg)
    run_id = _extract_run_id(viz_cfg)
    run_manifest = store.read_manifest("runs", run_id)
    run_dir = store.artifact_dir("runs", run_id)
    dataset_payload = load_run_dataset_payload(
        run_dir,
        xarray_missing_message=(
            "Run dataset not found at {dataset_path} (xarray missing: {error})."
        ),
    )

    coords = dataset_payload.get("coords", {})
    data_vars = dataset_payload.get("data_vars", {})
    attrs = dataset_payload.get("attrs", {})

    time_values = []
    time_coord = coords.get("time")
    if isinstance(time_coord, Mapping):
        time_data = time_coord.get("data", [])
        if isinstance(time_data, Sequence):
            time_values = list(time_data)
    time_series = _coerce_numeric(time_values) or list(range(len(time_values)))

    max_points = viz_cfg.get("max_points", DEFAULT_MAX_POINTS)
    if not isinstance(max_points, int) or max_points <= 0:
        raise ConfigError("viz.max_points must be a positive int.")
    top_species = viz_cfg.get("top_species", DEFAULT_TOP_SPECIES)
    if not isinstance(top_species, int) or top_species <= 0:
        raise ConfigError("viz.top_species must be a positive int.")

    indices = _downsample_indices(len(time_series), max_points)
    time_series = _downsample_series(time_series, indices)

    units = {}
    if isinstance(attrs, Mapping):
        units = (
            attrs.get("units", {})
            if isinstance(attrs.get("units", {}), Mapping)
            else {}
        )

    charts: list[tuple[str, str]] = []
    for key, label in (("T", "Temperature"), ("P", "Pressure")):
        series_payload = data_vars.get(key)
        if not isinstance(series_payload, Mapping):
            continue
        values = series_payload.get("data", [])
        if not isinstance(values, Sequence):
            continue
        values = _downsample_series(values, indices)
        numeric_values = _coerce_numeric(values)
        if numeric_values is None or len(numeric_values) != len(time_series):
            continue
        charts.append(
            (
                key.lower(),
                _build_svg_line_chart(
                    title=label,
                    times=time_series,
                    series={label: numeric_values},
                    unit=str(units.get(key, "")) if units else None,
                ),
            )
        )

    species_coord = coords.get("species")
    species = []
    if isinstance(species_coord, Mapping):
        species_data = species_coord.get("data", [])
        if isinstance(species_data, Sequence):
            species = list(species_data)
    x_payload = data_vars.get("X")
    if isinstance(x_payload, Mapping):
        x_series = x_payload.get("data", [])
        if isinstance(x_series, Sequence):
            x_series = _downsample_series(x_series, indices)
            top_series = _select_top_species(species, x_series, top_species)
            if top_series:
                series_map = {name: values for name, values in top_series}
                charts.append(
                    (
                        "top_species_x",
                        _build_svg_line_chart(
                            title="Top Species (X)",
                            times=time_series,
                            series=series_map,
                            unit=str(units.get("X", "")) if units else None,
                        ),
                    )
                )
            if x_series:
                last_row = x_series[-1]
                if isinstance(last_row, Sequence):
                    last_values = _coerce_numeric(last_row)
                    if last_values:
                        pairs = list(zip(species, last_values))
                        pairs.sort(key=lambda item: item[1], reverse=True)
                        if top_species > 0:
                            pairs = pairs[:top_species]
                        labels = [name for name, _ in pairs]
                        values = [value for _, value in pairs]
                        charts.append(
                            (
                                "final_composition_x",
                                _build_svg_bar_chart(
                                    title="Final Composition (X)",
                                    labels=labels,
                                    values=values,
                                    unit=str(units.get("X", "")) if units else None,
                                ),
                            )
                        )

    if charts:
        chart_cards = "".join(
            "<div style=\"border:1px solid var(--border);border-radius:16px;padding:12px;background:#fbfaf7;\">"
            + chart_html
            + "</div>"
            for _, chart_html in charts
        )
        charts_html = (
            "<section class=\"panel\">"
            "<h2>Run Time Series</h2>"
            "<div class=\"grid\">"
            + chart_cards
            + "</div>"
            "</section>"
        )
    else:
        charts_html = (
            "<section class=\"panel\">"
            "<h2>Run Time Series</h2>"
            "<p class=\"muted\">No chartable time series found in run artifact.</p>"
            "</section>"
        )

    export_cfg = viz_cfg.get("export_images", True)
    export_enabled = True
    export_dir_name = viz_cfg.get("image_dir", "images")
    export_formats = _normalize_image_formats(viz_cfg.get("image_formats"))
    if isinstance(export_cfg, Mapping):
        if export_cfg.get("enabled") is not None:
            export_enabled = bool(export_cfg.get("enabled"))
        if export_cfg.get("dir") is not None:
            export_dir_name = export_cfg.get("dir")
        if export_cfg.get("formats") is not None or export_cfg.get("image_formats") is not None:
            export_formats = _normalize_image_formats(
                export_cfg.get("formats", export_cfg.get("image_formats"))
            )
    else:
        export_enabled = bool(export_cfg)
    if not isinstance(export_dir_name, str) or not export_dir_name.strip():
        raise ConfigError("viz.image_dir must be a non-empty string.")

    svg_exports: list[tuple[str, str]] = []
    export_paths: list[str] = []
    export_notes: list[str] = []
    if export_enabled and charts:
        need_png = any(fmt in ("png", "jpg") for fmt in export_formats)
        cairosvg_ok = True
        pillow_ok = True
        if need_png:
            try:
                import cairosvg  # noqa: F401
            except Exception:
                cairosvg_ok = False
                export_notes.append("PNG/JPG export requires cairosvg (not installed).")
        if "jpg" in export_formats:
            try:
                from PIL import Image  # noqa: F401
            except Exception:
                pillow_ok = False
                export_notes.append("JPG export requires Pillow (not installed).")
        for name, chart_html in charts:
            svg = _extract_svg_fragment(chart_html)
            if svg is None:
                continue
            svg_exports.append((name, svg))
            for fmt in export_formats:
                if fmt == "jpg" and not (cairosvg_ok and pillow_ok):
                    continue
                if fmt in ("png", "jpg") and not cairosvg_ok:
                    continue
                export_paths.append(f"{export_dir_name}/{name}.{fmt}")

    image_export_section = ""
    if export_enabled and export_paths:
        note_items = "".join(
            f"<li>{html.escape(note)}</li>" for note in export_notes
        )
        path_items = "".join(
            f"<li>{html.escape(path)}</li>" for path in export_paths
        )
        image_export_section = (
            "<section class=\"panel\">"
            "<h2>Image Exports</h2>"
            + ("<ul>" + note_items + "</ul>" if note_items else "")
            + "<ul>"
            + path_items
            + "</ul>"
            + "</section>"
        )

    inputs_payload = {"artifacts": [{"kind": "runs", "id": run_id}]}
    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        parents=[run_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        notes=f"Generated from run {run_manifest.id}",
    )

    title = viz_cfg.get("title") or "Simulation Report"
    if not isinstance(title, str):
        raise ConfigError("viz.title must be a string.")

    html_doc = render_report_html(
        title=title,
        dashboard="sim",
        created_at=report_manifest.created_at,
        manifest=report_manifest,
        inputs=inputs_payload["artifacts"],
        config=manifest_cfg,
        placeholders=("Quicklook",),
    )
    html_doc = _inject_section(html_doc, charts_html + image_export_section)

    def _writer(base_dir: Path) -> None:
        if export_enabled and svg_exports:
            export_dir = base_dir / export_dir_name
            export_dir.mkdir(parents=True, exist_ok=True)
            exported: list[str] = []
            png_supported = "png" in export_formats or "jpg" in export_formats
            cairosvg = None
            if png_supported:
                try:
                    import cairosvg as _cairosvg  # type: ignore

                    cairosvg = _cairosvg
                except Exception:
                    cairosvg = None
            image_module = None
            if "jpg" in export_formats and cairosvg is not None:
                try:
                    from PIL import Image as _Image  # type: ignore

                    image_module = _Image
                except Exception:
                    image_module = None
            for name, svg in svg_exports:
                if "svg" in export_formats:
                    svg_path = export_dir / f"{name}.svg"
                    svg_path.write_text(svg, encoding="utf-8")
                    exported.append(f"{export_dir_name}/{svg_path.name}")
                if cairosvg is not None and (
                    "png" in export_formats or "jpg" in export_formats
                ):
                    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
                    if "png" in export_formats:
                        png_path = export_dir / f"{name}.png"
                        png_path.write_bytes(png_bytes)
                        exported.append(f"{export_dir_name}/{png_path.name}")
                    if "jpg" in export_formats and image_module is not None:
                        import io

                        with io.BytesIO(png_bytes) as buffer:
                            image = image_module.open(buffer).convert("RGB")
                            jpg_path = export_dir / f"{name}.jpg"
                            image.save(jpg_path, format="JPEG", quality=95)
                            exported.append(f"{export_dir_name}/{jpg_path.name}")
            if exported:
                write_json_atomic(export_dir / "index.json", {"files": exported})
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


register("task", "sim.viz", viz)

__all__ = ["run", "viz", "validate_config"]
