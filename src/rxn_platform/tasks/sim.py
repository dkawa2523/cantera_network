"""Simulation task entrypoints."""

from __future__ import annotations

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
from rxn_platform.backends.base import dump_run_dataset
from rxn_platform.core import (
    ArtifactManifest,
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
    resolve_repo_path,
)
from rxn_platform.errors import ArtifactError, BackendError, ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.registry import Registry, register, resolve_backend
from rxn_platform.reporting import render_report_html
from rxn_platform.store import ArtifactCacheResult, ArtifactStore

RUN_STATE_DIRNAME = "state.zarr"
DEFAULT_TOP_SPECIES = 3
DEFAULT_MAX_POINTS = 200


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


def _backend_name(sim_cfg: Mapping[str, Any]) -> str:
    name = sim_cfg.get("name") or sim_cfg.get("backend")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("sim.name must be a non-empty string.")
    return name


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


def _load_run_dataset_payload(run_dir: Path) -> dict[str, Any]:
    dataset_path = run_dir / RUN_STATE_DIRNAME / "dataset.json"
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
            f"Run dataset not found at {dataset_path} (xarray missing: {exc})."
        ) from exc
    dataset = xr.open_zarr(run_dir / RUN_STATE_DIRNAME)
    coords = {
        name: {"dims": [name], "data": dataset.coords[name].values.tolist()}
        for name in dataset.coords
    }
    data_vars = {
        name: {"dims": list(dataset[name].dims), "data": dataset[name].values.tolist()}
        for name in dataset.data_vars
    }
    return {"coords": coords, "data_vars": data_vars, "attrs": dict(dataset.attrs)}


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
    manifest = ArtifactManifest(
        schema_version=1,
        kind="runs",
        id=run_id,
        created_at=_utc_now_iso(),
        parents=[],
        inputs=inputs,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        dump_run_dataset(dataset, base_dir / RUN_STATE_DIRNAME)

    return store.ensure(manifest, writer=_writer)


register("task", "sim.run", run)

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
    dataset_payload = _load_run_dataset_payload(run_dir)

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

    charts: list[str] = []
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
            _build_svg_line_chart(
                title=label,
                times=time_series,
                series={label: numeric_values},
                unit=str(units.get(key, "")) if units else None,
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
                    _build_svg_line_chart(
                        title="Top Species (X)",
                        times=time_series,
                        series=series_map,
                        unit=str(units.get("X", "")) if units else None,
                    )
                )

    if charts:
        chart_cards = "".join(
            "<div style=\"border:1px solid var(--border);border-radius:16px;padding:12px;background:#fbfaf7;\">"
            + chart
            + "</div>"
            for chart in charts
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

    inputs_payload = {"artifacts": [{"kind": "runs", "id": run_id}]}
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
        parents=[run_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
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
    html_doc = _inject_section(html_doc, charts_html)

    def _writer(base_dir: Path) -> None:
        (base_dir / "index.html").write_text(html_doc, encoding="utf-8")

    return store.ensure(report_manifest, writer=_writer)


register("task", "sim.viz", viz)

__all__ = ["run", "viz", "validate_config"]
