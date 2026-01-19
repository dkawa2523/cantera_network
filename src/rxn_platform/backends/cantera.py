"""Cantera-backed 0D reactor simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rxn_platform.backends.base import RunDataset, SimulationBackend
from rxn_platform.core import normalize_reaction_multipliers, resolve_repo_path
from rxn_platform.errors import BackendError
from rxn_platform.registry import register

try:  # Optional dependency.
    import cantera as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None

try:  # Optional dependency.
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None

try:  # Optional dependency (needed for xarray.to_zarr).
    import zarr  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    zarr = None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BackendError(f"{label} must be a mapping.")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackendError(f"{label} must be a mapping.")
    return value


def _as_float(value: Any, label: str) -> float:
    if value is None:
        raise BackendError(f"{label} must be a float.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"{label} must be a float, got {value!r}.") from exc


def _as_int(value: Any, label: str) -> int:
    if value is None:
        raise BackendError(f"{label} must be an int.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"{label} must be an int, got {value!r}.") from exc


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendError(f"{label} must be a non-empty string.")
    return value


def _as_float_list(value: Any, label: str) -> list[float]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise BackendError(f"{label} must be a sequence of floats.")
    values: list[float] = []
    for entry in value:
        values.append(_as_float(entry, label))
    if not values:
        raise BackendError(f"{label} must contain at least one entry.")
    return values


def _linspace(start: float, stop: float, steps: int) -> list[float]:
    if steps <= 0:
        raise BackendError("time_grid.steps must be a positive integer.")
    if steps == 1:
        return [float(start)]
    step = (stop - start) / float(steps - 1)
    return [start + step * index for index in range(steps)]


def _build_time_grid(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        times = _as_float_list(value, "time_grid")
        return _validate_time_grid(times)
    cfg = _require_mapping(value, "time_grid")
    if "points" in cfg and cfg.get("points") is not None:
        times = _as_float_list(cfg.get("points"), "time_grid.points")
        return _validate_time_grid(times)
    start = _as_float(cfg.get("start", 0.0), "time_grid.start")
    stop = _as_float(cfg.get("stop", start), "time_grid.stop")
    if cfg.get("dt") is not None:
        dt = _as_float(cfg.get("dt"), "time_grid.dt")
        if dt <= 0:
            raise BackendError("time_grid.dt must be positive.")
        steps = int((stop - start) / dt) + 1
        return _validate_time_grid([start + dt * idx for idx in range(steps)])
    steps = _as_int(cfg.get("steps", 1), "time_grid.steps")
    return _validate_time_grid(_linspace(start, stop, steps))


def _validate_time_grid(times: Sequence[float]) -> list[float]:
    if not times:
        raise BackendError("time_grid must contain at least one time.")
    output: list[float] = []
    last_time = None
    for time in times:
        if last_time is not None and time < last_time:
            raise BackendError("time_grid must be non-decreasing.")
        output.append(float(time))
        last_time = float(time)
    return output


def _normalize_energy(value: Any) -> str:
    if value is None:
        return "on"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"on", "true", "yes", "1"}:
            return "on"
        if lowered in {"off", "false", "no", "0"}:
            return "off"
    raise BackendError("reactor.energy must be on/off or a boolean.")


def _coerce_composition(value: Any) -> str | dict[str, float]:
    if isinstance(value, str):
        if not value.strip():
            raise BackendError("initial.X must be a non-empty string.")
        return value
    if isinstance(value, Mapping):
        composition: dict[str, float] = {}
        for name, amount in value.items():
            if not isinstance(name, str) or not name.strip():
                raise BackendError("initial.X keys must be non-empty strings.")
            composition[name] = _as_float(amount, f"initial.X[{name}]")
        if not composition:
            raise BackendError("initial.X must contain at least one species.")
        return composition
    raise BackendError("initial.X must be a composition string or mapping.")


def _resolve_mechanism(value: str) -> str:
    resolved = resolve_repo_path(value)
    if resolved.exists():
        return str(resolved)
    return value


def _reaction_labels(solution: Any) -> list[str]:
    count = int(getattr(solution, "n_reactions", 0) or 0)
    if count <= 0:
        return []
    try:
        equations = list(solution.reaction_equations())
        if len(equations) == count:
            return [str(entry) for entry in equations]
    except Exception:
        pass
    labels: list[str] = []
    for idx in range(count):
        try:
            labels.append(str(solution.reaction_equation(idx)))
        except Exception:
            labels.append(f"R{idx + 1}")
    return labels


def _apply_reaction_multipliers(
    solution: Any,
    multipliers: Sequence[Mapping[str, Any]],
    reaction_names: Sequence[str],
) -> list[dict[str, Any]]:
    if not multipliers:
        return []
    if not reaction_names:
        raise BackendError("No reactions available to apply multipliers.")
    if not hasattr(solution, "set_multiplier"):
        raise BackendError("Cantera solution does not support multipliers.")

    label_to_index: dict[str, int] = {}
    for idx, label in enumerate(reaction_names):
        if label in label_to_index:
            raise BackendError(
                "Duplicate reaction identifiers detected in mechanism."
            )
        label_to_index[label] = idx

    applied: dict[int, dict[str, Any]] = {}
    for entry in multipliers:
        if "index" in entry:
            idx = entry["index"]
            if not isinstance(idx, int):
                raise BackendError("reaction multiplier index must be an int.")
            if idx < 0 or idx >= len(reaction_names):
                raise BackendError(
                    f"reaction multiplier index out of range: {idx}."
                )
            reaction_id = reaction_names[idx]
        else:
            reaction_id = entry.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id.strip():
                raise BackendError(
                    "reaction multiplier reaction_id must be a non-empty string."
                )
            if reaction_id not in label_to_index:
                raise BackendError(
                    f"reaction_id not found in mechanism: {reaction_id!r}."
                )
            idx = label_to_index[reaction_id]
        try:
            multiplier = float(entry.get("multiplier"))
        except (TypeError, ValueError) as exc:
            raise BackendError(
                f"reaction multiplier must be a float for {reaction_id!r}."
            ) from exc
        if idx in applied:
            if applied[idx]["multiplier"] != multiplier:
                raise BackendError(
                    f"Conflicting multipliers for reaction index {idx}."
                )
            continue
        try:
            solution.set_multiplier(multiplier, idx)
        except Exception as exc:
            raise BackendError(
                f"Failed to set multiplier for reaction {reaction_id!r}."
            ) from exc
        applied[idx] = {
            "index": idx,
            "reaction_id": reaction_names[idx],
            "multiplier": multiplier,
        }
    return [applied[idx] for idx in sorted(applied)]


class CanteraBackend(SimulationBackend):
    """Run 0D Cantera reactor simulations."""

    name = "cantera"

    def run(self, cfg: Mapping[str, Any]) -> Any:
        if ct is None:
            raise BackendError(
                "Cantera is not installed; install cantera to use the cantera backend."
            )
        if not isinstance(cfg, Mapping):
            raise BackendError("CanteraBackend config must be a mapping.")

        mechanism = cfg.get("mechanism") or cfg.get("solution")
        mechanism = _as_str(mechanism, "mechanism")
        mechanism = _resolve_mechanism(mechanism)
        phase = cfg.get("phase")
        if phase is not None:
            phase = _as_str(phase, "phase")

        if phase is None:
            gas = ct.Solution(mechanism)
        else:
            gas = ct.Solution(mechanism, name=phase)

        initial_cfg = _require_mapping(cfg.get("initial"), "initial")
        temperature = _as_float(initial_cfg.get("T"), "initial.T")
        pressure = _as_float(initial_cfg.get("P"), "initial.P")
        composition = _coerce_composition(initial_cfg.get("X"))
        gas.TPX = temperature, pressure, composition

        try:
            multipliers = normalize_reaction_multipliers(cfg)
        except (TypeError, ValueError) as exc:
            raise BackendError(
                f"reaction multipliers are invalid: {exc}"
            ) from exc
        reaction_names = _reaction_labels(gas)
        applied_multipliers = _apply_reaction_multipliers(
            gas, multipliers, reaction_names
        )

        time_cfg = cfg.get("time_grid", cfg.get("time"))
        time_grid = _build_time_grid(time_cfg)

        reactor_cfg = _optional_mapping(cfg.get("reactor"), "reactor")
        reactor_type = reactor_cfg.get("type", cfg.get("reactor_type"))
        reactor_type = _as_str(
            reactor_type or "IdealGasConstPressureReactor",
            "reactor.type",
        )
        energy = _normalize_energy(reactor_cfg.get("energy", cfg.get("energy")))

        if reactor_type == "IdealGasConstPressureReactor":
            reactor = ct.IdealGasConstPressureReactor(gas, energy=energy)
        elif reactor_type == "IdealGasReactor":
            reactor = ct.IdealGasReactor(gas, energy=energy)
            volume = reactor_cfg.get("volume", cfg.get("volume"))
            if volume is not None:
                reactor.volume = _as_float(volume, "reactor.volume")
        else:
            raise BackendError(f"Unsupported reactor.type: {reactor_type!r}.")

        sim = ct.ReactorNet([reactor])
        solver_cfg = _optional_mapping(cfg.get("solver"), "solver")
        rtol = solver_cfg.get("rtol", cfg.get("rtol"))
        atol = solver_cfg.get("atol", cfg.get("atol"))
        max_steps = solver_cfg.get("max_steps", cfg.get("max_steps"))
        if rtol is not None:
            sim.rtol = _as_float(rtol, "solver.rtol")
        if atol is not None:
            sim.atol = _as_float(atol, "solver.atol")
        if max_steps is not None:
            sim.max_steps = _as_int(max_steps, "solver.max_steps")

        species_names = list(gas.species_names)
        outputs_cfg = _optional_mapping(cfg.get("outputs"), "outputs")
        include_rop = bool(
            outputs_cfg.get("include_rop", outputs_cfg.get("rop", True))
        )
        include_wdot = bool(
            outputs_cfg.get("include_wdot", outputs_cfg.get("wdot", True))
        )
        include_creation = bool(
            outputs_cfg.get(
                "include_creation", outputs_cfg.get("creation", True)
            )
        )
        include_destruction = bool(
            outputs_cfg.get(
                "include_destruction", outputs_cfg.get("destruction", True)
            )
        )
        times: list[float] = []
        temperatures: list[float] = []
        pressures: list[float] = []
        mole_fractions: list[list[float]] = []
        rop_net: list[list[float]] = []
        rop_net_ok = include_rop and bool(reaction_names)
        wdot: list[list[float]] = []
        wdot_ok = include_wdot
        creation_rates: list[list[float]] = []
        creation_ok = include_creation
        destruction_rates: list[list[float]] = []
        destruction_ok = include_destruction

        for target_time in time_grid:
            if target_time > sim.time:
                sim.advance(float(target_time))
            times.append(float(sim.time))
            temperatures.append(float(reactor.T))
            pressures.append(float(reactor.thermo.P))
            mole_fractions.append([float(x) for x in reactor.thermo.X])
            if wdot_ok:
                try:
                    wdot.append([float(x) for x in gas.net_production_rates])
                except Exception:
                    wdot_ok = False
            if rop_net_ok:
                try:
                    rop_net.append(
                        [float(x) for x in gas.net_rates_of_progress]
                    )
                except Exception:
                    rop_net_ok = False
            if creation_ok:
                try:
                    creation_rates.append(
                        [float(x) for x in gas.creation_rates]
                    )
                except Exception:
                    creation_ok = False
            if destruction_ok:
                try:
                    destruction_rates.append(
                        [float(x) for x in gas.destruction_rates]
                    )
                except Exception:
                    destruction_ok = False

        units = {
            "time": "s",
            "T": "K",
            "P": "Pa",
            "X": "mole_fraction",
        }
        if wdot_ok:
            units["net_production_rates"] = "kmol/m^3/s"
        if rop_net_ok:
            units["rop_net"] = "kmol/m^3/s"
        if creation_ok:
            units["creation_rates"] = "kmol/m^3/s"
        if destruction_ok:
            units["destruction_rates"] = "kmol/m^3/s"
        attrs = {
            "backend": self.name,
            "model": reactor_type,
            "mechanism": mechanism,
            "units": units,
            "cantera_version": ct.__version__,
        }
        if applied_multipliers:
            attrs["reaction_multipliers"] = applied_multipliers
        if phase is not None:
            attrs["phase"] = phase

        coords = {
            "time": {"dims": ["time"], "data": times},
            "species": {"dims": ["species"], "data": species_names},
        }
        if reaction_names:
            coords["reaction"] = {"dims": ["reaction"], "data": reaction_names}
        data_vars = {
            "T": {"dims": ["time"], "data": temperatures},
            "P": {"dims": ["time"], "data": pressures},
            "X": {"dims": ["time", "species"], "data": mole_fractions},
        }
        if wdot_ok:
            data_vars["net_production_rates"] = {
                "dims": ["time", "species"],
                "data": wdot,
            }
        if rop_net_ok:
            data_vars["rop_net"] = {
                "dims": ["time", "reaction"],
                "data": rop_net,
            }
        if creation_ok:
            data_vars["creation_rates"] = {
                "dims": ["time", "species"],
                "data": creation_rates,
            }
        if destruction_ok:
            data_vars["destruction_rates"] = {
                "dims": ["time", "species"],
                "data": destruction_rates,
            }

        if xr is not None and zarr is not None:
            xr_vars = {
                name: (tuple(payload["dims"]), payload["data"])
                for name, payload in data_vars.items()
            }
            xr_coords = {
                name: payload["data"] for name, payload in coords.items()
            }
            return xr.Dataset(data_vars=xr_vars, coords=xr_coords, attrs=attrs)

        return RunDataset(coords=coords, data_vars=data_vars, attrs=attrs)


_DEFAULT_BACKEND = CanteraBackend()
register("backend", _DEFAULT_BACKEND.name, _DEFAULT_BACKEND)

__all__ = ["CanteraBackend"]
