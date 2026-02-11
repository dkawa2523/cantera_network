"""Deterministic dummy backend for fast simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rxn_platform.backends.base import RunDataset, SimulationBackend
from rxn_platform.errors import BackendError
from rxn_platform.registry import register

_DEFAULT_SPECIES = ("A", "B", "C")
_DEFAULT_REACTIONS = ("R1", "R2")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackendError(f"{label} must be a mapping, got {type(value)!r}.")
    return value


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"{label} must be a float, got {value!r}.") from exc


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"{label} must be an int, got {value!r}.") from exc


def _as_str_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise BackendError(f"{label} must be a sequence of strings.")
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise BackendError(f"{label} entries must be non-empty strings.")
        items.append(item)
    if not items:
        raise BackendError(f"{label} must contain at least one entry.")
    return items


def _linspace(start: float, stop: float, steps: int) -> list[float]:
    if steps <= 0:
        raise BackendError("time.steps must be a positive integer.")
    if steps == 1:
        return [float(start)]
    step = (stop - start) / float(steps - 1)
    return [start + step * index for index in range(steps)]


def _build_mole_fractions(steps: int, n_species: int) -> list[list[float]]:
    fractions: list[list[float]] = []
    for t_index in range(steps):
        raw = [1.0 + (idx + 1) * 0.01 * t_index for idx in range(n_species)]
        total = sum(raw)
        fractions.append([value / total for value in raw])
    return fractions


class DummyBackend(SimulationBackend):
    """Generate deterministic time series data without external dependencies."""

    name = "dummy"

    def run(self, cfg: Mapping[str, Any]) -> RunDataset:
        if not isinstance(cfg, Mapping):
            raise BackendError("DummyBackend config must be a mapping.")

        time_cfg = _as_mapping(cfg.get("time"), "time")
        start = _as_float(time_cfg.get("start", 0.0), "time.start")
        stop = _as_float(time_cfg.get("stop", 1.0), "time.stop")
        steps = _as_int(time_cfg.get("steps", 5), "time.steps")
        time = _linspace(start, stop, steps)

        initial_cfg = _as_mapping(cfg.get("initial"), "initial")
        ramp_cfg = _as_mapping(cfg.get("ramp"), "ramp")
        temperature_0 = _as_float(initial_cfg.get("T", 1000.0), "initial.T")
        pressure_0 = _as_float(initial_cfg.get("P", 101325.0), "initial.P")
        temperature_step = _as_float(ramp_cfg.get("T", 0.0), "ramp.T")
        pressure_step = _as_float(ramp_cfg.get("P", 0.0), "ramp.P")

        species = cfg.get("species", _DEFAULT_SPECIES)
        species_list = _as_str_list(species, "species")
        mole_fractions = _build_mole_fractions(steps, len(species_list))

        temperatures = [
            temperature_0 + temperature_step * index for index in range(steps)
        ]
        pressures = [
            pressure_0 + pressure_step * index for index in range(steps)
        ]

        coords: dict[str, Any] = {
            "time": {"dims": ["time"], "data": time},
            "species": {"dims": ["species"], "data": species_list},
        }
        data_vars: dict[str, Any] = {
            "T": {"dims": ["time"], "data": temperatures},
            "P": {"dims": ["time"], "data": pressures},
            "X": {"dims": ["time", "species"], "data": mole_fractions},
        }

        outputs_cfg = _as_mapping(cfg.get("outputs"), "outputs")
        include_rop = bool(
            outputs_cfg.get("include_rop", outputs_cfg.get("rop", False))
        )
        include_wdot = bool(
            outputs_cfg.get("include_wdot", outputs_cfg.get("wdot", False))
        )

        units = {
            "time": "s",
            "T": "K",
            "P": "Pa",
            "X": "mole_fraction",
        }

        if include_rop:
            reactions = cfg.get("reactions", _DEFAULT_REACTIONS)
            reaction_list = _as_str_list(reactions, "reactions")
            coords["reaction"] = {"dims": ["reaction"], "data": reaction_list}
            rop = [
                [(idx + 1) * 0.01 * (t + 1) for idx in range(len(reaction_list))]
                for t in range(steps)
            ]
            data_vars["rop_net"] = {"dims": ["time", "reaction"], "data": rop}
            units["rop_net"] = "arb"

        if include_wdot:
            wdot = [
                [(idx + 1) * 1.0e-3 * (t + 1) for idx in range(len(species_list))]
                for t in range(steps)
            ]
            data_vars["net_production_rates"] = {
                "dims": ["time", "species"],
                "data": wdot,
            }
            units["net_production_rates"] = "arb"

        attrs = {
            "backend": self.name,
            "model": "dummy",
            "units": units,
        }
        multipliers = cfg.get("reaction_multipliers")
        if multipliers:
            attrs["reaction_multipliers"] = multipliers
        return RunDataset(coords=coords, data_vars=data_vars, attrs=attrs)


_DEFAULT_BACKEND = DummyBackend()
register("backend", _DEFAULT_BACKEND.name, _DEFAULT_BACKEND)

__all__ = ["DummyBackend"]
