import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.backends.cantera  # noqa: F401

from rxn_platform.backends.base import RunDataset
from rxn_platform.registry import resolve_backend


def _coord_values(dataset: object, name: str) -> list[float] | list[str]:
    if isinstance(dataset, RunDataset):
        return list(dataset.coords[name]["data"])
    return dataset.coords[name].values.tolist()


def _var_values(dataset: object, name: str) -> list[list[float]] | list[float]:
    if isinstance(dataset, RunDataset):
        return dataset.data_vars[name]["data"]
    return dataset[name].values.tolist()


def _has_var(dataset: object, name: str) -> bool:
    if isinstance(dataset, RunDataset):
        return name in dataset.data_vars
    return name in dataset.data_vars


def test_cantera_outputs_include_rates() -> None:
    backend = resolve_backend("cantera")
    cfg = {
        "mechanism": "gri30.yaml",
        "phase": "gas",
        "initial": {"T": 1000.0, "P": 101325.0, "X": {"CH4": 1.0, "O2": 2.0, "N2": 7.52}},
        "time_grid": {"start": 0.0, "stop": 1.0e-4, "steps": 3},
        "reactor": {"type": "IdealGasConstPressureReactor", "energy": "on"},
        "solver": {"rtol": 1.0e-9, "atol": 1.0e-15, "max_steps": 100000},
    }

    dataset = backend.run(cfg)

    times = _coord_values(dataset, "time")
    species = _coord_values(dataset, "species")
    reactions = _coord_values(dataset, "reaction")

    wdot = _var_values(dataset, "net_production_rates")
    rop_net = _var_values(dataset, "rop_net")

    assert len(times) == 3
    assert len(species) > 0
    assert len(reactions) > 0
    assert len(wdot) == len(times)
    assert len(wdot[0]) == len(species)
    assert len(rop_net) == len(times)
    assert len(rop_net[0]) == len(reactions)

    if _has_var(dataset, "creation_rates"):
        creation = _var_values(dataset, "creation_rates")
        assert len(creation) == len(times)
        assert len(creation[0]) == len(species)

    if _has_var(dataset, "destruction_rates"):
        destruction = _var_values(dataset, "destruction_rates")
        assert len(destruction) == len(times)
        assert len(destruction[0]) == len(species)
