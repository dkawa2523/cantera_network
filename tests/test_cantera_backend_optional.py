import pytest

ct = pytest.importorskip("cantera")

import rxn_platform.backends.cantera  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.backends.base import RunDataset
from rxn_platform.registry import get, resolve_backend
from rxn_platform.store import ArtifactStore
from rxn_platform.validators import validate_run_artifact


def _coord_values(dataset: object, name: str) -> list[float]:
    if isinstance(dataset, RunDataset):
        return list(dataset.coords[name]["data"])
    return dataset.coords[name].values.tolist()


def _var_values(dataset: object, name: str) -> list[list[float]] | list[float]:
    if isinstance(dataset, RunDataset):
        return dataset.data_vars[name]["data"]
    return dataset[name].values.tolist()


def test_cantera_backend_runartifact(tmp_path) -> None:
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
    temperatures = _var_values(dataset, "T")
    pressures = _var_values(dataset, "P")
    mole_fractions = _var_values(dataset, "X")

    assert len(times) == 3
    assert len(species) > 0
    assert len(temperatures) == len(times)
    assert len(pressures) == len(times)
    assert len(mole_fractions) == len(times)
    assert len(mole_fractions[0]) == len(species)

    store = ArtifactStore(tmp_path / "artifacts")
    task = get("task", "sim.run")
    result = task({"sim": dict(cfg, name="cantera")}, store=store)
    validate_run_artifact(result.path)
