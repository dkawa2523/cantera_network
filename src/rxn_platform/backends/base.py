"""Backend interfaces and run dataset helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:  # Optional dependency.
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None


@dataclass(frozen=True)
class RunDataset:
    """Lightweight, JSON-serializable dataset placeholder."""

    coords: Mapping[str, Any]
    data_vars: Mapping[str, Any]
    attrs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": dict(self.coords),
            "data_vars": dict(self.data_vars),
            "attrs": dict(self.attrs),
        }


class SimulationBackend(ABC):
    """Base class for simulation backends."""

    name: str

    @abstractmethod
    def run(self, cfg: Mapping[str, Any]) -> Any:
        """Execute the simulation and return a dataset-like object."""


def dump_run_dataset(dataset: Any, path: Path) -> None:
    """Persist a run dataset to disk."""
    if xr is not None and isinstance(dataset, xr.Dataset):
        dataset.to_zarr(path, mode="w")
        return
    if isinstance(dataset, RunDataset):
        path.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            dataset.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        (path / "dataset.json").write_text(payload + "\n", encoding="utf-8")
        return
    raise TypeError(
        "Unsupported run dataset type; expected RunDataset or xarray.Dataset."
    )


__all__ = ["RunDataset", "SimulationBackend", "dump_run_dataset"]
