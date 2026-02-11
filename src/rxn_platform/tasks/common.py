"""Shared task helpers to minimize duplication."""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable, Optional

from rxn_platform.core import ArtifactManifest
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.io_utils import read_json
from rxn_platform.metadata import code_metadata, provenance_metadata
from rxn_platform.run_store import resolve_run_dataset_dir, utc_now_iso

try:  # Optional dependency.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:  # Optional dependency.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None


def resolve_cfg(cfg: Any) -> dict[str, Any]:
    try:
        resolved = resolve_config(cfg)
    except (ConfigError, TypeError, ValueError):
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise
    return resolved


def build_manifest(
    *,
    kind: str,
    artifact_id: str,
    config: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    parents: Optional[Iterable[str]] = None,
    notes: Optional[str] = None,
    created_at: Optional[str] = None,
    code: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind=kind,
        id=artifact_id,
        created_at=created_at or utc_now_iso(),
        parents=list(parents) if parents is not None else [],
        inputs=dict(inputs) if inputs is not None else {},
        config=dict(config) if config is not None else {},
        code=dict(code) if code is not None else code_metadata(),
        provenance=dict(provenance)
        if provenance is not None
        else provenance_metadata(),
        notes=notes,
    )


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = read_json(path)
    except JSONDecodeError as exc:
        raise ConfigError(f"Failed to parse JSON from {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return list(rows)
        return []
    if isinstance(payload, list):
        return list(payload)
    return []


def read_table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"table not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        parquet_error: Optional[Exception] = None
        if pd is not None:
            try:
                frame = pd.read_parquet(path)
                return frame.to_dict(orient="records")
            except Exception as exc:
                parquet_error = exc
        if pq is not None:
            try:
                table = pq.read_table(path)
                return table.to_pylist()
            except Exception as exc:
                parquet_error = exc
        json_path = path.with_suffix(".json")
        if json_path.exists():
            return _read_json_rows(json_path)
        try:
            return _read_json_rows(path)
        except ConfigError as exc:
            if parquet_error is not None:
                raise ConfigError(
                    f"Failed to read parquet table from {path}: {parquet_error}"
                ) from parquet_error
            raise ConfigError(
                f"Parquet support is required to read {path} (install pandas or pyarrow)."
            ) from exc
    return _read_json_rows(path)


def load_run_dataset_payload(
    run_dir: Path,
    *,
    dataset_dir: Optional[Path] = None,
    missing_message: str = "Run dataset not found; expected sim/timeseries.zarr or state.zarr.",
    xarray_missing_message: Optional[str] = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir or resolve_run_dataset_dir(run_dir)
    if dataset_dir is None:
        raise ArtifactError(missing_message)
    dataset_path = dataset_dir / "dataset.json"
    if dataset_path.exists():
        try:
            payload = read_json(dataset_path)
        except JSONDecodeError as exc:
            raise ArtifactError(f"Run dataset JSON is invalid: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("Run dataset JSON must be a mapping.")
        return dict(payload)
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:
        if xarray_missing_message is None:
            raise ArtifactError(
                "Run dataset not found; install xarray to load timeseries.zarr."
            ) from exc
        message = xarray_missing_message.format(
            dataset_path=dataset_path,
            error=exc,
        )
        raise ArtifactError(message) from exc
    dataset = xr.open_zarr(dataset_dir)
    coords = {
        name: {"dims": [name], "data": dataset.coords[name].values.tolist()}
        for name in dataset.coords
    }
    data_vars = {
        name: {"dims": list(dataset[name].dims), "data": dataset[name].values.tolist()}
        for name in dataset.data_vars
    }
    return {"coords": coords, "data_vars": data_vars, "attrs": dict(dataset.attrs)}

def load_run_ids_from_run_set(store: Any, run_set_id: str) -> list[str]:
    """Load run_ids from a run_sets artifact.

    This keeps multi-condition pipelines loosely coupled by passing a single
    `run_set_id` between tasks instead of duplicating run_id lists.
    """
    if not isinstance(run_set_id, str) or not run_set_id.strip():
        raise ConfigError("run_set_id must be a non-empty string.")
    run_set_id = run_set_id.strip()
    # Deliberately avoid importing ArtifactStore for type-check; tasks pass the store.
    manifest = store.read_manifest("run_sets", run_set_id)
    inputs = getattr(manifest, "inputs", None)
    if not isinstance(inputs, Mapping):
        raise ConfigError("run_set manifest.inputs must be a mapping.")
    run_ids = inputs.get("run_ids")
    if not isinstance(run_ids, list):
        raise ConfigError("run_set manifest.inputs.run_ids must be a list.")
    cleaned: list[str] = []
    for idx, entry in enumerate(run_ids):
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError(
                f"run_set manifest.inputs.run_ids[{idx}] must be a non-empty string."
            )
        cleaned.append(entry.strip())
    if not cleaned:
        raise ConfigError("run_set manifest.inputs.run_ids must not be empty.")
    return cleaned


__all__ = [
    "build_manifest",
    "code_metadata",
    "load_run_dataset_payload",
    "load_run_ids_from_run_set",
    "provenance_metadata",
    "read_table_rows",
    "resolve_cfg",
]
