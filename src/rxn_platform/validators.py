"""Artifact contract validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Optional

from rxn_platform.core import ArtifactManifest, load_manifest
from rxn_platform.errors import ValidationError
from rxn_platform.io_utils import read_json
from rxn_platform.run_store import resolve_run_dataset_dir

try:  # Optional dependency.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:  # Optional dependency.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None

try:  # Optional dependency.
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None

RUN_STATE_DIRNAME = "state.zarr"


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _raise_missing(label: str, missing: Sequence[str], *, path: Path) -> None:
    if not missing:
        return
    missing_list = ", ".join(missing)
    raise ValidationError(
        f"{label} validation failed at {path}: missing {missing_list}"
    )


def _load_manifest_for_dir(
    artifact_dir: Path,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]],
    missing: list[str],
) -> Optional[ArtifactManifest]:
    manifest_path = artifact_dir / "manifest.yaml"
    if not manifest_path.exists():
        missing.append("manifest.yaml")
        return None
    if manifest is None:
        try:
            return load_manifest(manifest_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ValidationError(
                f"Invalid manifest at {manifest_path}: {exc}"
            ) from exc
    if isinstance(manifest, ArtifactManifest):
        return manifest
    if isinstance(manifest, Mapping):
        try:
            return ArtifactManifest.from_dict(manifest)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid manifest payload: {exc}") from exc
    raise ValidationError("manifest must be an ArtifactManifest or mapping.")


def _validate_manifest_kind(
    manifest: Optional[ArtifactManifest],
    expected_kind: str,
    missing: list[str],
) -> None:
    if manifest is None:
        return
    if manifest.schema_version < 1:
        missing.append("manifest.schema_version>=1")
    if manifest.kind != expected_kind:
        missing.append(f"manifest.kind:{expected_kind}")


def _read_parquet_columns(path: Path) -> Optional[list[str]]:
    if pq is not None:
        try:
            return list(pq.read_schema(path).names)
        except Exception:
            return None
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return [str(col) for col in frame.columns]
        except Exception:
            return None
    return None


def _resolve_table_columns(
    path: Path,
    table_columns: Optional[Sequence[str]],
    missing: list[str],
) -> Optional[list[str]]:
    if table_columns is not None:
        return [str(col) for col in table_columns]
    columns = _read_parquet_columns(path)
    if columns is None:
        if pq is None and pd is None:
            missing.append(
                f"{path.name} columns (install pandas/pyarrow or provide columns)"
            )
        else:
            missing.append(f"{path.name} columns (parquet read failed)")
        return None
    return columns


def _read_run_dataset_metadata(
    dataset_dir: Path,
    missing: list[str],
) -> tuple[Optional[set[str]], Optional[Mapping[str, Any]]]:
    dataset_label = dataset_dir.name
    dataset_json = dataset_dir / "dataset.json"
    if dataset_json.exists():
        try:
            payload = read_json(dataset_json)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{dataset_label}/dataset.json is not valid JSON: {exc}"
            ) from exc
        coords = payload.get("coords")
        attrs = payload.get("attrs")
        if not isinstance(coords, Mapping):
            missing.append(f"{dataset_label}/dataset.json.coords")
            return None, None
        if not isinstance(attrs, Mapping):
            missing.append(f"{dataset_label}/dataset.json.attrs")
            return None, None
        return set(coords.keys()), attrs
    if xr is not None:
        try:
            dataset = xr.open_zarr(dataset_dir)
        except Exception as exc:
            raise ValidationError(
                f"{dataset_label} could not be opened with xarray: {exc}"
            ) from exc
        return set(dataset.coords.keys()), dataset.attrs
    missing.append(f"{dataset_label} dataset metadata")
    return None, None


def validate_run_artifact(
    artifact_dir: str | Path,
    *,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]] = None,
) -> None:
    """Validate a RunArtifact stored under artifact_dir."""
    path = _as_path(artifact_dir)
    missing: list[str] = []
    manifest_obj = _load_manifest_for_dir(path, manifest, missing)
    _validate_manifest_kind(manifest_obj, "runs", missing)

    dataset_dir = resolve_run_dataset_dir(path)
    if dataset_dir is None:
        missing.append("sim/timeseries.zarr|state.zarr")
    else:
        coords, attrs = _read_run_dataset_metadata(dataset_dir, missing)
        if coords is not None:
            if "time" not in coords:
                missing.append("coords.time")
            if not {"species", "surface_species"} & coords:
                missing.append("coords.species|surface_species")
        if attrs is not None:
            if "units" not in attrs:
                missing.append("attrs.units")
            if "model" not in attrs and "backend" not in attrs:
                missing.append("attrs.model_or_backend")

    _raise_missing("RunArtifact", missing, path=path)


def validate_observable_artifact(
    artifact_dir: str | Path,
    *,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]] = None,
    table_columns: Optional[Sequence[str]] = None,
) -> None:
    """Validate an ObservableArtifact stored under artifact_dir."""
    path = _as_path(artifact_dir)
    missing: list[str] = []
    manifest_obj = _load_manifest_for_dir(path, manifest, missing)
    _validate_manifest_kind(manifest_obj, "observables", missing)

    values_path = path / "values.parquet"
    if not values_path.exists():
        missing.append("values.parquet")
    else:
        columns = _resolve_table_columns(values_path, table_columns, missing)
        if columns is not None:
            required = ("run_id", "observable", "value", "unit", "meta_json")
            for col in required:
                if col not in columns:
                    missing.append(f"values.parquet:{col}")

    _raise_missing("ObservableArtifact", missing, path=path)


def validate_graph_artifact(
    artifact_dir: str | Path,
    *,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]] = None,
) -> None:
    """Validate a GraphArtifact stored under artifact_dir."""
    path = _as_path(artifact_dir)
    missing: list[str] = []
    manifest_obj = _load_manifest_for_dir(path, manifest, missing)
    _validate_manifest_kind(manifest_obj, "graphs", missing)

    graph_path = path / "graph.json"
    stoich_path = path / "stoich.npz"
    if not graph_path.exists() and not stoich_path.exists():
        missing.append("graph.json|stoich.npz")
    if graph_path.exists():
        try:
            read_json(graph_path)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"graph.json is not valid JSON: {exc}"
            ) from exc

    _raise_missing("GraphArtifact", missing, path=path)


def validate_feature_artifact(
    artifact_dir: str | Path,
    *,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]] = None,
    table_columns: Optional[Sequence[str]] = None,
) -> None:
    """Validate a FeatureArtifact stored under artifact_dir."""
    path = _as_path(artifact_dir)
    missing: list[str] = []
    manifest_obj = _load_manifest_for_dir(path, manifest, missing)
    _validate_manifest_kind(manifest_obj, "features", missing)

    features_path = path / "features.parquet"
    if not features_path.exists():
        missing.append("features.parquet")
    else:
        columns = _resolve_table_columns(
            features_path, table_columns, missing
        )
        if columns is not None:
            if "value" not in columns:
                missing.append("features.parquet:value")
            identifier_fields = {
                "run_id",
                "condition_id",
                "time_window",
                "target",
                "reaction_id",
                "species",
            }
            if not identifier_fields.intersection(columns):
                missing.append(
                    "features.parquet:identifier(run_id|condition_id|"
                    "time_window|target|reaction_id|species)"
                )

    _raise_missing("FeatureArtifact", missing, path=path)


def validate_sensitivity_artifact(
    artifact_dir: str | Path,
    *,
    manifest: Optional[ArtifactManifest | Mapping[str, Any]] = None,
    table_columns: Optional[Sequence[str]] = None,
) -> None:
    """Validate a SensitivityArtifact stored under artifact_dir."""
    path = _as_path(artifact_dir)
    missing: list[str] = []
    manifest_obj = _load_manifest_for_dir(path, manifest, missing)
    _validate_manifest_kind(manifest_obj, "sensitivity", missing)

    sensitivity_path = path / "sensitivity.parquet"
    if not sensitivity_path.exists():
        missing.append("sensitivity.parquet")
    else:
        columns = _resolve_table_columns(
            sensitivity_path, table_columns, missing
        )
        if columns is not None:
            if "value" not in columns:
                missing.append("sensitivity.parquet:value")
            identifier_fields = {
                "run_id",
                "condition_id",
                "time_window",
                "target",
                "reaction_id",
                "species",
            }
            if not identifier_fields.intersection(columns):
                missing.append(
                    "sensitivity.parquet:identifier(run_id|condition_id|"
                    "time_window|target|reaction_id|species)"
                )

    _raise_missing("SensitivityArtifact", missing, path=path)


__all__ = [
    "validate_run_artifact",
    "validate_observable_artifact",
    "validate_graph_artifact",
    "validate_feature_artifact",
    "validate_sensitivity_artifact",
]
