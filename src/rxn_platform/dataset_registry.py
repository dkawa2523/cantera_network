"""Dataset registry helpers for condition + time-series datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from rxn_platform.core import resolve_repo_path, stable_hash
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.run_store import utc_now_iso

DATASET_REGISTRY_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
DATASET_REGISTRY_NAME = "registry.json"


def resolve_dataset_registry_path(root: Optional[str | Path] = None) -> Path:
    if root is None:
        base = resolve_repo_path("datasets")
        return base / DATASET_REGISTRY_NAME
    base = Path(root)
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    if base.suffix == ".json":
        return base
    return base / DATASET_REGISTRY_NAME


def load_dataset_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": DATASET_REGISTRY_SCHEMA_VERSION, "datasets": []}
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset registry is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Dataset registry must be a JSON object.")
    registry = dict(payload)
    version = registry.get("schema_version", DATASET_REGISTRY_SCHEMA_VERSION)
    if version != DATASET_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset registry schema_version: {version}")
    datasets = registry.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("Dataset registry 'datasets' must be a list.")
    registry["schema_version"] = version
    registry["datasets"] = datasets
    return registry


def save_dataset_registry(path: Path, registry: Mapping[str, Any]) -> None:
    payload = dict(registry)
    payload["schema_version"] = DATASET_REGISTRY_SCHEMA_VERSION
    payload["updated_at"] = utc_now_iso()
    write_json_atomic(path, payload)


def compute_dataset_id(
    *,
    conditions_hash: str,
    mechanism_hash: str,
    schema_version: int = DATASET_SCHEMA_VERSION,
) -> str:
    if not isinstance(conditions_hash, str) or not conditions_hash.strip():
        raise ValueError("conditions_hash must be a non-empty string.")
    if not isinstance(mechanism_hash, str) or not mechanism_hash.strip():
        raise ValueError("mechanism_hash must be a non-empty string.")
    payload = {
        "conditions_hash": conditions_hash,
        "mechanism_hash": mechanism_hash,
        "schema_version": schema_version,
    }
    return stable_hash(payload, length=16)


def _match_dataset_entry(
    entry: Mapping[str, Any],
    *,
    conditions_hash: str,
    mechanism_hash: str,
    schema_version: int,
    dataset_id: str,
) -> bool:
    if entry.get("dataset_id") == dataset_id:
        return True
    if (
        entry.get("conditions_hash") == conditions_hash
        and entry.get("mechanism_hash") == mechanism_hash
        and entry.get("schema_version", schema_version) == schema_version
    ):
        return True
    return False


def _append_run_ref(entry: dict[str, Any], run_ref: Mapping[str, str]) -> bool:
    runs = entry.get("runs", [])
    if runs is None:
        runs = []
    if not isinstance(runs, list):
        raise ValueError("Dataset entry 'runs' must be a list.")
    run_id = run_ref.get("run_id")
    exp = run_ref.get("exp")
    for existing in runs:
        if not isinstance(existing, Mapping):
            continue
        if existing.get("run_id") == run_id and existing.get("exp") == exp:
            entry["runs"] = runs
            return False
    runs.append(dict(run_ref))
    entry["runs"] = runs
    return True


def register_dataset_entry(
    registry: dict[str, Any],
    *,
    conditions_hash: str,
    mechanism_hash: str,
    run_ref: Mapping[str, str],
    schema_version: int = DATASET_SCHEMA_VERSION,
) -> tuple[dict[str, Any], bool, bool]:
    dataset_id = compute_dataset_id(
        conditions_hash=conditions_hash,
        mechanism_hash=mechanism_hash,
        schema_version=schema_version,
    )
    datasets = registry.get("datasets", [])
    if datasets is None:
        datasets = []
    if not isinstance(datasets, list):
        raise ValueError("Dataset registry 'datasets' must be a list.")

    entry_idx: Optional[int] = None
    entry_payload: Optional[dict[str, Any]] = None
    for idx, entry in enumerate(datasets):
        if not isinstance(entry, Mapping):
            continue
        if _match_dataset_entry(
            entry,
            conditions_hash=conditions_hash,
            mechanism_hash=mechanism_hash,
            schema_version=schema_version,
            dataset_id=dataset_id,
        ):
            entry_idx = idx
            entry_payload = dict(entry)
            break

    reused = entry_payload is not None
    if not reused:
        entry_payload = {
            "dataset_id": dataset_id,
            "schema_version": schema_version,
            "created_at": utc_now_iso(),
            "conditions_hash": conditions_hash,
            "mechanism_hash": mechanism_hash,
            "runs": [],
        }
        datasets.append(entry_payload)
    else:
        if entry_payload is None:
            raise ValueError("Matched dataset entry is missing.")
        existing_id = entry_payload.get("dataset_id")
        if existing_id and existing_id != dataset_id:
            raise ValueError("Dataset entry id does not match computed id.")
        entry_payload["dataset_id"] = dataset_id
        entry_payload.setdefault("schema_version", schema_version)
        entry_payload.setdefault("conditions_hash", conditions_hash)
        entry_payload.setdefault("mechanism_hash", mechanism_hash)
        entry_payload.setdefault("runs", [])
        datasets[entry_idx] = entry_payload

    run_added = _append_run_ref(entry_payload, run_ref)
    registry["datasets"] = datasets
    return entry_payload, reused, run_added


__all__ = [
    "DATASET_REGISTRY_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "resolve_dataset_registry_path",
    "load_dataset_registry",
    "save_dataset_registry",
    "compute_dataset_id",
    "register_dataset_entry",
]
