"""Core schema and manifest I/O helpers."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import read_yaml_payload, write_yaml_payload

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from pydantic import BaseModel
    try:
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - pydantic v1
        ConfigDict = None
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None
    ConfigDict = None


_ALLOWED_FIELDS = {
    "schema_version",
    "kind",
    "id",
    "created_at",
    "parents",
    "inputs",
    "config",
    "code",
    "provenance",
    "notes",
}
_REQUIRED_FIELDS = _ALLOWED_FIELDS - {"notes"}
_MAPPING_FIELDS = ("inputs", "config", "code", "provenance")


def _require_nonempty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Manifest field '{field_name}' must be a non-empty string.")


def _validate_manifest_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("Manifest must be a mapping.")

    normalized = dict(data)
    keys = set(normalized.keys())
    unknown = sorted(keys - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"Unknown fields in manifest: {unknown}.")

    missing = sorted(_REQUIRED_FIELDS - keys)
    if missing:
        raise ValueError(f"Missing required fields in manifest: {missing}.")

    schema_version = normalized["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise TypeError("Manifest field 'schema_version' must be an integer.")
    if schema_version < 1:
        raise ValueError("Manifest field 'schema_version' must be >= 1.")

    _require_nonempty_str(normalized["kind"], "kind")
    _require_nonempty_str(normalized["id"], "id")
    _require_nonempty_str(normalized["created_at"], "created_at")

    parents = normalized["parents"]
    if isinstance(parents, tuple):
        parents = list(parents)
        normalized["parents"] = parents
    if not isinstance(parents, list):
        raise TypeError("Manifest field 'parents' must be a list of strings.")
    for index, parent in enumerate(parents):
        if not isinstance(parent, str):
            raise TypeError(
                f"Manifest field 'parents' item {index} must be a string."
            )

    for field_name in _MAPPING_FIELDS:
        value = normalized[field_name]
        if isinstance(value, Mapping) and not isinstance(value, dict):
            normalized[field_name] = dict(value)
            value = normalized[field_name]
        if not isinstance(value, dict):
            raise TypeError(f"Manifest field '{field_name}' must be a mapping.")

    notes = normalized.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise TypeError("Manifest field 'notes' must be a string or null.")

    return normalized


if BaseModel is not None:

    class ArtifactManifest(BaseModel):
        schema_version: int
        kind: str
        id: str
        created_at: str
        parents: List[str]
        inputs: Dict[str, Any]
        config: Dict[str, Any]
        code: Dict[str, Any]
        provenance: Dict[str, Any]
        notes: Optional[str] = None

        if ConfigDict is not None:
            model_config = ConfigDict(extra="forbid")
        else:
            class Config:
                extra = "forbid"

        @classmethod
        def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
            normalized = _validate_manifest_dict(data)
            return cls(**normalized)

        def to_dict(self) -> Dict[str, Any]:
            if hasattr(self, "model_dump"):
                data = self.model_dump()
            else:
                data = self.dict()
            if data.get("notes") is None:
                data.pop("notes", None)
            return data

else:

    @dataclass
    class ArtifactManifest:
        schema_version: int
        kind: str
        id: str
        created_at: str
        parents: List[str]
        inputs: Dict[str, Any]
        config: Dict[str, Any]
        code: Dict[str, Any]
        provenance: Dict[str, Any]
        notes: Optional[str] = None

        def __post_init__(self) -> None:
            normalized = _validate_manifest_dict(self.__dict__)
            for field_name, value in normalized.items():
                setattr(self, field_name, value)

        @classmethod
        def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
            normalized = _validate_manifest_dict(data)
            return cls(**normalized)

        def to_dict(self) -> Dict[str, Any]:
            data = asdict(self)
            if data.get("notes") is None:
                data.pop("notes", None)
            return data


def find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Return the repo root based on pyproject.toml or .git, if found."""
    base = start or Path.cwd()
    try:
        base = base.resolve()
    except OSError:
        base = base.absolute()
    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to cwd or repo root when possible."""
    target = Path(path)
    if target.is_absolute():
        return target
    cwd_candidate = (Path.cwd() / target).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    root = find_repo_root()
    if root is not None:
        root_candidate = (root / target).resolve()
        if root_candidate.exists():
            return root_candidate
    return target


def load_config(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    try:
        return read_yaml_payload(
            path,
            error_message=(
                "PyYAML is not available; manifest must be JSON-compatible YAML."
            ),
            error_cls=ValueError,
        )
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc


def load_manifest(path: Union[str, Path]) -> ArtifactManifest:
    path = Path(path)
    payload = read_yaml_payload(
        path,
        error_message="PyYAML is not available; manifest must be JSON-compatible YAML.",
        error_cls=ValueError,
    )
    try:
        return ArtifactManifest.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manifest in {path}: {exc}") from exc


def dump_manifest(path: Union[str, Path], manifest: ArtifactManifest) -> None:
    if not hasattr(manifest, "to_dict"):
        raise TypeError("manifest must be an ArtifactManifest instance.")
    payload = manifest.to_dict()
    normalized = _validate_manifest_dict(payload)
    if normalized.get("notes") is None:
        normalized.pop("notes", None)
    write_yaml_payload(Path(path), normalized, sort_keys=True)


def _coerce_multiplier_value(
    value: Any,
    label: str,
    *,
    default_multiplier: Optional[float],
    enforce_default: bool,
) -> float:
    if value is None:
        if default_multiplier is None:
            raise ValueError(f"{label} must include multiplier.")
        value = default_multiplier
    if isinstance(value, bool):
        raise TypeError(f"{label} multiplier must be a float, got {value!r}.")
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label} multiplier must be a float, got {value!r}."
        ) from exc
    if (
        enforce_default
        and default_multiplier is not None
        and multiplier != float(default_multiplier)
    ):
        raise ValueError(
            f"{label} multiplier must be {default_multiplier}, got {multiplier}."
        )
    return multiplier


def _coerce_reaction_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} reaction_id must be a non-empty string.")
    return value.strip()


def _coerce_reaction_index(value: Any, label: str) -> int:
    if value is None or isinstance(value, bool):
        raise TypeError(f"{label} index must be an integer.")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} index must be an integer.") from exc
    if index < 0:
        raise ValueError(f"{label} index must be >= 0, got {index}.")
    return index


def _parse_multiplier_item(
    entry: Any,
    label: str,
    *,
    default_multiplier: Optional[float],
    enforce_default: bool,
) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        index = entry.get("index")
        reaction_id = entry.get("reaction_id") or entry.get("reaction")
        if index is not None and reaction_id is not None:
            raise ValueError(f"{label} must set index or reaction_id, not both.")
        if index is None and reaction_id is None:
            raise ValueError(f"{label} must include index or reaction_id.")
        multiplier = _coerce_multiplier_value(
            entry.get("multiplier"),
            label,
            default_multiplier=default_multiplier,
            enforce_default=enforce_default,
        )
        if index is not None:
            return {
                "index": _coerce_reaction_index(index, f"{label}.index"),
                "multiplier": multiplier,
            }
        return {
            "reaction_id": _coerce_reaction_id(
                reaction_id,
                f"{label}.reaction_id",
            ),
            "multiplier": multiplier,
        }
    if isinstance(entry, str):
        multiplier = _coerce_multiplier_value(
            None,
            label,
            default_multiplier=default_multiplier,
            enforce_default=enforce_default,
        )
        return {
            "reaction_id": _coerce_reaction_id(entry, label),
            "multiplier": multiplier,
        }
    if isinstance(entry, bool):
        raise TypeError(f"{label} entry must be an index or reaction_id.")
    if isinstance(entry, int):
        multiplier = _coerce_multiplier_value(
            None,
            label,
            default_multiplier=default_multiplier,
            enforce_default=enforce_default,
        )
        return {
            "index": _coerce_reaction_index(entry, label),
            "multiplier": multiplier,
        }
    raise TypeError(f"{label} entry must be a mapping, index, or reaction_id.")


def _parse_multiplier_entries(
    value: Any,
    *,
    label: str,
    default_multiplier: Optional[float],
    enforce_default: bool,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        entries: list[dict[str, Any]] = []
        for key, raw_multiplier in value.items():
            item_label = f"{label}[{key!r}]"
            multiplier = _coerce_multiplier_value(
                raw_multiplier,
                item_label,
                default_multiplier=default_multiplier,
                enforce_default=enforce_default,
            )
            if isinstance(key, str):
                entries.append(
                    {
                        "reaction_id": _coerce_reaction_id(key, item_label),
                        "multiplier": multiplier,
                    }
                )
                continue
            entries.append(
                {
                    "index": _coerce_reaction_index(key, item_label),
                    "multiplier": multiplier,
                }
            )
        return entries
    if isinstance(value, SequenceABC) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        entries = []
        for index, entry in enumerate(value):
            entries.append(
                _parse_multiplier_item(
                    entry,
                    f"{label}[{index}]",
                    default_multiplier=default_multiplier,
                    enforce_default=enforce_default,
                )
            )
        return entries
    raise TypeError(f"{label} must be a mapping or sequence.")


def _multiplier_sort_key(entry: Mapping[str, Any]) -> tuple[int, Any]:
    if "index" in entry:
        return (0, entry["index"])
    return (1, entry["reaction_id"])


def _dedupe_multiplier_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: dict[tuple[str, Any], dict[str, Any]] = {}
    for entry in entries:
        if "index" in entry:
            key = ("index", entry["index"])
        else:
            key = ("reaction_id", entry["reaction_id"])
        existing = seen.get(key)
        if existing is not None and existing["multiplier"] != entry["multiplier"]:
            raise ValueError(
                f"Conflicting multipliers for {key[0]} {key[1]!r}."
            )
        seen[key] = entry
    return sorted(seen.values(), key=_multiplier_sort_key)


def normalize_reaction_multipliers(
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return a stable list of reaction multiplier specs from config."""
    if not isinstance(cfg, Mapping):
        raise TypeError("reaction multiplier config must be a mapping.")
    entries: list[dict[str, Any]] = []
    entries.extend(
        _parse_multiplier_entries(
            cfg.get("reaction_multipliers"),
            label="reaction_multipliers",
            default_multiplier=None,
            enforce_default=False,
        )
    )
    entries.extend(
        _parse_multiplier_entries(
            cfg.get("disabled_reactions"),
            label="disabled_reactions",
            default_multiplier=0.0,
            enforce_default=True,
        )
    )
    return _dedupe_multiplier_entries(entries)


def canonicalize(obj: Any, exclude_keys: Optional[Iterable[str]] = None) -> Any:
    exclude = {str(key) for key in exclude_keys or ()}
    return _canonicalize(obj, exclude)


def _stable_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonicalize(obj: Any, exclude_keys: Set[str]) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, PurePath):
        return obj.as_posix()

    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": obj.hex()}

    if isinstance(obj, Mapping):
        items = []
        for key, value in obj.items():
            key_str = str(key)
            if key_str in exclude_keys:
                continue
            items.append((key_str, _canonicalize(value, exclude_keys)))
        items.sort(key=lambda item: item[0])
        return {key: value for key, value in items}

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(item, exclude_keys) for item in obj]

    if isinstance(obj, SequenceABC):
        return [_canonicalize(item, exclude_keys) for item in obj]

    if isinstance(obj, (set, frozenset)):
        items = [_canonicalize(item, exclude_keys) for item in obj]
        items.sort(key=_stable_sort_key)
        return items

    if np is not None:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": _canonicalize(obj.tolist(), exclude_keys),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.generic):
            return _canonicalize(obj.item(), exclude_keys)

    if isinstance(obj, Enum):
        return _canonicalize(obj.value, exclude_keys)

    if is_dataclass(obj):
        return _canonicalize(asdict(obj), exclude_keys)

    if hasattr(obj, "model_dump"):
        return _canonicalize(obj.model_dump(), exclude_keys)

    if hasattr(obj, "dict"):
        return _canonicalize(obj.dict(), exclude_keys)

    if hasattr(obj, "__dict__"):
        return _canonicalize(vars(obj), exclude_keys)

    raise TypeError(f"Unsupported type for canonicalize: {type(obj)!r}")


def stable_hash(
    obj: Any,
    *,
    exclude_keys: Optional[Iterable[str]] = None,
    length: Optional[int] = 16,
) -> str:
    canonical = canonicalize(obj, exclude_keys=exclude_keys)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if length is None:
        return digest
    if length <= 0:
        raise ValueError("length must be a positive integer or None.")
    return digest[:length]


def make_run_id(
    cfg: Mapping[str, Any],
    *,
    exclude_keys: Optional[Iterable[str]] = None,
    length: Optional[int] = 16,
) -> str:
    return stable_hash(cfg, exclude_keys=exclude_keys, length=length)


def make_artifact_id(
    *,
    inputs: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    code: Optional[Mapping[str, Any]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
    length: Optional[int] = 16,
) -> str:
    payload = {
        "inputs": inputs or {},
        "config": config or {},
        "code": code or {},
    }
    return stable_hash(payload, exclude_keys=exclude_keys, length=length)


__all__ = [
    "ArtifactManifest",
    "load_config",
    "load_manifest",
    "dump_manifest",
    "find_repo_root",
    "resolve_repo_path",
    "normalize_reaction_multipliers",
    "canonicalize",
    "stable_hash",
    "make_run_id",
    "make_artifact_id",
]
