"""Shared mechanism compilation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from rxn_platform.core import stable_hash
from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import (
    read_yaml_payload as _read_yaml_payload,
    write_yaml_payload,
)


def read_yaml_payload(path: Path) -> Any:
    try:
        return _read_yaml_payload(
            path,
            error_message=(
                "PyYAML is not available; patch/mechanism must be JSON-compatible."
            ),
            error_cls=ConfigError,
        )
    except OSError as exc:
        raise ConfigError(f"Failed to read YAML from {path}: {exc}") from exc


def reaction_identifiers(reaction: Any, index: int) -> list[str]:
    identifiers: list[str] = [f"R{index + 1}"]
    if isinstance(reaction, Mapping):
        for key in ("id", "name", "equation", "reaction"):
            value = reaction.get(key)
            if isinstance(value, str) and value.strip():
                identifiers.append(value.strip())
    elif isinstance(reaction, str) and reaction.strip():
        identifiers.append(reaction.strip())
    else:
        raise ConfigError("reaction entries must be mappings or strings.")
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def reaction_id_index_map(
    reactions: Sequence[Any],
) -> dict[str, list[int]]:
    id_map: dict[str, list[int]] = {}
    for idx, reaction in enumerate(reactions):
        identifiers = reaction_identifiers(reaction, idx)
        for identifier in identifiers:
            id_map.setdefault(identifier, []).append(idx)
    return id_map


def resolve_patch_entries(
    entries: Sequence[Mapping[str, Any]],
    reactions: Sequence[Any],
) -> dict[int, dict[str, Any]]:
    if not reactions:
        raise ConfigError("mechanism has no reactions to patch.")
    id_map = reaction_id_index_map(reactions)
    resolved: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if "index" in entry:
            idx = entry["index"]
            if not isinstance(idx, int):
                raise ConfigError("patch reaction index must be an int.")
            if idx < 0 or idx >= len(reactions):
                raise ConfigError(f"patch reaction index out of range: {idx}.")
        else:
            reaction_id = entry.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id.strip():
                raise ConfigError("patch reaction_id must be a non-empty string.")
            indices = id_map.get(reaction_id)
            if not indices:
                raise ConfigError(f"reaction_id not found in mechanism: {reaction_id!r}.")
            if len(indices) > 1:
                raise ConfigError(
                    f"reaction_id matches multiple reactions: {reaction_id!r}."
                )
            idx = indices[0]
        resolved[idx] = dict(entry)
    return resolved


def apply_patch_entries(
    mechanism: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], set[int]]:
    reactions = mechanism.get("reactions")
    if reactions is None:
        raise ConfigError("mechanism must define a reactions list.")
    if not isinstance(reactions, Sequence) or isinstance(
        reactions,
        (str, bytes, bytearray),
    ):
        raise ConfigError("mechanism.reactions must be a sequence.")
    reaction_list = list(reactions)
    resolved = resolve_patch_entries(entries, reaction_list)
    disabled_indices = {
        idx
        for idx, entry in resolved.items()
        if float(entry.get("multiplier", 0.0)) == 0.0
    }
    filtered_reactions = [
        reaction
        for idx, reaction in enumerate(reaction_list)
        if idx not in disabled_indices
    ]
    updated = dict(mechanism)
    updated["reactions"] = filtered_reactions
    return updated, disabled_indices


class MechanismCompiler:
    """Compile reduced mechanisms from patch entries."""

    def __init__(
        self,
        mechanism: Mapping[str, Any],
        *,
        mechanism_path: Optional[str] = None,
    ) -> None:
        if not isinstance(mechanism, Mapping):
            raise ConfigError("mechanism payload must be a mapping.")
        reactions = mechanism.get("reactions")
        if reactions is None:
            raise ConfigError("mechanism must define a reactions list.")
        if not isinstance(reactions, Sequence) or isinstance(
            reactions,
            (str, bytes, bytearray),
        ):
            raise ConfigError("mechanism.reactions must be a sequence.")
        self.payload = dict(mechanism)
        self.reactions = list(reactions)
        self.mechanism_path = mechanism_path

    @classmethod
    def from_path(cls, path: str | Path) -> "MechanismCompiler":
        mech_path = Path(path)
        try:
            payload = read_yaml_payload(mech_path)
        except ConfigError:
            payload = None

        if isinstance(payload, Mapping):
            return cls(dict(payload), mechanism_path=str(path))

        # Fallback for environments without PyYAML: use Cantera's parser and
        # reconstruct a JSON/YAML-compatible mechanism payload.
        try:  # pragma: no cover - optional dependency path
            import cantera as ct  # type: ignore
        except Exception as exc:
            raise ConfigError(
                "mechanism YAML must be a mapping (and cantera fallback is unavailable)."
            ) from exc

        def _to_builtin(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(k): _to_builtin(v) for k, v in value.items()}
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [_to_builtin(v) for v in value]
            return value

        sol = ct.Solution(str(mech_path))
        species_payload = []
        for sp in sol.species():
            entry = getattr(sp, "input_data", None)
            if entry is None:
                continue
            species_payload.append(_to_builtin(entry))

        reactions_payload = []
        for idx in range(sol.n_reactions):
            rxn = sol.reaction(idx)
            entry = getattr(rxn, "input_data", None)
            if entry is None:
                entry = {"equation": getattr(rxn, "equation", f"R{idx + 1}")}
            reactions_payload.append(_to_builtin(entry))

        mech_payload = {
            "schema_version": 1,
            "source": {"path": str(mech_path), "phase": sol.name},
            "species": species_payload,
            "reactions": reactions_payload,
        }
        return cls(mech_payload, mechanism_path=str(path))

    def apply_patch_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], set[int]]:
        return apply_patch_entries(self.payload, entries)

    def reaction_count(self) -> int:
        return len(self.reactions)

    def mechanism_hash(self, payload: Optional[Mapping[str, Any]] = None) -> str:
        return stable_hash(payload or self.payload, length=16)
