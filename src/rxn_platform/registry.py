"""Plugin registry for backends, tasks, observables, features, and viz."""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from typing import Any

from rxn_platform.errors import BackendError

DEFAULT_KINDS = ("backend", "task", "observable", "feature", "viz")


def _validate_key(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string.")
    return value


def _format_options(options: Iterable[str]) -> str:
    values = builtins.list(options)
    if not values:
        return "<none>"
    return ", ".join(sorted(values))


class Registry:
    """Registry of plugin objects organized by kind/name."""

    def __init__(self, kinds: Iterable[str] = DEFAULT_KINDS) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        for kind in kinds:
            self.add_kind(kind)

    def add_kind(self, kind: str, *, overwrite: bool = False) -> None:
        kind = _validate_key("kind", kind)
        if kind in self._entries and not overwrite:
            raise ValueError(f"Registry kind already exists: {kind!r}.")
        self._entries[kind] = {}

    def register(self, kind: str, name: str, obj: Any, *, overwrite: bool = False) -> None:
        kind = _validate_key("kind", kind)
        name = _validate_key("name", name)
        bucket = self._entries.get(kind)
        if bucket is None:
            available = _format_options(self._entries.keys())
            raise KeyError(
                f"Unknown registry kind: {kind!r}. Available kinds: {available}."
            )
        if name in bucket and not overwrite:
            raise ValueError(
                f"{kind} {name!r} is already registered; use overwrite=True to replace."
            )
        bucket[name] = obj

    def get(self, kind: str, name: str) -> Any:
        kind = _validate_key("kind", kind)
        name = _validate_key("name", name)
        bucket = self._entries.get(kind)
        if bucket is None:
            available = _format_options(self._entries.keys())
            raise KeyError(
                f"Unknown registry kind: {kind!r}. Available kinds: {available}."
            )
        if name not in bucket:
            available = _format_options(bucket.keys())
            raise KeyError(
                f"{kind} {name!r} is not registered. Available: {available}."
            )
        return bucket[name]

    def list(self, kind: str) -> list[str]:
        kind = _validate_key("kind", kind)
        bucket = self._entries.get(kind)
        if bucket is None:
            available = _format_options(self._entries.keys())
            raise KeyError(
                f"Unknown registry kind: {kind!r}. Available kinds: {available}."
            )
        return builtins.list(bucket.keys())


_DEFAULT_REGISTRY = Registry()


def add_kind(kind: str, *, overwrite: bool = False) -> None:
    _DEFAULT_REGISTRY.add_kind(kind, overwrite=overwrite)


def register(kind: str, name: str, obj: Any, *, overwrite: bool = False) -> None:
    _DEFAULT_REGISTRY.register(kind, name, obj, overwrite=overwrite)


def get(kind: str, name: str) -> Any:
    return _DEFAULT_REGISTRY.get(kind, name)


def list(kind: str) -> list[str]:
    return _DEFAULT_REGISTRY.list(kind)


def resolve_backend(name: str, *, registry: Registry | None = None) -> Any:
    registry = registry or _DEFAULT_REGISTRY
    try:
        return registry.get("backend", name)
    except KeyError as exc:
        available = _format_options(registry.list("backend"))
        raise BackendError(
            f"Backend {name!r} is not registered. Available: {available}."
        ) from exc


__all__ = [
    "DEFAULT_KINDS",
    "Registry",
    "add_kind",
    "register",
    "get",
    "list",
    "resolve_backend",
]
