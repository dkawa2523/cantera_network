"""Task base definitions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Optional

from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.registry import Registry


@dataclass(frozen=True)
class TaskContext:
    """Shared task execution context."""

    store: ArtifactStore
    registry: Optional[Registry] = None
    logger: Optional[logging.Logger] = None


class Task:
    """Base class for task implementations."""

    name: str

    def run(
        self,
        store: ArtifactStore,
        cfg: Mapping[str, Any],
    ) -> ArtifactCacheResult:
        raise NotImplementedError("Task implementations must override run().")

    def __call__(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
    ) -> ArtifactCacheResult:
        return self.run(store=store, cfg=cfg)


__all__ = ["Task", "TaskContext"]
