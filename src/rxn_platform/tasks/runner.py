"""Task resolution and execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import logging
from pathlib import Path
from typing import Any, Optional

from rxn_platform.errors import ConfigError, RxnPlatformError, TaskError
from rxn_platform.hydra_utils import resolve_config
import rxn_platform.registry as registry_module
from rxn_platform.registry import Registry
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import TaskContext

DEFAULT_STORE_ROOT = "artifacts"


def _load_builtin_plugins() -> None:
    # Import for side effects: register built-in tasks/backends.
    import rxn_platform.tasks.sim  # noqa: F401
    import rxn_platform.tasks.viz  # noqa: F401
    import rxn_platform.tasks.observables  # noqa: F401
    import rxn_platform.tasks.graphs  # noqa: F401
    import rxn_platform.tasks.features  # noqa: F401
    import rxn_platform.tasks.sensitivity  # noqa: F401
    import rxn_platform.tasks.optimization  # noqa: F401
    import rxn_platform.tasks.assimilation  # noqa: F401
    import rxn_platform.tasks.reduction  # noqa: F401
    import rxn_platform.tasks.doe  # noqa: F401
    import rxn_platform.tasks.dimred  # noqa: F401
    import rxn_platform.tasks.sbi  # noqa: F401
    import rxn_platform.backends.dummy  # noqa: F401
    import rxn_platform.backends.cantera  # noqa: F401


def _resolve_cfg(cfg: Any) -> dict[str, Any]:
    try:
        resolved = resolve_config(cfg)
    except (ConfigError, TypeError, ValueError):
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise
    return resolved


def _select_registry(registry: Optional[Registry]) -> Registry | None:
    return registry


def resolve_task(name: str, *, registry: Optional[Registry] = None) -> Any:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("task name must be a non-empty string.")

    registry = _select_registry(registry)
    if registry is None:
        _load_builtin_plugins()
        try:
            return registry_module.get("task", name)
        except KeyError as exc:
            available = ", ".join(sorted(registry_module.list("task"))) or "<none>"
            raise TaskError(
                f"Task {name!r} is not registered. Available: {available}.",
                context={"task": name},
            ) from exc

    try:
        return registry.get("task", name)
    except KeyError as exc:
        available = ", ".join(sorted(registry.list("task"))) or "<none>"
        raise TaskError(
            f"Task {name!r} is not registered. Available: {available}.",
            context={"task": name},
        ) from exc


def _call_task(
    task_obj: Any,
    cfg: Any,
    context: TaskContext,
) -> ArtifactCacheResult:
    if hasattr(task_obj, "run"):
        func = task_obj.run
    else:
        func = task_obj
    if not callable(func):
        raise TaskError("Task entry is not callable.")

    signature = inspect.signature(func)
    params = signature.parameters
    kwargs: dict[str, Any] = {}

    if "context" in params:
        kwargs["context"] = context
    else:
        if "store" in params:
            kwargs["store"] = context.store
        if "registry" in params:
            kwargs["registry"] = context.registry

    if "cfg" in params:
        kwargs["cfg"] = cfg
    elif "config" in params:
        kwargs["config"] = cfg
    else:
        raise TaskError("Task must accept a cfg or config argument.")

    return func(**kwargs)


def _require_artifact_result(task_name: str, result: Any) -> ArtifactCacheResult:
    if isinstance(result, ArtifactCacheResult):
        return result
    raise TaskError(
        f"Task {task_name!r} must return an ArtifactCacheResult.",
        context={"task": task_name},
    )


def run_task(
    task_name: str,
    cfg: Any,
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
    logger: Optional[logging.Logger] = None,
) -> ArtifactCacheResult:
    task_logger = logger or logging.getLogger("rxn_platform.task_runner")
    registry = _select_registry(registry)
    context = TaskContext(store=store, registry=registry, logger=task_logger)

    task_obj = resolve_task(task_name, registry=registry)
    try:
        result = _call_task(task_obj, cfg, context)
    except RxnPlatformError:
        raise
    except Exception as exc:
        raise TaskError(
            f"Task {task_name!r} failed: {exc}",
            context={"task": task_name},
        ) from exc

    artifact_result = _require_artifact_result(task_name, result)
    task_logger.info(
        "Task %s produced artifact %s.",
        task_name,
        artifact_result.manifest.id,
    )
    return artifact_result


def _extract_task_name(cfg: Mapping[str, Any]) -> str:
    task_cfg = cfg.get("task")
    name: Any = None
    if isinstance(task_cfg, Mapping):
        name = task_cfg.get("name") or task_cfg.get("task")
    elif isinstance(task_cfg, str):
        name = task_cfg
    elif task_cfg is None:
        name = cfg.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ConfigError("task.name must be a non-empty string.")
    return name


def _extract_store_root(cfg: Mapping[str, Any]) -> Path:
    store_cfg = cfg.get("store")
    root: Any = DEFAULT_STORE_ROOT
    if isinstance(store_cfg, Mapping):
        root = store_cfg.get("root", DEFAULT_STORE_ROOT)
    elif store_cfg is None:
        root = DEFAULT_STORE_ROOT
    else:
        raise ConfigError("store must be a mapping if provided.")

    if not isinstance(root, str) or not root.strip():
        raise ConfigError("store.root must be a non-empty string.")
    return Path(root)


def run_task_from_config(
    cfg: Any,
    *,
    registry: Optional[Registry] = None,
    logger: Optional[logging.Logger] = None,
) -> ArtifactCacheResult:
    resolved = _resolve_cfg(cfg)
    task_name = _extract_task_name(resolved)
    store_root = _extract_store_root(resolved)
    store = ArtifactStore(store_root)
    return run_task(
        task_name,
        resolved,
        store=store,
        registry=registry,
        logger=logger,
    )


__all__ = ["DEFAULT_STORE_ROOT", "resolve_task", "run_task", "run_task_from_config"]
