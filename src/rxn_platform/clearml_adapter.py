"""ClearML integration for RunStore metadata (optional dependency)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Optional

try:  # Optional dependency.
    from clearml import Task
except ImportError:  # pragma: no cover - optional dependency
    Task = None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "y", "on"}:
            return True
        if cleaned in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if "," in cleaned:
            return [part.strip() for part in cleaned.split(",") if part.strip()]
        return [cleaned]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tags: list[str] = []
        for item in value:
            cleaned = _coerce_str(item)
            if cleaned:
                tags.append(cleaned)
        return tags
    return []


@dataclass(frozen=True)
class ClearMLSettings:
    enabled: bool
    dry_run: bool
    project: str
    task_name: str
    tags: tuple[str, ...]


def _resolve_settings(
    cfg: Mapping[str, Any], manifest: Mapping[str, Any]
) -> ClearMLSettings:
    clearml_value = cfg.get("clearml")
    enabled = False
    dry_run = _coerce_bool(cfg.get("dry_run"))
    project = None
    task_name = None
    tags: list[str] = []

    if isinstance(clearml_value, Mapping):
        enabled = _coerce_bool(clearml_value.get("enabled", True))
        dry_run = _coerce_bool(clearml_value.get("dry_run", dry_run))
        project = _coerce_str(clearml_value.get("project"))
        task_name = _coerce_str(clearml_value.get("task_name"))
        tags = _coerce_tags(clearml_value.get("tags"))
    else:
        enabled = _coerce_bool(clearml_value)
        project = _coerce_str(cfg.get("clearml_project"))
        task_name = _coerce_str(cfg.get("clearml_task_name"))
        tags = _coerce_tags(cfg.get("clearml_tags"))
        if cfg.get("clearml_dry_run") is not None:
            dry_run = _coerce_bool(cfg.get("clearml_dry_run"))

    exp = _coerce_str(manifest.get("exp") or cfg.get("exp")) or "default"
    run_id = _coerce_str(manifest.get("run_id") or cfg.get("run_id")) or "run"
    recipe = _coerce_str(manifest.get("recipe"))

    if project is None:
        project = f"rxn_platform/{exp}"
    if task_name is None:
        task_name = f"{run_id} ({recipe})" if recipe else run_id

    return ClearMLSettings(
        enabled=enabled,
        dry_run=dry_run,
        project=project,
        task_name=task_name,
        tags=tuple(tags),
    )


def maybe_register_clearml_task(
    cfg: Mapping[str, Any],
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    metrics: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not isinstance(cfg, Mapping):
        return
    logger = logger or logging.getLogger("rxn_platform.clearml")
    settings = _resolve_settings(cfg, manifest)
    if not settings.enabled:
        return
    if settings.dry_run:
        if Task is None:
            logger.info(
                "ClearML dry-run: clearml is not installed; would register run %s/%s "
                "to project '%s' (install with `pip install .[clearml]`).",
                manifest.get("exp"),
                manifest.get("run_id"),
                settings.project,
            )
        else:
            logger.info(
                "ClearML dry-run: would register run %s/%s to project '%s'.",
                manifest.get("exp"),
                manifest.get("run_id"),
                settings.project,
            )
        return
    if Task is None:
        logger.info(
            "ClearML integration requested but clearml is not installed. "
            "Install with `pip install .[clearml]`."
        )
        return

    task = None
    try:
        task = Task.init(project_name=settings.project, task_name=settings.task_name)
        if settings.tags:
            task.add_tags(list(settings.tags))
        meta = {
            "exp": manifest.get("exp"),
            "run_id": manifest.get("run_id"),
            "created_at": manifest.get("created_at"),
            "recipe": manifest.get("recipe"),
            "store_root": manifest.get("store_root"),
            "run_root": str(run_root),
        }
        task.connect(meta, name="runstore")
        task.upload_artifact("manifest", dict(manifest))
        if isinstance(metrics, Mapping):
            task.upload_artifact("metrics", dict(metrics))
        else:
            task.upload_artifact("metrics", {"payload": metrics})
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("ClearML integration failed: %s", exc)
    finally:
        if task is not None:
            try:
                task.close()
            except Exception:  # pragma: no cover - optional dependency
                logger.debug("ClearML task close failed.", exc_info=True)


__all__ = ["maybe_register_clearml_task", "ClearMLSettings"]
