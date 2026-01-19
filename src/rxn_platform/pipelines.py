"""Sequential pipeline runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import ArtifactManifest, load_config, make_run_id
from rxn_platform.errors import ConfigError, TaskError
from rxn_platform.registry import Registry
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.runner import DEFAULT_STORE_ROOT, run_task

DEFAULT_PIPELINE_ROOT = Path("configs") / "pipeline"


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping.")
    return dict(value)


def _as_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("pipeline.steps must be a list of step mappings.")
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(value):
        if not isinstance(step, Mapping):
            raise ConfigError(f"pipeline.steps[{index}] must be a mapping.")
        steps.append(dict(step))
    return steps


def _extract_pipeline_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if "pipeline" in cfg and isinstance(cfg.get("pipeline"), Mapping):
        return dict(cfg["pipeline"])
    return dict(cfg)


def _utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _code_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {"version": __version__}
    git_dir = Path.cwd() / ".git"
    if not git_dir.exists():
        return payload
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["git_commit"] = commit
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return payload
    return payload


def _provenance_metadata() -> dict[str, Any]:
    return {"python": platform.python_version()}


def _resolve_step_ref(
    value: Any,
    *,
    results: Mapping[str, str],
    last_result: Optional[str],
    step_id: str,
) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        ref = value[1:]
        if ref == "last":
            if last_result is None:
                raise ConfigError(
                    f"pipeline step {step_id!r} references @last but no previous step "
                    "exists."
                )
            return last_result
        if not ref:
            raise ConfigError(
                f"pipeline step {step_id!r} contains an empty step reference."
            )
        if ref not in results:
            raise ConfigError(
                f"pipeline step {step_id!r} references unknown step {ref!r}."
            )
        return results[ref]
    if isinstance(value, Mapping):
        return {
            key: _resolve_step_ref(
                item,
                results=results,
                last_result=last_result,
                step_id=step_id,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _resolve_step_ref(
                item,
                results=results,
                last_result=last_result,
                step_id=step_id,
            )
            for item in value
        ]
    return value


def _normalize_step(
    step: Mapping[str, Any],
    index: int,
) -> tuple[str, str, dict[str, Any]]:
    step_id = step.get("id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ConfigError(f"pipeline.steps[{index}].id must be a non-empty string.")
    task_name = step.get("task")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ConfigError(f"pipeline.steps[{index}].task must be a non-empty string.")
    inputs = step.get("inputs", {})
    if inputs is None:
        inputs = {}
    inputs_map = _as_mapping(inputs, f"pipeline.steps[{index}].inputs")
    params = step.get("params", {})
    if params is None:
        params = {}
    params_map = _as_mapping(params, f"pipeline.steps[{index}].params")
    cfg = dict(step)
    cfg.pop("id", None)
    cfg.pop("task", None)
    cfg["inputs"] = inputs_map
    cfg["params"] = params_map
    return step_id, task_name, cfg


def _resolve_pipeline_path(
    pipeline: str | Path,
    root: Path,
) -> Path:
    path = Path(pipeline)
    if path.suffix in {".yaml", ".yml"} or path.is_absolute() or len(path.parts) > 1:
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    candidate = root / f"{path.name}.yaml"
    if candidate.exists():
        return candidate
    fallback = root / f"{path.name}.yml"
    if fallback.exists():
        return fallback
    return candidate


class PipelineRunner:
    """Load and run pipeline steps sequentially."""

    def __init__(
        self,
        *,
        store: Optional[ArtifactStore] = None,
        store_root: Optional[str | Path] = None,
        registry: Optional[Registry] = None,
        logger: Optional[logging.Logger] = None,
        config_dir: Optional[str | Path] = None,
    ) -> None:
        if store is None:
            root = Path(store_root) if store_root is not None else Path(DEFAULT_STORE_ROOT)
            store = ArtifactStore(root)
        self.store = store
        self.registry = registry
        self.logger = logger or logging.getLogger("rxn_platform.pipeline")
        if config_dir is None:
            config_dir = DEFAULT_PIPELINE_ROOT
        config_path = Path(config_dir)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        self.config_dir = config_path

    def load(self, pipeline: str | Path) -> dict[str, Any]:
        path = _resolve_pipeline_path(pipeline, self.config_dir)
        return _as_mapping(load_config(path), "pipeline config")

    def run(self, pipeline: Mapping[str, Any] | str | Path) -> dict[str, str]:
        if isinstance(pipeline, Mapping):
            raw_cfg = dict(pipeline)
        else:
            raw_cfg = self.load(pipeline)
        pipeline_cfg = _extract_pipeline_cfg(raw_cfg)
        steps = _as_steps(pipeline_cfg.get("steps"))

        pipeline_run_id = make_run_id(pipeline_cfg, exclude_keys=("hydra",))
        pipeline_manifest = ArtifactManifest(
            schema_version=1,
            kind="pipelines",
            id=pipeline_run_id,
            created_at=_utc_now_iso(),
            parents=[],
            inputs={},
            config=pipeline_cfg,
            code=_code_metadata(),
            provenance=_provenance_metadata(),
        )

        results: dict[str, str] = {}
        step_records: list[dict[str, Any]] = []
        seen: set[str] = set()
        last_result: Optional[str] = None
        for index, step in enumerate(steps):
            step_id, task_name, step_cfg = _normalize_step(step, index)
            if step_id in seen:
                raise ConfigError(f"Duplicate pipeline step id: {step_id!r}.")
            seen.add(step_id)
            resolved_inputs = _resolve_step_ref(
                step_cfg["inputs"],
                results=results,
                last_result=last_result,
                step_id=step_id,
            )
            step_cfg["inputs"] = resolved_inputs
            self.logger.info("Running pipeline step %s (%s).", step_id, task_name)
            started_at = _utc_now_iso()
            start_clock = time.perf_counter()
            try:
                result = run_task(
                    task_name,
                    step_cfg,
                    store=self.store,
                    registry=self.registry,
                    logger=self.logger,
                )
            except Exception as exc:
                raise TaskError(
                    f"Pipeline step {step_id!r} failed: {exc}",
                    context={"step": step_id, "task": task_name},
                ) from exc
            ended_at = _utc_now_iso()
            elapsed = time.perf_counter() - start_clock
            artifact_id = result.manifest.id
            results[step_id] = artifact_id
            last_result = artifact_id
            step_records.append(
                {
                    "id": step_id,
                    "task": task_name,
                    "artifact_id": artifact_id,
                    "inputs": resolved_inputs,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "elapsed_seconds": elapsed,
                }
            )
        results_payload = {
            "schema_version": 1,
            "pipeline_run_id": pipeline_run_id,
            "steps": step_records,
            "results": results,
        }

        def _writer(base_dir: Path) -> None:
            payload = json.dumps(
                results_payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            (base_dir / "results.json").write_text(
                f"{payload}\n",
                encoding="utf-8",
            )

        self.store.ensure(pipeline_manifest, writer=_writer)
        return results


__all__ = ["DEFAULT_PIPELINE_ROOT", "PipelineRunner"]
