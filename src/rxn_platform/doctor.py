"""Environment diagnostics and smoke checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import logging
from pathlib import Path
import tempfile
from typing import Any, Optional

from rxn_platform.errors import (
    ArtifactError,
    BackendError,
    ConfigError,
    RxnPlatformError,
    TaskError,
    ValidationError,
)
from rxn_platform.hydra_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONFIG_PATH,
    compose_config,
    resolve_config,
)
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.runner import DEFAULT_STORE_ROOT
from rxn_platform.run_store import (
    RUNS_ROOT,
    RUN_MANIFEST_NAME,
    RUN_CONFIG_NAME,
    RUN_METRICS_NAME,
    LEGACY_RUN_MANIFEST_NAME,
    LEGACY_RUN_CONFIG_NAME,
    LEGACY_RUN_RESULTS_NAME,
    read_run_manifest,
    read_run_metrics,
)

_RECOMMENDED_DEPS = ("numpy", "xarray")
_OPTIONAL_DEPS = ("cantera",)
_REQUIRED_RUN_FIELDS = (
    "simulator",
    "mechanism_hash",
    "conditions_hash",
    "qoi_spec_hash",
)
_FAILURE_CATEGORIES = ("io", "contract", "sim", "graph", "ml")
_IO_TOKENS = ("no such file", "not found", "permission", "writable", "read", "write")
_CONTRACT_TOKENS = ("config", "schema", "manifest", "validation", "required", "must be", "contract")
_SIM_TOKENS = ("cantera", "backend", "simulation", "reactor")
_GRAPH_TOKENS = ("graph", "laplacian", "stoich", "temporal")
_ML_TOKENS = ("gnn", "diffpool", "torch", "pytorch", "optimization", "train", "model", "assim", "sbi", "dataset")


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure path is environment-specific
        return False, f"{module_name} import failed: {exc}"
    version = getattr(module, "__version__", None)
    if version:
        return True, f"{module_name} {version}"
    return True, f"{module_name} import ok"


def _category_for_task(task_name: str) -> str:
    name = task_name.strip().lower()
    if name.startswith("sim"):
        return "sim"
    if name.startswith("graphs") or "graph" in name:
        return "graph"
    if name.startswith("gnn") or "gnn" in name:
        return "ml"
    if name.startswith("reduction."):
        if "cnr" in name:
            return "graph"
        return "ml"
    if name.startswith(("optimization", "assimilation", "sbi", "dimred")):
        return "ml"
    return "contract"


def categorize_error(
    error: BaseException | str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if isinstance(error, ArtifactError):
        return "io"
    if isinstance(error, (ConfigError, ValidationError)):
        return "contract"
    if isinstance(error, BackendError):
        return "sim"
    if isinstance(error, TaskError):
        ctx = context
        if ctx is None and isinstance(error, RxnPlatformError):
            ctx = error.context
        if isinstance(ctx, Mapping):
            task_name = ctx.get("task")
            if isinstance(task_name, str) and task_name.strip():
                return _category_for_task(task_name)

    message = error if isinstance(error, str) else str(error)
    lowered = message.lower()
    if any(token in lowered for token in _SIM_TOKENS):
        return "sim"
    if any(token in lowered for token in _GRAPH_TOKENS):
        return "graph"
    if any(token in lowered for token in _ML_TOKENS):
        return "ml"
    if any(token in lowered for token in _IO_TOKENS):
        return "io"
    if any(token in lowered for token in _CONTRACT_TOKENS):
        return "contract"
    return "contract"


def _strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0].rstrip()


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if cleaned[0] in {"'", '"'} and cleaned[-1] == cleaned[0]:
        return cleaned[1:-1]
    try:
        if "." not in cleaned and "e" not in lowered:
            return int(cleaned)
        return float(cleaned)
    except ValueError:
        return cleaned


def _split_key_value(line: str, *, path: Path, line_no: int) -> tuple[str, Optional[str]]:
    if ":" not in line:
        raise ConfigError(f"Invalid YAML line in {path} at {line_no}: {line}")
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Empty YAML key in {path} at {line_no}.")
    value = value.strip()
    if value == "":
        return key, None
    return key, value


def _parse_block(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[Any, int]:
    while index < len(lines) and _is_blank(lines[index]):
        index += 1
    if index >= len(lines):
        return None, index
    line = _strip_inline_comment(lines[index])
    line_indent = _line_indent(line)
    if line_indent < indent:
        return None, index
    if line_indent > indent:
        raise ConfigError(
            f"Unexpected indent in {path} at line {index + 1}."
        )
    stripped = line.strip()
    if stripped.startswith("- "):
        return _parse_list(lines, index, indent, path=path)
    return _parse_mapping(lines, index, indent, path=path)


def _parse_mapping(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        if _is_blank(lines[index]):
            index += 1
            continue
        line = _strip_inline_comment(lines[index])
        line_indent = _line_indent(line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigError(
                f"Unexpected indent in {path} at line {index + 1}."
            )
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ConfigError(
                f"Unexpected list item in {path} at line {index + 1}."
            )
        key, value = _split_key_value(stripped, path=path, line_no=index + 1)
        if value is None:
            nested, index = _parse_block(
                lines,
                index + 1,
                indent + 2,
                path=path,
            )
            mapping[key] = nested
        else:
            mapping[key] = _parse_scalar(value)
            index += 1
    return mapping, index


def _parse_list(
    lines: list[str],
    index: int,
    indent: int,
    *,
    path: Path,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        if _is_blank(lines[index]):
            index += 1
            continue
        line = _strip_inline_comment(lines[index])
        line_indent = _line_indent(line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigError(
                f"Unexpected indent in {path} at line {index + 1}."
            )
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        content = stripped[2:].strip()
        if not content:
            item, index = _parse_block(
                lines,
                index + 1,
                indent + 2,
                path=path,
            )
            items.append(item)
            continue
        if ":" in content:
            key, value = _split_key_value(content, path=path, line_no=index + 1)
            item: dict[str, Any] = {}
            if value is None:
                nested, index = _parse_block(
                    lines,
                    index + 1,
                    indent + 2,
                    path=path,
                )
                item[key] = nested
            else:
                item[key] = _parse_scalar(value)
                index += 1
            while True:
                next_index = index
                while next_index < len(lines) and _is_blank(lines[next_index]):
                    next_index += 1
                if next_index >= len(lines):
                    break
                next_indent = _line_indent(lines[next_index])
                if next_indent <= indent:
                    break
                if next_indent < indent + 2:
                    raise ConfigError(
                        f"Unexpected indent in {path} at line {next_index + 1}."
                    )
                extra, index = _parse_mapping(
                    lines,
                    next_index,
                    indent + 2,
                    path=path,
                )
                item.update(extra)
                break
            items.append(item)
            continue
        items.append(_parse_scalar(content))
        index += 1
    return items, index


def _load_simple_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    # Prefer the shared YAML reader (PyYAML when available; otherwise a small
    # fallback parser that supports our config subset).
    try:
        from rxn_platform.io_utils import read_yaml_payload as _read_yaml_payload

        payload = _read_yaml_payload(path)
    except Exception as exc:
        raise ConfigError(f"Failed to load YAML from {path}: {exc}") from exc
    if payload is None:
        return {}
    return payload


def _normalize_config_name(name: str) -> str:
    if name.endswith((".yaml", ".yml")):
        return Path(name).stem
    return name


def _resolve_yaml_path(base_dir: Path, name: str) -> Path:
    candidate = base_dir / f"{name}.yaml"
    if candidate.exists():
        return candidate
    fallback = base_dir / f"{name}.yml"
    if fallback.exists():
        return fallback
    return candidate


def _extract_group_name(
    entry: Mapping[str, Any],
    group: str,
) -> Optional[str]:
    if group in entry:
        return str(entry[group])
    override_key = f"override /{group}"
    if override_key in entry:
        return str(entry[override_key])
    root_key = f"/{group}"
    if root_key in entry:
        return str(entry[root_key])
    return None


def _extract_defaults_names(
    defaults_cfg: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    defaults = defaults_cfg.get("defaults")
    if not isinstance(defaults, list):
        raise ConfigError("defaults must be a list when using fallback config load.")
    sim_name: Optional[str] = None
    pipeline_name: Optional[str] = None
    for entry in defaults:
        if isinstance(entry, Mapping):
            sim_override = _extract_group_name(entry, "sim")
            if sim_override is not None:
                sim_name = sim_override
            pipeline_override = _extract_group_name(entry, "pipeline")
            if pipeline_override is not None:
                pipeline_name = pipeline_override
    return sim_name, pipeline_name


def _extract_recipe_name(defaults_cfg: Mapping[str, Any]) -> Optional[str]:
    defaults = defaults_cfg.get("defaults")
    if not isinstance(defaults, list):
        return None
    for entry in defaults:
        if isinstance(entry, Mapping):
            recipe_name = _extract_group_name(entry, "recipe")
            if recipe_name:
                return recipe_name
    return None


def _extract_fallback_overrides(
    overrides: Optional[Sequence[str]],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    group_overrides: dict[str, str] = {}
    assignments: dict[str, str] = {}
    ignored: list[str] = []
    if not overrides:
        return group_overrides, assignments, ignored
    for item in overrides:
        if not item or item == "--":
            continue
        if "=" not in item:
            ignored.append(item)
            continue
        raw_key, value = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            continue
        if key.startswith("+"):
            key = key[1:]
        if key in {"sim", "pipeline", "recipe", "benchmarks"}:
            group_overrides[key] = value
            continue
        assignments[key] = value
    return group_overrides, assignments, ignored


def _select_dotted(cfg: Mapping[str, Any], path: str) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_dotted(cfg: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        return
    parts = [part for part in path.split(".") if part]
    if not parts:
        return
    current: Any = cfg
    for part in parts[:-1]:
        if not isinstance(current, Mapping):
            return
        next_value = current.get(part)
        if not isinstance(next_value, Mapping):
            next_value = {}
            current[part] = next_value
        current = next_value
    if isinstance(current, Mapping):
        current[parts[-1]] = value


def _resolve_interpolations(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        # Support nested `${...}` substitutions without Hydra/OmegaConf. We resolve
        # repeatedly until the string is no longer an interpolation or we detect a loop.
        current: Any = value
        seen: set[str] = set()
        while isinstance(current, str):
            text = current.strip()
            if not (text.startswith("${") and text.endswith("}")):
                return current
            if text in seen:
                return current
            seen.add(text)

            expr = text[2:-1].strip()
            candidate: Any = None
            if expr.startswith("oc.select:"):
                rest = expr[len("oc.select:") :]
                if "," in rest:
                    path_expr, default = rest.split(",", 1)
                else:
                    path_expr, default = rest, ""
                path_expr = path_expr.strip()
                default = default.strip()
                selected = _select_dotted(context, path_expr)
                if selected is None or (isinstance(selected, str) and not selected.strip()):
                    candidate = default
                else:
                    candidate = selected
            else:
                selected = _select_dotted(context, expr)
                if selected is None:
                    return current
                candidate = selected

            if isinstance(candidate, str):
                current = candidate
                continue
            return _resolve_interpolations(candidate, context)
        return _resolve_interpolations(current, context)
    if isinstance(value, Mapping):
        return {key: _resolve_interpolations(item, context) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolve_interpolations(item, context) for item in value]
    return value


def _compose_config_fallback(
    *,
    config_path: str | Path,
    config_name: str,
    overrides: Optional[Sequence[str]],
    logger: logging.Logger,
) -> dict[str, Any]:
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    defaults_path = _resolve_yaml_path(
        config_dir, _normalize_config_name(config_name)
    )
    defaults_cfg = _load_simple_yaml(defaults_path)
    sim_name, pipeline_name = _extract_defaults_names(defaults_cfg)
    recipe_name = _extract_recipe_name(defaults_cfg)
    group_overrides, assignments, ignored = _extract_fallback_overrides(overrides)
    if "recipe" in group_overrides:
        recipe_name = group_overrides["recipe"] or recipe_name
    if recipe_name:
        recipe_path = _resolve_yaml_path(config_dir / "recipe", recipe_name)
        if recipe_path.exists():
            recipe_cfg = _load_simple_yaml(recipe_path)
            recipe_sim, recipe_pipeline = _extract_defaults_names(recipe_cfg)
            if recipe_sim:
                sim_name = recipe_sim
            if recipe_pipeline:
                pipeline_name = recipe_pipeline
    if "sim" in group_overrides and group_overrides["sim"]:
        sim_name = group_overrides["sim"]
    if "pipeline" in group_overrides and group_overrides["pipeline"]:
        pipeline_name = group_overrides["pipeline"]
    if sim_name is None or pipeline_name is None:
        raise ConfigError(
            "Fallback config load requires defaults for sim and pipeline."
        )
    if ignored:
        logger.warning(
            "Fallback config loader ignored overrides: %s",
            ", ".join(ignored),
        )

    sim_path = _resolve_yaml_path(config_dir / "sim", sim_name)
    pipeline_path = _resolve_yaml_path(config_dir / "pipeline", pipeline_name)
    sim_cfg = _load_simple_yaml(sim_path)
    pipeline_cfg = _load_simple_yaml(pipeline_path)

    store_cfg = defaults_cfg.get("store", {})
    if not isinstance(store_cfg, Mapping):
        store_cfg = {}
    store_cfg = dict(store_cfg)
    composed: dict[str, Any] = {
        "pipeline": dict(pipeline_cfg),
        "sim": dict(sim_cfg),
        "store": store_cfg,
    }

    # Optional config groups referenced via interpolation in benchmark pipelines.
    benchmarks_name = group_overrides.get("benchmarks")
    if benchmarks_name:
        bench_path = _resolve_yaml_path(config_dir / "benchmarks", benchmarks_name)
        if bench_path.exists():
            composed["benchmarks"] = _load_simple_yaml(bench_path)
    composed.setdefault("benchmarks", {})
    composed.setdefault("mechanism", {"path": ""})
    composed.setdefault("assimilation", {})

    # Apply dotted assignments (exp/run_id, mechanism.path, benchmarks.case_id, store.root, etc).
    for key, value in assignments.items():
        _set_dotted(composed, key, value)

    # Resolve `${...}` placeholders inside pipeline/sim configs (Hydra-free interpolation).
    composed["sim"] = _resolve_interpolations(composed.get("sim", {}), composed)
    composed["pipeline"] = _resolve_interpolations(composed.get("pipeline", {}), composed)

    return composed


def _extract_store_root(cfg: Mapping[str, Any]) -> Path:
    store_cfg = cfg.get("store")
    if store_cfg is None:
        root: Any = DEFAULT_STORE_ROOT
    elif isinstance(store_cfg, Mapping):
        root = store_cfg.get("root", DEFAULT_STORE_ROOT)
    else:
        raise ConfigError("store must be a mapping if provided.")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError("store.root must be a non-empty string.")
    return Path(root)


def _check_artifact_writable(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".doctor.", dir=root) as tmp_dir:
            check_path = Path(tmp_dir) / "write_check.txt"
            check_path.write_text("ok\n", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Artifact root not writable: {root}") from exc


def _log_dependency(
    logger: logging.Logger,
    module_name: str,
    *,
    severity: str,
) -> bool:
    ok, detail = _check_import(module_name)
    if ok:
        logger.info("Dependency ok: %s", detail)
        return True
    if severity == "required":
        logger.error("Dependency missing: %s", detail)
    else:
        logger.warning("Dependency missing: %s", detail)
    return False


def _scan_runstore(root: Path) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    if not root.exists():
        return issues
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            missing_files: list[str] = []
            for name in (RUN_MANIFEST_NAME, RUN_CONFIG_NAME, RUN_METRICS_NAME):
                if not (run_dir / name).exists():
                    missing_files.append(name)
            legacy_files: list[str] = []
            for name in (
                LEGACY_RUN_MANIFEST_NAME,
                LEGACY_RUN_CONFIG_NAME,
                LEGACY_RUN_RESULTS_NAME,
            ):
                if (run_dir / name).exists():
                    legacy_files.append(name)
            has_metadata = any(
                (run_dir / name).exists()
                for name in (RUN_MANIFEST_NAME, RUN_CONFIG_NAME, RUN_METRICS_NAME)
            ) or bool(legacy_files)
            missing_fields: list[str] = []
            manifest = read_run_manifest(run_dir)
            if isinstance(manifest, Mapping):
                for field in _REQUIRED_RUN_FIELDS:
                    value = manifest.get(field)
                    if value is None or (isinstance(value, str) and not value.strip()):
                        missing_fields.append(field)
            else:
                missing_fields.extend(_REQUIRED_RUN_FIELDS)
            if missing_files or missing_fields:
                issues.append(
                    {
                        "exp": exp_dir.name,
                        "run_id": run_dir.name,
                        "run_root": run_dir,
                        "missing_files": missing_files,
                        "missing_fields": missing_fields,
                        "legacy_files": legacy_files,
                        "orphan": not has_metadata,
                    }
                )
    return issues


def _log_runstore_issues(
    logger: logging.Logger, issues: list[dict[str, object]]
) -> None:
    for issue in issues:
        run_root = issue["run_root"]
        missing_files = ", ".join(issue["missing_files"]) or "none"
        missing_fields = ", ".join(issue["missing_fields"]) or "none"
        legacy_files = ", ".join(issue["legacy_files"]) or "none"
        logger.warning(
            "RunStore contract violation at %s (missing files: %s; missing fields: %s; legacy: %s).",
            run_root,
            missing_files,
            missing_fields,
            legacy_files,
        )
        if issue.get("orphan"):
            logger.warning(
                "Orphan run directory detected at %s (artifacts without RunStore metadata). "
                "Re-run `rxn run ...` to create manifest/config/metrics.",
                run_root,
            )
        elif issue["legacy_files"]:
            logger.warning(
                "Legacy run detected at %s. Repair by re-running `rxn run ...` or "
                "register via `rxn import-legacy %s --exp <exp> --run-id <new_id>`.",
                run_root,
                run_root,
            )
        else:
            logger.warning(
                "Repair by re-running `rxn run ...` to regenerate RunStore metadata for %s.",
                run_root,
            )


def _scan_runstore_failures(root: Path) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    if not root.exists():
        return failures
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            metrics = read_run_metrics(run_dir)
            if not isinstance(metrics, Mapping):
                continue
            status = metrics.get("status")
            error = metrics.get("error")
            if status != "failed" and error is None:
                continue
            category = metrics.get("error_category")
            if not isinstance(category, str) or category not in _FAILURE_CATEGORIES:
                category = categorize_error(str(error) if error is not None else "")
            failures.append(
                {
                    "exp": exp_dir.name,
                    "run_id": run_dir.name,
                    "run_root": run_dir,
                    "category": category,
                    "error": error,
                }
            )
    return failures


def _log_runstore_failures(
    logger: logging.Logger, failures: list[dict[str, object]]
) -> None:
    for failure in failures:
        logger.error(
            "Run failed in %s/%s (category=%s): %s",
            failure.get("exp"),
            failure.get("run_id"),
            failure.get("category"),
            failure.get("error"),
        )


def run_doctor(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: Optional[Sequence[str]] = None,
    strict: bool = False,
    runstore_root: Optional[str | Path] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("rxn_platform.doctor")
    logger.info("Doctor starting.")
    hydra_ok, hydra_detail = _check_import("hydra")
    if hydra_ok:
        logger.info("Dependency ok: %s", hydra_detail)
    else:
        logger.warning(
            "Dependency missing: %s (using fallback config loader).",
            hydra_detail,
        )
    for module_name in _RECOMMENDED_DEPS:
        _log_dependency(logger, module_name, severity="recommended")
    for module_name in _OPTIONAL_DEPS:
        _log_dependency(logger, module_name, severity="optional")

    if hydra_ok:
        cfg = compose_config(
            config_path=config_path,
            config_name=config_name,
            overrides=overrides,
        )
        resolved = resolve_config(cfg)
    else:
        resolved = _compose_config_fallback(
            config_path=config_path,
            config_name=config_name,
            overrides=overrides,
            logger=logger,
        )
    store_root = _extract_store_root(resolved)
    _check_artifact_writable(store_root)
    logger.info("Artifact root writable: %s", store_root)

    store = ArtifactStore(store_root)
    runner = PipelineRunner(store=store, logger=logger)
    try:
        results = runner.run(resolved)
    except Exception as exc:
        category = categorize_error(exc)
        logger.error("Doctor smoke pipeline failed (category=%s).", category)
        raise RxnPlatformError(
            f"Doctor failed to run dummy smoke pipeline: {exc}",
            user_message="Doctor failed while running the dummy smoke pipeline.",
            context={"error": str(exc), "category": category},
        ) from exc
    logger.info("Dummy smoke pipeline completed: %s", results)

    if runstore_root is None:
        runstore_root = RUNS_ROOT
    runstore_root = Path(runstore_root)
    if not runstore_root.is_absolute():
        runstore_root = (Path.cwd() / runstore_root).resolve()
    issues = _scan_runstore(runstore_root)
    if issues:
        _log_runstore_issues(logger, issues)
        if strict:
            blocking = [
                issue
                for issue in issues
                if not issue.get("legacy_files") and not issue.get("orphan")
            ]
            if not blocking:
                logger.info(
                    "RunStore strict mode: only legacy/orphan runs detected; continuing."
                )
                logger.info("Doctor completed successfully.")
                return
            example = (
                "Re-run with `rxn run recipe=... run_id=...` or register legacy output "
                "via `rxn import-legacy <legacy_root> --exp <exp> --run-id <new_id>`."
            )
            raise RxnPlatformError(
                f"RunStore contract violations found under {runstore_root}.",
                user_message=(
                    f"RunStore contract violations found under {runstore_root}. "
                    f"{example}"
                ),
                context={"issues": blocking},
            )
    failures = _scan_runstore_failures(runstore_root)
    if failures:
        _log_runstore_failures(logger, failures)
    logger.info("RunStore contract check ok.")
    logger.info("Doctor completed successfully.")


__all__ = ["categorize_error", "run_doctor"]
