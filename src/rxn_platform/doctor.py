"""Environment diagnostics and smoke checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import logging
from pathlib import Path
import tempfile
from typing import Any, Optional

from rxn_platform.errors import ArtifactError, ConfigError, RxnPlatformError
from rxn_platform.hydra_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONFIG_PATH,
    compose_config,
    resolve_config,
)
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.runner import DEFAULT_STORE_ROOT

_RECOMMENDED_DEPS = ("numpy", "xarray")
_OPTIONAL_DEPS = ("cantera",)


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure path is environment-specific
        return False, f"{module_name} import failed: {exc}"
    version = getattr(module, "__version__", None)
    if version:
        return True, f"{module_name} {version}"
    return True, f"{module_name} import ok"


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
    lines = path.read_text(encoding="utf-8").splitlines()
    payload, _ = _parse_block(lines, 0, 0, path=path)
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


def _extract_defaults_names(
    defaults_cfg: Mapping[str, Any],
) -> tuple[str, str]:
    defaults = defaults_cfg.get("defaults")
    if not isinstance(defaults, list):
        raise ConfigError("defaults must be a list when using fallback config load.")
    sim_name: Optional[str] = None
    pipeline_name: Optional[str] = None
    for entry in defaults:
        if isinstance(entry, Mapping):
            if "sim" in entry:
                sim_name = str(entry["sim"])
            if "pipeline" in entry:
                pipeline_name = str(entry["pipeline"])
    if sim_name is None or pipeline_name is None:
        raise ConfigError(
            "Fallback config load requires defaults for sim and pipeline."
        )
    return sim_name, pipeline_name


def _extract_fallback_overrides(
    overrides: Optional[Sequence[str]],
) -> tuple[Optional[str], Optional[str], Optional[str], list[str]]:
    sim_override: Optional[str] = None
    pipeline_override: Optional[str] = None
    store_root_override: Optional[str] = None
    ignored: list[str] = []
    if not overrides:
        return sim_override, pipeline_override, store_root_override, ignored
    for item in overrides:
        if not item or item == "--":
            continue
        if "=" not in item:
            ignored.append(item)
            continue
        key, value = item.split("=", 1)
        if key == "sim":
            sim_override = value
            continue
        if key == "pipeline":
            pipeline_override = value
            continue
        if key == "store.root":
            store_root_override = value
            continue
        ignored.append(item)
    return sim_override, pipeline_override, store_root_override, ignored


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
    (
        sim_override,
        pipeline_override,
        store_root_override,
        ignored,
    ) = _extract_fallback_overrides(overrides)
    if sim_override:
        sim_name = sim_override
    if pipeline_override:
        pipeline_name = pipeline_override
    if ignored:
        logger.warning(
            "Fallback config loader ignored overrides: %s",
            ", ".join(ignored),
        )

    sim_path = _resolve_yaml_path(config_dir / "sim", sim_name)
    pipeline_path = _resolve_yaml_path(config_dir / "pipeline", pipeline_name)
    sim_cfg = _load_simple_yaml(sim_path)
    pipeline_cfg = _load_simple_yaml(pipeline_path)
    steps = pipeline_cfg.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping) and step.get("sim") == "${sim}":
                step["sim"] = sim_cfg

    store_cfg = defaults_cfg.get("store", {})
    if not isinstance(store_cfg, Mapping):
        store_cfg = {}
    store_cfg = dict(store_cfg)
    if store_root_override:
        store_cfg["root"] = store_root_override

    return {
        "pipeline": dict(pipeline_cfg),
        "sim": dict(sim_cfg),
        "store": store_cfg,
    }


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


def run_doctor(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: Optional[Sequence[str]] = None,
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
        raise RxnPlatformError(
            f"Doctor failed to run dummy smoke pipeline: {exc}",
            user_message="Doctor failed while running the dummy smoke pipeline.",
            context={"error": str(exc)},
        ) from exc
    logger.info("Dummy smoke pipeline completed: %s", results)
    logger.info("Doctor completed successfully.")


__all__ = ["run_doctor"]
