"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from rxn_platform.errors import ArtifactError, ConfigError, RxnPlatformError
from rxn_platform.hydra_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONFIG_PATH,
    compose_config,
    format_config,
    resolve_config,
    seed_everything,
)
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.logging_utils import (
    configure_logging,
    log_exception,
    run_with_error_handling,
)
from rxn_platform.mechanism import MechanismCompiler
from rxn_platform.doctor import _compose_config_fallback, categorize_error, run_doctor
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks import sim as sim_task
from rxn_platform.tasks.runner import DEFAULT_STORE_ROOT
from rxn_platform.tasks.runner import run_task_from_config
from rxn_platform.clearml_adapter import maybe_register_clearml_task
from rxn_platform.dataset_registry import (
    DATASET_SCHEMA_VERSION,
    load_dataset_registry,
    register_dataset_entry,
    resolve_dataset_registry_path,
    save_dataset_registry,
)
from rxn_platform.run_store import (
    RUNS_ROOT,
    LEGACY_RUN_CONFIG_NAME,
    LEGACY_RUN_MANIFEST_NAME,
    list_runs,
    read_run_config,
    read_run_manifest,
    read_run_metrics,
    derive_run_contract_metadata,
    normalize_component,
    normalize_run_id,
    resolve_run_info,
    sync_conditions_table,
    utc_now_iso,
    write_run_config,
    write_run_manifest,
    write_run_metrics,
)

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:  # Optional dependency.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

_SUBCOMMANDS: Sequence[str] = (
    "help",
    "cfg",
    "run",
    "sim",
    "task",
    "pipeline",
    "dataset",
    "viz",
    "report",
    "doctor",
    "artifacts",
    "show-graph",
    "diff-mech",
    "list-runs",
    "import-legacy",
)


def _placeholder_handler(args: argparse.Namespace) -> None:
    raise RxnPlatformError(
        f"Command '{args.command}' is not implemented yet.",
        user_message=(
            f"'{args.command}' is not implemented yet. "
            f"Run `rxn {args.command} --help` for available options."
        ),
    )


def _cfg_handler(args: argparse.Namespace) -> None:
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    output = format_config(cfg)
    print(output, end="")


def _register_help_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parser: argparse.ArgumentParser,
) -> None:
    def _handler(_args: argparse.Namespace) -> None:
        parser.print_help()

    help_parser = subparsers.add_parser(
        "help",
        help="Show top-level help.",
        description="Show top-level help.",
    )
    help_parser.set_defaults(handler=_handler)


def _register_cfg_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    cfg_parser = subparsers.add_parser(
        "cfg",
        help="Compose and print Hydra config.",
        description="Compose and print Hydra config.",
    )
    cfg_parser.add_argument(
        "--config-path",
        default="configs",
        help="Path to the Hydra config directory.",
    )
    cfg_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    cfg_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: sim=dummy common.seed=123).",
    )
    cfg_parser.set_defaults(handler=_cfg_handler)


def _normalize_run_config(
    resolved: dict[str, object],
) -> tuple[dict[str, object], Path, str, str]:
    try:
        run_info = resolve_run_info(resolved)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid run configuration: {exc}") from exc
    run_root = run_info.root
    run_cfg = resolved.get("run")
    if not isinstance(run_cfg, Mapping):
        run_cfg = {}
    run_cfg = dict(run_cfg)
    run_cfg["exp"] = run_info.exp
    run_cfg["run_id"] = run_info.run_id
    run_cfg["root"] = str(run_root)
    resolved["run"] = run_cfg
    resolved["exp"] = run_info.exp
    resolved["run_id"] = run_info.run_id

    store_cfg = resolved.get("store")
    if not isinstance(store_cfg, Mapping):
        store_cfg = {}
    store_cfg = dict(store_cfg)
    store_root = run_root / "artifacts"
    store_cfg["root"] = str(store_root)
    resolved["store"] = store_cfg
    return resolved, run_root, run_info.exp, run_info.run_id


def _extract_compare_override(
    overrides: Optional[Sequence[str]],
) -> tuple[list[str], Optional[str]]:
    if not overrides:
        return [], None
    cleaned: list[str] = []
    compare_value: Optional[str] = None
    iterator = iter(overrides)
    for item in iterator:
        if item == "--compare":
            try:
                value = next(iterator)
            except StopIteration as exc:
                raise ConfigError("--compare requires a value.") from exc
            if not value or value.startswith("--"):
                raise ConfigError("--compare requires a value.")
            compare_value = value
            continue
        if isinstance(item, str) and item.startswith("--compare="):
            value = item.split("=", 1)[1]
            if not value:
                raise ConfigError("--compare requires a value.")
            compare_value = value
            continue
        cleaned.append(item)
    return cleaned, compare_value


def _run_handler(args: argparse.Namespace) -> None:
    overrides, compare_value = _extract_compare_override(args.overrides)
    if compare_value is not None:
        overrides.append(f"compare={compare_value}")
    legacy_keys = ("sim=", "task=", "pipeline=")
    if any(isinstance(item, str) and item.startswith(legacy_keys) for item in overrides) and not any(
        isinstance(item, str) and item.startswith("recipe=") for item in overrides
    ):
        logging.getLogger("rxn_platform.run").warning(
            "Legacy config override detected (sim/task/pipeline). "
            "Prefer recipe=... overrides for new runs."
        )
    logger = logging.getLogger("rxn_platform.run")
    resolved = _compose_pipeline_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=overrides,
        logger=logger,
    )
    resolved, run_root, exp, run_id = _normalize_run_config(resolved)
    run_root.mkdir(parents=True, exist_ok=True)
    seed_everything(resolved)

    manifest_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "exp": exp,
        "created_at": utc_now_iso(),
        "recipe": None,
        "store_root": str(Path(resolved["store"]["root"])),
    }
    manifest_payload.update(derive_run_contract_metadata(resolved))
    recipe_cfg = resolved.get("recipe")
    if isinstance(recipe_cfg, Mapping):
        recipe_name = recipe_cfg.get("name")
        if isinstance(recipe_name, str) and recipe_name.strip():
            manifest_payload["recipe"] = recipe_name
    write_run_manifest(run_root, manifest_payload)
    write_run_config(run_root, resolved)
    conditions_path = sync_conditions_table(resolved, run_root)
    if conditions_path is not None:
        updated_manifest = dict(manifest_payload)
        updated_manifest["conditions_path"] = str(conditions_path)
        write_run_manifest(run_root, updated_manifest)

    store = ArtifactStore(Path(resolved["store"]["root"]))
    results: dict[str, object]
    try:
        if "pipeline" in resolved:
            runner = PipelineRunner(store=store, logger=logger)
            results = runner.run(resolved)
        elif "task" in resolved:
            result = run_task_from_config(resolved, logger=logger)
            results = {"artifact_id": result.manifest.id}
        elif "sim" in resolved:
            result = sim_task.run(resolved, store=store)
            results = {"run_id": result.manifest.id}
        else:
            raise ConfigError(
                "run config must include pipeline, task, or sim settings."
            )
    except Exception as exc:
        category = categorize_error(exc)
        metrics_payload = {
            "schema_version": 1,
            "status": "failed",
            "error": str(exc),
            "error_category": category,
        }
        write_run_metrics(run_root, metrics_payload)
        raise

    metrics_payload = {
        "schema_version": 1,
        "status": "ok",
        "results": results,
    }
    if isinstance(results, Mapping):
        reduction_metrics: list[dict[str, Any]] = []
        validation_metrics: list[dict[str, Any]] = []
        mapping_metrics: list[dict[str, Any]] = []
        for value in results.values():
            if not isinstance(value, str) or not value.strip():
                continue
            metrics_path = store.root / "reduction" / value / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                payload = read_json(metrics_path)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                reduction_metrics.append(
                    {"reduction_id": value, "metrics": dict(payload)}
                )
        for value in results.values():
            if not isinstance(value, str) or not value.strip():
                continue
            metrics_path = store.root / "validation" / value / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                payload = read_json(metrics_path)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            entry = {"validation_id": value, "metrics": dict(payload)}
            validation_metrics.append(entry)
            if payload.get("kind") == "mapping_eval":
                mapping_metrics.append(entry)
        if reduction_metrics:
            metrics_payload["reduction_metrics"] = reduction_metrics
        if validation_metrics:
            metrics_payload["validation_metrics"] = validation_metrics
    if mapping_metrics:
        metrics_payload["mapping_metrics"] = mapping_metrics
    write_run_metrics(run_root, metrics_payload)
    maybe_register_clearml_task(
        resolved,
        run_root=run_root,
        manifest=manifest_payload,
        metrics=metrics_payload,
        logger=logger,
    )
    # Human-friendly index of key artifacts and rendered outputs under RunStore.
    viz_root = run_root / "viz"
    summary_payload: dict[str, Any] = {
        "schema_version": 1,
        "exp": exp,
        "run_id": run_id,
        "created_at": manifest_payload.get("created_at"),
        "status": metrics_payload.get("status"),
        "results": results,
        "paths": {
            "manifest": "manifest.json",
            "config": "config_resolved.yaml",
            "metrics": "metrics.json",
            "viz_index": "viz/index.html" if (viz_root / "index.html").exists() else None,
            "viz_network_index": "viz/network/index.json"
            if (viz_root / "network" / "index.json").exists()
            else None,
        },
    }
    if (viz_root / "network").exists():
        summary_payload["viz_network_files"] = {
            "dot": sorted(path.name for path in (viz_root / "network").glob("*.dot")),
            "svg": sorted(path.name for path in (viz_root / "network").glob("*.svg")),
        }
    if (viz_root / "timeseries").exists():
        summary_payload["viz_timeseries_files"] = sorted(
            path.name for path in (viz_root / "timeseries").glob("*.svg")
        )
    if (viz_root / "reduction").exists():
        summary_payload["viz_reduction_files"] = sorted(
            path.name for path in (viz_root / "reduction").glob("*.svg")
        )
    write_json_atomic(run_root / "summary.json", summary_payload)
    payload = {
        "exp": exp,
        "run_id": run_id,
        "run_root": str(run_root),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _warn_legacy_command(label: str) -> None:
    logger = logging.getLogger("rxn_platform.cli")
    logger.warning(
        "%s is a legacy entrypoint. Prefer `rxn run recipe=<name> run_id=<id>`.",
        label,
    )


def _register_run_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Run a recipe and store results under RunStore.",
        description="Run a recipe and store results under RunStore.",
    )
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    run_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    run_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: recipe=smoke run_id=demo exp=local).",
    )
    run_parser.set_defaults(handler=_run_handler)


def _list_runs_handler(args: argparse.Namespace) -> None:
    root = _resolve_runstore_root(args.runstore_root)
    runs = list_runs(root)
    limit = args.last
    if limit is not None:
        runs = runs[: max(0, limit)]
    if not runs:
        print(f"No runs found under {root}.")
        return
    for item in runs:
        print(f"{item.exp}/{item.run_id} {item.created_at}")


def _register_list_runs_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    list_parser = subparsers.add_parser(
        "list-runs",
        help="List RunStore entries.",
        description="List RunStore entries.",
    )
    list_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    list_parser.add_argument(
        "--root",
        dest="runstore_root",
        default=None,
        help="Alias for --runstore-root.",
    )
    list_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Show only the most recent N runs.",
    )
    list_parser.set_defaults(handler=_list_runs_handler)


def _dataset_list_handler(args: argparse.Namespace) -> None:
    registry_path = resolve_dataset_registry_path(args.root)
    try:
        registry = load_dataset_registry(registry_path)
    except ValueError as exc:
        raise ConfigError(f"Dataset registry invalid: {exc}") from exc
    datasets = registry.get("datasets", [])
    if not datasets:
        print(f"No datasets registered under {registry_path.parent}.")
        return
    for entry in datasets:
        if not isinstance(entry, Mapping):
            continue
        dataset_id = entry.get("dataset_id", "<unknown>")
        conditions_hash = entry.get("conditions_hash", "")
        mechanism_hash = entry.get("mechanism_hash", "")
        runs = entry.get("runs", [])
        run_count = len(runs) if isinstance(runs, list) else 0
        print(
            f"{dataset_id} conditions={conditions_hash} "
            f"mechanism={mechanism_hash} runs={run_count}"
        )


def _dataset_register_handler(args: argparse.Namespace) -> None:
    run_id = _parse_run_id_arg(args.run_id)
    exp = normalize_component(args.exp, "exp")
    if args.run_root:
        run_root = Path(args.run_root)
        if not run_root.is_absolute():
            run_root = (Path.cwd() / run_root).resolve()
    else:
        runstore_root = _resolve_runstore_root(args.runstore_root)
        run_root = runstore_root / exp / run_id
    if not run_root.exists():
        raise ConfigError(
            f"RunStore entry not found: {run_root}",
            user_message=f"RunStore entry not found: {run_root}",
        )
    manifest = read_run_manifest(run_root)
    if not isinstance(manifest, Mapping):
        raise ConfigError("RunStore manifest is missing or invalid.")
    manifest_run_id = manifest.get("run_id")
    manifest_exp = manifest.get("exp")
    if isinstance(manifest_run_id, str) and manifest_run_id and manifest_run_id != run_id:
        raise ConfigError("RunStore manifest run_id does not match --run-id.")
    if isinstance(manifest_exp, str) and manifest_exp and manifest_exp != exp:
        raise ConfigError("RunStore manifest exp does not match --exp.")
    config_payload = read_run_config(run_root)
    meta: dict[str, Optional[str]] = {}
    if isinstance(config_payload, Mapping):
        meta = derive_run_contract_metadata(config_payload)

    def _pick_hash(key: str) -> str:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            value = meta.get(key) if meta else None
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"RunStore manifest missing {key}.")
        return value

    conditions_hash = _pick_hash("conditions_hash")
    mechanism_hash = _pick_hash("mechanism_hash")

    store_root = run_root / "artifacts"
    store_value = manifest.get("store_root")
    if isinstance(store_value, str) and store_value.strip():
        store_root = Path(store_value)

    registry_path = resolve_dataset_registry_path(args.root)
    try:
        registry = load_dataset_registry(registry_path)
    except ValueError as exc:
        raise ConfigError(f"Dataset registry invalid: {exc}") from exc
    run_ref = {
        "run_id": run_id,
        "exp": exp,
        "run_root": str(run_root),
        "store_root": str(store_root),
    }
    try:
        entry, reused, run_added = register_dataset_entry(
            registry,
            conditions_hash=conditions_hash,
            mechanism_hash=mechanism_hash,
            run_ref=run_ref,
            schema_version=DATASET_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise ConfigError(f"Dataset registry update failed: {exc}") from exc
    try:
        save_dataset_registry(registry_path, registry)
    except OSError as exc:
        raise ConfigError(f"Failed to write dataset registry: {exc}") from exc

    dataset_id = entry.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ConfigError("Dataset registry returned invalid dataset_id.")

    updated_manifest = dict(manifest)
    existing_id = updated_manifest.get("dataset_id")
    if isinstance(existing_id, str) and existing_id and existing_id != dataset_id:
        raise ConfigError("RunStore manifest dataset_id does not match registry.")
    updated_manifest["dataset_id"] = dataset_id
    updated_manifest["dataset_registry"] = str(registry_path)
    updated_manifest.setdefault("dataset_schema_version", DATASET_SCHEMA_VERSION)
    write_run_manifest(run_root, updated_manifest)

    payload = {
        "dataset_id": dataset_id,
        "reused": reused,
        "run_added": run_added,
        "registry_path": str(registry_path),
        "run_id": run_id,
        "exp": exp,
        "conditions_hash": conditions_hash,
        "mechanism_hash": mechanism_hash,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _register_dataset_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Dataset registry commands.",
        description="Dataset registry commands.",
    )
    dataset_parser.set_defaults(handler=_placeholder_handler, parser=dataset_parser)
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        metavar="DATASET_COMMAND",
    )

    list_parser = dataset_subparsers.add_parser(
        "list",
        help="List registered datasets.",
        description="List registered datasets.",
    )
    list_parser.add_argument(
        "--root",
        default="datasets",
        help="Path to the dataset registry root (default: datasets).",
    )
    list_parser.set_defaults(handler=_dataset_list_handler)

    register_parser = dataset_subparsers.add_parser(
        "register",
        help="Register a run into the dataset registry.",
        description="Register a run into the dataset registry.",
    )
    register_parser.add_argument(
        "--run-id",
        required=True,
        help="RunStore run_id to register.",
    )
    register_parser.add_argument(
        "--exp",
        default="default",
        help="Experiment name (default: default).",
    )
    register_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    register_parser.add_argument(
        "--run-root",
        help="Path to the RunStore run directory (overrides runs/<exp>/<run_id>).",
    )
    register_parser.add_argument(
        "--root",
        default="datasets",
        help="Path to the dataset registry root (default: datasets).",
    )
    register_parser.set_defaults(handler=_dataset_register_handler)


_KNOWN_ARTIFACT_KINDS = (
    "runs",
    "observables",
    "graphs",
    "features",
    "sensitivity",
    "models",
    "reduction",
    "validation",
    "reports",
    "pipelines",
)


def _looks_like_artifact_root(path: Path) -> bool:
    for kind in _KNOWN_ARTIFACT_KINDS:
        if (path / kind).exists():
            return True
    return False


def _detect_legacy_artifact_root(legacy_root: Path) -> tuple[Path, str]:
    if (legacy_root / "artifacts").is_dir():
        return legacy_root / "artifacts", "runstore"
    if (legacy_root / LEGACY_RUN_MANIFEST_NAME).exists() or (
        legacy_root / LEGACY_RUN_CONFIG_NAME
    ).exists():
        if (legacy_root / "artifacts").is_dir():
            return legacy_root / "artifacts", "runstore"
        raise ConfigError(
            f"Legacy runstore at {legacy_root} is missing artifacts/."
        )
    if _looks_like_artifact_root(legacy_root):
        return legacy_root, "artifact_store"
    raise ConfigError(
        f"Legacy root does not look like a runstore or artifact store: {legacy_root}"
    )


def _load_legacy_config_payload(legacy_root: Path) -> dict[str, object]:
    config_payload = read_run_config(legacy_root)
    if isinstance(config_payload, Mapping):
        return dict(config_payload)
    return {"legacy": {"root": str(legacy_root)}}


def _load_legacy_metrics_payload(legacy_root: Path) -> dict[str, object]:
    metrics = read_run_metrics(legacy_root)
    if isinstance(metrics, Mapping):
        payload = dict(metrics)
        if "schema_version" not in payload:
            return {
                "schema_version": 1,
                "status": "legacy",
                "results": payload,
            }
        return payload
    return {
        "schema_version": 1,
        "legacy": {"root": str(legacy_root)},
    }


def _register_legacy_artifacts(run_root: Path, artifact_root: Path) -> Path:
    target = run_root / "artifacts"
    if target.exists():
        return target
    try:
        target.symlink_to(artifact_root, target_is_directory=True)
        return target
    except OSError:
        return artifact_root


def _import_legacy_handler(args: argparse.Namespace) -> None:
    legacy_root = Path(args.legacy_root)
    if not legacy_root.is_absolute():
        legacy_root = (Path.cwd() / legacy_root).resolve()
    if not legacy_root.exists():
        raise ConfigError(f"Legacy root not found: {legacy_root}")

    run_id = normalize_run_id(args.run_id)
    exp = normalize_component(args.exp or "legacy", "exp")
    run_root = _resolve_run_root(exp, run_id, runstore_root=args.runstore_root)
    if run_root.exists():
        raise ConfigError(
            f"RunStore entry already exists: {run_root}",
            user_message=(
                f"RunStore entry already exists: {run_root}. "
                "Choose a different run_id or exp."
            ),
        )

    artifact_root, source_kind = _detect_legacy_artifact_root(legacy_root)
    run_root.mkdir(parents=True, exist_ok=True)
    store_root = _register_legacy_artifacts(run_root, artifact_root)

    config_payload = _load_legacy_config_payload(legacy_root)
    manifest_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "exp": exp,
        "created_at": utc_now_iso(),
        "recipe": None,
        "store_root": str(store_root),
        "legacy": {
            "root": str(legacy_root),
            "artifact_root": str(artifact_root),
            "source_kind": source_kind,
        },
    }
    manifest_payload.update(derive_run_contract_metadata(config_payload))
    write_run_manifest(run_root, manifest_payload)
    write_run_config(run_root, config_payload)

    metrics_payload = _load_legacy_metrics_payload(legacy_root)
    write_run_metrics(run_root, metrics_payload)

    payload = {
        "exp": exp,
        "run_id": run_id,
        "run_root": str(run_root),
        "store_root": str(store_root),
        "legacy_root": str(legacy_root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _sim_help_handler(args: argparse.Namespace) -> None:
    parser = getattr(args, "parser", None)
    if parser is not None:
        parser.print_help()
    raise SystemExit(2)


def _resolve_config_dir(path: str) -> Path:
    config_dir = Path(path)
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    return config_dir


def _resolve_runstore_root(value: Optional[str]) -> Path:
    root = Path(value) if value else RUNS_ROOT
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root


def _resolve_run_root(
    exp: str,
    run_id: str,
    *,
    runstore_root: Optional[str],
) -> Path:
    root = _resolve_runstore_root(runstore_root)
    return root / exp / run_id


def _select_sim_template(backend: str, config_dir: Path) -> Optional[Path]:
    sim_dir = config_dir / "sim"
    if not sim_dir.exists():
        return None
    direct = sim_dir / f"{backend}.yaml"
    if direct.exists():
        return direct
    if backend == "cantera":
        alt = sim_dir / "cantera_min.yaml"
        if alt.exists():
            return alt
    placeholder = sim_dir / "placeholder.yaml"
    if placeholder.exists():
        return placeholder
    return None


def _sim_init_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn sim init")
    backend = args.backend
    if not isinstance(backend, str) or not backend.strip():
        raise ConfigError("backend must be a non-empty string.")
    config_dir = _resolve_config_dir(args.config_path)
    if not config_dir.exists():
        raise ConfigError(f"Config directory not found: {config_dir}")

    output = args.output
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
    else:
        output_path = config_dir / "sim" / f"{backend}_init.yaml"

    if output_path.exists() and not args.force:
        raise ConfigError(
            f"Output file already exists: {output_path}",
            user_message=(
                f"Output file already exists: {output_path}. "
                "Use --force to overwrite."
            ),
        )

    template = _select_sim_template(backend, config_dir)
    if template is None:
        contents = f"name: {backend}\n"
    else:
        contents = template.read_text(encoding="utf-8")
        if template.name == "placeholder.yaml":
            contents = f"name: {backend}\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contents, encoding="utf-8")
    print(str(output_path))


def _sim_validate_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn sim validate")
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    sim_task.validate_config(cfg)
    print("sim config ok")


def _sim_run_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn sim run")
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    resolved = resolve_config(cfg)
    store_root = _extract_store_root_from_cfg(resolved)
    store = ArtifactStore(store_root)

    cache_bust = args.cache_bust
    skip_cache = bool(args.skip_cache or cache_bust)
    if skip_cache and not cache_bust:
        cache_bust = utc_now_iso()

    logger = logging.getLogger("rxn_platform.sim")
    if cache_bust:
        logger.info("Cache skip enabled: sim.cache_bust=%s", cache_bust)
    result = sim_task.run(resolved, store=store, cache_bust=cache_bust)
    if result.reused:
        logger.info(
            "Cache hit for run_id=%s; skipping simulation.", result.manifest.id
        )
    else:
        logger.info("Cache miss for run_id=%s; simulation executed.", result.manifest.id)
    print(f"run_id={result.manifest.id}")


def _sim_viz_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn sim viz")
    root = _resolve_store_root(args.root)
    _require_artifact_root(root)
    store = ArtifactStore(root)
    cfg = {
        "viz": {
            "run_id": args.run_id,
            "title": args.title,
            "top_species": args.top_species,
            "max_points": args.max_points,
        }
    }
    logger = logging.getLogger("rxn_platform.sim")
    result = sim_task.viz(cfg, store=store)
    if result.reused:
        logger.info("Report cache hit for report_id=%s.", result.manifest.id)
    else:
        logger.info("Report generated: report_id=%s.", result.manifest.id)
    print(f"report_id={result.manifest.id}")


def _register_sim_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sim_parser = subparsers.add_parser(
        "sim",
        help="Legacy simulation commands (prefer `run recipe=...`).",
        description="Legacy simulation commands (prefer `run recipe=...`).",
    )
    sim_parser.set_defaults(handler=_sim_help_handler, parser=sim_parser)
    sim_subparsers = sim_parser.add_subparsers(
        dest="sim_command",
        metavar="SIM_COMMAND",
    )

    init_parser = sim_subparsers.add_parser(
        "init",
        help="Generate a simulation config template.",
        description="Generate a simulation config template.",
    )
    init_parser.add_argument(
        "--backend",
        default="dummy",
        help="Backend name to use for the template.",
    )
    init_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    init_parser.add_argument(
        "--output",
        help="Output path for the generated config.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    init_parser.set_defaults(handler=_sim_init_handler)

    validate_parser = sim_subparsers.add_parser(
        "validate",
        help="Validate simulation config.",
        description="Validate simulation config.",
    )
    validate_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    validate_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    validate_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: sim=dummy sim.time.steps=3).",
    )
    validate_parser.set_defaults(handler=_sim_validate_handler)

    run_parser = sim_subparsers.add_parser(
        "run",
        help="Run a simulation backend.",
        description="Run a simulation backend.",
    )
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    run_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    run_parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Bypass cache by adding a cache_bust entry.",
    )
    run_parser.add_argument(
        "--cache-bust",
        help="Explicit cache-bust value (implies --skip-cache).",
    )
    run_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: sim=dummy store.root=artifacts).",
    )
    run_parser.set_defaults(handler=_sim_run_handler)

    viz_parser = sim_subparsers.add_parser(
        "viz",
        help="Render a quick report for a run artifact.",
        description="Render a quick report for a run artifact.",
    )
    viz_parser.add_argument("run_id", help="Run artifact id.")
    viz_parser.add_argument(
        "--root",
        default=DEFAULT_STORE_ROOT,
        help="Path to the artifact store root.",
    )
    viz_parser.add_argument(
        "--title",
        default="Simulation Report",
        help="Report title.",
    )
    viz_parser.add_argument(
        "--top-species",
        type=int,
        default=sim_task.DEFAULT_TOP_SPECIES,
        help="Number of top species to plot.",
    )
    viz_parser.add_argument(
        "--max-points",
        type=int,
        default=sim_task.DEFAULT_MAX_POINTS,
        help="Maximum number of time points to plot.",
    )
    viz_parser.set_defaults(handler=_sim_viz_handler)


def _task_help_handler(args: argparse.Namespace) -> None:
    parser = getattr(args, "parser", None)
    if parser is not None:
        parser.print_help()
    raise SystemExit(2)


def _task_run_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn task run")
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    logger = logging.getLogger("rxn_platform")
    result = run_task_from_config(cfg, logger=logger)
    print(f"artifact_id={result.manifest.id}")


def _register_task_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    task_parser = subparsers.add_parser(
        "task",
        help="Legacy task runner commands (prefer `run recipe=...`).",
        description="Legacy task runner commands (prefer `run recipe=...`).",
    )
    task_parser.set_defaults(handler=_task_help_handler, parser=task_parser)
    task_subparsers = task_parser.add_subparsers(
        dest="task_command",
        metavar="TASK_COMMAND",
    )
    run_parser = task_subparsers.add_parser(
        "run",
        help="Run a registered task.",
        description="Run a registered task.",
    )
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    run_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    run_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: task=foo task.name=foo).",
    )
    run_parser.set_defaults(handler=_task_run_handler)


def _pipeline_help_handler(args: argparse.Namespace) -> None:
    parser = getattr(args, "parser", None)
    if parser is not None:
        parser.print_help()
    raise SystemExit(2)


def _extract_store_root_from_cfg(cfg: Mapping[str, object]) -> Path:
    store_cfg = cfg.get("store")
    if store_cfg is None:
        root: object = DEFAULT_STORE_ROOT
    elif isinstance(store_cfg, Mapping):
        root = store_cfg.get("root", DEFAULT_STORE_ROOT)
    else:
        raise ConfigError("store must be a mapping if provided.")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError("store.root must be a non-empty string.")
    return Path(root)


def _compose_pipeline_config(
    *,
    config_path: str,
    config_name: str,
    overrides: Optional[Sequence[str]],
    logger: logging.Logger,
) -> dict[str, object]:
    try:
        import hydra  # noqa: F401
    except Exception as exc:
        logger.warning(
            "Dependency missing: hydra import failed: %s (using fallback config loader).",
            exc,
        )
        return _compose_config_fallback(
            config_path=config_path,
            config_name=config_name,
            overrides=overrides,
            logger=logger,
        )
    cfg = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )
    return resolve_config(cfg)


def _pipeline_run_handler(args: argparse.Namespace) -> None:
    _warn_legacy_command("rxn pipeline run")
    logger = logging.getLogger("rxn_platform.pipeline")
    resolved = _compose_pipeline_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
        logger=logger,
    )
    store_root = _extract_store_root_from_cfg(resolved)
    store = ArtifactStore(store_root)
    runner = PipelineRunner(store=store, logger=logger)
    results = runner.run(resolved)
    payload = json.dumps(results, indent=2, sort_keys=True, ensure_ascii=True)
    print(payload)


def _register_pipeline_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Legacy pipeline runner commands (prefer `run recipe=...`).",
        description="Legacy pipeline runner commands (prefer `run recipe=...`).",
    )
    pipeline_parser.set_defaults(handler=_pipeline_help_handler, parser=pipeline_parser)
    pipeline_subparsers = pipeline_parser.add_subparsers(
        dest="pipeline_command",
        metavar="PIPELINE_COMMAND",
    )
    run_parser = pipeline_subparsers.add_parser(
        "run",
        help="Run a pipeline from Hydra config.",
        description="Run a pipeline from Hydra config.",
    )
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    run_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    run_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: pipeline=smoke sim=dummy).",
    )
    run_parser.set_defaults(handler=_pipeline_run_handler)


def _doctor_handler(args: argparse.Namespace) -> None:
    run_doctor(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
        strict=args.strict,
        runstore_root=args.runstore_root,
        logger=logging.getLogger("rxn_platform.doctor"),
    )


def _register_doctor_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics and smoke checks.",
        description="Run environment diagnostics and smoke checks.",
    )
    doctor_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hydra config directory.",
    )
    doctor_parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name (without extension).",
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if RunStore contract violations are detected.",
    )
    doctor_parser.add_argument(
        "--runstore-root",
        help="Override the RunStore root used for contract checks.",
    )
    doctor_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: sim=dummy store.root=artifacts).",
    )
    doctor_parser.set_defaults(handler=_doctor_handler)


def _artifact_component(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} must be a non-empty string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ArtifactError(f"{label} must be a single path component: {value!r}")
    if value in {".", ".."}:
        raise ArtifactError(f"{label} must not be '.' or '..'.")
    return value


def _resolve_store_root(root: str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _require_artifact_root(root: Path) -> None:
    if not root.exists():
        raise ArtifactError(
            f"Artifact root not found: {root}",
            user_message=(
                f"Artifact root not found: {root}. "
                "Run a task to create artifacts or pass --root."
            ),
        )
    if not root.is_dir():
        raise ArtifactError(f"Artifact root is not a directory: {root}")


def _list_artifact_ids(kind_dir: Path) -> list[str]:
    artifact_ids: list[str] = []
    for entry in kind_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if (entry / "manifest.yaml").exists():
            artifact_ids.append(entry.name)
    return sorted(artifact_ids)


def _artifacts_help_handler(args: argparse.Namespace) -> None:
    parser = getattr(args, "parser", None)
    if parser is not None:
        parser.print_help()
    raise SystemExit(2)


def _artifacts_ls_handler(args: argparse.Namespace) -> None:
    root = _resolve_store_root(args.root)
    _require_artifact_root(root)
    kind_filter = args.kind
    if kind_filter:
        kind = _artifact_component("kind", kind_filter)
        kind_dir = root / kind
        kinds = [kind_dir]
    else:
        kinds = sorted(
            [
                entry
                for entry in root.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ],
            key=lambda entry: entry.name,
        )
    if not kinds:
        print(f"No artifacts found under {root}.")
        return
    for kind_dir in kinds:
        ids: list[str] = []
        if kind_dir.exists():
            if not kind_dir.is_dir():
                raise ArtifactError(
                    f"Artifact kind path is not a directory: {kind_dir}"
                )
            ids = _list_artifact_ids(kind_dir)
        print(f"{kind_dir.name}:")
        if ids:
            for artifact_id in ids:
                print(f"  {artifact_id}")
        else:
            print("  (none)")


def _artifacts_show_handler(args: argparse.Namespace) -> None:
    root = _resolve_store_root(args.root)
    _require_artifact_root(root)
    kind = _artifact_component("kind", args.kind)
    artifact_id = _artifact_component("artifact_id", args.artifact_id)
    store = ArtifactStore(root)
    try:
        manifest = store.read_manifest(kind, artifact_id)
    except ArtifactError as exc:
        manifest_path = store.manifest_path(kind, artifact_id)
        raise ArtifactError(
            f"Artifact not found: {kind}/{artifact_id}",
            user_message=(
                f"Artifact {kind}/{artifact_id} not found. "
                f"Expected manifest at {manifest_path}."
            ),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            f"Invalid manifest for {kind}/{artifact_id}: {exc}",
            user_message=f"Invalid manifest for {kind}/{artifact_id}: {exc}",
        ) from exc
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=True))


def _artifacts_open_report_handler(args: argparse.Namespace) -> None:
    root = _resolve_store_root(args.root)
    _require_artifact_root(root)
    artifact_id = _artifact_component("artifact_id", args.artifact_id)
    report_dir = root / "reports" / artifact_id
    report_path = report_dir / "index.html"
    if not report_path.exists():
        raise ArtifactError(
            f"Report not found: {artifact_id}",
            user_message=(
                f"Report {artifact_id} not found. "
                f"Expected report at {report_path}."
            ),
        )
    print(report_path.resolve().as_uri())


def _find_artifact_kind(store_root: Path, artifact_id: str) -> Optional[str]:
    if not store_root.exists():
        return None
    for entry in store_root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        manifest_path = entry / artifact_id / "manifest.yaml"
        if manifest_path.exists():
            return entry.name
    return None


def _report_handler(args: argparse.Namespace) -> None:
    run_id = _parse_run_id_arg(args.run_id)
    exp = normalize_component(args.exp, "exp")
    run_root = _resolve_run_root(exp, run_id, runstore_root=args.runstore_root)
    if not run_root.exists():
        raise ConfigError(
            f"RunStore entry not found: {run_root}",
            user_message=f"RunStore entry not found: {run_root}",
        )
    manifest = read_run_manifest(run_root)
    store_root = run_root / "artifacts"
    if isinstance(manifest, Mapping):
        store_value = manifest.get("store_root")
        if isinstance(store_value, str) and store_value.strip():
            store_root = Path(store_value)
    store = ArtifactStore(store_root)
    metrics = read_run_metrics(run_root)
    if not isinstance(metrics, Mapping):
        raise ConfigError("RunStore metrics are missing or invalid.")
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        raise ConfigError("RunStore metrics missing results mapping.")

    inputs: list[dict[str, str]] = []
    for value in results.values():
        if not isinstance(value, str) or not value.strip():
            continue
        kind = _find_artifact_kind(store_root, value)
        if kind is None:
            continue
        inputs.append({"kind": kind, "id": value})
    if not inputs:
        raise ConfigError("No artifacts found to report.")

    title = args.title or f"Run {run_id} report"
    dashboard = args.dashboard or "base"
    cfg = {"viz": {"inputs": inputs, "title": title, "dashboard": dashboard}}
    from rxn_platform.tasks import viz as viz_task

    result = viz_task.run(cfg, store=store)
    print(f"report_id={result.manifest.id}")


def _register_artifacts_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    artifacts_parser = subparsers.add_parser(
        "artifacts",
        help="Inspect stored artifacts.",
        description="Inspect stored artifacts.",
    )
    artifacts_parser.add_argument(
        "--root",
        default=DEFAULT_STORE_ROOT,
        help="Path to the artifact store root.",
    )
    artifacts_parser.set_defaults(handler=_artifacts_help_handler, parser=artifacts_parser)
    artifacts_subparsers = artifacts_parser.add_subparsers(
        dest="artifacts_command",
        metavar="ARTIFACTS_COMMAND",
    )

    list_parser = artifacts_subparsers.add_parser(
        "ls",
        help="List artifacts by kind.",
        description="List artifacts by kind.",
    )
    list_parser.add_argument(
        "--kind",
        help="Limit listing to a single artifact kind.",
    )
    list_parser.set_defaults(handler=_artifacts_ls_handler)

    show_parser = artifacts_subparsers.add_parser(
        "show",
        help="Show an artifact manifest.",
        description="Show an artifact manifest.",
    )
    show_parser.add_argument("kind", help="Artifact kind.")
    show_parser.add_argument("artifact_id", help="Artifact id.")
    show_parser.set_defaults(handler=_artifacts_show_handler)

    open_parser = artifacts_subparsers.add_parser(
        "open-report",
        help="Print the report URL for a report artifact.",
        description="Print the report URL for a report artifact.",
    )
    open_parser.add_argument("artifact_id", help="Report artifact id.")
    open_parser.set_defaults(handler=_artifacts_open_report_handler)


def _parse_run_id_arg(raw_value: str) -> str:
    value = raw_value.strip()
    if "=" in value:
        key, candidate = value.split("=", 1)
        if key.strip() != "run_id":
            raise ConfigError(
                "show-graph expects run_id or run_id=<value> as the first argument."
            )
        value = candidate.strip()
    return normalize_component(value, "run_id")


def _select_graph_id(
    results: Mapping[str, object],
    store: ArtifactStore,
) -> Optional[str]:
    for key in ("graph", "graphs", "temporal_graph", "temporal_flux", "species_graph"):
        value = results.get(key)
        if isinstance(value, str):
            return value
    artifact_id = results.get("artifact_id")
    if isinstance(artifact_id, str):
        return artifact_id
    for value in results.values():
        if not isinstance(value, str):
            continue
        try:
            store.read_manifest("graphs", value)
        except ArtifactError:
            continue
        return value
    return None


def _select_reduction_id(
    results: Mapping[str, object],
    store: ArtifactStore,
) -> Optional[str]:
    for key in ("amore", "reduction", "reduce", "amore_search"):
        value = results.get(key)
        if isinstance(value, str):
            return value
    artifact_id = results.get("artifact_id")
    if isinstance(artifact_id, str):
        return artifact_id
    for value in results.values():
        if not isinstance(value, str):
            continue
        try:
            store.read_manifest("reduction", value)
        except ArtifactError:
            continue
        return value
    return None


def _diff_mech_handler(args: argparse.Namespace) -> None:
    run_id = _parse_run_id_arg(args.run_id)
    exp = normalize_component(args.exp, "exp")
    run_root = _resolve_run_root(exp, run_id, runstore_root=args.runstore_root)
    if not run_root.exists():
        raise ConfigError(
            f"RunStore entry not found: {run_root}",
            user_message=f"RunStore entry not found: {run_root}",
        )
    manifest = read_run_manifest(run_root)
    store_root = run_root / "artifacts"
    if isinstance(manifest, Mapping):
        store_value = manifest.get("store_root")
        if isinstance(store_value, str) and store_value.strip():
            store_root = Path(store_value)
    store = ArtifactStore(store_root)
    metrics = read_run_metrics(run_root)
    if not isinstance(metrics, Mapping):
        raise ConfigError("RunStore metrics are missing or invalid.")
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        raise ConfigError("RunStore metrics missing results mapping.")

    reduction_id = args.reduction_id or _select_reduction_id(results, store)
    if reduction_id is None:
        raise ConfigError("No reduction artifact id found for diff-mech.")
    reduction_dir = store.artifact_dir("reduction", reduction_id)
    reduced_mech = reduction_dir / "reduced_mech.yaml"
    if not reduced_mech.exists():
        reduced_mech = reduction_dir / "mechanism.yaml"
    if not reduced_mech.exists():
        raise ConfigError(f"Reduced mechanism not found under {reduction_dir}.")

    manifest_payload = store.read_manifest("reduction", reduction_id).to_dict()
    inputs = manifest_payload.get("inputs", {})
    mechanism_path = None
    if isinstance(inputs, Mapping):
        mechanism_path = inputs.get("mechanism")
    if not isinstance(mechanism_path, str) or not mechanism_path.strip():
        raise ConfigError("Reduction manifest does not include mechanism path.")

    base_compiler = MechanismCompiler.from_path(mechanism_path)
    reduced_compiler = MechanismCompiler.from_path(reduced_mech)
    base_count = base_compiler.reaction_count()
    reduced_count = reduced_compiler.reaction_count()
    disabled_count = max(0, base_count - reduced_count)

    payload = {
        "run_id": run_id,
        "exp": exp,
        "reduction_id": reduction_id,
        "mechanism": mechanism_path,
        "reduced_mechanism": str(reduced_mech),
        "reaction_count_base": base_count,
        "reaction_count_reduced": reduced_count,
        "disabled_reactions": disabled_count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _show_graph_handler(args: argparse.Namespace) -> None:
    run_id = _parse_run_id_arg(args.run_id)
    exp = normalize_component(args.exp, "exp")
    run_root = _resolve_run_root(exp, run_id, runstore_root=args.runstore_root)
    if not run_root.exists():
        raise ConfigError(
            f"RunStore entry not found: {run_root}",
            user_message=f"RunStore entry not found: {run_root}",
        )
    manifest = read_run_manifest(run_root)
    store_root = run_root / "artifacts"
    if isinstance(manifest, Mapping):
        store_value = manifest.get("store_root")
        if isinstance(store_value, str) and store_value.strip():
            store_root = Path(store_value)
    store = ArtifactStore(store_root)
    metrics = read_run_metrics(run_root)
    if not isinstance(metrics, Mapping):
        raise ConfigError("RunStore metrics are missing or invalid.")
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        raise ConfigError("RunStore metrics missing results mapping.")
    graph_id = args.graph_id
    if graph_id is None:
        graph_id = _select_graph_id(results, store)
    if graph_id is None:
        raise ConfigError("Graph artifact id not found in RunStore metrics.")
    graph_id = normalize_component(graph_id, "graph_id")

    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_path = graph_dir / "graph.json"
    if not graph_path.exists():
        raise ConfigError(
            f"graph.json not found for graphs/{graph_id}",
            user_message=f"graph.json not found for graphs/{graph_id}",
        )
    try:
        graph_payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(graph_payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")

    layer_index = int(args.layer)
    species_graph = graph_payload.get("species_graph")
    layer_path = None
    if isinstance(species_graph, Mapping):
        layers = species_graph.get("layers")
        if isinstance(layers, Sequence):
            for entry in layers:
                if not isinstance(entry, Mapping):
                    continue
                if entry.get("index") == layer_index:
                    candidate = entry.get("path")
                    if isinstance(candidate, str):
                        layer_path = candidate
                        break
    if layer_path is None:
        layer_path = f"species_graph/layer_{layer_index:03d}.npz"

    matrix_path = graph_dir / layer_path
    if not matrix_path.exists():
        raise ConfigError(
            f"Layer file not found: {matrix_path}",
            user_message=f"Layer file not found: {matrix_path}",
        )
    if sp is not None:
        matrix = sp.load_npz(matrix_path)
        shape = matrix.shape
        nnz = int(matrix.nnz)
    else:
        if np is None:
            raise ConfigError("numpy is required to read layer matrices.")
        data = np.load(matrix_path)
        if "shape" in data:
            shape = tuple(int(x) for x in data["shape"])
        else:
            shape = (len(data["indptr"]) - 1, len(data["indptr"]) - 1)
        nnz = int(len(data["data"]))

    density = float(nnz) / float(shape[0] * shape[1]) if shape[0] and shape[1] else 0.0
    payload = {
        "run_id": run_id,
        "exp": exp,
        "graph_id": graph_id,
        "layer": layer_index,
        "layer_path": str(matrix_path),
        "shape": list(shape),
        "nnz": nnz,
        "density": density,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _register_show_graph_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    show_parser = subparsers.add_parser(
        "show-graph",
        help="Summarize a temporal graph layer stored under RunStore.",
        description="Summarize a temporal graph layer stored under RunStore.",
    )
    show_parser.add_argument(
        "run_id",
        help="RunStore run_id or run_id=<value>.",
    )
    show_parser.add_argument(
        "--exp",
        default="default",
        help="Experiment name (default: default).",
    )
    show_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    show_parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to summarize.",
    )
    show_parser.add_argument(
        "--graph-id",
        default=None,
        help="Override graph artifact id if metrics do not include one.",
    )
    show_parser.set_defaults(handler=_show_graph_handler)


def _register_diff_mech_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    diff_parser = subparsers.add_parser(
        "diff-mech",
        help="Summarize baseline vs reduced mechanism counts for a RunStore entry.",
        description="Summarize baseline vs reduced mechanism counts for a RunStore entry.",
    )
    diff_parser.add_argument(
        "run_id",
        help="RunStore run_id or run_id=<value>.",
    )
    diff_parser.add_argument(
        "--exp",
        default="default",
        help="Experiment name (default: default).",
    )
    diff_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    diff_parser.add_argument(
        "--reduction-id",
        default=None,
        help="Override reduction artifact id if metrics do not include one.",
    )
    diff_parser.set_defaults(handler=_diff_mech_handler)


def _register_report_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    report_parser = subparsers.add_parser(
        "report",
        help="Render a report for a RunStore entry.",
        description="Render a report for a RunStore entry.",
    )
    report_parser.add_argument(
        "run_id",
        help="RunStore run_id.",
    )
    report_parser.add_argument(
        "--exp",
        default="default",
        help="Experiment name (default: default).",
    )
    report_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    report_parser.add_argument(
        "--title",
        default=None,
        help="Override report title.",
    )
    report_parser.add_argument(
        "--dashboard",
        default="base",
        help="Dashboard type (default: base).",
    )
    report_parser.set_defaults(handler=_report_handler)


def _register_import_legacy_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    import_parser = subparsers.add_parser(
        "import-legacy",
        help="Register legacy outputs into RunStore.",
        description="Register legacy outputs into RunStore without copying data.",
    )
    import_parser.add_argument(
        "legacy_root",
        help="Path to legacy output (runstore or artifact root).",
    )
    import_parser.add_argument(
        "--exp",
        default="legacy",
        help="Experiment name to store under (default: legacy).",
    )
    import_parser.add_argument(
        "--run-id",
        default="auto",
        help="Run id to assign (default: auto).",
    )
    import_parser.add_argument(
        "--runstore-root",
        default=None,
        help="Path to the RunStore root (default: runs).",
    )
    import_parser.set_defaults(handler=_import_legacy_handler)


def _build_parser(subcommands: Iterable[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rxn",
        description="rxn_platform command line interface.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show full traceback on errors.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in subcommands:
        if name == "help":
            _register_help_subcommand(subparsers, parser)
            continue
        if name == "cfg":
            _register_cfg_subcommand(subparsers)
            continue
        if name == "run":
            _register_run_subcommand(subparsers)
            continue
        if name == "sim":
            _register_sim_subcommand(subparsers)
            continue
        if name == "task":
            _register_task_subcommand(subparsers)
            continue
        if name == "pipeline":
            _register_pipeline_subcommand(subparsers)
            continue
        if name == "dataset":
            _register_dataset_subcommand(subparsers)
            continue
        if name == "doctor":
            _register_doctor_subcommand(subparsers)
            continue
        if name == "import-legacy":
            _register_import_legacy_subcommand(subparsers)
            continue
        if name == "artifacts":
            _register_artifacts_subcommand(subparsers)
            continue
        if name == "show-graph":
            _register_show_graph_subcommand(subparsers)
            continue
        if name == "diff-mech":
            _register_diff_mech_subcommand(subparsers)
            continue
        if name == "report":
            _register_report_subcommand(subparsers)
            continue
        if name == "list-runs":
            _register_list_runs_subcommand(subparsers)
            continue
        subparser = subparsers.add_parser(
            name,
            help=f"{name} commands (placeholder).",
            description=f"{name} commands (placeholder).",
        )
        subparser.set_defaults(handler=_placeholder_handler)
    return parser


def _cli_main(
    *,
    cli_logger: logging.Logger,
    argv: Optional[Sequence[str]] = None,
) -> None:
    parser = _build_parser(_SUBCOMMANDS)
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        raise SystemExit(2)
    try:
        args.handler(args)
    except RxnPlatformError as exc:
        log_exception(cli_logger, exc, show_traceback=args.traceback)
        raise SystemExit(1) from None


def main() -> None:
    """Entry point with standard logging/error handling."""
    logger = configure_logging()
    run_with_error_handling(_cli_main, logger=logger, cli_logger=logger)


if __name__ == "__main__":
    main()
