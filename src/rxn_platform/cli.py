"""CLI entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Optional, Sequence

from rxn_platform.errors import ArtifactError, ConfigError, RxnPlatformError
from rxn_platform.hydra_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONFIG_PATH,
    compose_config,
    format_config,
    resolve_config,
)
from rxn_platform.logging_utils import (
    configure_logging,
    log_exception,
    run_with_error_handling,
)
from rxn_platform.doctor import _compose_config_fallback, run_doctor
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks import sim as sim_task
from rxn_platform.tasks.runner import DEFAULT_STORE_ROOT
from rxn_platform.tasks.runner import run_task_from_config

_SUBCOMMANDS: Sequence[str] = (
    "cfg",
    "sim",
    "task",
    "pipeline",
    "viz",
    "doctor",
    "artifacts",
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
        default="defaults",
        help="Hydra config name (without extension).",
    )
    cfg_parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (ex: sim=dummy common.seed=123).",
    )
    cfg_parser.set_defaults(handler=_cfg_handler)


def _sim_help_handler(args: argparse.Namespace) -> None:
    parser = getattr(args, "parser", None)
    if parser is not None:
        parser.print_help()
    raise SystemExit(2)


def _utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _resolve_config_dir(path: str) -> Path:
    config_dir = Path(path)
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()
    return config_dir


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
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    sim_task.validate_config(cfg)
    print("sim config ok")


def _sim_run_handler(args: argparse.Namespace) -> None:
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
        cache_bust = _utc_now_iso()

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
        help="Simulation commands.",
        description="Simulation commands.",
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
        help="Task runner commands.",
        description="Task runner commands.",
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
        help="Pipeline runner commands.",
        description="Pipeline runner commands.",
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
        if name == "cfg":
            _register_cfg_subcommand(subparsers)
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
        if name == "doctor":
            _register_doctor_subcommand(subparsers)
            continue
        if name == "artifacts":
            _register_artifacts_subcommand(subparsers)
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
