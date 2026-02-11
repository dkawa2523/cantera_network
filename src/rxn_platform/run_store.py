"""RunStore utilities for unified run output management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional, Sequence
import uuid

from rxn_platform.core import resolve_repo_path, stable_hash
from rxn_platform.io_utils import read_json, read_yaml_payload, write_json_atomic, write_yaml_payload

RUNS_ROOT = Path("runs")
RUN_MANIFEST_NAME = "manifest.json"
RUN_CONFIG_NAME = "config_resolved.yaml"
RUN_METRICS_NAME = "metrics.json"
RUN_SIM_DIRNAME = "sim"
RUN_TIMESERIES_DIRNAME = "timeseries.zarr"
RUN_GRAPH_DIRNAME = "graphs"
RUN_GRAPH_META_NAME = "meta.json"
RUN_GRAPH_SPECIES_DIRNAME = "species_graph"
RUN_VIZ_DIRNAME = "viz"

LEGACY_RUN_MANIFEST_NAME = "manifest.yaml"
LEGACY_RUN_CONFIG_NAME = "config.yaml"
LEGACY_RUN_RESULTS_NAME = "results.json"
LEGACY_RUN_STATE_DIRNAME = "state.zarr"


@dataclass(frozen=True)
class RunInfo:
    exp: str
    run_id: str
    root: Path


@dataclass(frozen=True)
class RunSummary:
    exp: str
    run_id: str
    root: Path
    created_at: str
    created_at_ts: float


def utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def normalize_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"{label} must be a single path component: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"{label} must not be '.' or '..'.")
    return value


def normalize_run_id(run_id: Optional[str]) -> str:
    if not run_id or run_id == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:8]
        return f"run_{timestamp}_{suffix}"
    return normalize_component(run_id, "run_id")


def resolve_run_info(cfg: Mapping[str, Any]) -> RunInfo:
    run_cfg = cfg.get("run") if isinstance(cfg.get("run"), Mapping) else {}
    exp = None
    run_id = None
    if isinstance(run_cfg, Mapping):
        exp = run_cfg.get("exp")
        run_id = run_cfg.get("run_id")
    exp = exp if isinstance(exp, str) and exp.strip() else cfg.get("exp")
    if not isinstance(exp, str) or not exp.strip():
        exp = "default"
    exp = normalize_component(exp, "exp")
    raw_run_id = run_id if isinstance(run_id, str) and run_id.strip() else cfg.get("run_id")
    raw_run_id = raw_run_id if isinstance(raw_run_id, str) else None
    normalized_run_id = normalize_run_id(raw_run_id)

    root_value = run_cfg.get("root") if isinstance(run_cfg, Mapping) else None
    if (
        isinstance(root_value, str)
        and root_value.strip()
        and raw_run_id
        and raw_run_id != "auto"
    ):
        root = Path(root_value)
    else:
        root = RUNS_ROOT / exp / normalized_run_id
    return RunInfo(exp=exp, run_id=normalized_run_id, root=root)


def write_run_manifest(run_root: Path, payload: Mapping[str, Any]) -> Path:
    path = run_root / RUN_MANIFEST_NAME
    write_json_atomic(path, dict(payload))
    return path


def write_run_config(run_root: Path, cfg: Mapping[str, Any]) -> Path:
    path = run_root / RUN_CONFIG_NAME
    write_yaml_payload(path, dict(cfg), sort_keys=True)
    return path


def write_run_metrics(run_root: Path, metrics: Any) -> Path:
    path = run_root / RUN_METRICS_NAME
    write_json_atomic(path, metrics)
    return path


def write_run_results(run_root: Path, results: Any) -> Path:
    return write_run_metrics(run_root, results)


def read_run_manifest(run_root: Path) -> Optional[Mapping[str, Any]]:
    path = run_root / RUN_MANIFEST_NAME
    if not path.exists():
        legacy = run_root / LEGACY_RUN_MANIFEST_NAME
        if not legacy.exists():
            return None
        payload = read_yaml_payload(legacy)
    else:
        payload = read_json(path)
    if isinstance(payload, Mapping):
        return payload
    return None


def read_run_config(run_root: Path) -> Optional[Mapping[str, Any]]:
    path = run_root / RUN_CONFIG_NAME
    if not path.exists():
        legacy = run_root / LEGACY_RUN_CONFIG_NAME
        if not legacy.exists():
            return None
        payload = read_yaml_payload(legacy)
    else:
        payload = read_yaml_payload(path)
    if isinstance(payload, Mapping):
        return payload
    return None


def read_run_metrics(run_root: Path) -> Optional[Any]:
    path = run_root / RUN_METRICS_NAME
    if not path.exists():
        legacy = run_root / LEGACY_RUN_RESULTS_NAME
        if not legacy.exists():
            return None
        return read_json(legacy)
    return read_json(path)


def resolve_run_root_from_store(store_root: Path) -> Optional[Path]:
    candidate = store_root.parent
    if (candidate / RUN_MANIFEST_NAME).exists():
        return candidate
    if (candidate / LEGACY_RUN_MANIFEST_NAME).exists():
        return candidate
    return None


def resolve_run_dataset_dir(run_root: Path) -> Optional[Path]:
    preferred = run_root / RUN_SIM_DIRNAME / RUN_TIMESERIES_DIRNAME
    if preferred.exists():
        return preferred
    legacy = run_root / LEGACY_RUN_STATE_DIRNAME
    if legacy.exists():
        return legacy
    return None


def _reset_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def sync_timeseries_from_artifact(
    artifact_dir: Path,
    run_root: Path,
) -> Optional[Path]:
    source = artifact_dir / LEGACY_RUN_STATE_DIRNAME
    if not source.exists():
        return None
    target = run_root / RUN_SIM_DIRNAME / RUN_TIMESERIES_DIRNAME
    target.parent.mkdir(parents=True, exist_ok=True)
    _reset_path(target)
    shutil.copytree(source, target)
    return target


def sync_temporal_graph_from_artifact(
    graph_dir: Path,
    run_root: Path,
) -> Optional[Path]:
    source_meta = graph_dir / "graph.json"
    source_layers = graph_dir / RUN_GRAPH_SPECIES_DIRNAME
    if not source_meta.exists() and not source_layers.exists():
        return None
    target_root = run_root / RUN_GRAPH_DIRNAME
    target_root.mkdir(parents=True, exist_ok=True)
    if source_meta.exists():
        target_meta = target_root / RUN_GRAPH_META_NAME
        _reset_path(target_meta)
        shutil.copy2(source_meta, target_meta)
    if source_layers.exists():
        target_layers = target_root / RUN_GRAPH_SPECIES_DIRNAME
        _reset_path(target_layers)
        shutil.copytree(source_layers, target_layers)
    return target_root


def sync_report_from_artifact(
    report_dir: Path,
    run_root: Path,
) -> Optional[Path]:
    if not report_dir.exists():
        return None
    target = run_root / RUN_VIZ_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    for entry in report_dir.iterdir():
        dest = target / entry.name
        if entry.is_dir():
            _reset_path(dest)
            shutil.copytree(entry, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dest)
    return target


def sync_conditions_table(
    cfg: Mapping[str, Any],
    run_root: Path,
) -> Optional[Path]:
    conditions_value = _find_value(
        cfg,
        ["conditions", "conditions_file", "conditions_path", "conditions_csv", "csv"],
    )
    if conditions_value is None:
        return None

    target_dir = run_root / "inputs"
    target_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(conditions_value, (str, Path)):
        value_str = str(conditions_value).strip()
        if not value_str:
            return None
        source = resolve_repo_path(value_str)
        if not source.exists() or not source.is_file():
            return None
        suffix = source.suffix.lower() or ".csv"
        if suffix not in {".csv", ".parquet"}:
            suffix = ".csv"
        target = target_dir / f"conditions{suffix}"
        _reset_path(target)
        shutil.copy2(source, target)
        return target

    if isinstance(conditions_value, Mapping):
        rows = [dict(conditions_value)]
    elif isinstance(conditions_value, Sequence) and not isinstance(
        conditions_value, (str, bytes, bytearray)
    ):
        rows = [dict(row) for row in conditions_value if isinstance(row, Mapping)]
        if not rows:
            return None
    else:
        return None

    fieldnames: list[str] = []
    fieldset: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in fieldset:
                fieldset.add(key)
                fieldnames.append(str(key))

    target = target_dir / "conditions.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return target


def _parse_iso_timestamp(value: str) -> Optional[float]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def list_runs(root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    if not root.exists():
        return summaries
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        exp_name = exp_dir.name
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            run_id = run_dir.name
            created_at = ""
            created_at_ts: Optional[float] = None
            manifest = read_run_manifest(run_dir)
            if isinstance(manifest, Mapping):
                created_at_value = manifest.get("created_at")
                if isinstance(created_at_value, str) and created_at_value.strip():
                    created_at = created_at_value
                    created_at_ts = _parse_iso_timestamp(created_at_value)
            if created_at_ts is None:
                try:
                    created_at_ts = run_dir.stat().st_mtime
                    created_at = datetime.fromtimestamp(
                        created_at_ts, tz=timezone.utc
                    ).isoformat().replace("+00:00", "Z")
                except OSError:
                    created_at_ts = 0.0
            summaries.append(
                RunSummary(
                    exp=exp_name,
                    run_id=run_id,
                    root=run_dir,
                    created_at=created_at,
                    created_at_ts=created_at_ts,
                )
            )
    summaries.sort(key=lambda item: item.created_at_ts, reverse=True)
    return summaries


def _hash_payload(value: Any) -> Optional[str]:
    if value is None:
        return stable_hash(None, length=16)
    try:
        return stable_hash(value, length=16)
    except Exception:
        try:
            return stable_hash(str(value), length=16)
        except Exception:
            return None


def _hash_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return digest[:16]


def _hash_path_or_payload(value: Any) -> Optional[str]:
    if value is None:
        return _hash_payload(None)
    if isinstance(value, (str, Path)):
        resolved = resolve_repo_path(str(value))
        if resolved.exists() and resolved.is_file():
            return _hash_file(resolved)
        return _hash_payload(str(value))
    return _hash_payload(value)


def _iter_candidate_maps(cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for key in ("benchmarks", "params", "inputs", "task", "pipeline", "sim"):
        value = cfg.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    sources.append(cfg)
    return sources


def _find_value(cfg: Mapping[str, Any], keys: list[str]) -> Any:
    for source in _iter_candidate_maps(cfg):
        for key in keys:
            if key in source:
                return source.get(key)
    return None


def derive_run_contract_metadata(cfg: Mapping[str, Any]) -> dict[str, Optional[str]]:
    sim_cfg = cfg.get("sim")
    simulator: Optional[str] = None
    if isinstance(sim_cfg, Mapping):
        name = sim_cfg.get("name") or sim_cfg.get("backend")
        if isinstance(name, str) and name.strip():
            simulator = name
    if simulator is None:
        value = cfg.get("simulator") or cfg.get("backend")
        if isinstance(value, str) and value.strip():
            simulator = value
    if simulator is None:
        simulator = "unknown"

    mechanism_value: Any = None
    if isinstance(sim_cfg, Mapping):
        for key in ("mechanism", "solution", "mechanism_path", "mechanism_file"):
            if key in sim_cfg:
                mechanism_value = sim_cfg.get(key)
                break
    if isinstance(mechanism_value, str) and not mechanism_value.strip():
        mechanism_value = None
    mechanism_hash = (
        _hash_path_or_payload(mechanism_value) if mechanism_value is not None else None
    )
    if mechanism_hash is None:
        mechanism_hash = stable_hash({"mechanism": None, "simulator": simulator}, length=16)

    conditions_value = _find_value(
        cfg,
        ["conditions", "conditions_file", "conditions_path", "conditions_csv", "csv"],
    )
    if isinstance(conditions_value, str) and not conditions_value.strip():
        conditions_value = None
    conditions_hash = (
        _hash_path_or_payload(conditions_value) if conditions_value is not None else None
    )

    qoi_value = _find_value(cfg, ["qoi", "qoi_spec", "observables", "observable"])
    if isinstance(qoi_value, str) and not qoi_value.strip():
        qoi_value = None
    qoi_spec_hash = _hash_payload(qoi_value) if qoi_value is not None else None

    return {
        "simulator": simulator,
        "mechanism_hash": mechanism_hash,
        "conditions_hash": conditions_hash,
        "qoi_spec_hash": qoi_spec_hash,
    }


__all__ = [
    "RUNS_ROOT",
    "RUN_MANIFEST_NAME",
    "RUN_CONFIG_NAME",
    "RUN_METRICS_NAME",
    "RUN_SIM_DIRNAME",
    "RUN_TIMESERIES_DIRNAME",
    "RUN_GRAPH_DIRNAME",
    "RUN_GRAPH_META_NAME",
    "RUN_GRAPH_SPECIES_DIRNAME",
    "RUN_VIZ_DIRNAME",
    "LEGACY_RUN_MANIFEST_NAME",
    "LEGACY_RUN_CONFIG_NAME",
    "LEGACY_RUN_RESULTS_NAME",
    "LEGACY_RUN_STATE_DIRNAME",
    "RunInfo",
    "RunSummary",
    "utc_now_iso",
    "normalize_component",
    "normalize_run_id",
    "resolve_run_info",
    "write_run_manifest",
    "write_run_config",
    "write_run_metrics",
    "write_run_results",
    "read_run_manifest",
    "read_run_config",
    "read_run_metrics",
    "resolve_run_root_from_store",
    "resolve_run_dataset_dir",
    "sync_timeseries_from_artifact",
    "sync_temporal_graph_from_artifact",
    "sync_report_from_artifact",
    "sync_conditions_table",
    "list_runs",
    "derive_run_contract_metadata",
]
