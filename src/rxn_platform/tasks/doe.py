"""Design of experiments task: rank candidate conditions via FIM metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import platform
from pathlib import Path
import subprocess
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import (
    ArtifactManifest,
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
)
from rxn_platform.errors import ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import Task
from rxn_platform.tasks.runner import run_task

try:  # Optional dependency.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:  # Optional dependency.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pq = None

DEFAULT_METRIC = "d_opt"
DEFAULT_REGULARIZATION = 1.0e-9
REQUIRED_COLUMNS = (
    "condition_id",
    "score",
    "rank",
    "metric",
    "n_params",
    "n_targets",
    "sensitivity_id",
    "meta_json",
)


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: Optional[str]
    sim_cfg: Optional[dict[str, Any]]


def _utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _resolve_cfg(cfg: Any) -> dict[str, Any]:
    try:
        resolved = resolve_config(cfg)
    except ConfigError:
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise
    return resolved


def _extract_doe_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ("doe", "design", "mbdoe"):
        if key in cfg:
            doe_cfg = cfg.get(key)
            if not isinstance(doe_cfg, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(cfg), dict(doe_cfg)
    return dict(cfg), dict(cfg)


def _extract_params(doe_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = doe_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("doe.params must be a mapping.")
    return dict(params)


def _extract_inputs(doe_cfg: Mapping[str, Any]) -> dict[str, Any]:
    inputs = doe_cfg.get("inputs")
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("doe.inputs must be a mapping.")
    return dict(inputs)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_str_sequence(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_require_nonempty_str(value, label)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for entry in value:
            items.append(_require_nonempty_str(entry, label))
        return items
    raise ConfigError(f"{label} must be a string or sequence of strings.")


def _extract_metric(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    metric = None
    for source in (params, doe_cfg):
        if "metric" in source:
            metric = source.get("metric")
            break
        if "criterion" in source:
            metric = source.get("criterion")
            break
    if metric is None:
        return DEFAULT_METRIC
    if not isinstance(metric, str):
        raise ConfigError("metric must be a string.")
    key = metric.strip().lower()
    alias = {
        "d_opt": "d_opt",
        "d-opt": "d_opt",
        "dopt": "d_opt",
        "logdet": "d_opt",
        "det": "d_opt",
        "trace": "trace",
        "tr": "trace",
    }
    if key not in alias:
        raise ConfigError("metric must be d_opt/logdet or trace.")
    return alias[key]


def _extract_regularization(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    value = None
    for source in (params, doe_cfg):
        for key in ("regularization", "ridge", "jitter", "lambda"):
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return DEFAULT_REGULARIZATION
    if isinstance(value, bool):
        raise ConfigError("regularization must be a float.")
    try:
        regularization = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("regularization must be a float.") from exc
    if regularization < 0.0:
        raise ConfigError("regularization must be non-negative.")
    return regularization


def _extract_targets(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[str]:
    targets = None
    for source in (params, doe_cfg):
        if "targets" in source:
            targets = source.get("targets")
            break
        if "target" in source:
            targets = source.get("target")
            break
    return _coerce_str_sequence(targets, "targets")


def _extract_weights(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, float]:
    weights = None
    for source in (params, doe_cfg):
        for key in ("weights", "target_weights", "weight"):
            if key in source:
                weights = source.get(key)
                break
        if weights is not None:
            break
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ConfigError("weights must be a mapping.")
    result: dict[str, float] = {}
    for target, raw_value in weights.items():
        target_name = _require_nonempty_str(target, "weights target")
        if isinstance(raw_value, bool):
            raise ConfigError("weights must be numeric.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("weights must be numeric.") from exc
        if value < 0.0:
            raise ConfigError("weights must be non-negative.")
        result[target_name] = value
    return result


def _normalize_condition_entry(entry: Any, label: str) -> ConditionSpec:
    if isinstance(entry, str):
        return ConditionSpec(condition_id=_require_nonempty_str(entry, label), sim_cfg=None)
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{label} must be a string or mapping.")
    condition_id = entry.get("id") or entry.get("condition_id") or entry.get("run_id")
    if condition_id is not None:
        condition_id = _require_nonempty_str(condition_id, f"{label}.id")
    sim_cfg: Optional[dict[str, Any]] = None
    if "sim" in entry:
        sim_value = entry.get("sim")
        if not isinstance(sim_value, Mapping):
            raise ConfigError(f"{label}.sim must be a mapping.")
        sim_cfg = dict(sim_value)
    elif condition_id is None:
        sim_cfg = dict(entry)
    return ConditionSpec(condition_id=condition_id, sim_cfg=sim_cfg)


def _extract_conditions(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[ConditionSpec]:
    value = None
    for source in (params, doe_cfg):
        for key in ("conditions", "condition_ids", "candidates"):
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("ids", "condition_ids", "conditions"):
            if key in value:
                value = value.get(key)
                break
    if isinstance(value, str):
        return [_normalize_condition_entry(value, "conditions")]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("conditions must be a sequence of strings or mappings.")
    specs: list[ConditionSpec] = []
    for index, entry in enumerate(value):
        specs.append(_normalize_condition_entry(entry, f"conditions[{index}]"))
    return specs


def _extract_sensitivity_id(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Optional[str]:
    for source in (inputs, params, doe_cfg):
        for key in ("sensitivity", "sensitivity_id", "sensitivity_artifact"):
            if key in source:
                value = source.get(key)
                if value is None:
                    return None
                return _require_nonempty_str(value, f"sensitivity {key}")
    return None


def _extract_sensitivity_cfg(
    doe_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    for source in (params, doe_cfg):
        if "sensitivity" in source:
            value = source.get("sensitivity")
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise ConfigError("sensitivity must be a mapping.")
            return dict(value)
    return None


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _reaction_key(row: Mapping[str, Any]) -> Optional[str]:
    reaction_id = row.get("reaction_id")
    if isinstance(reaction_id, str) and reaction_id.strip():
        return reaction_id.strip()
    reaction_index = row.get("reaction_index")
    if isinstance(reaction_index, int) and not isinstance(reaction_index, bool):
        return f"index:{reaction_index}"
    return None


def _group_sensitivities(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        condition_id = row.get("condition_id") or row.get("run_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            continue
        target = row.get("target") or row.get("observable")
        if not isinstance(target, str) or not target.strip():
            continue
        param_id = _reaction_key(row)
        if param_id is None:
            continue
        value = _coerce_numeric(row.get("value"))
        if value is None:
            continue
        grouped.setdefault(condition_id, {}).setdefault(target, {})[param_id] = value
    return grouped


def _normalize_sim_cfg(sim_cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(sim_cfg)
    try:
        multipliers = normalize_reaction_multipliers(sim_cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"reaction multipliers are invalid: {exc}") from exc
    if multipliers:
        normalized["reaction_multipliers"] = list(multipliers)
    else:
        normalized.pop("reaction_multipliers", None)
    normalized.pop("disabled_reactions", None)
    return normalized


def _sim_run_id(sim_cfg: Mapping[str, Any]) -> str:
    manifest_cfg = {"sim": sim_cfg, "inputs": {}, "params": {}}
    return make_run_id(manifest_cfg, exclude_keys=("hydra",))


def _resolve_condition_ids(specs: Sequence[ConditionSpec]) -> list[str]:
    condition_ids: list[str] = []
    for spec in specs:
        if spec.condition_id:
            condition_ids.append(spec.condition_id)
            continue
        if spec.sim_cfg is None:
            raise ConfigError("condition must include id or sim config.")
        sim_cfg = _normalize_sim_cfg(spec.sim_cfg)
        condition_ids.append(_sim_run_id(sim_cfg))
    return condition_ids


def _build_fim(
    targets: Mapping[str, Mapping[str, float]],
    param_ids: Sequence[str],
    weights: Mapping[str, float],
) -> tuple[list[list[float]], int]:
    size = len(param_ids)
    fim = [[0.0 for _ in range(size)] for _ in range(size)]
    used_targets = 0
    for target, values in targets.items():
        weight = weights.get(target, 1.0)
        if weight <= 0.0:
            continue
        vector = [values.get(param_id, 0.0) for param_id in param_ids]
        if not any(vector):
            continue
        used_targets += 1
        for i in range(size):
            vi = vector[i]
            if vi == 0.0:
                continue
            for j in range(i, size):
                fim[i][j] += weight * vi * vector[j]
    for i in range(size):
        for j in range(i + 1, size):
            fim[j][i] = fim[i][j]
    return fim, used_targets


def _trace(matrix: Sequence[Sequence[float]]) -> float:
    return sum(row[i] for i, row in enumerate(matrix))


def _logdet(matrix: Sequence[Sequence[float]]) -> tuple[float, str]:
    size = len(matrix)
    if size == 0:
        return math.nan, "no_params"
    work = [list(row) for row in matrix]
    logdet = 0.0
    sign = 1.0
    for i in range(size):
        pivot = max(range(i, size), key=lambda r: abs(work[r][i]))
        pivot_val = work[pivot][i]
        if abs(pivot_val) <= 0.0:
            return math.nan, "singular"
        if pivot != i:
            work[i], work[pivot] = work[pivot], work[i]
            sign *= -1.0
            pivot_val = work[i][i]
        if pivot_val < 0.0:
            sign *= -1.0
        logdet += math.log(abs(pivot_val))
        for r in range(i + 1, size):
            factor = work[r][i] / pivot_val
            if factor == 0.0:
                continue
            for c in range(i + 1, size):
                work[r][c] -= factor * work[i][c]
            work[r][i] = 0.0
    if sign <= 0.0:
        return math.nan, "non_positive"
    return logdet, "ok"


def _safe_score(value: Any) -> float:
    if value is None:
        return -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    if not math.isfinite(number):
        return -math.inf
    return number


def _build_row(
    condition_id: str,
    metric: str,
    score: Any,
    logdet: Any,
    trace_value: Any,
    n_params: int,
    n_targets: int,
    sensitivity_id: str,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    meta_json = json.dumps(
        dict(meta),
        ensure_ascii=True,
        sort_keys=True,
    )
    return {
        "condition_id": condition_id,
        "score": score,
        "rank": None,
        "metric": metric,
        "n_params": n_params,
        "n_targets": n_targets,
        "sensitivity_id": sensitivity_id,
        "meta_json": meta_json,
        "logdet": logdet,
        "trace": trace_value,
    }


def _collect_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(REQUIRED_COLUMNS)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_design_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = _collect_columns(rows)
    if pd is not None:
        frame = pd.DataFrame(list(rows), columns=columns)
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            pass
    if pa is not None and pq is not None:
        table = pa.Table.from_pylist(list(rows))
        table = table.select(columns)
        pq.write_table(table, path)
        return
    payload = {"columns": columns, "rows": list(rows)}
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger = logging.getLogger("rxn_platform.doe")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
    )


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


def _resolve_sensitivity_artifact(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry],
) -> str:
    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, doe_cfg = _extract_doe_cfg(resolved_cfg)
    params = _extract_params(doe_cfg)
    inputs = _extract_inputs(doe_cfg)

    sensitivity_id = _extract_sensitivity_id(doe_cfg, params, inputs)
    sensitivity_cfg = _extract_sensitivity_cfg(doe_cfg, params)

    if sensitivity_id and sensitivity_cfg:
        raise ConfigError("Specify only one of sensitivity id or sensitivity config.")
    if sensitivity_id:
        store.read_manifest("sensitivity", sensitivity_id)
        return sensitivity_id
    if sensitivity_cfg is None:
        raise ConfigError("sensitivity id or sensitivity config must be provided.")

    runner_cfg = {"sensitivity": sensitivity_cfg}
    result = run_task(
        "sensitivity.multiplier_fd",
        runner_cfg,
        store=store,
        registry=registry,
    )
    return result.manifest.id


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Rank candidate conditions via approximate FIM metrics."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, doe_cfg = _extract_doe_cfg(resolved_cfg)
    params = _extract_params(doe_cfg)
    inputs = _extract_inputs(doe_cfg)

    metric = _extract_metric(doe_cfg, params)
    regularization = _extract_regularization(doe_cfg, params)
    targets = _extract_targets(doe_cfg, params)
    weights = _extract_weights(doe_cfg, params)
    condition_specs = _extract_conditions(doe_cfg, params)

    sensitivity_id = _extract_sensitivity_id(doe_cfg, params, inputs)
    sensitivity_cfg = _extract_sensitivity_cfg(doe_cfg, params)
    if sensitivity_id and sensitivity_cfg:
        raise ConfigError("Specify only one of sensitivity id or sensitivity config.")
    if sensitivity_id is None:
        sensitivity_id = _resolve_sensitivity_artifact(
            resolved_cfg,
            store=store,
            registry=registry,
        )

    sensitivity_dir = store.artifact_dir("sensitivity", sensitivity_id)
    table_path = sensitivity_dir / "sensitivity.parquet"
    if not table_path.exists():
        raise ConfigError(f"Sensitivity table not found: {table_path}")
    rows = _read_table_rows(table_path)
    if not rows:
        raise ConfigError("Sensitivity table contains no rows.")

    grouped = _group_sensitivities(rows)
    if not grouped:
        raise ConfigError("No valid sensitivities found for DOE ranking.")

    if not targets:
        if weights:
            targets = sorted(weights.keys())
        else:
            target_set = {
                target for condition_data in grouped.values() for target in condition_data.keys()
            }
            targets = sorted(target_set)
    if not targets:
        raise ConfigError("No targets resolved for DOE ranking.")

    condition_ids = _resolve_condition_ids(condition_specs) if condition_specs else []
    if condition_ids:
        grouped = {cid: grouped[cid] for cid in condition_ids if cid in grouped}
        if not grouped:
            raise ConfigError("No sensitivities matched requested conditions.")
    else:
        condition_ids = sorted(grouped.keys())

    rows_out: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        target_values = grouped.get(condition_id, {})
        filtered_targets: dict[str, dict[str, float]] = {}
        missing_targets: list[str] = []
        for target in targets:
            values = target_values.get(target)
            if values is None:
                missing_targets.append(target)
                continue
            filtered_targets[target] = dict(values)
        param_ids = sorted(
            {param for values in filtered_targets.values() for param in values.keys()}
        )
        n_params = len(param_ids)
        fim, used_targets = _build_fim(filtered_targets, param_ids, weights)
        if regularization > 0.0:
            for index in range(n_params):
                fim[index][index] += regularization
        trace_value = _trace(fim) if n_params else math.nan
        logdet, status = _logdet(fim) if n_params else (math.nan, "no_params")
        score = logdet if metric == "d_opt" else trace_value
        meta = {
            "status": status,
            "metric": metric,
            "regularization": regularization,
            "missing_target_count": len(missing_targets),
        }
        rows_out.append(
            _build_row(
                condition_id,
                metric,
                score,
                logdet,
                trace_value,
                n_params,
                used_targets,
                sensitivity_id,
                meta,
            )
        )

    rows_out.sort(
        key=lambda row: (-_safe_score(row.get("score")), row.get("condition_id"))
    )
    for rank, row in enumerate(rows_out, start=1):
        row["rank"] = rank

    inputs_payload = {
        "sensitivity_id": sensitivity_id,
        "metric": metric,
        "regularization": regularization,
        "targets": list(targets),
        "condition_ids": list(condition_ids),
    }
    if weights:
        inputs_payload["weights"] = dict(weights)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    manifest = ArtifactManifest(
        schema_version=1,
        kind="designs",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[sensitivity_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_design_table(rows_out, base_dir / "design.parquet")

    return store.ensure(manifest, writer=_writer)


class FimRankingTask(Task):
    name = "doe.fim_rank"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run(cfg, store=store, registry=registry)


register("task", "doe.fim_rank", FimRankingTask())

__all__ = ["FimRankingTask", "run"]
