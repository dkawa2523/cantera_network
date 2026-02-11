"""Dimension reduction tasks: active subspace estimation from sensitivities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional
from rxn_platform.core import make_artifact_id
from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import write_json_atomic
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import Task
from rxn_platform.tasks.runner import run_task
from rxn_platform.tasks.common import (
    build_manifest,
    code_metadata as _code_metadata,
    read_table_rows as _read_table_rows,
    resolve_cfg as _resolve_cfg,
)

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

try:  # Optional dependency.
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

DEFAULT_K = 2
DEFAULT_MISSING_STRATEGY = "zero"
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1.0e-9
REQUIRED_COLUMNS = (
    "component",
    "rank",
    "param_id",
    "loading",
    "eigenvalue",
    "variance_ratio",
    "meta_json",
)


def _extract_dimred_cfg(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ("dimred", "dimension_reduction", "active_subspace", "subspace"):
        if key in cfg:
            dimred_cfg = cfg.get(key)
            if not isinstance(dimred_cfg, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(cfg), dict(dimred_cfg)
    return dict(cfg), dict(cfg)


def _extract_params(dimred_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = dimred_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("dimred.params must be a mapping.")
    return dict(params)


def _extract_inputs(dimred_cfg: Mapping[str, Any]) -> dict[str, Any]:
    inputs = dimred_cfg.get("inputs")
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("dimred.inputs must be a mapping.")
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
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items: list[str] = []
        for entry in value:
            items.append(_require_nonempty_str(entry, label))
        return items
    raise ConfigError(f"{label} must be a string or sequence of strings.")


def _extract_targets(
    dimred_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[str]:
    targets = None
    for source in (params, dimred_cfg):
        if "targets" in source:
            targets = source.get("targets")
            break
        if "target" in source:
            targets = source.get("target")
            break
    return _coerce_str_sequence(targets, "targets")


def _extract_k(
    dimred_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[int]:
    value = None
    for source in (params, dimred_cfg):
        for key in ("k", "rank", "dim", "dims", "n_dims", "components"):
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError("k must be a positive integer.")
    try:
        k_val = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("k must be a positive integer.") from exc
    if k_val <= 0:
        raise ConfigError("k must be a positive integer.")
    return k_val


def _extract_missing_strategy(
    dimred_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    value = None
    for source in (params, dimred_cfg):
        if "missing_strategy" in source:
            value = source.get("missing_strategy")
            break
    if value is None:
        return DEFAULT_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in {"zero", "skip"}:
        raise ConfigError("missing_strategy must be 'zero' or 'skip'.")
    return strategy


def _extract_sensitivity_id(
    dimred_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Optional[str]:
    for source in (inputs, params, dimred_cfg):
        for key in ("sensitivity", "sensitivity_id", "sensitivity_artifact"):
            if key in source:
                value = source.get(key)
                if isinstance(value, Mapping):
                    continue
                return _require_nonempty_str(value, f"sensitivity {key}")
    return None


def _extract_sensitivity_cfg(
    dimred_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    for source in (params, dimred_cfg):
        if "sensitivity" in source:
            value = source.get("sensitivity")
            if value is None:
                return None
            if not isinstance(value, Mapping):
                return None
            return dict(value)
    return None


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


def _param_sort_key(param_id: str) -> tuple[int, Any]:
    if param_id.startswith("index:"):
        try:
            return (0, int(param_id.split(":", 1)[1]))
        except (TypeError, ValueError):
            return (0, param_id)
    return (1, param_id)


def _group_gradients(
    rows: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
) -> tuple[dict[tuple[str, str], dict[str, float]], list[str]]:
    target_set = set(targets)
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    param_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        target = row.get("target") or row.get("observable")
        if not isinstance(target, str) or not target.strip():
            continue
        target = target.strip()
        if target_set and target not in target_set:
            continue
        condition_id = row.get("condition_id") or row.get("run_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            continue
        condition_id = condition_id.strip()
        param_id = _reaction_key(row)
        if param_id is None:
            continue
        value = _coerce_numeric(row.get("value"))
        if value is None:
            continue
        key = (condition_id, target)
        grouped.setdefault(key, {})[param_id] = value
        param_ids.add(param_id)
    return grouped, sorted(param_ids, key=_param_sort_key)


def _build_gradient_vectors(
    grouped: Mapping[tuple[str, str], Mapping[str, float]],
    param_ids: Sequence[str],
    *,
    missing_strategy: str,
) -> tuple[list[list[float]], int]:
    vectors: list[list[float]] = []
    for values in grouped.values():
        vector: list[float] = []
        missing = False
        for param_id in param_ids:
            if param_id in values:
                vector.append(values[param_id])
            else:
                vector.append(0.0)
                missing = True
        if missing and missing_strategy == "skip":
            continue
        vectors.append(vector)
    return vectors, len(vectors)


def _covariance_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    if not vectors:
        return []
    size = len(vectors[0])
    cov = [[0.0 for _ in range(size)] for _ in range(size)]
    for vector in vectors:
        for i in range(size):
            vi = vector[i]
            if vi == 0.0:
                continue
            for j in range(i, size):
                cov[i][j] += vi * vector[j]
    count = len(vectors)
    if count:
        scale = 1.0 / count
        for i in range(size):
            for j in range(i, size):
                cov[i][j] *= scale
    for i in range(size):
        for j in range(i + 1, size):
            cov[j][i] = cov[i][j]
    return cov


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(li * ri for li, ri in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _orient_vector(vector: list[float]) -> list[float]:
    if not vector:
        return vector
    max_index = max(range(len(vector)), key=lambda idx: abs(vector[idx]))
    if vector[max_index] < 0.0:
        return [-value for value in vector]
    return vector


def _power_iteration(
    matrix: Sequence[Sequence[float]],
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> tuple[float, list[float]]:
    size = len(matrix)
    if size == 0:
        return 0.0, []
    vector = [1.0 for _ in range(size)]
    norm = _norm(vector)
    if norm == 0.0:
        return 0.0, [0.0 for _ in range(size)]
    vector = [value / norm for value in vector]
    for _ in range(max_iter):
        product = _mat_vec(matrix, vector)
        norm = _norm(product)
        if norm == 0.0:
            return 0.0, [0.0 for _ in range(size)]
        next_vec = [value / norm for value in product]
        diff = max(abs(next_vec[i] - vector[i]) for i in range(size))
        vector = next_vec
        if diff < tol:
            break
    eigenvalue = _dot(vector, _mat_vec(matrix, vector))
    return eigenvalue, _orient_vector(vector)


def _eigh_fallback(
    matrix: Sequence[Sequence[float]],
    k: int,
) -> tuple[list[float], list[list[float]]]:
    size = len(matrix)
    work = [list(row) for row in matrix]
    values: list[float] = []
    vectors: list[list[float]] = []
    for _ in range(min(k, size)):
        eigenvalue, vector = _power_iteration(work)
        values.append(float(eigenvalue))
        vectors.append(vector)
        if size == 0:
            break
        for i in range(size):
            for j in range(size):
                work[i][j] -= eigenvalue * vector[i] * vector[j]
    return values, vectors


def _eigh_symmetric(
    matrix: Sequence[Sequence[float]],
    k: int,
) -> tuple[list[float], list[list[float]]]:
    size = len(matrix)
    if size == 0:
        return [], []
    if np is not None:
        array = np.array(matrix, dtype=float)
        values, vectors = np.linalg.eigh(array)
        order = list(reversed(range(len(values))))
        values_out: list[float] = []
        vectors_out: list[list[float]] = []
        for index in order[:k]:
            values_out.append(float(values[index]))
            vector = [float(vectors[row][index]) for row in range(size)]
            vectors_out.append(_orient_vector(vector))
        return values_out, vectors_out
    return _eigh_fallback(matrix, k)


def _collect_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(REQUIRED_COLUMNS)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_subspace_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
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
    write_json_atomic(path, payload)
    logger = logging.getLogger("rxn_platform.dimred")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
    )


def _resolve_sensitivity_artifact(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry],
) -> str:
    resolved_cfg = _resolve_cfg(cfg)
    _, dimred_cfg = _extract_dimred_cfg(resolved_cfg)
    params = _extract_params(dimred_cfg)
    inputs = _extract_inputs(dimred_cfg)

    sensitivity_id = _extract_sensitivity_id(dimred_cfg, params, inputs)
    sensitivity_cfg = _extract_sensitivity_cfg(dimred_cfg, params)

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
    """Estimate an active subspace from sensitivity gradients."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, dimred_cfg = _extract_dimred_cfg(resolved_cfg)
    params = _extract_params(dimred_cfg)
    inputs = _extract_inputs(dimred_cfg)

    targets = _extract_targets(dimred_cfg, params)
    missing_strategy = _extract_missing_strategy(dimred_cfg, params)
    k = _extract_k(dimred_cfg, params)

    sensitivity_id = _extract_sensitivity_id(dimred_cfg, params, inputs)
    sensitivity_cfg = _extract_sensitivity_cfg(dimred_cfg, params)
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

    if not targets:
        target_set = {row.get("target") for row in rows if row.get("target")}
        targets = sorted(target_set)

    grouped, param_ids = _group_gradients(rows, targets)
    if not param_ids:
        raise ConfigError("No parameters found in sensitivity table.")
    if not grouped:
        raise ConfigError("No gradients available for active subspace.")

    vectors, sample_count = _build_gradient_vectors(
        grouped,
        param_ids,
        missing_strategy=missing_strategy,
    )
    if not vectors:
        raise ConfigError("No gradient samples available after filtering.")

    cov = _covariance_matrix(vectors)
    n_params = len(param_ids)
    if n_params == 0:
        raise ConfigError("No parameters available for active subspace.")

    target_k = k if k is not None else min(DEFAULT_K, n_params)
    if target_k > n_params:
        target_k = n_params

    eigenvalues, eigenvectors = _eigh_symmetric(cov, target_k)
    if not eigenvectors:
        raise ConfigError("Failed to compute active subspace directions.")

    total_variance = sum(value for value in eigenvalues if value > 0.0)
    meta = {
        "sensitivity_id": sensitivity_id,
        "sample_count": sample_count,
        "param_count": n_params,
        "targets": list(targets),
        "missing_strategy": missing_strategy,
    }
    meta_json = json.dumps(meta, ensure_ascii=True, sort_keys=True)

    rows_out: list[dict[str, Any]] = []
    for comp_index, (eigenvalue, vector) in enumerate(
        zip(eigenvalues, eigenvectors),
        start=0,
    ):
        variance_ratio = (
            float(eigenvalue) / total_variance if total_variance > 0.0 else math.nan
        )
        for param_id, loading in zip(param_ids, vector):
            rows_out.append(
                {
                    "component": comp_index,
                    "rank": comp_index + 1,
                    "param_id": param_id,
                    "loading": loading,
                    "eigenvalue": float(eigenvalue),
                    "variance_ratio": variance_ratio,
                    "meta_json": meta_json,
                }
            )

    inputs_payload = {
        "sensitivity_id": sensitivity_id,
        "k": target_k,
        "targets": list(targets),
        "missing_strategy": missing_strategy,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    manifest = build_manifest(
        kind="subspaces",
        artifact_id=artifact_id,
        parents=[sensitivity_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_subspace_table(rows_out, base_dir / "subspace.parquet")

    return store.ensure(manifest, writer=_writer)


class ActiveSubspaceTask(Task):
    name = "dimred.active_subspace"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run(cfg, store=store, registry=registry)


register("task", "dimred.active_subspace", ActiveSubspaceTask())

__all__ = ["ActiveSubspaceTask", "run"]
