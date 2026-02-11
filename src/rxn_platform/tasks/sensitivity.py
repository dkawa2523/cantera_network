"""Sensitivity task: multiplier finite-difference scans and virtual deletion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional
from rxn_platform.core import (
    make_artifact_id,
    make_run_id,
    normalize_reaction_multipliers,
    resolve_repo_path,
)
from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import write_json_atomic
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
from rxn_platform.tasks.base import Task
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

DEFAULT_EPS = 1.0e-2
DEFAULT_DEFINITION = "dlogk"
DEFAULT_RANK_BY = "abs"
DEFAULT_MODE = "multiplier_fd"
DEFAULT_STABILITY_TOP_N = 10
REQUIRED_COLUMNS = (
    "run_id",
    "target",
    "reaction_id",
    "reaction_index",
    "value",
    "unit",
    "rank",
    "meta_json",
)


@dataclass(frozen=True)
class ReactionSpec:
    reaction_id: str
    reaction_index: Optional[int]
    key: tuple[str, Any]


@dataclass(frozen=True)
class ConditionSpec:
    sim_cfg: dict[str, Any]
    condition_id: Optional[str]


def _extract_sensitivity_cfg(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sens_cfg = cfg.get("sensitivity")
    if isinstance(sens_cfg, Mapping):
        return dict(cfg), dict(sens_cfg)
    return dict(cfg), dict(cfg)


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


def _extract_params(sens_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = sens_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("sensitivity.params must be a mapping.")
    return dict(params)


def _extract_mode(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    value = None
    for source in (params, sens_cfg):
        for key in ("mode", "method", "kind"):
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return DEFAULT_MODE
    if not isinstance(value, str):
        raise ConfigError("mode must be a string.")
    mode = value.strip().lower()
    alias_map = {
        "multiplier_fd": "multiplier_fd",
        "fd": "multiplier_fd",
        "finite_difference": "multiplier_fd",
        "finite-difference": "multiplier_fd",
        "virtual_deletion": "virtual_deletion",
        "virtual-delete": "virtual_deletion",
        "virtual_delete": "virtual_deletion",
        "deletion": "virtual_deletion",
        "multiplier_zero": "virtual_deletion",
        "multiplier0": "virtual_deletion",
        "zero": "virtual_deletion",
    }
    if mode not in alias_map:
        raise ConfigError("mode must be 'multiplier_fd' or 'virtual_deletion'.")
    return alias_map[mode]


def _extract_stability_top_n(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[int]:
    value = None
    for source in (params, sens_cfg):
        for key in ("stability_top_n", "rank_stability_top_n", "top_n"):
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError("stability_top_n must be an int.")
    try:
        top_n = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("stability_top_n must be an int.") from exc
    if top_n <= 0:
        raise ConfigError("stability_top_n must be positive.")
    return top_n


def _normalize_conditions(value: Any) -> list[ConditionSpec]:
    if value is None:
        raise ConfigError("conditions must be a non-empty list.")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError("conditions must be a sequence of mappings.")
    specs: list[ConditionSpec] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"conditions[{index}] must be a mapping.")
        if "sim" in entry:
            sim_cfg = entry.get("sim")
            if not isinstance(sim_cfg, Mapping):
                raise ConfigError(f"conditions[{index}].sim must be a mapping.")
            condition_id = entry.get("id", entry.get("condition_id"))
            if condition_id is not None:
                condition_id = _require_nonempty_str(
                    condition_id, f"conditions[{index}].id"
                )
            specs.append(
                ConditionSpec(sim_cfg=dict(sim_cfg), condition_id=condition_id)
            )
            continue
        sim_cfg = dict(entry)
        condition_id = sim_cfg.pop("condition_id", None)
        if condition_id is not None:
            condition_id = _require_nonempty_str(
                condition_id, f"conditions[{index}].condition_id"
            )
        specs.append(ConditionSpec(sim_cfg=sim_cfg, condition_id=condition_id))
    if not specs:
        raise ConfigError("conditions must not be empty.")
    return specs


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ConfigError(f"conditions_file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ConfigError(f"conditions_file is empty: {path}")
    return rows


def _coerce_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be numeric.") from exc


def _apply_csv_condition(
    sim_cfg: Mapping[str, Any],
    *,
    temperature: Optional[float],
    pressure_atm: Optional[float],
    phi: Optional[float],
    t_end: Optional[float],
    case_id: Optional[str],
) -> dict[str, Any]:
    updated = dict(sim_cfg)
    initial = dict(updated.get("initial", {}))
    if temperature is not None:
        initial["T"] = temperature
    if pressure_atm is not None:
        initial["P"] = pressure_atm * 101325.0
    if phi is not None:
        if phi <= 0.0:
            raise ConfigError("phi must be positive.")
        composition = dict(initial.get("X") or {})
        composition["CH4"] = 1.0
        composition["O2"] = 2.0 / float(phi)
        composition["N2"] = composition["O2"] * 3.76
        initial["X"] = composition
    if initial:
        updated["initial"] = initial

    if t_end is not None:
        if "time_grid" in updated and isinstance(updated.get("time_grid"), Mapping):
            time_grid = dict(updated.get("time_grid") or {})
            time_grid["stop"] = t_end
            updated["time_grid"] = time_grid
        elif "time" in updated and isinstance(updated.get("time"), Mapping):
            time_cfg = dict(updated.get("time") or {})
            time_cfg["stop"] = t_end
            updated["time"] = time_cfg
        else:
            updated["time_grid"] = {"start": 0.0, "stop": t_end, "steps": 4}
    if case_id:
        updated["condition_id"] = case_id
    return updated


def _coerce_case_ids(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ConfigError("case_id must be a non-empty string.")
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError("case_id entries must be non-empty strings.")
            items.append(entry.strip())
        return items
    raise ConfigError("case_id must be a string or list of strings.")


def _coerce_row_indices(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError("row_index must be an integer.")
    if isinstance(value, int):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        indices: list[int] = []
        for entry in value:
            if isinstance(entry, bool):
                raise ConfigError("row_index entries must be integers.")
            try:
                indices.append(int(entry))
            except (TypeError, ValueError) as exc:
                raise ConfigError("row_index entries must be integers.") from exc
        return indices
    raise ConfigError("row_index must be an integer or list of integers.")


def _extract_conditions_file_settings(
    cfg: Mapping[str, Any],
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[Path, Optional[list[str]], Optional[list[int]], str]:
    sources: list[Mapping[str, Any]] = []
    for source in (params, sens_cfg, cfg):
        if isinstance(source, Mapping):
            sources.append(source)
        benchmarks = source.get("benchmarks") if isinstance(source, Mapping) else None
        if isinstance(benchmarks, Mapping):
            sources.append(benchmarks)

    conditions_file: Optional[Any] = None
    for source in sources:
        for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
            if key in source:
                conditions_file = source.get(key)
                break
        if conditions_file is not None:
            break
    if conditions_file is None:
        raise ConfigError("conditions_file is required for sensitivity conditions_file.")
    if not isinstance(conditions_file, (str, Path)) or not str(conditions_file).strip():
        raise ConfigError("conditions_file must be a non-empty string or Path.")
    conditions_path = resolve_repo_path(conditions_file)

    case_ids: Optional[list[str]] = None
    row_indices: Optional[list[int]] = None
    case_col: Optional[str] = None
    for source in sources:
        if case_ids is None:
            for key in ("case_ids", "case_id", "condition_id"):
                if key in source and source.get(key) is not None:
                    case_ids = _coerce_case_ids(source.get(key))
                    break
        if row_indices is None and "row_index" in source:
            row_indices = _coerce_row_indices(source.get("row_index"))
        if case_col is None:
            for key in ("case_column", "case_col", "case_field"):
                if key in source and source.get(key) is not None:
                    case_col = source.get(key)
                    break
        if case_ids is not None and row_indices is not None and case_col is not None:
            break
    if case_col is None:
        case_col = "case_id"
    if not isinstance(case_col, str) or not case_col.strip():
        raise ConfigError("case_column must be a non-empty string.")
    case_col = case_col.strip()

    return conditions_path, case_ids, row_indices, case_col


def _build_condition_specs_from_csv(
    sim_cfg: Mapping[str, Any],
    *,
    conditions_path: Path,
    case_ids: Optional[list[str]],
    row_indices: Optional[list[int]],
    case_col: str,
) -> list[ConditionSpec]:
    rows = _load_csv_rows(conditions_path)
    selected: list[dict[str, str]] = []
    if case_ids:
        for row in rows:
            if row.get(case_col) in case_ids:
                selected.append(row)
        if not selected:
            raise ConfigError("No matching case_id found in conditions file.")
    elif row_indices:
        for idx in row_indices:
            if idx < 0 or idx >= len(rows):
                raise ConfigError("row_index out of range for conditions file.")
            selected.append(rows[idx])
    else:
        selected = rows

    specs: list[ConditionSpec] = []
    for row in selected:
        def _pick(keys: Sequence[str]) -> Optional[str]:
            for key in keys:
                if key in row and row.get(key) is not None:
                    value = row.get(key)
                    if isinstance(value, str) and not value.strip():
                        return None
                    return value
            return None

        temperature = _coerce_optional_float(
            _pick(("T0", "T", "temperature")), "temperature"
        )
        pressure_atm = _coerce_optional_float(
            _pick(("P0_atm", "P_atm", "P0", "pressure_atm", "pressure")),
            "pressure_atm",
        )
        phi = _coerce_optional_float(_pick(("phi",)), "phi")
        t_end = _coerce_optional_float(_pick(("t_end", "t_end_s", "t_end_seconds")), "t_end")
        row_case_id = row.get(case_col) if case_col in row else None
        if isinstance(row_case_id, str) and not row_case_id.strip():
            row_case_id = None

        updated = _apply_csv_condition(
            sim_cfg,
            temperature=temperature,
            pressure_atm=pressure_atm,
            phi=phi,
            t_end=t_end,
            case_id=row_case_id,
        )
        specs.append(ConditionSpec(sim_cfg=updated, condition_id=row_case_id))
    if not specs:
        raise ConfigError("No conditions resolved for sensitivity computation.")
    return specs

def _normalize_sim_cfgs(value: Any) -> list[ConditionSpec]:
    if isinstance(value, Mapping):
        sim_cfg = dict(value)
        condition_id = sim_cfg.pop("condition_id", None)
        if condition_id is not None:
            condition_id = _require_nonempty_str(condition_id, "sim.condition_id")
        return [ConditionSpec(sim_cfg=sim_cfg, condition_id=condition_id)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        specs: list[ConditionSpec] = []
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise ConfigError(f"sim[{index}] must be a mapping.")
            sim_cfg = dict(entry)
            condition_id = sim_cfg.pop("condition_id", None)
            if condition_id is not None:
                condition_id = _require_nonempty_str(
                    condition_id, f"sim[{index}].condition_id"
                )
            specs.append(ConditionSpec(sim_cfg=sim_cfg, condition_id=condition_id))
        if not specs:
            raise ConfigError("sim list must not be empty.")
        return specs
    raise ConfigError("sensitivity requires a sim config mapping or list.")


def _extract_sim_cfgs(
    cfg: Mapping[str, Any],
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[ConditionSpec]:
    conditions = None
    for source in (params, sens_cfg, cfg):
        if isinstance(source, Mapping) and "conditions" in source:
            conditions = source.get("conditions")
            break
    if conditions is not None:
        return _normalize_conditions(conditions)

    conditions_file_present = False
    for source in (params, sens_cfg, cfg):
        if not isinstance(source, Mapping):
            continue
        if "conditions_file" in source or "conditions_path" in source or "conditions_csv" in source:
            conditions_file_present = True
            break
        benchmarks = source.get("benchmarks")
        if isinstance(benchmarks, Mapping) and "conditions_file" in benchmarks:
            conditions_file_present = True
            break
    if conditions_file_present:
        sim_cfg: Any = None
        for source in (sens_cfg, params, cfg):
            if isinstance(source, Mapping) and "sim" in source:
                sim_cfg = source.get("sim")
                break
        if sim_cfg is None:
            inputs = sens_cfg.get("inputs")
            if isinstance(inputs, Mapping) and "sim" in inputs:
                sim_cfg = inputs.get("sim")
        if not isinstance(sim_cfg, Mapping):
            raise ConfigError("sim config mapping is required with conditions_file.")
        conditions_path, case_ids, row_indices, case_col = _extract_conditions_file_settings(
            cfg,
            sens_cfg,
            params,
        )
        return _build_condition_specs_from_csv(
            dict(sim_cfg),
            conditions_path=conditions_path,
            case_ids=case_ids,
            row_indices=row_indices,
            case_col=case_col,
        )

    sim_cfg: Any = None
    for source in (sens_cfg, params, cfg):
        if isinstance(source, Mapping) and "sim" in source:
            sim_cfg = source.get("sim")
            break
    if sim_cfg is None:
        inputs = sens_cfg.get("inputs")
        if isinstance(inputs, Mapping) and "sim" in inputs:
            sim_cfg = inputs.get("sim")
    return _normalize_sim_cfgs(sim_cfg)


def _extract_observables(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Any:
    for source in (params, sens_cfg):
        if "observables" in source:
            return source.get("observables")
    raise ConfigError("sensitivity observables must be provided.")


def _extract_targets(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[str]:
    targets: Any = None
    for source in (params, sens_cfg):
        if "targets" in source:
            targets = source.get("targets")
            break
    return _coerce_str_sequence(targets, "targets")


def _extract_missing_strategy(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    value = None
    for source in (params, sens_cfg):
        if "missing_strategy" in source:
            value = source.get("missing_strategy")
            break
    if value is None:
        return "nan"
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    strategy = value.strip().lower()
    if strategy not in {"nan", "skip"}:
        raise ConfigError("missing_strategy must be 'nan' or 'skip'.")
    return strategy


def _extract_eps(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    for source in (params, sens_cfg):
        for key in ("eps", "epsilon", "delta"):
            if key in source:
                value = source.get(key)
                break
        else:
            continue
        break
    else:
        return DEFAULT_EPS
    try:
        eps = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("eps must be a float.") from exc
    if eps <= 0.0:
        raise ConfigError("eps must be positive.")
    return eps


def _extract_definition(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    definition = None
    for source in (params, sens_cfg):
        if "definition" in source:
            definition = source.get("definition")
            break
    if definition is None:
        return DEFAULT_DEFINITION
    if not isinstance(definition, str):
        raise ConfigError("definition must be a string.")
    return definition.strip().lower()


def _extract_rank_by(
    sens_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    rank_by = None
    for source in (params, sens_cfg):
        if "rank_by" in source:
            rank_by = source.get("rank_by")
            break
    if rank_by is None:
        return DEFAULT_RANK_BY
    if not isinstance(rank_by, str):
        raise ConfigError("rank_by must be a string.")
    rank_by = rank_by.strip().lower()
    if rank_by not in {"abs", "raw"}:
        raise ConfigError("rank_by must be 'abs' or 'raw'.")
    return rank_by


def _normalize_reaction_entry(entry: Any, label: str) -> ReactionSpec:
    if isinstance(entry, bool):
        raise ConfigError(f"{label} must be an int or string.")
    if isinstance(entry, int):
        if entry < 0:
            raise ConfigError(f"{label} index must be non-negative.")
        reaction_id = f"index:{entry}"
        return ReactionSpec(reaction_id=reaction_id, reaction_index=entry, key=("index", entry))
    if isinstance(entry, str):
        reaction_id = _require_nonempty_str(entry, label)
        return ReactionSpec(reaction_id=reaction_id, reaction_index=None, key=("reaction_id", reaction_id))
    if isinstance(entry, Mapping):
        if "index" in entry:
            index = entry.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ConfigError(f"{label}.index must be an int.")
            if index < 0:
                raise ConfigError(f"{label}.index must be non-negative.")
            reaction_id = f"index:{index}"
            return ReactionSpec(reaction_id=reaction_id, reaction_index=index, key=("index", index))
        for key in ("reaction_id", "id"):
            if key in entry:
                reaction_id = _require_nonempty_str(entry.get(key), f"{label}.{key}")
                return ReactionSpec(
                    reaction_id=reaction_id,
                    reaction_index=None,
                    key=("reaction_id", reaction_id),
                )
    raise ConfigError(f"{label} must be an int, string, or mapping.")


def _normalize_reactions(value: Any) -> list[ReactionSpec]:
    if value is None:
        raise ConfigError("reactions list is required.")
    if isinstance(value, Mapping):
        if any(key in value for key in ("indices", "reaction_indices")):
            value = value.get("indices", value.get("reaction_indices"))
        elif any(key in value for key in ("ids", "reaction_ids")):
            value = value.get("ids", value.get("reaction_ids"))
        else:
            return [_normalize_reaction_entry(value, "reactions")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        specs: list[ReactionSpec] = []
        seen: set[tuple[str, Any]] = set()
        for index, entry in enumerate(value):
            spec = _normalize_reaction_entry(entry, f"reactions[{index}]")
            if spec.key in seen:
                continue
            seen.add(spec.key)
            specs.append(spec)
        if not specs:
            raise ConfigError("reactions list must not be empty.")
        return specs
    raise ConfigError("reactions must be a sequence or mapping.")


def _coerce_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    return {"detail": meta}


def _coerce_value(value: Any, meta: dict[str, Any]) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        meta["error"] = "value must be numeric"
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        meta["error"] = f"value not numeric: {value!r}"
        return math.nan


def _build_row(
    run_id: str,
    target: str,
    reaction_id: str,
    reaction_index: Optional[int],
    value: Any,
    unit: Any,
    rank: Optional[int],
    meta: Any,
) -> dict[str, Any]:
    target_name = _require_nonempty_str(target, "target")
    reaction_name = _require_nonempty_str(reaction_id, "reaction_id")
    unit_str = "" if unit is None else str(unit)
    meta_payload = _coerce_meta(meta)
    value_float = _coerce_value(value, meta_payload)
    try:
        meta_json = json.dumps(
            meta_payload,
            ensure_ascii=True,
            sort_keys=True,
        )
    except TypeError:
        meta_json = json.dumps(
            {"detail": str(meta_payload)},
            ensure_ascii=True,
            sort_keys=True,
        )
    return {
        "run_id": run_id,
        "target": target_name,
        "reaction_id": reaction_name,
        "reaction_index": reaction_index,
        "value": value_float,
        "unit": unit_str,
        "rank": rank,
        "meta_json": meta_json,
    }


def _collect_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(REQUIRED_COLUMNS)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_sensitivity_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
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
    json_path = path.with_suffix(".json")
    if json_path != path:
        write_json_atomic(json_path, payload)
    logger = logging.getLogger("rxn_platform.sensitivity")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s and %s.",
        path,
        json_path,
    )


def _multiplier_sort_key(entry: Mapping[str, Any]) -> tuple[int, Any]:
    if "index" in entry:
        return (0, entry["index"])
    return (1, entry["reaction_id"])


def _build_multiplier_map(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, Any], float]:
    mapping: dict[tuple[str, Any], float] = {}
    for entry in entries:
        if "index" in entry:
            key = ("index", entry["index"])
        else:
            key = ("reaction_id", entry["reaction_id"])
        mapping[key] = float(entry.get("multiplier", 1.0))
    return mapping


def _rebuild_multipliers(mapping: Mapping[tuple[str, Any], float]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, multiplier in mapping.items():
        kind, value = key
        entry: dict[str, Any] = {"multiplier": multiplier}
        if kind == "index":
            entry["index"] = value
        else:
            entry["reaction_id"] = value
        entries.append(entry)
    return sorted(entries, key=_multiplier_sort_key)


def _normalize_sim_cfg(sim_cfg: Mapping[str, Any], multipliers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = dict(sim_cfg)
    if multipliers:
        normalized["reaction_multipliers"] = list(multipliers)
    else:
        normalized.pop("reaction_multipliers", None)
    normalized.pop("disabled_reactions", None)
    return normalized


def _sim_run_id(sim_cfg: Mapping[str, Any]) -> str:
    manifest_cfg = {"sim": sim_cfg, "inputs": {}, "params": {}}
    return make_run_id(manifest_cfg, exclude_keys=("hydra",))


def _compute_sensitivity(
    base_value: float,
    pert_value: float,
    *,
    definition: str,
    eps: float,
) -> tuple[float, str]:
    if math.isnan(base_value) or math.isnan(pert_value):
        return math.nan, ""
    delta = pert_value - base_value
    if definition in {"dlogk", "dydlogk", "dy_dlogk"}:
        denom = math.log1p(eps)
        return delta / denom, ""
    if definition in {"dy", "delta"}:
        return delta, ""
    if definition in {"dlny_dlnk", "dlogy_dlogk"}:
        if base_value == 0.0:
            return math.nan, ""
        denom = math.log1p(eps)
        return delta / (base_value * denom), "1"
    if definition in {"relative"}:
        if base_value == 0.0:
            return math.nan, ""
        return delta / base_value, "1"
    raise ConfigError(f"Unknown sensitivity definition: {definition!r}.")


def _compute_impact(base_value: float, pert_value: float) -> float:
    if math.isnan(base_value) or math.isnan(pert_value):
        return math.nan
    return pert_value - base_value


def _assign_ranks(rows: list[dict[str, Any]], rank_by: str) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {}
    for row in rows:
        value = row.get("value")
        if not isinstance(value, (int, float)) or math.isnan(float(value)):
            row["rank"] = None
            continue
        metric = abs(float(value)) if rank_by == "abs" else float(value)
        target = row.get("target")
        if not isinstance(target, str) or not target.strip():
            row["rank"] = None
            continue
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            condition_id = row.get("run_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            row["rank"] = None
            continue
        grouped.setdefault((target, condition_id), []).append((metric, row))

    for entries in grouped.values():
        entries.sort(key=lambda item: item[0], reverse=True)
        for rank, (_, row) in enumerate(entries, start=1):
            row["rank"] = rank


def _safe_metric_value(value: Any) -> float:
    if value is None:
        return -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    if math.isnan(number):
        return -math.inf
    return number


def _select_top_items(values: Mapping[str, Any], top_n: Optional[int]) -> list[str]:
    ranked = sorted(
        values.items(),
        key=lambda item: (-_safe_metric_value(item[1]), item[0]),
    )
    if top_n is None or top_n >= len(ranked):
        return [item[0] for item in ranked]
    return [item[0] for item in ranked[:top_n]]


def _filter_finite(values: Mapping[str, Any]) -> dict[str, float]:
    filtered: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            filtered[key] = number
    return filtered


def _rank_nodes(values: Mapping[str, float]) -> dict[str, float]:
    items = sorted(
        values.items(),
        key=lambda item: (-item[1], item[0]),
    )
    ranks: dict[str, float] = {}
    index = 0
    count = len(items)
    while index < count:
        value = items[index][1]
        start = index
        while index < count and items[index][1] == value:
            index += 1
        end = index
        rank_value = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[items[pos][0]] = rank_value
    return ranks


def _pearson_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return math.nan
    mean_a = sum(values_a) / float(len(values_a))
    mean_b = sum(values_b) / float(len(values_b))
    var_a = 0.0
    var_b = 0.0
    cov = 0.0
    for a, b in zip(values_a, values_b):
        da = a - mean_a
        db = b - mean_b
        var_a += da * da
        var_b += db * db
        cov += da * db
    if var_a <= 0.0 or var_b <= 0.0:
        return math.nan
    return cov / math.sqrt(var_a * var_b)


def _mean_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / float(len(values))


def _min_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return min(values)


def _max_value(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return max(values)


def _default_top_k(values_by_run: Mapping[str, Mapping[str, float]]) -> Optional[int]:
    min_nodes = min((len(values) for values in values_by_run.values()), default=0)
    if min_nodes <= 0:
        return None
    return min(DEFAULT_STABILITY_TOP_N, min_nodes)


def _compute_rank_stability(
    values_by_run: Mapping[str, Mapping[str, Any]],
    *,
    top_n: Optional[int],
) -> dict[str, Any]:
    run_ids = sorted(values_by_run.keys())
    stability: dict[str, Any] = {
        "run_count": len(run_ids),
        "pair_count": 0,
    }
    if len(run_ids) < 2:
        stability["status"] = "insufficient_runs"
        return stability

    filtered_values = {rid: _filter_finite(values) for rid, values in values_by_run.items()}
    ranks_by_run = {rid: _rank_nodes(values) for rid, values in filtered_values.items()}

    spearman_values: list[float] = []
    for idx_a, run_a in enumerate(run_ids):
        for run_b in run_ids[idx_a + 1 :]:
            common = set(ranks_by_run[run_a]).intersection(ranks_by_run[run_b])
            if len(common) < 2:
                continue
            scores_a = [ranks_by_run[run_a][node] for node in common]
            scores_b = [ranks_by_run[run_b][node] for node in common]
            corr = _pearson_corr(scores_a, scores_b)
            if math.isfinite(corr):
                spearman_values.append(corr)
    stability["pair_count"] = len(run_ids) * (len(run_ids) - 1) // 2
    stability["spearman_mean"] = _mean_value(spearman_values)
    stability["spearman_min"] = _min_value(spearman_values)
    stability["spearman_max"] = _max_value(spearman_values)

    top_k = top_n if top_n is not None else _default_top_k(filtered_values)
    stability["top_k"] = top_k
    jaccard_values: list[float] = []
    if top_k is not None and top_k > 0:
        top_sets: dict[str, set[str]] = {}
        for run_id, values in filtered_values.items():
            top_sets[run_id] = set(_select_top_items(values, top_k))
        for idx_a, run_a in enumerate(run_ids):
            for run_b in run_ids[idx_a + 1 :]:
                union = top_sets[run_a].union(top_sets[run_b])
                if not union:
                    continue
                intersect = top_sets[run_a].intersection(top_sets[run_b])
                jaccard_values.append(len(intersect) / float(len(union)))
    stability["top_k_jaccard_mean"] = _mean_value(jaccard_values)
    stability["top_k_jaccard_min"] = _min_value(jaccard_values)
    stability["top_k_jaccard_max"] = _max_value(jaccard_values)

    stability["status"] = "computed" if spearman_values or jaccard_values else "insufficient_overlap"
    return stability


def _apply_rank_stability(
    rows: list[dict[str, Any]],
    *,
    rank_by: str,
    top_n: Optional[int],
) -> None:
    values_by_target: dict[str, dict[str, dict[str, float]]] = {}
    target_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        target = row.get("target")
        run_id = row.get("condition_id") or row.get("run_id")
        reaction_id = row.get("reaction_id")
        if not isinstance(target, str) or not target.strip():
            continue
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        if not isinstance(reaction_id, str) or not reaction_id.strip():
            continue
        metric_value = row.get("value")
        try:
            metric = float(metric_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric):
            continue
        if rank_by == "abs":
            metric = abs(metric)
        values_by_target.setdefault(target, {}).setdefault(run_id, {})[reaction_id] = metric
        target_rows.setdefault(target, []).append(index)

    for target, values_by_run in values_by_target.items():
        stability = _compute_rank_stability(values_by_run, top_n=top_n)
        for index in target_rows.get(target, []):
            meta_json = rows[index].get("meta_json")
            if isinstance(meta_json, str):
                try:
                    meta = json.loads(meta_json)
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = {}
            meta["rank_stability"] = stability
            rows[index]["meta_json"] = json.dumps(
                meta,
                ensure_ascii=True,
                sort_keys=True,
            )


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_observable_rows(
    store: ArtifactStore,
    observable_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    obs_dir = store.artifact_dir("observables", observable_id)
    values_path = obs_dir / "values.parquet"
    if not values_path.exists():
        raise ConfigError(f"Observable values not found: {values_path}")
    rows = _read_table_rows(values_path)
    return [row for row in rows if row.get("run_id") == run_id]


def _values_by_target(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapping: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = row.get("observable")
        if not isinstance(name, str) or not name.strip():
            continue
        if name in mapping:
            continue
        mapping[name] = row
    return mapping


def _run_observables(
    runner: PipelineRunner,
    run_id: str,
    observables_cfg: Mapping[str, Any],
) -> str:
    pipeline_cfg = {
        "steps": [
            {
                "id": "obs",
                "task": "observables.run",
                "inputs": {"run_id": run_id},
                "params": dict(observables_cfg),
            }
        ]
    }
    results = runner.run(pipeline_cfg)
    return results["obs"]


def _run_sim_and_observables(
    runner: PipelineRunner,
    store: ArtifactStore,
    sim_cfg: Mapping[str, Any],
    observables_cfg: Mapping[str, Any],
) -> tuple[str, str]:
    run_id = _sim_run_id(sim_cfg)
    if store.exists("runs", run_id):
        obs_id = _run_observables(runner, run_id, observables_cfg)
        return run_id, obs_id
    pipeline_cfg = {
        "steps": [
            {"id": "sim", "task": "sim.run", "sim": dict(sim_cfg)},
            {
                "id": "obs",
                "task": "observables.run",
                "inputs": {"run_id": "@sim"},
                "params": dict(observables_cfg),
            },
        ]
    }
    results = runner.run(pipeline_cfg)
    return results["sim"], results["obs"]


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Compute sensitivity rankings using finite-difference or virtual deletion."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, sens_cfg = _extract_sensitivity_cfg(resolved_cfg)
    params = _extract_params(sens_cfg)

    condition_specs = _extract_sim_cfgs(resolved_cfg, sens_cfg, params)
    observables_raw = _extract_observables(sens_cfg, params)
    targets = _extract_targets(sens_cfg, params)
    missing_strategy = _extract_missing_strategy(sens_cfg, params)
    eps = _extract_eps(sens_cfg, params)
    definition = _extract_definition(sens_cfg, params)
    rank_by = _extract_rank_by(sens_cfg, params)
    mode = _extract_mode(sens_cfg, params)
    stability_top_n = _extract_stability_top_n(sens_cfg, params)
    reactions = _normalize_reactions(params.get("reactions", sens_cfg.get("reactions")))

    observables_cfg = {
        "observables": observables_raw,
        "missing_strategy": missing_strategy,
    }

    logger = logging.getLogger("rxn_platform.sensitivity")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    run_obs_cache: dict[str, str] = {}
    obs_rows_cache: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    condition_states: list[dict[str, Any]] = []
    seen_baseline: set[str] = set()
    target_candidates: set[str] = set()

    for spec in condition_specs:
        try:
            base_multipliers = normalize_reaction_multipliers(spec.sim_cfg)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"reaction multipliers are invalid: {exc}") from exc
        base_multiplier_map = _build_multiplier_map(base_multipliers)
        base_sim_cfg = _normalize_sim_cfg(spec.sim_cfg, base_multipliers)

        run_id = _sim_run_id(base_sim_cfg)
        if run_id not in run_obs_cache:
            baseline_run_id, baseline_obs_id = _run_sim_and_observables(
                runner,
                store,
                base_sim_cfg,
                observables_cfg,
            )
            run_obs_cache[baseline_run_id] = baseline_obs_id
        baseline_run_id = run_id
        baseline_obs_id = run_obs_cache[baseline_run_id]
        condition_id = spec.condition_id or baseline_run_id

        if baseline_run_id not in seen_baseline:
            baseline_rows = _extract_observable_rows(
                store, baseline_obs_id, baseline_run_id
            )
            baseline_by_target = _values_by_target(baseline_rows)
            obs_rows_cache[(baseline_obs_id, baseline_run_id)] = baseline_by_target
            target_candidates.update(baseline_by_target.keys())
            condition_states.append(
                {
                    "baseline_run_id": baseline_run_id,
                    "baseline_obs_id": baseline_obs_id,
                    "baseline_by_target": baseline_by_target,
                    "base_multiplier_map": base_multiplier_map,
                    "sim_cfg": spec.sim_cfg,
                    "condition_id": condition_id,
                }
            )
            seen_baseline.add(baseline_run_id)

    if not condition_states:
        raise ConfigError("No conditions resolved for sensitivity computation.")
    if not targets:
        targets = sorted(target_candidates)
    if not targets:
        raise ConfigError("No targets resolved for sensitivity computation.")

    rows: list[dict[str, Any]] = []
    perturbed_run_ids: list[str] = []
    perturbed_obs_ids: list[str] = []

    for state in condition_states:
        baseline_run_id = state["baseline_run_id"]
        baseline_obs_id = state["baseline_obs_id"]
        baseline_by_target = state["baseline_by_target"]
        base_multiplier_map = state["base_multiplier_map"]
        sim_cfg = state["sim_cfg"]
        condition_id = state["condition_id"]

        for reaction in reactions:
            perturbed_map = dict(base_multiplier_map)
            base_multiplier = perturbed_map.get(reaction.key, 1.0)
            if mode == "virtual_deletion":
                perturbed_multiplier = 0.0
            else:
                perturbed_multiplier = base_multiplier * (1.0 + eps)
            perturbed_map[reaction.key] = perturbed_multiplier
            perturbed_multipliers = _rebuild_multipliers(perturbed_map)
            pert_sim_cfg = _normalize_sim_cfg(sim_cfg, perturbed_multipliers)

            pert_run_id = _sim_run_id(pert_sim_cfg)
            if pert_run_id not in run_obs_cache:
                run_id, obs_id = _run_sim_and_observables(
                    runner,
                    store,
                    pert_sim_cfg,
                    observables_cfg,
                )
                run_obs_cache[run_id] = obs_id
            pert_obs_id = run_obs_cache[pert_run_id]
            perturbed_run_ids.append(pert_run_id)
            perturbed_obs_ids.append(pert_obs_id)

            cache_key = (pert_obs_id, pert_run_id)
            pert_by_target = obs_rows_cache.get(cache_key)
            if pert_by_target is None:
                pert_rows = _extract_observable_rows(store, pert_obs_id, pert_run_id)
                pert_by_target = _values_by_target(pert_rows)
                obs_rows_cache[cache_key] = pert_by_target

            for target in targets:
                base_row = baseline_by_target.get(target)
                pert_row = pert_by_target.get(target)
                meta: dict[str, Any] = {
                    "baseline_run_id": baseline_run_id,
                    "perturbed_run_id": pert_run_id,
                    "baseline_observable_id": baseline_obs_id,
                    "perturbed_observable_id": pert_obs_id,
                    "mode": mode,
                    "base_multiplier": base_multiplier,
                    "perturbed_multiplier": perturbed_multiplier,
                }
                if mode != "virtual_deletion":
                    meta["definition"] = definition
                    meta["eps"] = eps
                else:
                    meta["definition"] = "impact"
                base_value = math.nan
                pert_value = math.nan
                base_unit = ""
                pert_unit = ""
                missing: list[str] = []
                if base_row is None:
                    missing.append("baseline")
                else:
                    base_unit = base_row.get("unit", "")
                    base_value = _coerce_value(base_row.get("value"), meta)
                    meta["baseline_meta_json"] = base_row.get("meta_json")
                if pert_row is None:
                    missing.append("perturbed")
                else:
                    pert_unit = pert_row.get("unit", "")
                    pert_value = _coerce_value(pert_row.get("value"), meta)
                    meta["perturbed_meta_json"] = pert_row.get("meta_json")
                if missing:
                    meta["status"] = "missing_target"
                    meta["missing"] = missing
                unit = base_unit if base_unit or not pert_unit else pert_unit
                impact = _compute_impact(base_value, pert_value)
                if mode == "virtual_deletion":
                    value = impact
                else:
                    value, unit_override = _compute_sensitivity(
                        base_value,
                        pert_value,
                        definition=definition,
                        eps=eps,
                    )
                    if unit_override:
                        unit = unit_override
                row = _build_row(
                    baseline_run_id,
                    target,
                    reaction.reaction_id,
                    reaction.reaction_index,
                    value,
                    unit,
                    None,
                    meta,
                )
                row["impact"] = impact
                row["condition_id"] = condition_id
                rows.append(row)

    _assign_ranks(rows, rank_by)
    _apply_rank_stability(rows, rank_by=rank_by, top_n=stability_top_n)

    inputs_payload = {
        "perturbed_run_ids": perturbed_run_ids,
        "reactions": [spec.reaction_id for spec in reactions],
        "targets": list(targets),
    }
    if len(condition_states) == 1:
        inputs_payload["baseline_run_id"] = condition_states[0]["baseline_run_id"]
    else:
        inputs_payload["baseline_run_ids"] = [
            state["baseline_run_id"] for state in condition_states
        ]
        inputs_payload["condition_ids"] = [
            state["condition_id"] for state in condition_states
        ]
    if mode != DEFAULT_MODE:
        inputs_payload["mode"] = mode
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    baseline_run_ids = [state["baseline_run_id"] for state in condition_states]
    baseline_obs_ids = [state["baseline_obs_id"] for state in condition_states]
    parents = _dedupe_preserve(
        baseline_run_ids + perturbed_run_ids + baseline_obs_ids + perturbed_obs_ids
    )
    manifest = build_manifest(
        kind="sensitivity",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        _write_sensitivity_table(rows, base_dir / "sensitivity.parquet")

    return store.ensure(manifest, writer=_writer)


class SensitivityTask(Task):
    name = "sensitivity.multiplier_fd"

    def run(
        self,
        cfg: Mapping[str, Any],
        *,
        store: ArtifactStore,
        registry: Optional[Registry] = None,
    ) -> ArtifactCacheResult:
        return run(cfg, store=store, registry=registry)


register("task", "sensitivity.multiplier_fd", SensitivityTask())

__all__ = ["SensitivityTask", "run"]
