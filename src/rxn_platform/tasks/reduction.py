"""Reduction task: apply mechanism patches (disable reactions / multipliers)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import copy
import importlib
import json
import logging
import math
from pathlib import Path
import random
import re
import statistics
import tempfile
import time
from typing import Any, Optional
from rxn_platform.core import (
    ArtifactManifest,
    make_artifact_id,
    normalize_reaction_multipliers,
    resolve_repo_path,
    stable_hash,
)
from rxn_platform.errors import ConfigError
from rxn_platform.io_utils import read_json, write_json_atomic
from rxn_platform.mechanism import (
    MechanismCompiler,
    apply_patch_entries as _apply_patch_entries_shared,
    read_yaml_payload,
    reaction_id_index_map as _reaction_id_index_map_shared,
    reaction_identifiers as _reaction_identifiers_shared,
    resolve_patch_entries as _resolve_patch_entries_shared,
    write_yaml_payload,
)
from rxn_platform.pipelines import PipelineRunner
import rxn_platform.registry as registry_module
from rxn_platform.registry import Registry, register
from rxn_platform.run_store import RUNS_ROOT, read_run_metrics, utc_now_iso as _utc_now_iso
from rxn_platform.store import ArtifactCacheResult, ArtifactStore
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

try:  # Optional dependency.
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

try:  # Optional dependency.
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None

try:  # Optional dependency.
    import cantera as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None

PATCH_SCHEMA_VERSION = 1
PATCH_FILENAME = "mechanism_patch.yaml"
MECHANISM_FILENAME = "mechanism.yaml"
DEFAULT_IMPORTANCE_COLUMN = "value"
DEFAULT_IMPORTANCE_MODE = "abs"
DEFAULT_IMPORTANCE_AGGREGATE = "max"
DEFAULT_VALIDATION_METRIC = "abs"
DEFAULT_VALIDATION_TOLERANCE = 1.0e-6
DEFAULT_VALIDATION_REL_EPS = 1.0e-12
DEFAULT_VALIDATION_MISSING_STRATEGY = "fail"
DEFAULT_AMORE_BEAM_WIDTH = 4
DEFAULT_AMORE_MAX_DEPTH = 3
DEFAULT_AMORE_EXPAND_TOP = 6
DEFAULT_AMORE_CHEAP_SCORE_MIN = 0.85
DEFAULT_AMORE_SURROGATE_DATASET_NAME = "surrogate"
DEFAULT_AMORE_SURROGATE_MIN_SAMPLES = 6
DEFAULT_AMORE_SURROGATE_WARMUP = 4
DEFAULT_AMORE_SURROGATE_K = 5
DEFAULT_AMORE_SURROGATE_UNCERTAINTY = 0.5
DEFAULT_AMORE_SURROGATE_FAIL_SKIP = 0.8
DEFAULT_AMORE_SURROGATE_ERROR_FACTOR = 10.0
DEFAULT_AMORE_SURROGATE_MAX_TYPES = 4
DEFAULT_AMORE_SURROGATE_UPDATE_EVERY = 1
NODE_LUMPING_SCHEMA_VERSION = 1
NODE_LUMPING_FILENAME = "node_lumping.json"
NODE_LUMPING_PRUNE_SCHEMA_VERSION = 1
REACTION_LUMPING_SCHEMA_VERSION = 1
REACTION_LUMPING_FILENAME = "reaction_lumping.json"
REACTION_LUMPING_PRUNE_SCHEMA_VERSION = 1
DEFAULT_LUMPING_THRESHOLD = 0.85
DEFAULT_LUMPING_METHOD = "threshold"
DEFAULT_LUMPING_CHARGE_SCALE = 1.0
DEFAULT_LUMPING_WEIGHTS = {
    "elements": 1.0,
    "charge": 1.0,
    "phase": 1.0,
    "state": 1.0,
    # Optional network-aware similarity components (default off for backward compatibility).
    "reaction_type_profile": 0.0,
    "neighbor_reaction": 0.0,
}
DEFAULT_REPRESENTATIVE_METRIC = "degree"
DEFAULT_REACTION_SIMILARITY_METRIC = "jaccard"
DEFAULT_REACTION_SIMILARITY_MODE = "both"
DEFAULT_REACTION_SIMILARITY_WEIGHTS = {
    "reactants": 1.0,
    "products": 1.0,
}
DEFAULT_REACTION_INCLUDE_PARTICIPANTS = False

# Lightweight periodic-table helpers for "projection-only" superstate mapping.
# Keep this dependency-free (no external installs) and sufficient for common
# combustion mechanisms like GRI30.
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.00784,
    "He": 4.002602,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Br": 79.904,
    "I": 126.90447,
}

DEFAULT_SUPERSTATE_MAPPING_THRESHOLD = 0.93
DEFAULT_SUPERSTATE_MAPPING_MAX_CLUSTER_SIZE = 20
DEFAULT_SUPERSTATE_MAPPING_WEIGHTS = {
    "element_set_jaccard": 0.35,
    "elements_cosine": 0.10,
    # Reaction-id Jaccard tends to be extremely sparse for real mechanisms; keep this
    # component low by default to avoid suppressing all merges.
    "neighbor_reaction": 0.05,
    # Reaction type/role profiles are a more stable signal; make it the primary
    # network-aware component for "merge firing" on GRI30-scale mechanisms.
    "reaction_type_profile": 0.45,
    "charge": 0.05,
}
DEFAULT_SUPERSTATE_MAPPING_MW_LOG_TOL = 1.0
DEFAULT_SUPERSTATE_MAPPING_ELEMENTS_COSINE_MIN = 0.2
DEFAULT_CNR_MAPPING_ID = "cnr_coarse"
DEFAULT_CNR_CONSTRAINT_FIELDS = ("phase", "charge")
DEFAULT_CNR_COMMUNITY_METHOD = "louvain"
DEFAULT_CNR_MIN_WEIGHT = 0.0
DEFAULT_CNR_SYMMETRIZE = True
DEFAULT_CNR_RESOLUTION = 1.0
DEFAULT_GNN_POOL_MAPPING_ID = "gnn_pool"
DEFAULT_GNN_POOL_TIME_DIM = 2
DEFAULT_GNN_POOL_TIME_BASE = 10.0
DEFAULT_GNN_POOL_TEMP = 1.0
DEFAULT_GNN_POOL_MAX_ITER = 25
DEFAULT_EVAL_QOI_SELECTOR = "all"
DEFAULT_EVAL_QOI_RADIUS = 0
DEFAULT_EVAL_MAX_BINS = 20
DEFAULT_EVAL_REPORT_FORMAT = "markdown"
DEFAULT_EVAL_REPORT_MAX_WINDOWS = 12
DEFAULT_GNN_POOL_MESSAGE_PASSING = 1
DEFAULT_GNN_POOL_COVERAGE_SLACK = 0.02
DEFAULT_LEARNCK_STABLE_STATES_PATH = Path("reduction/learnck/stable_states.yaml")
REDUCTION_METHOD_TASKS = {
    "cnr_coarse": "reduction.cnr_coarse",
    "amore_search": "reduction.amore_search",
    "learnck_style": "reduction.learnck_style",
    "gnn_pool_temporal": "reduction.gnn_pool_temporal",
    "gnn_importance_prune": "reduction.gnn_importance_prune",
}
DEFAULT_OVERALL_REACTION_TOL = 1.0e-8
REQUIRED_VALIDATION_COLUMNS = (
    "patch_index",
    "patch_id",
    "passed",
    "status",
    "kind",
    "name",
    "unit",
    "meta_json",
    "item_index",
    "baseline_value",
    "reduced_value",
    "abs_diff",
    "rel_diff",
    "metric",
    "tolerance",
    "baseline_run_id",
    "reduced_run_id",
    "baseline_artifact_id",
    "reduced_artifact_id",
)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _normalize_str_list(raw: Any, label: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [_require_nonempty_str(raw, label)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values: list[str] = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, str) or not entry.strip():
                raise ConfigError(f"{label}[{idx}] must be a non-empty string.")
            values.append(entry.strip())
        return values
    raise ConfigError(f"{label} must be a string or list of strings.")


def _normalize_int_list(raw: Any, label: str) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, bool):
        raise ConfigError(f"{label} must be an integer or list of integers.")
    if isinstance(raw, int):
        if raw < 0:
            raise ConfigError(f"{label} entries must be >= 0.")
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values: list[int] = []
        for idx, entry in enumerate(raw):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ConfigError(f"{label}[{idx}] must be an integer.")
            if entry < 0:
                raise ConfigError(f"{label}[{idx}] must be >= 0.")
            values.append(entry)
        return values
    raise ConfigError(f"{label} must be an integer or list of integers.")


def _normalize_window_split_cfg(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError("window_split must be a mapping.")
    return dict(raw)


def _window_split_from_cfg(
    all_window_ids: Sequence[int],
    split_cfg: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    if not all_window_ids:
        return [], []
    train_ids_raw = split_cfg.get("train_window_ids") or split_cfg.get("train_windows")
    val_ids_raw = split_cfg.get("val_window_ids") or split_cfg.get("val_windows")
    if train_ids_raw is not None or val_ids_raw is not None:
        train_ids = _normalize_int_list(
            train_ids_raw, "window_split.train_window_ids"
        )
        val_ids = _normalize_int_list(val_ids_raw, "window_split.val_window_ids")
        return train_ids, val_ids

    try:
        train_ratio = float(split_cfg.get("train_ratio", split_cfg.get("train", 0.8)))
    except (TypeError, ValueError) as exc:
        raise ConfigError("window_split.train_ratio must be numeric.") from exc
    val_ratio_raw = split_cfg.get("val_ratio", split_cfg.get("val"))
    if val_ratio_raw is None:
        val_ratio = max(0.0, 1.0 - train_ratio)
    else:
        try:
            val_ratio = float(val_ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("window_split.val_ratio must be numeric.") from exc

    if train_ratio < 0.0 or val_ratio < 0.0:
        raise ConfigError("window_split ratios must be non-negative.")
    if train_ratio + val_ratio > 1.0 + 1e-8:
        raise ConfigError("window_split ratios must sum to <= 1.0.")

    seed = split_cfg.get("seed", 0)
    if isinstance(seed, bool):
        raise ConfigError("window_split.seed must be an integer.")
    try:
        seed_value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError("window_split.seed must be an integer.") from exc
    shuffle = bool(split_cfg.get("shuffle", True))

    ids = list(all_window_ids)
    if shuffle:
        rng = random.Random(seed_value)
        rng.shuffle(ids)

    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    if train_count + val_count > len(ids):
        val_count = max(0, len(ids) - train_count)

    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    if not train_ids and ids:
        train_ids = [ids[0]]
    if not val_ids and ids:
        candidates = [window_id for window_id in ids if window_id not in train_ids]
        val_ids = [candidates[-1]] if candidates else [ids[-1]]
    return train_ids, val_ids


def _extract_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = cfg.get("params")
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("params must be a mapping when provided.")
    return dict(params)


def _coerce_optional_float(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number.") from exc


def _coerce_positive_int(value: Any, label: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer.")
    if value <= 0:
        raise ConfigError(f"{label} must be a positive integer.")
    return value


def _load_stable_states_payload(
    value: Any,
) -> tuple[dict[str, Any], Optional[str]]:
    if value is None:
        path = resolve_repo_path(DEFAULT_LEARNCK_STABLE_STATES_PATH)
        payload = read_yaml_payload(path)
        if not isinstance(payload, Mapping):
            raise ConfigError("stable_states config must be a mapping.")
        return dict(payload), str(path)
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        path = resolve_repo_path(value)
        if not path.exists():
            raise ConfigError(f"stable_states file not found: {value}")
        payload = read_yaml_payload(path)
        if not isinstance(payload, Mapping):
            raise ConfigError("stable_states file must contain a mapping.")
        return dict(payload), str(path)
    raise ConfigError("stable_states must be a mapping or path to YAML/JSON.")


def _normalize_phase(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("stable_states.phase must be a string when provided.")
    text = value.strip().lower()
    if not text:
        return None
    if "surf" in text or "surface" in text:
        return "surface"
    if "gas" in text:
        return "gas"
    return text


def _coerce_element_map(raw: Any, label: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping.")
    elements: dict[str, float] = {}
    for element, count in raw.items():
        if element is None:
            continue
        try:
            value = float(count)
        except (TypeError, ValueError):
            raise ConfigError(f"{label} entries must be numeric.")
        if value == 0.0:
            continue
        elements[str(element)] = value
    return elements


def _coerce_site_map(raw: Any, label: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping.")
    sites: dict[str, float] = {}
    for site, count in raw.items():
        if site is None:
            continue
        try:
            value = float(count)
        except (TypeError, ValueError):
            raise ConfigError(f"{label} entries must be numeric.")
        if value == 0.0:
            continue
        sites[str(site)] = value
    return sites


def _parse_formula_elements(name: str) -> dict[str, float]:
    elements: dict[str, float] = {}
    for element, raw_count in re.findall(r"([A-Z][a-z]?)([0-9]*\\.?[0-9]*)", name):
        if not element:
            continue
        if raw_count:
            try:
                count = float(raw_count)
            except ValueError:
                count = 1.0
        else:
            count = 1.0
        elements[element] = elements.get(element, 0.0) + count
    return elements


def _extract_mechanism_species_elements(
    mechanism_payload: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    species_entries = mechanism_payload.get("species")
    if not isinstance(species_entries, Sequence) or isinstance(
        species_entries, (str, bytes, bytearray)
    ):
        return {}
    mapping: dict[str, dict[str, float]] = {}
    for entry in species_entries:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name") or entry.get("species")
        if not isinstance(name, str) or not name.strip():
            continue
        elements = _coerce_element_map(entry.get("composition"), "species.composition")
        if elements:
            mapping[str(name)] = elements
    return mapping


def _normalize_stable_entries(
    payload: Mapping[str, Any],
    mechanism_elements: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    entries_raw: Any = None
    for key in ("stable_species", "stable_states", "states", "species"):
        if key in payload:
            entries_raw = payload.get(key)
            break
    if entries_raw is None:
        raise ConfigError("stable_states must define stable_species.")
    if isinstance(entries_raw, str):
        entries_raw = [entries_raw]
    if not isinstance(entries_raw, Sequence) or isinstance(
        entries_raw,
        (bytes, bytearray),
    ):
        raise ConfigError("stable_species must be a list of entries.")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries_raw):
        if isinstance(entry, str):
            name = _require_nonempty_str(entry, f"stable_species[{index}]")
            elements = dict(mechanism_elements.get(name, {}))
            if not elements:
                elements = _parse_formula_elements(name)
            entries.append(
                {
                    "name": name,
                    "phase": None,
                    "elements": elements,
                    "sites": {},
                }
            )
            continue
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("species") or entry.get("id")
            name = _require_nonempty_str(name, f"stable_species[{index}].name")
            phase = _normalize_phase(entry.get("phase") or entry.get("kind"))
            elements = _coerce_element_map(
                entry.get("elements") or entry.get("composition"),
                f"stable_species[{index}].elements",
            )
            if not elements:
                elements = dict(mechanism_elements.get(name, {}))
            if not elements:
                elements = _parse_formula_elements(name)
            sites = _coerce_site_map(
                entry.get("sites") or entry.get("site"),
                f"stable_species[{index}].sites",
            )
            if not sites and "*" in name:
                sites = {"*": 1.0}
            entries.append(
                {
                    "name": name,
                    "phase": phase,
                    "elements": elements,
                    "sites": sites,
                }
            )
            continue
        raise ConfigError("stable_species entries must be strings or mappings.")
    return entries


def _normalize_overall_entries(
    raw: Any,
    label: str,
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise ConfigError(f"{label} must be a list of entries.")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            name = _require_nonempty_str(entry, f"{label}[{index}]")
            entries.append({"species": name, "coefficient": 1.0})
            continue
        if isinstance(entry, Mapping):
            name = entry.get("species") or entry.get("name") or entry.get("id")
            name = _require_nonempty_str(name, f"{label}[{index}].species")
            coeff = entry.get("coefficient", entry.get("stoich", entry.get("nu", 1.0)))
            try:
                coeff_value = float(coeff)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{label}[{index}].coefficient must be numeric.") from exc
            entries.append({"species": name, "coefficient": coeff_value})
            continue
        raise ConfigError(f"{label} entries must be strings or mappings.")
    return entries


def _build_overall_reaction_template(
    stable_entries: Sequence[Mapping[str, Any]],
    template_cfg: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if template_cfg:
        reactants = _normalize_overall_entries(template_cfg.get("reactants"), "overall_reaction.reactants")
        products = _normalize_overall_entries(template_cfg.get("products"), "overall_reaction.products")
        if not reactants or not products:
            raise ConfigError("overall_reaction requires reactants and products.")
    else:
        reactants = [
            {"species": entry["name"], "coefficient": 1.0} for entry in stable_entries
        ]
        products = [
            {"species": entry["name"], "coefficient": 1.0} for entry in stable_entries
        ]
    return {
        "schema_version": 1,
        "reactants": reactants,
        "products": products,
    }


def _sum_balance(
    entries: Sequence[Mapping[str, Any]],
    species_meta: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in entries:
        name = entry.get("species")
        coeff = float(entry.get("coefficient", 0.0))
        meta = species_meta.get(name) or {}
        values = meta.get(field) or {}
        for key, count in values.items():
            totals[key] = totals.get(key, 0.0) + coeff * float(count)
    return totals


def _check_overall_reaction_conservation(
    template: Mapping[str, Any],
    stable_entries: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    species_meta = {
        entry["name"]: {"elements": entry.get("elements", {}), "sites": entry.get("sites", {})}
        for entry in stable_entries
    }
    reactants = template.get("reactants") or []
    products = template.get("products") or []

    element_data = _sum_balance(products, species_meta, "elements")
    element_react = _sum_balance(reactants, species_meta, "elements")
    element_diff = {
        element: element_data.get(element, 0.0) - element_react.get(element, 0.0)
        for element in set(element_data) | set(element_react)
    }
    elements_missing = not any(species_meta[name]["elements"] for name in species_meta)
    if elements_missing:
        raise ConfigError("overall reaction elements are missing.")
    element_passed = all(abs(value) <= tolerance for value in element_diff.values())

    site_data = _sum_balance(products, species_meta, "sites")
    site_react = _sum_balance(reactants, species_meta, "sites")
    site_diff = {
        site: site_data.get(site, 0.0) - site_react.get(site, 0.0)
        for site in set(site_data) | set(site_react)
    }
    sites_missing = not any(species_meta[name]["sites"] for name in species_meta)
    site_passed = True if sites_missing else all(
        abs(value) <= tolerance for value in site_diff.values()
    )

    return {
        "elements": {
            "passed": element_passed,
            "diff": element_diff,
            "tolerance": tolerance,
        },
        "sites": {
            "passed": site_passed,
            "diff": site_diff,
            "tolerance": tolerance,
            "status": "skipped" if sites_missing else "checked",
        },
    }


def _select_patch_reaction_id(mechanism_payload: Mapping[str, Any]) -> str:
    reactions = mechanism_payload.get("reactions")
    if not isinstance(reactions, Sequence) or isinstance(
        reactions, (str, bytes, bytearray)
    ):
        raise ConfigError("mechanism must define a reactions list.")
    if not reactions:
        raise ConfigError("mechanism has no reactions.")
    identifiers = _reaction_identifiers(reactions[0], 0)
    return identifiers[0]


def _coerce_bool(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean.")
    return value


def _coerce_optional_str(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_str(value, label)


def _load_graph_payload(store: ArtifactStore, graph_id: str) -> dict[str, Any]:
    store.read_manifest("graphs", graph_id)
    graph_path = store.artifact_dir("graphs", graph_id) / "graph.json"
    try:
        payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")
    return dict(payload)


def _load_csr_matrix(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"matrix not found: {path}")
    if sp is not None:
        return sp.load_npz(path)
    if np is None:
        raise ConfigError("numpy is required to load sparse matrices.")
    data = np.load(path)
    if "data" not in data or "indices" not in data or "indptr" not in data:
        raise ConfigError(f"Invalid sparse matrix file: {path}")
    shape = data.get("shape")
    if shape is None:
        raise ConfigError(f"Missing shape in sparse matrix file: {path}")
    return {
        "data": data["data"],
        "indices": data["indices"],
        "indptr": data["indptr"],
        "shape": tuple(int(x) for x in shape),
    }


def _dense_from_sparse(matrix: Any) -> Any:
    if sp is not None and hasattr(matrix, "toarray"):
        return matrix.toarray()
    if np is None:
        raise ConfigError("numpy is required to densify sparse matrices.")
    if isinstance(matrix, np.ndarray):
        return matrix
    if isinstance(matrix, Mapping):
        shape = matrix.get("shape")
        data = matrix.get("data")
        indices = matrix.get("indices")
        indptr = matrix.get("indptr")
        if shape is None or data is None or indices is None or indptr is None:
            raise ConfigError("Invalid sparse matrix payload.")
        rows = shape[0]
        cols = shape[1]
        dense = np.zeros((rows, cols), dtype=float)
        for row in range(rows):
            start = int(indptr[row])
            end = int(indptr[row + 1])
            for idx in range(start, end):
                col = int(indices[idx])
                dense[row, col] = float(data[idx])
        return dense
    raise ConfigError("Unsupported sparse matrix payload.")


def _parse_charge_from_name(name: str) -> Optional[float]:
    if not name:
        return None
    match = re.search(r"([+-])(\d*)$", name)
    if not match:
        return None
    sign = -1.0 if match.group(1) == "-" else 1.0
    magnitude = match.group(2)
    if not magnitude:
        return sign
    try:
        return sign * float(int(magnitude))
    except ValueError:
        return None


def _classify_state(name: str, charge: Optional[float]) -> str:
    if charge is not None and not math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "ion"
    lowered = name.lower()
    if name.endswith(".") or "rad" in lowered:
        return "radical"
    return "neutral"


def _normalize_constraint_fields(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_CNR_CONSTRAINT_FIELDS)
    if isinstance(raw, Mapping):
        raw = raw.get("fields") or raw.get("constraints")
    if isinstance(raw, str):
        return [_require_nonempty_str(raw, "constraints")]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        fields: list[str] = []
        for entry in raw:
            fields.append(_require_nonempty_str(entry, "constraints"))
        return fields
    raise ConfigError("constraints must be a string or list of strings.")


def _normalize_charge_value(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-6):
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _species_metadata_from_graph(
    payload: Mapping[str, Any],
    species_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    nodes = payload.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes, bytearray)):
        return meta
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if node.get("kind") not in (None, "species"):
            continue
        label = node.get("label") or node.get("species")
        node_id = node.get("id")
        name = None
        if isinstance(label, str) and label.strip():
            name = label.strip()
        elif isinstance(node_id, str) and node_id.startswith("species_"):
            name = node_id.split("species_", 1)[1]
        elif isinstance(node_id, str) and node_id.strip():
            name = node_id.strip()
        if name is None or name not in species_names:
            continue
        meta[name] = {
            "phase": node.get("phase"),
            "charge": node.get("charge"),
            "state": node.get("state"),
            "elements": node.get("elements"),
        }
    return meta


def _species_metadata_from_mechanism(
    mechanism: Optional[str],
    phase: Optional[str],
) -> dict[str, dict[str, Any]]:
    if mechanism is None or ct is None:
        return {}
    try:
        solution = ct.Solution(mechanism, phase) if phase else ct.Solution(mechanism)
    except Exception:
        return {}
    try:
        from rxn_platform.tasks import graphs as graph_tasks
    except Exception:
        return {}
    try:
        annotations = graph_tasks.annotate_species(solution, phase=phase)
    except Exception:
        return {}
    return annotations if isinstance(annotations, Mapping) else {}


def _build_constraint_groups(
    species_names: Sequence[str],
    *,
    constraints: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    phase_default: Optional[str],
) -> tuple[dict[tuple[str, ...], list[int]], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    entries: list[dict[str, Any]] = []

    def _elements_set(meta: Mapping[str, Any]) -> set[str]:
        elements = meta.get("elements")
        if not isinstance(elements, Mapping):
            return set()
        out: set[str] = set()
        for key, value in elements.items():
            if key is None:
                continue
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count == 0.0:
                continue
            out.add(str(key))
        return out

    def _heavy_signature(meta: Mapping[str, Any]) -> str:
        heavy = sorted(elem for elem in _elements_set(meta) if elem != "H")
        return ",".join(heavy) if heavy else "none"

    def _infer_kind(meta: Mapping[str, Any], *, species_name: str) -> str:
        raw = meta.get("kind")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        charge_value = meta.get("charge")
        try:
            charge = float(charge_value)
        except (TypeError, ValueError):
            charge = _parse_charge_from_name(species_name) or 0.0
        if not math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return "ion"
        elements = meta.get("elements")
        if isinstance(elements, Mapping):
            total_atoms = 0.0
            nonzero = 0
            for value in elements.values():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if v == 0.0:
                    continue
                total_atoms += v
                nonzero += 1
            if nonzero == 1 and abs(total_atoms - 1.0) < 1.0e-12:
                return "atom"
        state = meta.get("state")
        if isinstance(state, str) and state.strip().lower() == "radical":
            return "radical"
        return "molecule"

    for idx, name in enumerate(species_names):
        meta = metadata.get(name, {})
        entry: dict[str, Any] = {"species": name}
        key_parts: list[str] = []
        for field in constraints:
            field_lower = field.lower()
            if field_lower == "phase":
                value = meta.get("phase") or phase_default or "unknown"
                value = str(value) if value is not None else "unknown"
            elif field_lower == "charge":
                raw_charge = meta.get("charge")
                if raw_charge is None:
                    raw_charge = _parse_charge_from_name(name)
                try:
                    charge_value = float(raw_charge)
                except (TypeError, ValueError):
                    charge_value = None
                value = _normalize_charge_value(charge_value)
            elif field_lower == "state":
                raw_charge = meta.get("charge")
                try:
                    charge_value = float(raw_charge)
                except (TypeError, ValueError):
                    charge_value = _parse_charge_from_name(name)
                value = meta.get("state") or _classify_state(name, charge_value)
            elif field_lower == "kind":
                value = _infer_kind(meta, species_name=name)
            elif field_lower in {"heavy_elements_signature", "heavy_elements"}:
                value = _heavy_signature(meta)
            else:
                raw_value = meta.get(field)
                value = "unknown" if raw_value is None else str(raw_value)
            key_parts.append(value)
            entry[field] = value
        key = tuple(key_parts)
        groups.setdefault(key, []).append(idx)
        entries.append(entry)
    return groups, entries


def _detect_communities(
    graph: Any,
    *,
    method: str,
    resolution: float,
    seed: Optional[int],
) -> list[list[int]]:
    if nx is None:
        raise ConfigError("networkx is required for community detection.")
    nodes = list(graph.nodes())
    if not nodes:
        return []
    if graph.number_of_edges() == 0:
        return [[node] for node in nodes]
    method = method.lower()
    if method == "louvain":
        louvain = getattr(nx.community, "louvain_communities", None)
        if louvain is not None:
            communities = louvain(graph, weight="weight", seed=seed, resolution=resolution)
        else:
            method = "greedy"
    if method == "greedy":
        communities = nx.algorithms.community.greedy_modularity_communities(
            graph, weight="weight"
        )
    elif method not in {"louvain"}:
        raise ConfigError("community.method must be 'louvain' or 'greedy'.")
    return [sorted(list(group)) for group in communities]


def _cluster_size_stats(sizes: Sequence[int]) -> dict[str, Any]:
    if not sizes:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
    mean = float(statistics.mean(sizes))
    median = float(statistics.median(sizes))
    stdev = float(statistics.pstdev(sizes)) if len(sizes) > 1 else 0.0
    return {
        "count": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": mean,
        "median": median,
        "stdev": stdev,
    }


def _load_gnn_dataset_payload(
    store: ArtifactStore,
    dataset_id: str,
) -> tuple[dict[str, Any], Path]:
    store.read_manifest("gnn_datasets", dataset_id)
    dataset_dir = store.artifact_dir("gnn_datasets", dataset_id)
    payload_path = dataset_dir / "dataset.json"
    if not payload_path.exists():
        raise ConfigError(f"gnn dataset metadata not found: {payload_path}")
    try:
        payload = read_json(payload_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"gnn dataset dataset.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("gnn dataset dataset.json must be a JSON object.")
    return dict(payload), dataset_dir


def _load_gnn_dataset_items(
    payload: Mapping[str, Any],
    *,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        files = {}
    dataset_root = payload.get("dataset_root")
    if isinstance(dataset_root, str) and dataset_root:
        data_root = Path(dataset_root)
    else:
        data_root = dataset_dir
    splits: dict[str, list[int]] = {}
    items: list[dict[str, Any]] = []
    data_json = files.get("data_json")
    data_pt = files.get("data_pt")
    if isinstance(data_json, str) and data_json:
        data_path = data_root / data_json
        if not data_path.exists():
            raise ConfigError(f"gnn dataset data not found: {data_path}")
        try:
            payload_data = read_json(data_path)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"gnn dataset data.json is not valid JSON: {exc}") from exc
        if not isinstance(payload_data, Mapping):
            raise ConfigError("gnn dataset data.json must be a JSON object.")
        raw_items = payload_data.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(
            raw_items, (str, bytes, bytearray)
        ):
            raise ConfigError("gnn dataset data.json items must be a list.")
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            window_id = entry.get("window_id")
            features = entry.get("x")
            if window_id is None or features is None:
                continue
            items.append(
                {
                    "window_id": int(window_id),
                    "features": features,
                    "window": entry.get("window", {}),
                }
            )
        raw_splits = payload_data.get("splits")
        if isinstance(raw_splits, Mapping):
            for key, value in raw_splits.items():
                if isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes, bytearray)
                ):
                    splits[str(key)] = [int(v) for v in value]
    elif isinstance(data_pt, str) and data_pt:
        data_path = data_root / data_pt
        if not data_path.exists():
            raise ConfigError(f"gnn dataset data not found: {data_path}")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "torch is required to load pyg datasets; "
                "rerun gnn_dataset with json output or install rxn-platform[gnn]."
            ) from exc
        try:
            data_payload = torch.load(data_path, weights_only=False)
        except TypeError:
            data_payload = torch.load(data_path)
        if not isinstance(data_payload, Mapping):
            raise ConfigError("gnn dataset data.pt must contain a mapping.")
        raw_items = data_payload.get("data_list")
        if not isinstance(raw_items, Sequence):
            raise ConfigError("gnn dataset data.pt missing data_list.")
        for data in raw_items:
            window_id = getattr(data, "window_id", None)
            features = getattr(data, "x", None)
            if window_id is None or features is None:
                continue
            try:
                features_np = features.detach().cpu().numpy().tolist()
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ConfigError("gnn dataset features could not be converted.") from exc
            items.append(
                {
                    "window_id": int(window_id),
                    "features": features_np,
                    "window": {
                        "start_time": getattr(data, "window_start", None),
                        "end_time": getattr(data, "window_end", None),
                        "start_idx": getattr(data, "window_index", None),
                    },
                }
            )
        raw_splits = data_payload.get("splits")
        if isinstance(raw_splits, Mapping):
            for key, value in raw_splits.items():
                if isinstance(value, Sequence):
                    splits[str(key)] = [int(v) for v in value]
    else:
        raise ConfigError("gnn dataset does not include data_json or data_pt entries.")

    if not items:
        raise ConfigError("gnn dataset items are empty.")
    if not splits and isinstance(payload.get("splits"), Mapping):
        for key, value in payload.get("splits", {}).items():
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                splits[str(key)] = [int(v) for v in value]
    return items, splits


def _normalize_k_sweep(value: Any, *, total_nodes: int) -> list[int]:
    if total_nodes <= 0:
        raise ConfigError("total_nodes must be positive for k_sweep.")
    if value is None:
        ratios = (0.1, 0.2, 0.3)
        ks = sorted({max(2, int(math.ceil(total_nodes * r))) for r in ratios})
        return [min(total_nodes, k) for k in ks]
    if isinstance(value, bool):
        raise ConfigError("k_sweep must be an integer or list of integers.")
    if isinstance(value, int):
        if value <= 0:
            raise ConfigError("k_sweep must be positive.")
        return [min(total_nodes, value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ks: list[int] = []
        for idx, entry in enumerate(value):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ConfigError(f"k_sweep[{idx}] must be an integer.")
            if entry <= 0:
                raise ConfigError(f"k_sweep[{idx}] must be positive.")
            ks.append(min(total_nodes, entry))
        return sorted(set(ks))
    raise ConfigError("k_sweep must be an integer or list of integers.")


def _allocate_group_clusters(
    *,
    total_k: int,
    group_sizes: Sequence[int],
) -> list[int]:
    if total_k <= 0:
        raise ConfigError("total_k must be positive.")
    if not group_sizes:
        return []
    total_nodes = sum(group_sizes)
    total_k = min(total_k, total_nodes)
    group_count = len(group_sizes)
    if total_k < group_count:
        total_k = group_count
    counts = [1 for _ in group_sizes]
    remaining = total_k - group_count
    if remaining <= 0:
        return counts
    weighted = [size / float(total_nodes) for size in group_sizes]
    extra = [int(round(remaining * weight)) for weight in weighted]
    counts = [base + add for base, add in zip(counts, extra)]
    while sum(counts) < total_k:
        idx = max(range(len(counts)), key=lambda i: group_sizes[i] - counts[i])
        counts[idx] += 1
    while sum(counts) > total_k:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break
    counts = [min(count, size) for count, size in zip(counts, group_sizes)]
    deficit = total_k - sum(counts)
    while deficit > 0:
        idx = max(range(len(counts)), key=lambda i: group_sizes[i] - counts[i])
        if counts[idx] < group_sizes[idx]:
            counts[idx] += 1
            deficit -= 1
        else:
            break
    return counts


def _temporal_encoding(
    t_value: float,
    *,
    time_dim: int,
    time_base: float,
) -> list[float]:
    if time_dim <= 0:
        return []
    enc: list[float] = []
    for idx in range(time_dim):
        denom = time_base**idx
        angle = 2.0 * math.pi * t_value / float(denom)
        enc.append(math.sin(angle))
        enc.append(math.cos(angle))
    return enc


def _row_normalize(matrix: Any) -> Any:
    if np is None:
        raise ConfigError("numpy is required for normalization.")
    dense = np.asarray(matrix, dtype=float)
    row_sum = dense.sum(axis=1)
    row_sum[row_sum == 0.0] = 1.0
    return dense / row_sum[:, None]


def _kmeans_assign(
    embeddings: Any,
    *,
    k: int,
    seed: Optional[int],
    max_iter: int,
    tol: float = 1.0e-4,
) -> tuple[Any, list[int]]:
    if np is None:
        raise ConfigError("numpy is required for kmeans.")
    data = np.asarray(embeddings, dtype=float)
    n, dim = data.shape
    if k <= 0:
        raise ConfigError("k must be positive for kmeans.")
    if k > n:
        k = n
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centers = data[indices].copy()
    labels = [0 for _ in range(n)]
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        shift = 0.0
        for idx in range(k):
            members = data[new_labels == idx]
            if members.size == 0:
                centers[idx] = data[rng.integers(0, n)]
                continue
            new_center = members.mean(axis=0)
            shift = max(shift, float(np.linalg.norm(new_center - centers[idx])))
            centers[idx] = new_center
        labels = new_labels.tolist()
        if shift < tol:
            break
    return centers, labels


def _soft_assignment(
    embeddings: Any,
    centers: Any,
    *,
    temperature: float,
) -> Any:
    if np is None:
        raise ConfigError("numpy is required for soft assignment.")
    data = np.asarray(embeddings, dtype=float)
    centers = np.asarray(centers, dtype=float)
    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    if temperature <= 0.0:
        hard = distances.argmin(axis=1)
        assign = np.zeros((data.shape[0], centers.shape[0]), dtype=float)
        for idx, label in enumerate(hard):
            assign[idx, int(label)] = 1.0
        return assign
    logits = -distances / float(temperature)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    norm = exp_logits.sum(axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return exp_logits / norm


def _edge_reconstruction_loss(
    adjacency: Any,
    assignment: Any,
) -> float:
    if np is None:
        raise ConfigError("numpy is required for reconstruction loss.")
    a_mat = np.asarray(adjacency, dtype=float)
    s_mat = np.asarray(assignment, dtype=float)
    if a_mat.size == 0:
        return 0.0
    pooled = s_mat.T @ a_mat @ s_mat
    recon = s_mat @ pooled @ s_mat.T
    diff = a_mat - recon
    return float(np.mean(diff * diff))


def _flux_coverage(
    adjacency: Any,
    membership: Sequence[int],
    *,
    symmetrize: bool,
) -> tuple[float, float, float]:
    if np is None:
        raise ConfigError("numpy is required for flux coverage.")
    dense = np.asarray(adjacency, dtype=float)
    if symmetrize:
        dense = 0.5 * (dense + dense.T)
    total = float(np.sum(dense))
    within = 0.0
    n_nodes = len(membership)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if membership[i] != membership[j]:
                continue
            within += float(dense[i, j])
    if not symmetrize:
        for i in range(n_nodes):
            within += float(dense[i, i])
    coverage = within / total if total > 0.0 else 0.0
    return total, within, coverage


def _extract_importance_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("importance", "scoring", "score"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(reduction_cfg)


def _extract_threshold_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("threshold", "thresholds", "prune"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(reduction_cfg)


def _extract_protection_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("protect", "protected", "protection"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return {}


def _extract_optional_artifact_id(
    reduction_cfg: Mapping[str, Any],
    *,
    keys: Sequence[str],
    label: str,
) -> Optional[str]:
    inputs = _extract_inputs(reduction_cfg)
    value: Any = None
    for key in keys:
        if key in inputs:
            value = inputs.get(key)
            break
    if value is None:
        for key in keys:
            if key in reduction_cfg:
                value = reduction_cfg.get(key)
                break
    if value is None:
        return None
    return _require_nonempty_str(value, label)


def _parse_meta_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _extract_reaction_id(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("reaction_id", "reaction", "reaction_name", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = _parse_meta_json(row.get("meta_json"))
    for key in ("reaction_id", "reaction", "reaction_name", "id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_reaction_index(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("reaction_index", "index"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    meta = _parse_meta_json(row.get("meta_json"))
    for key in ("reaction_index", "index"):
        value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_score(value: Any, mode: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if mode == "abs":
        return abs(number)
    return number


def _aggregate_scores(values: Sequence[float], method: str) -> float:
    if not values:
        return math.nan
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if method == "mean":
        return sum(values) / float(len(values))
    if method == "sum":
        return sum(values)
    if method == "median":
        return statistics.median(values)
    raise ConfigError(f"importance.aggregate must be one of: max, min, mean, sum, median.")


def _reaction_key_sort(key: tuple[str, Any]) -> tuple[int, str]:
    kind, value = key
    if kind == "reaction_id":
        return (0, str(value))
    return (1, str(value))


def _extract_reaction_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    column: str,
    mode: str,
    aggregate: str,
) -> tuple[dict[tuple[str, Any], float], dict[tuple[str, Any], dict[str, Any]]]:
    raw_scores: dict[tuple[str, Any], list[float]] = {}
    info: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reaction_id = _extract_reaction_id(row)
        reaction_index = _extract_reaction_index(row)
        if reaction_id is None and reaction_index is None:
            continue
        if reaction_id is not None:
            key = ("reaction_id", reaction_id)
        else:
            key = ("index", reaction_index)
        score = _coerce_score(row.get(column), mode)
        if score is None:
            continue
        raw_scores.setdefault(key, []).append(score)
        if key not in info:
            info[key] = {
                "reaction_id": reaction_id,
                "reaction_index": reaction_index,
            }
    if not raw_scores:
        raise ConfigError("No reaction importance scores found.")
    aggregated: dict[tuple[str, Any], float] = {}
    for key, values in raw_scores.items():
        aggregated[key] = _aggregate_scores(values, aggregate)
    if not aggregated:
        raise ConfigError("No reaction importance scores found.")
    return aggregated, info


def _aggregate_weighted(
    values: Sequence[float],
    weights: Sequence[float],
    method: str,
) -> float:
    if method == "weighted":
        total = sum(weights)
        if total <= 0.0:
            return math.nan
        return sum(v * w for v, w in zip(values, weights)) / total
    return _aggregate_scores(values, method)


def _extract_window_id(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("window_id", "window_index", "time_window"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    meta = _parse_meta_json(row.get("meta_json"))
    for key in ("window_id", "window_index", "time_window"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_species_name(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("species", "species_name", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = _parse_meta_json(row.get("meta_json"))
    for key in ("species", "species_name", "name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_float_list(value: Any, label: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[float] = []
        for idx, entry in enumerate(value):
            if isinstance(entry, bool):
                raise ConfigError(f"{label}[{idx}] must be numeric.")
            try:
                values.append(float(entry))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{label}[{idx}] must be numeric.") from exc
        return values
    raise ConfigError(f"{label} must be numeric or a list of numerics.")


def _select_disabled_keys(
    scores: Mapping[tuple[str, Any], float],
    *,
    top_k: Optional[int],
    score_threshold: Optional[float],
) -> set[tuple[str, Any]]:
    disabled: set[tuple[str, Any]] = set()
    if top_k is not None:
        sorted_items = sorted(
            scores.items(),
            key=lambda item: (-item[1], _reaction_key_sort(item[0])),
        )
        if top_k < len(sorted_items):
            disabled.update(key for key, _ in sorted_items[top_k:])
    if score_threshold is not None:
        disabled.update(
            {key for key, score in scores.items() if score < score_threshold}
        )
    if not disabled:
        raise ConfigError("Threshold prune produced no disabled reactions.")
    return disabled


def _load_importance_rows(
    store: ArtifactStore,
    *,
    kind: str,
    artifact_id: str,
) -> list[dict[str, Any]]:
    if kind == "sensitivity":
        filename = "sensitivity.parquet"
    elif kind == "features":
        filename = "features.parquet"
    else:
        raise ConfigError(f"Unsupported importance source kind: {kind!r}.")
    store.read_manifest(kind, artifact_id)
    table_path = store.artifact_dir(kind, artifact_id) / filename
    return _read_table_rows(table_path)


def _load_graph_reaction_types(
    store: ArtifactStore,
    graph_id: str,
) -> tuple[dict[str, str], dict[int, str]]:
    store.read_manifest("graphs", graph_id)
    graph_path = store.artifact_dir("graphs", graph_id) / "graph.json"
    try:
        payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")
    nodes: Any = None
    bipartite = payload.get("bipartite")
    if isinstance(bipartite, Mapping):
        data = bipartite.get("data")
        if isinstance(data, Mapping):
            nodes = data.get("nodes")
    if not isinstance(nodes, Sequence):
        return {}, {}
    types_by_id: dict[str, str] = {}
    types_by_index: dict[int, str] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if node.get("kind") != "reaction":
            continue
        reaction_id = node.get("reaction_id")
        reaction_index = node.get("reaction_index")
        reaction_type = node.get("reaction_type", node.get("type"))
        if not isinstance(reaction_type, str) or not reaction_type.strip():
            continue
        normalized = reaction_type.strip().lower()
        if isinstance(reaction_id, str) and reaction_id.strip():
            types_by_id[reaction_id.strip()] = normalized
        if isinstance(reaction_index, int) and not isinstance(reaction_index, bool):
            types_by_index[reaction_index] = normalized
    return types_by_id, types_by_index


def _extract_node_lumping_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("node_lumping", "lumping", "node_lump"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(reduction_cfg)


def _extract_reaction_lumping_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("reaction_lumping", "reaction_lump", "lumping"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(reduction_cfg)


def _coerce_similarity_threshold(value: Any) -> float:
    if value is None:
        return DEFAULT_LUMPING_THRESHOLD
    if isinstance(value, bool):
        raise ConfigError("lumping.threshold must be a number.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("lumping.threshold must be a number.") from exc
    if math.isnan(threshold) or math.isinf(threshold):
        raise ConfigError("lumping.threshold must be finite.")
    if threshold < 0.0 or threshold > 1.0:
        raise ConfigError("lumping.threshold must be between 0 and 1.")
    return threshold


def _coerce_similarity_method(value: Any) -> str:
    if value is None:
        return DEFAULT_LUMPING_METHOD
    text = str(value).strip().lower().replace("_", "-")
    if text in {"threshold", "single", "single-link", "single-linkage"}:
        return "threshold"
    if text in {"hierarchical", "agglomerative"}:
        return "hierarchical"
    raise ConfigError("lumping.method must be 'threshold' or 'hierarchical'.")


def _coerce_reaction_similarity_metric(value: Any) -> str:
    if value is None:
        return DEFAULT_REACTION_SIMILARITY_METRIC
    text = str(value).strip().lower().replace("_", "-")
    if text in {"jaccard", "jaccard-index", "jaccardindex"}:
        return "jaccard"
    raise ConfigError("lumping.similarity.metric must be 'jaccard'.")


def _coerce_reaction_similarity_mode(value: Any) -> str:
    if value is None:
        return DEFAULT_REACTION_SIMILARITY_MODE
    text = str(value).strip().lower().replace("_", "-")
    if text in {"reactant", "reactants", "lhs", "inputs"}:
        return "reactants"
    if text in {"product", "products", "rhs", "outputs"}:
        return "products"
    if text in {"both", "separate", "pair", "reactants-products"}:
        return "both"
    if text in {"union", "combined", "all"}:
        return "union"
    raise ConfigError(
        "lumping.similarity.mode must be 'reactants', 'products', 'both', or 'union'."
    )


def _extract_similarity_weights(lumping_cfg: Mapping[str, Any]) -> dict[str, float]:
    weights_cfg = lumping_cfg.get("weights") or lumping_cfg.get("similarity_weights")
    if weights_cfg is None:
        return dict(DEFAULT_LUMPING_WEIGHTS)
    if not isinstance(weights_cfg, Mapping):
        raise ConfigError("lumping.weights must be a mapping.")
    weights: dict[str, float] = {}
    for key, default in DEFAULT_LUMPING_WEIGHTS.items():
        raw = weights_cfg.get(key, default)
        if raw is None:
            weight = float(default)
        else:
            if isinstance(raw, bool):
                raise ConfigError(f"lumping.weights.{key} must be a number.")
            try:
                weight = float(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"lumping.weights.{key} must be a number.") from exc
        if weight < 0.0:
            raise ConfigError(f"lumping.weights.{key} must be >= 0.")
        weights[key] = weight
    if sum(weights.values()) <= 0.0:
        raise ConfigError("lumping.weights must include a positive value.")
    return weights


def _extract_reaction_similarity_weights(
    similarity_cfg: Mapping[str, Any],
) -> dict[str, float]:
    weights_cfg = similarity_cfg.get("weights") or similarity_cfg.get("role_weights")
    if weights_cfg is None:
        return dict(DEFAULT_REACTION_SIMILARITY_WEIGHTS)
    if not isinstance(weights_cfg, Mapping):
        raise ConfigError("lumping.similarity.weights must be a mapping.")
    weights = dict(DEFAULT_REACTION_SIMILARITY_WEIGHTS)
    for key, raw in weights_cfg.items():
        if key in ("reactant", "reactants"):
            norm_key = "reactants"
        elif key in ("product", "products"):
            norm_key = "products"
        else:
            raise ConfigError(
                "lumping.similarity.weights keys must be reactants/products."
            )
        if raw is None:
            value = float(weights[norm_key])
        else:
            if isinstance(raw, bool):
                raise ConfigError(
                    f"lumping.similarity.weights.{norm_key} must be a number."
                )
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"lumping.similarity.weights.{norm_key} must be a number."
                ) from exc
        if value < 0.0:
            raise ConfigError(
                f"lumping.similarity.weights.{norm_key} must be >= 0."
            )
        weights[norm_key] = value
    if sum(weights.values()) <= 0.0:
        raise ConfigError("lumping.similarity.weights must include a positive value.")
    return weights


def _coerce_reaction_include_participants(similarity_cfg: Mapping[str, Any]) -> bool:
    raw = similarity_cfg.get("include_participants")
    if raw is None:
        raw = similarity_cfg.get("include_unknown_role")
    if raw is None:
        return DEFAULT_REACTION_INCLUDE_PARTICIPANTS
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    raise ConfigError(
        "lumping.similarity.include_participants must be a boolean."
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _coerce_charge_scale(lumping_cfg: Mapping[str, Any]) -> float:
    raw = lumping_cfg.get("charge_scale") or lumping_cfg.get("charge_diff_scale")
    if raw is None:
        return DEFAULT_LUMPING_CHARGE_SCALE
    if isinstance(raw, bool):
        raise ConfigError("lumping.charge_scale must be a number.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("lumping.charge_scale must be a number.") from exc
    if value <= 0.0 or math.isnan(value) or math.isinf(value):
        raise ConfigError("lumping.charge_scale must be positive and finite.")
    return value


def _node_ref_to_id(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key in ("id", "name", "key"):
            candidate = value.get(key)
            if candidate is not None:
                return str(candidate)
        return None
    if value is None:
        return None
    return str(value)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def _species_name_from_node(node: Mapping[str, Any]) -> str:
    for key in ("label", "species", "name"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id.startswith("species_"):
        trimmed = node_id[len("species_") :]
        if trimmed:
            return trimmed
    if node_id is None:
        return "unknown"
    return str(node_id)


def _reaction_name_from_node(node: Mapping[str, Any]) -> str:
    for key in ("reaction_id", "reaction", "label", "name", "reaction_equation"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id.startswith("reaction_"):
        trimmed = node_id[len("reaction_") :]
        if trimmed:
            return trimmed
    if node_id is None:
        return "unknown"
    return str(node_id)


def _extract_elements_map(node: Mapping[str, Any]) -> dict[str, float]:
    elements: dict[str, float] = {}
    raw = node.get("elements")
    if isinstance(raw, Mapping):
        for element, count in raw.items():
            if element is None:
                continue
            try:
                value = float(count)
            except (TypeError, ValueError):
                continue
            if value == 0.0:
                continue
            elements[str(element)] = value
        return elements
    vector = node.get("elements_vector") or node.get("elementsVector")
    if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes, bytearray)):
        for entry in vector:
            element = None
            count = None
            if isinstance(entry, Mapping):
                element = entry.get("element") or entry.get("name")
                count = entry.get("count") or entry.get("value")
            elif isinstance(entry, Sequence) and not isinstance(
                entry, (str, bytes, bytearray)
            ):
                if len(entry) >= 2:
                    element = entry[0]
                    count = entry[1]
            if element is None:
                continue
            try:
                value = float(count)
            except (TypeError, ValueError):
                continue
            if value == 0.0:
                continue
            elements[str(element)] = value
    return elements


def _extract_charge_value(node: Mapping[str, Any]) -> Optional[float]:
    raw = node.get("charge")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _load_graph_nodes_and_links(
    store: ArtifactStore,
    graph_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store.read_manifest("graphs", graph_id)
    graph_path = store.artifact_dir("graphs", graph_id) / "graph.json"
    try:
        payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")
    data: Optional[Mapping[str, Any]] = None
    bipartite = payload.get("bipartite")
    if isinstance(bipartite, Mapping):
        candidate = bipartite.get("data")
        if isinstance(candidate, Mapping):
            data = candidate
    if data is None and "nodes" in payload:
        data = payload
    if data is None:
        raise ConfigError("graph.json has no node-link data.")
    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    links_raw = data.get("links") or data.get("edges") or []
    if not isinstance(links_raw, Sequence) or isinstance(
        links_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph links must be a sequence.")
    nodes: list[dict[str, Any]] = []
    for idx, entry in enumerate(nodes_raw):
        if isinstance(entry, Mapping):
            node = dict(entry)
        elif isinstance(entry, str):
            node = {"id": entry}
        else:
            continue
        if node.get("id") is None:
            node["id"] = f"node_{idx}"
        nodes.append(node)
    links: list[dict[str, Any]] = []
    for entry in links_raw:
        if isinstance(entry, Mapping):
            links.append(dict(entry))
    return nodes, links


def _species_degree_map(
    links: Sequence[Mapping[str, Any]],
    species_ids: set[str],
) -> dict[str, int]:
    degree = {node_id: 0 for node_id in species_ids}
    for link in links:
        source = _node_ref_to_id(link.get("source"))
        target = _node_ref_to_id(link.get("target"))
        if source in degree:
            degree[source] += 1
        if target in degree:
            degree[target] += 1
    return degree


def _prepare_species_entries(
    nodes: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("kind") != "species":
            continue
        node_id = _node_ref_to_id(node.get("id"))
        if node_id is None:
            continue
        species_index = node.get("species_index")
        if not isinstance(species_index, int) or isinstance(species_index, bool):
            species_index = None
        entries.append(
            {
                "node_id": node_id,
                "species": _species_name_from_node(node),
                "species_index": species_index,
                "elements": _extract_elements_map(node),
                "charge": _extract_charge_value(node),
                "phase": _normalize_text(node.get("phase")),
                "state": _normalize_text(node.get("state")),
            }
        )
    if not entries:
        raise ConfigError("graph has no species nodes.")
    species_ids = {entry["node_id"] for entry in entries}
    degree = _species_degree_map(links, species_ids)
    for entry in entries:
        entry["degree"] = degree.get(entry["node_id"], 0)
    return entries


def _reaction_role_from_link(link: Mapping[str, Any]) -> Optional[str]:
    raw = link.get("role") or link.get("reaction_role") or link.get("side")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"reactant", "reactants", "lhs", "input", "inputs", "source"}:
            return "reactant"
        if text in {"product", "products", "rhs", "output", "outputs", "sink"}:
            return "product"
    stoich = link.get("stoich") or link.get("stoichiometry")
    if stoich is None or isinstance(stoich, bool):
        return None
    try:
        value = float(stoich)
    except (TypeError, ValueError):
        return None
    if value < 0.0:
        return "reactant"
    if value > 0.0:
        return "product"
    return None


def _prepare_reaction_entries(
    nodes: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    node_kinds: dict[str, str] = {}
    species_names: dict[str, str] = {}
    for node in nodes:
        node_id = _node_ref_to_id(node.get("id"))
        if node_id is None:
            continue
        kind = node.get("kind")
        kind_text = str(kind).strip().lower() if kind is not None else ""
        if kind_text:
            node_kinds[node_id] = kind_text
        if kind_text == "species":
            species_names[node_id] = _species_name_from_node(node)
        if kind_text != "reaction":
            continue
        reaction_type = _normalize_text(
            node.get("reaction_type") or node.get("type")
        ) or "unknown"
        reaction_index = node.get("reaction_index")
        if not isinstance(reaction_index, int) or isinstance(reaction_index, bool):
            reaction_index = None
        reaction_equation = node.get("reaction_equation")
        if reaction_equation is not None and not isinstance(reaction_equation, str):
            reaction_equation = str(reaction_equation)
        entries[node_id] = {
            "node_id": node_id,
            "reaction_id": _reaction_name_from_node(node),
            "reaction_type": reaction_type,
            "reaction_index": reaction_index,
            "reaction_equation": reaction_equation,
            "reactants": set(),
            "products": set(),
            "participants": set(),
            "degree": 0,
        }

    if not entries:
        raise ConfigError("graph has no reaction nodes.")

    for link in links:
        source_id = _node_ref_to_id(link.get("source"))
        target_id = _node_ref_to_id(link.get("target"))
        if source_id is None or target_id is None:
            continue
        source_kind = node_kinds.get(source_id)
        target_kind = node_kinds.get(target_id)
        reaction_node_id: Optional[str] = None
        species_node_id: Optional[str] = None
        if source_kind == "reaction" and target_kind == "species":
            reaction_node_id = source_id
            species_node_id = target_id
        elif source_kind == "species" and target_kind == "reaction":
            reaction_node_id = target_id
            species_node_id = source_id
        if reaction_node_id is None or species_node_id is None:
            continue
        entry = entries.get(reaction_node_id)
        if entry is None:
            continue
        species_name = species_names.get(species_node_id) or species_node_id
        role = _reaction_role_from_link(link)
        if role == "reactant":
            entry["reactants"].add(species_name)
        elif role == "product":
            entry["products"].add(species_name)
        else:
            entry["participants"].add(species_name)
        entry["degree"] += 1

    return list(entries.values())


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _apply_protection(
    disabled_keys: set[tuple[str, Any]],
    *,
    reaction_info: Mapping[tuple[str, Any], Mapping[str, Any]],
    protected_ids: Sequence[str],
    protected_indices: Sequence[int],
    protected_types: Sequence[str],
    types_by_id: Mapping[str, str],
    types_by_index: Mapping[int, str],
) -> list[tuple[str, Any]]:
    protected_keys: set[tuple[str, Any]] = set()
    for reaction_id in protected_ids:
        protected_keys.add(("reaction_id", reaction_id))
    for reaction_index in protected_indices:
        protected_keys.add(("index", reaction_index))

    filtered: list[tuple[str, Any]] = []
    for key in disabled_keys:
        if key in protected_keys:
            continue
        if protected_types:
            info = reaction_info.get(key, {})
            reaction_id = info.get("reaction_id")
            reaction_index = info.get("reaction_index")
            reaction_type = None
            if isinstance(reaction_id, str):
                reaction_type = types_by_id.get(reaction_id)
            if reaction_type is None and isinstance(reaction_index, int):
                reaction_type = types_by_index.get(reaction_index)
            if reaction_type in protected_types:
                continue
        filtered.append(key)
    return filtered


def _build_disabled_entries(
    keys: Sequence[tuple[str, Any]],
    info: Mapping[tuple[str, Any], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in sorted(keys, key=_reaction_key_sort):
        kind, value = key
        detail = info.get(key, {})
        reaction_index = detail.get("reaction_index")
        reaction_id = detail.get("reaction_id")
        # Prefer reaction indices whenever available. Equation strings are not
        # guaranteed unique in real mechanisms and break multiplier application.
        if reaction_index is not None and not isinstance(reaction_index, bool):
            entries.append({"index": int(reaction_index)})
            continue
        if isinstance(reaction_id, str) and reaction_id.strip():
            entries.append({"reaction_id": reaction_id})
            continue
        if kind == "reaction_id":
            reaction_id = detail.get("reaction_id") or value
            entries.append({"reaction_id": reaction_id})
            continue
        if reaction_index is not None and not isinstance(reaction_index, bool):
            entries.append({"index": int(reaction_index)})
            continue
        reaction_index = value
        entries.append({"index": reaction_index})
    return entries


def _extract_reduction_cfg(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ("reduction", "reduce"):
        if key in cfg and isinstance(cfg.get(key), Mapping):
            reduction_cfg = cfg.get(key)
            if not isinstance(reduction_cfg, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(cfg), dict(reduction_cfg)
    return dict(cfg), dict(cfg)


def _resolve_reducer_task_name(
    method: str,
    registry: Optional[Registry],
) -> str:
    method_key = method.strip()
    method_norm = method_key.lower()
    task_name = REDUCTION_METHOD_TASKS.get(method_norm)
    if task_name is None and method_key.startswith("reduction."):
        task_name = method_key
    if task_name is None:
        available = sorted(REDUCTION_METHOD_TASKS.keys())
        raise ConfigError(
            "reduction.method is not registered. Available: " + ", ".join(available)
        )
    return task_name


def _extract_inputs(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    inputs = reduction_cfg.get("inputs")
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("reduction.inputs must be a mapping.")
    return dict(inputs)


def _coerce_path(value: Any, label: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _extract_mechanism(reduction_cfg: Mapping[str, Any]) -> str:
    inputs = _extract_inputs(reduction_cfg)
    mechanism: Any = None
    for key in ("mechanism", "mechanism_path", "solution"):
        if key in inputs:
            mechanism = inputs.get(key)
            break
    if mechanism is None:
        for key in ("mechanism", "mechanism_path", "solution"):
            if key in reduction_cfg:
                mechanism = reduction_cfg.get(key)
                break
    mech_value = _coerce_path(mechanism, "reduction.mechanism")
    mech_path = Path(mech_value)
    if mech_path.is_absolute() or len(mech_path.parts) > 1:
        if not mech_path.exists():
            raise ConfigError(f"mechanism file not found: {mech_value}")
    return mech_value


def _extract_patch_source(reduction_cfg: Mapping[str, Any]) -> Any:
    inputs = _extract_inputs(reduction_cfg)
    patch_value: Any = None
    for key in ("patch", "patches", "mechanism_patch", "patch_file"):
        if key in inputs:
            patch_value = inputs.get(key)
            break
    if patch_value is None:
        for key in ("patch", "patches", "mechanism_patch", "patch_file"):
            if key in reduction_cfg:
                patch_value = reduction_cfg.get(key)
                break
    if patch_value is None:
        if "disabled_reactions" in reduction_cfg or "reaction_multipliers" in reduction_cfg:
            patch_value = reduction_cfg
        else:
            raise ConfigError("reduction patch is required.")
    return patch_value


def _load_patch_payload(patch_value: Any) -> tuple[dict[str, Any], Optional[str]]:
    if isinstance(patch_value, Mapping):
        return dict(patch_value), None
    if isinstance(patch_value, Path):
        patch_value = str(patch_value)
    if isinstance(patch_value, str):
        patch_path = Path(patch_value)
        if not patch_path.exists():
            raise ConfigError(f"patch file not found: {patch_value}")
        payload = read_yaml_payload(patch_path)
        if not isinstance(payload, Mapping):
            raise ConfigError("patch file must contain a mapping.")
        return dict(payload), str(patch_path)
    raise ConfigError("patch must be a mapping or path to YAML/JSON.")


def _normalize_patch_payload(
    patch_data: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema_version = patch_data.get("schema_version", PATCH_SCHEMA_VERSION)
    if schema_version is None:
        schema_version = PATCH_SCHEMA_VERSION
    if isinstance(schema_version, bool):
        raise ConfigError("patch.schema_version must be an integer.")
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ConfigError("patch.schema_version must be an integer.") from exc
    if schema_version < 1:
        raise ConfigError("patch.schema_version must be >= 1.")

    disabled_raw = patch_data.get("disabled_reactions")
    multipliers_raw = patch_data.get("reaction_multipliers")
    try:
        disabled_entries = normalize_reaction_multipliers(
            {"disabled_reactions": disabled_raw}
        )
        multiplier_entries = normalize_reaction_multipliers(
            {"reaction_multipliers": multipliers_raw}
        )
        combined_entries = normalize_reaction_multipliers(
            {
                "disabled_reactions": disabled_raw,
                "reaction_multipliers": multipliers_raw,
            }
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"patch entries are invalid: {exc}") from exc

    if not combined_entries:
        raise ConfigError("patch must include disabled_reactions or reaction_multipliers.")

    disabled_payload: list[dict[str, Any]] = []
    for entry in disabled_entries:
        if "index" in entry:
            disabled_payload.append({"index": entry["index"]})
        else:
            disabled_payload.append({"reaction_id": entry["reaction_id"]})

    normalized_patch = {
        "schema_version": schema_version,
        "disabled_reactions": disabled_payload,
        "reaction_multipliers": list(multiplier_entries),
    }
    return normalized_patch, combined_entries


def _reaction_identifiers(reaction: Any, index: int) -> list[str]:
    return _reaction_identifiers_shared(reaction, index)


def _reaction_id_index_map(
    reactions: Sequence[Any],
) -> dict[str, list[int]]:
    return _reaction_id_index_map_shared(reactions)


def _resolve_patch_entries(
    entries: Sequence[Mapping[str, Any]],
    reactions: Sequence[Any],
) -> dict[int, dict[str, Any]]:
    return _resolve_patch_entries_shared(entries, reactions)


def _apply_patch_entries(
    mechanism: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], set[int]]:
    return _apply_patch_entries_shared(mechanism, entries)


def threshold_prune(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Generate a reduction patch by pruning reactions below importance thresholds."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)

    sensitivity_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("sensitivity", "sensitivity_id", "sensitivity_artifact"),
        label="reduction.sensitivity",
    )
    features_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("features", "feature", "features_id", "feature_id"),
        label="reduction.features",
    )
    if sensitivity_id and features_id:
        raise ConfigError("Specify only one of sensitivity or features as input.")
    if not sensitivity_id and not features_id:
        raise ConfigError("reduction threshold prune requires sensitivity or features input.")

    source_kind = "sensitivity" if sensitivity_id else "features"
    source_id = sensitivity_id or features_id

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )

    importance_cfg = _extract_importance_cfg(reduction_cfg)
    importance_column = _require_nonempty_str(
        importance_cfg.get("column")
        or importance_cfg.get("field")
        or DEFAULT_IMPORTANCE_COLUMN,
        "importance.column",
    )
    importance_mode = str(importance_cfg.get("mode", DEFAULT_IMPORTANCE_MODE)).lower()
    if importance_mode in ("abs", "absolute"):
        importance_mode = "abs"
    elif importance_mode in ("raw", "value"):
        importance_mode = "raw"
    else:
        raise ConfigError("importance.mode must be 'abs' or 'raw'.")

    importance_aggregate = str(
        importance_cfg.get("aggregate")
        or importance_cfg.get("aggregation")
        or DEFAULT_IMPORTANCE_AGGREGATE
    ).lower()
    if importance_aggregate not in {"max", "min", "mean", "sum", "median"}:
        raise ConfigError(
            "importance.aggregate must be one of: max, min, mean, sum, median."
        )

    threshold_cfg = _extract_threshold_cfg(reduction_cfg)
    top_k = threshold_cfg.get(
        "top_k",
        threshold_cfg.get("topK", threshold_cfg.get("keep_top_k")),
    )
    if top_k is not None:
        if isinstance(top_k, bool):
            raise ConfigError("threshold.top_k must be an integer.")
        try:
            top_k = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ConfigError("threshold.top_k must be an integer.") from exc
        if top_k <= 0:
            raise ConfigError("threshold.top_k must be a positive integer.")

    score_threshold = threshold_cfg.get("score_lt")
    if score_threshold is None:
        score_threshold = threshold_cfg.get(
            "score_threshold", threshold_cfg.get("min_score")
        )
    if score_threshold is None and "threshold" in threshold_cfg:
        threshold_value = threshold_cfg.get("threshold")
        if not isinstance(threshold_value, Mapping):
            score_threshold = threshold_value
    if score_threshold is not None:
        if isinstance(score_threshold, bool):
            raise ConfigError("threshold.score_lt must be a number.")
        try:
            score_threshold = float(score_threshold)
        except (TypeError, ValueError) as exc:
            raise ConfigError("threshold.score_lt must be a number.") from exc
        if math.isnan(score_threshold) or math.isinf(score_threshold):
            raise ConfigError("threshold.score_lt must be a finite number.")

    if top_k is None and score_threshold is None:
        raise ConfigError("threshold prune requires top_k or score_lt.")

    protection_cfg = _extract_protection_cfg(reduction_cfg)
    protected_types = [
        value.lower()
        for value in _normalize_str_list(
            protection_cfg.get("reaction_types")
            or protection_cfg.get("reaction_type")
            or protection_cfg.get("types"),
            "protect.reaction_types",
        )
    ]
    protected_ids = _normalize_str_list(
        protection_cfg.get("reaction_ids") or protection_cfg.get("reaction_id"),
        "protect.reaction_ids",
    )
    protected_indices = _normalize_int_list(
        protection_cfg.get("reaction_indices") or protection_cfg.get("reaction_index"),
        "protect.reaction_indices",
    )

    types_by_id: dict[str, str] = {}
    types_by_index: dict[int, str] = {}
    if graph_id is not None:
        types_by_id, types_by_index = _load_graph_reaction_types(store, graph_id)

    type_map_cfg = protection_cfg.get("reaction_type_map") or protection_cfg.get(
        "reaction_types_by_id"
    )
    if type_map_cfg is not None:
        if not isinstance(type_map_cfg, Mapping):
            raise ConfigError("protect.reaction_type_map must be a mapping.")
        for key, value in type_map_cfg.items():
            if not isinstance(key, str) or not key.strip():
                raise ConfigError("protect.reaction_type_map keys must be strings.")
            if not isinstance(value, str) or not value.strip():
                raise ConfigError("protect.reaction_type_map values must be strings.")
            types_by_id[key.strip()] = value.strip().lower()

    if protected_types and not (types_by_id or types_by_index):
        raise ConfigError("protect.reaction_types requires graph or type map input.")

    rows = _load_importance_rows(store, kind=source_kind, artifact_id=source_id)
    scores, reaction_info = _extract_reaction_scores(
        rows,
        column=importance_column,
        mode=importance_mode,
        aggregate=importance_aggregate,
    )
    disabled_keys = _select_disabled_keys(
        scores,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    filtered_keys = _apply_protection(
        disabled_keys,
        reaction_info=reaction_info,
        protected_ids=protected_ids,
        protected_indices=protected_indices,
        protected_types=protected_types,
        types_by_id=types_by_id,
        types_by_index=types_by_index,
    )
    if not filtered_keys:
        raise ConfigError("Threshold prune produced no disabled reactions.")

    disabled_entries = _build_disabled_entries(filtered_keys, reaction_info)
    patch_payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "disabled_reactions": disabled_entries,
        "reaction_multipliers": [],
    }

    inputs_payload: dict[str, Any] = {source_kind: source_id}
    if graph_id is not None:
        inputs_payload["graph"] = graph_id
    inputs_payload["importance"] = {
        "column": importance_column,
        "mode": importance_mode,
        "aggregate": importance_aggregate,
    }
    threshold_payload: dict[str, Any] = {}
    if top_k is not None:
        threshold_payload["top_k"] = top_k
    if score_threshold is not None:
        threshold_payload["score_lt"] = score_threshold
    if threshold_payload:
        inputs_payload["threshold"] = threshold_payload
    protect_payload: dict[str, Any] = {}
    if protected_types:
        protect_payload["reaction_types"] = protected_types
    if protected_ids:
        protect_payload["reaction_ids"] = protected_ids
    if protected_indices:
        protect_payload["reaction_indices"] = protected_indices
    if protect_payload:
        inputs_payload["protect"] = protect_payload

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = [source_id]
    if graph_id is not None:
        parents.append(graph_id)
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(
            base_dir / PATCH_FILENAME,
            patch_payload,
            sort_keys=True,
        )

        disabled_count = 0
        disabled_raw = patch_payload.get("disabled_reactions") or []
        if isinstance(disabled_raw, dict):
            disabled_count = len(disabled_raw)
        elif isinstance(disabled_raw, list):
            disabled_count = len(disabled_raw)

        species_before = None
        reactions_before = None
        if graph_id is not None:
            try:
                graph_payload = _load_graph_payload(store, graph_id)
            except Exception:
                graph_payload = None
            if isinstance(graph_payload, dict):
                species_section = graph_payload.get("species")
                reactions_section = graph_payload.get("reactions")
                if isinstance(species_section, dict):
                    species_before = _coerce_optional_int(species_section.get("count"))
                elif isinstance(species_section, list):
                    species_before = len(species_section)
                if isinstance(reactions_section, dict):
                    reactions_before = _coerce_optional_int(reactions_section.get("count"))
                elif isinstance(reactions_section, list):
                    reactions_before = len(reactions_section)

        reactions_after = (
            (reactions_before - disabled_count)
            if isinstance(reactions_before, int)
            else None
        )
        metrics = {
            "schema_version": 1,
            "kind": "threshold_prune_metrics",
            "graph_id": graph_id,
            "counts": {
                "species_before": species_before,
                "species_after": species_before,
                "merged_species": 0,
                "reactions_before": reactions_before,
                "reactions_after": reactions_after,
                "disabled_reactions": disabled_count,
                "merged_reactions": 0,
            },
            "selection": {
                "threshold": threshold_payload,
                "importance": {
                    "column": importance_column,
                    "mode": importance_mode,
                    "aggregate": importance_aggregate,
                },
            },
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def dispatch(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Dispatch reduction methods through a shared interface."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    _, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    method = (
        reduction_cfg.get("method")
        or reduction_cfg.get("reducer")
        or reduction_cfg.get("reduction_method")
        or params.get("method")
        or params.get("reducer")
    )
    if not isinstance(method, str) or not method.strip():
        raise ConfigError("reduction.method must be a non-empty string.")
    task_name = _resolve_reducer_task_name(method, registry)
    if task_name == "reduction.dispatch":
        raise ConfigError("reduction.method must target a concrete reduction task.")
    if registry is None:
        task = registry_module.get("task", task_name)
    else:
        task = registry.get("task", task_name)
    return task(cfg=cfg, store=store, registry=registry)


def gnn_importance_prune(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Prune mechanisms using GNN importance scores with QoI-guided threshold search."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    importance_cfg = _extract_importance_cfg(reduction_cfg)
    threshold_cfg = _extract_threshold_cfg(reduction_cfg)

    mechanism_path = _extract_mechanism(reduction_cfg)
    importance_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=(
            "importance",
            "importance_id",
            "features",
            "features_id",
            "feature_id",
        ),
        label="reduction.importance",
    )
    if importance_id is None:
        raise ConfigError("gnn_importance_prune requires importance features input.")

    store.read_manifest("features", importance_id)
    features_dir = store.artifact_dir("features", importance_id)
    rows = _read_table_rows(features_dir / "features.parquet")

    reaction_feature = (
        importance_cfg.get("reaction_feature")
        or importance_cfg.get("reaction")
        or "gnn_reaction_importance"
    )
    species_feature = (
        importance_cfg.get("species_feature")
        or importance_cfg.get("species")
        or "gnn_species_importance"
    )
    if not isinstance(reaction_feature, str) or not reaction_feature.strip():
        raise ConfigError("importance.reaction_feature must be a non-empty string.")
    if not isinstance(species_feature, str) or not species_feature.strip():
        raise ConfigError("importance.species_feature must be a non-empty string.")

    importance_column = _require_nonempty_str(
        importance_cfg.get("column")
        or importance_cfg.get("field")
        or DEFAULT_IMPORTANCE_COLUMN,
        "importance.column",
    )
    importance_mode = str(importance_cfg.get("mode", DEFAULT_IMPORTANCE_MODE)).lower()
    if importance_mode in ("abs", "absolute"):
        importance_mode = "abs"
    elif importance_mode in ("raw", "value"):
        importance_mode = "raw"
    else:
        raise ConfigError("importance.mode must be 'abs' or 'raw'.")

    importance_aggregate = str(
        importance_cfg.get("aggregate")
        or importance_cfg.get("aggregation")
        or DEFAULT_IMPORTANCE_AGGREGATE
    ).lower()
    if importance_aggregate not in {"max", "min", "mean", "sum", "median", "weighted"}:
        raise ConfigError(
            "importance.aggregate must be one of: max, min, mean, sum, median, weighted."
        )

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    qoi_cfg = params.get("qoi") or importance_cfg.get("qoi") or {}
    if qoi_cfg is None:
        qoi_cfg = {}
    if not isinstance(qoi_cfg, Mapping):
        raise ConfigError("qoi config must be a mapping.")
    qoi_cfg = dict(qoi_cfg)

    window_meta: list[dict[str, Any]] = []
    if graph_id is not None:
        graph_payload = _load_graph_payload(store, graph_id)
        species_graph = graph_payload.get("species_graph")
        if isinstance(species_graph, Mapping):
            layers_meta = species_graph.get("layers")
            if isinstance(layers_meta, Sequence) and not isinstance(
                layers_meta, (str, bytes, bytearray)
            ):
                for entry in layers_meta:
                    if not isinstance(entry, Mapping):
                        continue
                    window_meta.append(
                        {
                            "index": entry.get("index"),
                            "window": entry.get("window", {}),
                        }
                    )

    qoi_window_ids: list[int] = []
    if qoi_cfg:
        if window_meta:
            qoi_window_ids = _select_qoi_windows(qoi_cfg, window_meta=window_meta)
        else:
            raw_ids = qoi_cfg.get("window_ids") or qoi_cfg.get("windows")
            if raw_ids is not None:
                qoi_window_ids = _normalize_int_list(raw_ids, "qoi.window_ids")
    qoi_window_set = set(qoi_window_ids)

    case_weights_cfg = importance_cfg.get("case_weights") or {}
    if case_weights_cfg is None:
        case_weights_cfg = {}
    if not isinstance(case_weights_cfg, Mapping):
        raise ConfigError("importance.case_weights must be a mapping.")
    case_weights: dict[str, float] = {}
    for key, value in case_weights_cfg.items():
        if key is None:
            continue
        try:
            case_weights[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("importance.case_weights values must be numeric.") from exc

    window_weights_cfg = importance_cfg.get("window_weights") or {}
    if window_weights_cfg is None:
        window_weights_cfg = {}
    if not isinstance(window_weights_cfg, Mapping):
        raise ConfigError("importance.window_weights must be a mapping.")
    window_weights: dict[int, float] = {}
    for key, value in window_weights_cfg.items():
        if key is None:
            continue
        try:
            window_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ConfigError("importance.window_weights keys must be integers.") from exc
        try:
            window_weights[window_id] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("importance.window_weights values must be numeric.") from exc

    qoi_weight = _coerce_optional_float(
        importance_cfg.get("qoi_weight") or importance_cfg.get("qoi_focus_weight"),
        "importance.qoi_weight",
    )
    if qoi_weight is None:
        qoi_weight = 1.0
    non_qoi_weight = _coerce_optional_float(
        importance_cfg.get("non_qoi_weight"),
        "importance.non_qoi_weight",
    )
    if non_qoi_weight is None:
        non_qoi_weight = 1.0

    reaction_scores_raw: dict[tuple[str, Any], list[float]] = {}
    reaction_weights: dict[tuple[str, Any], list[float]] = {}
    reaction_info: dict[tuple[str, Any], dict[str, Any]] = {}
    reaction_by_window: dict[tuple[tuple[str, Any], int], list[float]] = {}
    species_scores_raw: dict[str, list[float]] = {}
    species_weights: dict[str, list[float]] = {}

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        feature = row.get("feature")
        if not isinstance(feature, str):
            continue
        score = _coerce_score(row.get(importance_column), importance_mode)
        if score is None:
            continue
        window_id = _extract_window_id(row)
        case_id = row.get("run_id") or row.get("case_id")
        weight = 1.0
        if window_id is not None:
            if window_id in window_weights:
                weight *= window_weights[window_id]
            elif qoi_window_set:
                weight *= qoi_weight if window_id in qoi_window_set else non_qoi_weight
        if isinstance(case_id, str) and case_id in case_weights:
            weight *= case_weights[case_id]

        if feature == reaction_feature:
            reaction_id = _extract_reaction_id(row)
            reaction_index = _extract_reaction_index(row)
            if reaction_id is None and reaction_index is None:
                continue
            if reaction_id is not None:
                key = ("reaction_id", reaction_id)
            else:
                key = ("index", reaction_index)
            reaction_scores_raw.setdefault(key, []).append(score)
            reaction_weights.setdefault(key, []).append(weight)
            if key not in reaction_info:
                reaction_info[key] = {
                    "reaction_id": reaction_id,
                    "reaction_index": reaction_index,
                }
            if window_id is not None:
                reaction_by_window.setdefault((key, window_id), []).append(score)
        elif feature == species_feature:
            species_name = _extract_species_name(row)
            if species_name is None:
                continue
            species_scores_raw.setdefault(species_name, []).append(score)
            species_weights.setdefault(species_name, []).append(weight)

    if not reaction_scores_raw:
        raise ConfigError("No reaction importance scores found for gnn_importance_prune.")

    reaction_scores: dict[tuple[str, Any], float] = {}
    for key, values in reaction_scores_raw.items():
        weights = reaction_weights.get(key, [])
        reaction_scores[key] = _aggregate_weighted(values, weights, importance_aggregate)

    species_scores: dict[str, float] = {}
    for name, values in species_scores_raw.items():
        weights = species_weights.get(name, [])
        value = _aggregate_weighted(values, weights, importance_aggregate)
        if value is None or math.isnan(value):
            value = 0.0
        species_scores[name] = value

    compiler = MechanismCompiler.from_path(mechanism_path)
    reaction_count = compiler.reaction_count()
    id_map = _reaction_id_index_map(compiler.reactions)
    index_scores = {idx: 0.0 for idx in range(reaction_count)}
    for key, score in reaction_scores.items():
        if score is None or math.isnan(score):
            continue
        kind, value = key
        if kind == "index":
            if isinstance(value, int) and 0 <= value < reaction_count:
                index_scores[value] = float(score)
        elif kind == "reaction_id":
            indices = id_map.get(str(value))
            if indices and len(indices) == 1:
                index_scores[indices[0]] = float(score)

    reaction_labels: dict[int, str] = {}
    for idx, reaction in enumerate(compiler.reactions):
        identifiers = _reaction_identifiers(reaction, idx)
        reaction_labels[idx] = identifiers[0] if identifiers else f"R{idx + 1}"

    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        if q <= 0.0:
            return min(values)
        if q >= 1.0:
            return max(values)
        sorted_vals = sorted(values)
        pos = q * (len(sorted_vals) - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return sorted_vals[lower]
        fraction = pos - lower
        return sorted_vals[lower] + fraction * (
            sorted_vals[upper] - sorted_vals[lower]
        )

    def _normalize_candidates() -> list[dict[str, Any]]:
        top_k_values = _normalize_int_list(
            threshold_cfg.get("top_k")
            or threshold_cfg.get("topK")
            or threshold_cfg.get("keep_top_k"),
            "threshold.top_k",
        )
        score_values = _normalize_float_list(
            threshold_cfg.get("score_lt")
            or threshold_cfg.get("score_threshold")
            or threshold_cfg.get("min_score"),
            "threshold.score_lt",
        )
        percentile_values = _normalize_float_list(
            threshold_cfg.get("percentile") or threshold_cfg.get("percentiles"),
            "threshold.percentile",
        )
        keep_ratio = threshold_cfg.get("keep_ratio") or threshold_cfg.get("ratio")
        if keep_ratio is not None:
            if isinstance(keep_ratio, bool):
                raise ConfigError("threshold.keep_ratio must be numeric.")
            try:
                keep_ratio = float(keep_ratio)
            except (TypeError, ValueError) as exc:
                raise ConfigError("threshold.keep_ratio must be numeric.") from exc
            if not 0.0 < keep_ratio <= 1.0:
                raise ConfigError("threshold.keep_ratio must be in (0, 1].")
            top_k_values.append(max(1, int(round(reaction_count * keep_ratio))))

        candidates: list[dict[str, Any]] = []
        for k in top_k_values:
            if k <= 0:
                continue
            candidates.append({"top_k": int(k)})
        for score in score_values:
            candidates.append({"score_lt": float(score)})
        if percentile_values:
            scores_list = list(index_scores.values())
            for percentile in percentile_values:
                if percentile < 0.0 or percentile > 100.0:
                    raise ConfigError("threshold.percentile must be in [0, 100].")
                threshold_value = _quantile(scores_list, percentile / 100.0)
                candidates.append(
                    {
                        "score_lt": float(threshold_value),
                        "percentile": float(percentile),
                    }
                )

        combine = bool(threshold_cfg.get("combine", False))
        if combine and top_k_values and score_values:
            for k in top_k_values:
                for score in score_values:
                    candidates.append({"top_k": int(k), "score_lt": float(score)})

        if not candidates:
            default_keep = max(1, int(round(0.5 * reaction_count)))
            candidates = [{"top_k": default_keep}]
        max_candidates = threshold_cfg.get("max_candidates")
        if max_candidates is not None:
            if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
                raise ConfigError("threshold.max_candidates must be an integer.")
            if max_candidates > 0:
                candidates = candidates[:max_candidates]
        return candidates

    min_keep = threshold_cfg.get("min_keep") or threshold_cfg.get("min_remaining")
    if min_keep is None:
        min_keep = 1
    if isinstance(min_keep, bool):
        raise ConfigError("threshold.min_keep must be an integer.")
    try:
        min_keep = int(min_keep)
    except (TypeError, ValueError) as exc:
        raise ConfigError("threshold.min_keep must be an integer.") from exc
    if min_keep <= 0:
        raise ConfigError("threshold.min_keep must be > 0.")
    min_keep = min(min_keep, reaction_count)

    candidates = _normalize_candidates()

    validation_cfg = _extract_validation_cfg(reduction_cfg)
    if "pipeline" not in validation_cfg:
        candidate = params.get("validation") if isinstance(params, Mapping) else None
        if isinstance(candidate, Mapping):
            validation_cfg = dict(candidate)
    pipeline_value = _extract_pipeline_value(validation_cfg)

    metric = _extract_metric_name(validation_cfg)
    tolerance = _extract_tolerance(validation_cfg)
    rel_eps = _extract_rel_eps(validation_cfg)
    missing_strategy = _normalize_missing_strategy(
        validation_cfg.get("missing_strategy")
    )

    logger = logging.getLogger("rxn_platform.reduction")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    pipeline_cfg = _normalize_pipeline_cfg(pipeline_value, runner)
    steps = pipeline_cfg.get("steps", [])

    sim_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("sim_step_id"),
        task_name="sim.run",
        label="validation.sim_step_id",
        required=True,
    )
    obs_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("observables_step_id"),
        task_name="observables.run",
        label="validation.observables_step_id",
        required=False,
    )
    feat_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("features_step_id"),
        task_name="features.run",
        label="validation.features_step_id",
        required=False,
    )
    if obs_step_id is None and feat_step_id is None:
        raise ConfigError("validation requires an observables or features step.")

    baseline_sim_cfg = _extract_sim_override(validation_cfg)
    if baseline_sim_cfg is None:
        baseline_sim_cfg = None
        for step in steps:
            if step.get("id") == sim_step_id:
                sim_cfg = step.get("sim")
                if not isinstance(sim_cfg, Mapping):
                    raise ConfigError("pipeline sim step must include a sim mapping.")
                baseline_sim_cfg = dict(sim_cfg)
                break
    if baseline_sim_cfg is None:
        raise ConfigError("baseline sim config could not be resolved.")

    baseline_sim_cfg = dict(baseline_sim_cfg)
    baseline_sim_cfg["mechanism"] = mechanism_path

    baseline_pipeline_cfg = copy.deepcopy(pipeline_cfg)
    for step in baseline_pipeline_cfg.get("steps", []):
        if step.get("id") == sim_step_id:
            step["sim"] = dict(baseline_sim_cfg)
            break

    baseline_results = runner.run(baseline_pipeline_cfg)
    baseline_run_id = baseline_results.get(sim_step_id)
    if baseline_run_id is None:
        raise ConfigError("baseline sim step did not produce a run_id.")

    baseline_obs_id = baseline_results.get(obs_step_id) if obs_step_id else None
    baseline_feat_id = baseline_results.get(feat_step_id) if feat_step_id else None

    baseline_obs_groups: dict[tuple[str, str, str], list[Any]] = {}
    baseline_feat_groups: dict[tuple[str, str, str], list[Any]] = {}
    if obs_step_id:
        if baseline_obs_id is None:
            raise ConfigError("baseline observables step did not produce an artifact.")
        obs_rows = _extract_run_rows(
            store,
            kind="observables",
            artifact_id=baseline_obs_id,
            run_id=baseline_run_id,
            filename="values.parquet",
        )
        baseline_obs_groups = _group_metric_rows(obs_rows, name_field="observable")
        if not baseline_obs_groups:
            raise ConfigError("baseline observables produced no rows for comparison.")
    if feat_step_id:
        if baseline_feat_id is None:
            raise ConfigError("baseline features step did not produce an artifact.")
        feat_rows = _extract_run_rows(
            store,
            kind="features",
            artifact_id=baseline_feat_id,
            run_id=baseline_run_id,
            filename="features.parquet",
        )
        baseline_feat_groups = _group_metric_rows(feat_rows, name_field="feature")
        if not baseline_feat_groups:
            raise ConfigError("baseline features produced no rows for comparison.")

    cache: dict[str, dict[str, Any]] = {}
    candidate_entries: list[dict[str, Any]] = []

    sorted_indices = sorted(
        index_scores.keys(),
        key=lambda idx: (-index_scores[idx], idx),
    )

    def _select_disabled_indices(candidate: Mapping[str, Any]) -> list[int]:
        disabled: set[int] = set()
        top_k = candidate.get("top_k")
        score_lt = candidate.get("score_lt")
        if top_k is not None:
            try:
                top_k = int(top_k)
            except (TypeError, ValueError) as exc:
                raise ConfigError("threshold.top_k must be an integer.") from exc
            top_k = max(0, min(top_k, reaction_count))
            if top_k < len(sorted_indices):
                disabled.update(sorted_indices[top_k:])
        if score_lt is not None:
            try:
                score_lt = float(score_lt)
            except (TypeError, ValueError) as exc:
                raise ConfigError("threshold.score_lt must be numeric.") from exc
            disabled.update(
                idx for idx, score in index_scores.items() if score < score_lt
            )
        if reaction_count - len(disabled) < min_keep:
            keep = sorted_indices[:min_keep]
            disabled = {idx for idx in range(reaction_count) if idx not in keep}
        return sorted(disabled)

    def _evaluate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        disabled_indices = _select_disabled_indices(candidate)
        entries = [{"index": idx} for idx in disabled_indices]
        reduced_payload, disabled_set = compiler.apply_patch_entries(entries)
        remaining = reaction_count - len(disabled_set)
        mechanism_hash = compiler.mechanism_hash(reduced_payload)

        entry_base = {
            "threshold": dict(candidate),
            "disabled_count": len(disabled_set),
            "remaining_reactions": remaining,
            "mechanism_hash": mechanism_hash,
        }

        if mechanism_hash in cache:
            cached = cache[mechanism_hash]
            result = dict(entry_base)
            result.update(
                {
                    "passed": cached.get("passed"),
                    "qoi_error": cached.get("qoi_error"),
                    "run_id": cached.get("run_id"),
                    "observables_id": cached.get("observables_id"),
                    "features_id": cached.get("features_id"),
                }
            )
            return result

        temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        try:
            temp_dir = tempfile.TemporaryDirectory(prefix="rxn_gnn_prune_")
            reduced_mechanism = Path(temp_dir.name) / MECHANISM_FILENAME
            write_yaml_payload(reduced_mechanism, reduced_payload, sort_keys=False)

            reduced_sim_cfg = dict(baseline_sim_cfg)
            reduced_sim_cfg["mechanism"] = str(reduced_mechanism)
            reduced_sim_cfg.pop("reaction_multipliers", None)
            reduced_sim_cfg.pop("disabled_reactions", None)

            reduced_pipeline_cfg = copy.deepcopy(pipeline_cfg)
            for step in reduced_pipeline_cfg.get("steps", []):
                if step.get("id") == sim_step_id:
                    step["sim"] = dict(reduced_sim_cfg)
                    break

            reduced_results = runner.run(reduced_pipeline_cfg)
            reduced_run_id = reduced_results.get(sim_step_id)
            if reduced_run_id is None:
                raise ConfigError("reduced sim step did not produce a run_id.")

            reduced_obs_id = reduced_results.get(obs_step_id) if obs_step_id else None
            reduced_feat_id = reduced_results.get(feat_step_id) if feat_step_id else None

            patch_rows: list[dict[str, Any]] = []
            patch_pass = True
            evaluated_total = 0

            if obs_step_id:
                if reduced_obs_id is None:
                    raise ConfigError(
                        "reduced observables step did not produce an artifact."
                    )
                reduced_obs_rows = _extract_run_rows(
                    store,
                    kind="observables",
                    artifact_id=reduced_obs_id,
                    run_id=reduced_run_id,
                    filename="values.parquet",
                )
                reduced_obs_groups = _group_metric_rows(
                    reduced_obs_rows, name_field="observable"
                )
                rows_out, passed, evaluated = _compare_grouped_values(
                    kind="observable",
                    baseline_groups=baseline_obs_groups,
                    reduced_groups=reduced_obs_groups,
                    metric=metric,
                    tolerance=tolerance,
                    rel_eps=rel_eps,
                    missing_strategy=missing_strategy,
                    patch_index=0,
                    patch_id=mechanism_hash,
                    baseline_run_id=baseline_run_id,
                    reduced_run_id=reduced_run_id,
                    baseline_artifact_id=baseline_obs_id,
                    reduced_artifact_id=reduced_obs_id,
                )
                patch_rows.extend(rows_out)
                patch_pass = patch_pass and passed
                evaluated_total += evaluated

            if feat_step_id:
                if reduced_feat_id is None:
                    raise ConfigError(
                        "reduced features step did not produce an artifact."
                    )
                reduced_feat_rows = _extract_run_rows(
                    store,
                    kind="features",
                    artifact_id=reduced_feat_id,
                    run_id=reduced_run_id,
                    filename="features.parquet",
                )
                reduced_feat_groups = _group_metric_rows(
                    reduced_feat_rows, name_field="feature"
                )
                rows_out, passed, evaluated = _compare_grouped_values(
                    kind="feature",
                    baseline_groups=baseline_feat_groups,
                    reduced_groups=reduced_feat_groups,
                    metric=metric,
                    tolerance=tolerance,
                    rel_eps=rel_eps,
                    missing_strategy=missing_strategy,
                    patch_index=0,
                    patch_id=mechanism_hash,
                    baseline_run_id=baseline_run_id,
                    reduced_run_id=reduced_run_id,
                    baseline_artifact_id=baseline_feat_id,
                    reduced_artifact_id=reduced_feat_id,
                )
                patch_rows.extend(rows_out)
                patch_pass = patch_pass and passed
                evaluated_total += evaluated

            if evaluated_total == 0:
                patch_pass = False

            metric_values: list[float] = []
            for row in patch_rows:
                if row.get("status") != "ok":
                    continue
                value = row.get("abs_diff") if metric == "abs" else row.get("rel_diff")
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
                metric_values.append(value)
            qoi_error = max(metric_values) if metric_values else math.inf

            result = dict(entry_base)
            result.update(
                {
                    "passed": patch_pass and qoi_error <= tolerance,
                    "qoi_error": qoi_error,
                    "run_id": reduced_run_id,
                    "observables_id": reduced_obs_id,
                    "features_id": reduced_feat_id,
                }
            )
            cache[mechanism_hash] = dict(result)
            return result
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    for candidate in candidates:
        candidate_entries.append(_evaluate_candidate(candidate))

    passing = [row for row in candidate_entries if row.get("passed")]
    if passing:
        selected_row = min(
            passing,
            key=lambda row: (
                row.get("remaining_reactions", reaction_count),
                row.get("qoi_error", math.inf),
            ),
        )
    else:
        selected_row = min(
            candidate_entries,
            key=lambda row: (
                row.get("qoi_error", math.inf),
                row.get("remaining_reactions", reaction_count),
            ),
        )

    selected_disabled = _select_disabled_indices(selected_row.get("threshold", {}))
    selected_entries = [{"index": idx} for idx in selected_disabled]
    selected_payload, _ = compiler.apply_patch_entries(selected_entries)
    selected_patch = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "disabled_reactions": [{"index": idx} for idx in selected_disabled],
        "reaction_multipliers": [],
    }

    keep_reactions = [
        reaction_labels[idx]
        for idx in range(reaction_count)
        if idx not in set(selected_disabled)
    ]

    species_keep_top_k = importance_cfg.get("species_top_k") or importance_cfg.get(
        "keep_species_top_k"
    )
    species_threshold = importance_cfg.get("species_score_threshold")
    keep_species: list[str] = []
    if species_scores:
        sorted_species = sorted(
            species_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if species_keep_top_k is None and species_threshold is None:
            species_keep_top_k = min(10, len(sorted_species))
        if species_keep_top_k is not None:
            try:
                species_keep_top_k = int(species_keep_top_k)
            except (TypeError, ValueError) as exc:
                raise ConfigError("importance.species_top_k must be an integer.") from exc
            keep_species = [
                name for name, _ in sorted_species[: max(0, species_keep_top_k)]
            ]
        elif species_threshold is not None:
            try:
                threshold_val = float(species_threshold)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "importance.species_score_threshold must be numeric."
                ) from exc
            keep_species = [
                name for name, score in sorted_species if score >= threshold_val
            ]

    top_reactions = sorted(
        index_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    top_reaction_labels = [
        {"reaction": reaction_labels[idx], "score": score}
        for idx, score in top_reactions[: min(20, len(top_reactions))]
    ]

    inputs_payload: dict[str, Any] = {
        "mechanism": mechanism_path,
        "importance_id": importance_id,
        "importance": {
            "reaction_feature": reaction_feature,
            "species_feature": species_feature,
            "aggregate": importance_aggregate,
            "mode": importance_mode,
        },
        "thresholds": {
            "candidates": candidates,
            "min_keep": min_keep,
        },
        "validation": {
            "metric": metric,
            "tolerance": tolerance,
            "rel_eps": rel_eps,
            "missing_strategy": missing_strategy,
        },
        "selected": {
            "threshold": selected_row.get("threshold"),
            "disabled_count": selected_row.get("disabled_count"),
            "remaining_reactions": selected_row.get("remaining_reactions"),
            "qoi_error": selected_row.get("qoi_error"),
            "passed": selected_row.get("passed"),
        },
        "qoi": {
            "window_ids": sorted(qoi_window_set),
            "qoi_weight": qoi_weight,
            "non_qoi_weight": non_qoi_weight,
        },
    }

    parents: list[str] = [importance_id, baseline_run_id]
    if graph_id is not None:
        parents.append(graph_id)
    if baseline_obs_id:
        parents.append(baseline_obs_id)
    if baseline_feat_id:
        parents.append(baseline_feat_id)

    artifact_id = reduction_cfg.get("artifact_id") or reduction_cfg.get("id")
    if artifact_id is None:
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
    artifact_id = _require_nonempty_str(artifact_id, "artifact_id")

    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=_dedupe_preserve(parents),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _heat_color(value: Optional[float], max_value: float) -> str:
        if value is None:
            return "#efefef"
        if max_value <= 0.0:
            return "#f5f5f5"
        ratio = max(min(value / max_value, 1.0), 0.0)
        base = (245, 245, 245)
        target = (197, 74, 74)
        r = int(base[0] + (target[0] - base[0]) * ratio)
        g = int(base[1] + (target[1] - base[1]) * ratio)
        b = int(base[2] + (target[2] - base[2]) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _heatmap_html(
        *,
        title: str,
        z_values: Sequence[Sequence[Optional[float]]],
        x_labels: Sequence[str],
        y_labels: Sequence[str],
    ) -> str:
        flat = [value for row in z_values for value in row if value is not None]
        max_value = max(flat) if flat else 0.0
        header = "".join(f"<th>{label}</th>" for label in x_labels)
        rows_html: list[str] = []
        for label, row in zip(y_labels, z_values):
            cells = []
            for value in row:
                color = _heat_color(value, max_value)
                display = "" if value is None else f"{value:.3g}"
                cells.append(
                    f"<td style=\"background:{color};text-align:right;\">{display}</td>"
                )
            rows_html.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
        return (
            "<html><head><meta charset=\"utf-8\">"
            "<style>table{border-collapse:collapse;font-size:12px;}th,td{border:1px solid #ddd;padding:4px;}</style>"
            "</head><body>"
            f"<h2>{title}</h2>"
            "<table>"
            f"<tr><th>Reaction</th>{header}</tr>"
            + "".join(rows_html)
            + "</table></body></html>"
        )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, selected_patch, sort_keys=True)
        write_yaml_payload(base_dir / "reduced_mech.yaml", selected_payload, sort_keys=False)
        write_yaml_payload(base_dir / MECHANISM_FILENAME, selected_payload, sort_keys=False)
        def _disabled_count(patch: Mapping[str, Any]) -> int:
            count = 0
            disabled_raw = patch.get("disabled_reactions") or []
            if isinstance(disabled_raw, Mapping):
                count += len(disabled_raw)
            elif isinstance(disabled_raw, Sequence) and not isinstance(
                disabled_raw, (str, bytes, bytearray)
            ):
                count += len(disabled_raw)
            multipliers_raw = patch.get("reaction_multipliers") or []
            if isinstance(multipliers_raw, Mapping):
                multipliers_raw = [multipliers_raw]
            if isinstance(multipliers_raw, Sequence) and not isinstance(
                multipliers_raw, (str, bytes, bytearray)
            ):
                for entry in multipliers_raw:
                    if not isinstance(entry, Mapping):
                        continue
                    try:
                        multiplier = float(entry.get("multiplier", 1.0))
                    except (TypeError, ValueError):
                        continue
                    if multiplier == 0.0:
                        count += 1
            return int(count)

        reactions_after = _count_reactions(selected_payload)
        species_after = None
        species_raw = selected_payload.get("species")
        if isinstance(species_raw, Sequence) and not isinstance(
            species_raw, (str, bytes, bytearray)
        ):
            species_after = len(species_raw)
        species_before = None
        base_species_raw = compiler.payload.get("species")
        if isinstance(base_species_raw, Sequence) and not isinstance(
            base_species_raw, (str, bytes, bytearray)
        ):
            species_before = len(base_species_raw)

        disabled_count = _disabled_count(selected_patch)
        metrics_payload = {
            "schema_version": 1,
            "kind": "gnn_importance_prune_metrics",
            "counts": {
                "species_before": species_before,
                "species_after": species_after,
                "merged_species": 0,
                "reactions_before": reaction_count,
                "reactions_after": reactions_after,
                "disabled_reactions": disabled_count,
                "merged_reactions": 0,
            },
            "selected": inputs_payload["selected"],
            "candidates": candidate_entries,
            "keep_reactions": keep_reactions,
            "keep_species": keep_species,
            "top_reactions": top_reaction_labels,
        }
        write_json_atomic(base_dir / "metrics.json", metrics_payload)

        if reaction_by_window:
            window_ids = sorted(
                {window_id for _, window_id in reaction_by_window.keys()}
            )
            top_keys = [idx for idx, _ in top_reactions[: min(12, len(top_reactions))]]
            key_lookup: dict[int, tuple[str, Any]] = {}
            for key, info in reaction_info.items():
                idx_val = info.get("reaction_index")
                if idx_val is None:
                    reaction_id = info.get("reaction_id")
                    if reaction_id is not None:
                        indices = id_map.get(str(reaction_id))
                        if indices and len(indices) == 1:
                            idx_val = indices[0]
                if idx_val is not None:
                    key_lookup[int(idx_val)] = key
            z_values: list[list[Optional[float]]] = []
            y_labels: list[str] = []
            for idx in top_keys:
                key = key_lookup.get(idx)
                if key is None:
                    continue
                row_vals: list[Optional[float]] = []
                for window_id in window_ids:
                    values = reaction_by_window.get((key, window_id), [])
                    if values:
                        row_vals.append(sum(values) / float(len(values)))
                    else:
                        row_vals.append(None)
                z_values.append(row_vals)
                y_labels.append(reaction_labels.get(idx, f"R{idx + 1}"))
            if z_values and y_labels:
                viz_dir = base_dir / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                heatmap = _heatmap_html(
                    title="Reaction Importance Heatmap",
                    z_values=z_values,
                    x_labels=[str(wid) for wid in window_ids],
                    y_labels=y_labels,
                )
                (viz_dir / "importance_heatmap.html").write_text(
                    heatmap + "\n",
                    encoding="utf-8",
                )

    return store.ensure(manifest, writer=_writer)


def propose_node_lumping(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Propose species node lumping based on annotated graph metadata."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    lumping_cfg = _extract_node_lumping_cfg(reduction_cfg)

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            lumping_cfg,
            keys=("graph", "graph_id", "graph_artifact"),
            label="lumping.graph",
        )
    if graph_id is None:
        raise ConfigError("node lumping requires a graph input.")

    threshold_value = lumping_cfg.get("threshold")
    if threshold_value is None:
        threshold_value = lumping_cfg.get("similarity_threshold") or lumping_cfg.get(
            "min_similarity"
        )
    threshold = _coerce_similarity_threshold(threshold_value)
    method = _coerce_similarity_method(
        lumping_cfg.get("method") or lumping_cfg.get("cluster_method")
    )
    weights = _extract_similarity_weights(lumping_cfg)
    charge_scale = _coerce_charge_scale(lumping_cfg)

    representative_cfg = lumping_cfg.get("representative") or {}
    if representative_cfg is None:
        representative_cfg = {}
    if not isinstance(representative_cfg, Mapping):
        raise ConfigError("lumping.representative must be a mapping.")
    metric_raw = representative_cfg.get("metric") or representative_cfg.get(
        "score"
    ) or DEFAULT_REPRESENTATIVE_METRIC
    metric = str(metric_raw).strip().lower()
    if metric in {"centrality", "degree_centrality"}:
        metric = "degree"
    if metric in {"lexical", "lexicographic"}:
        metric = "name"
    if metric not in {"degree", "name"}:
        raise ConfigError("lumping.representative.metric must be 'degree' or 'name'.")

    nodes, links = _load_graph_nodes_and_links(store, graph_id)
    species_entries = _prepare_species_entries(nodes, links)

    # Optional network-aware similarities derived from the bipartite graph context.
    species_ids = {entry["node_id"] for entry in species_entries}
    reaction_type_by_id: dict[str, str] = {}
    for node in nodes:
        node_id = _node_ref_to_id(node.get("id"))
        if node_id is None:
            continue
        if node.get("kind") != "reaction":
            continue
        reaction_type_by_id[node_id] = (
            _normalize_text(node.get("reaction_type") or node.get("type")) or "unknown"
        )
    reaction_ids = set(reaction_type_by_id.keys())

    neighbor_reactions: dict[str, set[str]] = {sid: set() for sid in species_ids}
    reaction_type_profiles: dict[str, dict[str, float]] = {sid: {} for sid in species_ids}
    for link in links:
        source = _node_ref_to_id(link.get("source"))
        target = _node_ref_to_id(link.get("target"))
        if source is None or target is None:
            continue
        species_id: Optional[str] = None
        reaction_id: Optional[str] = None
        if source in species_ids and target in reaction_ids:
            species_id = source
            reaction_id = target
        elif target in species_ids and source in reaction_ids:
            species_id = target
            reaction_id = source
        if species_id is None or reaction_id is None:
            continue
        neighbor_reactions[species_id].add(reaction_id)
        rtype = reaction_type_by_id.get(reaction_id, "unknown")
        role = _reaction_role_from_link(link) or "other"
        key = f"{rtype}:{role}"
        profile = reaction_type_profiles[species_id]
        profile[key] = profile.get(key, 0.0) + 1.0

    def _profile_norm(profile: Mapping[str, float]) -> float:
        return math.sqrt(sum(value * value for value in profile.values()))

    profile_norms = {sid: _profile_norm(profile) for sid, profile in reaction_type_profiles.items()}

    def _cosine_sparse(
        left: Mapping[str, float],
        right: Mapping[str, float],
        left_norm: float,
        right_norm: float,
    ) -> float:
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
            left_norm, right_norm = right_norm, left_norm
        dot = 0.0
        for k, v in left.items():
            other = right.get(k)
            if other is not None:
                dot += float(v) * float(other)
        return dot / (left_norm * right_norm)

    element_names = sorted(
        {element for entry in species_entries for element in entry["elements"]}
    )
    vectors: dict[str, list[float]] = {}
    for entry in species_entries:
        vector = [float(entry["elements"].get(element, 0.0)) for element in element_names]
        vectors[entry["node_id"]] = vector

    def _sort_key(node_id: str) -> tuple[str, str]:
        name = node_to_species[node_id]
        return (name.lower(), node_id)

    node_to_species = {entry["node_id"]: entry["species"] for entry in species_entries}
    species_entries.sort(key=lambda entry: _sort_key(entry["node_id"]))

    pairs: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = {
        entry["node_id"]: set() for entry in species_entries
    }
    total_weight = sum(weights.values())

    for idx, left in enumerate(species_entries):
        for right in species_entries[idx + 1 :]:
            left_id = left["node_id"]
            right_id = right["node_id"]
            element_sim = _cosine_similarity(vectors[left_id], vectors[right_id])
            charge_sim = 0.0
            if left["charge"] is not None and right["charge"] is not None:
                diff = abs(left["charge"] - right["charge"])
                charge_sim = max(0.0, 1.0 - diff / charge_scale)
            phase_sim = (
                1.0
                if left["phase"] and left["phase"] == right["phase"]
                else 0.0
            )
            state_sim = (
                1.0
                if left["state"] and left["state"] == right["state"]
                else 0.0
            )
            reaction_type_profile_sim = 0.0
            if weights.get("reaction_type_profile", 0.0) > 0.0:
                reaction_type_profile_sim = _cosine_sparse(
                    reaction_type_profiles.get(left_id, {}),
                    reaction_type_profiles.get(right_id, {}),
                    float(profile_norms.get(left_id, 0.0)),
                    float(profile_norms.get(right_id, 0.0)),
                )

            neighbor_reaction_sim = 0.0
            if weights.get("neighbor_reaction", 0.0) > 0.0:
                neighbor_reaction_sim = _jaccard_similarity(
                    neighbor_reactions.get(left_id, set()),
                    neighbor_reactions.get(right_id, set()),
                )

            components = {
                "elements": element_sim,
                "charge": charge_sim,
                "phase": phase_sim,
                "state": state_sim,
                "reaction_type_profile": reaction_type_profile_sim,
                "neighbor_reaction": neighbor_reaction_sim,
            }
            score = (
                weights["elements"] * element_sim
                + weights["charge"] * charge_sim
                + weights["phase"] * phase_sim
                + weights["state"] * state_sim
                + weights.get("reaction_type_profile", 0.0) * reaction_type_profile_sim
                + weights.get("neighbor_reaction", 0.0) * neighbor_reaction_sim
            ) / total_weight
            pairs.append(
                {
                    "node_id_a": left_id,
                    "species_a": left["species"],
                    "node_id_b": right_id,
                    "species_b": right["species"],
                    "similarity": score,
                    "components": components,
                }
            )
            if score >= threshold:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    visited: set[str] = set()
    clusters: list[list[str]] = []
    for node_id in sorted(adjacency.keys(), key=_sort_key):
        if node_id in visited:
            continue
        stack = [node_id]
        cluster: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in sorted(adjacency.get(current, set()), key=_sort_key):
                if neighbor not in visited:
                    stack.append(neighbor)
        cluster.sort(key=_sort_key)
        clusters.append(cluster)

    node_meta = {entry["node_id"]: entry for entry in species_entries}
    cluster_payloads: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []

    for cluster_id, members in enumerate(clusters):
        scores: dict[str, float] = {}
        for node_id in members:
            if metric == "degree":
                score = float(node_meta[node_id].get("degree", 0))
            else:
                score = 0.0
            scores[node_id] = score
        representative = sorted(
            members,
            key=lambda node_id: (
                -scores.get(node_id, 0.0),
                node_to_species[node_id].lower(),
                node_id,
            ),
        )[0]

        cluster_payloads.append(
            {
                "cluster_id": cluster_id,
                "members": [node_to_species[node_id] for node_id in members],
                "member_node_ids": list(members),
                "representative": node_to_species[representative],
                "representative_node_id": representative,
                "selection": {
                    "metric": metric,
                    "scores": {
                        node_to_species[node_id]: scores[node_id] for node_id in members
                    },
                    "reason": "max_score",
                },
                "summary": {"size": len(members)},
            }
        )
        for node_id in members:
            mapping.append(
                {
                    "node_id": node_id,
                    "species": node_to_species[node_id],
                    "representative_node_id": representative,
                    "representative": node_to_species[representative],
                    "cluster_id": cluster_id,
                }
            )

    mapping.sort(
        key=lambda entry: (entry["species"].lower(), entry["node_id"]),
    )

    species_index_by_node_id = {
        entry["node_id"]: entry.get("species_index") for entry in species_entries
    }
    for entry in mapping:
        if not isinstance(entry, dict):
            continue
        if "species_index" in entry:
            continue
        node_id = entry.get("node_id")
        if isinstance(node_id, str):
            entry["species_index"] = species_index_by_node_id.get(node_id)

    species_payload = [
        {
            "node_id": entry["node_id"],
            "species": entry["species"],
            "elements": entry["elements"],
            "charge": entry["charge"],
            "phase": entry["phase"],
            "state": entry["state"],
            "species_index": entry.get("species_index"),
            "degree": entry["degree"],
        }
        for entry in species_entries
    ]

    payload = {
        "schema_version": NODE_LUMPING_SCHEMA_VERSION,
        "kind": "node_lumping",
        "source": {"graph_id": graph_id},
        "similarity": {
            "method": method,
            "threshold": threshold,
            "weights": weights,
            "charge_scale": charge_scale,
            "elements": {"order": element_names},
            "pairs": pairs,
        },
        "clusters": cluster_payloads,
        "mapping": mapping,
        "species": species_payload,
        "meta": {
            "species_count": len(species_entries),
            "cluster_count": len(clusters),
            "selection_metric": metric,
            "selection_source": "graph_edges",
        },
    }

    inputs_payload = {
        "mode": "node_lumping",
        "graph": graph_id,
        "similarity": {
            "method": method,
            "threshold": threshold,
            "weights": weights,
            "charge_scale": charge_scale,
        },
        "representative": {"metric": metric},
    }

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / NODE_LUMPING_FILENAME, payload)

        # Emit a mapping.json in the shared "superstate mapping" shape so downstream
        # tasks can consume node_lumping outputs without special-casing.
        superstates: list[dict[str, Any]] = []
        for cluster in cluster_payloads:
            cid = cluster.get("cluster_id")
            if not isinstance(cid, int) or isinstance(cid, bool):
                continue
            superstates.append(
                {
                    "superstate_id": int(cid),
                    "name": f"S{int(cid):03d}",
                    "representative": cluster.get("representative"),
                    "members": cluster.get("members") or [],
                    "summary": cluster.get("summary") or {},
                }
            )
        mapping_payload = {
            "schema_version": 1,
            "kind": "superstate_mapping",
            "source": {"graph_id": graph_id},
            "superstates": superstates,
            "mapping": [
                {
                    "species": entry.get("species"),
                    "species_index": entry.get("species_index"),
                    "superstate_id": entry.get("cluster_id"),
                    "representative": entry.get("representative"),
                }
                for entry in mapping
            ],
            # Minimal shared fields for downstream consumers that expect a uniform
            # superstate mapping contract.
            "guards": {},
            "policy": {"origin": "node_lumping"},
            "composition_meta": [
                {
                    "species": entry.get("species"),
                    "species_index": entry.get("species_index"),
                    "node_id": entry.get("node_id"),
                    "formula": None,
                    "elements": dict(entry.get("elements") or {}),
                    "charge": entry.get("charge"),
                    "kind": None,
                    "mw": None,
                    "phase": entry.get("phase"),
                    "state": entry.get("state"),
                }
                for entry in species_payload
                if isinstance(entry, Mapping)
            ],
            "meta": {
                "selection_metric": metric,
                "similarity": {
                    "method": method,
                    "threshold": threshold,
                    "weights": weights,
                    "charge_scale": charge_scale,
                },
            },
        }
        write_json_atomic(base_dir / "mapping.json", mapping_payload)

        metrics = {
            "schema_version": 1,
            "kind": "node_lumping_metrics",
            "species_count": len(species_entries),
            "cluster_count": len(clusters),
            "cluster_sizes": _cluster_size_stats([len(members) for members in clusters]),
            "similarity": {
                "method": method,
                "threshold": threshold,
                "weights": weights,
                "charge_scale": charge_scale,
            },
            "representative_metric": metric,
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def superstate_mapping(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build a projection-only superstate mapping from a stoichiometric bipartite graph.

    This mapping is designed for analysis/projection (QoI on superstates, reaction
    aggregation) and intentionally allows merging species with different elemental
    compositions. It does NOT attempt to produce an element-conserving reduced
    mechanism suitable for Cantera re-simulation.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    inputs = _extract_inputs(reduction_cfg)

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            inputs,
            keys=("graph", "graph_id", "graph_artifact"),
            label="reduction.inputs.graph",
        )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            params,
            keys=("graph", "graph_id", "graph_artifact"),
            label="reduction.params.graph",
        )
    if graph_id is None:
        raise ConfigError("superstate_mapping requires graph_id.")

    guards_cfg = params.get("guards") or reduction_cfg.get("guards") or {}
    if guards_cfg is None:
        guards_cfg = {}
    if not isinstance(guards_cfg, Mapping):
        raise ConfigError("guards must be a mapping when provided.")
    guards_cfg = dict(guards_cfg)

    require_element_overlap = bool(
        guards_cfg.get("require_element_overlap", True)
    )
    require_heavy_element_overlap = bool(
        guards_cfg.get("require_heavy_element_overlap", False)
    )
    require_same_kind = bool(guards_cfg.get("require_same_kind", True))
    require_same_phase = bool(guards_cfg.get("require_same_phase", True))

    protected_raw = guards_cfg.get("protected_species", params.get("protected_species"))
    if protected_raw is None:
        protected_species = {"CO", "CO2"}
    else:
        protected_species = set(_normalize_str_list(protected_raw, "protected_species"))

    def _normalize_pair_groups(raw: Any, label: str) -> list[list[str]]:
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            raw = list(raw.values())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ConfigError(f"{label} must be a list of string lists.")
        groups: list[list[str]] = []
        for idx, entry in enumerate(raw):
            if entry is None:
                continue
            if isinstance(entry, str):
                raise ConfigError(f"{label}[{idx}] must be a list of strings.")
            if not isinstance(entry, Sequence) or isinstance(entry, (bytes, bytearray)):
                raise ConfigError(f"{label}[{idx}] must be a list of strings.")
            values = _normalize_str_list(list(entry), f"{label}[{idx}]")
            if len(values) < 2:
                continue
            groups.append(values)
        return groups

    cannot_link_groups = _normalize_pair_groups(
        guards_cfg.get("cannot_link") or guards_cfg.get("cannot_merge"),
        "guards.cannot_link",
    )
    must_link_groups = _normalize_pair_groups(
        guards_cfg.get("must_link") or guards_cfg.get("must_merge"),
        "guards.must_link",
    )

    similarity_cfg = params.get("similarity") or reduction_cfg.get("similarity") or {}
    if similarity_cfg is None:
        similarity_cfg = {}
    if not isinstance(similarity_cfg, Mapping):
        raise ConfigError("similarity must be a mapping when provided.")
    similarity_cfg = dict(similarity_cfg)

    threshold_raw = similarity_cfg.get("threshold", similarity_cfg.get("min_similarity"))
    if threshold_raw is None:
        threshold = DEFAULT_SUPERSTATE_MAPPING_THRESHOLD
    else:
        if isinstance(threshold_raw, bool):
            raise ConfigError("similarity.threshold must be a number.")
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.threshold must be a number.") from exc
    if threshold < 0.0 or threshold > 1.0 or math.isnan(threshold) or math.isinf(threshold):
        raise ConfigError("similarity.threshold must be in [0, 1] and finite.")

    threshold_mode_raw = similarity_cfg.get("threshold_mode")
    if threshold_mode_raw is None:
        threshold_mode = "fixed"
    else:
        threshold_mode = str(threshold_mode_raw).strip().lower()
        if threshold_mode in {"fix", "fixed", "value"}:
            threshold_mode = "fixed"
        elif threshold_mode in {"quantile", "q"}:
            threshold_mode = "quantile"
        elif threshold_mode in {"target_clusters", "target_cluster", "clusters"}:
            threshold_mode = "target_clusters"
        elif threshold_mode in {
            "target_merged_species",
            "target_merged",
            "merged_species",
            "merged",
        }:
            threshold_mode = "target_merged_species"
        else:
            raise ConfigError(
                "similarity.threshold_mode must be one of: fixed, quantile, target_clusters, target_merged_species."
            )

    quantile_q_raw = similarity_cfg.get("quantile_q")
    if quantile_q_raw is None:
        quantile_q = 0.98
    else:
        if isinstance(quantile_q_raw, bool):
            raise ConfigError("similarity.quantile_q must be a number.")
        try:
            quantile_q = float(quantile_q_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.quantile_q must be a number.") from exc
    if quantile_q < 0.0 or quantile_q > 1.0 or math.isnan(quantile_q) or math.isinf(quantile_q):
        raise ConfigError("similarity.quantile_q must be in [0, 1] and finite.")

    target_clusters_raw = similarity_cfg.get("target_clusters")
    if target_clusters_raw is None:
        target_clusters: Optional[int] = None
    else:
        if isinstance(target_clusters_raw, bool):
            raise ConfigError("similarity.target_clusters must be an integer.")
        try:
            target_clusters = int(target_clusters_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.target_clusters must be an integer.") from exc

    target_merged_raw = similarity_cfg.get("target_merged_species")
    if target_merged_raw is None:
        target_merged_species: Optional[int] = None
    else:
        if isinstance(target_merged_raw, bool):
            raise ConfigError("similarity.target_merged_species must be an integer.")
        try:
            target_merged_species = int(target_merged_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.target_merged_species must be an integer.") from exc

    max_iter_raw = similarity_cfg.get("max_iter")
    if max_iter_raw is None:
        max_iter = 30
    else:
        if isinstance(max_iter_raw, bool):
            raise ConfigError("similarity.max_iter must be an integer.")
        try:
            max_iter = int(max_iter_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.max_iter must be an integer.") from exc
    if max_iter <= 0:
        raise ConfigError("similarity.max_iter must be positive.")

    weights_raw = similarity_cfg.get("weights")
    weights: dict[str, float] = dict(DEFAULT_SUPERSTATE_MAPPING_WEIGHTS)
    if weights_raw is not None:
        if not isinstance(weights_raw, Mapping):
            raise ConfigError("similarity.weights must be a mapping.")
        for key, value in weights_raw.items():
            if key is None:
                continue
            k = str(key).strip()
            if not k:
                continue
            if k not in weights:
                raise ConfigError(f"similarity.weights has unknown key: {k}")
            if value is None:
                continue
            if isinstance(value, bool):
                raise ConfigError(f"similarity.weights.{k} must be a number.")
            try:
                w = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"similarity.weights.{k} must be a number.") from exc
            if w < 0.0 or math.isnan(w) or math.isinf(w):
                raise ConfigError(f"similarity.weights.{k} must be >= 0 and finite.")
            weights[k] = w
    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        raise ConfigError("similarity.weights must include at least one positive value.")

    max_cluster_size_raw = similarity_cfg.get("max_cluster_size", params.get("max_cluster_size"))
    if max_cluster_size_raw is None:
        max_cluster_size = DEFAULT_SUPERSTATE_MAPPING_MAX_CLUSTER_SIZE
    else:
        if isinstance(max_cluster_size_raw, bool):
            raise ConfigError("similarity.max_cluster_size must be an integer.")
        try:
            max_cluster_size = int(max_cluster_size_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.max_cluster_size must be an integer.") from exc
    if max_cluster_size <= 0:
        raise ConfigError("similarity.max_cluster_size must be positive.")

    rep_cfg = similarity_cfg.get("representative") or {}
    if rep_cfg is None:
        rep_cfg = {}
    if not isinstance(rep_cfg, Mapping):
        raise ConfigError("similarity.representative must be a mapping when provided.")
    rep_metric_raw = rep_cfg.get("metric") or rep_cfg.get("score") or DEFAULT_REPRESENTATIVE_METRIC
    rep_metric = str(rep_metric_raw).strip().lower()
    if rep_metric in {"centrality", "degree_centrality"}:
        rep_metric = "degree"
    if rep_metric in {"lexical", "lexicographic"}:
        rep_metric = "name"
    if rep_metric not in {"degree", "name"}:
        raise ConfigError("similarity.representative.metric must be 'degree' or 'name'.")

    # Soft constraints (kept intentionally loose by default).
    mw_log_tol_raw = params.get("mw_log_tol", guards_cfg.get("mw_log_tol"))
    if mw_log_tol_raw is None:
        mw_log_tol = DEFAULT_SUPERSTATE_MAPPING_MW_LOG_TOL
    else:
        if isinstance(mw_log_tol_raw, bool):
            raise ConfigError("mw_log_tol must be numeric.")
        try:
            mw_log_tol = float(mw_log_tol_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("mw_log_tol must be numeric.") from exc
    if mw_log_tol < 0.0 or math.isnan(mw_log_tol) or math.isinf(mw_log_tol):
        raise ConfigError("mw_log_tol must be >= 0 and finite.")

    elem_cos_min_raw = params.get(
        "elements_cosine_min",
        guards_cfg.get("elements_cosine_min"),
    )
    if elem_cos_min_raw is None:
        elements_cosine_min = DEFAULT_SUPERSTATE_MAPPING_ELEMENTS_COSINE_MIN
    else:
        if isinstance(elem_cos_min_raw, bool):
            raise ConfigError("elements_cosine_min must be numeric.")
        try:
            elements_cosine_min = float(elem_cos_min_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("elements_cosine_min must be numeric.") from exc
    if (
        elements_cosine_min < -1.0
        or elements_cosine_min > 1.0
        or math.isnan(elements_cosine_min)
        or math.isinf(elements_cosine_min)
    ):
        raise ConfigError("elements_cosine_min must be in [-1, 1] and finite.")

    charge_scale_raw = similarity_cfg.get("charge_scale", params.get("charge_scale", 2.0))
    if charge_scale_raw is None:
        charge_scale = 2.0
    else:
        if isinstance(charge_scale_raw, bool):
            raise ConfigError("charge_scale must be numeric.")
        try:
            charge_scale = float(charge_scale_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("charge_scale must be numeric.") from exc
    if charge_scale <= 0.0 or math.isnan(charge_scale) or math.isinf(charge_scale):
        raise ConfigError("charge_scale must be positive and finite.")

    nodes, links = _load_graph_nodes_and_links(store, graph_id)
    species_entries = _prepare_species_entries(nodes, links)

    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = _node_ref_to_id(node.get("id"))
        if node_id is None:
            continue
        if isinstance(node, Mapping):
            node_by_id[node_id] = dict(node)

    def _canonical_charge(charge: Optional[float]) -> int:
        if charge is None:
            return 0
        try:
            value = float(charge)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(value):
            return 0
        rounded = round(value)
        if abs(value - rounded) < 1.0e-6:
            return int(rounded)
        return int(value)

    def _molecular_weight(elements: Mapping[str, float]) -> Optional[float]:
        total = 0.0
        for element, count in elements.items():
            weight = ATOMIC_WEIGHTS.get(str(element))
            if weight is None:
                return None
            total += float(weight) * float(count)
        if not math.isfinite(total) or total <= 0.0:
            return None
        return float(total)

    def _infer_kind(elements: Mapping[str, float], charge: Optional[float]) -> str:
        q = _canonical_charge(charge)
        if q != 0:
            return "ion"
        total_atoms = float(sum(float(v) for v in elements.values()))
        if abs(total_atoms - 1.0) < 1.0e-12 and len(elements) == 1:
            return "atom"
        electrons = 0.0
        for element, count in elements.items():
            z = ATOMIC_NUMBERS.get(str(element))
            if z is None:
                return "unknown"
            electrons += float(z) * float(count)
        electrons -= float(q)
        if not math.isfinite(electrons):
            return "unknown"
        n_e = int(round(electrons))
        if abs(electrons - n_e) > 1.0e-6:
            return "unknown"
        if n_e % 2 != 0:
            return "radical"
        return "molecule"

    def _element_set(elements: Mapping[str, float]) -> set[str]:
        return {str(k) for k, v in elements.items() if float(v) != 0.0}

    def _element_set_jaccard(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        left_set = _element_set(left)
        right_set = _element_set(right)
        if not left_set and not right_set:
            return 0.0
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    # Attach enriched metadata to species entries.
    node_to_species = {entry["node_id"]: entry["species"] for entry in species_entries}
    species_to_node_id: dict[str, str] = {
        entry["species"]: entry["node_id"] for entry in species_entries
    }
    for entry in species_entries:
        node_id = entry["node_id"]
        node = node_by_id.get(node_id, {})
        entry["formula"] = node.get("formula")
        entry["mw"] = _molecular_weight(entry["elements"])
        entry["kind"] = _infer_kind(entry["elements"], entry.get("charge"))
        entry["element_set"] = _element_set(entry["elements"])
        entry["heavy_element_set"] = {e for e in entry["element_set"] if e != "H"}

    # Build optional network-aware contexts from the bipartite graph.
    species_ids = {entry["node_id"] for entry in species_entries}
    reaction_type_by_id: dict[str, str] = {}
    reaction_ids: set[str] = set()
    reaction_count = 0
    for node in nodes:
        node_id = _node_ref_to_id(node.get("id"))
        if node_id is None:
            continue
        if node.get("kind") != "reaction":
            continue
        reaction_count += 1
        reaction_type_by_id[node_id] = (
            _normalize_text(node.get("reaction_type") or node.get("type")) or "unknown"
        )
        reaction_ids.add(node_id)

    neighbor_reactions: dict[str, set[str]] = {sid: set() for sid in species_ids}
    reaction_type_profiles: dict[str, dict[str, float]] = {sid: {} for sid in species_ids}
    for link in links:
        source = _node_ref_to_id(link.get("source"))
        target = _node_ref_to_id(link.get("target"))
        if source is None or target is None:
            continue
        species_id: Optional[str] = None
        reaction_id: Optional[str] = None
        if source in species_ids and target in reaction_ids:
            species_id = source
            reaction_id = target
        elif target in species_ids and source in reaction_ids:
            species_id = target
            reaction_id = source
        if species_id is None or reaction_id is None:
            continue
        neighbor_reactions[species_id].add(reaction_id)
        rtype = reaction_type_by_id.get(reaction_id, "unknown")
        role = _reaction_role_from_link(link) or "other"
        key = f"{rtype}:{role}"
        profile = reaction_type_profiles[species_id]
        profile[key] = profile.get(key, 0.0) + 1.0

    def _profile_norm(profile: Mapping[str, float]) -> float:
        return math.sqrt(sum(value * value for value in profile.values()))

    profile_norms = {sid: _profile_norm(profile) for sid, profile in reaction_type_profiles.items()}

    def _cosine_sparse(
        left: Mapping[str, float],
        right: Mapping[str, float],
        left_norm: float,
        right_norm: float,
    ) -> float:
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
            left_norm, right_norm = right_norm, left_norm
        dot = 0.0
        for k, v in left.items():
            other = right.get(k)
            if other is not None:
                dot += float(v) * float(other)
        return dot / (left_norm * right_norm)

    element_names = sorted(
        {element for entry in species_entries for element in entry["elements"]}
    )
    vectors: dict[str, list[float]] = {}
    for entry in species_entries:
        vector = [float(entry["elements"].get(element, 0.0)) for element in element_names]
        vectors[entry["node_id"]] = vector

    def _sort_key(node_id: str) -> tuple[str, str]:
        name = node_to_species[node_id]
        return (name.lower(), node_id)

    species_entries.sort(key=lambda entry: _sort_key(entry["node_id"]))

    protected_node_ids = {
        species_to_node_id[name]
        for name in protected_species
        if name in species_to_node_id
    }

    cannot_link_pairs: set[tuple[str, str]] = set()
    for group in cannot_link_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                cannot_link_pairs.add((a, b))
                cannot_link_pairs.add((b, a))

    skip_reasons: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []

    def _is_blocked_by_cannot_link(a: str, b: str) -> bool:
        return (a, b) in cannot_link_pairs

    for idx, left in enumerate(species_entries):
        for right in species_entries[idx + 1 :]:
            left_id = left["node_id"]
            right_id = right["node_id"]
            left_name = left["species"]
            right_name = right["species"]

            if left_id in protected_node_ids or right_id in protected_node_ids:
                skip_reasons["protected_species"] += 1
                continue
            if _is_blocked_by_cannot_link(left_name, right_name):
                skip_reasons["cannot_link"] += 1
                continue

            if require_same_phase:
                lp = left.get("phase")
                rp = right.get("phase")
                if lp and rp and lp != rp:
                    skip_reasons["phase_mismatch"] += 1
                    continue
            if require_same_kind:
                if left.get("kind") != right.get("kind"):
                    skip_reasons["kind_mismatch"] += 1
                    continue
            if require_element_overlap:
                if not (set(left.get("element_set") or set()) & set(right.get("element_set") or set())):
                    skip_reasons["no_element_overlap"] += 1
                    continue
            if require_heavy_element_overlap:
                if not (set(left.get("heavy_element_set") or set()) & set(right.get("heavy_element_set") or set())):
                    skip_reasons["no_heavy_element_overlap"] += 1
                    continue

            element_sim = _cosine_similarity(vectors[left_id], vectors[right_id])
            if element_sim < elements_cosine_min:
                skip_reasons["elements_cosine_min"] += 1
                continue

            left_mw = left.get("mw")
            right_mw = right.get("mw")
            if (
                mw_log_tol > 0.0
                and isinstance(left_mw, (int, float))
                and isinstance(right_mw, (int, float))
                and left_mw > 0.0
                and right_mw > 0.0
            ):
                ratio = float(left_mw) / float(right_mw)
                if ratio <= 0.0:
                    skip_reasons["mw_invalid"] += 1
                    continue
                if abs(math.log(ratio)) > mw_log_tol:
                    skip_reasons["mw_log_tol"] += 1
                    continue

            element_set_sim = _element_set_jaccard(left["elements"], right["elements"])
            neighbor_reaction_sim = _jaccard_similarity(
                neighbor_reactions.get(left_id, set()),
                neighbor_reactions.get(right_id, set()),
            )
            reaction_type_profile_sim = _cosine_sparse(
                reaction_type_profiles.get(left_id, {}),
                reaction_type_profiles.get(right_id, {}),
                float(profile_norms.get(left_id, 0.0)),
                float(profile_norms.get(right_id, 0.0)),
            )

            charge_sim = 0.0
            if left.get("charge") is not None and right.get("charge") is not None:
                diff = abs(float(left["charge"]) - float(right["charge"]))
                charge_sim = max(0.0, 1.0 - diff / charge_scale)

            components = {
                "element_set_jaccard": element_set_sim,
                "elements_cosine": element_sim,
                "neighbor_reaction": neighbor_reaction_sim,
                "reaction_type_profile": reaction_type_profile_sim,
                "charge": charge_sim,
            }
            score = (
                weights["element_set_jaccard"] * element_set_sim
                + weights["elements_cosine"] * element_sim
                + weights["neighbor_reaction"] * neighbor_reaction_sim
                + weights["reaction_type_profile"] * reaction_type_profile_sim
                + weights["charge"] * charge_sim
            ) / total_weight

            pairs.append(
                {
                    "node_id_a": left_id,
                    "species_a": left_name,
                    "node_id_b": right_id,
                    "species_b": right_name,
                    "similarity": float(score),
                    "components": components,
                }
            )

    entry_by_node_id = {entry["node_id"]: entry for entry in species_entries}

    # Enforce must-link groups (soft override), but never break hard guards.
    must_link_edges: list[tuple[str, str]] = []
    for group in must_link_groups:
        group_node_ids = [species_to_node_id.get(name) for name in group]
        group_node_ids = [nid for nid in group_node_ids if isinstance(nid, str)]
        for i in range(len(group_node_ids)):
            for j in range(i + 1, len(group_node_ids)):
                a = group_node_ids[i]
                b = group_node_ids[j]
                if a in protected_node_ids or b in protected_node_ids:
                    skip_reasons["must_link_protected"] += 1
                    continue
                left_entry = entry_by_node_id.get(a)
                right_entry = entry_by_node_id.get(b)
                if not left_entry or not right_entry:
                    skip_reasons["must_link_missing_entry"] += 1
                    continue
                if require_element_overlap and not (
                    set(left_entry.get("element_set") or set())
                    & set(right_entry.get("element_set") or set())
                ):
                    skip_reasons["must_link_no_element_overlap"] += 1
                    continue
                if require_heavy_element_overlap and not (
                    set(left_entry.get("heavy_element_set") or set())
                    & set(right_entry.get("heavy_element_set") or set())
                ):
                    skip_reasons["must_link_no_heavy_element_overlap"] += 1
                    continue
                if require_same_kind and left_entry.get("kind") != right_entry.get("kind"):
                    skip_reasons["must_link_kind_mismatch"] += 1
                    continue
                if require_same_phase:
                    lp = left_entry.get("phase")
                    rp = right_entry.get("phase")
                    if lp and rp and lp != rp:
                        skip_reasons["must_link_phase_mismatch"] += 1
                        continue
                must_link_edges.append((a, b))

    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 1.0
        q = min(max(float(q), 0.0), 1.0)
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = q * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        frac = pos - lo
        return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])

    def _clusters_for_threshold(thr: float) -> tuple[list[list[str]], int]:
        adjacency: dict[str, set[str]] = {
            entry["node_id"]: set() for entry in species_entries
        }
        for pair in pairs:
            if float(pair.get("similarity", 0.0)) < thr:
                continue
            a = pair.get("node_id_a")
            b = pair.get("node_id_b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
        for a, b in must_link_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)

        visited: set[str] = set()
        clusters: list[list[str]] = []
        for node_id in sorted(adjacency.keys(), key=_sort_key):
            if node_id in visited:
                continue
            stack = [node_id]
            cluster: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in sorted(adjacency.get(current, set()), key=_sort_key):
                    if neighbor not in visited:
                        stack.append(neighbor)
            cluster.sort(key=_sort_key)
            clusters.append(cluster)

        split_clusters = 0
        capped: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= max_cluster_size:
                capped.append(cluster)
                continue
            split_clusters += 1
            for start in range(0, len(cluster), max_cluster_size):
                capped.append(cluster[start : start + max_cluster_size])
        return capped, split_clusters

    score_values = [float(pair.get("similarity", 0.0)) for pair in pairs]
    if threshold_mode == "fixed":
        threshold_used = float(threshold)
        clusters, split_clusters = _clusters_for_threshold(threshold_used)
    elif threshold_mode == "quantile":
        threshold_used = _quantile(score_values, quantile_q)
        clusters, split_clusters = _clusters_for_threshold(threshold_used)
    else:
        species_count = len(species_entries)
        if threshold_mode == "target_clusters":
            if target_clusters is None:
                raise ConfigError("similarity.target_clusters is required for threshold_mode=target_clusters.")
            if target_clusters <= 0:
                raise ConfigError("similarity.target_clusters must be positive.")
            target_cluster_count = int(target_clusters)
        else:
            if target_merged_species is None:
                raise ConfigError(
                    "similarity.target_merged_species is required for threshold_mode=target_merged_species."
                )
            if target_merged_species < 0:
                raise ConfigError("similarity.target_merged_species must be >= 0.")
            desired = int(species_count) - int(target_merged_species)
            target_cluster_count = max(1, desired)

        # Search for a threshold whose connected-components count is near the target.
        best_threshold = float(threshold)
        best_clusters, best_split = _clusters_for_threshold(best_threshold)
        best_err = abs(len(best_clusters) - target_cluster_count)
        lo = 0.0
        hi = 1.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            mid_clusters, mid_split = _clusters_for_threshold(mid)
            mid_count = len(mid_clusters)
            mid_err = abs(mid_count - target_cluster_count)
            if mid_err < best_err or (mid_err == best_err and mid > best_threshold):
                best_threshold = mid
                best_clusters = mid_clusters
                best_split = mid_split
                best_err = mid_err
            if mid_count > target_cluster_count:
                hi = mid
            elif mid_count < target_cluster_count:
                lo = mid
            else:
                break

        threshold_used = float(best_threshold)
        clusters = best_clusters
        split_clusters = int(best_split)

    threshold_used = float(min(max(float(threshold_used), 0.0), 1.0))
    threshold_used = float(round(threshold_used, 12))
    edges_above_threshold = sum(1 for value in score_values if value >= threshold_used)

    degree_by_node_id = {
        entry["node_id"]: int(entry.get("degree", 0)) for entry in species_entries
    }
    entry_by_node_id = {entry["node_id"]: entry for entry in species_entries}

    superstates: list[dict[str, Any]] = []
    clusters_payload: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []

    for super_id, members in enumerate(clusters):
        if not members:
            continue
        if rep_metric == "degree":
            representative_node_id = sorted(
                members,
                key=lambda nid: (
                    -degree_by_node_id.get(nid, 0),
                    node_to_species[nid].lower(),
                    nid,
                ),
            )[0]
        else:
            representative_node_id = sorted(members, key=_sort_key)[0]
        representative = node_to_species[representative_node_id]
        name = f"S{super_id:03d}"
        member_species = [node_to_species[nid] for nid in members]
        kinds = {str(entry_by_node_id[nid].get("kind")) for nid in members if nid in entry_by_node_id}
        phases = {str(entry_by_node_id[nid].get("phase")) for nid in members if entry_by_node_id[nid].get("phase") is not None}
        element_union: set[str] = set()
        mw_values: list[float] = []
        for nid in members:
            entry = entry_by_node_id.get(nid, {})
            element_union |= set(entry.get("element_set") or set())
            mw = entry.get("mw")
            if isinstance(mw, (int, float)) and math.isfinite(float(mw)):
                mw_values.append(float(mw))
        summary = {
            "size": len(members),
            "kinds": sorted(kinds),
            "phases": sorted(phases),
            "elements": sorted(element_union),
        }
        if mw_values:
            summary["mw_min"] = min(mw_values)
            summary["mw_max"] = max(mw_values)

        cluster_entry = {
            "superstate_id": int(super_id),
            "name": name,
            "representative": representative,
            "members": member_species,
            "member_node_ids": list(members),
            "representative_node_id": representative_node_id,
            "summary": summary,
        }
        clusters_payload.append(cluster_entry)
        superstates.append(
            {
                "superstate_id": int(super_id),
                "name": name,
                "representative": representative,
                "members": member_species,
                "summary": summary,
            }
        )
        for nid in members:
            entry = entry_by_node_id.get(nid, {})
            mapping.append(
                {
                    "species": node_to_species[nid],
                    "species_index": entry.get("species_index"),
                    "superstate_id": int(super_id),
                    "representative": representative,
                }
            )

    mapping.sort(key=lambda row: (str(row.get("species", "")).lower(), str(row.get("species", ""))))

    composition_meta: list[dict[str, Any]] = []
    for entry in species_entries:
        composition_meta.append(
            {
                "species": entry["species"],
                "species_index": entry.get("species_index"),
                "node_id": entry["node_id"],
                "formula": entry.get("formula"),
                "elements": dict(entry.get("elements") or {}),
                "charge": entry.get("charge"),
                "kind": entry.get("kind"),
                "mw": entry.get("mw"),
                "phase": entry.get("phase"),
            }
        )

    coverage = 0.0
    if species_entries:
        mapped = {row.get("species") for row in mapping if isinstance(row.get("species"), str)}
        coverage = len(mapped) / float(len(species_entries))

    mapping_payload = {
        "schema_version": 1,
        "kind": "superstate_mapping",
        "source": {"graph_id": graph_id},
        # Provide both aliases for compatibility with existing consumers.
        "clusters": clusters_payload,
        "superstates": superstates,
        "mapping": mapping,
        "guards": {
            "require_element_overlap": require_element_overlap,
            "require_heavy_element_overlap": require_heavy_element_overlap,
            "require_same_kind": require_same_kind,
            "require_same_phase": require_same_phase,
            "protected_species": sorted(protected_species),
            "cannot_link": cannot_link_groups,
            "must_link": must_link_groups,
        },
        "similarity": {
            "threshold_mode": threshold_mode,
            "threshold": threshold_used,
            "threshold_config": float(threshold),
            "quantile_q": float(quantile_q),
            "target_clusters": target_clusters,
            "target_merged_species": target_merged_species,
            "max_iter": int(max_iter),
            "weights": weights,
            "charge_scale": charge_scale,
            "elements": {"order": element_names},
        },
        "soft_constraints": {
            "mw_log_tol": mw_log_tol,
            "elements_cosine_min": elements_cosine_min,
            "max_cluster_size": max_cluster_size,
        },
        "composition_meta": composition_meta,
        "meta": {
            "species_count": len(species_entries),
            "superstate_count": len(superstates),
            "coverage": coverage,
            "representative_metric": rep_metric,
            "split_clusters": split_clusters,
        },
    }

    inputs_payload = {
        "mode": "superstate_mapping",
        "graph_id": graph_id,
        "guards": mapping_payload["guards"],
        "similarity": {
            "threshold_mode": threshold_mode,
            "threshold": threshold_used,
            "threshold_config": float(threshold),
            "quantile_q": float(quantile_q),
            "target_clusters": target_clusters,
            "target_merged_species": target_merged_species,
            "max_iter": int(max_iter),
            "weights": weights,
            "charge_scale": charge_scale,
            "representative_metric": rep_metric,
        },
        "soft_constraints": mapping_payload["soft_constraints"],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    score_stats: dict[str, Any] = {"count": int(len(score_values))}
    if score_values:
        score_stats.update(
            {
                "min": float(min(score_values)),
                "p10": float(_quantile(score_values, 0.10)),
                "p50": float(_quantile(score_values, 0.50)),
                "p90": float(_quantile(score_values, 0.90)),
                "p95": float(_quantile(score_values, 0.95)),
                "max": float(max(score_values)),
            }
        )
    else:
        score_stats.update(
            {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None}
        )

    top_pairs_raw = sorted(
        pairs,
        key=lambda row: (
            -float(row.get("similarity", 0.0)),
            str(row.get("species_a", "")).lower(),
            str(row.get("species_b", "")).lower(),
        ),
    )[:20]
    top_pairs = [
        {
            "species_a": row.get("species_a"),
            "species_b": row.get("species_b"),
            "similarity": float(row.get("similarity", 0.0)),
            "components": row.get("components") or {},
        }
        for row in top_pairs_raw
        if isinstance(row, Mapping)
    ]

    cluster_sizes = [len(cluster) for cluster in clusters]
    metrics_payload = {
        "schema_version": 1,
        "kind": "superstate_mapping_metrics",
        "counts": {
            "species_before": len(species_entries),
            "species_after": len(superstates),
            "merged_species": max(0, len(species_entries) - len(superstates)),
            "reactions_before": reaction_count if reaction_count else None,
            "reactions_after": reaction_count if reaction_count else None,
            "disabled_reactions": 0,
            "merged_reactions": 0,
        },
        "skip_reasons": dict(skip_reasons),
        "score_stats": score_stats,
        "edges_above_threshold": int(edges_above_threshold),
        "top_pairs": top_pairs,
        "cluster_sizes": _cluster_size_stats(cluster_sizes),
        "similarity": mapping_payload["similarity"],
        "guards": mapping_payload["guards"],
        "soft_constraints": mapping_payload["soft_constraints"],
    }

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "mapping.json", mapping_payload)
        write_json_atomic(base_dir / "metrics.json", metrics_payload)

    return store.ensure(manifest, writer=_writer)


def propose_reaction_lumping(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Propose reaction lumping candidates within each reaction type."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    lumping_cfg = _extract_reaction_lumping_cfg(reduction_cfg)

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            lumping_cfg,
            keys=("graph", "graph_id", "graph_artifact"),
            label="lumping.graph",
        )
    if graph_id is None:
        raise ConfigError("reaction lumping requires a graph input.")

    threshold_value = lumping_cfg.get("threshold")
    if threshold_value is None:
        threshold_value = lumping_cfg.get("similarity_threshold") or lumping_cfg.get(
            "min_similarity"
        )
    threshold = _coerce_similarity_threshold(threshold_value)
    method = _coerce_similarity_method(
        lumping_cfg.get("method") or lumping_cfg.get("cluster_method")
    )

    similarity_cfg = lumping_cfg.get("similarity") or lumping_cfg.get("similarity_cfg")
    if similarity_cfg is None:
        similarity_cfg = {}
    if not isinstance(similarity_cfg, Mapping):
        raise ConfigError("lumping.similarity must be a mapping.")
    similarity_cfg = dict(similarity_cfg)

    metric = _coerce_reaction_similarity_metric(
        similarity_cfg.get("metric") or similarity_cfg.get("method")
    )
    mode = _coerce_reaction_similarity_mode(
        similarity_cfg.get("mode") or similarity_cfg.get("set")
    )
    weights = _extract_reaction_similarity_weights(similarity_cfg)
    include_participants = _coerce_reaction_include_participants(similarity_cfg)

    representative_cfg = lumping_cfg.get("representative") or {}
    if representative_cfg is None:
        representative_cfg = {}
    if not isinstance(representative_cfg, Mapping):
        raise ConfigError("lumping.representative must be a mapping.")
    rep_metric_raw = representative_cfg.get("metric") or representative_cfg.get(
        "score"
    ) or DEFAULT_REPRESENTATIVE_METRIC
    rep_metric = str(rep_metric_raw).strip().lower()
    if rep_metric in {"centrality", "degree_centrality"}:
        rep_metric = "degree"
    if rep_metric in {"lexical", "lexicographic"}:
        rep_metric = "name"
    if rep_metric not in {"degree", "name"}:
        raise ConfigError("lumping.representative.metric must be 'degree' or 'name'.")

    nodes, links = _load_graph_nodes_and_links(store, graph_id)
    reaction_entries = _prepare_reaction_entries(nodes, links)

    def _sort_key(entry: Mapping[str, Any]) -> tuple[str, str]:
        reaction_id = str(entry.get("reaction_id") or "")
        node_id = str(entry.get("node_id") or "")
        return (reaction_id.lower(), node_id)

    reaction_entries.sort(key=_sort_key)
    entry_by_id = {entry["node_id"]: entry for entry in reaction_entries}

    pairs: list[dict[str, Any]] = []
    cluster_payloads: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []

    type_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in reaction_entries:
        reaction_type = entry.get("reaction_type") or "unknown"
        type_groups.setdefault(str(reaction_type), []).append(entry)

    cluster_id = 0
    weights_sum = weights["reactants"] + weights["products"]
    for reaction_type, group_entries in sorted(type_groups.items()):
        group_entries.sort(key=_sort_key)
        adjacency: dict[str, set[str]] = {
            entry["node_id"]: set() for entry in group_entries
        }
        for idx, left in enumerate(group_entries):
            left_reactants = set(left["reactants"])
            left_products = set(left["products"])
            left_participants = set(left["participants"])
            if include_participants:
                left_reactants |= left_participants
                left_products |= left_participants
            left_union = left_reactants | left_products | (
                left_participants if include_participants else set()
            )
            for right in group_entries[idx + 1 :]:
                right_reactants = set(right["reactants"])
                right_products = set(right["products"])
                right_participants = set(right["participants"])
                if include_participants:
                    right_reactants |= right_participants
                    right_products |= right_participants
                right_union = right_reactants | right_products | (
                    right_participants if include_participants else set()
                )
                reactant_score = _jaccard_similarity(left_reactants, right_reactants)
                product_score = _jaccard_similarity(left_products, right_products)
                union_score = _jaccard_similarity(left_union, right_union)
                if mode == "reactants":
                    similarity = reactant_score
                elif mode == "products":
                    similarity = product_score
                elif mode == "union":
                    similarity = union_score
                else:
                    similarity = (
                        weights["reactants"] * reactant_score
                        + weights["products"] * product_score
                    ) / weights_sum
                pairs.append(
                    {
                        "reaction_type": reaction_type,
                        "node_id_a": left["node_id"],
                        "reaction_id_a": left["reaction_id"],
                        "node_id_b": right["node_id"],
                        "reaction_id_b": right["reaction_id"],
                        "similarity": similarity,
                        "components": {
                            "reactants": reactant_score,
                            "products": product_score,
                            "union": union_score,
                        },
                    }
                )
                if similarity >= threshold:
                    adjacency[left["node_id"]].add(right["node_id"])
                    adjacency[right["node_id"]].add(left["node_id"])

        visited: set[str] = set()
        for entry in group_entries:
            node_id = entry["node_id"]
            if node_id in visited:
                continue
            stack = [node_id]
            members: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                members.append(current)
                for neighbor in sorted(adjacency.get(current, set())):
                    if neighbor not in visited:
                        stack.append(neighbor)
            members.sort(key=lambda node_id: _sort_key(entry_by_id[node_id]))
            scores: dict[str, float] = {}
            for member_id in members:
                member_entry = entry_by_id[member_id]
                if rep_metric == "degree":
                    score = float(member_entry.get("degree", 0))
                else:
                    score = 0.0
                scores[member_id] = score
            representative = sorted(
                members,
                key=lambda node_id: (
                    -scores.get(node_id, 0.0),
                    str(entry_by_id[node_id]["reaction_id"]).lower(),
                    node_id,
                ),
            )[0]
            cluster_payloads.append(
                {
                    "cluster_id": cluster_id,
                    "reaction_type": reaction_type,
                    "members": [
                        entry_by_id[node_id]["reaction_id"] for node_id in members
                    ],
                    "member_node_ids": list(members),
                    "representative": entry_by_id[representative]["reaction_id"],
                    "representative_node_id": representative,
                    "selection": {
                        "metric": rep_metric,
                        "scores": {
                            entry_by_id[node_id]["reaction_id"]: scores[node_id]
                            for node_id in members
                        },
                        "reason": "max_score",
                    },
                    "summary": {"size": len(members)},
                }
            )
            for node_id in members:
                entry = entry_by_id[node_id]
                mapping.append(
                    {
                        "node_id": node_id,
                        "reaction_id": entry["reaction_id"],
                        "reaction_index": entry["reaction_index"],
                        "reaction_type": entry["reaction_type"],
                        "representative_node_id": representative,
                        "representative": entry_by_id[representative]["reaction_id"],
                        "cluster_id": cluster_id,
                    }
                )
            cluster_id += 1

    mapping.sort(
        key=lambda entry: (
            str(entry["reaction_id"]).lower(),
            str(entry["node_id"]),
        )
    )

    reaction_payload = [
        {
            "node_id": entry["node_id"],
            "reaction_id": entry["reaction_id"],
            "reaction_index": entry["reaction_index"],
            "reaction_type": entry["reaction_type"],
            "reaction_equation": entry["reaction_equation"],
            "reactants": sorted(entry["reactants"]),
            "products": sorted(entry["products"]),
            "participants": sorted(entry["participants"]),
            "degree": entry["degree"],
        }
        for entry in reaction_entries
    ]

    type_summary = {
        reaction_type: len(entries) for reaction_type, entries in type_groups.items()
    }

    payload = {
        "schema_version": REACTION_LUMPING_SCHEMA_VERSION,
        "kind": "reaction_lumping",
        "source": {"graph_id": graph_id},
        "similarity": {
            "method": metric,
            "mode": mode,
            "threshold": threshold,
            "weights": weights,
            "include_participants": include_participants,
            "cluster_method": method,
            "pairs": pairs,
        },
        "clusters": cluster_payloads,
        "mapping": mapping,
        "reactions": reaction_payload,
        "meta": {
            "reaction_count": len(reaction_entries),
            "cluster_count": len(cluster_payloads),
            "reaction_types": type_summary,
            "selection_metric": rep_metric,
        },
    }

    inputs_payload = {
        "mode": "reaction_lumping",
        "graph": graph_id,
        "similarity": {
            "metric": metric,
            "mode": mode,
            "threshold": threshold,
            "weights": weights,
            "include_participants": include_participants,
            "cluster_method": method,
        },
        "representative": {"metric": rep_metric},
    }

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / REACTION_LUMPING_FILENAME, payload)
        cluster_sizes = [
            int(entry.get("summary", {}).get("size", 0))
            for entry in cluster_payloads
            if isinstance(entry, Mapping)
        ]
        metrics = {
            "schema_version": 1,
            "kind": "reaction_lumping_metrics",
            "reaction_count": len(reaction_entries),
            "cluster_count": len(cluster_payloads),
            "cluster_sizes": _cluster_size_stats(cluster_sizes),
            "similarity": {
                "method": method,
                "threshold": threshold,
                "metric": metric,
                "mode": mode,
                "weights": weights,
                "include_participants": include_participants,
            },
            "representative_metric": rep_metric,
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def reaction_lumping_prune(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Convert a reaction_lumping artifact into a disable-reaction patch.

    This is a pragmatic "reaction merge" reduction: for each reaction cluster, keep
    the representative and disable all other member reactions. The output is a
    standard `mechanism_patch.yaml` so existing QoI validation can be reused.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    mechanism_path = _extract_mechanism(reduction_cfg)

    lumping_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=(
            "reaction_lumping",
            "reaction_lumping_id",
            "lumping",
            "lumping_id",
            "mapping_id",
        ),
        label="reduction.reaction_lumping_id",
    )
    if lumping_id is None:
        raise ConfigError("reaction_lumping_prune requires a reaction_lumping artifact id.")

    store.read_manifest("reduction", lumping_id)
    lumping_path = store.artifact_dir("reduction", lumping_id) / REACTION_LUMPING_FILENAME
    payload = read_json(lumping_path)
    if not isinstance(payload, Mapping):
        raise ConfigError("reaction_lumping payload must be a mapping.")

    clusters = payload.get("clusters") or []
    mapping = payload.get("mapping") or []
    if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes, bytearray)):
        raise ConfigError("reaction_lumping.clusters must be a sequence.")
    if not isinstance(mapping, Sequence) or isinstance(mapping, (str, bytes, bytearray)):
        raise ConfigError("reaction_lumping.mapping must be a sequence.")

    index_by_node_id: dict[str, int] = {}
    for entry in mapping:
        if not isinstance(entry, Mapping):
            continue
        node_id = entry.get("node_id")
        idx = entry.get("reaction_index")
        if not isinstance(node_id, str) or not node_id:
            continue
        if isinstance(idx, bool) or idx is None:
            continue
        try:
            index_by_node_id[node_id] = int(idx)
        except (TypeError, ValueError):
            continue

    rep_indices: set[int] = set()
    member_indices: set[int] = set()
    cluster_sizes: list[int] = []
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        members = cluster.get("member_node_ids") or []
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes, bytearray)):
            cluster_sizes.append(len(members))
            for node_id in members:
                if isinstance(node_id, str) and node_id in index_by_node_id:
                    member_indices.add(index_by_node_id[node_id])
        rep_node_id = cluster.get("representative_node_id")
        if isinstance(rep_node_id, str) and rep_node_id in index_by_node_id:
            rep_indices.add(index_by_node_id[rep_node_id])

    disabled = sorted(idx for idx in member_indices if idx not in rep_indices)
    patch_payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "disabled_reactions": [{"index": int(idx)} for idx in disabled],
        # Keep the patch structurally non-empty even if disabled is empty.
        "reaction_multipliers": [{"index": 0, "multiplier": 1.0}],
    }
    normalized_patch, combined_entries = _normalize_patch_payload(patch_payload)

    compiler = MechanismCompiler.from_path(mechanism_path)
    reduced_payload, _ = compiler.apply_patch_entries(combined_entries)

    inputs_payload = {
        "mode": "reaction_lumping_prune",
        "mechanism": mechanism_path,
        "reaction_lumping_id": lumping_id,
        "keep_representatives": True,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[lumping_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, normalized_patch, sort_keys=True)
        write_yaml_payload(base_dir / MECHANISM_FILENAME, reduced_payload, sort_keys=False)
        species_before = None
        species_raw = compiler.payload.get("species")
        if isinstance(species_raw, Sequence) and not isinstance(
            species_raw, (str, bytes, bytearray)
        ):
            species_before = len(species_raw)

        reactions_after = _count_reactions(reduced_payload)
        metrics = {
            "schema_version": REACTION_LUMPING_PRUNE_SCHEMA_VERSION,
            "kind": "reaction_lumping_prune",
            "reaction_lumping_id": lumping_id,
            "counts": {
                "species_before": species_before,
                "species_after": species_before,
                "merged_species": 0,
                "reactions_before": compiler.reaction_count(),
                "reactions_after": reactions_after,
                "disabled_reactions": len(disabled),
                "merged_reactions": 0,
            },
            "cluster_sizes": _cluster_size_stats(cluster_sizes),
            "kept_reaction_indices": sorted(rep_indices),
            "disabled_reaction_indices": disabled,
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def node_lumping_prune(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Convert a node_lumping artifact into a reduced Cantera mechanism.

    This implements an executable "state merge" reduction by merging species
    into their cluster representative.

    Safety defaults (to keep reduced simulation and QoI comparable without
    redefining QoIs on superstates):
    - Only merge when elemental composition and charge match the representative.
    - Never merge away protected species (default: CO, CO2).
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    if ct is None:  # pragma: no cover - optional dependency
        raise ConfigError("node_lumping_prune requires cantera to be installed.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    mechanism_path = _extract_mechanism(reduction_cfg)

    lumping_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=(
            "node_lumping",
            "node_lumping_id",
            "lumping",
            "lumping_id",
            "mapping_id",
        ),
        label="reduction.node_lumping_id",
    )
    if lumping_id is None:
        raise ConfigError("node_lumping_prune requires a node_lumping artifact id.")

    params = _extract_params(reduction_cfg)
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ConfigError("reduction.params must be a mapping.")

    merge_constraints = params.get("merge_constraints") or params.get("constraints")
    if merge_constraints is None:
        merge_constraints = {}
    if not isinstance(merge_constraints, Mapping):
        raise ConfigError("params.merge_constraints must be a mapping when provided.")
    merge_constraints = dict(merge_constraints)

    # Backward compatible: legacy flat params are supported, but merge_constraints wins.
    require_same_composition = bool(
        merge_constraints.get(
            "require_same_composition",
            params.get("require_same_composition", True),
        )
    )
    require_same_charge = bool(
        merge_constraints.get(
            "require_same_charge",
            params.get("require_same_charge", True),
        )
    )
    require_same_phase = bool(
        merge_constraints.get(
            "require_same_phase",
            params.get("require_same_phase", True),
        )
    )
    require_same_kind = bool(
        merge_constraints.get(
            "require_same_kind",
            params.get("require_same_kind", False),
        )
    )

    if not (require_same_composition or require_same_charge or require_same_phase):
        raise ConfigError(
            "merge_constraints disables all safety constraints; use projection-only instead."
        )
    cache_bust = params.get("cache_bust")

    thermo_constraints_raw = params.get("thermo_constraints")
    if thermo_constraints_raw is None:
        thermo_constraints = {}
    elif not isinstance(thermo_constraints_raw, Mapping):
        raise ConfigError("params.thermo_constraints must be a mapping when provided.")
    else:
        thermo_constraints = dict(thermo_constraints_raw)

    kinetics_constraints_raw = params.get("kinetics_constraints")
    if kinetics_constraints_raw is None:
        kinetics_constraints = {}
    elif not isinstance(kinetics_constraints_raw, Mapping):
        raise ConfigError("params.kinetics_constraints must be a mapping when provided.")
    else:
        kinetics_constraints = dict(kinetics_constraints_raw)

    thermo_enabled = bool(thermo_constraints.get("enabled", False))
    thermo_missing_strategy = str(thermo_constraints.get("missing_strategy", "skip")).strip().lower()
    if thermo_missing_strategy not in {"skip", "fail"}:
        raise ConfigError("thermo_constraints.missing_strategy must be 'skip' or 'fail'.")
    thermo_t_grid_raw = thermo_constraints.get("T_grid")
    if thermo_t_grid_raw is None:
        thermo_t_grid = [600.0, 1000.0, 1500.0, 2000.0]
    elif isinstance(thermo_t_grid_raw, Sequence) and not isinstance(
        thermo_t_grid_raw, (str, bytes, bytearray)
    ):
        thermo_t_grid = []
        for item in thermo_t_grid_raw:
            if item is None or isinstance(item, bool):
                continue
            try:
                thermo_t_grid.append(float(item))
            except (TypeError, ValueError):
                raise ConfigError("thermo_constraints.T_grid must be a list of numbers.")
        if not thermo_t_grid:
            raise ConfigError("thermo_constraints.T_grid must be non-empty when provided.")
    else:
        raise ConfigError("thermo_constraints.T_grid must be a list of numbers when provided.")
    try:
        thermo_p_ref = float(thermo_constraints.get("P_ref", 101325.0))
    except (TypeError, ValueError):
        raise ConfigError("thermo_constraints.P_ref must be a number.")
    try:
        thermo_max_rel_cp = float(thermo_constraints.get("max_rel_cp", 0.25))
        thermo_max_rel_h = float(thermo_constraints.get("max_rel_h", 0.25))
        thermo_max_rel_s = float(thermo_constraints.get("max_rel_s", 0.25))
    except (TypeError, ValueError):
        raise ConfigError("thermo_constraints max_rel_* values must be numbers.")

    kinetics_enabled = bool(kinetics_constraints.get("enabled", False))
    kinetics_missing_strategy = str(kinetics_constraints.get("missing_strategy", "skip")).strip().lower()
    if kinetics_missing_strategy not in {"skip", "fail"}:
        raise ConfigError("kinetics_constraints.missing_strategy must be 'skip' or 'fail'.")
    kinetics_signature = str(
        kinetics_constraints.get("species_signature", "adjacent_logk_mean")
    ).strip()
    if kinetics_signature != "adjacent_logk_mean":
        raise ConfigError("kinetics_constraints.species_signature must be 'adjacent_logk_mean'.")
    kinetics_t_grid_raw = kinetics_constraints.get("T_grid")
    if kinetics_t_grid_raw is None:
        kinetics_t_grid = [600.0, 1000.0, 1500.0, 2000.0]
    elif isinstance(kinetics_t_grid_raw, Sequence) and not isinstance(
        kinetics_t_grid_raw, (str, bytes, bytearray)
    ):
        kinetics_t_grid = []
        for item in kinetics_t_grid_raw:
            if item is None or isinstance(item, bool):
                continue
            try:
                kinetics_t_grid.append(float(item))
            except (TypeError, ValueError):
                raise ConfigError("kinetics_constraints.T_grid must be a list of numbers.")
        if not kinetics_t_grid:
            raise ConfigError("kinetics_constraints.T_grid must be non-empty when provided.")
    else:
        raise ConfigError("kinetics_constraints.T_grid must be a list of numbers when provided.")
    try:
        kinetics_p_ref = float(kinetics_constraints.get("P_ref", 101325.0))
    except (TypeError, ValueError):
        raise ConfigError("kinetics_constraints.P_ref must be a number.")
    kinetics_x_ref_raw = kinetics_constraints.get("X_ref", "N2:0.79, O2:0.21")
    kinetics_x_ref = str(kinetics_x_ref_raw).strip()
    try:
        kinetics_eps = float(kinetics_constraints.get("eps", 1.0e-300))
    except (TypeError, ValueError):
        raise ConfigError("kinetics_constraints.eps must be a number.")
    try:
        kinetics_max_abs_log10k_diff = float(kinetics_constraints.get("max_abs_log10k_diff", 1.0))
    except (TypeError, ValueError):
        raise ConfigError("kinetics_constraints.max_abs_log10k_diff must be a number.")
    try:
        kinetics_min_adj_size = int(kinetics_constraints.get("min_adj_size", 2))
    except (TypeError, ValueError):
        raise ConfigError("kinetics_constraints.min_adj_size must be an integer.")
    if kinetics_min_adj_size < 1:
        raise ConfigError("kinetics_constraints.min_adj_size must be >= 1.")

    merge_mode_raw = params.get("merge_mode", merge_constraints.get("merge_mode"))
    if merge_mode_raw is None:
        merge_mode = "signature_split"
    else:
        merge_mode = str(merge_mode_raw).strip().lower()
        if merge_mode in {"signature_split", "signature", "split"}:
            merge_mode = "signature_split"
        elif merge_mode in {
            "cluster_representative",
            "cluster",
            "representative",
            "cluster_rep",
        }:
            merge_mode = "cluster_representative"
        else:
            raise ConfigError(
                "params.merge_mode must be 'signature_split' or 'cluster_representative'."
            )
    protected_raw = params.get("protected_species")
    if protected_raw is None:
        protected = {"CO", "CO2"}
    elif isinstance(protected_raw, Sequence) and not isinstance(
        protected_raw, (str, bytes, bytearray)
    ):
        protected = {str(item).strip() for item in protected_raw if str(item).strip()}
    else:
        protected = {str(protected_raw).strip()} if str(protected_raw).strip() else set()

    store.read_manifest("reduction", lumping_id)
    lumping_path = store.artifact_dir("reduction", lumping_id) / NODE_LUMPING_FILENAME
    payload = read_json(lumping_path)
    if not isinstance(payload, Mapping):
        raise ConfigError("node_lumping payload must be a mapping.")

    mapping_entries = payload.get("mapping") or []
    clusters = payload.get("clusters") or []
    if not isinstance(mapping_entries, Sequence) or isinstance(
        mapping_entries, (str, bytes, bytearray)
    ):
        raise ConfigError("node_lumping.mapping must be a sequence.")
    if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes, bytearray)):
        raise ConfigError("node_lumping.clusters must be a sequence.")

    phase_by_species: dict[str, Optional[str]] = {}
    degree_by_species: dict[str, int] = {}
    species_meta = payload.get("species") or []
    if isinstance(species_meta, Sequence) and not isinstance(
        species_meta, (str, bytes, bytearray)
    ):
        for entry in species_meta:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("species")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            phase_by_species[name] = _normalize_text(entry.get("phase"))
            degree = entry.get("degree")
            if degree is None or isinstance(degree, bool):
                continue
            try:
                degree_by_species[name] = int(degree)
            except (TypeError, ValueError):
                continue

    base = ct.Solution(mechanism_path)
    base_species_names = [sp.name for sp in base.species()]
    base_species_set = set(base_species_names)

    kind_by_species: dict[str, str] = {}
    if require_same_kind:
        element_z: dict[str, int] = {}
        for element in base.element_names:
            try:
                element_z[str(element)] = int(ct.Element(str(element)).atomic_number)
            except Exception:
                continue

        def _kind(name: str) -> str:
            try:
                sp = base.species(name)
            except Exception:
                return "unknown"
            try:
                charge_value = float(sp.charge)
            except Exception:
                charge_value = 0.0
            if abs(charge_value) > 1.0e-12:
                return "ion"
            comp = dict(sp.composition)
            atoms = 0.0
            for count in comp.values():
                try:
                    atoms += float(count)
                except Exception:
                    continue
            if abs(atoms - 1.0) < 1.0e-12:
                return "atom"
            electrons = 0.0
            for element, count in comp.items():
                z = element_z.get(str(element))
                if z is None:
                    return "unknown"
                electrons += float(z) * float(count)
            electrons -= charge_value
            if int(round(abs(electrons))) % 2 == 1:
                return "radical"
            return "molecule"

        for name in base_species_names:
            kind_by_species[name] = _kind(name)

    thermo_cache: dict[str, Any] = {
        "enabled": bool(thermo_enabled),
        "T_grid": list(thermo_t_grid),
        "P_ref": float(thermo_p_ref),
        "max_rel_cp": float(thermo_max_rel_cp),
        "max_rel_h": float(thermo_max_rel_h),
        "max_rel_s": float(thermo_max_rel_s),
        "missing_strategy": thermo_missing_strategy,
    }
    thermo_by_T: dict[float, tuple[Any, Any, Any]] = {}
    if thermo_enabled:
        for T in thermo_t_grid:
            base.TP = float(T), float(thermo_p_ref)
            thermo_by_T[float(T)] = (
                base.standard_cp_R,
                base.standard_enthalpies_RT,
                base.standard_entropies_R,
            )

    kinetics_cache: dict[str, Any] = {
        "enabled": bool(kinetics_enabled),
        "T_grid": list(kinetics_t_grid),
        "P_ref": float(kinetics_p_ref),
        "X_ref": str(kinetics_x_ref),
        "eps": float(kinetics_eps),
        "species_signature": kinetics_signature,
        "max_abs_log10k_diff": float(kinetics_max_abs_log10k_diff),
        "min_adj_size": int(kinetics_min_adj_size),
        "missing_strategy": kinetics_missing_strategy,
    }
    x_ref_used = kinetics_x_ref
    adj_reactant: dict[str, list[int]] = {}
    adj_product: dict[str, list[int]] = {}
    logk_by_T: list[list[float]] = []
    if kinetics_enabled:
        # Build adjacency lists from the base mechanism.
        for r_idx in range(base.n_reactions):
            rxn = base.reaction(r_idx)
            try:
                reactants = dict(rxn.reactants)
            except Exception:
                reactants = {}
            try:
                products = dict(rxn.products)
            except Exception:
                products = {}
            for name in reactants.keys():
                if name in base_species_set:
                    adj_reactant.setdefault(str(name), []).append(int(r_idx))
            for name in products.keys():
                if name in base_species_set:
                    adj_product.setdefault(str(name), []).append(int(r_idx))

        # Resolve a usable reference composition.
        def _try_set_X(value: str) -> bool:
            try:
                base.X = value
            except Exception:
                return False
            return True

        if not _try_set_X(x_ref_used):
            fallback = None
            for candidate in ("N2:1.0", "AR:1.0"):
                if _try_set_X(candidate):
                    fallback = candidate
                    break
            if fallback is None and base_species_names:
                candidate = f"{base_species_names[0]}:1.0"
                if _try_set_X(candidate):
                    fallback = candidate
            if fallback is None:
                raise ConfigError(
                    "kinetics_constraints.X_ref could not be applied and no fallback composition was usable."
                )
            x_ref_used = fallback
        kinetics_cache["X_ref_used"] = x_ref_used

        for T in kinetics_t_grid:
            base.TP = float(T), float(kinetics_p_ref)
            # base.X already set above (to X_ref_used).
            ks = base.forward_rate_constants
            logk = []
            for value in ks:
                try:
                    k = float(value)
                except Exception:
                    k = 0.0
                k = k + float(kinetics_eps)
                if k <= 0.0:
                    k = float(kinetics_eps)
                logk.append(float(math.log10(k)))
            logk_by_T.append(logk)

    def _thermo_distance(species: str, rep: str) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
        if not thermo_enabled:
            return None, None, None, None
        try:
            s_idx = int(base.species_index(species))
            r_idx = int(base.species_index(rep))
        except Exception:
            if thermo_missing_strategy == "fail":
                raise ConfigError(f"thermo_constraints missing species indices for {species!r}->{rep!r}.")
            return None, None, None, "thermo_missing"
        eps = 1.0e-300
        rel_cp = 0.0
        rel_h = 0.0
        rel_s = 0.0
        for T in thermo_t_grid:
            cp_R, h_RT, s_R = thermo_by_T.get(float(T), (None, None, None))
            if cp_R is None:
                continue
            try:
                cp_a = float(cp_R[s_idx])
                cp_r = float(cp_R[r_idx])
                h_a = float(h_RT[s_idx])
                h_r = float(h_RT[r_idx])
                s_a = float(s_R[s_idx])
                s_r = float(s_R[r_idx])
            except Exception:
                if thermo_missing_strategy == "fail":
                    raise ConfigError(
                        f"thermo_constraints could not extract cp/h/s for {species!r}->{rep!r}."
                    )
                return None, None, None, "thermo_missing"
            rel_cp = max(rel_cp, abs(cp_a - cp_r) / max(abs(cp_r), eps))
            rel_h = max(rel_h, abs(h_a - h_r) / max(abs(h_r), eps))
            rel_s = max(rel_s, abs(s_a - s_r) / max(abs(s_r), eps))
        return float(rel_cp), float(rel_h), float(rel_s), None

    sig_reactant: dict[str, Optional[list[float]]] = {}
    sig_product: dict[str, Optional[list[float]]] = {}
    adj_sizes: dict[str, dict[str, int]] = {"reactant": {}, "product": {}}
    if kinetics_enabled:
        for name in base_species_names:
            r_adj = adj_reactant.get(name, [])
            p_adj = adj_product.get(name, [])
            adj_sizes["reactant"][name] = int(len(r_adj))
            adj_sizes["product"][name] = int(len(p_adj))

            if len(r_adj) >= kinetics_min_adj_size:
                vals = []
                for logk in logk_by_T:
                    vals.append(float(sum(logk[idx] for idx in r_adj) / len(r_adj)))
                sig_reactant[name] = vals
            else:
                sig_reactant[name] = None

            if len(p_adj) >= kinetics_min_adj_size:
                vals = []
                for logk in logk_by_T:
                    vals.append(float(sum(logk[idx] for idx in p_adj) / len(p_adj)))
                sig_product[name] = vals
            else:
                sig_product[name] = None

    def _kinetics_distance(species: str, rep: str) -> tuple[Optional[float], dict[str, int], Optional[str]]:
        if not kinetics_enabled:
            return None, {"reactant": 0, "product": 0}, None
        s_r = sig_reactant.get(species)
        r_r = sig_reactant.get(rep)
        s_p = sig_product.get(species)
        r_p = sig_product.get(rep)
        sizes = {
            "reactant": int(adj_sizes["reactant"].get(species, 0)),
            "product": int(adj_sizes["product"].get(species, 0)),
        }
        diffs: list[float] = []
        if s_r is not None and r_r is not None:
            diffs.append(max(abs(a - b) for a, b in zip(s_r, r_r)))
        if s_p is not None and r_p is not None:
            diffs.append(max(abs(a - b) for a, b in zip(s_p, r_p)))
        if not diffs:
            if kinetics_missing_strategy == "fail":
                raise ConfigError(f"kinetics_constraints missing signatures for {species!r}->{rep!r}.")
            return None, sizes, "kinetics_missing"
        return float(max(diffs)), sizes, None

    def _composition(name: str) -> dict[str, float]:
        return dict(base.species(name).composition)

    def _charge(name: str) -> float:
        return float(base.species(name).charge)

    def _composition_key(name: str) -> tuple[tuple[str, float], ...]:
        comp = _composition(name)
        items: list[tuple[str, float]] = []
        for element, count in comp.items():
            value = float(count)
            if abs(value - round(value)) < 1.0e-12:
                value = float(int(round(value)))
            else:
                value = float(round(value, 12))
            if abs(value) < 1.0e-15:
                continue
            items.append((str(element), value))
        items.sort(key=lambda pair: pair[0])
        return tuple(items)

    def _canonical_charge_key(value: float) -> Any:
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1.0e-9:
            return int(rounded)
        return float(round(float(value), 12))

    signature_groups_total = 0
    signature_groups_merged = 0
    signature_groups_trace: list[dict[str, Any]] = []

    proposed: dict[str, str] = {}
    if merge_mode == "cluster_representative":
        for entry in mapping_entries:
            if not isinstance(entry, Mapping):
                continue
            species = entry.get("species")
            rep = entry.get("representative")
            if not isinstance(species, str) or not species.strip():
                continue
            if not isinstance(rep, str) or not rep.strip():
                continue
            species = species.strip()
            rep = rep.strip()
            if species == rep:
                continue
            proposed[species] = rep
    else:
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            members_raw = cluster.get("members") or []
            if not isinstance(members_raw, Sequence) or isinstance(
                members_raw, (str, bytes, bytearray)
            ):
                continue
            cluster_id = cluster.get("cluster_id")
            members = [
                str(item).strip()
                for item in members_raw
                if isinstance(item, str) and item.strip()
            ]
            if not members:
                continue

            groups: dict[tuple[Any, ...], list[str]] = {}
            for name in members:
                if name in protected:
                    groups.setdefault(("protected", name), []).append(name)
                    continue
                if name not in base_species_set:
                    groups.setdefault(("missing_species", name), []).append(name)
                    continue
                phase_key = phase_by_species.get(name) if require_same_phase else None
                charge_key = (
                    _canonical_charge_key(_charge(name)) if require_same_charge else None
                )
                comp_key = _composition_key(name) if require_same_composition else None
                groups.setdefault((phase_key, charge_key, comp_key), []).append(name)

            signature_groups_total += len(groups)
            for signature, group_members in groups.items():
                if len(group_members) <= 1:
                    continue
                # Merge only into an existing representative species.
                candidates = [
                    name
                    for name in group_members
                    if name in base_species_set and name not in protected
                ]
                if not candidates:
                    continue
                representative = sorted(
                    candidates,
                    key=lambda name: (
                        -int(degree_by_species.get(name, 0)),
                        name.lower(),
                        name,
                    ),
                )[0]
                merges = [
                    name
                    for name in group_members
                    if name != representative and name in base_species_set and name not in protected
                ]
                if merges:
                    signature_groups_merged += 1
                    signature_payload: dict[str, Any] = {}
                    phase_key, charge_key, comp_key = signature
                    if require_same_phase:
                        signature_payload["phase"] = phase_key
                    if require_same_charge:
                        signature_payload["charge"] = charge_key
                    if require_same_composition:
                        signature_payload["composition"] = {
                            element: count for element, count in (comp_key or ())
                        }
                    signature_groups_trace.append(
                        {
                            "cluster_id": int(cluster_id)
                            if isinstance(cluster_id, int) and not isinstance(cluster_id, bool)
                            else None,
                            "signature": signature_payload,
                            "representative": representative,
                            "members": sorted(
                                {name for name in group_members if name in base_species_set},
                                key=lambda value: value.lower(),
                            ),
                        }
                    )
                for name in merges:
                    proposed[name] = representative

    # Resolve representative chains (just in case).
    def _final_rep(name: str) -> str:
        seen: set[str] = set()
        cur = name
        while cur in proposed and cur not in seen:
            seen.add(cur)
            cur = proposed[cur]
        return cur

    for key in list(proposed.keys()):
        proposed[key] = _final_rep(proposed[key])

    accepted: dict[str, str] = {}
    skipped: dict[str, str] = {}
    accepted_details: list[dict[str, Any]] = []
    merge_eval_by_species: dict[str, dict[str, Any]] = {}
    for species, rep in sorted(proposed.items()):
        if species in protected:
            skipped[species] = "protected_species"
            continue
        if species not in base_species_set:
            skipped[species] = "missing_species"
            continue
        if rep not in base_species_set:
            skipped[species] = "missing_representative"
            continue
        if require_same_kind:
            sp_kind = kind_by_species.get(species, "unknown")
            rep_kind = kind_by_species.get(rep, "unknown")
            if sp_kind != rep_kind:
                skipped[species] = "kind_mismatch"
                continue
        if require_same_composition and _composition(species) != _composition(rep):
            skipped[species] = "composition_mismatch"
            continue
        if require_same_charge and abs(_charge(species) - _charge(rep)) > 1e-9:
            skipped[species] = "charge_mismatch"
            continue
        if require_same_phase:
            sp_phase = phase_by_species.get(species)
            rep_phase = phase_by_species.get(rep)
            if sp_phase and rep_phase and sp_phase != rep_phase:
                skipped[species] = "phase_mismatch"
                continue
        rel_cp, rel_h, rel_s, thermo_reason = _thermo_distance(species, rep)
        logk_diff, sizes, kinetics_reason = _kinetics_distance(species, rep)
        eval_detail = {
            "species": species,
            "representative": rep,
            "thermo": {
                "rel_cp": rel_cp,
                "rel_h": rel_h,
                "rel_s": rel_s,
            },
            "kinetics": {
                "log10k_diff": logk_diff,
                "adj_sizes": sizes,
            },
        }
        merge_eval_by_species[species] = eval_detail
        if thermo_reason is not None:
            skipped[species] = thermo_reason
            continue
        if thermo_enabled:
            assert rel_cp is not None and rel_h is not None and rel_s is not None
            if rel_cp > thermo_max_rel_cp or rel_h > thermo_max_rel_h or rel_s > thermo_max_rel_s:
                skipped[species] = "thermo_mismatch"
                continue
        if kinetics_reason is not None:
            skipped[species] = kinetics_reason
            continue
        if kinetics_enabled:
            assert logk_diff is not None
            if logk_diff > kinetics_max_abs_log10k_diff:
                skipped[species] = "kinetics_mismatch"
                continue
        accepted[species] = rep
        accepted_details.append(
            {
                "species": species,
                "representative": rep,
                "rel_cp": rel_cp,
                "rel_h": rel_h,
                "rel_s": rel_s,
                "logk_diff": logk_diff,
                "adj_sizes": sizes,
            }
        )

    mapped_away = set(accepted.keys())
    keep_species_names = [name for name in base_species_names if name not in mapped_away]
    keep_species_set = set(keep_species_names)

    def _format_coeff(value: float) -> str:
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.12g}"

    def _apply_mapping_to_equation(equation: str) -> str:
        arrow = None
        for token in ("<=>", "=>"):
            if token in equation:
                arrow = token
                break
        if arrow is None:
            return equation
        left_raw, right_raw = equation.split(arrow, 1)
        left_raw = left_raw.strip()
        right_raw = right_raw.strip()

        def _parse_side(side: str) -> tuple[dict[tuple[str, bool], float], float]:
            terms = [t.strip() for t in side.split(" + ") if t.strip()]
            counts: dict[tuple[str, bool], float] = {}
            m_coeff = 0.0
            for term in terms:
                if term == "M":
                    m_coeff += 1.0
                    continue
                has_suffix = term.endswith("(+M)")
                core = term[:-4].strip() if has_suffix else term
                coef = 1.0
                species = core
                match = re.match(
                    r"^([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+(.+)$",
                    core,
                )
                if match:
                    coef = float(match.group(1))
                    species = match.group(2).strip()
                mapped = accepted.get(species, species)
                key = (mapped, has_suffix)
                counts[key] = counts.get(key, 0.0) + coef
            return counts, m_coeff

        def _render_side(counts: dict[tuple[str, bool], float], m_coeff: float) -> str:
            parts: list[str] = []
            if m_coeff:
                parts.append("M")
            merged: dict[str, tuple[float, bool]] = {}
            for (name, has_suffix), coef in counts.items():
                if name == "M":
                    continue
                prev = merged.get(name)
                if prev is None:
                    merged[name] = (coef, has_suffix)
                else:
                    merged[name] = (prev[0] + coef, prev[1] or has_suffix)
            for name in sorted(merged.keys(), key=lambda x: x.lower()):
                coef, has_suffix = merged[name]
                if abs(coef) < 1e-15:
                    continue
                if abs(coef - 1.0) < 1e-12:
                    term = name
                else:
                    term = f"{_format_coeff(coef)} {name}"
                if has_suffix:
                    term = f"{term} (+M)"
                parts.append(term)
            return " + ".join(parts) if parts else ""

        left_counts, left_m = _parse_side(left_raw)
        right_counts, right_m = _parse_side(right_raw)
        left = _render_side(left_counts, left_m)
        right = _render_side(right_counts, right_m)
        return f"{left} {arrow} {right}".strip()

    species_objs = [
        ct.Species.from_dict(dict(base.species(name).input_data))
        for name in keep_species_names
    ]

    reaction_dicts: list[dict[str, Any]] = []
    for i in range(base.n_reactions):
        r = base.reaction(i)
        d = dict(r.input_data)
        eq = d.get("equation")
        if isinstance(eq, str) and eq.strip():
            d["equation"] = _apply_mapping_to_equation(eq.strip())

        def _rewrite_species_keyed_map(key: str) -> None:
            raw = d.get(key)
            if raw is None or not isinstance(raw, Mapping):
                return
            merged: dict[str, Any] = {}
            for k, v in raw.items():
                name = str(k)
                mapped = accepted.get(name, name)
                if mapped not in keep_species_set:
                    continue
                if mapped in merged:
                    try:
                        merged[mapped] = max(float(merged[mapped]), float(v))
                    except Exception:
                        merged[mapped] = v
                else:
                    merged[mapped] = v
            d[key] = merged

        _rewrite_species_keyed_map("efficiencies")
        _rewrite_species_keyed_map("orders")
        _rewrite_species_keyed_map("negative-orders")
        _rewrite_species_keyed_map("coverage-dependencies")

        reaction_dicts.append(d)

    def _canonical_equation_key(equation: str) -> Any:
        """Directionless stoichiometry key for duplicate detection.

        Cantera flags duplicates even if one equation is written in reverse,
        so we canonicalize by sorting both sides and then sorting the pair.
        """
        arrow = None
        for token in ("<=>", "=>"):
            if token in equation:
                arrow = token
                break
        if arrow is None:
            return ("raw", equation)
        left_raw, right_raw = equation.split(arrow, 1)

        def _side_key(side: str) -> tuple[tuple[str, float], ...]:
            terms = [t.strip() for t in side.strip().split(" + ") if t.strip()]
            stoich: dict[str, float] = {}
            for term in terms:
                if term == "M":
                    continue
                if term.endswith("(+M)"):
                    term = term[:-4].strip()
                coef = 1.0
                species = term
                match = re.match(
                    r"^([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+(.+)$",
                    term,
                )
                if match:
                    coef = float(match.group(1))
                    species = match.group(2).strip()
                stoich[species] = stoich.get(species, 0.0) + coef
            items = []
            for species, coef in stoich.items():
                if abs(coef) < 1e-15:
                    continue
                items.append((species, round(float(coef), 12)))
            items.sort(key=lambda pair: pair[0].lower())
            return tuple(items)

        left = _side_key(left_raw)
        right = _side_key(right_raw)
        if left <= right:
            return (left, right)
        return (right, left)

    by_key: dict[Any, list[int]] = {}
    for idx, d in enumerate(reaction_dicts):
        eq = d.get("equation")
        if isinstance(eq, str) and eq.strip():
            by_key.setdefault(_canonical_equation_key(eq.strip()), []).append(idx)
    for indices in by_key.values():
        if len(indices) <= 1:
            continue
        for idx in indices:
            reaction_dicts[idx]["duplicate"] = True

    reaction_objs = [ct.Reaction.from_dict(d, base) for d in reaction_dicts]

    reduced = ct.Solution(
        thermo=base.thermo_model,
        kinetics="gas",
        transport_model=base.transport_model,
        species=species_objs,
        reactions=reaction_objs,
    )

    inputs_payload = {
        "mode": "node_lumping_prune",
        "mechanism": mechanism_path,
        "node_lumping_id": lumping_id,
        "merge_mode": merge_mode,
        "require_same_composition": require_same_composition,
        "require_same_charge": require_same_charge,
        "require_same_phase": require_same_phase,
        "protected_species": sorted(protected),
    }
    if cache_bust is not None:
        inputs_payload["cache_bust"] = str(cache_bust)
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[lumping_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        signature_groups_out: list[dict[str, Any]] = []
        for group in signature_groups_trace:
            if not isinstance(group, Mapping):
                continue
            representative = group.get("representative")
            members = group.get("members") or []
            if not isinstance(representative, str) or not representative.strip():
                continue
            if not isinstance(members, Sequence) or isinstance(members, (str, bytes, bytearray)):
                continue
            rep = representative.strip()
            member_list = [
                str(item).strip()
                for item in members
                if isinstance(item, str) and item.strip()
            ]
            members_info = []
            for name in member_list:
                info = merge_eval_by_species.get(name, {})
                merged_to = accepted.get(name)
                members_info.append(
                    {
                        "species": name,
                        "is_representative": bool(name == rep),
                        "merged_to": merged_to,
                        "accepted": bool(name in accepted),
                        "skip_reason": skipped.get(name),
                        "thermo": info.get("thermo"),
                        "kinetics": info.get("kinetics"),
                    }
                )
            group_out = dict(group)
            group_out["members_info"] = members_info
            signature_groups_out.append(group_out)

        patch_payload = {
            "schema_version": PATCH_SCHEMA_VERSION,
            "disabled_reactions": [],
            # Keep structurally non-empty for patch tooling.
            "reaction_multipliers": [{"index": 0, "multiplier": 1.0}],
            "state_merge": {
                "kind": "node_lumping_prune",
                "node_lumping_id": lumping_id,
                "merge_mode": merge_mode,
                "species_to_representative": dict(accepted),
                "protected_species": sorted(protected),
                "signature_groups": signature_groups_out,
                "filters": {
                    "require_same_composition": require_same_composition,
                    "require_same_charge": require_same_charge,
                    "require_same_phase": require_same_phase,
                    "require_same_kind": require_same_kind,
                },
                "thermo_constraints": dict(thermo_cache),
                "kinetics_constraints": dict(kinetics_cache),
            },
        }
        write_yaml_payload(base_dir / PATCH_FILENAME, patch_payload, sort_keys=True)
        # Mechanism must be real YAML for Cantera; don't rely on PyYAML.
        base_dir.mkdir(parents=True, exist_ok=True)
        reduced.write_yaml(str(base_dir / MECHANISM_FILENAME))

        cluster_sizes: list[int] = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            members = cluster.get("members") or []
            if not isinstance(members, Sequence) or isinstance(
                members, (str, bytes, bytearray)
            ):
                continue
            cluster_sizes.append(
                len([m for m in members if isinstance(m, str) and m.strip()])
            )

        metrics = {
            "schema_version": NODE_LUMPING_PRUNE_SCHEMA_VERSION,
            "kind": "node_lumping_prune",
            "node_lumping_id": lumping_id,
            "mechanism": mechanism_path,
            "counts": {
                "species_before": int(base.n_species),
                "species_after": int(reduced.n_species),
                "merged_species": int(len(accepted)),
                "reactions_before": int(base.n_reactions),
                "reactions_after": int(reduced.n_reactions),
                "disabled_reactions": 0,
                "merged_reactions": 0,
            },
            # Backward compatible flat fields.
            "species_before": int(base.n_species),
            "species_after": int(reduced.n_species),
            "merged_species": int(len(accepted)),
            "proposed_merges": int(len(proposed)),
            "accepted_merges": int(len(accepted)),
            "skipped_species": int(len(skipped)),
            "skip_reasons": dict(Counter(skipped.values())),
            "skipped_merges_by_reason": dict(Counter(skipped.values())),
            "merge_mode": merge_mode,
            "signature_groups_total": int(signature_groups_total),
            "signature_groups_merged": int(signature_groups_merged),
            "cluster_sizes": _cluster_size_stats(cluster_sizes),
            "protected_species": sorted(protected),
            "merge_constraints": {
                "require_same_composition": bool(require_same_composition),
                "require_same_charge": bool(require_same_charge),
                "require_same_phase": bool(require_same_phase),
                "require_same_kind": bool(require_same_kind),
            },
            "thermo_constraints": dict(thermo_cache),
            "kinetics_constraints": dict(kinetics_cache),
            "accepted_merges_detail": list(accepted_details),
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Apply a mechanism patch and store the reduction artifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    mechanism_path = _extract_mechanism(reduction_cfg)
    patch_source = _extract_patch_source(reduction_cfg)
    patch_data, _ = _load_patch_payload(patch_source)
    normalized_patch, combined_entries = _normalize_patch_payload(patch_data)

    mechanism_payload = read_yaml_payload(Path(mechanism_path))
    if not isinstance(mechanism_payload, Mapping):
        raise ConfigError("mechanism YAML must be a mapping.")
    reduced_payload, _ = _apply_patch_entries(
        dict(mechanism_payload),
        combined_entries,
    )

    inputs_payload = {
        "mechanism": mechanism_path,
        "patch": normalized_patch,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(
            base_dir / PATCH_FILENAME,
            normalized_patch,
            sort_keys=True,
        )
        write_yaml_payload(
            base_dir / MECHANISM_FILENAME,
            reduced_payload,
            sort_keys=False,
        )

    return store.ensure(manifest, writer=_writer)


def _extract_validation_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("validation", "validate"):
        if key in reduction_cfg:
            value = reduction_cfg.get(key)
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(reduction_cfg)

def _expand_inject_templates(value: Any, *, inputs_cfg: Mapping[str, Any]) -> Any:
    """Expand $inputs.* templates for validation.inject_inputs.

    PipelineRunner resolves @step refs only inside step.inputs. The validation config
    is not part of step.inputs, so we support a small template mechanism that can
    reference already-resolved outer-step inputs.
    """
    if isinstance(value, str):
        raw = value.strip()
        if raw == "$inputs":
            return dict(inputs_cfg)
        if raw.startswith("$inputs."):
            path = raw[len("$inputs.") :].strip()
            if not path:
                raise ConfigError("inject_inputs template $inputs. requires a key path.")
            current: Any = inputs_cfg
            for part in path.split("."):
                if not isinstance(current, Mapping):
                    raise ConfigError(
                        f"inject_inputs template {raw!r} traversed a non-mapping at {part!r}."
                    )
                if part not in current:
                    raise ConfigError(f"inject_inputs template key missing: {raw!r}.")
                current = current[part]
            return current
        return value
    if isinstance(value, Mapping):
        return {
            key: _expand_inject_templates(item, inputs_cfg=inputs_cfg)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _expand_inject_templates(item, inputs_cfg=inputs_cfg) for item in value
        ]
    return value


def _inject_pipeline_inputs(
    pipeline_cfg: Mapping[str, Any],
    inject_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge inject_inputs into pipeline.steps[*].inputs by step id."""
    updated = copy.deepcopy(dict(pipeline_cfg))
    steps = updated.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        raise ConfigError("pipeline.steps must be a list to inject inputs.")
    remaining = {str(key): value for key, value in inject_inputs.items()}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            continue
        if step_id not in remaining:
            continue
        payload = remaining.pop(step_id)
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise ConfigError("validation.inject_inputs values must be mappings.")
        existing = step.get("inputs")
        if existing is None:
            existing = {}
        if not isinstance(existing, Mapping):
            raise ConfigError("pipeline step inputs must be a mapping.")
        merged = dict(existing)
        merged.update(dict(payload))
        # Mutate the copied pipeline step mapping.
        step["inputs"] = merged
    if remaining:
        missing = ", ".join(sorted(remaining.keys()))
        raise ConfigError(f"validation.inject_inputs references unknown step ids: {missing}.")
    return updated


def _extract_pipeline_value(validation_cfg: Mapping[str, Any]) -> Any:
    pipeline = validation_cfg.get("pipeline")
    if pipeline is None:
        pipeline = validation_cfg.get("pipeline_cfg") or validation_cfg.get("pipeline_config")
    if pipeline is None and "steps" in validation_cfg:
        pipeline = {"steps": validation_cfg.get("steps")}
    if pipeline is None:
        raise ConfigError("validation.pipeline must be provided.")
    return pipeline


def _normalize_pipeline_cfg(
    pipeline_value: Any,
    runner: PipelineRunner,
) -> dict[str, Any]:
    if isinstance(pipeline_value, Mapping):
        pipeline_cfg = dict(pipeline_value)
    elif isinstance(pipeline_value, Sequence) and not isinstance(
        pipeline_value, (str, bytes, bytearray)
    ):
        pipeline_cfg = {"steps": list(pipeline_value)}
    else:
        pipeline_cfg = runner.load(pipeline_value)
    steps = pipeline_cfg.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        raise ConfigError("pipeline.steps must be a list of step mappings.")
    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ConfigError(f"pipeline.steps[{index}] must be a mapping.")
        normalized_steps.append(dict(step))
    pipeline_cfg["steps"] = normalized_steps
    return pipeline_cfg


def _extract_step_id(
    steps: Sequence[Mapping[str, Any]],
    *,
    override: Any,
    task_name: str,
    label: str,
    required: bool,
) -> Optional[str]:
    if override is not None:
        step_id = _require_nonempty_str(override, label)
        for step in steps:
            if step.get("id") == step_id:
                return step_id
        raise ConfigError(f"{label} does not match any pipeline step id.")
    for step in steps:
        if step.get("task") == task_name:
            step_id = step.get("id")
            if isinstance(step_id, str) and step_id.strip():
                return step_id
    if required:
        raise ConfigError(f"pipeline must include a {task_name} step.")
    return None


def _extract_sim_override(validation_cfg: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    sim_cfg = validation_cfg.get("sim") or validation_cfg.get("sim_cfg")
    if sim_cfg is None:
        return None
    if not isinstance(sim_cfg, Mapping):
        raise ConfigError("validation.sim must be a mapping.")
    return dict(sim_cfg)


def _normalize_case_mode(value: Any) -> str:
    if value is None:
        return "single"
    if not isinstance(value, str):
        raise ConfigError("validation.case_mode must be a string.")
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"single", "one", "default"}:
        return "single"
    if normalized in {"all", "all-cases", "csv-all"}:
        return "all"
    raise ConfigError("validation.case_mode must be 'single' or 'all'.")


def _extract_case_col(validation_cfg: Mapping[str, Any]) -> str:
    value = (
        validation_cfg.get("case_col")
        or validation_cfg.get("case_column")
        or validation_cfg.get("case_field")
        or "case_id"
    )
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("validation.case_col must be a non-empty string.")
    return value.strip()


def _extract_conditions_path(
    validation_cfg: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    sim_step_id: str,
) -> Path:
    value: Any = None
    for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
        if key in validation_cfg and validation_cfg.get(key) is not None:
            value = validation_cfg.get(key)
            break
    if value is None:
        for step in steps:
            if step.get("id") != sim_step_id:
                continue
            params = step.get("params")
            if not isinstance(params, Mapping):
                params = {}
            for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
                if key in params and params.get(key) is not None:
                    value = params.get(key)
                    break
            break
    if value is None:
        raise ConfigError(
            "validation.case_mode='all' requires conditions_file (or sim step params)."
        )
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError("validation.conditions_file must be a non-empty string.")
    return resolve_repo_path(value)


def _load_case_ids_from_csv(path: Path, *, case_col: str) -> list[str]:
    if not path.exists():
        raise ConfigError(f"conditions_file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ConfigError(f"conditions_file is empty: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        raw = row.get(case_col)
        if raw is None:
            continue
        cid = str(raw).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
    if not ids:
        raise ConfigError(f"conditions_file missing {case_col} values: {path}")
    return ids


def _normalize_missing_strategy(value: Any) -> str:
    if value is None:
        return DEFAULT_VALIDATION_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("validation.missing_strategy must be a string.")
    normalized = value.strip().lower()
    if normalized in {"fail", "error"}:
        return "fail"
    if normalized in {"skip", "ignore"}:
        return "skip"
    raise ConfigError("validation.missing_strategy must be 'fail' or 'skip'.")


def _extract_metric_name(validation_cfg: Mapping[str, Any]) -> str:
    metric = validation_cfg.get("metric") or validation_cfg.get("diff_metric")
    if isinstance(metric, Mapping):
        metric = metric.get("name") or metric.get("type")
    if metric is None:
        metric = DEFAULT_VALIDATION_METRIC
    if not isinstance(metric, str) or not metric.strip():
        raise ConfigError("validation.metric must be a non-empty string.")
    metric = metric.strip().lower()
    if metric in {"abs", "absolute"}:
        return "abs"
    if metric in {"rel", "relative", "ratio"}:
        return "rel"
    raise ConfigError("validation.metric must be 'abs' or 'rel'.")


def _extract_tolerance(validation_cfg: Mapping[str, Any]) -> float:
    raw = validation_cfg.get("tolerance")
    if raw is None:
        raw = validation_cfg.get("tol") or validation_cfg.get("threshold")
    metric_cfg = validation_cfg.get("metric")
    if raw is None and isinstance(metric_cfg, Mapping):
        raw = metric_cfg.get("tolerance") or metric_cfg.get("tol")
    if raw is None:
        return DEFAULT_VALIDATION_TOLERANCE
    if isinstance(raw, bool):
        raise ConfigError("validation.tolerance must be a number.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("validation.tolerance must be a number.") from exc
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ConfigError("validation.tolerance must be a finite non-negative number.")
    return value


def _extract_rel_eps(validation_cfg: Mapping[str, Any]) -> float:
    raw = (
        validation_cfg.get("rel_eps")
        or validation_cfg.get("relative_epsilon")
        or validation_cfg.get("rel_epsilon")
    )
    metric_cfg = validation_cfg.get("metric")
    if raw is None and isinstance(metric_cfg, Mapping):
        raw = metric_cfg.get("rel_eps") or metric_cfg.get("rel_epsilon")
    if raw is None:
        return DEFAULT_VALIDATION_REL_EPS
    if isinstance(raw, bool):
        raise ConfigError("validation.rel_eps must be a number.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("validation.rel_eps must be a number.") from exc
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ConfigError("validation.rel_eps must be a finite non-negative number.")
    return value


def _amore_extract_search_cfg(reduction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = _extract_params(reduction_cfg)
    search_cfg = params.get("search") or params.get("amore") or reduction_cfg.get("search")
    if search_cfg is None:
        return {}
    if not isinstance(search_cfg, Mapping):
        raise ConfigError("reduction.search must be a mapping when provided.")
    return dict(search_cfg)


def _amore_candidate_hash(base_hash: str, disabled_indices: Sequence[int]) -> str:
    payload = {
        "base_mechanism_hash": base_hash,
        "disabled_indices": list(disabled_indices),
    }
    return stable_hash(payload, length=16)


def _amore_load_graph_payload(store: ArtifactStore, graph_id: str) -> dict[str, Any]:
    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_path = graph_dir / "graph.json"
    if not graph_path.exists():
        raise ConfigError(f"graph.json not found for graphs/{graph_id}")
    try:
        payload = read_json(graph_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json payload must be a mapping.")
    return dict(payload)


def _amore_reaction_activity(
    graph_payload: Mapping[str, Any],
    compiler: MechanismCompiler,
    *,
    logger: logging.Logger,
) -> list[float]:
    reaction_count = compiler.reaction_count()
    stats = graph_payload.get("reaction_stats")
    activity_values: list[float] = []
    if isinstance(stats, Mapping):
        activity = stats.get("activity")
        if isinstance(activity, Mapping):
            values = activity.get("values")
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                activity_values = [
                    float(value) if value is not None else 0.0 for value in values
                ]

    if activity_values:
        reactions_payload = graph_payload.get("reactions", {})
        reactions_order: list[str] = []
        if isinstance(reactions_payload, Mapping):
            order = reactions_payload.get("order")
            if isinstance(order, Sequence) and not isinstance(
                order, (str, bytes, bytearray)
            ):
                reactions_order = [str(item) for item in order]
        if len(activity_values) == reaction_count:
            return activity_values
        if reactions_order and len(reactions_order) == len(activity_values):
            index_map: dict[str, int] = {}
            for idx, name in enumerate(reactions_order):
                if name and name not in index_map:
                    index_map[name] = idx
            aligned: list[float] = []
            matched = 0
            for idx, reaction in enumerate(compiler.reactions):
                value = 0.0
                for identifier in _reaction_identifiers(reaction, idx):
                    mapped = index_map.get(identifier)
                    if mapped is not None:
                        value = activity_values[mapped]
                        matched += 1
                        break
                aligned.append(float(value))
            if matched:
                logger.warning(
                    "Aligned reaction activity using identifiers (%d/%d matched).",
                    matched,
                    reaction_count,
                )
                return aligned
        logger.warning(
            "Reaction activity length mismatch; falling back to uniform weights."
        )

    return [1.0 for _ in range(reaction_count)]


def _amore_cheap_score(
    activity_values: Sequence[float],
    disabled_indices: Sequence[int],
) -> float:
    total = float(sum(activity_values))
    if total <= 0.0:
        return 1.0
    removed = 0.0
    for idx in disabled_indices:
        if idx < 0 or idx >= len(activity_values):
            continue
        removed += float(activity_values[idx])
    score = (total - removed) / total
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _amore_write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    if not columns:
        columns = ["mechanism_hash"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _amore_extract_surrogate_cfg(
    reduction_cfg: Mapping[str, Any],
    resolved_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    params = _extract_params(reduction_cfg)
    surrogate_cfg = params.get("surrogate") or reduction_cfg.get("surrogate") or {}
    if surrogate_cfg is None:
        surrogate_cfg = {}
    if not isinstance(surrogate_cfg, Mapping):
        raise ConfigError("reduction.surrogate must be a mapping when provided.")
    surrogate_cfg = dict(surrogate_cfg)

    enabled = surrogate_cfg.get("enabled")
    if enabled is None:
        for source in (params, reduction_cfg, resolved_cfg):
            if not isinstance(source, Mapping):
                continue
            if "use_surrogate" in source:
                enabled = source.get("use_surrogate")
                break
            if "surrogate" in source and isinstance(source.get("surrogate"), bool):
                enabled = source.get("surrogate")
                break
    enabled = _coerce_bool(enabled, "surrogate.enabled", default=False)

    dataset_name = surrogate_cfg.get("dataset_name") or surrogate_cfg.get("dataset")
    if dataset_name is None:
        dataset_name = DEFAULT_AMORE_SURROGATE_DATASET_NAME
    dataset_name = _require_nonempty_str(dataset_name, "surrogate.dataset_name")

    def _coerce_positive(value: Any, label: str, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{label} must be an integer.")
        if value <= 0:
            raise ConfigError(f"{label} must be > 0.")
        return value

    min_samples = _coerce_positive(
        surrogate_cfg.get("min_samples") or surrogate_cfg.get("min_train"),
        "surrogate.min_samples",
        DEFAULT_AMORE_SURROGATE_MIN_SAMPLES,
    )
    warmup = surrogate_cfg.get("warmup")
    if warmup is None:
        warmup = DEFAULT_AMORE_SURROGATE_WARMUP
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigError("surrogate.warmup must be a non-negative integer.")

    k_neighbors = _coerce_positive(
        surrogate_cfg.get("k_neighbors") or surrogate_cfg.get("k"),
        "surrogate.k_neighbors",
        DEFAULT_AMORE_SURROGATE_K,
    )
    max_types = surrogate_cfg.get("max_types") or surrogate_cfg.get("type_buckets")
    if max_types is None:
        max_types = DEFAULT_AMORE_SURROGATE_MAX_TYPES
    if isinstance(max_types, bool) or not isinstance(max_types, int) or max_types < 0:
        raise ConfigError("surrogate.max_types must be a non-negative integer.")

    uncertainty_gate = _coerce_optional_float(
        surrogate_cfg.get("uncertainty_gate") or surrogate_cfg.get("uncertainty"),
        "surrogate.uncertainty_gate",
    )
    if uncertainty_gate is None:
        uncertainty_gate = DEFAULT_AMORE_SURROGATE_UNCERTAINTY
    if uncertainty_gate < 0.0 or uncertainty_gate > 1.0:
        raise ConfigError("surrogate.uncertainty_gate must be in [0, 1].")

    fail_skip = _coerce_optional_float(
        surrogate_cfg.get("fail_prob_skip") or surrogate_cfg.get("fail_skip"),
        "surrogate.fail_prob_skip",
    )
    if fail_skip is None:
        fail_skip = DEFAULT_AMORE_SURROGATE_FAIL_SKIP
    if fail_skip < 0.0 or fail_skip > 1.0:
        raise ConfigError("surrogate.fail_prob_skip must be in [0, 1].")

    error_skip = _coerce_optional_float(
        surrogate_cfg.get("error_skip"),
        "surrogate.error_skip",
    )
    error_factor = _coerce_optional_float(
        surrogate_cfg.get("error_skip_factor") or surrogate_cfg.get("error_factor"),
        "surrogate.error_skip_factor",
    )
    if error_factor is None:
        error_factor = DEFAULT_AMORE_SURROGATE_ERROR_FACTOR
    if error_factor <= 0.0:
        raise ConfigError("surrogate.error_skip_factor must be > 0.")

    update_every = surrogate_cfg.get("update_every")
    if update_every is None:
        update_every = DEFAULT_AMORE_SURROGATE_UPDATE_EVERY
    if isinstance(update_every, bool) or not isinstance(update_every, int) or update_every <= 0:
        raise ConfigError("surrogate.update_every must be a positive integer.")

    return {
        "enabled": enabled,
        "dataset_name": dataset_name,
        "min_samples": min_samples,
        "warmup": warmup,
        "k_neighbors": k_neighbors,
        "max_types": max_types,
        "uncertainty_gate": uncertainty_gate,
        "fail_prob_skip": fail_skip,
        "error_skip": error_skip,
        "error_skip_factor": error_factor,
        "update_every": update_every,
    }


def _amore_resolve_surrogate_root(store_root: Path, dataset_name: str) -> Path:
    dataset_path = Path(dataset_name)
    if dataset_path.is_absolute() or ".." in dataset_path.parts:
        raise ConfigError("surrogate.dataset_name must be a relative path without '..'.")
    if len(dataset_path.parts) > 1:
        raise ConfigError("surrogate.dataset_name must be a single path component.")
    run_root = store_root.parent if store_root.name == "artifacts" else store_root
    return run_root / "datasets" / dataset_path.name


def _amore_load_surrogate_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta_path = dataset_dir / "dataset.json"
    data_path = dataset_dir / "data.json"
    if not meta_path.exists() or not data_path.exists():
        return {}, []
    try:
        meta_payload = read_json(meta_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"surrogate dataset.json is invalid: {exc}") from exc
    if not isinstance(meta_payload, Mapping):
        raise ConfigError("surrogate dataset.json must be a JSON object.")
    try:
        data_payload = read_json(data_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"surrogate data.json is invalid: {exc}") from exc
    if not isinstance(data_payload, Sequence) or isinstance(
        data_payload, (str, bytes, bytearray)
    ):
        raise ConfigError("surrogate data.json must contain a list.")
    rows: list[dict[str, Any]] = []
    for entry in data_payload:
        if isinstance(entry, Mapping):
            rows.append(dict(entry))
    return dict(meta_payload), rows


def _amore_write_surrogate_dataset(
    dataset_dir: Path,
    meta: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = dict(meta)
    meta_payload.setdefault("schema_version", 1)
    meta_payload["rows"] = len(rows)
    meta_payload["updated_at"] = _utc_now_iso()
    write_json_atomic(dataset_dir / "dataset.json", meta_payload)
    write_json_atomic(dataset_dir / "data.json", list(rows))


def _amore_reaction_type_label(reaction: Any) -> str:
    if isinstance(reaction, Mapping):
        for key in ("reaction_type", "type"):
            value = reaction.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        if reaction.get("duplicate"):
            return "duplicate"
    return "unknown"


def _amore_select_type_buckets(
    reaction_types: Sequence[str],
    *,
    max_types: int,
) -> list[str]:
    counts: dict[str, int] = {}
    for entry in reaction_types:
        key = entry if entry else "unknown"
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    buckets = [name for name, _ in ordered]
    if max_types > 0:
        buckets = buckets[:max_types]
    if "other" not in buckets:
        buckets.append("other")
    return buckets


def _amore_feature_key(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return "unknown"
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value)
    return cleaned or "unknown"


def _amore_feature_names(type_buckets: Sequence[str]) -> list[str]:
    base = [
        "disabled_ratio",
        "disabled_count",
        "cheap_score",
        "removed_activity_ratio",
        "removed_activity_mean",
    ]
    type_features = [f"type_ratio_{_amore_feature_key(name)}" for name in type_buckets]
    return base + type_features


def _amore_candidate_features(
    disabled_indices: Sequence[int],
    *,
    reaction_count: int,
    cheap_score: float,
    reaction_types: Sequence[str],
    type_buckets: Sequence[str],
    activity_values: Sequence[float],
) -> tuple[list[float], dict[str, float]]:
    disabled_set = {int(idx) for idx in disabled_indices if isinstance(idx, int)}
    disabled_count = len(disabled_set)
    disabled_ratio = (
        float(disabled_count) / float(reaction_count) if reaction_count > 0 else 0.0
    )
    total_activity = float(sum(activity_values))
    removed_activity = 0.0
    for idx in disabled_set:
        if 0 <= idx < len(activity_values):
            removed_activity += float(activity_values[idx])
    removed_activity_ratio = (
        removed_activity / total_activity if total_activity > 0.0 else 0.0
    )
    removed_activity_mean = (
        removed_activity / float(disabled_count) if disabled_count > 0 else 0.0
    )

    bucket_set = {bucket for bucket in type_buckets}
    total_by_bucket: dict[str, int] = {bucket: 0 for bucket in type_buckets}
    removed_by_bucket: dict[str, int] = {bucket: 0 for bucket in type_buckets}
    for idx, reaction_type in enumerate(reaction_types):
        bucket = reaction_type if reaction_type in bucket_set else "other"
        if bucket not in total_by_bucket:
            bucket = "other"
        total_by_bucket[bucket] = total_by_bucket.get(bucket, 0) + 1
        if idx in disabled_set:
            removed_by_bucket[bucket] = removed_by_bucket.get(bucket, 0) + 1

    feature_values: dict[str, float] = {
        "disabled_ratio": disabled_ratio,
        "disabled_count": float(disabled_count),
        "cheap_score": float(cheap_score),
        "removed_activity_ratio": removed_activity_ratio,
        "removed_activity_mean": removed_activity_mean,
    }
    for bucket in type_buckets:
        total = total_by_bucket.get(bucket, 0)
        removed = removed_by_bucket.get(bucket, 0)
        ratio = float(removed) / float(total) if total > 0 else 0.0
        feature_values[f"type_ratio_{_amore_feature_key(bucket)}"] = ratio

    feature_names = _amore_feature_names(type_buckets)
    feature_vector = [feature_values.get(name, 0.0) for name in feature_names]
    return feature_vector, feature_values


def _amore_fit_surrogate_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    min_samples: int,
    k_neighbors: int,
) -> dict[str, Any]:
    samples: list[list[float]] = []
    error_targets: list[Optional[float]] = []
    fail_targets: list[Optional[float]] = []
    for row in rows:
        features = row.get("features")
        if not isinstance(features, Mapping):
            continue
        vector: list[float] = []
        for name in feature_names:
            value, ok = _coerce_numeric(features.get(name))
            vector.append(value if ok else 0.0)
        samples.append(vector)

        qoi_error, error_ok = _coerce_numeric(row.get("qoi_error"))
        error_targets.append(qoi_error if error_ok else None)

        passed = row.get("passed")
        if isinstance(passed, bool):
            fail_targets.append(0.0 if passed else 1.0)
        else:
            failed = row.get("failed")
            if isinstance(failed, bool):
                fail_targets.append(1.0 if failed else 0.0)
            else:
                fail_targets.append(None)

    count = len(samples)
    if count == 0:
        return {
            "ready": False,
            "feature_names": list(feature_names),
            "samples": [],
            "error_targets": [],
            "fail_targets": [],
            "mean": [],
            "std": [],
            "k_neighbors": k_neighbors,
            "distance_scale": 1.0,
        }

    n_features = len(samples[0])
    means = [0.0 for _ in range(n_features)]
    for sample in samples:
        for idx, value in enumerate(sample):
            means[idx] += float(value)
    means = [value / float(count) for value in means]

    variances = [0.0 for _ in range(n_features)]
    for sample in samples:
        for idx, value in enumerate(sample):
            diff = float(value) - means[idx]
            variances[idx] += diff * diff
    stds = [math.sqrt(value / float(count)) for value in variances]
    stds = [value if value > 1.0e-12 else 1.0 for value in stds]

    standardized: list[list[float]] = []
    for sample in samples:
        standardized.append(
            [(sample[idx] - means[idx]) / stds[idx] for idx in range(n_features)]
        )

    has_error = any(value is not None for value in error_targets)
    has_fail = any(value is not None for value in fail_targets)
    ready = count >= min_samples and (has_error or has_fail)

    return {
        "ready": ready,
        "feature_names": list(feature_names),
        "samples": standardized,
        "error_targets": error_targets,
        "fail_targets": fail_targets,
        "mean": means,
        "std": stds,
        "k_neighbors": k_neighbors,
        "distance_scale": math.sqrt(float(n_features)),
    }


def _amore_predict_surrogate(
    model: Mapping[str, Any],
    feature_vector: Sequence[float],
) -> dict[str, Optional[float]]:
    samples = model.get("samples") or []
    if not isinstance(samples, Sequence) or not samples:
        return {"pred_error": None, "pred_fail_prob": None, "uncertainty": 1.0}
    mean = model.get("mean") or []
    std = model.get("std") or []
    if not mean or not std:
        return {"pred_error": None, "pred_fail_prob": None, "uncertainty": 1.0}

    standardized = [
        (float(value) - mean[idx]) / std[idx] for idx, value in enumerate(feature_vector)
    ]

    distances: list[tuple[float, int]] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, Sequence):
            continue
        dist_sq = 0.0
        for jdx, value in enumerate(sample):
            diff = float(value) - standardized[jdx]
            dist_sq += diff * diff
        distances.append((math.sqrt(dist_sq), idx))
    if not distances:
        return {"pred_error": None, "pred_fail_prob": None, "uncertainty": 1.0}

    distances.sort(key=lambda item: item[0])
    k = model.get("k_neighbors") or len(distances)
    k = min(int(k), len(distances))
    neighbors = distances[:k]
    eps = 1.0e-8

    def _weighted_mean(targets: Sequence[Optional[float]]) -> tuple[Optional[float], float]:
        weight_sum = 0.0
        value_sum = 0.0
        for dist, idx in neighbors:
            if idx >= len(targets):
                continue
            value = targets[idx]
            if value is None:
                continue
            weight = 1.0 / (dist + eps)
            weight_sum += weight
            value_sum += weight * float(value)
        if weight_sum <= 0.0:
            return None, 0.0
        return value_sum / weight_sum, weight_sum

    error_targets = model.get("error_targets") or []
    fail_targets = model.get("fail_targets") or []

    pred_error, _ = _weighted_mean(error_targets)
    pred_fail, _ = _weighted_mean(fail_targets)

    error_std = 0.0
    if pred_error is not None:
        weight_sum = 0.0
        for dist, idx in neighbors:
            if idx >= len(error_targets):
                continue
            value = error_targets[idx]
            if value is None:
                continue
            weight = 1.0 / (dist + eps)
            diff = float(value) - pred_error
            error_std += weight * diff * diff
            weight_sum += weight
        if weight_sum > 0.0:
            error_std = math.sqrt(error_std / weight_sum)

    distance_scale = float(model.get("distance_scale") or 1.0)
    avg_distance = sum(item[0] for item in neighbors) / float(len(neighbors))
    distance_score = min(1.0, avg_distance / (distance_scale + eps))

    if pred_fail is None:
        fail_uncertainty = 1.0
    else:
        fail_uncertainty = min(1.0, max(0.0, 4.0 * pred_fail * (1.0 - pred_fail)))

    if pred_error is None:
        error_uncertainty = 1.0
    else:
        denom = abs(pred_error) + eps
        error_uncertainty = min(1.0, error_std / denom)

    uncertainty = max(distance_score, fail_uncertainty, error_uncertainty)

    return {
        "pred_error": pred_error,
        "pred_fail_prob": pred_fail,
        "uncertainty": uncertainty,
    }


def _normalize_patch_candidates(validation_cfg: Mapping[str, Any]) -> list[Any]:
    patches = validation_cfg.get("patches")
    if patches is None:
        patches = validation_cfg.get("patch") or validation_cfg.get("candidates")
    if patches is None:
        raise ConfigError("validation.patches must be provided.")
    if isinstance(patches, Sequence) and not isinstance(
        patches, (str, bytes, bytearray)
    ):
        return list(patches)
    return [patches]


def _load_reduction_artifact(
    store: ArtifactStore,
    reduction_id: str,
) -> tuple[Path, dict[str, Any]]:
    store.read_manifest("reduction", reduction_id)
    reduction_dir = store.artifact_dir("reduction", reduction_id)
    patch_payload = read_yaml_payload(reduction_dir / PATCH_FILENAME)
    if not isinstance(patch_payload, Mapping):
        raise ConfigError("reduction patch payload must be a mapping.")
    return reduction_dir, dict(patch_payload)


def _resolve_reduction_candidate(
    candidate: Any,
    *,
    mechanism_path: str,
    store: ArtifactStore,
    registry: Optional[Registry],
) -> tuple[ArtifactCacheResult, dict[str, Any]]:
    reduction_id: Optional[str] = None
    patch_value: Any = None
    if isinstance(candidate, Mapping):
        if (
            "disabled_reactions" not in candidate
            and "reaction_multipliers" not in candidate
        ):
            reduction_id = candidate.get("reduction_id") or candidate.get("artifact_id")
        if reduction_id is None:
            patch_value = dict(candidate)
    elif isinstance(candidate, Path):
        patch_value = str(candidate)
    elif isinstance(candidate, str):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            patch_value = candidate
        else:
            reduction_id = candidate
    else:
        raise ConfigError("validation patch entries must be mappings or strings.")

    if reduction_id is not None:
        reduction_id = _require_nonempty_str(reduction_id, "reduction_id")
        reduction_dir, patch_payload = _load_reduction_artifact(store, reduction_id)
        result = ArtifactCacheResult(
            path=reduction_dir,
            reused=True,
            manifest=store.read_manifest("reduction", reduction_id),
        )
        return result, patch_payload

    cfg = {"reduction": {"mechanism": mechanism_path, "patch": patch_value}}
    result = run(cfg, store=store, registry=registry)
    patch_payload = read_yaml_payload(result.path / PATCH_FILENAME)
    if not isinstance(patch_payload, Mapping):
        raise ConfigError("reduction patch payload must be a mapping.")
    return result, dict(patch_payload)


def _extract_patch_multipliers(patch_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    multipliers_raw = patch_payload.get("reaction_multipliers") or []
    try:
        multipliers = normalize_reaction_multipliers(
            {"reaction_multipliers": multipliers_raw}
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"patch reaction_multipliers are invalid: {exc}") from exc
    filtered: list[dict[str, Any]] = []
    for entry in multipliers:
        try:
            multiplier = float(entry.get("multiplier", 1.0))
        except (TypeError, ValueError):
            multiplier = 0.0
        if multiplier == 0.0:
            continue
        filtered.append(dict(entry))
    return filtered


def _build_patch_multipliers_with_disabled(
    patch_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    disabled = patch_payload.get("disabled_reactions") or []
    if isinstance(disabled, Sequence) and not isinstance(
        disabled, (str, bytes, bytearray)
    ):
        for item in disabled:
            if not isinstance(item, Mapping):
                continue
            entry = dict(item)
            entry.setdefault("multiplier", 0.0)
            entries.append(entry)
    multipliers_raw = patch_payload.get("reaction_multipliers") or []
    if isinstance(multipliers_raw, Sequence) and not isinstance(
        multipliers_raw, (str, bytes, bytearray)
    ):
        for item in multipliers_raw:
            if isinstance(item, Mapping):
                entries.append(dict(item))
    try:
        return normalize_reaction_multipliers({"reaction_multipliers": entries})
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"patch multipliers are invalid: {exc}") from exc


def _normalize_meta_json_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return ""
        parsed = _parse_meta_json(cleaned)
        if parsed:
            return json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        return cleaned
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
    return json.dumps({"detail": str(value)}, ensure_ascii=True, sort_keys=True)


def _group_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name_field: str,
) -> dict[tuple[str, str, str], list[Any]]:
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    for row in rows:
        name = row.get(name_field)
        if not isinstance(name, str) or not name.strip():
            continue
        meta_json = _normalize_meta_json_key(row.get("meta_json"))
        unit_value = row.get("unit")
        unit_str = "" if unit_value is None else str(unit_value)
        key = (name.strip(), meta_json, unit_str)
        grouped.setdefault(key, []).append(row.get("value"))
    return grouped


def _coerce_numeric(value: Any) -> tuple[float, bool]:
    if value is None:
        return math.nan, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan, False
    if math.isnan(number) or math.isinf(number):
        return math.nan, False
    return number, True


def _compare_grouped_values(
    *,
    kind: str,
    baseline_groups: Mapping[tuple[str, str, str], Sequence[Any]],
    reduced_groups: Mapping[tuple[str, str, str], Sequence[Any]],
    metric: str,
    tolerance: float,
    rel_eps: float,
    missing_strategy: str,
    patch_index: int,
    patch_id: str,
    baseline_run_id: str,
    reduced_run_id: str,
    baseline_artifact_id: str,
    reduced_artifact_id: str,
) -> tuple[list[dict[str, Any]], bool, int]:
    rows: list[dict[str, Any]] = []
    overall_pass = True
    evaluated = 0
    keys = sorted(set(baseline_groups) | set(reduced_groups))
    for key in keys:
        name, meta_json, unit = key
        baseline_values = list(baseline_groups.get(key, []))
        reduced_values = list(reduced_groups.get(key, []))
        max_len = max(len(baseline_values), len(reduced_values))
        for index in range(max_len):
            base_raw = baseline_values[index] if index < len(baseline_values) else None
            reduced_raw = reduced_values[index] if index < len(reduced_values) else None
            base_value = math.nan
            reduced_value = math.nan
            abs_diff = math.nan
            rel_diff = math.nan
            status = "ok"
            passed = True

            if base_raw is None or reduced_raw is None:
                status = "missing"
                passed = missing_strategy == "skip"
            else:
                base_value, base_ok = _coerce_numeric(base_raw)
                reduced_value, reduced_ok = _coerce_numeric(reduced_raw)
                if not base_ok or not reduced_ok:
                    status = "invalid"
                    passed = missing_strategy == "skip"
                else:
                    abs_diff = abs(reduced_value - base_value)
                    rel_diff = abs_diff / (abs(base_value) + rel_eps)
                    metric_value = abs_diff if metric == "abs" else rel_diff
                    passed = metric_value <= tolerance
                    evaluated += 1

            if status in {"missing", "invalid"} and missing_strategy == "skip":
                status = "skipped"
            if not passed and status != "skipped":
                overall_pass = False

            rows.append(
                {
                    "patch_index": patch_index,
                    "patch_id": patch_id,
                    "passed": passed,
                    "status": status,
                    "kind": kind,
                    "name": name,
                    "unit": unit,
                    "meta_json": meta_json,
                    "item_index": index,
                    "baseline_value": base_value,
                    "reduced_value": reduced_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "metric": metric,
                    "tolerance": tolerance,
                    "baseline_run_id": baseline_run_id,
                    "reduced_run_id": reduced_run_id,
                    "baseline_artifact_id": baseline_artifact_id,
                    "reduced_artifact_id": reduced_artifact_id,
                }
            )
    if evaluated == 0:
        overall_pass = False
    return rows, overall_pass, evaluated


def _collect_validation_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(REQUIRED_VALIDATION_COLUMNS)
    extras: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in columns:
                extras.add(str(key))
    return columns + sorted(extras)


def _write_validation_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = _collect_validation_columns(rows)
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
    logger = logging.getLogger("rxn_platform.reduction")
    logger.warning(
        "Parquet writer unavailable; stored JSON payload at %s.",
        path,
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


def _extract_run_rows(
    store: ArtifactStore,
    *,
    kind: str,
    artifact_id: str,
    run_id: str,
    filename: str,
) -> list[dict[str, Any]]:
    store.read_manifest(kind, artifact_id)
    table_path = store.artifact_dir(kind, artifact_id) / filename
    rows = _read_table_rows(table_path)
    return [row for row in rows if row.get("run_id") == run_id]


def validate_reduction(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Validate reduction patches by rerunning pipelines and comparing outputs."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    validation_cfg = _extract_validation_cfg(reduction_cfg)
    if "pipeline" not in validation_cfg:
        params = _extract_params(reduction_cfg)
        candidate = params.get("validation") if isinstance(params, Mapping) else None
        if isinstance(candidate, Mapping):
            validation_cfg = dict(candidate)
    inputs_cfg = reduction_cfg.get("inputs")
    if isinstance(inputs_cfg, Mapping):
        if not any(
            key in validation_cfg for key in ("patches", "patch", "candidates")
        ):
            for key in ("patches", "patch", "candidates"):
                if key in inputs_cfg:
                    validation_cfg[key] = inputs_cfg[key]
                    break

    inject_inputs_value = (
        validation_cfg.get("inject_inputs")
        or validation_cfg.get("inject")
        or validation_cfg.get("pipeline_inject")
    )
    inject_inputs: dict[str, Any] = {}
    if inject_inputs_value is None:
        inject_inputs = {}
    elif not isinstance(inject_inputs_value, Mapping):
        raise ConfigError("validation.inject_inputs must be a mapping when provided.")
    else:
        inject_inputs = dict(inject_inputs_value)

    mechanism_path = _extract_mechanism(reduction_cfg)
    pipeline_value = _extract_pipeline_value(validation_cfg)

    metric = _extract_metric_name(validation_cfg)
    tolerance = _extract_tolerance(validation_cfg)
    rel_eps = _extract_rel_eps(validation_cfg)
    missing_strategy = _normalize_missing_strategy(
        validation_cfg.get("missing_strategy")
    )
    stop_on_fail = bool(validation_cfg.get("stop_on_fail", True))
    use_multipliers_only = bool(validation_cfg.get("use_multipliers_only", False))

    logger = logging.getLogger("rxn_platform.reduction")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    pipeline_cfg = _normalize_pipeline_cfg(pipeline_value, runner)
    steps = pipeline_cfg.get("steps", [])
    if inject_inputs:
        inputs_map = dict(inputs_cfg) if isinstance(inputs_cfg, Mapping) else {}
        expanded = _expand_inject_templates(inject_inputs, inputs_cfg=inputs_map)
        if not isinstance(expanded, Mapping):
            raise ConfigError("validation.inject_inputs expansion must produce a mapping.")
        pipeline_cfg = _inject_pipeline_inputs(pipeline_cfg, dict(expanded))
        steps = pipeline_cfg.get("steps", [])

    sim_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("sim_step_id"),
        task_name="sim.run",
        label="validation.sim_step_id",
        required=True,
    )
    obs_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("observables_step_id"),
        task_name="observables.run",
        label="validation.observables_step_id",
        required=False,
    )
    feat_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("features_step_id"),
        task_name="features.run",
        label="validation.features_step_id",
        required=False,
    )
    if obs_step_id is None and feat_step_id is None:
        raise ConfigError("validation requires an observables or features step.")

    baseline_sim_cfg = _extract_sim_override(validation_cfg)
    if baseline_sim_cfg is None:
        baseline_sim_cfg = None
        for step in steps:
            if step.get("id") == sim_step_id:
                sim_cfg = step.get("sim")
                if not isinstance(sim_cfg, Mapping):
                    raise ConfigError("pipeline sim step must include a sim mapping.")
                baseline_sim_cfg = dict(sim_cfg)
                break
    if baseline_sim_cfg is None:
        raise ConfigError("baseline sim config could not be resolved.")

    baseline_sim_cfg = dict(baseline_sim_cfg)
    baseline_sim_cfg["mechanism"] = mechanism_path

    sim_task_name: Optional[str] = None
    sim_step_params: dict[str, Any] = {}
    for step in steps:
        if step.get("id") != sim_step_id:
            continue
        task_name = step.get("task")
        if isinstance(task_name, str) and task_name.strip():
            sim_task_name = task_name.strip()
        params_value = step.get("params")
        if isinstance(params_value, Mapping):
            sim_step_params = dict(params_value)
        break

    case_mode = _normalize_case_mode(validation_cfg.get("case_mode"))
    case_col = _extract_case_col(validation_cfg)
    conditions_path: Optional[Path] = None
    case_ids: list[Optional[str]]
    if case_mode == "all":
        if sim_task_name is not None and sim_task_name != "sim.run_csv":
            raise ConfigError(
                "validation.case_mode='all' requires sim.run_csv in the validation pipeline."
            )
        conditions_path = _extract_conditions_path(validation_cfg, steps, sim_step_id)
        case_ids = list(_load_case_ids_from_csv(conditions_path, case_col=case_col))
    else:
        default_case_id: Optional[str] = None
        for key in ("case_id", "condition_id", "case"):
            value = sim_step_params.get(key)
            if isinstance(value, str) and value.strip():
                default_case_id = value.strip()
                break
        case_ids = [default_case_id]

    def _apply_case_overrides(
        cfg_to_run: Mapping[str, Any],
        *,
        case_id: Optional[str],
    ) -> dict[str, Any]:
        updated = copy.deepcopy(cfg_to_run)
        for step in updated.get("steps", []):
            if step.get("id") != sim_step_id:
                continue
            if conditions_path is None and case_id is None:
                return updated
            params_value = step.get("params")
            if params_value is None:
                params_map: dict[str, Any] = {}
            elif isinstance(params_value, Mapping):
                params_map = dict(params_value)
            else:
                raise ConfigError("pipeline sim step params must be a mapping.")
            if conditions_path is not None:
                for key in ("conditions_file", "conditions_path", "conditions_csv", "csv"):
                    if key in params_map:
                        params_map[key] = str(conditions_path)
                        break
                else:
                    params_map["conditions_file"] = str(conditions_path)
            if case_id is not None:
                for key in ("case_id", "condition_id", "case"):
                    if key in params_map:
                        params_map[key] = case_id
                        break
                else:
                    params_map["case_id"] = case_id
            step["params"] = params_map
            break
        return updated

    baseline_pipeline_cfg = copy.deepcopy(pipeline_cfg)
    for step in baseline_pipeline_cfg.get("steps", []):
        if step.get("id") == sim_step_id:
            step["sim"] = dict(baseline_sim_cfg)
            break

    baseline_cases: list[dict[str, Any]] = []
    parents: list[str] = []
    for case_id in case_ids:
        baseline_results = runner.run(
            _apply_case_overrides(baseline_pipeline_cfg, case_id=case_id)
        )
        baseline_run_id = baseline_results.get(sim_step_id)
        if baseline_run_id is None:
            raise ConfigError("baseline sim step did not produce a run_id.")

        baseline_obs_id = baseline_results.get(obs_step_id) if obs_step_id else None
        baseline_feat_id = baseline_results.get(feat_step_id) if feat_step_id else None

        baseline_obs_groups: dict[tuple[str, str, str], list[Any]] = {}
        baseline_feat_groups: dict[tuple[str, str, str], list[Any]] = {}
        if obs_step_id:
            if baseline_obs_id is None:
                raise ConfigError("baseline observables step did not produce an artifact.")
            obs_rows = _extract_run_rows(
                store,
                kind="observables",
                artifact_id=baseline_obs_id,
                run_id=baseline_run_id,
                filename="values.parquet",
            )
            baseline_obs_groups = _group_metric_rows(obs_rows, name_field="observable")
            if not baseline_obs_groups:
                raise ConfigError("baseline observables produced no rows for comparison.")
        if feat_step_id:
            if baseline_feat_id is None:
                raise ConfigError("baseline features step did not produce an artifact.")
            feat_rows = _extract_run_rows(
                store,
                kind="features",
                artifact_id=baseline_feat_id,
                run_id=baseline_run_id,
                filename="features.parquet",
            )
            baseline_feat_groups = _group_metric_rows(feat_rows, name_field="feature")
            if not baseline_feat_groups:
                raise ConfigError("baseline features produced no rows for comparison.")

        baseline_cases.append(
            {
                "case_id": case_id,
                "baseline_run_id": baseline_run_id,
                "baseline_observables_id": baseline_obs_id,
                "baseline_features_id": baseline_feat_id,
                "obs_groups": baseline_obs_groups,
                "feat_groups": baseline_feat_groups,
            }
        )
        parents.append(baseline_run_id)
        if baseline_obs_id:
            parents.append(baseline_obs_id)
        if baseline_feat_id:
            parents.append(baseline_feat_id)

    patch_candidates = _normalize_patch_candidates(validation_cfg)
    metrics_rows: list[dict[str, Any]] = []
    patch_summaries: list[dict[str, Any]] = []
    selected_patch: Optional[dict[str, Any]] = None
    if not baseline_cases:
        raise ConfigError("baseline produced no cases for validation.")

    for index, candidate in enumerate(patch_candidates):
        reduction_result, patch_payload = _resolve_reduction_candidate(
            candidate,
            mechanism_path=mechanism_path,
            store=store,
            registry=registry,
        )
        reduction_id = reduction_result.manifest.id
        reduction_dir = reduction_result.path
        reduced_mechanism = reduction_dir / MECHANISM_FILENAME
        temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        if use_multipliers_only:
            reduced_mechanism = Path(mechanism_path)
            multipliers = _build_patch_multipliers_with_disabled(patch_payload)
        else:
            if not reduced_mechanism.exists():
                temp_dir = tempfile.TemporaryDirectory(prefix="rxn_reduction_")
                reduced_mechanism = Path(temp_dir.name) / MECHANISM_FILENAME
                _, combined_entries = _normalize_patch_payload(patch_payload)
                mechanism_payload = read_yaml_payload(Path(mechanism_path))
                if not isinstance(mechanism_payload, Mapping):
                    raise ConfigError("mechanism YAML must be a mapping.")
                reduced_payload, _ = _apply_patch_entries(
                    dict(mechanism_payload),
                    combined_entries,
                )
                write_yaml_payload(reduced_mechanism, reduced_payload, sort_keys=False)
            multipliers = _extract_patch_multipliers(patch_payload)

        reduced_sim_cfg = dict(baseline_sim_cfg)
        reduced_sim_cfg["mechanism"] = str(reduced_mechanism)
        if multipliers:
            reduced_sim_cfg["reaction_multipliers"] = multipliers
        else:
            reduced_sim_cfg.pop("reaction_multipliers", None)
        reduced_sim_cfg.pop("disabled_reactions", None)

        patch_rows: list[dict[str, Any]] = []
        patch_pass = True
        evaluated_total = 0
        reduced_run_ids: dict[str, str] = {}
        reduced_obs_ids: dict[str, str] = {}
        reduced_feat_ids: dict[str, str] = {}

        reduced_pipeline_cfg = copy.deepcopy(pipeline_cfg)
        for step in reduced_pipeline_cfg.get("steps", []):
            if step.get("id") == sim_step_id:
                step["sim"] = dict(reduced_sim_cfg)
                break

        try:
            for baseline_case in baseline_cases:
                case_id = baseline_case.get("case_id")
                reduced_results = runner.run(
                    _apply_case_overrides(
                        reduced_pipeline_cfg,
                        case_id=case_id if isinstance(case_id, str) else None,
                    )
                )
                reduced_run_id = reduced_results.get(sim_step_id)
                if reduced_run_id is None:
                    raise ConfigError("reduced sim step did not produce a run_id.")

                reduced_obs_id = reduced_results.get(obs_step_id) if obs_step_id else None
                reduced_feat_id = reduced_results.get(feat_step_id) if feat_step_id else None

                case_key = str(case_id) if case_id is not None else "default"
                reduced_run_ids[case_key] = str(reduced_run_id)
                if reduced_obs_id is not None:
                    reduced_obs_ids[case_key] = str(reduced_obs_id)
                if reduced_feat_id is not None:
                    reduced_feat_ids[case_key] = str(reduced_feat_id)

                if obs_step_id:
                    if reduced_obs_id is None:
                        raise ConfigError(
                            "reduced observables step did not produce an artifact."
                        )
                    reduced_obs_rows = _extract_run_rows(
                        store,
                        kind="observables",
                        artifact_id=reduced_obs_id,
                        run_id=reduced_run_id,
                        filename="values.parquet",
                    )
                    reduced_obs_groups = _group_metric_rows(
                        reduced_obs_rows, name_field="observable"
                    )
                    rows, passed, evaluated = _compare_grouped_values(
                        kind="observable",
                        baseline_groups=baseline_case.get("obs_groups", {}),
                        reduced_groups=reduced_obs_groups,
                        metric=metric,
                        tolerance=tolerance,
                        rel_eps=rel_eps,
                        missing_strategy=missing_strategy,
                        patch_index=index,
                        patch_id=reduction_id,
                        baseline_run_id=baseline_case["baseline_run_id"],
                        reduced_run_id=reduced_run_id,
                        baseline_artifact_id=baseline_case["baseline_observables_id"],
                        reduced_artifact_id=reduced_obs_id,
                    )
                    if case_id is not None:
                        for row in rows:
                            row["case_id"] = case_id
                    patch_rows.extend(rows)
                    patch_pass = patch_pass and passed
                    evaluated_total += evaluated

                if feat_step_id:
                    if reduced_feat_id is None:
                        raise ConfigError("reduced features step did not produce an artifact.")
                    reduced_feat_rows = _extract_run_rows(
                        store,
                        kind="features",
                        artifact_id=reduced_feat_id,
                        run_id=reduced_run_id,
                        filename="features.parquet",
                    )
                    reduced_feat_groups = _group_metric_rows(
                        reduced_feat_rows, name_field="feature"
                    )
                    rows, passed, evaluated = _compare_grouped_values(
                        kind="feature",
                        baseline_groups=baseline_case.get("feat_groups", {}),
                        reduced_groups=reduced_feat_groups,
                        metric=metric,
                        tolerance=tolerance,
                        rel_eps=rel_eps,
                        missing_strategy=missing_strategy,
                        patch_index=index,
                        patch_id=reduction_id,
                        baseline_run_id=baseline_case["baseline_run_id"],
                        reduced_run_id=reduced_run_id,
                        baseline_artifact_id=baseline_case["baseline_features_id"],
                        reduced_artifact_id=reduced_feat_id,
                    )
                    if case_id is not None:
                        for row in rows:
                            row["case_id"] = case_id
                    patch_rows.extend(rows)
                    patch_pass = patch_pass and passed
                    evaluated_total += evaluated

                parents.append(reduction_id)
                parents.append(reduced_run_id)
                if reduced_obs_id:
                    parents.append(reduced_obs_id)
                if reduced_feat_id:
                    parents.append(reduced_feat_id)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        if evaluated_total == 0:
            patch_pass = False

        metrics_rows.extend(patch_rows)
        summary = {
            "patch_index": index,
            "reduction_id": reduction_id,
            "passed": patch_pass,
            "run_id": next(iter(reduced_run_ids.values()), None),
            "observables": next(iter(reduced_obs_ids.values()), None),
            "features": next(iter(reduced_feat_ids.values()), None),
        }
        if case_mode == "all":
            summary["case_ids"] = [cid for cid in case_ids if cid is not None]
            summary["reduced_runs"] = dict(reduced_run_ids)
        patch_summaries.append(summary)
        if patch_pass:
            selected_patch = summary
        if stop_on_fail and not patch_pass:
            break

    if not metrics_rows:
        raise ConfigError("validation produced no comparison metrics.")

    first_case = baseline_cases[0]
    inputs_payload: dict[str, Any] = {
        "baseline_run_id": first_case["baseline_run_id"],
        "baseline_observables_id": first_case.get("baseline_observables_id"),
        "baseline_features_id": first_case.get("baseline_features_id"),
        "baseline_cases": [
            {
                "case_id": entry.get("case_id"),
                "baseline_run_id": entry.get("baseline_run_id"),
                "baseline_observables_id": entry.get("baseline_observables_id"),
                "baseline_features_id": entry.get("baseline_features_id"),
            }
            for entry in baseline_cases
        ],
        "patches": [
            {"patch_index": entry["patch_index"], "reduction_id": entry["reduction_id"]}
            for entry in patch_summaries
        ],
        "selected_patch": selected_patch,
        "passed": selected_patch is not None,
        "case_mode": case_mode,
        "metric": {
            "name": metric,
            "tolerance": tolerance,
            "rel_eps": rel_eps,
            "missing_strategy": missing_strategy,
        },
    }
    if conditions_path is not None:
        inputs_payload["conditions_file"] = str(conditions_path)
        inputs_payload["case_col"] = case_col

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="validation",
        artifact_id=artifact_id,
        parents=_dedupe_preserve(parents),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    created_at = _utc_now_iso()
    per_patch_stats: dict[int, dict[str, Any]] = {}
    worst_candidates: list[tuple[float, dict[str, Any]]] = []
    for row in metrics_rows:
        idx_raw = row.get("patch_index")
        if idx_raw is None or isinstance(idx_raw, bool):
            continue
        try:
            patch_index = int(idx_raw)
        except (TypeError, ValueError):
            continue
        stats = per_patch_stats.setdefault(
            patch_index,
            {
                "rows_total": 0,
                "skipped": 0,
                "evaluated": 0,
                "passed": 0,
                "rel_values": [],
                "abs_values": [],
            },
        )
        stats["rows_total"] += 1
        status = row.get("status")
        if status == "skipped":
            stats["skipped"] += 1
            continue
        stats["evaluated"] += 1
        if row.get("passed") is True:
            stats["passed"] += 1
        rel = row.get("rel_diff")
        try:
            rel_v = float(rel)
        except (TypeError, ValueError):
            rel_v = math.nan
        if math.isfinite(rel_v):
            stats["rel_values"].append(rel_v)
        abs_diff = row.get("abs_diff")
        try:
            abs_v = float(abs_diff)
        except (TypeError, ValueError):
            abs_v = math.nan
        if math.isfinite(abs_v):
            stats["abs_values"].append(abs_v)

        metric_value = abs_v if metric == "abs" else rel_v
        if math.isfinite(metric_value):
            worst_candidates.append((float(metric_value), dict(row)))

    patch_reports: list[dict[str, Any]] = []
    for summary in patch_summaries:
        patch_index = summary.get("patch_index")
        reduction_id = summary.get("reduction_id")
        if not isinstance(patch_index, int) or isinstance(patch_index, bool):
            continue
        if not isinstance(reduction_id, str) or not reduction_id.strip():
            continue
        stats = per_patch_stats.get(int(patch_index), {})
        evaluated = int(stats.get("evaluated") or 0)
        passed_count = int(stats.get("passed") or 0)
        pass_rate = float(passed_count) / float(evaluated) if evaluated > 0 else 0.0
        rel_vals = list(stats.get("rel_values") or [])
        abs_vals = list(stats.get("abs_values") or [])
        rel_vals = [float(v) for v in rel_vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        abs_vals = [float(v) for v in abs_vals if isinstance(v, (int, float)) and math.isfinite(float(v))]

        patch_reports.append(
            {
                "patch_index": int(patch_index),
                "reduction_id": reduction_id.strip(),
                "passed": bool(summary.get("passed")),
                "pass_rate": float(pass_rate),
                "evaluated": evaluated,
                "rows_total": int(stats.get("rows_total") or 0),
                "skipped": int(stats.get("skipped") or 0),
                "mean_rel_diff": float(sum(rel_vals) / len(rel_vals)) if rel_vals else None,
                "max_rel_diff": float(max(rel_vals)) if rel_vals else None,
                "mean_abs_diff": float(sum(abs_vals) / len(abs_vals)) if abs_vals else None,
                "max_abs_diff": float(max(abs_vals)) if abs_vals else None,
            }
        )

    worst_candidates.sort(key=lambda item: item[0], reverse=True)
    worst_rows: list[dict[str, Any]] = []
    for value, row in worst_candidates[:20]:
        entry = {
            "metric_value": float(value),
            "metric": metric,
            "patch_index": row.get("patch_index"),
            "patch_id": row.get("patch_id"),
            "case_id": row.get("case_id"),
            "kind": row.get("kind"),
            "name": row.get("name"),
            "unit": row.get("unit"),
            "status": row.get("status"),
            "passed": row.get("passed"),
            "baseline_value": row.get("baseline_value"),
            "reduced_value": row.get("reduced_value"),
            "abs_diff": row.get("abs_diff"),
            "rel_diff": row.get("rel_diff"),
        }
        worst_rows.append(entry)

    report_payload = {
        "schema_version": 1,
        "kind": "validation_report",
        "created_at": created_at,
        "level0": {
            "available": False,
            "note": "cheap metrics are not computed by reduction.validate; use features.reduction_cheap_metrics.",
        },
        "level1": {
            "available": True,
            "passed": selected_patch is not None,
            "selected_patch": selected_patch,
            "case_mode": case_mode,
            "metric": {
                "name": metric,
                "tolerance": tolerance,
                "rel_eps": rel_eps,
                "missing_strategy": missing_strategy,
            },
            "patches": patch_reports,
            "worst": worst_rows,
        },
        "level2": {
            "available": False,
            "note": "path-retention metrics are not computed by reduction.validate (future work).",
        },
    }
    if conditions_path is not None:
        report_payload["level1"]["conditions_file"] = str(conditions_path)
        report_payload["level1"]["case_col"] = case_col

    def _writer(base_dir: Path) -> None:
        _write_validation_table(metrics_rows, base_dir / "metrics.parquet")
        write_json_atomic(base_dir / "report.json", report_payload)

    return store.ensure(manifest, writer=_writer)


def repair_topk(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Select the best patch from a validation sweep and emit a repaired reduction artifact.

    This is a Stage3 "repair" primitive: it does not invent a new patch, it chooses an
    existing patch candidate (e.g., top-k sweep) according to a policy and re-emits
    it as a new reduction artifact with a `repair.json` log for traceability.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    inputs = _extract_inputs(reduction_cfg)

    validation_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("validation_id", "validation", "validation_artifact"),
        label="reduction.validation_id",
    )
    if validation_id is None:
        validation_id = _extract_optional_artifact_id(
            inputs,
            keys=("validation_id", "validation", "validation_artifact"),
            label="reduction.inputs.validation_id",
        )
    if validation_id is None:
        validation_id = _extract_optional_artifact_id(
            params,
            keys=("validation_id", "validation", "validation_artifact"),
            label="reduction.params.validation_id",
        )
    if validation_id is None:
        raise ConfigError("repair_topk requires validation_id.")
    validation_id = _require_nonempty_str(validation_id, "validation_id")

    policy_cfg = params.get("policy") or {}
    if policy_cfg is None:
        policy_cfg = {}
    if not isinstance(policy_cfg, Mapping):
        raise ConfigError("params.policy must be a mapping when provided.")
    policy = dict(policy_cfg)

    objective_raw = policy.get("objective", "disabled_reactions")
    if not isinstance(objective_raw, str) or not objective_raw.strip():
        raise ConfigError("policy.objective must be a non-empty string when provided.")
    objective = objective_raw.strip().lower()
    if objective not in {"disabled_reactions", "merged_species"}:
        raise ConfigError("policy.objective must be one of: disabled_reactions, merged_species.")

    target_pass_rate_raw = policy.get("target_pass_rate", 1.0)
    if isinstance(target_pass_rate_raw, bool):
        raise ConfigError("policy.target_pass_rate must be numeric.")
    try:
        target_pass_rate = float(target_pass_rate_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("policy.target_pass_rate must be numeric.") from exc
    target_pass_rate = max(0.0, min(1.0, target_pass_rate))

    target_max_rel_raw = policy.get("target_max_rel_diff")
    target_max_rel: Optional[float]
    if target_max_rel_raw is None:
        target_max_rel = None
    else:
        if isinstance(target_max_rel_raw, bool):
            raise ConfigError("policy.target_max_rel_diff must be numeric when provided.")
        try:
            target_max_rel = float(target_max_rel_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("policy.target_max_rel_diff must be numeric when provided.") from exc
        if not math.isfinite(target_max_rel):
            raise ConfigError("policy.target_max_rel_diff must be finite.")

    store.read_manifest("validation", validation_id)
    validation_manifest = store.read_manifest("validation", validation_id)
    table_path = store.artifact_dir("validation", validation_id) / "metrics.parquet"
    rows = _read_table_rows(table_path)

    patch_specs = validation_manifest.inputs.get("patches")
    patch_index_to_id: dict[int, str] = {}
    if isinstance(patch_specs, Sequence) and not isinstance(
        patch_specs, (str, bytes, bytearray)
    ):
        for entry in patch_specs:
            if not isinstance(entry, Mapping):
                continue
            patch_index = entry.get("patch_index")
            reduction_id = entry.get("reduction_id")
            if not isinstance(patch_index, int) or isinstance(patch_index, bool):
                continue
            if isinstance(reduction_id, str) and reduction_id.strip():
                patch_index_to_id[int(patch_index)] = reduction_id.strip()

    if not patch_index_to_id:
        raise ConfigError("validation artifact has no patches mapping to select from.")

    per_patch: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx_raw = row.get("patch_index")
        if idx_raw is None or isinstance(idx_raw, bool):
            continue
        try:
            patch_index = int(idx_raw)
        except (TypeError, ValueError):
            continue
        stats = per_patch.setdefault(
            patch_index,
            {"evaluated": 0, "passed": 0, "rel": [], "abs": []},
        )
        status = row.get("status")
        if status == "skipped":
            continue
        stats["evaluated"] += 1
        if row.get("passed") is True:
            stats["passed"] += 1
        rel = row.get("rel_diff")
        try:
            rel_v = float(rel)
        except (TypeError, ValueError):
            rel_v = math.nan
        if math.isfinite(rel_v):
            stats["rel"].append(rel_v)
        abs_diff = row.get("abs_diff")
        try:
            abs_v = float(abs_diff)
        except (TypeError, ValueError):
            abs_v = math.nan
        if math.isfinite(abs_v):
            stats["abs"].append(abs_v)

    def _disabled_count(reduction_id: str) -> int:
        patch_path = store.artifact_dir("reduction", reduction_id) / PATCH_FILENAME
        payload = read_yaml_payload(patch_path) if patch_path.exists() else {}
        if not isinstance(payload, Mapping):
            return 0
        disabled = payload.get("disabled_reactions") or []
        count = 0
        if isinstance(disabled, Mapping):
            count += len(disabled)
        elif isinstance(disabled, Sequence) and not isinstance(disabled, (str, bytes, bytearray)):
            count += len(disabled)
        multipliers = payload.get("reaction_multipliers") or []
        if isinstance(multipliers, Mapping):
            multipliers = [multipliers]
        if isinstance(multipliers, Sequence) and not isinstance(multipliers, (str, bytes, bytearray)):
            for entry in multipliers:
                if not isinstance(entry, Mapping):
                    continue
                try:
                    multiplier = float(entry.get("multiplier", 1.0))
                except (TypeError, ValueError):
                    continue
                if multiplier == 0.0:
                    count += 1
        return int(count)

    def _merged_species_count(reduction_id: str) -> int:
        patch_path = store.artifact_dir("reduction", reduction_id) / PATCH_FILENAME
        payload = read_yaml_payload(patch_path) if patch_path.exists() else {}
        if not isinstance(payload, Mapping):
            return 0
        state_merge = payload.get("state_merge")
        if not isinstance(state_merge, Mapping):
            return 0
        mapping = state_merge.get("species_to_representative")
        if not isinstance(mapping, Mapping):
            return 0
        return int(len(mapping))

    def _select_best(candidates: list[tuple[int, str]]) -> str:
        good: list[tuple[tuple[Any, ...], str]] = []
        fallback: list[tuple[tuple[float, float, int, int], str]] = []
        for patch_index, reduction_id in candidates:
            stats = per_patch.get(patch_index, {})
            evaluated = int(stats.get("evaluated") or 0)
            passed_count = int(stats.get("passed") or 0)
            pass_rate = (passed_count / evaluated) if evaluated > 0 else 0.0
            rel_vals = [float(v) for v in (stats.get("rel") or []) if math.isfinite(float(v))]
            mean_rel = (sum(rel_vals) / len(rel_vals)) if rel_vals else math.inf
            max_rel = max(rel_vals) if rel_vals else math.inf
            disabled = _disabled_count(reduction_id)
            merged = _merged_species_count(reduction_id)

            meets = pass_rate >= target_pass_rate - 1e-12
            if target_max_rel is not None:
                meets = meets and (max_rel <= target_max_rel + 1e-12)

            if meets:
                if objective == "merged_species":
                    # Prefer maximum merged, then minimum mean_rel, then minimum max_rel.
                    good.append(
                        (
                            (int(-merged), float(mean_rel), float(max_rel), int(patch_index)),
                            reduction_id,
                        )
                    )
                else:
                    # Prefer maximum disabled, then minimum mean_rel, then stable tie-break.
                    good.append(
                        (
                            (int(-disabled), float(mean_rel), float(max_rel), int(patch_index)),
                            reduction_id,
                        )
                    )
            else:
                # Otherwise: maximize pass_rate, then minimize mean_rel, then maximize objective.
                obj_score = merged if objective == "merged_species" else disabled
                fallback.append(
                    (
                        (
                            float(-pass_rate),
                            float(mean_rel),
                            int(-obj_score),
                            int(patch_index),
                        ),
                        reduction_id,
                    )
                )

        if good:
            good.sort(key=lambda item: item[0])
            return good[0][1]
        fallback.sort(key=lambda item: item[0])
        return fallback[0][1]

    candidates = sorted(patch_index_to_id.items(), key=lambda item: item[0])
    selected_reduction_id = _select_best([(idx, rid) for idx, rid in candidates])

    store.read_manifest("reduction", selected_reduction_id)
    selected_dir = store.artifact_dir("reduction", selected_reduction_id)
    patch_path = selected_dir / PATCH_FILENAME
    if not patch_path.exists():
        raise ConfigError(f"Selected reduction patch is missing: reduction/{selected_reduction_id}/{PATCH_FILENAME}")

    selected_patch_payload = read_yaml_payload(patch_path)
    if not isinstance(selected_patch_payload, Mapping):
        raise ConfigError("Selected mechanism_patch.yaml must be a mapping.")
    selected_patch_payload = dict(selected_patch_payload)

    mechanism_payload = None
    mechanism_path = selected_dir / MECHANISM_FILENAME
    if mechanism_path.exists():
        raw = read_yaml_payload(mechanism_path)
        if isinstance(raw, Mapping):
            mechanism_payload = dict(raw)

    merged_species = 0
    state_merge = selected_patch_payload.get("state_merge")
    if isinstance(state_merge, Mapping):
        mapping = state_merge.get("species_to_representative")
        if isinstance(mapping, Mapping):
            merged_species = len(mapping)

    counts = {
        "species_before": None,
        "species_after": None,
        "merged_species": int(merged_species),
        "reactions_before": None,
        "reactions_after": None,
        "disabled_reactions": _disabled_count(selected_reduction_id),
        "merged_reactions": 0,
    }

    repair_payload = {
        "schema_version": 1,
        "kind": "repair_topk",
        "validation_id": validation_id,
        "selected_reduction_id": selected_reduction_id,
        "policy": {
            "objective": objective,
            "target_pass_rate": float(target_pass_rate),
            "target_max_rel_diff": float(target_max_rel) if target_max_rel is not None else None,
        },
    }

    inputs_payload = {
        "mode": "repair_topk",
        "validation_id": validation_id,
        "selected_reduction_id": selected_reduction_id,
        "policy": repair_payload["policy"],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[validation_id, selected_reduction_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, selected_patch_payload, sort_keys=True)
        if mechanism_payload is not None:
            write_yaml_payload(base_dir / MECHANISM_FILENAME, mechanism_payload, sort_keys=False)
        write_json_atomic(base_dir / "repair.json", repair_payload)
        write_json_atomic(
            base_dir / "metrics.json",
            {
                "schema_version": 1,
                "kind": "repair_topk_metrics",
                "counts": counts,
                "selected_reduction_id": selected_reduction_id,
                "validation_id": validation_id,
                "objective": objective,
            },
        )

    return store.ensure(manifest, writer=_writer)


def repair_restore(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Restore disabled reactions for failing cases using per-case importance.

    This is a Stage3 repair primitive intended for multi-case validation:
    start from an aggressive patch, validate across cases, then restore a small
    number of high-importance reactions *only* for the failing cases.

    Notes:
    - This task does not re-run validation; it consumes an existing validation
      artifact (metrics table) and produces a new patch.
    - Importance is taken from a features artifact that includes per-run rows
      (e.g., rop_wdot_summary or gnn_importance), and mapped via a run_set's
      case_id -> run_id table.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    inputs = _extract_inputs(reduction_cfg)

    base_reduction_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("base_reduction_id", "base_patch_id", "patch_id", "reduction_id"),
        label="reduction.base_reduction_id",
    )
    if base_reduction_id is None:
        base_reduction_id = _extract_optional_artifact_id(
            inputs,
            keys=("base_reduction_id", "base_patch_id", "patch_id", "reduction_id"),
            label="reduction.inputs.base_reduction_id",
        )
    if base_reduction_id is None:
        raise ConfigError("repair_restore requires base_reduction_id.")
    base_reduction_id = _require_nonempty_str(base_reduction_id, "base_reduction_id")

    validation_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("validation_id", "validation", "validation_artifact"),
        label="reduction.validation_id",
    )
    if validation_id is None:
        validation_id = _extract_optional_artifact_id(
            inputs,
            keys=("validation_id", "validation", "validation_artifact"),
            label="reduction.inputs.validation_id",
        )
    if validation_id is None:
        raise ConfigError("repair_restore requires validation_id.")
    validation_id = _require_nonempty_str(validation_id, "validation_id")

    features_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("features_id", "features", "importance_features_id"),
        label="reduction.features_id",
    )
    if features_id is None:
        features_id = _extract_optional_artifact_id(
            inputs,
            keys=("features_id", "features", "importance_features_id"),
            label="reduction.inputs.features_id",
        )
    if features_id is None:
        raise ConfigError("repair_restore requires features_id.")
    features_id = _require_nonempty_str(features_id, "features_id")

    run_set_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("run_set_id", "run_set", "run_sets"),
        label="reduction.run_set_id",
    )
    if run_set_id is None:
        run_set_id = _extract_optional_artifact_id(
            inputs,
            keys=("run_set_id", "run_set", "run_sets"),
            label="reduction.inputs.run_set_id",
        )
    if run_set_id is None:
        raise ConfigError("repair_restore requires run_set_id.")
    run_set_id = _require_nonempty_str(run_set_id, "run_set_id")

    target_patch_id = _require_nonempty_str(
        str(params.get("target_patch_id") or params.get("patch_id") or base_reduction_id),
        "params.target_patch_id",
    )

    restore_per_case_raw = params.get("restore_per_case", 5)
    if isinstance(restore_per_case_raw, bool):
        raise ConfigError("params.restore_per_case must be an integer.")
    try:
        restore_per_case = int(restore_per_case_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("params.restore_per_case must be an integer.") from exc
    if restore_per_case <= 0:
        raise ConfigError("params.restore_per_case must be positive.")

    max_total_raw = params.get("max_total_restored", None)
    max_total: Optional[int]
    if max_total_raw is None:
        max_total = None
    else:
        if isinstance(max_total_raw, bool):
            raise ConfigError("params.max_total_restored must be an integer when provided.")
        try:
            max_total = int(max_total_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("params.max_total_restored must be an integer when provided.") from exc
        if max_total <= 0:
            raise ConfigError("params.max_total_restored must be positive when provided.")

    importance_mode = str(params.get("importance_mode", "abs")).strip().lower()
    if importance_mode in ("abs", "absolute"):
        importance_mode = "abs"
    elif importance_mode in ("raw", "value"):
        importance_mode = "raw"
    else:
        raise ConfigError("params.importance_mode must be 'abs' or 'raw'.")

    store.read_manifest("reduction", base_reduction_id)
    base_patch_path = store.artifact_dir("reduction", base_reduction_id) / PATCH_FILENAME
    if not base_patch_path.exists():
        raise ConfigError(f"Base patch missing: reduction/{base_reduction_id}/{PATCH_FILENAME}")
    base_patch_payload = read_yaml_payload(base_patch_path)
    if not isinstance(base_patch_payload, Mapping):
        raise ConfigError("Base mechanism_patch.yaml must be a mapping.")
    base_patch_payload = dict(base_patch_payload)

    disabled_entries = base_patch_payload.get("disabled_reactions") or []
    if not isinstance(disabled_entries, Sequence) or isinstance(
        disabled_entries, (str, bytes, bytearray)
    ):
        raise ConfigError("disabled_reactions must be a list.")
    disabled_indices: list[int] = []
    for entry in disabled_entries:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("index")
        if idx is None or isinstance(idx, bool):
            continue
        try:
            disabled_indices.append(int(idx))
        except (TypeError, ValueError):
            continue
    disabled_set = set(disabled_indices)
    if not disabled_set:
        raise ConfigError("Base patch has no disabled reactions to restore from.")

    store.read_manifest("validation", validation_id)
    val_rows = _read_table_rows(store.artifact_dir("validation", validation_id) / "metrics.parquet")
    failing_case_ids: list[str] = []
    seen_cases: set[str] = set()
    for row in val_rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("patch_id") != target_patch_id:
            continue
        if row.get("passed") is True:
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            continue
        case_id = case_id.strip()
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        failing_case_ids.append(case_id)

    if not failing_case_ids:
        raise ConfigError(
            "repair_restore found no failing cases for the target patch. "
            "If the patch already passes, this task is unnecessary."
        )

    runs_json_path = store.artifact_dir("run_sets", run_set_id) / "runs.json"
    if not runs_json_path.exists():
        raise ConfigError(f"run_sets/{run_set_id}/runs.json is missing.")
    try:
        runs_payload = json.loads(runs_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError("run_sets runs.json must be valid JSON.") from exc
    case_to_run = runs_payload.get("case_to_run")
    if not isinstance(case_to_run, Mapping):
        raise ConfigError("run_sets runs.json must contain case_to_run mapping.")

    store.read_manifest("features", features_id)
    features_dir = store.artifact_dir("features", features_id)
    features_path = features_dir / "features.parquet"
    if not features_path.exists():
        # Some tests/dummy artifacts may store JSON rows in features.json.
        alt = features_dir / "features.json"
        if alt.exists():
            features_path = alt
        else:
            raise ConfigError(f"features/{features_id}/features.parquet is missing.")
    feat_rows = _read_table_rows(features_path)

    # Build per-run reaction importance list: [(abs(score), score, reaction_index), ...]
    per_run: dict[str, list[tuple[float, float, int]]] = {}
    for row in feat_rows:
        if not isinstance(row, Mapping):
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        meta_raw = row.get("meta_json") or "{}"
        if not isinstance(meta_raw, str):
            continue
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
        rxn_index = meta.get("reaction_index")
        if rxn_index is None or isinstance(rxn_index, bool):
            continue
        try:
            rxn_index_i = int(rxn_index)
        except (TypeError, ValueError):
            continue
        value = row.get("value")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        key = run_id.strip()
        per_run.setdefault(key, []).append((abs(score), score, rxn_index_i))

    for run_id, items in per_run.items():
        # Sort descending by |score|.
        items.sort(key=lambda t: (-t[0], -t[1]))
        per_run[run_id] = items

    restored_indices: list[int] = []
    restored_by_case: dict[str, list[int]] = {}
    restored_set: set[int] = set()

    for case_id in failing_case_ids:
        run_id = case_to_run.get(case_id)
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        run_id = run_id.strip()
        items = per_run.get(run_id) or []
        chosen: list[int] = []
        for abs_score, score, rxn_index_i in items:
            if importance_mode == "raw" and score == 0.0:
                continue
            if rxn_index_i not in disabled_set:
                continue
            if rxn_index_i in restored_set:
                continue
            chosen.append(rxn_index_i)
            restored_set.add(rxn_index_i)
            restored_indices.append(rxn_index_i)
            if len(chosen) >= restore_per_case:
                break
            if max_total is not None and len(restored_indices) >= max_total:
                break
        if chosen:
            restored_by_case[case_id] = chosen
        if max_total is not None and len(restored_indices) >= max_total:
            break

    if not restored_indices:
        raise ConfigError(
            "repair_restore could not find any reactions to restore. "
            "This likely means the importance features do not cover the failing cases "
            "or the base patch disabled reactions are not present in the importance ranking."
        )

    # Apply restoration: remove restored indices from disabled entries (by index).
    restored_set_final = set(restored_indices)
    new_disabled_entries: list[dict[str, Any]] = []
    for entry in disabled_entries:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("index")
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_i in restored_set_final:
            continue
        new_disabled_entries.append({"index": idx_i})
    new_disabled_entries.sort(key=lambda d: int(d.get("index", 0)))

    new_patch_payload = dict(base_patch_payload)
    new_patch_payload["disabled_reactions"] = new_disabled_entries

    repair_payload = {
        "schema_version": 1,
        "kind": "repair_restore",
        "base_reduction_id": base_reduction_id,
        "validation_id": validation_id,
        "features_id": features_id,
        "run_set_id": run_set_id,
        "target_patch_id": target_patch_id,
        "failing_cases": failing_case_ids,
        "restore_per_case": int(restore_per_case),
        "max_total_restored": int(max_total) if max_total is not None else None,
        "restored_indices": sorted(restored_indices),
        "restored_by_case": restored_by_case,
    }

    inputs_payload = {
        "mode": "repair_restore",
        "base_reduction_id": base_reduction_id,
        "validation_id": validation_id,
        "features_id": features_id,
        "run_set_id": run_set_id,
        "target_patch_id": target_patch_id,
        "restore_per_case": int(restore_per_case),
        "max_total_restored": int(max_total) if max_total is not None else None,
        "importance_mode": importance_mode,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[base_reduction_id, validation_id, features_id, run_set_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, new_patch_payload, sort_keys=True)
        write_json_atomic(base_dir / "repair.json", repair_payload)
        write_json_atomic(
            base_dir / "metrics.json",
            {
                "schema_version": 1,
                "kind": "repair_restore_metrics",
                "counts": {
                    "disabled_reactions_before": int(len(disabled_set)),
                    "disabled_reactions_after": int(len(new_disabled_entries)),
                    "restored_reactions": int(len(restored_set_final)),
                },
                "restored_indices": sorted(restored_indices),
                "failing_cases": failing_case_ids,
            },
        )

    return store.ensure(manifest, writer=_writer)


def _extract_qoi_species_from_name(name: Any) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None
    text = name.strip()
    gas_match = re.search(r"(?:^|[.])gas[.]([A-Za-z0-9_()+\-]+)[.]", text)
    if gas_match:
        species = gas_match.group(1).strip()
        return species or None
    if text.startswith("qoi."):
        label = text[len("qoi.") :].strip()
        for suffix in (
            "_final_super",
            "_final",
            "_max_super",
            "_max",
            "_super",
        ):
            if label.endswith(suffix):
                label = label[: -len(suffix)]
                break
        if not label:
            return None
        # qoi.CO2_final -> CO2, qoi.ignition_delay -> None
        head = label.split("_", 1)[0]
        if not head:
            return None
        if head.lower() in {"ignition", "delay", "t", "temperature"}:
            return None
        return head
    return None


def _build_patch_index_map(validation_manifest: ArtifactManifest) -> dict[int, str]:
    mapping: dict[int, str] = {}
    patch_specs = validation_manifest.inputs.get("patches")
    if not isinstance(patch_specs, Sequence) or isinstance(
        patch_specs, (str, bytes, bytearray)
    ):
        return mapping
    for entry in patch_specs:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("patch_index")
        rid = entry.get("reduction_id")
        if isinstance(idx, int) and not isinstance(idx, bool):
            if isinstance(rid, str) and rid.strip():
                mapping[int(idx)] = rid.strip()
    return mapping


def _resolve_patch_id_for_row(
    row: Mapping[str, Any],
    *,
    patch_index_to_id: Mapping[int, str],
) -> Optional[str]:
    patch_id = row.get("patch_id")
    if isinstance(patch_id, str) and patch_id.strip():
        return patch_id.strip()
    idx = row.get("patch_index")
    if idx is None or isinstance(idx, bool):
        return None
    try:
        idx_i = int(idx)
    except (TypeError, ValueError):
        return None
    return patch_index_to_id.get(idx_i)


def _collect_failing_cases_and_targets(
    *,
    store: ArtifactStore,
    validation_id: str,
    target_patch_id: str,
    default_qoi_species: Sequence[str],
) -> tuple[list[str], dict[str, set[str]], list[dict[str, Any]]]:
    validation_manifest = store.read_manifest("validation", validation_id)
    patch_index_to_id = _build_patch_index_map(validation_manifest)
    rows = _read_table_rows(store.artifact_dir("validation", validation_id) / "metrics.parquet")

    case_order: list[str] = []
    seen_cases: set[str] = set()
    targets_by_case: dict[str, set[str]] = {}
    worst_rows: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        resolved_patch = _resolve_patch_id_for_row(row, patch_index_to_id=patch_index_to_id)
        if resolved_patch != target_patch_id:
            continue
        if row.get("passed") is True:
            continue
        case_id_raw = row.get("case_id")
        if not isinstance(case_id_raw, str) or not case_id_raw.strip():
            continue
        case_id = case_id_raw.strip()
        if case_id not in seen_cases:
            seen_cases.add(case_id)
            case_order.append(case_id)
        species = _extract_qoi_species_from_name(row.get("name"))
        if species:
            targets_by_case.setdefault(case_id, set()).add(species)

    report_path = store.artifact_dir("validation", validation_id) / "report.json"
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_payload = {}
        level1 = report_payload.get("level1")
        if isinstance(level1, Mapping):
            worst = level1.get("worst")
            if isinstance(worst, Sequence) and not isinstance(
                worst, (str, bytes, bytearray)
            ):
                for entry in worst:
                    if not isinstance(entry, Mapping):
                        continue
                    patch_id = entry.get("patch_id")
                    if isinstance(patch_id, str) and patch_id.strip():
                        if patch_id.strip() != target_patch_id:
                            continue
                    case_id = entry.get("case_id")
                    if not isinstance(case_id, str) or not case_id.strip():
                        continue
                    case_id = case_id.strip()
                    if case_id not in seen_cases:
                        seen_cases.add(case_id)
                        case_order.append(case_id)
                    species = _extract_qoi_species_from_name(entry.get("name"))
                    if species:
                        targets_by_case.setdefault(case_id, set()).add(species)
                    worst_rows.append(dict(entry))

    defaults = {str(item).strip() for item in default_qoi_species if str(item).strip()}
    if defaults:
        for case_id in case_order:
            targets_by_case.setdefault(case_id, set()).update(defaults)

    return case_order, targets_by_case, worst_rows


def _load_case_to_run_mapping(store: ArtifactStore, run_set_id: str) -> dict[str, str]:
    runs_json_path = store.artifact_dir("run_sets", run_set_id) / "runs.json"
    if not runs_json_path.exists():
        raise ConfigError(f"run_sets/{run_set_id}/runs.json is missing.")
    try:
        payload = json.loads(runs_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError("run_sets runs.json must be valid JSON.") from exc
    case_to_run_raw = payload.get("case_to_run")
    if not isinstance(case_to_run_raw, Mapping):
        raise ConfigError("run_sets runs.json must contain case_to_run mapping.")
    case_to_run: dict[str, str] = {}
    for key, value in case_to_run_raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        case_to_run[key.strip()] = value.strip()
    return case_to_run


def _load_reaction_scores_by_run(
    *,
    store: ArtifactStore,
    features_id: str,
    importance_mode: str,
) -> dict[str, dict[int, float]]:
    store.read_manifest("features", features_id)
    features_dir = store.artifact_dir("features", features_id)
    features_path = features_dir / "features.parquet"
    if not features_path.exists():
        alt = features_dir / "features.json"
        if alt.exists():
            features_path = alt
        else:
            raise ConfigError(f"features/{features_id}/features.parquet is missing.")
    rows = _read_table_rows(features_path)

    per_run: dict[str, dict[int, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        meta_raw = row.get("meta_json") or "{}"
        if not isinstance(meta_raw, str):
            continue
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
        reaction_index = meta.get("reaction_index")
        if reaction_index is None or isinstance(reaction_index, bool):
            continue
        try:
            r_idx = int(reaction_index)
        except (TypeError, ValueError):
            continue
        value = row.get("value")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        if importance_mode == "abs":
            score_v = abs(score)
        else:
            score_v = score
            if score_v <= 0.0:
                continue
        key = run_id.strip()
        run_scores = per_run.setdefault(key, {})
        previous = run_scores.get(r_idx)
        if previous is None or score_v > previous:
            run_scores[r_idx] = float(score_v)

    return per_run


def _load_species_reaction_index_map(
    *,
    store: ArtifactStore,
    graph_id: str,
) -> dict[str, set[int]]:
    nodes, links = _load_graph_nodes_and_links(store, graph_id)
    reaction_entries = _prepare_reaction_entries(nodes, links)
    mapping: dict[str, set[int]] = {}
    for entry in reaction_entries:
        if not isinstance(entry, Mapping):
            continue
        r_idx = entry.get("reaction_index")
        if not isinstance(r_idx, int) or isinstance(r_idx, bool):
            continue
        participants: set[str] = set()
        for key in ("reactants", "products", "participants"):
            raw = entry.get(key) or set()
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                for name in raw:
                    if isinstance(name, str) and name.strip():
                        participants.add(name.strip())
            elif isinstance(raw, set):
                for name in raw:
                    if isinstance(name, str) and name.strip():
                        participants.add(name.strip())
        for species in participants:
            mapping.setdefault(species, set()).add(int(r_idx))
    return mapping


def repair_cover_restore(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Targeted restore for failing multi-case QoI via greedy case-cover selection."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    inputs = _extract_inputs(reduction_cfg)

    base_reduction_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("base_reduction_id", "base_patch_id", "patch_id", "reduction_id"),
        label="reduction.base_reduction_id",
    )
    if base_reduction_id is None:
        base_reduction_id = _extract_optional_artifact_id(
            inputs,
            keys=("base_reduction_id", "base_patch_id", "patch_id", "reduction_id"),
            label="reduction.inputs.base_reduction_id",
        )
    if base_reduction_id is None:
        raise ConfigError("repair_cover_restore requires base_reduction_id.")
    base_reduction_id = _require_nonempty_str(base_reduction_id, "base_reduction_id")

    validation_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("validation_id", "validation", "validation_artifact"),
        label="reduction.validation_id",
    )
    if validation_id is None:
        validation_id = _extract_optional_artifact_id(
            inputs,
            keys=("validation_id", "validation", "validation_artifact"),
            label="reduction.inputs.validation_id",
        )
    if validation_id is None:
        raise ConfigError("repair_cover_restore requires validation_id.")
    validation_id = _require_nonempty_str(validation_id, "validation_id")

    features_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("features_id", "features", "importance_features_id"),
        label="reduction.features_id",
    )
    if features_id is None:
        features_id = _extract_optional_artifact_id(
            inputs,
            keys=("features_id", "features", "importance_features_id"),
            label="reduction.inputs.features_id",
        )
    if features_id is None:
        raise ConfigError("repair_cover_restore requires features_id.")
    features_id = _require_nonempty_str(features_id, "features_id")

    run_set_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("run_set_id", "run_set", "run_sets"),
        label="reduction.run_set_id",
    )
    if run_set_id is None:
        run_set_id = _extract_optional_artifact_id(
            inputs,
            keys=("run_set_id", "run_set", "run_sets"),
            label="reduction.inputs.run_set_id",
        )
    if run_set_id is None:
        raise ConfigError("repair_cover_restore requires run_set_id.")
    run_set_id = _require_nonempty_str(run_set_id, "run_set_id")

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph_id", "graph", "mechanism_graph_id"),
        label="reduction.graph_id",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            inputs,
            keys=("graph_id", "graph", "mechanism_graph_id"),
            label="reduction.inputs.graph_id",
        )
    if graph_id is None:
        raise ConfigError("repair_cover_restore requires graph_id.")
    graph_id = _require_nonempty_str(graph_id, "graph_id")

    target_patch_id = _require_nonempty_str(
        str(params.get("target_patch_id") or params.get("patch_id") or base_reduction_id),
        "params.target_patch_id",
    )
    importance_mode = str(params.get("importance_mode", "abs")).strip().lower()
    if importance_mode in {"abs", "absolute"}:
        importance_mode = "abs"
    elif importance_mode in {"raw", "value"}:
        importance_mode = "raw"
    else:
        raise ConfigError("params.importance_mode must be 'abs' or 'raw'.")

    max_total_raw = params.get("max_total_restored", 40)
    if isinstance(max_total_raw, bool):
        raise ConfigError("params.max_total_restored must be numeric.")
    try:
        max_total = int(max_total_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("params.max_total_restored must be an integer.") from exc
    if max_total <= 0:
        raise ConfigError("params.max_total_restored must be positive.")

    fallback_per_case_raw = params.get("fallback_restore_per_case", params.get("restore_per_case", 0))
    if isinstance(fallback_per_case_raw, bool):
        raise ConfigError("params.fallback_restore_per_case must be an integer.")
    try:
        fallback_per_case = int(fallback_per_case_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("params.fallback_restore_per_case must be an integer.") from exc
    if fallback_per_case < 0:
        raise ConfigError("params.fallback_restore_per_case must be >= 0.")

    max_candidates_raw = params.get("max_candidates_per_case", 200)
    if isinstance(max_candidates_raw, bool):
        raise ConfigError("params.max_candidates_per_case must be an integer.")
    try:
        max_candidates_per_case = int(max_candidates_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("params.max_candidates_per_case must be an integer.") from exc
    if max_candidates_per_case <= 0:
        raise ConfigError("params.max_candidates_per_case must be positive.")

    qoi_species_raw = params.get("qoi_species")
    if qoi_species_raw is None:
        qoi_species_default: list[str] = []
    elif isinstance(qoi_species_raw, str):
        qoi_species_default = [_require_nonempty_str(qoi_species_raw, "qoi_species")]
    elif isinstance(qoi_species_raw, Sequence) and not isinstance(
        qoi_species_raw, (str, bytes, bytearray)
    ):
        qoi_species_default = [
            _require_nonempty_str(entry, "qoi_species")
            for entry in qoi_species_raw
            if entry is not None
        ]
    else:
        raise ConfigError("qoi_species must be a string or a list of strings.")

    store.read_manifest("reduction", base_reduction_id)
    base_patch_path = store.artifact_dir("reduction", base_reduction_id) / PATCH_FILENAME
    if not base_patch_path.exists():
        raise ConfigError(f"Base patch missing: reduction/{base_reduction_id}/{PATCH_FILENAME}")
    base_patch_payload = read_yaml_payload(base_patch_path)
    if not isinstance(base_patch_payload, Mapping):
        raise ConfigError("Base mechanism_patch.yaml must be a mapping.")
    base_patch_payload = dict(base_patch_payload)

    disabled_entries = base_patch_payload.get("disabled_reactions") or []
    if not isinstance(disabled_entries, Sequence) or isinstance(
        disabled_entries, (str, bytes, bytearray)
    ):
        raise ConfigError("disabled_reactions must be a list.")
    disabled_indices: list[int] = []
    for entry in disabled_entries:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("index")
        if idx is None or isinstance(idx, bool):
            continue
        try:
            disabled_indices.append(int(idx))
        except (TypeError, ValueError):
            continue
    disabled_set = set(disabled_indices)
    if not disabled_set:
        raise ConfigError("Base patch has no disabled reactions to restore from.")

    failing_case_ids, targets_by_case, worst_rows = _collect_failing_cases_and_targets(
        store=store,
        validation_id=validation_id,
        target_patch_id=target_patch_id,
        default_qoi_species=qoi_species_default,
    )
    if not failing_case_ids:
        raise ConfigError(
            "repair_cover_restore found no failing cases for the target patch. "
            "If the patch already passes, this task is unnecessary."
        )

    case_to_run = _load_case_to_run_mapping(store, run_set_id)
    scores_by_run = _load_reaction_scores_by_run(
        store=store,
        features_id=features_id,
        importance_mode=importance_mode,
    )
    species_to_reactions = _load_species_reaction_index_map(store=store, graph_id=graph_id)

    case_candidates: dict[str, list[tuple[int, float]]] = {}
    for case_id in failing_case_ids:
        run_id = case_to_run.get(case_id)
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        run_scores = scores_by_run.get(run_id.strip()) or {}
        if not run_scores:
            continue
        target_species = targets_by_case.get(case_id, set())
        allowed_reactions: Optional[set[int]] = None
        if target_species:
            union: set[int] = set()
            for species in target_species:
                union.update(species_to_reactions.get(species, set()))
            if union:
                allowed_reactions = union
        candidates: list[tuple[int, float]] = []
        for reaction_index, score in run_scores.items():
            if reaction_index not in disabled_set:
                continue
            if allowed_reactions is not None and reaction_index not in allowed_reactions:
                continue
            candidates.append((int(reaction_index), float(score)))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if candidates:
            case_candidates[case_id] = candidates[:max_candidates_per_case]

    if not case_candidates:
        raise ConfigError(
            "repair_cover_restore has no candidate reactions for failing cases. "
            "Check graph_id/features_id alignment and qoi_species filters."
        )

    reaction_to_cases: dict[int, set[str]] = {}
    reaction_score_sum: dict[int, float] = {}
    for case_id, candidates in case_candidates.items():
        for reaction_index, score in candidates:
            reaction_to_cases.setdefault(reaction_index, set()).add(case_id)
            reaction_score_sum[reaction_index] = reaction_score_sum.get(reaction_index, 0.0) + float(score)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_by_case: dict[str, list[int]] = {case_id: [] for case_id in case_candidates}
    uncovered: set[str] = set(case_candidates.keys())

    while uncovered and len(selected) < max_total:
        best_reaction: Optional[int] = None
        best_cover = -1
        best_score = -math.inf
        for reaction_index, covered_cases in reaction_to_cases.items():
            if reaction_index in selected_set:
                continue
            cover = len(uncovered.intersection(covered_cases))
            if cover <= 0:
                continue
            score = float(reaction_score_sum.get(reaction_index, 0.0))
            if cover > best_cover:
                best_cover = cover
                best_score = score
                best_reaction = reaction_index
                continue
            if cover == best_cover:
                if score > best_score or (
                    math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
                    and (best_reaction is None or reaction_index < best_reaction)
                ):
                    best_score = score
                    best_reaction = reaction_index
        if best_reaction is None:
            break
        selected.append(int(best_reaction))
        selected_set.add(int(best_reaction))
        for case_id in list(uncovered):
            if case_id in reaction_to_cases.get(best_reaction, set()):
                selected_by_case.setdefault(case_id, []).append(int(best_reaction))
                uncovered.remove(case_id)

    if fallback_per_case > 0 and len(selected) < max_total:
        # Ensure each failing case has at least N restored reactions, even if
        # greedy cover already marked it as "covered" by one shared reaction.
        for case_id in failing_case_ids:
            if len(selected) >= max_total:
                break
            candidates = case_candidates.get(case_id) or []
            already = len(selected_by_case.get(case_id, []))
            need = max(0, fallback_per_case - already)
            if need <= 0:
                continue
            for reaction_index, _score in candidates:
                if len(selected) >= max_total:
                    break
                if reaction_index in selected_set:
                    continue
                selected.append(int(reaction_index))
                selected_set.add(int(reaction_index))
                selected_by_case.setdefault(case_id, []).append(int(reaction_index))
                need -= 1
                if need <= 0:
                    break

    if not selected:
        raise ConfigError(
            "repair_cover_restore could not select any reaction to restore. "
            "Try increasing max_candidates_per_case or relaxing qoi_species filtering."
        )

    new_disabled_entries: list[dict[str, Any]] = []
    for entry in disabled_entries:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("index")
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_i in selected_set:
            continue
        new_disabled_entries.append({"index": idx_i})
    new_disabled_entries.sort(key=lambda item: int(item.get("index", 0)))

    new_patch_payload = dict(base_patch_payload)
    new_patch_payload["disabled_reactions"] = new_disabled_entries

    repair_payload = {
        "schema_version": 1,
        "kind": "repair_cover_restore",
        "base_reduction_id": base_reduction_id,
        "validation_id": validation_id,
        "features_id": features_id,
        "run_set_id": run_set_id,
        "graph_id": graph_id,
        "target_patch_id": target_patch_id,
        "failing_cases": failing_case_ids,
        "targets_by_case": {key: sorted(value) for key, value in targets_by_case.items()},
        "selected_reactions": sorted(selected_set),
        "selected_by_case": {key: sorted(value) for key, value in selected_by_case.items() if value},
        "uncovered_cases_after_greedy": sorted(uncovered),
        "params": {
            "importance_mode": importance_mode,
            "max_total_restored": int(max_total),
            "fallback_restore_per_case": int(fallback_per_case),
            "max_candidates_per_case": int(max_candidates_per_case),
            "qoi_species": qoi_species_default,
        },
        "worst_rows_sample": worst_rows[:20],
    }

    inputs_payload = {
        "mode": "repair_cover_restore",
        "base_reduction_id": base_reduction_id,
        "validation_id": validation_id,
        "features_id": features_id,
        "run_set_id": run_set_id,
        "graph_id": graph_id,
        "target_patch_id": target_patch_id,
        "importance_mode": importance_mode,
        "max_total_restored": int(max_total),
        "fallback_restore_per_case": int(fallback_per_case),
        "max_candidates_per_case": int(max_candidates_per_case),
        "qoi_species": qoi_species_default,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[base_reduction_id, validation_id, features_id, run_set_id, graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, new_patch_payload, sort_keys=True)
        write_json_atomic(base_dir / "repair.json", repair_payload)
        write_json_atomic(
            base_dir / "metrics.json",
            {
                "schema_version": 1,
                "kind": "repair_cover_restore_metrics",
                "counts": {
                    "disabled_reactions_before": int(len(disabled_set)),
                    "disabled_reactions_after": int(len(new_disabled_entries)),
                    "restored_reactions": int(len(selected_set)),
                    "failing_cases_total": int(len(failing_case_ids)),
                    "failing_cases_with_candidates": int(len(case_candidates)),
                    "uncovered_cases_after_greedy": int(len(uncovered)),
                },
                "selected_reactions": sorted(selected_set),
                "selected_by_case": {key: sorted(value) for key, value in selected_by_case.items() if value},
                "targets_by_case": {key: sorted(value) for key, value in targets_by_case.items()},
            },
        )

    return store.ensure(manifest, writer=_writer)


def repair_mapping_split(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Split a superstate mapping based on merge-quality signals (purity) and guards.

    This is a Stage3 "repair" primitive for mapping-based reducers. It refines an
    existing mapping by splitting problematic superstates while preserving coverage.
    """
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)
    inputs = _extract_inputs(reduction_cfg)

    mapping_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("mapping_id", "mapping", "reduction_id"),
        label="reduction.mapping_id",
    )
    if mapping_id is None:
        mapping_id = _extract_optional_artifact_id(
            inputs,
            keys=("mapping_id", "mapping", "reduction_id"),
            label="reduction.inputs.mapping_id",
        )
    if mapping_id is None:
        raise ConfigError("repair_mapping_split requires mapping_id.")
    mapping_id = _require_nonempty_str(mapping_id, "mapping_id")

    merge_quality_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("merge_quality_id", "features_id", "merge_quality"),
        label="reduction.merge_quality_id",
    )
    if merge_quality_id is None:
        merge_quality_id = _extract_optional_artifact_id(
            inputs,
            keys=("merge_quality_id", "features_id", "merge_quality"),
            label="reduction.inputs.merge_quality_id",
        )
    if merge_quality_id is None:
        raise ConfigError("repair_mapping_split requires merge_quality_id.")
    merge_quality_id = _require_nonempty_str(merge_quality_id, "merge_quality_id")

    policy_cfg = params.get("policy") or {}
    if policy_cfg is None:
        policy_cfg = {}
    if not isinstance(policy_cfg, Mapping):
        raise ConfigError("params.policy must be a mapping when provided.")
    policy = dict(policy_cfg)

    purity_min_raw = policy.get("purity_min", 0.5)
    if isinstance(purity_min_raw, bool):
        raise ConfigError("policy.purity_min must be numeric.")
    try:
        purity_min = float(purity_min_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("policy.purity_min must be numeric.") from exc
    if not math.isfinite(purity_min) or purity_min < 0.0:
        raise ConfigError("policy.purity_min must be finite and >= 0.")

    max_splits_raw = policy.get("max_splits", 20)
    if isinstance(max_splits_raw, bool):
        raise ConfigError("policy.max_splits must be an integer.")
    try:
        max_splits = int(max_splits_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("policy.max_splits must be an integer.") from exc
    if max_splits <= 0:
        raise ConfigError("policy.max_splits must be positive.")

    mapping_payload = _load_mapping_payload(store, mapping_id)
    mapping_entries = mapping_payload.get("mapping") or []
    if not isinstance(mapping_entries, Sequence) or isinstance(
        mapping_entries, (str, bytes, bytearray)
    ):
        raise ConfigError("mapping.json mapping entries must be a list.")

    # Resolve members + representative per superstate id.
    members_by_id: dict[int, list[str]] = {}
    rep_by_id: dict[int, str] = {}
    for entry in mapping_entries:
        if not isinstance(entry, Mapping):
            continue
        species = entry.get("species")
        sid = entry.get("superstate_id")
        if isinstance(species, str) and isinstance(sid, int):
            members_by_id.setdefault(int(sid), []).append(species)
            rep = entry.get("representative")
            if isinstance(rep, str) and rep.strip():
                rep_by_id.setdefault(int(sid), rep.strip())

    clusters = mapping_payload.get("superstates") or mapping_payload.get("clusters") or []
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes, bytearray)):
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            sid = cluster.get("superstate_id")
            rep = cluster.get("representative")
            if isinstance(sid, int) and isinstance(rep, str) and rep.strip():
                rep_by_id.setdefault(int(sid), rep.strip())
            members = cluster.get("members") or []
            if isinstance(sid, int) and isinstance(members, Sequence) and not isinstance(
                members, (str, bytes, bytearray)
            ):
                members_by_id.setdefault(int(sid), []).extend([str(m) for m in members])

    for sid, members in list(members_by_id.items()):
        deduped = []
        seen: set[str] = set()
        for name in members:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        deduped.sort(key=lambda s: s.lower())
        members_by_id[sid] = deduped

    # Species -> heavy element signature (best-effort).
    heavy_sig_by_species: dict[str, str] = {}
    comp_meta = mapping_payload.get("composition_meta") or []
    if isinstance(comp_meta, Sequence) and not isinstance(comp_meta, (str, bytes, bytearray)):
        for entry in comp_meta:
            if not isinstance(entry, Mapping):
                continue
            species = entry.get("species")
            if not isinstance(species, str) or not species.strip():
                continue
            elements = entry.get("elements") or {}
            if not isinstance(elements, Mapping):
                elements = {}
            heavy: list[str] = []
            for key, value in elements.items():
                if str(key) == "H":
                    continue
                try:
                    count = float(value)
                except (TypeError, ValueError):
                    continue
                if count == 0.0:
                    continue
                heavy.append(str(key))
            heavy.sort()
            heavy_sig_by_species[species.strip()] = ",".join(heavy) if heavy else "none"

    # Purity per superstate id from merge-quality features.
    store.read_manifest("features", merge_quality_id)
    quality_rows = _read_table_rows(store.artifact_dir("features", merge_quality_id) / "features.parquet")
    purity_by_id: dict[int, float] = {}
    for row in quality_rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("feature") != "merge.superstate_purity":
            continue
        meta_raw = row.get("meta_json") or "{}"
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        except Exception:
            meta = {}
        sid = meta.get("superstate_id")
        if sid is None or isinstance(sid, bool):
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        value = row.get("value")
        try:
            purity = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(purity):
            purity_by_id[sid_int] = float(purity)

    guards = mapping_payload.get("guards") or {}
    protected = guards.get("protected_species") if isinstance(guards, Mapping) else None
    protected_species = set(_normalize_str_list(protected, "protected_species")) if protected is not None else set()
    extra_protected = policy.get("protected_species")
    if extra_protected is not None:
        protected_species |= set(_normalize_str_list(extra_protected, "policy.protected_species"))

    splits: list[dict[str, Any]] = []

    # Mutable working structure: list of clusters (members).
    cluster_order = sorted(members_by_id.keys())
    working: list[dict[str, Any]] = []
    for sid in cluster_order:
        members = list(members_by_id.get(sid) or [])
        rep = rep_by_id.get(sid) or (members[0] if members else f"S{sid:03d}")
        working.append(
            {
                "sid": int(sid),
                "members": members,
                "representative": rep,
            }
        )

    def _new_cluster(members: list[str], representative: Optional[str] = None) -> dict[str, Any]:
        rep = representative or (members[0] if members else "unknown")
        return {"sid": None, "members": list(members), "representative": rep}

    # 1) Enforce protected species as singleton superstates.
    if protected_species:
        for cluster in list(working):
            members = list(cluster.get("members") or [])
            if len(members) <= 1:
                continue
            to_extract = [s for s in members if s in protected_species]
            if not to_extract:
                continue
            for species in to_extract:
                if species in members and len(members) > 1:
                    members.remove(species)
                    working.append(_new_cluster([species], representative=species))
                    splits.append(
                        {
                            "reason": "protected_species_singleton",
                            "species": species,
                            "from_superstate": cluster.get("sid"),
                        }
                    )
            cluster["members"] = members
            if cluster.get("representative") in to_extract and members:
                cluster["representative"] = members[0]

    # 2) Split low-purity clusters by heavy-element signature when available.
    split_count = 0
    for cluster in list(working):
        if split_count >= max_splits:
            break
        members = list(cluster.get("members") or [])
        if len(members) <= 1:
            continue
        sid = cluster.get("sid")
        purity = purity_by_id.get(int(sid)) if isinstance(sid, int) else None
        if purity is None or purity >= purity_min:
            continue
        groups: dict[str, list[str]] = {}
        for species in members:
            sig = heavy_sig_by_species.get(species, "none")
            groups.setdefault(sig, []).append(species)
        if len(groups) <= 1:
            continue
        rep = cluster.get("representative")
        rep_sig = heavy_sig_by_species.get(rep, "none") if isinstance(rep, str) else "none"
        keep_sig = rep_sig if rep_sig in groups else sorted(groups.keys())[0]
        keep_members = groups.pop(keep_sig)
        keep_members.sort(key=lambda s: s.lower())
        cluster["members"] = keep_members
        cluster["representative"] = rep if rep in keep_members else keep_members[0]
        for sig, sig_members in sorted(groups.items(), key=lambda item: item[0]):
            sig_members.sort(key=lambda s: s.lower())
            working.append(_new_cluster(sig_members))
            split_count += 1
            splits.append(
                {
                    "reason": "purity_split_heavy_signature",
                    "from_superstate": sid,
                    "signature": sig,
                    "members": list(sig_members),
                }
            )
            if split_count >= max_splits:
                break

    # Re-index superstates sequentially.
    new_superstates: list[dict[str, Any]] = []
    new_mapping: list[dict[str, Any]] = []
    species_before = sum(len(cluster.get("members") or []) for cluster in working)
    next_id = 0
    for cluster in working:
        members = list(cluster.get("members") or [])
        if not members:
            continue
        members.sort(key=lambda s: s.lower())
        rep = cluster.get("representative")
        if not isinstance(rep, str) or rep not in members:
            rep = members[0]
        sid = next_id
        next_id += 1
        name = f"S{sid:03d}"
        new_superstates.append(
            {
                "superstate_id": int(sid),
                "name": name,
                "representative": rep,
                "members": members,
                "summary": {"size": len(members)},
            }
        )
        for species in members:
            new_mapping.append(
                {
                    "species": species,
                    "superstate_id": int(sid),
                    "representative": rep,
                }
            )

    new_mapping.sort(key=lambda row: (str(row.get("species", "")).lower(), str(row.get("species", ""))))
    species_after = len(new_superstates)

    repaired_payload = dict(mapping_payload)
    repaired_payload["schema_version"] = 1
    repaired_payload["kind"] = "superstate_mapping"
    repaired_payload["source"] = dict((mapping_payload.get("source") or {})) if isinstance(mapping_payload.get("source"), Mapping) else {}
    repaired_payload["source"]["mapping_id"] = mapping_id
    repaired_payload["superstates"] = new_superstates
    repaired_payload["clusters"] = new_superstates
    repaired_payload["mapping"] = new_mapping
    repaired_payload["meta"] = dict(repaired_payload.get("meta") or {})
    repaired_payload["meta"].update(
        {
            "species_count": int(species_before),
            "superstate_count": int(species_after),
            "coverage": 1.0,
            "repair": {"splits": splits, "merge_quality_id": merge_quality_id},
        }
    )

    metrics_payload = {
        "schema_version": 1,
        "kind": "repair_mapping_split_metrics",
        "counts": {
            "species_before": int(species_before),
            "species_after": int(species_after),
            "merged_species": max(0, int(species_before - species_after)),
            "reactions_before": None,
            "reactions_after": None,
            "disabled_reactions": 0,
            "merged_reactions": 0,
        },
        "policy": {
            "purity_min": float(purity_min),
            "max_splits": int(max_splits),
            "protected_species": sorted(protected_species),
        },
        "repair": {"splits": splits, "merge_quality_id": merge_quality_id},
    }

    inputs_payload = {
        "mode": "repair_mapping_split",
        "mapping_id": mapping_id,
        "merge_quality_id": merge_quality_id,
        "policy": metrics_payload["policy"],
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[mapping_id, merge_quality_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "mapping.json", repaired_payload)
        write_json_atomic(base_dir / "metrics.json", metrics_payload)
        write_json_atomic(
            base_dir / "repair.json",
            {
                "schema_version": 1,
                "kind": "repair_mapping_split",
                "mapping_id": mapping_id,
                "merge_quality_id": merge_quality_id,
                "policy": metrics_payload["policy"],
                "splits": splits,
            },
        )

    return store.ensure(manifest, writer=_writer)


def learnck_style(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Generate LearnCK-style reduction scaffolding artifacts."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)

    mechanism_path = _extract_mechanism(reduction_cfg)
    mechanism_payload = MechanismCompiler.from_path(mechanism_path).payload

    stable_value = (
        params.get("stable_states")
        or reduction_cfg.get("stable_states")
        or reduction_cfg.get("stable")
    )
    stable_payload, stable_path = _load_stable_states_payload(stable_value)
    mechanism_elements = _extract_mechanism_species_elements(mechanism_payload)
    stable_entries = _normalize_stable_entries(stable_payload, mechanism_elements)
    if not stable_entries:
        raise ConfigError("stable_states produced no stable species entries.")

    overall_cfg = None
    for key in ("overall_reaction", "overall_template"):
        candidate = params.get(key) or reduction_cfg.get(key)
        if isinstance(candidate, Mapping):
            overall_cfg = dict(candidate)
            break

    template = _build_overall_reaction_template(stable_entries, overall_cfg)
    tolerance = _coerce_optional_float(
        params.get("overall_tolerance") or reduction_cfg.get("overall_tolerance"),
        "overall_tolerance",
    )
    if tolerance is None:
        tolerance = DEFAULT_OVERALL_REACTION_TOL
    checks = _check_overall_reaction_conservation(
        template,
        stable_entries,
        tolerance=tolerance,
    )
    if not checks["elements"]["passed"] or not checks["sites"]["passed"]:
        raise ConfigError("overall reaction conservation check failed.")

    # LearnCK-style scaffolding does not yet synthesize an overall reaction in the
    # mechanism. Emit a no-op patch that is safe to validate (no reaction_id-based
    # multipliers, which can fail on mechanisms with duplicate reaction equations).
    patch_payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "disabled_reactions": [],
        # Keep this patch "semantically no-op" while still structurally non-empty.
        # Using an index avoids reaction-equation ambiguity (duplicate equations).
        "reaction_multipliers": [{"index": 0, "multiplier": 1.0}],
    }
    normalized_patch, combined_entries = _normalize_patch_payload(patch_payload)
    reduced_payload, _ = _apply_patch_entries(
        dict(mechanism_payload),
        combined_entries,
    )

    inputs = _extract_inputs(reduction_cfg)
    run_id = inputs.get("run_id") or reduction_cfg.get("run_id")
    stable_features_id = (
        inputs.get("stable_features_id")
        or inputs.get("features_id")
        or reduction_cfg.get("stable_features_id")
    )
    parents: list[str] = []
    if isinstance(run_id, str) and run_id.strip():
        run_id = run_id.strip()
        store.read_manifest("runs", run_id)
        parents.append(run_id)
    else:
        run_id = None
    if isinstance(stable_features_id, str) and stable_features_id.strip():
        stable_features_id = stable_features_id.strip()
        store.read_manifest("features", stable_features_id)
        parents.append(stable_features_id)
    else:
        stable_features_id = None

    inputs_payload = {
        "mechanism": mechanism_path,
        "stable_states_path": stable_path,
        "stable_species": [entry["name"] for entry in stable_entries],
        "overall_reaction": template,
        "patch": normalized_patch,
        "run_id": run_id,
        "stable_features_id": stable_features_id,
    }
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, normalized_patch, sort_keys=True)
        write_yaml_payload(base_dir / MECHANISM_FILENAME, reduced_payload, sort_keys=False)
        stable_out = dict(stable_payload)
        if "stable_species" not in stable_out:
            stable_out["stable_species"] = [entry["name"] for entry in stable_entries]
        write_yaml_payload(base_dir / "stable_states.yaml", stable_out, sort_keys=True)
        write_yaml_payload(base_dir / "overall_reaction.yaml", template, sort_keys=True)
        species_before = None
        reactions_before = None
        species_raw = mechanism_payload.get("species")
        if isinstance(species_raw, Sequence) and not isinstance(
            species_raw, (str, bytes, bytearray)
        ):
            species_before = len(species_raw)
        reactions_raw = mechanism_payload.get("reactions")
        if isinstance(reactions_raw, Sequence) and not isinstance(
            reactions_raw, (str, bytes, bytearray)
        ):
            reactions_before = len(reactions_raw)

        species_after = None
        reactions_after = _count_reactions(reduced_payload)
        reduced_species_raw = reduced_payload.get("species")
        if isinstance(reduced_species_raw, Sequence) and not isinstance(
            reduced_species_raw, (str, bytes, bytearray)
        ):
            species_after = len(reduced_species_raw)

        metrics = {
            "schema_version": 1,
            "kind": "learnck_metrics",
            "counts": {
                "species_before": species_before,
                "species_after": species_after,
                "merged_species": 0,
                "reactions_before": reactions_before,
                "reactions_after": reactions_after,
                "disabled_reactions": 0,
                "merged_reactions": 0,
            },
            "stable_states_path": stable_path,
            "stable_species_count": len(stable_entries),
            "overall_reaction_checks": checks,
            "patch_noop": True,
        }
        write_json_atomic(base_dir / "metrics.json", metrics)

    return store.ensure(manifest, writer=_writer)


def amore_search(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run AMORE-style beam search with cheap score filtering."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    mechanism_path = _extract_mechanism(reduction_cfg)
    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        raise ConfigError("reduction.graph is required for amore_search.")

    search_cfg = _amore_extract_search_cfg(reduction_cfg)
    beam_width = _coerce_positive_int(
        search_cfg.get("beam_width") or search_cfg.get("beam"),
        "reduction.search.beam_width",
        default=DEFAULT_AMORE_BEAM_WIDTH,
    )
    max_depth = _coerce_positive_int(
        search_cfg.get("max_depth") or search_cfg.get("depth"),
        "reduction.search.max_depth",
        default=DEFAULT_AMORE_MAX_DEPTH,
    )
    expand_top = _coerce_positive_int(
        search_cfg.get("expand_top") or search_cfg.get("expand_per_parent"),
        "reduction.search.expand_top",
        default=DEFAULT_AMORE_EXPAND_TOP,
    )
    cheap_min = _coerce_optional_float(
        search_cfg.get("cheap_score_min")
        or search_cfg.get("cheap_score")
        or search_cfg.get("cheap_min"),
        "reduction.search.cheap_score_min",
    )
    if cheap_min is None:
        cheap_min = DEFAULT_AMORE_CHEAP_SCORE_MIN
    if cheap_min < 0.0 or cheap_min > 1.0:
        raise ConfigError("reduction.search.cheap_score_min must be in [0, 1].")

    max_removed = search_cfg.get("max_removed") or search_cfg.get("max_remove")
    if max_removed is not None:
        if isinstance(max_removed, bool):
            raise ConfigError("reduction.search.max_removed must be an integer.")
        try:
            max_removed = int(max_removed)
        except (TypeError, ValueError) as exc:
            raise ConfigError("reduction.search.max_removed must be an integer.") from exc
        if max_removed < 0:
            raise ConfigError("reduction.search.max_removed must be >= 0.")
    min_keep = search_cfg.get("min_keep") or search_cfg.get("min_remaining")
    if min_keep is None:
        min_keep = 1
    if isinstance(min_keep, bool):
        raise ConfigError("reduction.search.min_keep must be an integer.")
    try:
        min_keep = int(min_keep)
    except (TypeError, ValueError) as exc:
        raise ConfigError("reduction.search.min_keep must be an integer.") from exc
    if min_keep <= 0:
        raise ConfigError("reduction.search.min_keep must be > 0.")

    logger = logging.getLogger("rxn_platform.reduction")
    compiler = MechanismCompiler.from_path(mechanism_path)
    reaction_count = compiler.reaction_count()
    if reaction_count <= min_keep:
        raise ConfigError("mechanism has no removable reactions for amore_search.")

    if max_removed is None:
        max_removed = max(0, reaction_count - min_keep)
    else:
        max_removed = min(max_removed, reaction_count - min_keep)
    if max_depth > max_removed:
        max_depth = max_removed
    if max_depth < 0:
        max_depth = 0

    graph_payload = _amore_load_graph_payload(store, graph_id)
    activity_values = _amore_reaction_activity(
        graph_payload,
        compiler,
        logger=logger,
    )
    ranked_indices = sorted(
        range(reaction_count),
        key=lambda idx: activity_values[idx],
    )
    base_hash = compiler.mechanism_hash()

    surrogate_cfg = _amore_extract_surrogate_cfg(reduction_cfg, resolved_cfg)
    use_surrogate = bool(surrogate_cfg.get("enabled"))

    validation_cfg = _extract_validation_cfg(reduction_cfg)
    if "pipeline" not in validation_cfg:
        params = _extract_params(reduction_cfg)
        candidate = params.get("validation") if isinstance(params, Mapping) else None
        if isinstance(candidate, Mapping):
            validation_cfg = dict(candidate)
    pipeline_value = _extract_pipeline_value(validation_cfg)
    metric = _extract_metric_name(validation_cfg)
    tolerance = _extract_tolerance(validation_cfg)
    rel_eps = _extract_rel_eps(validation_cfg)
    missing_strategy = _normalize_missing_strategy(
        validation_cfg.get("missing_strategy")
    )

    runner = PipelineRunner(store=store, registry=registry, logger=logger)
    pipeline_cfg = _normalize_pipeline_cfg(pipeline_value, runner)
    steps = pipeline_cfg.get("steps", [])

    sim_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("sim_step_id"),
        task_name="sim.run",
        label="validation.sim_step_id",
        required=True,
    )
    obs_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("observables_step_id"),
        task_name="observables.run",
        label="validation.observables_step_id",
        required=False,
    )
    feat_step_id = _extract_step_id(
        steps,
        override=validation_cfg.get("features_step_id"),
        task_name="features.run",
        label="validation.features_step_id",
        required=False,
    )
    if obs_step_id is None and feat_step_id is None:
        raise ConfigError("validation requires an observables or features step.")

    baseline_sim_cfg = _extract_sim_override(validation_cfg)
    if baseline_sim_cfg is None:
        baseline_sim_cfg = None
        for step in steps:
            if step.get("id") == sim_step_id:
                sim_cfg = step.get("sim")
                if not isinstance(sim_cfg, Mapping):
                    raise ConfigError("pipeline sim step must include a sim mapping.")
                baseline_sim_cfg = dict(sim_cfg)
                break
    if baseline_sim_cfg is None:
        raise ConfigError("baseline sim config could not be resolved.")

    baseline_sim_cfg = dict(baseline_sim_cfg)
    baseline_sim_cfg["mechanism"] = mechanism_path

    baseline_pipeline_cfg = copy.deepcopy(pipeline_cfg)
    for step in baseline_pipeline_cfg.get("steps", []):
        if step.get("id") == sim_step_id:
            step["sim"] = dict(baseline_sim_cfg)
            break

    baseline_results = runner.run(baseline_pipeline_cfg)
    baseline_run_id = baseline_results.get(sim_step_id)
    if baseline_run_id is None:
        raise ConfigError("baseline sim step did not produce a run_id.")

    baseline_obs_id = baseline_results.get(obs_step_id) if obs_step_id else None
    baseline_feat_id = baseline_results.get(feat_step_id) if feat_step_id else None

    baseline_obs_groups: dict[tuple[str, str, str], list[Any]] = {}
    baseline_feat_groups: dict[tuple[str, str, str], list[Any]] = {}
    if obs_step_id:
        if baseline_obs_id is None:
            raise ConfigError("baseline observables step did not produce an artifact.")
        obs_rows = _extract_run_rows(
            store,
            kind="observables",
            artifact_id=baseline_obs_id,
            run_id=baseline_run_id,
            filename="values.parquet",
        )
        baseline_obs_groups = _group_metric_rows(obs_rows, name_field="observable")
        if not baseline_obs_groups:
            raise ConfigError("baseline observables produced no rows for comparison.")
    if feat_step_id:
        if baseline_feat_id is None:
            raise ConfigError("baseline features step did not produce an artifact.")
        feat_rows = _extract_run_rows(
            store,
            kind="features",
            artifact_id=baseline_feat_id,
            run_id=baseline_run_id,
            filename="features.parquet",
        )
        baseline_feat_groups = _group_metric_rows(feat_rows, name_field="feature")
        if not baseline_feat_groups:
            raise ConfigError("baseline features produced no rows for comparison.")

    surrogate_state: Optional[dict[str, Any]] = None
    if use_surrogate:
        reaction_types = [
            _amore_reaction_type_label(reaction) for reaction in compiler.reactions
        ]
        type_buckets = _amore_select_type_buckets(
            reaction_types, max_types=int(surrogate_cfg["max_types"])
        )
        feature_names = _amore_feature_names(type_buckets)
        dataset_dir = _amore_resolve_surrogate_root(
            store.root, str(surrogate_cfg["dataset_name"])
        )
        dataset_meta, dataset_rows = _amore_load_surrogate_dataset(dataset_dir)
        dataset_hash = dataset_meta.get("mechanism_hash")
        if dataset_hash and dataset_hash != base_hash:
            logger.warning(
                "Surrogate dataset mechanism_hash mismatch; starting new dataset."
            )
            dataset_meta = {}
            dataset_rows = []
        stored_features = dataset_meta.get("feature_names")
        if stored_features and stored_features != feature_names:
            logger.warning(
                "Surrogate dataset feature_names mismatch; starting new dataset."
            )
            dataset_meta = {}
            dataset_rows = []

        surrogate_state = {
            "enabled": True,
            "dataset_dir": dataset_dir,
            "dataset_meta": dict(dataset_meta),
            "rows": list(dataset_rows),
            "new_rows": [],
            "feature_names": feature_names,
            "type_buckets": type_buckets,
            "reaction_types": reaction_types,
            "seen_hashes": {
                row.get("mechanism_hash")
                for row in dataset_rows
                if isinstance(row.get("mechanism_hash"), str)
            },
            "cfg": dict(surrogate_cfg),
            "stats": {
                "baseline_candidates": 0,
                "evaluated": 0,
                "surrogate_filtered": 0,
                "surrogate_uncertain": 0,
                "surrogate_predicted": 0,
                "cheap_filtered": 0,
                "cache_hits": 0,
            },
        }
        surrogate_state["cfg"]["error_skip"] = surrogate_cfg.get("error_skip")
        if surrogate_state["cfg"]["error_skip"] is None:
            surrogate_state["cfg"]["error_skip"] = (
                tolerance * float(surrogate_cfg["error_skip_factor"])
            )
        surrogate_state["model"] = _amore_fit_surrogate_model(
            surrogate_state["rows"],
            feature_names=feature_names,
            min_samples=int(surrogate_cfg["min_samples"]),
            k_neighbors=int(surrogate_cfg["k_neighbors"]),
        )

    cache: dict[str, dict[str, Any]] = {}
    edit_entries: list[dict[str, Any]] = []
    filtered_keys: set[tuple[int, ...]] = set()

    def _refresh_surrogate_model() -> None:
        if surrogate_state is None:
            return
        surrogate_state["model"] = _amore_fit_surrogate_model(
            surrogate_state["rows"],
            feature_names=surrogate_state["feature_names"],
            min_samples=int(surrogate_state["cfg"]["min_samples"]),
            k_neighbors=int(surrogate_state["cfg"]["k_neighbors"]),
        )

    def _append_surrogate_row(row: Mapping[str, Any]) -> None:
        if surrogate_state is None:
            return
        mechanism_hash = row.get("mechanism_hash")
        if not isinstance(mechanism_hash, str) or not mechanism_hash.strip():
            return
        if mechanism_hash in surrogate_state["seen_hashes"]:
            return
        surrogate_state["seen_hashes"].add(mechanism_hash)
        surrogate_state["rows"].append(dict(row))
        surrogate_state["new_rows"].append(dict(row))
        update_every = int(surrogate_state["cfg"]["update_every"])
        if update_every > 0 and len(surrogate_state["new_rows"]) % update_every == 0:
            _refresh_surrogate_model()

    def _evaluate_candidate(
        disabled_indices: Sequence[int],
        *,
        depth: int,
        cheap_score: float,
    ) -> None:
        disabled_list = sorted(set(disabled_indices))
        candidate_key = tuple(disabled_list)
        entry_base: dict[str, Any] = {
            "disabled_indices": disabled_list,
            "disabled_count": len(disabled_list),
            "remaining_reactions": reaction_count - len(disabled_list),
            "cheap_score": cheap_score,
            "depth": depth,
        }
        surrogate_features: Optional[dict[str, float]] = None
        surrogate_vector: Optional[list[float]] = None
        surrogate_pred: Optional[dict[str, Optional[float]]] = None
        surrogate_decision: Optional[str] = None

        if cheap_score < cheap_min:
            if surrogate_state is not None:
                surrogate_state["stats"]["cheap_filtered"] += 1
            entry = dict(entry_base)
            entry.update(
                {
                    "status": "filtered",
                    "mechanism_hash": _amore_candidate_hash(base_hash, disabled_list),
                    "passed": False,
                }
            )
            edit_entries.append(entry)
            filtered_keys.add(candidate_key)
            return

        patch_entries = [
            {"index": idx} for idx in disabled_list
        ]
        mechanism_hash = _amore_candidate_hash(base_hash, disabled_list)

        if mechanism_hash in cache:
            if surrogate_state is not None:
                surrogate_state["stats"]["cache_hits"] += 1
            cached = cache[mechanism_hash]
            entry = dict(entry_base)
            entry.update(
                {
                    "status": "cached",
                    "mechanism_hash": mechanism_hash,
                    "passed": cached.get("passed"),
                    "qoi_error": cached.get("qoi_error"),
                    "run_id": cached.get("run_id"),
                    "observables_id": cached.get("observables_id"),
                    "features_id": cached.get("features_id"),
                }
            )
            edit_entries.append(entry)
            return

        if surrogate_state is not None:
            surrogate_state["stats"]["baseline_candidates"] += 1
            surrogate_vector, surrogate_features = _amore_candidate_features(
                disabled_list,
                reaction_count=reaction_count,
                cheap_score=cheap_score,
                reaction_types=surrogate_state["reaction_types"],
                type_buckets=surrogate_state["type_buckets"],
                activity_values=activity_values,
            )
            model = surrogate_state.get("model") or {}
            if model.get("ready"):
                surrogate_pred = _amore_predict_surrogate(model, surrogate_vector)
                surrogate_state["stats"]["surrogate_predicted"] += 1
                entry_base.update(
                    {
                        "surrogate_pred_error": surrogate_pred.get("pred_error"),
                        "surrogate_fail_prob": surrogate_pred.get("pred_fail_prob"),
                        "surrogate_uncertainty": surrogate_pred.get("uncertainty"),
                    }
                )
                warmup = max(1, int(surrogate_state["cfg"]["warmup"]))
                if surrogate_state["stats"]["evaluated"] >= warmup:
                    uncertainty_gate = float(surrogate_state["cfg"]["uncertainty_gate"])
                    if surrogate_pred.get("uncertainty", 1.0) >= uncertainty_gate:
                        surrogate_decision = "uncertain"
                        surrogate_state["stats"]["surrogate_uncertain"] += 1
                    else:
                        fail_skip = float(surrogate_state["cfg"]["fail_prob_skip"])
                        error_skip = float(surrogate_state["cfg"]["error_skip"])
                        pred_fail = surrogate_pred.get("pred_fail_prob")
                        pred_error = surrogate_pred.get("pred_error")
                        if (
                            pred_fail is not None
                            and pred_fail >= fail_skip
                            or pred_error is not None
                            and pred_error >= error_skip
                        ):
                            surrogate_decision = "filtered"
                        else:
                            surrogate_decision = "evaluate"
                else:
                    surrogate_decision = "warmup"

        if surrogate_decision is not None:
            entry_base["surrogate_decision"] = surrogate_decision

        if surrogate_decision == "filtered":
            entry = dict(entry_base)
            entry.update(
                {
                    "status": "surrogate_filtered",
                    "mechanism_hash": mechanism_hash,
                    "passed": False,
                }
            )
            edit_entries.append(entry)
            if surrogate_state is not None:
                surrogate_state["stats"]["surrogate_filtered"] += 1
            return

        temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        try:
            if surrogate_state is not None:
                surrogate_state["stats"]["evaluated"] += 1
            reduced_sim_cfg = dict(baseline_sim_cfg)
            reduced_sim_cfg["mechanism"] = mechanism_path
            reduced_sim_cfg.pop("reaction_multipliers", None)
            reduced_sim_cfg["disabled_reactions"] = list(patch_entries)

            reduced_pipeline_cfg = copy.deepcopy(pipeline_cfg)
            for step in reduced_pipeline_cfg.get("steps", []):
                if step.get("id") == sim_step_id:
                    step["sim"] = dict(reduced_sim_cfg)
                    break

            reduced_results = runner.run(reduced_pipeline_cfg)
            reduced_run_id = reduced_results.get(sim_step_id)
            if reduced_run_id is None:
                raise ConfigError("reduced sim step did not produce a run_id.")

            reduced_obs_id = (
                reduced_results.get(obs_step_id) if obs_step_id else None
            )
            reduced_feat_id = (
                reduced_results.get(feat_step_id) if feat_step_id else None
            )

            patch_rows: list[dict[str, Any]] = []
            patch_pass = True
            evaluated_total = 0

            if obs_step_id:
                if reduced_obs_id is None:
                    raise ConfigError(
                        "reduced observables step did not produce an artifact."
                    )
                reduced_obs_rows = _extract_run_rows(
                    store,
                    kind="observables",
                    artifact_id=reduced_obs_id,
                    run_id=reduced_run_id,
                    filename="values.parquet",
                )
                reduced_obs_groups = _group_metric_rows(
                    reduced_obs_rows, name_field="observable"
                )
                rows, passed, evaluated = _compare_grouped_values(
                    kind="observable",
                    baseline_groups=baseline_obs_groups,
                    reduced_groups=reduced_obs_groups,
                    metric=metric,
                    tolerance=tolerance,
                    rel_eps=rel_eps,
                    missing_strategy=missing_strategy,
                    patch_index=depth,
                    patch_id=mechanism_hash,
                    baseline_run_id=baseline_run_id,
                    reduced_run_id=reduced_run_id,
                    baseline_artifact_id=baseline_obs_id,
                    reduced_artifact_id=reduced_obs_id,
                )
                patch_rows.extend(rows)
                patch_pass = patch_pass and passed
                evaluated_total += evaluated

            if feat_step_id:
                if reduced_feat_id is None:
                    raise ConfigError(
                        "reduced features step did not produce an artifact."
                    )
                reduced_feat_rows = _extract_run_rows(
                    store,
                    kind="features",
                    artifact_id=reduced_feat_id,
                    run_id=reduced_run_id,
                    filename="features.parquet",
                )
                reduced_feat_groups = _group_metric_rows(
                    reduced_feat_rows, name_field="feature"
                )
                rows, passed, evaluated = _compare_grouped_values(
                    kind="feature",
                    baseline_groups=baseline_feat_groups,
                    reduced_groups=reduced_feat_groups,
                    metric=metric,
                    tolerance=tolerance,
                    rel_eps=rel_eps,
                    missing_strategy=missing_strategy,
                    patch_index=depth,
                    patch_id=mechanism_hash,
                    baseline_run_id=baseline_run_id,
                    reduced_run_id=reduced_run_id,
                    baseline_artifact_id=baseline_feat_id,
                    reduced_artifact_id=reduced_feat_id,
                )
                patch_rows.extend(rows)
                patch_pass = patch_pass and passed
                evaluated_total += evaluated

            if evaluated_total == 0:
                patch_pass = False

            metric_values: list[float] = []
            for row in patch_rows:
                if row.get("status") != "ok":
                    continue
                value = row.get("abs_diff") if metric == "abs" else row.get("rel_diff")
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
                metric_values.append(value)
            qoi_error = max(metric_values) if metric_values else math.inf

            result = dict(entry_base)
            result.update(
                {
                    "status": "evaluated",
                    "mechanism_hash": mechanism_hash,
                    "passed": patch_pass,
                    "qoi_error": qoi_error,
                    "run_id": reduced_run_id,
                    "observables_id": reduced_obs_id,
                    "features_id": reduced_feat_id,
                }
            )
            cache[mechanism_hash] = dict(result)
            if surrogate_state is not None and surrogate_features is not None:
                _append_surrogate_row(
                    {
                        "mechanism_hash": mechanism_hash,
                        "disabled_indices": disabled_list,
                        "disabled_count": len(disabled_list),
                        "remaining_reactions": reaction_count - len(disabled_list),
                        "cheap_score": cheap_score,
                        "depth": depth,
                        "features": surrogate_features,
                        "qoi_error": qoi_error,
                        "passed": patch_pass,
                        "failed": not patch_pass,
                        "status": "evaluated",
                    }
                )
            edit_entries.append(result)
        except Exception as exc:
            entry = dict(entry_base)
            entry.update(
                {
                    "status": "failed",
                    "mechanism_hash": _amore_candidate_hash(base_hash, disabled_list),
                    "passed": False,
                    "error": str(exc),
                }
            )
            if surrogate_state is not None and surrogate_features is not None:
                _append_surrogate_row(
                    {
                        "mechanism_hash": mechanism_hash,
                        "disabled_indices": disabled_list,
                        "disabled_count": len(disabled_list),
                        "remaining_reactions": reaction_count - len(disabled_list),
                        "cheap_score": cheap_score,
                        "depth": depth,
                        "features": surrogate_features,
                        "qoi_error": None,
                        "passed": False,
                        "failed": True,
                        "status": "failed",
                    }
                )
            edit_entries.append(entry)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    beam: list[dict[str, Any]] = [
        {"disabled": set(), "cheap_score": 1.0}
    ]
    seen: set[tuple[int, ...]] = {()}

    for depth in range(max_depth + 1):
        next_beam: list[dict[str, Any]] = []
        for candidate in beam:
            disabled_set = candidate["disabled"]
            disabled_key = tuple(sorted(disabled_set))
            if disabled_key in filtered_keys:
                continue
            _evaluate_candidate(
                disabled_key,
                depth=depth,
                cheap_score=candidate["cheap_score"],
            )

        if depth >= max_depth:
            break

        for candidate in beam:
            if candidate["cheap_score"] < cheap_min:
                continue
            disabled_set = candidate["disabled"]
            expanded = 0
            for idx in ranked_indices:
                if idx in disabled_set:
                    continue
                new_set = set(disabled_set)
                new_set.add(idx)
                if len(new_set) > max_removed:
                    continue
                key = tuple(sorted(new_set))
                if key in seen:
                    continue
                seen.add(key)
                cheap_score = _amore_cheap_score(activity_values, key)
                next_beam.append({"disabled": new_set, "cheap_score": cheap_score})
                expanded += 1
                if expanded >= expand_top:
                    break

        if not next_beam:
            break
        next_beam.sort(key=lambda item: item["cheap_score"], reverse=True)
        beam = next_beam[:beam_width]

    evaluated_rows = [row for row in cache.values() if row.get("qoi_error") is not None]
    passing_rows = [row for row in evaluated_rows if row.get("passed")]
    selected_row: Optional[dict[str, Any]] = None
    if passing_rows:
        selected_row = min(
            passing_rows,
            key=lambda row: (row.get("remaining_reactions", reaction_count), row.get("qoi_error", math.inf)),
        )
    elif evaluated_rows:
        selected_row = min(
            evaluated_rows,
            key=lambda row: (row.get("remaining_reactions", reaction_count), row.get("qoi_error", math.inf)),
        )

    if selected_row is None:
        raise ConfigError("amore_search produced no evaluated candidates.")

    selected_disabled = selected_row.get("disabled_indices", [])
    selected_patch = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "disabled_reactions": [{"index": idx} for idx in selected_disabled],
        "reaction_multipliers": [],
    }

    pareto_rows: list[dict[str, Any]] = []
    for row in evaluated_rows:
        remaining = row.get("remaining_reactions", reaction_count)
        qoi_error = row.get("qoi_error", math.inf)
        pareto_rows.append(
            {
                "mechanism_hash": row.get("mechanism_hash"),
                "disabled_count": row.get("disabled_count"),
                "remaining_reactions": remaining,
                "qoi_error": qoi_error,
                "passed": row.get("passed"),
                "cheap_score": row.get("cheap_score"),
                "status": row.get("status"),
            }
        )

    pareto_front: list[dict[str, Any]] = []
    for row in pareto_rows:
        dominated = False
        for other in pareto_rows:
            if other is row:
                continue
            if (
                other.get("remaining_reactions", math.inf)
                <= row.get("remaining_reactions", math.inf)
                and other.get("qoi_error", math.inf) <= row.get("qoi_error", math.inf)
            ):
                if (
                    other.get("remaining_reactions", math.inf)
                    < row.get("remaining_reactions", math.inf)
                    or other.get("qoi_error", math.inf) < row.get("qoi_error", math.inf)
                ):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(row)

    surrogate_summary: Optional[dict[str, Any]] = None
    if surrogate_state is not None:
        _refresh_surrogate_model()
        stats = surrogate_state["stats"]
        baseline = int(stats.get("baseline_candidates", 0))
        evaluated = int(stats.get("evaluated", 0))
        reduction = max(0, baseline - evaluated)
        reduction_ratio = float(reduction) / float(baseline) if baseline > 0 else 0.0
        dataset_meta = dict(surrogate_state.get("dataset_meta") or {})
        dataset_meta.setdefault("kind", "amore_surrogate")
        dataset_meta.setdefault("created_at", _utc_now_iso())
        dataset_meta.update(
            {
                "mechanism_hash": base_hash,
                "graph_id": graph_id,
                "feature_names": surrogate_state["feature_names"],
                "type_buckets": surrogate_state["type_buckets"],
                "config": dict(surrogate_state["cfg"]),
            }
        )
        _amore_write_surrogate_dataset(
            surrogate_state["dataset_dir"],
            dataset_meta,
            surrogate_state["rows"],
        )
        surrogate_summary = {
            "enabled": True,
            "dataset_dir": str(surrogate_state["dataset_dir"]),
            "feature_names": list(surrogate_state["feature_names"]),
            "type_buckets": list(surrogate_state["type_buckets"]),
            "training_rows": len(surrogate_state["rows"]),
            "new_rows": len(surrogate_state["new_rows"]),
            "model_ready": bool(
                (surrogate_state.get("model") or {}).get("ready", False)
            ),
            "evaluation": {
                "baseline": baseline,
                "evaluated": evaluated,
                "surrogate_filtered": int(stats.get("surrogate_filtered", 0)),
                "uncertain_evaluated": int(stats.get("surrogate_uncertain", 0)),
                "reduction": reduction,
                "reduction_ratio": reduction_ratio,
            },
            "gates": {
                "uncertainty_gate": float(surrogate_state["cfg"]["uncertainty_gate"]),
                "fail_prob_skip": float(surrogate_state["cfg"]["fail_prob_skip"]),
                "error_skip": float(surrogate_state["cfg"]["error_skip"]),
            },
        }

    inputs_payload: dict[str, Any] = {
        "mechanism": mechanism_path,
        "graph_id": graph_id,
        "search": {
            "beam_width": beam_width,
            "max_depth": max_depth,
            "expand_top": expand_top,
            "cheap_score_min": cheap_min,
            "max_removed": max_removed,
            "min_keep": min_keep,
        },
        "validation": {
            "metric": metric,
            "tolerance": tolerance,
            "rel_eps": rel_eps,
            "missing_strategy": missing_strategy,
        },
        "selected": {
            "mechanism_hash": selected_row.get("mechanism_hash"),
            "disabled_count": selected_row.get("disabled_count"),
            "remaining_reactions": selected_row.get("remaining_reactions"),
            "qoi_error": selected_row.get("qoi_error"),
            "passed": selected_row.get("passed"),
        },
    }
    if surrogate_summary is not None:
        inputs_payload["surrogate"] = {
            "enabled": True,
            "dataset_dir": surrogate_summary.get("dataset_dir"),
            "feature_names": surrogate_summary.get("feature_names"),
            "type_buckets": surrogate_summary.get("type_buckets"),
            "gates": surrogate_summary.get("gates"),
        }

    artifact_id = reduction_cfg.get("artifact_id") or reduction_cfg.get("id")
    if artifact_id is None:
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
    artifact_id = _require_nonempty_str(artifact_id, "artifact_id")

    parents = [graph_id, baseline_run_id]
    if baseline_obs_id:
        parents.append(baseline_obs_id)
    if baseline_feat_id:
        parents.append(baseline_feat_id)

    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=_dedupe_preserve(parents),
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    species_before = None
    species_raw = compiler.payload.get("species")
    if isinstance(species_raw, Sequence) and not isinstance(
        species_raw, (str, bytes, bytearray)
    ):
        species_before = len(species_raw)

    selected_disabled = inputs_payload.get("selected", {}).get("disabled_count")
    try:
        disabled_count = int(selected_disabled) if selected_disabled is not None else None
    except (TypeError, ValueError):
        disabled_count = None
    selected_remaining = inputs_payload.get("selected", {}).get("remaining_reactions")
    try:
        reactions_after = int(selected_remaining) if selected_remaining is not None else None
    except (TypeError, ValueError):
        reactions_after = None

    metrics_payload = {
        "schema_version": 1,
        "kind": "amore_metrics",
        "counts": {
            "species_before": species_before,
            "species_after": species_before,
            "merged_species": 0,
            "reactions_before": reaction_count,
            "reactions_after": reactions_after,
            "disabled_reactions": disabled_count,
            "merged_reactions": 0,
        },
        "selected": inputs_payload["selected"],
        "baseline": {
            "run_id": baseline_run_id,
            "observables_id": baseline_obs_id,
            "features_id": baseline_feat_id,
        },
        "pareto_size": len(pareto_front),
        "evaluated": len(evaluated_rows),
    }
    if surrogate_summary is not None:
        metrics_payload["surrogate"] = surrogate_summary

    def _writer(base_dir: Path) -> None:
        write_yaml_payload(base_dir / PATCH_FILENAME, selected_patch, sort_keys=True)
        write_json_atomic(
            base_dir / "edit_log.json",
            {
                "schema_version": 1,
                "baseline": {
                    "run_id": baseline_run_id,
                    "observables_id": baseline_obs_id,
                    "features_id": baseline_feat_id,
                },
                "search": inputs_payload["search"],
                "candidates": edit_entries,
                "selected": inputs_payload["selected"],
                "surrogate": surrogate_summary,
            },
        )
        if surrogate_summary is not None:
            write_json_atomic(base_dir / "surrogate_report.json", surrogate_summary)
        _amore_write_csv(pareto_front, base_dir / "pareto.csv")
        write_json_atomic(base_dir / "metrics.json", metrics_payload)

    return store.ensure(manifest, writer=_writer)


def cnr_coarse(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Generate CNR-Coarse mapping from temporal graphs with hard constraints."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for cnr_coarse mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            params,
            keys=("graph", "graph_id", "graph_artifact"),
            label="reduction.graph",
        )
    if graph_id is None:
        raise ConfigError("cnr_coarse requires graph_id.")

    graph_payload = _load_graph_payload(store, graph_id)
    species_section = graph_payload.get("species") or {}
    species_order = species_section.get("order")
    if not isinstance(species_order, Sequence) or isinstance(
        species_order, (str, bytes, bytearray)
    ):
        raise ConfigError("temporal graph species.order must be a list.")
    species_names = [str(name) for name in species_order]
    if not species_names:
        raise ConfigError("temporal graph has no species.")

    species_graph = graph_payload.get("species_graph")
    if not isinstance(species_graph, Mapping):
        raise ConfigError("temporal graph is missing species_graph metadata.")
    layers_meta = species_graph.get("layers")
    if not isinstance(layers_meta, Sequence) or isinstance(
        layers_meta, (str, bytes, bytearray)
    ):
        raise ConfigError("temporal graph layers must be a list.")
    if not layers_meta:
        raise ConfigError("temporal graph has no layers.")

    graph_dir = store.artifact_dir("graphs", graph_id)
    aggregated = None
    window_summaries: list[dict[str, Any]] = []
    total_flux = 0.0
    for entry in layers_meta:
        if not isinstance(entry, Mapping):
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        layer_path = graph_dir / path_value
        layer_matrix = _load_csr_matrix(layer_path)
        dense_layer = None
        if sp is not None and hasattr(layer_matrix, "sum"):
            layer_flux = float(layer_matrix.sum())
        else:
            dense_layer = _dense_from_sparse(layer_matrix)
            layer_flux = float(dense_layer.sum())
        total_flux += layer_flux
        if aggregated is None:
            aggregated = layer_matrix if dense_layer is None else dense_layer
        else:
            # scipy.sparse may be unavailable; fall back to dense numpy addition.
            if sp is not None and hasattr(aggregated, "shape") and hasattr(layer_matrix, "shape"):
                aggregated = aggregated + layer_matrix
            else:
                if dense_layer is None:
                    dense_layer = _dense_from_sparse(layer_matrix)
                aggregated = _dense_from_sparse(aggregated) + dense_layer
        window_summaries.append(
            {
                "index": entry.get("index"),
                "path": path_value,
                "flux_sum": layer_flux,
                "window": entry.get("window"),
            }
        )

    if aggregated is None:
        raise ConfigError("temporal graph layers could not be loaded.")

    dense_agg = _dense_from_sparse(aggregated)
    if dense_agg.shape[0] != len(species_names):
        raise ConfigError("temporal graph matrix shape does not match species list.")

    community_cfg = params.get("community") or reduction_cfg.get("community") or {}
    if community_cfg is None:
        community_cfg = {}
    if not isinstance(community_cfg, Mapping):
        raise ConfigError("community config must be a mapping.")
    community_cfg = dict(community_cfg)
    method = str(
        community_cfg.get("method") or community_cfg.get("algo") or DEFAULT_CNR_COMMUNITY_METHOD
    )
    resolution = _coerce_optional_float(community_cfg.get("resolution"), "community.resolution")
    if resolution is None:
        resolution = DEFAULT_CNR_RESOLUTION
    min_weight = _coerce_optional_float(community_cfg.get("min_weight"), "community.min_weight")
    if min_weight is None:
        min_weight = DEFAULT_CNR_MIN_WEIGHT
    symmetrize = _coerce_bool(
        community_cfg.get("symmetrize"),
        "community.symmetrize",
        default=DEFAULT_CNR_SYMMETRIZE,
    )

    constraints_cfg = params.get("constraints") or reduction_cfg.get("constraints")
    constraint_fields = _normalize_constraint_fields(constraints_cfg)
    source_meta = graph_payload.get("source")
    mechanism = None
    phase_default = None
    if isinstance(source_meta, Mapping):
        mechanism = _coerce_optional_str(source_meta.get("mechanism"), "mechanism")
        phase_default = _coerce_optional_str(source_meta.get("phase"), "phase")

    meta_from_graph = _species_metadata_from_graph(graph_payload, species_names)
    meta_from_mech = _species_metadata_from_mechanism(mechanism, phase_default)
    merged_meta: dict[str, dict[str, Any]] = {}
    for name in species_names:
        merged = {}
        if name in meta_from_mech:
            merged.update(meta_from_mech[name])
        if name in meta_from_graph:
            merged.update(meta_from_graph[name])
        merged_meta[name] = merged

    groups, constraint_entries = _build_constraint_groups(
        species_names,
        constraints=constraint_fields,
        metadata=merged_meta,
        phase_default=phase_default,
    )

    if symmetrize:
        dense_agg = 0.5 * (dense_agg + dense_agg.T)

    seed = None
    common_cfg = manifest_cfg.get("common")
    if isinstance(common_cfg, Mapping):
        raw_seed = common_cfg.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed
    if seed is None:
        raw_seed = reduction_cfg.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed
    if seed is None:
        seed = 0

    degree_weights = dense_agg.sum(axis=1)
    if hasattr(degree_weights, "tolist"):
        degree_weights = np.asarray(degree_weights, dtype=float).ravel()
    else:
        degree_weights = np.asarray(degree_weights, dtype=float).ravel()

    group_items = sorted(groups.items(), key=lambda item: item[0])
    clusters: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    superstate_id = 0

    logger = logging.getLogger("rxn_platform.reduction")
    requested_method = method
    if nx is None:
        # No optional dependency installs allowed in some environments; fall back
        # to a deterministic, dependency-free clustering.
        if method.lower() not in {"components", "connected_components", "connected"}:
            logger.warning(
                "cnr_coarse community.method=%r requested but networkx is unavailable; "
                "falling back to 'components' clustering.",
                requested_method,
            )
        method = "components"

    def _components_from_dense(submatrix: Any) -> list[list[int]]:
        # Build an unweighted graph by thresholding edge weights, then return
        # connected components (member indices are local to this submatrix).
        n = int(submatrix.shape[0])
        if n <= 0:
            return []
        if n == 1:
            return [[0]]
        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    w = float(submatrix[i, j])
                except Exception:
                    continue
                if w <= min_weight:
                    continue
                adj[i].append(j)
                adj[j].append(i)
        seen = [False] * n
        comps: list[list[int]] = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            comp: list[int] = []
            while stack:
                node = stack.pop()
                if seen[node]:
                    continue
                seen[node] = True
                comp.append(node)
                for nb in adj[node]:
                    if not seen[nb]:
                        stack.append(nb)
            comp.sort()
            comps.append(comp)
        return comps

    for group_key, indices in group_items:
        if not indices:
            continue
        submatrix = dense_agg[np.ix_(indices, indices)]
        if nx is not None:
            graph = nx.Graph()
            graph.add_nodes_from(range(len(indices)))
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    weight = float(submatrix[i, j])
                    if weight <= min_weight:
                        continue
                    graph.add_edge(i, j, weight=weight)
            communities = _detect_communities(
                graph,
                method=method,
                resolution=resolution,
                seed=seed,
            )
        else:
            communities = _components_from_dense(submatrix)
        if not communities:
            communities = [[node] for node in range(len(indices))]
        for members_local in communities:
            members_global = [indices[idx] for idx in members_local]
            members_global.sort()
            rep_idx = max(
                members_global,
                key=lambda idx: (degree_weights[idx], species_names[idx].lower()),
            )
            name = f"S{superstate_id:03d}"
            cluster_meta = {
                "superstate_id": superstate_id,
                "name": name,
                "members": [species_names[idx] for idx in members_global],
                "member_indices": list(members_global),
                "size": len(members_global),
                "representative": species_names[rep_idx],
                "representative_index": int(rep_idx),
                "constraints": {
                    field: group_key[pos] for pos, field in enumerate(constraint_fields)
                },
            }
            clusters.append(cluster_meta)
            for idx in members_global:
                mapping.append(
                    {
                        "species": species_names[idx],
                        "species_index": int(idx),
                        "superstate_id": superstate_id,
                        "superstate": name,
                        "representative": species_names[rep_idx],
                        "constraints": {
                            field: group_key[pos]
                            for pos, field in enumerate(constraint_fields)
                        },
                    }
                )
            superstate_id += 1

    if not clusters:
        raise ConfigError("cnr_coarse produced no clusters.")

    mapping.sort(key=lambda entry: (entry["species"].lower(), entry["species_index"]))
    cluster_sizes = [cluster["size"] for cluster in clusters]
    size_stats = _cluster_size_stats(cluster_sizes)

    total_flux_weight = float(np.triu(dense_agg, k=1).sum()) if symmetrize else float(dense_agg.sum())
    within_flux = 0.0
    membership = {entry["species_index"]: entry["superstate_id"] for entry in mapping}
    for i in range(len(species_names)):
        for j in range(i + 1, len(species_names)):
            if membership.get(i) != membership.get(j):
                continue
            within_flux += float(dense_agg[i, j])
    if not symmetrize:
        for i in range(len(species_names)):
            if i not in membership:
                continue
            within_flux += float(dense_agg[i, i])
    flux_coverage = (
        within_flux / total_flux_weight if total_flux_weight > 0.0 else 0.0
    )

    mapping_payload = {
        "schema_version": 1,
        # Shared mapping contract: downstream consumers treat this as a superstate mapping.
        "kind": "superstate_mapping",
        "source": {"graph_id": graph_id},
        "producer": {"method": "cnr_coarse"},
        "policy": {
            "constraints": {"fields": list(constraint_fields)},
            "community": {
                "method": method,
                "requested_method": requested_method,
                "resolution": resolution,
                "min_weight": min_weight,
                "symmetrize": symmetrize,
            },
        },
        # Keep legacy fields for compatibility / traceability.
        "constraints": {"fields": list(constraint_fields), "entries": constraint_entries},
        "community": {
            "method": method,
            "requested_method": requested_method,
            "resolution": resolution,
            "min_weight": min_weight,
            "symmetrize": symmetrize,
        },
        "windowing": graph_payload.get("windowing"),
        "aggregation": graph_payload.get("aggregation"),
        "clusters": clusters,
        # Alias in the shared contract shape.
        "superstates": [
            {
                "superstate_id": cluster.get("superstate_id"),
                "name": cluster.get("name"),
                "representative": cluster.get("representative"),
                "members": cluster.get("members") or [],
                "summary": {
                    "size": cluster.get("size"),
                    "constraints": cluster.get("constraints") or {},
                },
            }
            for cluster in clusters
            if isinstance(cluster, Mapping)
        ],
        "mapping": mapping,
        "guards": {},
        "composition_meta": [],
    }

    metrics_payload = {
        "flux": {
            "total": total_flux_weight,
            "within_cluster": within_flux,
            "coverage": flux_coverage,
        },
        "cluster_sizes": size_stats,
        "window_count": len(window_summaries),
        "window_summaries": window_summaries,
    }

    inputs_payload = {
        "mode": "cnr_coarse",
        "graph": graph_id,
        "constraints": list(constraint_fields),
        "community": {
            "method": method,
            "resolution": resolution,
            "min_weight": min_weight,
            "symmetrize": symmetrize,
        },
    }

    artifact_id = reduction_cfg.get("artifact_id") or reduction_cfg.get("id")
    if artifact_id is None:
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
    artifact_id = _require_nonempty_str(artifact_id, "artifact_id")
    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    cluster_edges: list[dict[str, Any]] = []
    cluster_matrix = np.zeros((len(clusters), len(clusters)), dtype=float)
    for i in range(len(species_names)):
        for j in range(i + 1, len(species_names)):
            weight = float(dense_agg[i, j])
            if weight <= 0.0:
                continue
            ci = membership.get(i)
            cj = membership.get(j)
            if ci is None or cj is None:
                continue
            cluster_matrix[ci, cj] += weight
            cluster_matrix[cj, ci] += weight

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            weight = cluster_matrix[i, j]
            if weight <= 0.0:
                continue
            cluster_edges.append(
                {
                    "source": clusters[i]["name"],
                    "target": clusters[j]["name"],
                    "weight": float(weight),
                }
            )

    viz_payload = {
        "nodes": [
            {
                "id": cluster["name"],
                "size": cluster["size"],
                "representative": cluster["representative"],
            }
            for cluster in clusters
        ],
        "edges": cluster_edges,
    }

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "mapping.json", mapping_payload)
        write_json_atomic(base_dir / "metrics.json", metrics_payload)
        viz_dir = base_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(viz_dir / "cluster_graph.json", viz_payload)
        summary_lines = [
            "<html><head><meta charset=\"utf-8\"></head><body>",
            "<h1>CNR-Coarse Clusters</h1>",
            "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">",
            "<tr><th>Superstate</th><th>Size</th><th>Representative</th><th>Members</th></tr>",
        ]
        for cluster in clusters:
            members = ", ".join(cluster["members"])
            summary_lines.append(
                f"<tr><td>{cluster['name']}</td><td>{cluster['size']}</td>"
                f"<td>{cluster['representative']}</td><td>{members}</td></tr>"
            )
        summary_lines.append("</table></body></html>")
        (viz_dir / "cluster_summary.html").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


def gnn_pool_temporal(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Generate a DiffPool-style mapping with temporal encodings and constraints."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for gnn_pool_temporal mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)

    graph_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("graph", "graph_id", "graph_artifact"),
        label="reduction.graph",
    )
    if graph_id is None:
        graph_id = _extract_optional_artifact_id(
            params,
            keys=("graph", "graph_id", "graph_artifact"),
            label="reduction.graph",
        )
    if graph_id is None:
        raise ConfigError("gnn_pool_temporal requires graph_id.")

    dataset_id = _extract_optional_artifact_id(
        reduction_cfg,
        keys=("dataset", "dataset_id", "gnn_dataset", "gnn_dataset_id"),
        label="reduction.dataset",
    )
    if dataset_id is None:
        dataset_id = _extract_optional_artifact_id(
            params,
            keys=("dataset", "dataset_id", "gnn_dataset", "gnn_dataset_id"),
            label="reduction.dataset",
        )

    graph_payload = _load_graph_payload(store, graph_id)
    species_section = graph_payload.get("species") or {}
    species_order = species_section.get("order")
    if not isinstance(species_order, Sequence) or isinstance(
        species_order, (str, bytes, bytearray)
    ):
        raise ConfigError("temporal graph species.order must be a list.")
    species_names = [str(name) for name in species_order]
    if not species_names:
        raise ConfigError("temporal graph has no species.")

    species_graph = graph_payload.get("species_graph")
    if not isinstance(species_graph, Mapping):
        raise ConfigError("temporal graph is missing species_graph metadata.")
    layers_meta = species_graph.get("layers")
    if not isinstance(layers_meta, Sequence) or isinstance(
        layers_meta, (str, bytes, bytearray)
    ):
        raise ConfigError("temporal graph layers must be a list.")
    if not layers_meta:
        raise ConfigError("temporal graph has no layers.")

    pool_cfg = params.get("pool") or params.get("diffpool") or {}
    if pool_cfg is None:
        pool_cfg = {}
    if not isinstance(pool_cfg, Mapping):
        raise ConfigError("pool config must be a mapping.")
    pool_cfg = dict(pool_cfg)

    temporal_cfg = params.get("temporal") or params.get("temporal_encoder") or {}
    if temporal_cfg is None:
        temporal_cfg = {}
    if not isinstance(temporal_cfg, Mapping):
        raise ConfigError("temporal config must be a mapping.")
    temporal_cfg = dict(temporal_cfg)

    selection_cfg = params.get("selection") or {}
    if selection_cfg is None:
        selection_cfg = {}
    if not isinstance(selection_cfg, Mapping):
        raise ConfigError("selection config must be a mapping.")
    selection_cfg = dict(selection_cfg)

    loss_cfg = params.get("self_supervised") or params.get("loss") or {}
    if loss_cfg is None:
        loss_cfg = {}
    if not isinstance(loss_cfg, Mapping):
        raise ConfigError("loss config must be a mapping.")
    loss_cfg = dict(loss_cfg)

    constraints_cfg = params.get("constraints") or reduction_cfg.get("constraints")
    constraint_fields = _normalize_constraint_fields(constraints_cfg)

    symmetrize = _coerce_bool(
        pool_cfg.get("symmetrize"),
        "pool.symmetrize",
        default=DEFAULT_CNR_SYMMETRIZE,
    )
    time_dim = temporal_cfg.get("time_dim", temporal_cfg.get("dim", DEFAULT_GNN_POOL_TIME_DIM))
    if isinstance(time_dim, bool) or not isinstance(time_dim, int) or time_dim < 0:
        raise ConfigError("temporal.time_dim must be a non-negative integer.")
    time_base = _coerce_optional_float(
        temporal_cfg.get("time_base", temporal_cfg.get("base")),
        "temporal.time_base",
    )
    if time_base is None:
        time_base = DEFAULT_GNN_POOL_TIME_BASE
    if time_base <= 0.0:
        raise ConfigError("temporal.time_base must be positive.")
    message_passing = temporal_cfg.get("message_passing", DEFAULT_GNN_POOL_MESSAGE_PASSING)
    if isinstance(message_passing, bool) or not isinstance(message_passing, int):
        raise ConfigError("temporal.message_passing must be an integer.")
    if message_passing < 0:
        raise ConfigError("temporal.message_passing must be >= 0.")

    temperature = pool_cfg.get("temperature", pool_cfg.get("temp", DEFAULT_GNN_POOL_TEMP))
    if temperature is None:
        temperature = DEFAULT_GNN_POOL_TEMP
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ConfigError("pool.temperature must be numeric.") from exc
    max_iter = pool_cfg.get("max_iter", DEFAULT_GNN_POOL_MAX_ITER)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ConfigError("pool.max_iter must be an integer.")
    if max_iter <= 0:
        raise ConfigError("pool.max_iter must be positive.")

    k_raw = (
        pool_cfg.get("k_sweep")
        or pool_cfg.get("k")
        or params.get("k_sweep")
        or params.get("k")
    )
    k_sweep = _normalize_k_sweep(k_raw, total_nodes=len(species_names))
    k_sweep = [k for k in k_sweep if k > 0]
    if not k_sweep:
        raise ConfigError("k_sweep produced no valid cluster counts.")

    min_val_cov = selection_cfg.get("min_val_flux_coverage")
    if min_val_cov is not None:
        try:
            min_val_cov = float(min_val_cov)
        except (TypeError, ValueError) as exc:
            raise ConfigError("selection.min_val_flux_coverage must be numeric.") from exc
        if not 0.0 <= min_val_cov <= 1.0:
            raise ConfigError("selection.min_val_flux_coverage must be in [0, 1].")
    coverage_slack = selection_cfg.get("coverage_slack", DEFAULT_GNN_POOL_COVERAGE_SLACK)
    try:
        coverage_slack = float(coverage_slack)
    except (TypeError, ValueError) as exc:
        raise ConfigError("selection.coverage_slack must be numeric.") from exc
    if coverage_slack < 0.0:
        raise ConfigError("selection.coverage_slack must be >= 0.")

    seed = None
    common_cfg = manifest_cfg.get("common")
    if isinstance(common_cfg, Mapping):
        raw_seed = common_cfg.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed
    if seed is None:
        raw_seed = reduction_cfg.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            seed = raw_seed
    if seed is None:
        seed = 0

    source_meta = graph_payload.get("source")
    mechanism = None
    phase_default = None
    if isinstance(source_meta, Mapping):
        mechanism = _coerce_optional_str(source_meta.get("mechanism"), "mechanism")
        phase_default = _coerce_optional_str(source_meta.get("phase"), "phase")

    meta_from_graph = _species_metadata_from_graph(graph_payload, species_names)
    meta_from_mech = _species_metadata_from_mechanism(mechanism, phase_default)
    merged_meta: dict[str, dict[str, Any]] = {}
    for name in species_names:
        merged = {}
        if name in meta_from_mech:
            merged.update(meta_from_mech[name])
        if name in meta_from_graph:
            merged.update(meta_from_graph[name])
        merged_meta[name] = merged

    groups, constraint_entries = _build_constraint_groups(
        species_names,
        constraints=constraint_fields,
        metadata=merged_meta,
        phase_default=phase_default,
    )
    group_items = sorted(groups.items(), key=lambda item: item[0])
    if not group_items:
        raise ConfigError("gnn_pool_temporal produced no constraint groups.")

    graph_dir = store.artifact_dir("graphs", graph_id)
    layer_lookup: dict[int, dict[str, Any]] = {}
    for entry in layers_meta:
        if not isinstance(entry, Mapping):
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        index_value = entry.get("index")
        if index_value is None:
            index_value = len(layer_lookup)
        try:
            index_value = int(index_value)
        except (TypeError, ValueError):
            continue
        layer_lookup[index_value] = {
            "index": index_value,
            "path": path_value,
            "window": entry.get("window", {}),
        }
    if not layer_lookup:
        raise ConfigError("temporal graph layers could not be indexed.")

    dataset_payload: Optional[dict[str, Any]] = None
    dataset_items: list[dict[str, Any]] = []
    dataset_splits: dict[str, list[int]] = {}
    if dataset_id is not None:
        dataset_payload, dataset_dir = _load_gnn_dataset_payload(store, dataset_id)
        dataset_items, dataset_splits = _load_gnn_dataset_items(
            dataset_payload, dataset_dir=dataset_dir
        )

    features_by_window: dict[int, Any] = {}
    if dataset_items:
        accum: dict[int, list[Any]] = {}
        for item in dataset_items:
            window_id = item.get("window_id")
            features = item.get("features")
            if window_id is None or features is None:
                continue
            window_id = int(window_id)
            try:
                matrix = np.asarray(features, dtype=float)
            except Exception as exc:
                raise ConfigError("dataset features must be numeric.") from exc
            if matrix.shape[0] != len(species_names):
                raise ConfigError(
                    "dataset node feature count does not match species count."
                )
            accum.setdefault(window_id, []).append(matrix)
        for window_id, matrices in accum.items():
            if not matrices:
                continue
            stacked = np.stack(matrices, axis=0)
            features_by_window[window_id] = np.nan_to_num(stacked.mean(axis=0))

    if not features_by_window:
        for window_id, meta in layer_lookup.items():
            layer_path = graph_dir / str(meta["path"])
            matrix = _load_csr_matrix(layer_path)
            dense = _dense_from_sparse(matrix)
            degree = np.sum(dense, axis=1)
            features_by_window[window_id] = degree.reshape(-1, 1)

    all_window_ids = sorted(layer_lookup.keys())
    if len(all_window_ids) < 2:
        raise ConfigError(
            "gnn_pool_temporal requires at least 2 windows to split train/val "
            f"(windows={len(all_window_ids)}, window_split={window_split_cfg!r}). "
            "Increase windowing.count or provide distinct window_split.train_window_ids/"
            "val_window_ids."
        )
    valid_window_ids = set(all_window_ids)
    train_window_ids: list[int] = []
    val_window_ids: list[int] = []
    window_split_cfg = _normalize_window_split_cfg(
        params.get("window_split")
        or params.get("window_splits")
        or reduction_cfg.get("window_split")
        or reduction_cfg.get("window_splits")
    )
    if window_split_cfg:
        train_window_ids, val_window_ids = _window_split_from_cfg(
            all_window_ids, window_split_cfg
        )
    else:
        if dataset_items and dataset_splits:
            for idx in dataset_splits.get("train", []):
                if 0 <= idx < len(dataset_items):
                    train_window_ids.append(int(dataset_items[idx]["window_id"]))
            for idx in dataset_splits.get("val", []):
                if 0 <= idx < len(dataset_items):
                    val_window_ids.append(int(dataset_items[idx]["window_id"]))
        if not train_window_ids:
            split_at = max(1, int(0.8 * len(all_window_ids)))
            train_window_ids = all_window_ids[:split_at]
        if not val_window_ids:
            val_window_ids = all_window_ids[len(train_window_ids) :]
        if not val_window_ids and all_window_ids:
            val_window_ids = [all_window_ids[-1]]

    train_window_ids = sorted(
        {wid for wid in train_window_ids if wid in valid_window_ids}
    )
    val_window_ids = sorted({wid for wid in val_window_ids if wid in valid_window_ids})
    overlap = set(train_window_ids) & set(val_window_ids)
    if overlap:
        val_window_ids = [wid for wid in val_window_ids if wid not in overlap]
    if not val_window_ids and all_window_ids:
        train_set = set(train_window_ids)
        candidates = [wid for wid in all_window_ids if wid not in train_set]
        val_window_ids = [candidates[-1]] if candidates else [all_window_ids[-1]]

    time_values: dict[int, float] = {}
    for window_id in all_window_ids:
        meta = layer_lookup[window_id]
        window_info = meta.get("window", {})
        t_value = None
        if isinstance(window_info, Mapping):
            start_time = window_info.get("start_time")
            end_time = window_info.get("end_time")
            if start_time is not None and end_time is not None:
                try:
                    t_value = 0.5 * (float(start_time) + float(end_time))
                except (TypeError, ValueError):
                    t_value = None
            if t_value is None:
                start_idx = window_info.get("start_idx")
                end_idx = window_info.get("end_idx")
                if start_idx is not None and end_idx is not None:
                    try:
                        t_value = 0.5 * (float(start_idx) + float(end_idx))
                    except (TypeError, ValueError):
                        t_value = None
        if t_value is None:
            t_value = float(window_id)
        time_values[window_id] = t_value

    t_min = min(time_values.values()) if time_values else 0.0
    t_max = max(time_values.values()) if time_values else 1.0
    t_span = t_max - t_min if t_max > t_min else 1.0

    def _window_encoding(window_id: int) -> list[float]:
        t_norm = (time_values[window_id] - t_min) / t_span if t_span > 0.0 else 0.0
        return _temporal_encoding(t_norm, time_dim=time_dim, time_base=time_base)

    def _aggregate_adjacency(window_ids: Sequence[int]) -> tuple[Any, float]:
        aggregated = None
        total_flux = 0.0
        for window_id in window_ids:
            meta = layer_lookup.get(window_id)
            if meta is None:
                continue
            layer_path = graph_dir / str(meta["path"])
            matrix = _load_csr_matrix(layer_path)
            dense_layer = None
            if sp is not None and hasattr(matrix, "sum"):
                layer_flux = float(matrix.sum())
            else:
                dense_layer = _dense_from_sparse(matrix)
                layer_flux = float(dense_layer.sum())
            total_flux += layer_flux
            if aggregated is None:
                aggregated = matrix if dense_layer is None else dense_layer
            else:
                if sp is not None and hasattr(aggregated, "shape") and hasattr(matrix, "shape"):
                    aggregated = aggregated + matrix
                else:
                    if dense_layer is None:
                        dense_layer = _dense_from_sparse(matrix)
                    aggregated = _dense_from_sparse(aggregated) + dense_layer
        if aggregated is None:
            aggregated = np.zeros((len(species_names), len(species_names)), dtype=float)
        dense = _dense_from_sparse(aggregated)
        if dense.shape[0] != len(species_names):
            raise ConfigError("temporal graph matrix shape does not match species list.")
        if symmetrize:
            dense = 0.5 * (dense + dense.T)
        return dense, total_flux

    adjacency_all, total_flux_all = _aggregate_adjacency(all_window_ids)
    adjacency_train, total_flux_train = _aggregate_adjacency(train_window_ids)
    adjacency_val, total_flux_val = _aggregate_adjacency(val_window_ids)

    window_summaries: list[dict[str, Any]] = []
    for window_id in all_window_ids:
        meta = layer_lookup.get(window_id)
        if meta is None:
            continue
        window_info = meta.get("window", {})
        window_summaries.append(
            {
                "index": window_id,
                "path": meta.get("path"),
                "window": window_info,
            }
        )

    feature_mats: list[Any] = []
    for window_id in train_window_ids:
        features = features_by_window.get(window_id)
        if features is None:
            features = np.zeros((len(species_names), 1), dtype=float)
        matrix = np.asarray(features, dtype=float)
        if matrix.shape[0] != len(species_names):
            raise ConfigError("feature matrix has invalid node count.")
        matrix = np.nan_to_num(matrix)
        enc = _window_encoding(window_id)
        if enc:
            enc_mat = np.tile(np.asarray(enc, dtype=float), (matrix.shape[0], 1))
            matrix = np.concatenate([matrix, enc_mat], axis=1)
        feature_mats.append(matrix)
    if not feature_mats:
        raise ConfigError("No feature matrices available for training windows.")

    agg_features = np.mean(np.stack(feature_mats, axis=0), axis=0)
    embeddings = np.asarray(agg_features, dtype=float)
    if message_passing > 0:
        norm_adj = _row_normalize(adjacency_train + np.eye(len(species_names)))
        for _ in range(message_passing):
            embeddings = norm_adj @ embeddings

    degree_weights = np.sum(adjacency_train, axis=1)
    degree_weights = np.asarray(degree_weights, dtype=float).ravel()

    group_sizes = [len(indices) for _, indices in group_items]
    sweep_results: list[dict[str, Any]] = []

    for k_total in k_sweep:
        group_counts = _allocate_group_clusters(
            total_k=k_total,
            group_sizes=group_sizes,
        )
        clusters: list[dict[str, Any]] = []
        mapping: list[dict[str, Any]] = []
        membership = [-1 for _ in range(len(species_names))]
        recon_losses: list[float] = []
        recon_weights: list[float] = []
        recon_val_losses: list[float] = []
        cluster_id = 0
        for (group_key, indices), k_group in zip(group_items, group_counts):
            if not indices:
                continue
            if k_group <= 0:
                k_group = 1
            k_group = min(k_group, len(indices))
            local_embeddings = embeddings[indices]
            if k_group >= len(indices):
                labels_local = list(range(len(indices)))
                centers = np.asarray(local_embeddings, dtype=float)
                assignment = np.eye(len(indices), dtype=float)
            else:
                centers, labels_local = _kmeans_assign(
                    local_embeddings,
                    k=k_group,
                    seed=seed,
                    max_iter=max_iter,
                )
                assignment = _soft_assignment(
                    local_embeddings,
                    centers,
                    temperature=temperature,
                )
            sub_adj = adjacency_train[np.ix_(indices, indices)]
            recon_losses.append(_edge_reconstruction_loss(sub_adj, assignment))
            recon_weights.append(float(np.sum(sub_adj)))
            if adjacency_val is not None:
                sub_val = adjacency_val[np.ix_(indices, indices)]
                recon_val_losses.append(_edge_reconstruction_loss(sub_val, assignment))

            cluster_members: dict[int, list[int]] = {}
            for local_idx, global_idx in enumerate(indices):
                label = int(labels_local[local_idx])
                cluster_members.setdefault(label, []).append(global_idx)

            for local_label, member_indices in cluster_members.items():
                if not member_indices:
                    continue
                member_indices.sort()
                rep_idx = max(
                    member_indices,
                    key=lambda idx: (degree_weights[idx], species_names[idx].lower()),
                )
                name = f"S{cluster_id:03d}"
                cluster_meta = {
                    "superstate_id": cluster_id,
                    "name": name,
                    "members": [species_names[idx] for idx in member_indices],
                    "member_indices": list(member_indices),
                    "size": len(member_indices),
                    "representative": species_names[rep_idx],
                    "representative_index": int(rep_idx),
                    "constraints": {
                        field: group_key[pos] for pos, field in enumerate(constraint_fields)
                    },
                }
                clusters.append(cluster_meta)
                for idx in member_indices:
                    membership[idx] = cluster_id
                    mapping.append(
                        {
                            "species": species_names[idx],
                            "species_index": int(idx),
                            "superstate_id": cluster_id,
                            "superstate": name,
                            "representative": species_names[rep_idx],
                            "constraints": {
                                field: group_key[pos]
                                for pos, field in enumerate(constraint_fields)
                            },
                        }
                    )
                cluster_id += 1

        if any(value < 0 for value in membership):
            raise ConfigError("gnn_pool_temporal produced incomplete membership mapping.")

        total_train, within_train, coverage_train = _flux_coverage(
            adjacency_train, membership, symmetrize=symmetrize
        )
        total_val, within_val, coverage_val = _flux_coverage(
            adjacency_val, membership, symmetrize=symmetrize
        )
        total_all, within_all, coverage_all = _flux_coverage(
            adjacency_all, membership, symmetrize=symmetrize
        )

        if recon_weights and sum(recon_weights) > 0.0:
            train_loss = float(
                sum(loss * weight for loss, weight in zip(recon_losses, recon_weights))
                / sum(recon_weights)
            )
        else:
            train_loss = float(statistics.mean(recon_losses)) if recon_losses else 0.0
        if recon_val_losses:
            val_loss = float(statistics.mean(recon_val_losses))
        else:
            val_loss = train_loss

        cluster_sizes = [cluster["size"] for cluster in clusters]
        size_stats = _cluster_size_stats(cluster_sizes)

        sweep_results.append(
            {
                "k_requested": int(k_total),
                "k_actual": int(len(clusters)),
                "group_counts": list(group_counts),
                "clusters": clusters,
                "mapping": mapping,
                "membership": membership,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "coverage": {
                    "train": coverage_train,
                    "val": coverage_val,
                    "all": coverage_all,
                },
                "flux": {
                    "train": {"total": total_train, "within_cluster": within_train},
                    "val": {"total": total_val, "within_cluster": within_val},
                    "all": {"total": total_all, "within_cluster": within_all},
                },
                "cluster_sizes": size_stats,
            }
        )

    if not sweep_results:
        raise ConfigError("gnn_pool_temporal produced no sweep results.")

    max_cov = max(result["coverage"]["val"] for result in sweep_results)
    if min_val_cov is not None:
        threshold = min_val_cov
    else:
        threshold = max(0.0, max_cov - coverage_slack)

    eligible = [
        result for result in sweep_results if result["coverage"]["val"] >= threshold
    ]
    if not eligible:
        eligible = list(sweep_results)
    eligible.sort(
        key=lambda result: (
            result["k_actual"],
            result["val_loss"],
            -result["coverage"]["val"],
        )
    )
    selected = eligible[0]

    selected_clusters = selected["clusters"]
    selected_mapping = selected["mapping"]
    selected_mapping.sort(
        key=lambda entry: (entry["species"].lower(), entry["species_index"])
    )
    selected_sizes = [cluster["size"] for cluster in selected_clusters]

    mapping_payload = {
        "schema_version": 1,
        # Shared mapping contract: downstream consumers treat this as a superstate mapping.
        "kind": "superstate_mapping",
        "source": {"graph_id": graph_id, "dataset_id": dataset_id},
        "producer": {"method": "gnn_pool_temporal"},
        "policy": {
            "constraints": {"fields": list(constraint_fields)},
            "temporal_encoder": {
                "type": "sincos",
                "time_dim": int(time_dim),
                "time_base": float(time_base),
                "message_passing": int(message_passing),
            },
            "pooling": {
                "method": "diffpool",
                "k_sweep": [int(item["k_requested"]) for item in sweep_results],
                "k_selected": int(selected["k_actual"]),
                "temperature": float(temperature),
                "max_iter": int(max_iter),
            },
            "self_supervised": {
                "loss": "edge_reconstruction",
                "train_loss": float(selected["train_loss"]),
                "val_loss": float(selected["val_loss"]),
            },
        },
        # Keep legacy fields for compatibility / traceability.
        "constraints": {"fields": list(constraint_fields), "entries": constraint_entries},
        "temporal_encoder": {
            "type": "sincos",
            "time_dim": int(time_dim),
            "time_base": float(time_base),
            "message_passing": int(message_passing),
        },
        "pooling": {
            "method": "diffpool",
            "k_sweep": [int(item["k_requested"]) for item in sweep_results],
            "k_selected": int(selected["k_actual"]),
            "temperature": float(temperature),
            "max_iter": int(max_iter),
        },
        "self_supervised": {
            "loss": "edge_reconstruction",
            "train_loss": float(selected["train_loss"]),
            "val_loss": float(selected["val_loss"]),
        },
        "windowing": graph_payload.get("windowing"),
        "aggregation": graph_payload.get("aggregation"),
        "clusters": selected_clusters,
        "superstates": [
            {
                "superstate_id": cluster.get("superstate_id"),
                "name": cluster.get("name"),
                "representative": cluster.get("representative"),
                "members": cluster.get("members") or [],
                "summary": {
                    "size": cluster.get("size"),
                    "constraints": cluster.get("constraints") or {},
                },
            }
            for cluster in selected_clusters
            if isinstance(cluster, Mapping)
        ],
        "mapping": selected_mapping,
        "guards": {},
        "composition_meta": [],
    }

    metrics_payload = {
        "flux": {
            "train": {
                "total": selected["flux"]["train"]["total"],
                "within_cluster": selected["flux"]["train"]["within_cluster"],
                "coverage": selected["coverage"]["train"],
            },
            "val": {
                "total": selected["flux"]["val"]["total"],
                "within_cluster": selected["flux"]["val"]["within_cluster"],
                "coverage": selected["coverage"]["val"],
            },
            "all": {
                "total": selected["flux"]["all"]["total"],
                "within_cluster": selected["flux"]["all"]["within_cluster"],
                "coverage": selected["coverage"]["all"],
            },
        },
        "loss": {
            "edge_reconstruction": {
                "train": float(selected["train_loss"]),
                "val": float(selected["val_loss"]),
            }
        },
        "selection": {
            "min_val_flux_coverage": min_val_cov,
            "coverage_slack": float(coverage_slack),
            "coverage_threshold": float(threshold),
            "k_selected": int(selected["k_actual"]),
            "k_requested": int(selected["k_requested"]),
        },
        "cluster_sizes": _cluster_size_stats(selected_sizes),
        "window_count": len(all_window_ids),
        "window_summaries": window_summaries,
        "sweep": [
            {
                "k_requested": item["k_requested"],
                "k_actual": item["k_actual"],
                "train_loss": item["train_loss"],
                "val_loss": item["val_loss"],
                "coverage_val": item["coverage"]["val"],
                "coverage_train": item["coverage"]["train"],
                "coverage_all": item["coverage"]["all"],
            }
            for item in sweep_results
        ],
        "dataset": {
            "dataset_id": dataset_id,
            "train_windows": train_window_ids,
            "val_windows": val_window_ids,
        },
        "flux_totals": {
            "train": float(total_flux_train),
            "val": float(total_flux_val),
            "all": float(total_flux_all),
        },
    }

    inputs_payload = {
        "mode": "gnn_pool_temporal",
        "graph": graph_id,
        "dataset": dataset_id,
        "constraints": list(constraint_fields),
        "pool": {
            "k_sweep": [int(item["k_requested"]) for item in sweep_results],
            "temperature": float(temperature),
        },
    }

    artifact_id = reduction_cfg.get("artifact_id") or reduction_cfg.get("id")
    if artifact_id is None:
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
    artifact_id = _require_nonempty_str(artifact_id, "artifact_id")
    parents = [graph_id]
    if dataset_id:
        parents.append(dataset_id)

    manifest = build_manifest(
        kind="reduction",
        artifact_id=artifact_id,
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "mapping.json", mapping_payload)
        write_json_atomic(base_dir / "metrics.json", metrics_payload)

    return store.ensure(manifest, writer=_writer)


def _load_mapping_payload(store: ArtifactStore, mapping_id: str) -> dict[str, Any]:
    store.read_manifest("reduction", mapping_id)
    mapping_path = store.artifact_dir("reduction", mapping_id) / "mapping.json"
    if not mapping_path.exists():
        raise ConfigError(f"mapping.json not found for mapping {mapping_id!r}.")
    try:
        payload = read_json(mapping_path)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"mapping.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("mapping.json must contain a JSON object.")
    return dict(payload)


def _read_run_results(run_root: Path, label: str) -> dict[str, Any]:
    metrics = read_run_metrics(run_root)
    if not isinstance(metrics, Mapping):
        raise ConfigError(f"{label} run metrics not found: {run_root}")
    results = metrics.get("results")
    if not isinstance(results, Mapping):
        raise ConfigError(f"{label} run metrics missing results mapping.")
    return dict(results)


def _coerce_optional_path(value: Any, label: str) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return Path(value)


def _extract_eval_group(
    inputs_cfg: Mapping[str, Any],
    *,
    group: str,
) -> dict[str, Any]:
    group_cfg: dict[str, Any] = {}
    raw = inputs_cfg.get(group)
    if isinstance(raw, Mapping):
        group_cfg.update(raw)
    prefix = f"{group}_"
    for key, value in inputs_cfg.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        group_cfg[key[len(prefix) :]] = value
    return group_cfg


def _select_result_artifact(
    results: Mapping[str, Any],
    *,
    label: str,
    step_name: str,
    store_root: Path,
    artifact_kind: str,
) -> Optional[str]:
    if step_name in results and isinstance(results.get(step_name), str):
        return str(results.get(step_name))
    fallback_steps = ("mapping", "reduction", "artifact_id")
    for key in fallback_steps:
        value = results.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in results.values():
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = store_root / artifact_kind / value / "metrics.json"
        if candidate.exists():
            return value
    return None


def _select_graph_artifact(
    results: Mapping[str, Any],
    *,
    graph_step: str,
) -> Optional[str]:
    if graph_step in results and isinstance(results.get(graph_step), str):
        return str(results.get(graph_step))
    for key in ("graph", "graph_id"):
        value = results.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _expand_window_ids(
    window_ids: Sequence[int],
    *,
    radius: int,
    all_ids: Sequence[int],
) -> list[int]:
    if radius <= 0:
        return sorted(set(window_ids))
    all_set = set(all_ids)
    expanded: set[int] = set()
    for window_id in window_ids:
        for offset in range(-radius, radius + 1):
            candidate = int(window_id) + offset
            if candidate in all_set:
                expanded.add(candidate)
    return sorted(expanded)


def _select_qoi_windows(
    qoi_cfg: Mapping[str, Any],
    *,
    window_meta: Sequence[Mapping[str, Any]],
) -> list[int]:
    selector = qoi_cfg.get("selector") or DEFAULT_EVAL_QOI_SELECTOR
    if not isinstance(selector, str) or not selector.strip():
        raise ConfigError("qoi.selector must be a non-empty string.")
    selector = selector.strip()
    window_ids = [int(entry.get("index", idx)) for idx, entry in enumerate(window_meta)]
    if selector == "all":
        return sorted(set(window_ids))
    if selector == "window_ids":
        raw_ids = qoi_cfg.get("window_ids") or qoi_cfg.get("windows") or []
        return _coerce_int_sequence(raw_ids, "qoi.window_ids")
    if selector == "time_range":
        start = qoi_cfg.get("start_time")
        end = qoi_cfg.get("end_time")
        if start is None or end is None:
            raise ConfigError("qoi.time_range requires start_time and end_time.")
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError) as exc:
            raise ConfigError("qoi.start_time/end_time must be numeric.") from exc
        selected: list[int] = []
        for entry in window_meta:
            info = entry.get("window", {})
            if not isinstance(info, Mapping):
                continue
            w_start = info.get("start_time")
            w_end = info.get("end_time")
            if w_start is None or w_end is None:
                continue
            try:
                w_start_f = float(w_start)
                w_end_f = float(w_end)
            except (TypeError, ValueError):
                continue
            if w_end_f < start_f or w_start_f > end_f:
                continue
            selected.append(int(entry.get("index", 0)))
        return sorted(set(selected))
    if selector == "centered":
        center = qoi_cfg.get("center_index")
        if center is None:
            raise ConfigError("qoi.centered requires center_index.")
        if isinstance(center, bool) or not isinstance(center, int):
            raise ConfigError("qoi.center_index must be an integer.")
        return [int(center)]
    if "." in selector:
        module_name, func_name = selector.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        except Exception as exc:
            raise ConfigError(f"qoi.selector import failed: {selector}") from exc
        if not callable(func):
            raise ConfigError("qoi.selector must resolve to a callable.")
        try:
            result = func(window_meta=window_meta, qoi_cfg=dict(qoi_cfg))
        except TypeError:
            result = func(window_meta, qoi_cfg)
        return _coerce_int_sequence(result, "qoi.selector")
    raise ConfigError(f"Unsupported qoi.selector: {selector}")


def _cluster_flux_vectors(
    window_matrices: Sequence[Any],
    *,
    cluster_indices: Sequence[Sequence[int]],
    symmetrize: bool,
) -> list[list[float]]:
    if np is None:
        raise ConfigError("numpy is required for cluster stability metrics.")
    vectors: list[list[float]] = []
    for dense in window_matrices:
        matrix = np.asarray(dense, dtype=float)
        vector: list[float] = []
        for indices in cluster_indices:
            if not indices:
                vector.append(0.0)
                continue
            sub = matrix[np.ix_(indices, indices)]
            if symmetrize:
                value = float(np.triu(sub, k=1).sum())
            else:
                value = float(sub.sum())
            vector.append(value)
        vectors.append(vector)
    return vectors


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if np is None:
        raise ConfigError("numpy is required for cluster stability metrics.")
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_histogram_bins(max_size: int, max_bins: int) -> tuple[list[int], list[str], bool]:
    if max_size <= max_bins:
        bins = list(range(1, max_size + 1))
        labels = [str(val) for val in bins]
        return bins, labels, False
    bins = list(range(1, max_bins)) + [max_bins]
    labels = [str(val) for val in bins[:-1]] + [f">={max_bins}"]
    return bins, labels, True


def _hist_counts(
    sizes: Sequence[int],
    *,
    bins: Sequence[int],
    overflow: bool,
) -> list[int]:
    counts = [0 for _ in bins]
    for size in sizes:
        if overflow and size >= bins[-1]:
            counts[-1] += 1
            continue
        for idx, bin_value in enumerate(bins):
            if size == bin_value:
                counts[idx] += 1
                break
    return counts


def _render_cluster_size_svg(
    *,
    labels: Sequence[str],
    gnn_counts: Sequence[int],
    cnr_counts: Sequence[int],
    title: str,
) -> str:
    width = 720
    height = 360
    margin = 48
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    series = max(len(labels), len(gnn_counts), len(cnr_counts))
    if series <= 0:
        series = 1
    max_count = max(max(gnn_counts, default=0), max(cnr_counts, default=0), 1)
    bar_group_width = chart_width / float(series)
    bar_width = bar_group_width * 0.35
    gap = bar_group_width * 0.15
    svg_lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">",
        f"<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\" />",
        f"<text x=\"{margin}\" y=\"{margin - 18}\" font-family=\"Arial\" font-size=\"16\" fill=\"#222\">{title}</text>",
        f"<line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#666\" />",
        f"<line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#666\" />",
        f"<text x=\"{width - margin - 120}\" y=\"{margin}\" font-family=\"Arial\" font-size=\"12\" fill=\"#0f6f68\">GNN</text>",
        f"<text x=\"{width - margin - 60}\" y=\"{margin}\" font-family=\"Arial\" font-size=\"12\" fill=\"#c05a3a\">CNR</text>",
    ]
    for idx in range(series):
        label = labels[idx] if idx < len(labels) else ""
        gnn = gnn_counts[idx] if idx < len(gnn_counts) else 0
        cnr = cnr_counts[idx] if idx < len(cnr_counts) else 0
        x_base = margin + idx * bar_group_width
        gnn_height = chart_height * (gnn / float(max_count))
        cnr_height = chart_height * (cnr / float(max_count))
        y_gnn = height - margin - gnn_height
        y_cnr = height - margin - cnr_height
        svg_lines.append(
            f"<rect x=\"{x_base + gap}\" y=\"{y_gnn}\" width=\"{bar_width}\" height=\"{gnn_height}\" fill=\"#0f6f68\" />"
        )
        svg_lines.append(
            f"<rect x=\"{x_base + gap + bar_width + gap}\" y=\"{y_cnr}\" width=\"{bar_width}\" height=\"{cnr_height}\" fill=\"#c05a3a\" />"
        )
        svg_lines.append(
            f"<text x=\"{x_base + bar_group_width / 2}\" y=\"{height - margin + 14}\" font-family=\"Arial\" font-size=\"10\" text-anchor=\"middle\" fill=\"#333\">{label}</text>"
        )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def _format_mapping_eval_report(
    *,
    report_format: str,
    gnn_metrics: Mapping[str, Any],
    cnr_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    qoi_info: Mapping[str, Any],
    report_cfg: Mapping[str, Any],
    viz_filename: str,
) -> str:
    format_lower = report_format.lower()
    title = "Mapping Evaluation Report"
    def _fmt(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "n/a"
        value_f = float(value)
        abs_val = abs(value_f)
        if abs_val != 0.0 and abs_val < 1.0e-3:
            return f"{value_f:.2e}"
        return f"{value_f:.4f}"
    summary_rows = [
        ("flux_coverage", gnn_metrics.get("flux", {}).get("coverage"), cnr_metrics.get("flux", {}).get("coverage")),
        ("qoi_coverage", gnn_metrics.get("qoi", {}).get("coverage"), cnr_metrics.get("qoi", {}).get("coverage")),
        ("qoi_retention", gnn_metrics.get("qoi", {}).get("retention"), cnr_metrics.get("qoi", {}).get("retention")),
        ("cluster_stability", gnn_metrics.get("stability", {}).get("cluster_similarity"), cnr_metrics.get("stability", {}).get("cluster_similarity")),
    ]
    if format_lower == "html":
        rows_html = []
        for name, gnn_val, cnr_val in summary_rows:
            delta = comparison.get(name)
            rows_html.append(
                "<tr>"
                f"<td>{name}</td>"
                f"<td>{_fmt(gnn_val)}</td>"
                f"<td>{_fmt(cnr_val)}</td>"
                f"<td>{_fmt(delta)}</td>"
                "</tr>"
            )
        return (
            "<html><head><meta charset=\"utf-8\"></head><body>"
            f"<h1>{title}</h1>"
            "<h2>Metric Summary</h2>"
            "<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">"
            "<tr><th>Metric</th><th>GNN</th><th>CNR</th><th>Delta</th></tr>"
            + "".join(rows_html)
            + "</table>"
            f"<h2>QoI Windows</h2><p>Selector: {qoi_info.get('selector')}</p>"
            f"<p>Window count: {qoi_info.get('count')}</p>"
            f"<h2>Cluster Size Distribution</h2><img src=\"{viz_filename}\" alt=\"cluster sizes\" />"
            "</body></html>\n"
        )
    lines = [f"# {title}", "", "## Metric Summary", "| Metric | GNN | CNR | Delta |", "| --- | --- | --- | --- |"]
    for name, gnn_val, cnr_val in summary_rows:
        delta = comparison.get(name)
        lines.append(f"| {name} | {_fmt(gnn_val)} | {_fmt(cnr_val)} | {_fmt(delta)} |")
    lines.append("")
    lines.append("## QoI Windows")
    lines.append(f"- selector: {qoi_info.get('selector')}")
    lines.append(f"- window_count: {qoi_info.get('count')}")
    lines.append("")
    lines.append("## Cluster Size Distribution")
    lines.append(f"![cluster sizes]({viz_filename})")
    max_windows = report_cfg.get("max_windows", DEFAULT_EVAL_REPORT_MAX_WINDOWS)
    if isinstance(max_windows, int) and max_windows > 0:
        lines.append("")
        lines.append("## Coverage by Window (GNN)")
        coverage_rows = gnn_metrics.get("coverage_by_window", [])
        lines.append("| window | coverage | flux_total |")
        lines.append("| --- | --- | --- |")
        for entry in coverage_rows[: max_windows]:
            idx = entry.get("index")
            cov = entry.get("coverage")
            total = entry.get("flux_total")
            lines.append(f"| {idx} | {_fmt(cov)} | {_fmt(total)} |")

    qoi_species_rows = gnn_metrics.get("qoi_species_purity") or []
    if isinstance(qoi_species_rows, Sequence) and not isinstance(
        qoi_species_rows, (str, bytes, bytearray)
    ):
        lines.append("")
        lines.append("## QoI Species Purity (Last Time Point)")
        lines.append(
            "This is a proxy metric for state-merge risk: if a species is clustered with others,"
            " it becomes indistinguishable after merging."
        )
        lines.append("")
        lines.append("| mapping | species | cluster_size | purity_last | purity_min | contamination_last | contamination_max | members |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for mapping_name, metrics in (("GNN", gnn_metrics), ("CNR", cnr_metrics)):
            entries = metrics.get("qoi_species_purity") or []
            if not isinstance(entries, Sequence) or isinstance(
                entries, (str, bytes, bytearray)
            ):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                sp = entry.get("species")
                size = entry.get("cluster_size")
                purity = entry.get("purity_last")
                contam = entry.get("contamination_last")
                purity_min = entry.get("purity_min")
                contam_max = entry.get("contamination_max")
                members = entry.get("cluster_members") or []
                if isinstance(members, Sequence) and not isinstance(
                    members, (str, bytes, bytearray)
                ):
                    members_list = [str(m) for m in members]
                else:
                    members_list = []
                members_preview = ", ".join(members_list[:10])
                if len(members_list) > 10:
                    members_preview += ", ..."
                lines.append(
                    f"| {mapping_name} | {sp} | {size} | {_fmt(purity)} | {_fmt(purity_min)} | {_fmt(contam)} | {_fmt(contam_max)} | {members_preview} |"
                )
    return "\n".join(lines) + "\n"


def evaluate_mapping(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Evaluate mapping quality and compare GNN vs CNR mappings."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")
    if np is None:
        raise ConfigError("numpy is required for mapping evaluation.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    inputs_cfg = _extract_inputs(reduction_cfg)
    params = _extract_params(reduction_cfg)

    gnn_cfg = _extract_eval_group(inputs_cfg, group="gnn")
    cnr_cfg = _extract_eval_group(inputs_cfg, group="compare")
    if not cnr_cfg:
        cnr_cfg = _extract_eval_group(inputs_cfg, group="cnr")

    qoi_cfg = params.get("qoi") or {}
    if qoi_cfg is None:
        qoi_cfg = {}
    if not isinstance(qoi_cfg, Mapping):
        raise ConfigError("params.qoi must be a mapping when provided.")
    qoi_cfg = dict(qoi_cfg)

    viz_cfg = params.get("viz") or {}
    if viz_cfg is None:
        viz_cfg = {}
    if not isinstance(viz_cfg, Mapping):
        raise ConfigError("params.viz must be a mapping when provided.")
    viz_cfg = dict(viz_cfg)

    report_cfg = params.get("report") or {}
    if report_cfg is None:
        report_cfg = {}
    if not isinstance(report_cfg, Mapping):
        raise ConfigError("params.report must be a mapping when provided.")
    report_cfg = dict(report_cfg)

    default_exp = "default"

    def _resolve_target(label: str, group_cfg: Mapping[str, Any]) -> dict[str, Any]:
        run_id = group_cfg.get("run_id") or group_cfg.get("run")
        exp = group_cfg.get("exp") or default_exp
        run_root = _coerce_optional_path(group_cfg.get("run_root"), f"{label}.run_root")
        store_root = _coerce_optional_path(
            group_cfg.get("store_root") or group_cfg.get("artifacts_root"),
            f"{label}.store_root",
        )
        mapping_id = group_cfg.get("mapping_id") or group_cfg.get("mapping")
        graph_id = group_cfg.get("graph_id") or group_cfg.get("graph")
        mapping_step = group_cfg.get("mapping_step") or params.get("mapping_step") or "mapping"
        graph_step = group_cfg.get("graph_step") or params.get("graph_step") or "graph"
        if run_root is None and run_id:
            if not isinstance(run_id, str) or not run_id.strip():
                raise ConfigError(f"{label}.run_id must be a non-empty string.")
            if not isinstance(exp, str) or not exp.strip():
                exp = default_exp
            run_root = resolve_repo_path(RUNS_ROOT / exp / str(run_id))
        if store_root is None:
            if run_root is not None:
                store_root = run_root / "artifacts"
            else:
                store_root = store.root
        target_store = ArtifactStore(store_root)
        if mapping_id is None:
            if run_root is None:
                raise ConfigError(f"{label} requires mapping_id or run_id.")
            results = _read_run_results(run_root, label)
            mapping_id = _select_result_artifact(
                results,
                label=label,
                step_name=str(mapping_step),
                store_root=store_root,
                artifact_kind="reduction",
            )
        if not isinstance(mapping_id, str) or not mapping_id.strip():
            raise ConfigError(f"{label} mapping_id not resolved.")
        mapping_payload = _load_mapping_payload(target_store, mapping_id)
        mapping_manifest = target_store.read_manifest("reduction", mapping_id)
        if graph_id is None:
            if run_root is not None:
                results = _read_run_results(run_root, label)
                graph_id = _select_graph_artifact(results, graph_step=str(graph_step))
        if graph_id is None:
            source = mapping_payload.get("source", {})
            if isinstance(source, Mapping):
                graph_id = source.get("graph_id") or source.get("graph")
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise ConfigError(f"{label} graph_id not resolved.")
        graph_payload = _load_graph_payload(target_store, graph_id)
        return {
            "run_id": run_id,
            "exp": exp,
            "run_root": str(run_root) if run_root else None,
            "store_root": str(store_root),
            "mapping_id": mapping_id,
            "graph_id": graph_id,
            "mapping_payload": mapping_payload,
            "mapping_manifest": mapping_manifest,
            "graph_payload": graph_payload,
            "store": target_store,
        }

    if not gnn_cfg:
        raise ConfigError("gnn mapping inputs are required.")
    gnn_target = _resolve_target("gnn", gnn_cfg)
    if not cnr_cfg:
        raise ConfigError("compare mapping inputs are required.")
    cnr_target = _resolve_target("compare", cnr_cfg)

    def _extract_symmetrize(payload: Mapping[str, Any], manifest: ArtifactManifest) -> bool:
        for section in (payload.get("community"), payload.get("pooling")):
            if isinstance(section, Mapping) and "symmetrize" in section:
                return _coerce_bool(section.get("symmetrize"), "symmetrize", default=True)
        cfg = manifest.config if isinstance(manifest.config, Mapping) else {}
        for root_key in ("reduction", "reduce"):
            node = cfg.get(root_key)
            if not isinstance(node, Mapping):
                continue
            for key in ("params", "pool", "community"):
                sub = node.get(key)
                if not isinstance(sub, Mapping):
                    continue
                if "symmetrize" in sub:
                    return _coerce_bool(sub.get("symmetrize"), "symmetrize", default=True)
        return True

    sym_override = params.get("symmetrize")
    if sym_override is not None and not isinstance(sym_override, bool):
        raise ConfigError("params.symmetrize must be a boolean when provided.")

    qoi_species_raw = qoi_cfg.get("species") or qoi_cfg.get("qoi_species")
    if qoi_species_raw is None:
        qoi_species = ["CO", "CO2"]
    else:
        qoi_species = _normalize_str_list(qoi_species_raw, "qoi.species")

    def _build_membership(
        mapping_payload: Mapping[str, Any],
        *,
        species_count: int,
    ) -> tuple[list[int], dict[int, list[int]]]:
        mapping_entries = mapping_payload.get("mapping")
        if not isinstance(mapping_entries, Sequence) or isinstance(
            mapping_entries, (str, bytes, bytearray)
        ):
            raise ConfigError("mapping payload missing mapping list.")
        membership = [-1 for _ in range(species_count)]
        cluster_members: dict[int, list[int]] = {}
        for entry in mapping_entries:
            if not isinstance(entry, Mapping):
                continue
            idx = entry.get("species_index")
            super_id = entry.get("superstate_id")
            if isinstance(idx, bool) or isinstance(super_id, bool):
                continue
            if not isinstance(idx, int) or not isinstance(super_id, int):
                continue
            if idx < 0 or idx >= species_count:
                continue
            membership[idx] = super_id
            cluster_members.setdefault(super_id, []).append(idx)
        if any(value < 0 for value in membership):
            raise ConfigError("mapping produced incomplete membership mapping.")
        return membership, cluster_members

    def _load_run_x_matrix(
        target_store: ArtifactStore,
        *,
        run_id: str,
    ) -> tuple[list[str], list[list[float]]]:
        target_store.read_manifest("runs", run_id)
        run_dir = target_store.artifact_dir("runs", run_id)
        payload = read_json(run_dir / "state.zarr" / "dataset.json")
        if not isinstance(payload, Mapping):
            raise ConfigError("run dataset.json must be a mapping.")
        coords = payload.get("coords") or {}
        if not isinstance(coords, Mapping):
            raise ConfigError("run dataset coords must be a mapping.")
        species_payload = coords.get("species")
        if not isinstance(species_payload, Mapping):
            raise ConfigError("run dataset missing coords.species.")
        species_data = species_payload.get("data")
        if not isinstance(species_data, Sequence) or isinstance(
            species_data, (str, bytes, bytearray)
        ):
            raise ConfigError("coords.species.data must be a list.")
        species_names = [str(name) for name in species_data]
        data_vars = payload.get("data_vars") or {}
        if not isinstance(data_vars, Mapping):
            raise ConfigError("run dataset data_vars must be a mapping.")
        x_var = data_vars.get("X")
        if not isinstance(x_var, Mapping):
            raise ConfigError("run dataset missing data_vars.X.")
        dims = x_var.get("dims")
        data = x_var.get("data")
        if isinstance(dims, str) or not isinstance(dims, Sequence):
            raise ConfigError("data_vars.X.dims must be a sequence.")
        dims_list = list(dims)
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
            raise ConfigError("data_vars.X.data must be a matrix.")
        if dims_list == ["time", "species"]:
            matrix = data
        elif dims_list == ["species", "time"]:
            # transpose into [time][species]
            matrix = [list(row) for row in zip(*data)]
        else:
            raise ConfigError("data_vars.X.dims must be [time, species] or [species, time].")
        if not matrix:
            raise ConfigError("data_vars.X.data must be non-empty.")
        x_matrix: list[list[float]] = []
        for row in matrix:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise ConfigError("data_vars.X rows must be sequences.")
            if len(row) != len(species_names):
                raise ConfigError("data_vars.X species axis length mismatch.")
            coerced: list[float] = []
            for entry in row:
                try:
                    coerced.append(float(entry))
                except (TypeError, ValueError) as exc:
                    raise ConfigError("data_vars.X entries must be numeric.") from exc
            x_matrix.append(coerced)
        return species_names, x_matrix

    def _qoi_species_purity(target: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not qoi_species:
            return []
        mapping_payload = target["mapping_payload"]
        graph_payload = target["graph_payload"]
        species_section = graph_payload.get("species") or {}
        species_order = species_section.get("order")
        if not isinstance(species_order, Sequence) or isinstance(
            species_order, (str, bytes, bytearray)
        ):
            return []
        species_names = [str(name) for name in species_order]
        membership, cluster_members = _build_membership(
            mapping_payload, species_count=len(species_names)
        )
        # Resolve the underlying run id from the temporal graph source.
        source = graph_payload.get("source") or {}
        run_ids = []
        if isinstance(source, Mapping):
            raw = source.get("run_ids") or source.get("run_id")
            if isinstance(raw, str):
                run_ids = [raw]
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
                run_ids = [str(rid) for rid in raw]
        if not run_ids:
            return []
        target_store: ArtifactStore = target["store"]
        run_species, x_matrix = _load_run_x_matrix(target_store, run_id=str(run_ids[0]))

        # Align by name if ordering differs.
        index_by_species = {name: idx for idx, name in enumerate(run_species)}
        x_aligned: list[list[float]] = []
        if run_species == species_names:
            x_aligned = x_matrix
        else:
            for row in x_matrix:
                aligned_row: list[float] = []
                for name in species_names:
                    idx = index_by_species.get(name)
                    aligned_row.append(float(row[idx]) if idx is not None else 0.0)
                x_aligned.append(aligned_row)
        if not x_aligned:
            return []
        x_last_aligned = x_aligned[-1]

        rows: list[dict[str, Any]] = []
        species_set = set(species_names)
        for sp in qoi_species:
            if sp not in species_set:
                rows.append(
                    {
                        "species": sp,
                        "status": "missing_species",
                        "cluster_id": None,
                        "cluster_size": None,
                        "purity_last": None,
                        "contamination_last": None,
                        "purity_min": None,
                        "contamination_max": None,
                        "cluster_members": [],
                    }
                )
                continue
            s_idx = species_names.index(sp)
            cluster_id = membership[s_idx]
            members = cluster_members.get(cluster_id, [])
            cluster_total = sum(x_last_aligned[idx] for idx in members) if members else 0.0
            x_sp = x_last_aligned[s_idx]
            purity = (x_sp / cluster_total) if cluster_total > 0.0 else None
            contam = (1.0 - purity) if purity is not None else None

            # Min purity over time gives a pessimistic view of "indistinguishability"
            # if interpreting merged-cluster totals as the species value.
            purity_over_time: list[float] = []
            for row in x_aligned:
                if not members:
                    continue
                total_t = sum(row[idx] for idx in members)
                if total_t <= 0.0:
                    continue
                purity_over_time.append(row[s_idx] / total_t)
            purity_min = min(purity_over_time) if purity_over_time else None
            contam_max = (1.0 - purity_min) if purity_min is not None else None

            member_names = [species_names[idx] for idx in sorted(members)]
            rows.append(
                {
                    "species": sp,
                    "status": "ok",
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "purity_last": float(purity) if purity is not None else None,
                    "contamination_last": float(contam) if contam is not None else None,
                    "purity_min": float(purity_min) if purity_min is not None else None,
                    "contamination_max": float(contam_max) if contam_max is not None else None,
                    "cluster_members": member_names,
                }
            )
        return rows

    def _evaluate_target(target: Mapping[str, Any]) -> dict[str, Any]:
        mapping_payload = target["mapping_payload"]
        graph_payload = target["graph_payload"]
        mapping_manifest = target["mapping_manifest"]
        symmetrize = (
            sym_override if sym_override is not None else _extract_symmetrize(mapping_payload, mapping_manifest)
        )
        species_section = graph_payload.get("species") or {}
        species_order = species_section.get("order")
        if not isinstance(species_order, Sequence) or isinstance(
            species_order, (str, bytes, bytearray)
        ):
            raise ConfigError("temporal graph species.order must be a list.")
        species_count = len(species_order)
        mapping_entries = mapping_payload.get("mapping")
        if not isinstance(mapping_entries, Sequence) or isinstance(
            mapping_entries, (str, bytes, bytearray)
        ):
            raise ConfigError("mapping payload missing mapping list.")
        membership = [-1 for _ in range(species_count)]
        cluster_members: dict[int, list[int]] = {}
        for entry in mapping_entries:
            if not isinstance(entry, Mapping):
                continue
            idx = entry.get("species_index")
            super_id = entry.get("superstate_id")
            if isinstance(idx, bool) or isinstance(super_id, bool):
                continue
            if not isinstance(idx, int) or not isinstance(super_id, int):
                continue
            if idx < 0 or idx >= species_count:
                continue
            membership[idx] = super_id
            cluster_members.setdefault(super_id, []).append(idx)
        if any(value < 0 for value in membership):
            raise ConfigError("mapping produced incomplete membership mapping.")

        clusters_payload = mapping_payload.get("clusters")
        if isinstance(clusters_payload, Sequence) and not isinstance(
            clusters_payload, (str, bytes, bytearray)
        ):
            cluster_sizes = [int(entry.get("size", 0)) for entry in clusters_payload if isinstance(entry, Mapping)]
        else:
            cluster_sizes = [len(members) for _, members in sorted(cluster_members.items())]
        size_stats = _cluster_size_stats(cluster_sizes)

        species_graph = graph_payload.get("species_graph") or {}
        layers_meta = species_graph.get("layers")
        if not isinstance(layers_meta, Sequence) or isinstance(
            layers_meta, (str, bytes, bytearray)
        ):
            raise ConfigError("temporal graph layers must be a list.")
        graph_dir = target["store"].artifact_dir("graphs", target["graph_id"])
        window_metrics: list[dict[str, Any]] = []
        window_dense: list[Any] = []
        window_meta: list[dict[str, Any]] = []
        for entry in layers_meta:
            if not isinstance(entry, Mapping):
                continue
            path_value = entry.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            index_value = entry.get("index")
            index_value = int(index_value) if index_value is not None else len(window_metrics)
            layer_path = graph_dir / path_value
            matrix = _load_csr_matrix(layer_path)
            dense = _dense_from_sparse(matrix)
            total, within, coverage = _flux_coverage(
                dense, membership, symmetrize=symmetrize
            )
            window_metrics.append(
                {
                    "index": index_value,
                    "path": path_value,
                    "window": entry.get("window"),
                    "flux_total": total,
                    "within_cluster": within,
                    "coverage": coverage,
                }
            )
            window_dense.append(dense)
            window_meta.append({"index": index_value, "window": entry.get("window")})

        total_flux = sum(entry["flux_total"] for entry in window_metrics)
        within_flux = sum(entry["within_cluster"] for entry in window_metrics)
        overall_coverage = within_flux / total_flux if total_flux > 0.0 else 0.0

        qoi_selector = qoi_cfg.get("selector") or DEFAULT_EVAL_QOI_SELECTOR
        qoi_window_ids = _select_qoi_windows(qoi_cfg, window_meta=window_meta)
        radius = qoi_cfg.get("window_radius")
        if radius is None:
            radius = DEFAULT_EVAL_QOI_RADIUS
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ConfigError("qoi.window_radius must be a non-negative integer.")
        if qoi_window_ids:
            all_ids = [entry["index"] for entry in window_meta]
            qoi_window_ids = _expand_window_ids(
                qoi_window_ids, radius=radius, all_ids=all_ids
            )
        qoi_entries = [
            entry for entry in window_metrics if entry["index"] in set(qoi_window_ids)
        ]
        qoi_total = sum(entry["flux_total"] for entry in qoi_entries)
        qoi_within = sum(entry["within_cluster"] for entry in qoi_entries)
        qoi_coverage = qoi_within / qoi_total if qoi_total > 0.0 else 0.0
        qoi_retention = (
            qoi_coverage / overall_coverage if overall_coverage > 0.0 else 0.0
        )

        cluster_indices = [
            sorted(members)
            for _, members in sorted(cluster_members.items(), key=lambda item: item[0])
        ]
        cluster_vectors = _cluster_flux_vectors(
            window_dense, cluster_indices=cluster_indices, symmetrize=symmetrize
        )
        weights = [entry["flux_total"] for entry in window_metrics]
        norm_vectors: list[list[float]] = []
        weighted_vectors: list[list[float]] = []
        norm_weights: list[float] = []
        total_weight = 0.0
        for vector, weight in zip(cluster_vectors, weights):
            total = sum(vector)
            if total <= 0.0:
                continue
            normalized = [value / total for value in vector]
            norm_vectors.append(normalized)
            weighted_vectors.append([value * weight for value in normalized])
            norm_weights.append(weight)
            total_weight += weight
        if weighted_vectors and total_weight > 0.0:
            mean_vector = [
                sum(values) / total_weight for values in zip(*weighted_vectors)
            ]
            similarities = [
                _cosine_similarity(vec, mean_vector) for vec in norm_vectors
            ]
            stability = float(
                sum(sim * w for sim, w in zip(similarities, norm_weights))
                / total_weight
            )
        else:
            stability = 0.0

        return {
            "mapping_id": target["mapping_id"],
            "graph_id": target["graph_id"],
            "symmetrize": symmetrize,
            "cluster_sizes": size_stats,
            "cluster_size_list": cluster_sizes,
            "flux": {
                "total": total_flux,
                "within_cluster": within_flux,
                "coverage": overall_coverage,
            },
            "coverage_by_window": window_metrics,
            "qoi": {
                "selector": qoi_selector,
                "window_ids": qoi_window_ids,
                "coverage": qoi_coverage,
                "retention": qoi_retention,
            },
            "stability": {
                "cluster_similarity": stability,
            },
        }

    gnn_metrics = _evaluate_target(gnn_target)
    cnr_metrics = _evaluate_target(cnr_target)
    # Optional proxy metrics for state-merge interpretation.
    try:
        gnn_metrics["qoi_species_purity"] = _qoi_species_purity(gnn_target)
    except Exception as exc:
        gnn_metrics["qoi_species_purity"] = []
        gnn_metrics["qoi_species_purity_error"] = str(exc)
    try:
        cnr_metrics["qoi_species_purity"] = _qoi_species_purity(cnr_target)
    except Exception as exc:
        cnr_metrics["qoi_species_purity"] = []
        cnr_metrics["qoi_species_purity_error"] = str(exc)

    comparison: dict[str, Any] = {}
    for name, path in (
        ("flux_coverage", ("flux", "coverage")),
        ("qoi_coverage", ("qoi", "coverage")),
        ("qoi_retention", ("qoi", "retention")),
        ("cluster_stability", ("stability", "cluster_similarity")),
    ):
        gnn_val = gnn_metrics
        cnr_val = cnr_metrics
        for key in path:
            gnn_val = gnn_val.get(key, {}) if isinstance(gnn_val, Mapping) else {}
            cnr_val = cnr_val.get(key, {}) if isinstance(cnr_val, Mapping) else {}
        if isinstance(gnn_val, (int, float)) and isinstance(cnr_val, (int, float)):
            comparison[name] = float(gnn_val) - float(cnr_val)

    max_bins = viz_cfg.get("max_bins", DEFAULT_EVAL_MAX_BINS)
    if isinstance(max_bins, bool) or not isinstance(max_bins, int) or max_bins <= 0:
        raise ConfigError("viz.max_bins must be a positive integer.")
    max_size = max(
        max(gnn_metrics["cluster_size_list"], default=0),
        max(cnr_metrics["cluster_size_list"], default=0),
    )
    bins, labels, overflow = _build_histogram_bins(max_size, max_bins)
    gnn_counts = _hist_counts(
        gnn_metrics["cluster_size_list"], bins=bins, overflow=overflow
    )
    cnr_counts = _hist_counts(
        cnr_metrics["cluster_size_list"], bins=bins, overflow=overflow
    )
    svg = _render_cluster_size_svg(
        labels=labels,
        gnn_counts=gnn_counts,
        cnr_counts=cnr_counts,
        title="Cluster Size Distribution",
    )

    report_format = report_cfg.get("format", DEFAULT_EVAL_REPORT_FORMAT)
    if not isinstance(report_format, str) or not report_format.strip():
        raise ConfigError("report.format must be a non-empty string.")
    report_format = report_format.strip().lower()
    if report_format not in ("markdown", "html"):
        raise ConfigError("report.format must be 'markdown' or 'html'.")
    report_filename = "report.md" if report_format == "markdown" else "report.html"
    viz_filename = "viz/cluster_size_comparison.svg"

    qoi_info = {
        "selector": qoi_cfg.get("selector") or DEFAULT_EVAL_QOI_SELECTOR,
        "count": len(gnn_metrics.get("qoi", {}).get("window_ids", [])),
    }
    report_text = _format_mapping_eval_report(
        report_format=report_format,
        gnn_metrics=gnn_metrics,
        cnr_metrics=cnr_metrics,
        comparison=comparison,
        qoi_info=qoi_info,
        report_cfg=report_cfg,
        viz_filename=viz_filename,
    )

    metrics_payload = {
        "schema_version": 1,
        "kind": "mapping_eval",
        "gnn": gnn_metrics,
        "compare": cnr_metrics,
        "comparison": comparison,
    }

    inputs_payload = {
        "gnn": {
            "run_id": gnn_target.get("run_id"),
            "exp": gnn_target.get("exp"),
            "mapping_id": gnn_target.get("mapping_id"),
            "graph_id": gnn_target.get("graph_id"),
        },
        "compare": {
            "run_id": cnr_target.get("run_id"),
            "exp": cnr_target.get("exp"),
            "mapping_id": cnr_target.get("mapping_id"),
            "graph_id": cnr_target.get("graph_id"),
        },
        "qoi": {
            "selector": qoi_cfg.get("selector") or DEFAULT_EVAL_QOI_SELECTOR,
            "window_radius": qoi_cfg.get("window_radius", DEFAULT_EVAL_QOI_RADIUS),
        },
    }

    artifact_id = reduction_cfg.get("artifact_id") or reduction_cfg.get("id")
    if artifact_id is None:
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
    artifact_id = _require_nonempty_str(artifact_id, "artifact_id")

    manifest = build_manifest(
        kind="validation",
        artifact_id=artifact_id,
        parents=[
            gnn_target["mapping_id"],
            cnr_target["mapping_id"],
            gnn_target["graph_id"],
            cnr_target["graph_id"],
        ],
        inputs=inputs_payload,
        config=manifest_cfg,
    )

    def _writer(base_dir: Path) -> None:
        write_json_atomic(base_dir / "metrics.json", metrics_payload)
        (base_dir / report_filename).write_text(
            report_text,
            encoding="utf-8",
        )
        viz_dir = base_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "cluster_size_comparison.svg").write_text(
            svg,
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _lookup_context_value(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_placeholders(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("${") and stripped.endswith("}"):
            path = stripped[2:-1].strip()
            resolved = _lookup_context_value(context, path) if path else None
            if resolved is not None:
                return resolved
        return value
    if isinstance(value, Mapping):
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolve_placeholders(item, context) for item in value]
    return value


def _infer_step_id(steps: Sequence[Mapping[str, Any]], task_name: str) -> Optional[str]:
    for step in steps:
        if step.get("task") == task_name:
            step_id = step.get("id")
            if isinstance(step_id, str) and step_id.strip():
                return step_id
    return None


def _list_reducer_tasks(registry: Optional[Registry]) -> list[str]:
    try:
        if registry is None:
            tasks = registry_module.list("task")
        else:
            tasks = registry.list("task")
    except KeyError:
        tasks = []
    exclude = {
        "reduction.apply",
        "reduction.validate",
        "reduction.evaluate_mapping",
        "reduction.benchmark_compare",
        "reduction.node_lumping",
        "reduction.reaction_lumping",
    }
    return sorted(
        {
            task
            for task in tasks
            if isinstance(task, str)
            and task.startswith("reduction.")
            and task not in exclude
        }
    )


def _infer_step_id_for_tasks(
    steps: Sequence[Mapping[str, Any]],
    task_names: Sequence[str],
) -> Optional[str]:
    if not task_names:
        return None
    task_set = set(task_names)
    for step in steps:
        if step.get("task") in task_set:
            step_id = step.get("id")
            if isinstance(step_id, str) and step_id.strip():
                return step_id
    return None


def _apply_step_overrides(
    steps: Sequence[Mapping[str, Any]],
    overrides: Any,
) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Sequence) or isinstance(
        overrides, (str, bytes, bytearray)
    ):
        raise ConfigError("step_overrides must be a list of mappings.")
    for entry in overrides:
        if not isinstance(entry, Mapping):
            raise ConfigError("step_overrides entries must be mappings.")
        step_id = entry.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ConfigError("step_overrides.id must be a non-empty string.")
        updates = entry.get("updates") or entry.get("set") or entry.get("patch")
        if not isinstance(updates, Mapping):
            raise ConfigError("step_overrides.updates must be a mapping.")
        target: Optional[dict[str, Any]] = None
        for step in steps:
            if step.get("id") == step_id:
                target = step  # type: ignore[assignment]
                break
        if target is None:
            raise ConfigError(f"step_overrides.id {step_id!r} not found.")
        target.update(_deep_merge_dicts(target, dict(updates)))


def _inject_split_seed(steps: Sequence[Mapping[str, Any]], seed: int) -> None:
    for step in steps:
        task_name = step.get("task")
        if task_name == "gnn_dataset.temporal_graph_pyg":
            params = step.get("params")
            if params is None:
                params = {}
                step["params"] = params
            if not isinstance(params, Mapping):
                raise ConfigError("gnn_dataset params must be a mapping.")
            params_map = dict(params)
            split_cfg = params_map.get("split") or params_map.get("splits") or {}
            if split_cfg is None:
                split_cfg = {}
            if not isinstance(split_cfg, Mapping):
                raise ConfigError("gnn_dataset split config must be a mapping.")
            split_map = dict(split_cfg)
            existing_seed = split_map.get("seed")
            if not isinstance(existing_seed, int) or isinstance(
                existing_seed, bool
            ):
                split_map["seed"] = seed
            params_map["split"] = split_map
            step["params"] = params_map
            continue
        if task_name != "reduction.gnn_pool_temporal":
            continue
        params = step.get("params")
        if params is None:
            params = {}
            step["params"] = params
        if not isinstance(params, Mapping):
            raise ConfigError("reduction.gnn_pool_temporal params must be a mapping.")
        params_map = dict(params)
        window_split = params_map.get("window_split") or {}
        if window_split is None:
            window_split = {}
        if not isinstance(window_split, Mapping):
            raise ConfigError("window_split must be a mapping.")
        window_split_map = dict(window_split)
        existing_seed = window_split_map.get("seed")
        if not isinstance(existing_seed, int) or isinstance(existing_seed, bool):
            window_split_map["seed"] = seed
        params_map["window_split"] = window_split_map
        step["params"] = params_map


def _read_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def _count_reactions(payload: Mapping[str, Any]) -> Optional[int]:
    reactions = payload.get("reactions")
    if isinstance(reactions, Sequence) and not isinstance(
        reactions, (str, bytes, bytearray)
    ):
        return len(reactions)
    return None


def _summarize_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    patch_id: Optional[str],
    metric: str,
) -> dict[str, Any]:
    values: list[float] = []
    passed_count = 0
    total = 0
    for row in rows:
        if patch_id is not None and row.get("patch_id") != patch_id:
            continue
        status = row.get("status")
        if status == "skipped":
            continue
        value = None
        if metric == "rel":
            value = row.get("rel_diff")
        else:
            value = row.get("abs_diff")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        values.append(numeric)
        total += 1
        if row.get("passed") is True:
            passed_count += 1
    max_value = max(values) if values else None
    mean_value = statistics.mean(values) if values else None
    pass_rate = passed_count / total if total > 0 else 0.0
    return {
        "count": total,
        "passed": passed_count,
        "pass_rate": pass_rate,
        "max_metric": max_value,
        "mean_metric": mean_value,
    }


def benchmark_compare(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run multiple reducers on a shared benchmark and emit a comparison report."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, reduction_cfg = _extract_reduction_cfg(resolved_cfg)
    params = _extract_params(reduction_cfg)

    reducers_value = params.get("reducers") or reduction_cfg.get("reducers")
    if reducers_value is None:
        raise ConfigError("reducers must be provided.")

    reducers: list[dict[str, Any]] = []
    if isinstance(reducers_value, Mapping):
        for key, value in reducers_value.items():
            entry: dict[str, Any] = {"id": key}
            if isinstance(value, Mapping):
                entry.update(dict(value))
            else:
                entry["pipeline"] = value
            reducers.append(entry)
    elif isinstance(reducers_value, Sequence) and not isinstance(
        reducers_value, (str, bytes, bytearray)
    ):
        for entry in reducers_value:
            if not isinstance(entry, Mapping):
                raise ConfigError("reducers entries must be mappings.")
            reducers.append(dict(entry))
    else:
        raise ConfigError("reducers must be a list or mapping.")

    sim_cfg = reduction_cfg.get("sim") or params.get("sim")
    if sim_cfg is not None and not isinstance(sim_cfg, Mapping):
        raise ConfigError("sim must be a mapping.")
    base_sim = dict(sim_cfg) if isinstance(sim_cfg, Mapping) else None

    use_surrogate = bool(
        reduction_cfg.get("use_surrogate")
        or params.get("use_surrogate")
        or resolved_cfg.get("use_surrogate")
    )

    split_seed = params.get("split_seed") or reduction_cfg.get("split_seed")
    split_seed_value: Optional[int] = None
    if split_seed is not None:
        try:
            split_seed_value = int(split_seed)
        except (TypeError, ValueError) as exc:
            raise ConfigError("split_seed must be an integer.") from exc

    validation_cfg = params.get("validation") or reduction_cfg.get("validation") or {}
    if validation_cfg is None:
        validation_cfg = {}
    if not isinstance(validation_cfg, Mapping):
        raise ConfigError("validation must be a mapping.")
    base_validation_cfg = dict(validation_cfg)

    mapping_eval_cfg = params.get("mapping_eval") or reduction_cfg.get("mapping_eval")
    if mapping_eval_cfg is not None and not isinstance(mapping_eval_cfg, Mapping):
        raise ConfigError("mapping_eval must be a mapping.")
    mapping_eval_cfg = dict(mapping_eval_cfg) if isinstance(mapping_eval_cfg, Mapping) else None

    report_cfg = params.get("report") or reduction_cfg.get("report") or {}
    if report_cfg is None:
        report_cfg = {}
    if not isinstance(report_cfg, Mapping):
        raise ConfigError("report must be a mapping.")
    report_cfg = dict(report_cfg)

    benchmark_cfg = params.get("benchmark") or reduction_cfg.get("benchmark") or {}
    if benchmark_cfg is None:
        benchmark_cfg = {}
    if not isinstance(benchmark_cfg, Mapping):
        raise ConfigError("benchmark must be a mapping.")
    benchmark_cfg = dict(benchmark_cfg)

    logger = logging.getLogger("rxn_platform.reduction")
    runner = PipelineRunner(store=store, registry=registry, logger=logger)

    run_root = store.root.parent if store.root.name == "artifacts" else store.root
    report_dir = run_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    reducer_results: list[dict[str, Any]] = []
    reducer_index: dict[str, dict[str, Any]] = {}

    for entry in reducers:
        enabled = entry.get("enabled", True)
        if enabled is False:
            continue
        reducer_id = entry.get("id") or entry.get("name")
        reducer_id = _require_nonempty_str(reducer_id, "reducers.id")
        label = entry.get("label") or reducer_id
        if not isinstance(label, str):
            raise ConfigError("reducers.label must be a string when provided.")
        pipeline_value = (
            entry.get("pipeline")
            or entry.get("pipeline_cfg")
            or entry.get("pipeline_config")
        )
        if pipeline_value is None:
            raise ConfigError(f"reducers[{reducer_id}] pipeline is required.")

        pipeline_cfg = _normalize_pipeline_cfg(pipeline_value, runner)
        steps = pipeline_cfg.get("steps", [])
        if not isinstance(steps, Sequence):
            raise ConfigError("pipeline.steps must be a list.")

        sim_override = entry.get("sim") or entry.get("sim_overrides")
        if sim_override is not None and not isinstance(sim_override, Mapping):
            raise ConfigError("reducers.sim_overrides must be a mapping.")
        if base_sim is not None:
            sim_used = _deep_merge_dicts(base_sim, dict(sim_override or {}))
        else:
            sim_used = dict(sim_override) if isinstance(sim_override, Mapping) else None

        context = {"sim": sim_used, "use_surrogate": use_surrogate}
        pipeline_cfg = _resolve_placeholders(pipeline_cfg, context)
        steps = pipeline_cfg.get("steps", [])
        if not isinstance(steps, Sequence):
            raise ConfigError("pipeline.steps must be a list.")

        if sim_used is not None:
            for step in steps:
                if step.get("task", "").startswith("sim."):
                    step["sim"] = copy.deepcopy(sim_used)
        if split_seed_value is not None:
            _inject_split_seed(steps, split_seed_value)
        _apply_step_overrides(steps, entry.get("step_overrides"))

        pipeline_cfg["steps"] = list(steps)

        started = time.perf_counter()
        results = runner.run(pipeline_cfg)
        elapsed = time.perf_counter() - started

        reduction_step = (
            entry.get("reduction_step")
            or entry.get("result_step")
            or entry.get("mapping_step")
        )
        if reduction_step is None:
            reduction_step = _infer_step_id_for_tasks(
                steps,
                _list_reducer_tasks(registry),
            )
        if reduction_step is None:
            raise ConfigError(f"reducers[{reducer_id}] reduction_step not resolved.")

        reduction_id = results.get(reduction_step)
        if not isinstance(reduction_id, str) or not reduction_id.strip():
            raise ConfigError(f"reducers[{reducer_id}] reduction result missing.")

        validation_step = entry.get("validation_step")
        if validation_step is None:
            validation_step = _infer_step_id(steps, "reduction.validate")
        validation_id = results.get(validation_step) if validation_step else None

        graph_step = entry.get("graph_step") or _infer_step_id(steps, "graphs.temporal_flux")
        graph_id = results.get(graph_step) if graph_step else None
        run_step = entry.get("run_step") or _infer_step_id(steps, "sim.run")
        run_id = results.get(run_step) if run_step else None

        reduction_dir = store.artifact_dir("reduction", reduction_id)
        reduction_metrics = _read_metrics_json(reduction_dir / "metrics.json")

        reducer_type = entry.get("type") or "reduction"
        if reducer_type not in {"reduction", "mapping"}:
            raise ConfigError("reducers.type must be 'reduction' or 'mapping'.")

        validation_summary: Optional[dict[str, Any]] = None
        validation_metric = None
        validation_elapsed = 0.0

        do_validate = entry.get("validate")
        if do_validate is None:
            do_validate = reducer_type == "reduction"

        if do_validate and (validation_id is None or entry.get("force_validation")):
            mechanism_path = (
                base_validation_cfg.get("mechanism")
                or (sim_used or {}).get("mechanism")
                or reduction_cfg.get("mechanism")
            )
            if mechanism_path is None:
                logger.warning(
                    "Validation skipped for %s: mechanism not provided.",
                    reducer_id,
                )
            else:
                validator_cfg = dict(base_validation_cfg)
                entry_validation = entry.get("validation")
                if entry_validation is not None:
                    if not isinstance(entry_validation, Mapping):
                        raise ConfigError("reducers.validation must be a mapping.")
                    validator_cfg = _deep_merge_dicts(validator_cfg, dict(entry_validation))
                if "sim" not in validator_cfg and sim_used is not None:
                    validator_cfg["sim"] = copy.deepcopy(sim_used)
                validation_payload = {
                    "inputs": {"patches": [reduction_id]},
                    "mechanism": mechanism_path,
                    "validation": validator_cfg,
                }
                validation_started = time.perf_counter()
                validation_result = validate_reduction(
                    validation_payload,
                    store=store,
                    registry=registry,
                )
                validation_elapsed = time.perf_counter() - validation_started
                validation_id = validation_result.manifest.id
                elapsed += validation_elapsed

        if isinstance(validation_id, str) and validation_id.strip():
            validation_dir = store.artifact_dir("validation", validation_id)
            table_path = validation_dir / "metrics.parquet"
            if not table_path.exists():
                table_path = validation_dir / "metrics.json"
            try:
                rows = _read_table_rows(table_path)
                metric_name = (
                    base_validation_cfg.get("metric")
                    or base_validation_cfg.get("diff_metric")
                    or "abs"
                )
                if not isinstance(metric_name, str):
                    metric_name = "abs"
                validation_metric = metric_name.lower()
                validation_summary = _summarize_validation_rows(
                    rows,
                    patch_id=reduction_id,
                    metric=validation_metric,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to summarize validation metrics for %s: %s",
                    reducer_id,
                    exc,
                )

        size_value: Optional[int] = None
        size_label = None
        mechanism_path = reduction_dir / MECHANISM_FILENAME
        if mechanism_path.exists():
            mech_payload = read_yaml_payload(mechanism_path)
            if isinstance(mech_payload, Mapping):
                size_value = _count_reactions(mech_payload)
                size_label = "reactions"
        if size_value is None and "cluster_sizes" in reduction_metrics:
            cluster_stats = reduction_metrics.get("cluster_sizes", {})
            if isinstance(cluster_stats, Mapping) and "count" in cluster_stats:
                size_value = int(cluster_stats.get("count", 0))
                size_label = "clusters"

        reducer_payload = {
            "id": reducer_id,
            "label": label,
            "type": reducer_type,
            "pipeline": pipeline_value,
            "elapsed_seconds": elapsed,
            "pipeline_seconds": elapsed - validation_elapsed,
            "validation_seconds": validation_elapsed,
            "reduction_step": reduction_step,
            "reduction_id": reduction_id,
            "validation_step": validation_step,
            "validation_id": validation_id,
            "run_id": run_id,
            "graph_id": graph_id,
            "metrics": reduction_metrics,
            "validation_summary": validation_summary,
            "validation_metric": validation_metric,
            "size_value": size_value,
            "size_label": size_label,
        }
        reducer_results.append(reducer_payload)
        reducer_index[reducer_id] = reducer_payload

    mapping_eval_id: Optional[str] = None
    mapping_eval_metrics: Optional[dict[str, Any]] = None
    if mapping_eval_cfg is not None:
        gnn_key = mapping_eval_cfg.get("gnn") or mapping_eval_cfg.get("gnn_id")
        compare_key = mapping_eval_cfg.get("compare") or mapping_eval_cfg.get("cnr")
        if isinstance(gnn_key, str) and isinstance(compare_key, str):
            gnn_entry = reducer_index.get(gnn_key)
            cnr_entry = reducer_index.get(compare_key)
            if gnn_entry and cnr_entry:
                eval_cfg: dict[str, Any] = {
                    "inputs": {
                        "gnn": {"mapping_id": gnn_entry["reduction_id"]},
                        "compare": {"mapping_id": cnr_entry["reduction_id"]},
                    },
                    "params": {},
                }
                for key in ("qoi", "viz", "report", "symmetrize"):
                    if key in mapping_eval_cfg:
                        eval_cfg["params"][key] = mapping_eval_cfg[key]
                eval_result = evaluate_mapping(
                    eval_cfg,
                    store=store,
                    registry=registry,
                )
                mapping_eval_id = eval_result.manifest.id
                metrics_path = store.artifact_dir("validation", mapping_eval_id) / "metrics.json"
                mapping_eval_metrics = _read_metrics_json(metrics_path)

    title = report_cfg.get("title") or "Benchmark Comparison"
    report_filename = report_cfg.get("filename") or "comparison.md"
    if not isinstance(title, str) or not title.strip():
        raise ConfigError("report.title must be a non-empty string.")
    if not isinstance(report_filename, str) or not report_filename.strip():
        raise ConfigError("report.filename must be a non-empty string.")
    report_path = Path(report_filename)
    if report_path.is_absolute() or ".." in report_path.parts or len(report_path.parts) > 1:
        raise ConfigError("report.filename must be a single relative filename.")

    metric_label = "abs"
    for entry in reducer_results:
        metric_val = entry.get("validation_metric")
        if isinstance(metric_val, str):
            metric_label = metric_val
            break

    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            num = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if math.isnan(num) or math.isinf(num):
            return "n/a"
        return f"{num:.3g}"

    reduction_rows: list[str] = []
    mapping_rows: list[str] = []

    reduction_header = (
        f"| Method | QoI max ({metric_label}) | QoI mean ({metric_label}) | Pass rate | "
        "Size | Elapsed (s) | Reduction ID | Validation ID |"
    )
    reduction_rows.append(reduction_header)
    reduction_rows.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    mapping_header = (
        "| Method | Flux coverage | QoI coverage | QoI retention | "
        "Cluster stability | Clusters | Elapsed (s) | Mapping ID |"
    )
    mapping_rows.append(mapping_header)
    mapping_rows.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for entry in reducer_results:
        if entry["type"] == "mapping":
            flux_cov = None
            qoi_cov = None
            qoi_ret = None
            stability = None
            cluster_count = None
            if mapping_eval_metrics is not None:
                if mapping_eval_metrics.get("gnn", {}).get("mapping_id") == entry["reduction_id"]:
                    metrics_block = mapping_eval_metrics.get("gnn", {})
                elif mapping_eval_metrics.get("compare", {}).get("mapping_id") == entry["reduction_id"]:
                    metrics_block = mapping_eval_metrics.get("compare", {})
                else:
                    metrics_block = {}
                if isinstance(metrics_block, Mapping):
                    flux_cov = metrics_block.get("flux", {}).get("coverage")
                    qoi_cov = metrics_block.get("qoi", {}).get("coverage")
                    qoi_ret = metrics_block.get("qoi", {}).get("retention")
                    stability = metrics_block.get("stability", {}).get("cluster_similarity")
                    cluster_stats = metrics_block.get("cluster_sizes", {})
                    if isinstance(cluster_stats, Mapping):
                        cluster_count = cluster_stats.get("count")
            else:
                metrics_block = entry.get("metrics", {})
                if isinstance(metrics_block, Mapping):
                    flux_cov = (
                        metrics_block.get("flux", {})
                        .get("coverage")
                        if isinstance(metrics_block.get("flux"), Mapping)
                        else None
                    )
                    cluster_stats = metrics_block.get("cluster_sizes", {})
                    if isinstance(cluster_stats, Mapping):
                        cluster_count = cluster_stats.get("count")

            mapping_rows.append(
                "| "
                + " | ".join(
                    [
                        str(entry["label"]),
                        _fmt(flux_cov),
                        _fmt(qoi_cov),
                        _fmt(qoi_ret),
                        _fmt(stability),
                        _fmt(cluster_count),
                        _fmt(entry["elapsed_seconds"]),
                        str(entry["reduction_id"]),
                    ]
                )
                + " |"
            )
            continue

        summary = entry.get("validation_summary") or {}
        size_value = entry.get("size_value")
        size_label = entry.get("size_label") or "n/a"
        size_text = _fmt(size_value)
        if size_value is not None and size_label != "n/a":
            size_text = f"{size_text} {size_label}"
        reduction_rows.append(
            "| "
            + " | ".join(
                [
                    str(entry["label"]),
                    _fmt(summary.get("max_metric")),
                    _fmt(summary.get("mean_metric")),
                    _fmt(summary.get("pass_rate")),
                    size_text,
                    _fmt(entry["elapsed_seconds"]),
                    str(entry["reduction_id"]),
                    str(entry.get("validation_id") or "n/a"),
                ]
            )
            + " |"
        )

    svg_sections: list[tuple[str, str]] = []
    if report_cfg.get("include_svg", True):
        labels = [entry["label"] for entry in reducer_results if entry["type"] == "reduction"]
        qoi_values = []
        size_values = []
        for entry in reducer_results:
            if entry["type"] != "reduction":
                continue
            summary = entry.get("validation_summary") or {}
            qoi_values.append(summary.get("max_metric") or 0.0)
            size_values.append(entry.get("size_value") or 0.0)

        def _render_bar_svg(title: str, values: list[float]) -> str:
            width = int(report_cfg.get("svg_width", 640))
            bar_height = int(report_cfg.get("svg_bar_height", 18))
            gap = 6
            margin_left = 140
            margin_top = 30
            max_value = max(values) if values else 0.0
            if max_value <= 0.0:
                max_value = 1.0
            height = margin_top + len(values) * (bar_height + gap) + 30
            bar_max = width - margin_left - 20
            rows = [f"<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">"]
            rows.append(f"<text x=\"10\" y=\"20\" font-size=\"14\">{title}</text>")
            for idx, (label, value) in enumerate(zip(labels, values)):
                y = margin_top + idx * (bar_height + gap)
                bar_len = int(bar_max * (float(value) / max_value))
                rows.append(
                    f"<text x=\"10\" y=\"{y + bar_height - 4}\" font-size=\"12\">{label}</text>"
                )
                rows.append(
                    f"<rect x=\"{margin_left}\" y=\"{y}\" width=\"{bar_len}\" height=\"{bar_height}\" fill=\"#4f7cac\"/>"
                )
                rows.append(
                    f"<text x=\"{margin_left + bar_len + 6}\" y=\"{y + bar_height - 4}\" font-size=\"12\">{_fmt(value)}</text>"
                )
            rows.append("</svg>")
            return "\n".join(rows) + "\n"

        if labels:
            svg_sections.append(("qoi_error.svg", _render_bar_svg("QoI max error", qoi_values)))
            svg_sections.append(("size.svg", _render_bar_svg("Reduction size", size_values)))

    created_at = _utc_now_iso()
    report_lines: list[str] = []
    report_lines.append(f"# {title}")
    report_lines.append("")
    report_lines.append(f"- created_at: {created_at}")
    report_lines.append(f"- run_root: {run_root}")
    report_lines.append("")
    if benchmark_cfg:
        report_lines.append("## Benchmark")
        for key, value in sorted(benchmark_cfg.items()):
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")

    report_lines.append("## Reduction Methods")
    report_lines.extend(reduction_rows)
    report_lines.append("")

    report_lines.append("## Mapping Methods")
    report_lines.extend(mapping_rows)
    report_lines.append("")

    report_lines.append("## Conclusion")
    best_entry = None
    best_metric = None
    for entry in reducer_results:
        if entry["type"] != "reduction":
            continue
        summary = entry.get("validation_summary") or {}
        metric_value = summary.get("max_metric")
        if metric_value is None:
            continue
        try:
            metric_value = float(metric_value)
        except (TypeError, ValueError):
            continue
        if best_metric is None or metric_value < best_metric:
            best_metric = metric_value
            best_entry = entry
    if best_entry is None:
        report_lines.append("- insufficient QoI metrics to rank reducers.")
    else:
        size_text = _fmt(best_entry.get("size_value"))
        if best_entry.get("size_label"):
            size_text = f"{size_text} {best_entry.get('size_label')}"
        report_lines.append(
            f"- lowest QoI max ({metric_label}): {best_entry['label']} "
            f"(error={_fmt(best_metric)}, size={size_text}, "
            f"elapsed={_fmt(best_entry['elapsed_seconds'])}s)"
        )
    report_lines.append("")

    if svg_sections:
        report_lines.append("## Figures")
        for filename, _ in svg_sections:
            report_lines.append(f"![{filename}](figures/{filename})")
        report_lines.append("")

    report_lines.append("## Reducer Artifacts")
    for entry in reducer_results:
        pipeline_label = entry.get("pipeline")
        if not isinstance(pipeline_label, str):
            pipeline_label = "<inline>"
        report_lines.append(
            f"- {entry['label']}: pipeline={pipeline_label} reduction={entry['reduction_id']} "
            f"validation={entry.get('validation_id') or 'n/a'}"
        )
    report_lines.append("")

    if mapping_eval_id:
        report_lines.append("## Mapping Evaluation")
        report_lines.append(f"- mapping_eval_id: {mapping_eval_id}")
        report_lines.append("")

    report_text = "\n".join(report_lines) + "\n"
    summary_payload = {
        "schema_version": 1,
        "created_at": created_at,
        "benchmark": benchmark_cfg,
        "reducers": reducer_results,
        "mapping_eval_id": mapping_eval_id,
        "mapping_eval_metrics": mapping_eval_metrics,
    }

    inputs_payload = {
        "reducers": [
            {"id": entry["id"], "reduction_id": entry["reduction_id"]}
            for entry in reducer_results
        ],
        "mapping_eval": mapping_eval_id,
        "benchmark": benchmark_cfg,
    }
    report_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )

    parent_ids: list[str] = []
    for entry in reducer_results:
        parent_ids.append(entry["reduction_id"])
        if entry.get("validation_id"):
            parent_ids.append(entry["validation_id"])
        if entry.get("graph_id"):
            parent_ids.append(entry["graph_id"])
        if entry.get("run_id"):
            parent_ids.append(entry["run_id"])
    if mapping_eval_id:
        parent_ids.append(mapping_eval_id)

    report_manifest = build_manifest(
        kind="reports",
        artifact_id=report_id,
        created_at=created_at,
        parents=_dedupe_preserve(parent_ids),
        inputs=inputs_payload,
        config=manifest_cfg,
        notes="Benchmark comparison report",
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / report_filename).write_text(report_text, encoding="utf-8")
        write_json_atomic(base_dir / "comparison_summary.json", summary_payload)
        if svg_sections:
            fig_dir = base_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for filename, svg_text in svg_sections:
                (fig_dir / filename).write_text(svg_text, encoding="utf-8")

    result = store.ensure(report_manifest, writer=_writer)

    report_path = report_dir / report_filename
    report_path.write_text(report_text, encoding="utf-8")
    summary_path = report_dir / "comparison_summary.json"
    write_json_atomic(summary_path, summary_payload)
    if svg_sections:
        fig_dir = report_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for filename, svg_text in svg_sections:
            (fig_dir / filename).write_text(svg_text, encoding="utf-8")

    return result


register("task", "reduction.apply", run)
register("task", "reduction.dispatch", dispatch)
register("task", "reduction.threshold_prune", threshold_prune)
register("task", "reduction.gnn_importance_prune", gnn_importance_prune)
register("task", "reduction.node_lumping", propose_node_lumping)
register("task", "reduction.superstate_mapping", superstate_mapping)
register("task", "reduction.node_lumping_prune", node_lumping_prune)
register("task", "reduction.reaction_lumping", propose_reaction_lumping)
register("task", "reduction.reaction_lumping_prune", reaction_lumping_prune)
register("task", "reduction.validate", validate_reduction)
register("task", "reduction.repair_topk", repair_topk)
register("task", "reduction.repair_restore", repair_restore)
register("task", "reduction.repair_cover_restore", repair_cover_restore)
register("task", "reduction.repair_mapping_split", repair_mapping_split)
register("task", "reduction.learnck_style", learnck_style)
register("task", "reduction.amore_search", amore_search)
register("task", "reduction.cnr_coarse", cnr_coarse)
register("task", "reduction.gnn_pool_temporal", gnn_pool_temporal)
register("task", "reduction.evaluate_mapping", evaluate_mapping)
register("task", "reduction.benchmark_compare", benchmark_compare)

__all__ = [
    "run",
    "dispatch",
    "threshold_prune",
    "propose_node_lumping",
    "superstate_mapping",
    "node_lumping_prune",
    "propose_reaction_lumping",
    "reaction_lumping_prune",
    "validate_reduction",
    "repair_topk",
    "repair_restore",
    "repair_cover_restore",
    "repair_mapping_split",
    "learnck_style",
    "amore_search",
    "cnr_coarse",
    "gnn_pool_temporal",
    "evaluate_mapping",
    "benchmark_compare",
]
