"""Reduction task: apply mechanism patches (disable reactions / multipliers)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import copy
import json
import logging
import math
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import (
    ArtifactManifest,
    make_artifact_id,
    normalize_reaction_multipliers,
)
from rxn_platform.errors import ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore

try:  # Optional dependency.
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

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
NODE_LUMPING_SCHEMA_VERSION = 1
NODE_LUMPING_FILENAME = "node_lumping.json"
REACTION_LUMPING_SCHEMA_VERSION = 1
REACTION_LUMPING_FILENAME = "reaction_lumping.json"
DEFAULT_LUMPING_THRESHOLD = 0.85
DEFAULT_LUMPING_METHOD = "threshold"
DEFAULT_LUMPING_CHARGE_SCALE = 1.0
DEFAULT_LUMPING_WEIGHTS = {
    "elements": 1.0,
    "charge": 1.0,
    "phase": 1.0,
    "state": 1.0,
}
DEFAULT_REPRESENTATIVE_METRIC = "degree"
DEFAULT_REACTION_SIMILARITY_METRIC = "jaccard"
DEFAULT_REACTION_SIMILARITY_MODE = "both"
DEFAULT_REACTION_SIMILARITY_WEIGHTS = {
    "reactants": 1.0,
    "products": 1.0,
}
DEFAULT_REACTION_INCLUDE_PARTICIPANTS = False
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


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if yaml is None:
                try:
                    return json.load(handle)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        "PyYAML is not available; patch/mechanism must be JSON-compatible."
                    ) from exc
            return yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"Failed to read YAML from {path}: {exc}") from exc


def _write_yaml(path: Path, payload: Mapping[str, Any], *, sort_keys: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=sort_keys,
                ensure_ascii=True,
            )
            handle.write("\n")
            return
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=sort_keys,
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


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"table not found: {path}")
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
    rows: Any = []
    if isinstance(payload, Mapping):
        rows = payload.get("rows", [])
    elif isinstance(payload, Sequence):
        rows = payload
    if not isinstance(rows, Sequence):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


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
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
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
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
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
        entries.append(
            {
                "node_id": node_id,
                "species": _species_name_from_node(node),
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
        if kind == "reaction_id":
            reaction_id = detail.get("reaction_id") or value
            entries.append({"reaction_id": reaction_id})
        else:
            reaction_index = detail.get("reaction_index")
            if reaction_index is None:
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
        payload = _read_yaml(patch_path)
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
    identifiers: list[str] = [f"R{index + 1}"]
    if isinstance(reaction, Mapping):
        for key in ("id", "name", "equation", "reaction"):
            value = reaction.get(key)
            if isinstance(value, str) and value.strip():
                identifiers.append(value.strip())
    elif isinstance(reaction, str) and reaction.strip():
        identifiers.append(reaction.strip())
    else:
        raise ConfigError("reaction entries must be mappings or strings.")
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def _reaction_id_index_map(
    reactions: Sequence[Any],
) -> dict[str, list[int]]:
    id_map: dict[str, list[int]] = {}
    for idx, reaction in enumerate(reactions):
        identifiers = _reaction_identifiers(reaction, idx)
        for identifier in identifiers:
            id_map.setdefault(identifier, []).append(idx)
    return id_map


def _resolve_patch_entries(
    entries: Sequence[Mapping[str, Any]],
    reactions: Sequence[Any],
) -> dict[int, dict[str, Any]]:
    if not reactions:
        raise ConfigError("mechanism has no reactions to patch.")
    id_map = _reaction_id_index_map(reactions)
    resolved: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if "index" in entry:
            idx = entry["index"]
            if not isinstance(idx, int):
                raise ConfigError("patch reaction index must be an int.")
            if idx < 0 or idx >= len(reactions):
                raise ConfigError(f"patch reaction index out of range: {idx}.")
        else:
            reaction_id = entry.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id.strip():
                raise ConfigError("patch reaction_id must be a non-empty string.")
            indices = id_map.get(reaction_id)
            if not indices:
                raise ConfigError(f"reaction_id not found in mechanism: {reaction_id!r}.")
            if len(indices) > 1:
                raise ConfigError(
                    f"reaction_id matches multiple reactions: {reaction_id!r}."
                )
            idx = indices[0]
        resolved[idx] = dict(entry)
    return resolved


def _apply_patch_entries(
    mechanism: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], set[int]]:
    reactions = mechanism.get("reactions")
    if reactions is None:
        raise ConfigError("mechanism must define a reactions list.")
    if not isinstance(reactions, Sequence) or isinstance(
        reactions,
        (str, bytes, bytearray),
    ):
        raise ConfigError("mechanism.reactions must be a sequence.")
    reaction_list = list(reactions)
    resolved = _resolve_patch_entries(entries, reaction_list)
    disabled_indices = {
        idx for idx, entry in resolved.items() if float(entry.get("multiplier", 0.0)) == 0.0
    }
    filtered_reactions = [
        reaction
        for idx, reaction in enumerate(reaction_list)
        if idx not in disabled_indices
    ]
    updated = dict(mechanism)
    updated["reactions"] = filtered_reactions
    return updated, disabled_indices


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
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=parents,
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_yaml(
            base_dir / PATCH_FILENAME,
            patch_payload,
            sort_keys=True,
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
            components = {
                "elements": element_sim,
                "charge": charge_sim,
                "phase": phase_sim,
                "state": state_sim,
            }
            score = (
                weights["elements"] * element_sim
                + weights["charge"] * charge_sim
                + weights["phase"] * phase_sim
                + weights["state"] * state_sim
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

    species_payload = [
        {
            "node_id": entry["node_id"],
            "species": entry["species"],
            "elements": entry["elements"],
            "charge": entry["charge"],
            "phase": entry["phase"],
            "state": entry["state"],
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
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / NODE_LUMPING_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / REACTION_LUMPING_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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

    mechanism_payload = _read_yaml(Path(mechanism_path))
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
    manifest = ArtifactManifest(
        schema_version=1,
        kind="reduction",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_yaml(
            base_dir / PATCH_FILENAME,
            normalized_patch,
            sort_keys=True,
        )
        _write_yaml(
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
    patch_payload = _read_yaml(reduction_dir / PATCH_FILENAME)
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
    patch_payload = _read_yaml(result.path / PATCH_FILENAME)
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
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    inputs_cfg = reduction_cfg.get("inputs")
    if isinstance(inputs_cfg, Mapping):
        if not any(
            key in validation_cfg for key in ("patches", "patch", "candidates")
        ):
            for key in ("patches", "patch", "candidates"):
                if key in inputs_cfg:
                    validation_cfg[key] = inputs_cfg[key]
                    break

    mechanism_path = _extract_mechanism(reduction_cfg)
    pipeline_value = _extract_pipeline_value(validation_cfg)

    metric = _extract_metric_name(validation_cfg)
    tolerance = _extract_tolerance(validation_cfg)
    rel_eps = _extract_rel_eps(validation_cfg)
    missing_strategy = _normalize_missing_strategy(
        validation_cfg.get("missing_strategy")
    )
    stop_on_fail = bool(validation_cfg.get("stop_on_fail", True))

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

    patch_candidates = _normalize_patch_candidates(validation_cfg)
    metrics_rows: list[dict[str, Any]] = []
    patch_summaries: list[dict[str, Any]] = []
    selected_patch: Optional[dict[str, Any]] = None
    parents: list[str] = []
    if baseline_obs_id:
        parents.append(baseline_obs_id)
    if baseline_feat_id:
        parents.append(baseline_feat_id)
    parents.append(baseline_run_id)

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
        if not reduced_mechanism.exists():
            temp_dir = tempfile.TemporaryDirectory(prefix="rxn_reduction_")
            reduced_mechanism = Path(temp_dir.name) / MECHANISM_FILENAME
            _, combined_entries = _normalize_patch_payload(patch_payload)
            mechanism_payload = _read_yaml(Path(mechanism_path))
            if not isinstance(mechanism_payload, Mapping):
                raise ConfigError("mechanism YAML must be a mapping.")
            reduced_payload, _ = _apply_patch_entries(
                dict(mechanism_payload),
                combined_entries,
            )
            _write_yaml(reduced_mechanism, reduced_payload, sort_keys=False)
        multipliers = _extract_patch_multipliers(patch_payload)

        reduced_sim_cfg = dict(baseline_sim_cfg)
        reduced_sim_cfg["mechanism"] = str(reduced_mechanism)
        if multipliers:
            reduced_sim_cfg["reaction_multipliers"] = multipliers
        else:
            reduced_sim_cfg.pop("reaction_multipliers", None)
        reduced_sim_cfg.pop("disabled_reactions", None)

        reduced_pipeline_cfg = copy.deepcopy(pipeline_cfg)
        for step in reduced_pipeline_cfg.get("steps", []):
            if step.get("id") == sim_step_id:
                step["sim"] = dict(reduced_sim_cfg)
                break

        try:
            reduced_results = runner.run(reduced_pipeline_cfg)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
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
                raise ConfigError("reduced observables step did not produce an artifact.")
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
                patch_index=index,
                patch_id=reduction_id,
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
                baseline_groups=baseline_feat_groups,
                reduced_groups=reduced_feat_groups,
                metric=metric,
                tolerance=tolerance,
                rel_eps=rel_eps,
                missing_strategy=missing_strategy,
                patch_index=index,
                patch_id=reduction_id,
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

        metrics_rows.extend(patch_rows)
        summary = {
            "patch_index": index,
            "reduction_id": reduction_id,
            "passed": patch_pass,
            "run_id": reduced_run_id,
            "observables": reduced_obs_id,
            "features": reduced_feat_id,
        }
        patch_summaries.append(summary)
        if patch_pass:
            selected_patch = summary
        if stop_on_fail and not patch_pass:
            parents.append(reduction_id)
            parents.append(reduced_run_id)
            if reduced_obs_id:
                parents.append(reduced_obs_id)
            if reduced_feat_id:
                parents.append(reduced_feat_id)
            break

        parents.append(reduction_id)
        parents.append(reduced_run_id)
        if reduced_obs_id:
            parents.append(reduced_obs_id)
        if reduced_feat_id:
            parents.append(reduced_feat_id)

    if not metrics_rows:
        raise ConfigError("validation produced no comparison metrics.")

    inputs_payload: dict[str, Any] = {
        "baseline_run_id": baseline_run_id,
        "baseline_observables_id": baseline_obs_id,
        "baseline_features_id": baseline_feat_id,
        "patches": [
            {"patch_index": entry["patch_index"], "reduction_id": entry["reduction_id"]}
            for entry in patch_summaries
        ],
        "selected_patch": selected_patch,
        "passed": selected_patch is not None,
        "metric": {
            "name": metric,
            "tolerance": tolerance,
            "rel_eps": rel_eps,
            "missing_strategy": missing_strategy,
        },
    }

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="validation",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=_dedupe_preserve(parents),
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        _write_validation_table(metrics_rows, base_dir / "metrics.parquet")

    return store.ensure(manifest, writer=_writer)


register("task", "reduction.apply", run)
register("task", "reduction.threshold_prune", threshold_prune)
register("task", "reduction.node_lumping", propose_node_lumping)
register("task", "reduction.reaction_lumping", propose_reaction_lumping)
register("task", "reduction.validate", validate_reduction)

__all__ = [
    "run",
    "threshold_prune",
    "propose_node_lumping",
    "propose_reaction_lumping",
    "validate_reduction",
]
