"""Graph tasks for stoichiometric matrix artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import re
import subprocess
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import ArtifactManifest, make_artifact_id
from rxn_platform.errors import ArtifactError, ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore

try:  # Optional dependency.
    import cantera as ct
except ImportError:  # pragma: no cover - optional dependency
    ct = None

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
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None


@dataclass(frozen=True)
class StoichResult:
    matrix: Any
    species: list[str]
    reaction_ids: list[str]
    reaction_equations: list[str]
    format: str


@dataclass(frozen=True)
class LaplacianResult:
    laplacian: Any
    degree: Any
    nodes: list[str]
    normalized_laplacian: Optional[Any]
    format: str
    normalized_format: Optional[str]


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


def _extract_graph_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "graphs" in cfg and isinstance(cfg.get("graphs"), Mapping):
        graph_cfg = cfg.get("graphs")
        if not isinstance(graph_cfg, Mapping):
            raise ConfigError("graphs config must be a mapping.")
        return dict(cfg), dict(graph_cfg)
    if "graph" in cfg and isinstance(cfg.get("graph"), Mapping):
        graph_cfg = cfg.get("graph")
        if not isinstance(graph_cfg, Mapping):
            raise ConfigError("graph config must be a mapping.")
        return dict(cfg), dict(graph_cfg)
    return dict(cfg), dict(cfg)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_optional_str(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_str(value, label)


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


def _normalize_mechanism(value: Any) -> str:
    mech = _require_nonempty_str(value, "mechanism")
    mech_path = Path(mech)
    if mech_path.is_absolute() or len(mech_path.parts) > 1:
        if not mech_path.exists():
            raise ConfigError(f"mechanism file not found: {mech}")
    return mech


def _extract_mechanism(graph_cfg: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    mechanism: Any = None
    phase: Any = None
    inputs = graph_cfg.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise ConfigError("graphs.inputs must be a mapping.")
        for key in ("mechanism", "solution", "mechanism_path", "source"):
            if key in inputs:
                mechanism = inputs.get(key)
                break
        if "phase" in inputs:
            phase = inputs.get("phase")
    if mechanism is None:
        for key in ("mechanism", "solution", "mechanism_path", "source"):
            if key in graph_cfg:
                mechanism = graph_cfg.get(key)
                break
    if phase is None and "phase" in graph_cfg:
        phase = graph_cfg.get("phase")

    mechanism = _normalize_mechanism(mechanism)

    if phase is not None:
        phase = _require_nonempty_str(phase, "phase")
    return mechanism, phase


def _coerce_single_run_id(value: Any) -> str:
    if value is None:
        raise ConfigError("run_id is required for run-based graphs.")
    if isinstance(value, str):
        return _require_nonempty_str(value, "run_id")
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items = [item for item in value if item is not None]
        if len(items) != 1:
            raise ConfigError("run_id must include exactly one entry.")
        return _require_nonempty_str(items[0], "run_id")
    raise ConfigError("run_id must be a string or single-item sequence.")


def _extract_run_id(graph_cfg: Mapping[str, Any]) -> str:
    run_id: Any = None
    inputs = graph_cfg.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise ConfigError("graphs.inputs must be a mapping.")
        for key in ("run_id", "run", "runs", "run_ids"):
            if key in inputs:
                run_id = inputs.get(key)
                break
    if run_id is None:
        for key in ("run_id", "run", "runs", "run_ids"):
            if key in graph_cfg:
                run_id = graph_cfg.get(key)
                break
    return _coerce_single_run_id(run_id)


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


def _load_run_dataset_payload(run_dir: Path) -> dict[str, Any]:
    dataset_path = run_dir / "state.zarr" / "dataset.json"
    if dataset_path.exists():
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Run dataset JSON is invalid: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ArtifactError("Run dataset JSON must be a mapping.")
        return dict(payload)
    if xr is None:
        raise ArtifactError(
            "Run dataset not found; install xarray to load state.zarr."
        )
    dataset = xr.open_zarr(run_dir / "state.zarr")
    coords = {
        name: {"dims": [name], "data": dataset.coords[name].values.tolist()}
        for name in dataset.coords
    }
    data_vars = {
        name: {"dims": list(dataset[name].dims), "data": dataset[name].values.tolist()}
        for name in dataset.data_vars
    }
    return {"coords": coords, "data_vars": data_vars, "attrs": dict(dataset.attrs)}


def _extract_coord_names(payload: Mapping[str, Any], coord: str) -> list[str]:
    coords = payload.get("coords", {})
    if not isinstance(coords, Mapping):
        raise ArtifactError("Run dataset coords must be a mapping.")
    coord_payload = coords.get(coord)
    if not isinstance(coord_payload, Mapping):
        return []
    return _coerce_str_sequence(coord_payload.get("data"), f"coords.{coord}.data")


def _build_run_bipartite_graph(
    *,
    gas_species: Sequence[str],
    surface_species: Sequence[str],
    reactions: Sequence[str],
) -> dict[str, Any]:
    species_nodes: list[dict[str, Any]] = []
    reaction_nodes: list[dict[str, Any]] = []
    species_node_ids: list[str] = []
    reaction_node_ids: list[str] = []

    for idx, name in enumerate(gas_species):
        node_id = f"species_{name}"
        species_node_ids.append(node_id)
        species_nodes.append(
            {
                "id": node_id,
                "kind": "species",
                "label": name,
                "species": name,
                "phase": "gas",
                "species_index": idx,
            }
        )

    for idx, name in enumerate(surface_species):
        node_id = f"surface_{name}"
        species_node_ids.append(node_id)
        species_nodes.append(
            {
                "id": node_id,
                "kind": "species",
                "label": name,
                "species": name,
                "phase": "surface",
                "species_index": idx,
            }
        )

    for idx, reaction_id in enumerate(reactions):
        node_id = f"reaction_{idx + 1}"
        reaction_node_ids.append(node_id)
        reaction_nodes.append(
            {
                "id": node_id,
                "kind": "reaction",
                "label": reaction_id,
                "reaction_id": reaction_id,
                "reaction_index": idx,
            }
        )

    links: list[dict[str, Any]] = []
    if reaction_node_ids and species_node_ids:
        for r_index, reaction_node_id in enumerate(reaction_node_ids):
            for s_index, species_node_id in enumerate(species_node_ids):
                value = -1.0 if (r_index + s_index) % 2 == 0 else 1.0
                links.append(
                    {
                        "source": species_node_id,
                        "target": reaction_node_id,
                        "stoich": value,
                        "role": "reactant" if value < 0 else "product",
                    }
                )

    graph_attrs = {
        "bipartite": "species-reaction",
        "edge_direction": "species_to_reaction",
        "stoich_sign": "synthetic",
    }
    return {
        "directed": True,
        "multigraph": False,
        "graph": graph_attrs,
        "nodes": species_nodes + reaction_nodes,
        "links": links,
    }



def _reaction_equations(solution: Any, count: int) -> list[str]:
    if count <= 0:
        return []
    try:
        equations = list(solution.reaction_equations())
        if len(equations) == count:
            return [str(entry) for entry in equations]
    except Exception:
        pass
    labels: list[str] = []
    for idx in range(count):
        try:
            labels.append(str(solution.reaction_equation(idx)))
        except Exception:
            labels.append(f"R{idx + 1}")
    return labels


def _reaction_ids(solution: Any, count: int) -> list[str]:
    ids: list[str] = []
    for idx in range(count):
        reaction_id: Optional[str] = None
        try:
            reaction = solution.reaction(idx)
            reaction_id = getattr(reaction, "id", None)
        except Exception:
            reaction = None
        if reaction_id:
            ids.append(str(reaction_id))
        else:
            ids.append(f"R{idx + 1}")
    return ids


def _normalize_phase_label(value: str) -> str:
    label = value.strip().lower()
    if not label:
        return "unknown"
    if "gas" in label:
        return "gas"
    if "surf" in label or "surface" in label or "interface" in label:
        return "surface"
    if "solid" in label or "bulk" in label:
        return "solid"
    return "unknown"


def _infer_phase_label(phase: Optional[str], solution: Any) -> tuple[str, bool]:
    if phase:
        normalized = _normalize_phase_label(str(phase))
        if normalized != "unknown":
            return normalized, False
    n_sites = getattr(solution, "n_sites", None)
    if isinstance(n_sites, (int, float)) and n_sites:
        return "surface", True
    for attr in ("phase_name", "name", "thermo_model", "transport_model"):
        value = getattr(solution, attr, None)
        if value:
            normalized = _normalize_phase_label(str(value))
            if normalized != "unknown":
                return normalized, True
    return "unknown", True


def _coerce_element_counts(
    composition: Any,
) -> tuple[dict[str, float], bool]:
    if not isinstance(composition, Mapping):
        return {}, True
    elements: dict[str, float] = {}
    inferred = False
    for element, count in composition.items():
        if element is None:
            inferred = True
            continue
        try:
            value = float(count)
        except (TypeError, ValueError):
            inferred = True
            continue
        if value == 0.0:
            continue
        elements[str(element)] = value
    if not elements:
        inferred = True
    return elements, inferred


def _formula_element_order(elements: Mapping[str, float]) -> list[str]:
    names = sorted(elements.keys())
    if "C" in elements:
        order = ["C"]
        if "H" in elements:
            order.append("H")
        for name in names:
            if name not in ("C", "H"):
                order.append(name)
        return order
    return names


def _format_count(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-8):
        int_value = int(round(value))
        return "" if int_value == 1 else str(int_value)
    return "" if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-8) else f"{value:g}"


def _format_formula(elements: Mapping[str, float]) -> str:
    if not elements:
        return ""
    parts: list[str] = []
    for element in _formula_element_order(elements):
        count = elements[element]
        parts.append(f"{element}{_format_count(count)}")
    return "".join(parts)


def _elements_vector(elements: Mapping[str, float]) -> list[list[Any]]:
    return [[element, elements[element]] for element in _formula_element_order(elements)]


def _parse_charge_from_name(name: str) -> Optional[int]:
    match = re.search(r"([+-]+)$", name)
    if match:
        signs = match.group(1)
        return signs.count("+") - signs.count("-")
    match = re.search(r"([+-])(\d+)$", name)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        return sign * int(match.group(2))
    return None


def _extract_charge(species: Any, name: str) -> tuple[Optional[float], bool]:
    charge_value: Optional[Any] = None
    if species is not None:
        value = getattr(species, "charge", None)
        if callable(value):
            try:
                charge_value = value()
            except Exception:
                charge_value = None
        else:
            charge_value = value
    if charge_value is not None:
        try:
            charge = float(charge_value)
        except (TypeError, ValueError):
            charge = None
        if charge is not None:
            if math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
                charge = 0.0
            return charge, False
    parsed = _parse_charge_from_name(name)
    if parsed is not None:
        return float(parsed), True
    return None, True


def _radical_heuristic(name: str) -> Optional[bool]:
    if name.endswith(".") or "RAD" in name or "rad" in name:
        return True
    return None


def _classify_state(name: str, charge: Optional[float]) -> tuple[str, bool]:
    if charge is None:
        return "unknown", True
    if not math.isclose(charge, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "ion", False
    if _radical_heuristic(name):
        return "radical", True
    return "neutral", True


def annotate_species(
    solution: Any,
    *,
    phase: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Annotate species with formula/elements/charge/state/phase metadata."""
    species_names = list(getattr(solution, "species_names", []))
    if not species_names:
        raise ConfigError("mechanism has no species.")
    phase_label, phase_inferred = _infer_phase_label(phase, solution)

    annotations: dict[str, dict[str, Any]] = {}
    for name in species_names:
        species_obj = None
        species_attr = getattr(solution, "species", None)
        if callable(species_attr):
            try:
                species_obj = species_attr(name)
            except Exception:
                species_obj = None
        composition = getattr(species_obj, "composition", None)
        elements, elements_inferred = _coerce_element_counts(composition)
        formula = _format_formula(elements)
        charge, charge_inferred = _extract_charge(species_obj, name)
        state, state_inferred = _classify_state(name, charge)

        inferred_fields: list[str] = []
        if elements_inferred:
            inferred_fields.extend(["elements", "formula"])
        if charge_inferred:
            inferred_fields.append("charge")
        if phase_inferred:
            inferred_fields.append("phase")
        if state_inferred:
            inferred_fields.append("state")

        annotations[name] = {
            "formula": formula or None,
            "elements": elements,
            "elements_vector": _elements_vector(elements),
            "charge": charge,
            "phase": phase_label,
            "state": state,
            "is_inferred": bool(inferred_fields),
            "inferred_fields": sorted(set(inferred_fields)),
        }
    return annotations


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(value)
    return text if text else None


def _normalize_reaction_type(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return "unknown"
    lowered = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    if not lowered:
        return "unknown"
    if "plog" in lowered:
        return "plog"
    if "chebyshev" in lowered:
        return "chebyshev"
    if "falloff" in lowered or "lindemann" in lowered or "troe" in lowered:
        return "falloff"
    if "chem" in lowered and "activ" in lowered:
        return "chemically-activated"
    if "three" in lowered and "body" in lowered:
        return "three-body"
    if "arrhenius" in lowered or "blowers" in lowered:
        return "elementary"
    if "pressure" in lowered and ("dep" in lowered or "depend" in lowered):
        return "pressure-dependent"
    if "pdep" in lowered:
        return "pressure-dependent"
    if "electro" in lowered:
        return "electrochemical"
    if "adsorption" in lowered:
        return "adsorption"
    if "desorption" in lowered:
        return "desorption"
    if "surface" in lowered:
        return "surface"
    if "interface" in lowered:
        return "interface"
    if "elementary" in lowered:
        return "elementary"
    return "unknown"


def _reaction_type_from_attr(reaction: Any) -> tuple[Optional[str], Optional[str]]:
    for attr in ("reaction_type", "type"):
        value = getattr(reaction, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        text = _as_text(value)
        if text:
            return text, f"attr:{attr}"
    return None, None


def _has_third_body(reaction: Any) -> bool:
    if getattr(reaction, "third_body", None) is not None:
        return True
    efficiencies = getattr(reaction, "efficiencies", None)
    if isinstance(efficiencies, Mapping) and efficiencies:
        return True
    if getattr(reaction, "default_efficiency", None) is not None:
        return True
    return False


def _has_falloff(reaction: Any) -> bool:
    for attr in ("falloff", "falloff_params", "falloff_parameters", "falloff_coeffs"):
        if getattr(reaction, attr, None) is not None:
            return True
    return False


def _classify_reaction_type(
    reaction: Any,
    *,
    phase_label: Optional[str],
) -> tuple[str, str, bool]:
    explicit, source = _reaction_type_from_attr(reaction)
    fallback_reason: Optional[str] = None
    if explicit:
        normalized = _normalize_reaction_type(explicit)
        if normalized != "unknown":
            return normalized, f"{source}:{explicit}", False
        fallback_reason = f"{source}:{explicit}"
    class_name = type(reaction).__name__
    normalized = _normalize_reaction_type(class_name)
    if normalized != "unknown":
        return normalized, f"class_name:{class_name}", True
    rate = getattr(reaction, "rate", None)
    if rate is not None:
        rate_class = type(rate).__name__
        normalized = _normalize_reaction_type(rate_class)
        if normalized != "unknown":
            return normalized, f"rate_class:{rate_class}", True
        for attr in ("type", "rate_type"):
            value = getattr(rate, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            normalized = _normalize_reaction_type(value)
            if normalized != "unknown":
                return normalized, f"rate_attr:{attr}:{value}", True
    if _has_falloff(reaction):
        return "falloff", "heuristic:falloff", True
    if _has_third_body(reaction):
        return "three-body", "heuristic:third_body", True
    if phase_label == "surface":
        return "surface", "heuristic:phase", True
    if fallback_reason:
        return "unknown", f"unknown: {fallback_reason}", True
    return "unknown", "unknown: no reaction type match", True


def _extract_bool_flag(reaction: Any, attr: str) -> tuple[Optional[bool], bool]:
    value = getattr(reaction, attr, None)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    if value is None:
        return None, True
    if isinstance(value, bool):
        return value, False
    if isinstance(value, (int, float)):
        return bool(value), True
    return None, True


def _estimate_reaction_order(reaction: Any) -> tuple[Optional[float], str, bool]:
    reactants = getattr(reaction, "reactants", None)
    if not isinstance(reactants, Mapping):
        return None, "missing_reactants", True
    if not reactants:
        return None, "empty_reactants", True
    total = 0.0
    valid = 0
    invalid = False
    for coeff in reactants.values():
        try:
            value = float(coeff)
        except (TypeError, ValueError):
            invalid = True
            continue
        total += value
        valid += 1
    if valid == 0:
        return None, "non_numeric_reactants", True
    if invalid:
        return total, "partial_stoich_sum", True
    return total, "stoich_sum", True


def annotate_reactions(
    solution: Any,
    *,
    phase: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Annotate reactions with type/order/reversible/duplicate metadata."""
    n_reactions = int(getattr(solution, "n_reactions", 0) or 0)
    if n_reactions <= 0:
        raise ConfigError("mechanism has no reactions.")
    phase_label, _ = _infer_phase_label(phase, solution)
    reaction_ids = _reaction_ids(solution, n_reactions)
    annotations: dict[str, dict[str, Any]] = {}
    for idx in range(n_reactions):
        reaction = solution.reaction(idx)
        reaction_id = reaction_ids[idx]
        reaction_type, type_reason, type_inferred = _classify_reaction_type(
            reaction,
            phase_label=phase_label,
        )
        order, order_source, order_inferred = _estimate_reaction_order(reaction)
        reversible, reversible_inferred = _extract_bool_flag(reaction, "reversible")
        duplicate, duplicate_inferred = _extract_bool_flag(reaction, "duplicate")

        inferred_fields: list[str] = []
        if type_inferred:
            inferred_fields.append("reaction_type")
        if order_inferred:
            inferred_fields.append("order")
        if reversible_inferred:
            inferred_fields.append("reversible")
        if duplicate_inferred:
            inferred_fields.append("duplicate")

        annotations[reaction_id] = {
            "reaction_index": idx,
            "reaction_type": reaction_type,
            "reaction_type_reason": type_reason,
            "order": order,
            "order_source": order_source,
            "reversible": reversible,
            "duplicate": duplicate,
            "is_inferred": bool(inferred_fields),
            "inferred_fields": sorted(set(inferred_fields)),
        }
    return annotations


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a float, got {value!r}.") from exc


def _stoich_entries(
    solution: Any,
    species_index: Mapping[str, int],
    n_reactions: int,
) -> dict[tuple[int, int], float]:
    entries: dict[tuple[int, int], float] = {}
    for rxn_idx in range(n_reactions):
        reaction = solution.reaction(rxn_idx)
        reactants = getattr(reaction, "reactants", None)
        products = getattr(reaction, "products", None)
        if not isinstance(reactants, Mapping) or not isinstance(products, Mapping):
            raise ConfigError("Reaction stoichiometry must be mappings.")
        for name, coeff in reactants.items():
            if name not in species_index:
                raise ConfigError(f"Unknown reactant species: {name}")
            value = -_as_float(coeff, f"reactants[{name}]")
            key = (species_index[name], rxn_idx)
            entries[key] = entries.get(key, 0.0) + value
        for name, coeff in products.items():
            if name not in species_index:
                raise ConfigError(f"Unknown product species: {name}")
            value = _as_float(coeff, f"products[{name}]")
            key = (species_index[name], rxn_idx)
            entries[key] = entries.get(key, 0.0) + value
    return entries


def build_stoich(solution: Any) -> StoichResult:
    """Build a species x reaction stoichiometric matrix from a Cantera Solution."""
    species_names = list(getattr(solution, "species_names", []))
    if not species_names:
        raise ConfigError("mechanism has no species.")
    n_reactions = int(getattr(solution, "n_reactions", 0) or 0)
    if n_reactions <= 0:
        raise ConfigError("mechanism has no reactions.")

    reaction_equations = _reaction_equations(solution, n_reactions)
    reaction_ids = _reaction_ids(solution, n_reactions)

    species_index = {name: idx for idx, name in enumerate(species_names)}
    entries = _stoich_entries(solution, species_index, n_reactions)
    rows = [key[0] for key in entries]
    cols = [key[1] for key in entries]
    data = [entries[key] for key in entries]

    if sp is not None:
        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(species_names), n_reactions),
            dtype=float,
        )
        fmt = "scipy.sparse.coo"
    else:
        if np is None:
            raise ConfigError("numpy is required to build dense stoichiometry.")
        matrix = np.zeros((len(species_names), n_reactions), dtype=float)
        for (row, col), value in entries.items():
            matrix[row, col] = value
        fmt = "numpy.ndarray"

    return StoichResult(
        matrix=matrix,
        species=species_names,
        reaction_ids=reaction_ids,
        reaction_equations=reaction_equations,
        format=fmt,
    )


def _iter_stoich_entries(result: StoichResult) -> list[tuple[int, int, float]]:
    matrix = result.matrix
    entries: list[tuple[int, int, float]] = []
    if sp is not None and hasattr(matrix, "tocoo"):
        coo = matrix.tocoo()
        for row, col, value in zip(coo.row, coo.col, coo.data):
            if value != 0:
                entries.append((int(row), int(col), float(value)))
        return entries
    if np is not None and isinstance(matrix, np.ndarray):
        rows, cols = np.nonzero(matrix)
        for row, col in zip(rows, cols):
            value = float(matrix[row, col])
            if value != 0.0:
                entries.append((int(row), int(col), value))
        return entries
    try:
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                if value:
                    entries.append((row_idx, col_idx, float(value)))
    except TypeError as exc:
        raise ConfigError(
            "Unsupported stoichiometric matrix type for bipartite graph."
        ) from exc
    return entries


def build_bipartite_graph(
    result: StoichResult,
    *,
    species_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    reaction_annotations: Optional[Mapping[Any, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a node-link bipartite graph from the stoichiometric matrix."""
    species_nodes: list[dict[str, Any]] = []
    reaction_nodes: list[dict[str, Any]] = []
    species_node_ids: list[str] = []
    reaction_node_ids: list[str] = []

    for idx, name in enumerate(result.species):
        node_id = f"species_{name}"
        species_node_ids.append(node_id)
        node = {
            "id": node_id,
            "kind": "species",
            "label": name,
            "species_index": idx,
        }
        if species_annotations and name in species_annotations:
            annotation = species_annotations.get(name)
            if isinstance(annotation, Mapping):
                node.update(annotation)
        species_nodes.append(node)

    for idx, reaction_id in enumerate(result.reaction_ids):
        node_id = f"reaction_{idx + 1}"
        reaction_node_ids.append(node_id)
        node = {
            "id": node_id,
            "kind": "reaction",
            "reaction_id": reaction_id,
            "reaction_index": idx,
            "reaction_equation": result.reaction_equations[idx],
        }
        if reaction_annotations:
            annotation = None
            if reaction_id in reaction_annotations:
                annotation = reaction_annotations.get(reaction_id)
            elif idx in reaction_annotations:
                annotation = reaction_annotations.get(idx)
            elif str(idx) in reaction_annotations:
                annotation = reaction_annotations.get(str(idx))
            if isinstance(annotation, Mapping):
                node.update(annotation)
        reaction_nodes.append(node)

    links: list[dict[str, Any]] = []
    for row, col, value in _iter_stoich_entries(result):
        role = "reactant" if value < 0 else "product"
        links.append(
            {
                "source": species_node_ids[row],
                "target": reaction_node_ids[col],
                "stoich": value,
                "role": role,
            }
        )

    graph_attrs = {
        "bipartite": "species-reaction",
        "edge_direction": "species_to_reaction",
        "stoich_sign": "reactants_negative_products_positive",
    }

    nodes = species_nodes + reaction_nodes
    if nx is not None:
        graph = nx.DiGraph()
        graph.graph.update(graph_attrs)
        for node in nodes:
            node_id = node["id"]
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for link in links:
            source = link["source"]
            target = link["target"]
            attrs = {
                key: value
                for key, value in link.items()
                if key not in {"source", "target"}
            }
            graph.add_edge(source, target, **attrs)
        return nx.readwrite.json_graph.node_link_data(graph)

    return {
        "directed": True,
        "multigraph": False,
        "graph": graph_attrs,
        "nodes": nodes,
        "links": links,
    }


def _edge_weight(
    link: Mapping[str, Any],
    *,
    weight_key: Optional[str],
    use_abs_weights: bool,
) -> float:
    value: Any = None
    if weight_key:
        value = link.get(weight_key)
    else:
        for key in ("weight", "value", "stoich"):
            if key in link:
                value = link.get(key)
                break
    if value is None:
        weight = 1.0
    else:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = 1.0
    if not math.isfinite(weight):
        raise ConfigError("edge weight must be finite.")
    if use_abs_weights:
        weight = abs(weight)
    if weight < 0.0:
        raise ConfigError("edge weight must be non-negative.")
    return weight


def build_laplacian(
    graph_payload: Mapping[str, Any],
    *,
    normalized: bool = False,
    weight_key: Optional[str] = None,
    use_abs_weights: bool = True,
    symmetrize: bool = True,
) -> LaplacianResult:
    """Build degree and Laplacian matrices from a node-link graph payload."""
    if np is None:
        raise ConfigError("numpy is required to build Laplacian matrices.")
    if not isinstance(graph_payload, Mapping):
        raise ConfigError("graph payload must be a mapping.")

    if "nodes" in graph_payload and ("links" in graph_payload or "edges" in graph_payload):
        graph_data = dict(graph_payload)
    else:
        graph_data, _ = _extract_node_link_payload(graph_payload)

    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    nodes, node_map = _normalize_nodes(nodes_raw)
    links = _normalize_links(links_raw)

    node_ids = [node["id"] for node in nodes]
    edge_nodes = set()
    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is not None:
            edge_nodes.add(source)
        if target is not None:
            edge_nodes.add(target)
    missing_nodes = sorted(edge_nodes.difference(node_map))
    for node_id in missing_nodes:
        node_map[node_id] = {"id": node_id}
        node_ids.append(node_id)

    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    size = len(node_ids)
    adjacency = np.zeros((size, size), dtype=float)

    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is None or target is None:
            continue
        weight = _edge_weight(
            link,
            weight_key=weight_key,
            use_abs_weights=use_abs_weights,
        )
        if weight == 0.0:
            continue
        row = index.get(source)
        col = index.get(target)
        if row is None or col is None:
            continue
        adjacency[row, col] += weight
        if symmetrize and row != col:
            adjacency[col, row] += weight

    degree = adjacency.sum(axis=1)
    laplacian = -adjacency
    np.fill_diagonal(laplacian, degree)

    normalized_laplacian: Optional[Any] = None
    normalized_format: Optional[str] = None
    if normalized:
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0.0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        scaled = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
        norm = -scaled
        np.fill_diagonal(norm, 1.0)
        if np.any(~nonzero):
            zero_idx = np.where(~nonzero)[0]
            for idx in zero_idx:
                norm[idx, idx] = 0.0
        normalized_laplacian = norm
        normalized_format = "numpy.ndarray"

    return LaplacianResult(
        laplacian=laplacian,
        degree=degree,
        nodes=node_ids,
        normalized_laplacian=normalized_laplacian,
        format="numpy.ndarray",
        normalized_format=normalized_format,
    )


def _load_solution(mechanism: str, phase: Optional[str]) -> Any:
    if ct is None:
        raise ConfigError("Cantera is required to build stoichiometric graphs.")
    if phase is None:
        return ct.Solution(mechanism)
    return ct.Solution(mechanism, phase)


def _write_stoich_npz(path: Path, result: StoichResult) -> None:
    if sp is not None and hasattr(result.matrix, "tocsr"):
        sp.save_npz(path, result.matrix.tocsr())
        return
    if np is None:
        raise ConfigError("numpy is required to write stoich.npz without scipy.")
    np.savez_compressed(path, stoich=result.matrix)


def _write_laplacian_npz(path: Path, result: LaplacianResult) -> None:
    if np is None:
        raise ConfigError("numpy is required to write laplacian.npz.")
    payload: dict[str, Any] = {
        "laplacian": np.asarray(result.laplacian, dtype=float),
        "degree": np.asarray(result.degree, dtype=float),
    }
    if result.normalized_laplacian is not None:
        payload["laplacian_norm"] = np.asarray(
            result.normalized_laplacian, dtype=float
        )
    np.savez_compressed(path, **payload)


def _graph_metadata(
    result: StoichResult,
    *,
    mechanism: str,
    phase: Optional[str],
    bipartite_graph: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    reactions = [
        {"id": rid, "equation": eq}
        for rid, eq in zip(result.reaction_ids, result.reaction_equations)
    ]
    metadata: dict[str, Any] = {
        "kind": "stoichiometric_matrix",
        "shape": [len(result.species), len(result.reaction_ids)],
        "species": list(result.species),
        "reactions": reactions,
        "stoich": {"path": "stoich.npz", "format": result.format},
        "source": {"mechanism": mechanism},
    }
    if phase is not None:
        metadata["source"]["phase"] = phase
    if bipartite_graph is not None:
        links_key = "links" if "links" in bipartite_graph else "edges"
        metadata["bipartite"] = {
            "format": "node_link",
            "node_count": len(bipartite_graph.get("nodes", [])),
            "edge_count": len(bipartite_graph.get(links_key, [])),
            "data": bipartite_graph,
        }
    return metadata


def _extract_analysis_cfg(graph_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("analysis", "analytics", "graph_analysis"):
        if key in graph_cfg:
            value = graph_cfg.get(key)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(graph_cfg)


def _extract_laplacian_cfg(graph_cfg: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("laplacian", "graph_laplacian"):
        if key in graph_cfg:
            value = graph_cfg.get(key)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} config must be a mapping.")
            return dict(value)
    return dict(graph_cfg)


def _extract_graph_id(analysis_cfg: Mapping[str, Any]) -> str:
    graph_id: Any = None
    inputs = analysis_cfg.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("graph_id", "graph", "id", "source"):
            if key in inputs:
                graph_id = inputs.get(key)
                break
    if graph_id is None:
        for key in ("graph_id", "graph", "id", "source"):
            if key in analysis_cfg:
                graph_id = analysis_cfg.get(key)
                break
    if graph_id is None:
        graph_section = analysis_cfg.get("graph")
        if isinstance(graph_section, Mapping):
            graph_id = graph_section.get("id") or graph_section.get("graph_id")
    return _require_nonempty_str(graph_id, "graph_id")


def _coerce_positive_int(
    value: Any,
    label: str,
    *,
    default: int,
) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer.") from exc
    if number <= 0:
        raise ConfigError(f"{label} must be a positive integer.")
    return number


def _coerce_bool(value: Any, label: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise ConfigError(f"{label} must be a boolean.")


def _load_graph_payload(path: Path) -> dict[str, Any]:
    graph_path = path / "graph.json"
    if not graph_path.exists():
        raise ConfigError(f"graph.json not found in {path}.")
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"graph.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("graph.json must contain a JSON object.")
    return dict(payload)


def _extract_node_link_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if "bipartite" in payload and isinstance(payload.get("bipartite"), Mapping):
        bipartite = payload.get("bipartite")
        data = bipartite.get("data") if isinstance(bipartite, Mapping) else None
        if isinstance(data, Mapping):
            return dict(data), True
    if "nodes" in payload and ("links" in payload or "edges" in payload):
        graph_meta = payload.get("graph")
        is_bipartite = False
        if isinstance(graph_meta, Mapping) and graph_meta.get("bipartite"):
            is_bipartite = True
        return dict(payload), is_bipartite
    raise ConfigError("graph.json has no node-link graph data.")


def _node_id_from_entry(entry: Any) -> str:
    if isinstance(entry, Mapping):
        for key in ("id", "name", "key"):
            if key in entry and entry.get(key) is not None:
                return str(entry.get(key))
    if isinstance(entry, str):
        return entry
    if entry is None:
        raise ConfigError("graph node id must not be null.")
    return str(entry)


def _normalize_nodes(
    nodes_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(nodes_raw, Sequence) or isinstance(
        nodes_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph nodes must be a sequence.")
    nodes: list[dict[str, Any]] = []
    node_map: dict[str, dict[str, Any]] = {}
    for entry in nodes_raw:
        node_id = _node_id_from_entry(entry)
        if isinstance(entry, Mapping):
            node = dict(entry)
        else:
            node = {}
        node["id"] = node_id
        node_map[node_id] = node
    nodes = list(node_map.values())
    return nodes, node_map


def _coerce_node_ref(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("id") or value.get("name") or value.get("key")
    if value is None:
        return None
    return str(value)


def _normalize_links(links_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(links_raw, Sequence) or isinstance(
        links_raw, (str, bytes, bytearray)
    ):
        raise ConfigError("graph links must be a sequence.")
    links: list[dict[str, Any]] = []
    for entry in links_raw:
        if not isinstance(entry, Mapping):
            continue
        source = _coerce_node_ref(entry.get("source"))
        target = _coerce_node_ref(entry.get("target"))
        if source is None or target is None:
            continue
        link = dict(entry)
        link["source"] = source
        link["target"] = target
        links.append(link)
    return links


def _normalize_direction_mode(value: Any, *, is_bipartite: bool) -> str:
    if value is None:
        return "role" if is_bipartite else "as_is"
    if not isinstance(value, str):
        raise ConfigError("direction_mode must be a string.")
    normalized = value.strip().lower()
    if normalized in ("as_is", "asis"):
        return "as_is"
    if normalized in ("role", "bipartite_role", "stoich_role"):
        return "role"
    raise ConfigError("direction_mode must be one of: as_is, role.")


def _infer_link_role(link: Mapping[str, Any]) -> Optional[str]:
    role = link.get("role")
    if isinstance(role, str):
        role_lower = role.strip().lower()
        if role_lower in ("reactant", "product"):
            return role_lower
    stoich = link.get("stoich")
    if stoich is None:
        return None
    try:
        value = float(stoich)
    except (TypeError, ValueError):
        return None
    if value > 0:
        return "product"
    if value < 0:
        return "reactant"
    return None


def _apply_role_direction(
    source: str,
    target: str,
    *,
    link: Mapping[str, Any],
    node_kind: Mapping[str, str],
) -> tuple[str, str]:
    role = _infer_link_role(link)
    if role is None:
        return source, target
    if role == "reactant":
        desired_source = "species"
        desired_target = "reaction"
    else:
        desired_source = "reaction"
        desired_target = "species"
    source_kind = node_kind.get(source)
    target_kind = node_kind.get(target)
    if source_kind == desired_source and target_kind == desired_target:
        return source, target
    if source_kind == desired_target and target_kind == desired_source:
        return target, source
    if role == "product":
        return target, source
    return source, target


def _build_directed_edges(
    links: Sequence[Mapping[str, Any]],
    *,
    node_kind: Mapping[str, str],
    direction_mode: str,
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for link in links:
        source = _coerce_node_ref(link.get("source"))
        target = _coerce_node_ref(link.get("target"))
        if source is None or target is None:
            continue
        if direction_mode == "role":
            source, target = _apply_role_direction(
                source,
                target,
                link=link,
                node_kind=node_kind,
            )
        edges.append((source, target))
    return edges


def _build_adjacency(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    adjacency = {node: set() for node in nodes}
    reverse = {node: set() for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        reverse.setdefault(target, set()).add(source)
        adjacency.setdefault(target, set())
        reverse.setdefault(source, set())
    return adjacency, reverse


def _finish_order(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []
    for node in nodes:
        if node in visited:
            continue
        stack: list[tuple[str, bool]] = [(node, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            if current in visited:
                continue
            visited.add(current)
            stack.append((current, True))
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    stack.append((neighbor, False))
    return order


def _collect_component(
    start: str,
    adjacency: Mapping[str, set[str]],
    visited: set[str],
) -> list[str]:
    component: list[str] = []
    stack = [start]
    visited.add(start)
    while stack:
        node = stack.pop()
        component.append(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return component


def _strongly_connected_components(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
) -> list[list[str]]:
    order = _finish_order(nodes, adjacency)
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in reversed(order):
        if node in visited:
            continue
        component = _collect_component(node, reverse, visited)
        components.append(component)
    return components


def _undirected_components(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> list[list[str]]:
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in visited:
            continue
        component = _collect_component(node, adjacency, visited)
        components.append(component)
    return components


def _summarize_components(
    components: Sequence[Sequence[str]],
    *,
    max_components: int,
    max_component_nodes: int,
) -> dict[str, Any]:
    summarized: list[dict[str, Any]] = []
    sorted_components = sorted(
        components,
        key=lambda comp: (-len(comp), sorted(comp)[0] if comp else ""),
    )
    for comp in sorted_components[:max_components]:
        nodes_sorted = sorted(comp)
        truncated = len(nodes_sorted) > max_component_nodes
        summarized.append(
            {
                "size": len(comp),
                "nodes": nodes_sorted[:max_component_nodes],
                "truncated": truncated,
            }
        )
    largest = summarized[0] if summarized else {"size": 0, "nodes": [], "truncated": False}
    return {
        "count": len(components),
        "components": summarized,
        "largest": largest,
    }


def _node_label(node: Mapping[str, Any], node_id: str) -> str:
    for key in ("label", "name", "reaction_id", "species"):
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return node_id


def _degree_centrality(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
    reverse: Mapping[str, set[str]],
    *,
    node_meta: Mapping[str, Mapping[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    node_count = len(nodes)
    ranking: list[dict[str, Any]] = []
    denom = float(node_count - 1) if node_count > 1 else 0.0
    for node_id in nodes:
        out_degree = len(adjacency.get(node_id, set()))
        in_degree = len(reverse.get(node_id, set()))
        degree = in_degree + out_degree
        score = degree / denom if denom else 0.0
        meta = node_meta.get(node_id, {})
        ranking.append(
            {
                "node_id": node_id,
                "label": _node_label(meta, node_id),
                "kind": meta.get("kind", "unknown"),
                "degree": degree,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "score": score,
            }
        )
    ranking.sort(key=lambda item: (-item["score"], item["node_id"]))
    if top_n < len(ranking):
        ranking = ranking[:top_n]
    return {"count": node_count, "top_n": top_n, "ranking": ranking}


def _betweenness_centrality(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    directed: bool,
    node_meta: Mapping[str, Mapping[str, Any]],
    top_n: int,
    max_nodes: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "skipped", "reason": "disabled", "ranking": []}
    if nx is None:
        return {"status": "skipped", "reason": "networkx_unavailable", "ranking": []}
    if len(nodes) > max_nodes:
        return {
            "status": "skipped",
            "reason": "node_limit",
            "max_nodes": max_nodes,
            "ranking": [],
        }
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    scores = nx.betweenness_centrality(graph)
    ranking = []
    for node_id, score in scores.items():
        meta = node_meta.get(node_id, {})
        ranking.append(
            {
                "node_id": node_id,
                "label": _node_label(meta, node_id),
                "kind": meta.get("kind", "unknown"),
                "score": float(score),
            }
        )
    ranking.sort(key=lambda item: (-item["score"], item["node_id"]))
    if top_n < len(ranking):
        ranking = ranking[:top_n]
    return {"status": "computed", "count": len(nodes), "top_n": top_n, "ranking": ranking}


def analyze_graph(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Analyze a stored GraphArtifact and emit graph analytics."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    analysis_cfg = _extract_analysis_cfg(graph_cfg)
    graph_id = _extract_graph_id(analysis_cfg)

    top_n = _coerce_positive_int(analysis_cfg.get("top_n"), "top_n", default=25)
    max_components = _coerce_positive_int(
        analysis_cfg.get("max_components"),
        "max_components",
        default=25,
    )
    max_component_nodes = _coerce_positive_int(
        analysis_cfg.get("max_component_nodes"),
        "max_component_nodes",
        default=200,
    )
    betweenness_enabled = _coerce_bool(
        analysis_cfg.get("compute_betweenness"),
        "compute_betweenness",
        default=False,
    )
    betweenness_max_nodes = _coerce_positive_int(
        analysis_cfg.get("betweenness_max_nodes"),
        "betweenness_max_nodes",
        default=2000,
    )

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    payload = _load_graph_payload(graph_dir)
    graph_data, is_bipartite = _extract_node_link_payload(payload)
    nodes_raw = graph_data.get("nodes") or []
    links_raw = graph_data.get("links") or graph_data.get("edges") or []
    nodes, node_map = _normalize_nodes(nodes_raw)
    links = _normalize_links(links_raw)

    node_ids = [node["id"] for node in nodes]
    node_kind = {
        node_id: str(node.get("kind", "unknown"))
        for node_id, node in node_map.items()
    }
    direction_mode = _normalize_direction_mode(
        analysis_cfg.get("direction_mode"),
        is_bipartite=is_bipartite,
    )
    edges = _build_directed_edges(
        links,
        node_kind=node_kind,
        direction_mode=direction_mode,
    )
    edge_nodes = {node_id for edge in edges for node_id in edge}
    missing_nodes = sorted(edge_nodes.difference(node_map))
    for node_id in missing_nodes:
        node_map[node_id] = {"id": node_id}
    node_ids = list(node_map.keys())
    adjacency, reverse = _build_adjacency(node_ids, edges)

    scc_components = _strongly_connected_components(node_ids, adjacency, reverse)
    scc_summary = _summarize_components(
        scc_components,
        max_components=max_components,
        max_component_nodes=max_component_nodes,
    )

    undirected_adjacency: dict[str, set[str]] = {
        node: set(neighbors) for node, neighbors in adjacency.items()
    }
    for source, target in edges:
        undirected_adjacency.setdefault(source, set()).add(target)
        undirected_adjacency.setdefault(target, set()).add(source)
    communities = _undirected_components(node_ids, undirected_adjacency)
    community_summary = _summarize_components(
        communities,
        max_components=max_components,
        max_component_nodes=max_component_nodes,
    )

    degree_summary = _degree_centrality(
        node_ids,
        adjacency,
        reverse,
        node_meta=node_map,
        top_n=top_n,
    )
    betweenness_summary = _betweenness_centrality(
        node_ids,
        edges,
        directed=bool(graph_data.get("directed", True)),
        node_meta=node_map,
        top_n=top_n,
        max_nodes=betweenness_max_nodes,
        enabled=betweenness_enabled,
    )

    edge_count = sum(len(neighbors) for neighbors in adjacency.values())
    analysis_payload = {
        "source": {"graph_id": graph_id},
        "summary": {
            "node_count": len(node_ids),
            "edge_count": edge_count,
            "directed": bool(graph_data.get("directed", True)),
            "bipartite": is_bipartite,
        },
        "analysis": {
            "direction_mode": direction_mode,
            "limits": {
                "top_n": top_n,
                "max_components": max_components,
                "max_component_nodes": max_component_nodes,
                "betweenness_max_nodes": betweenness_max_nodes,
            },
            "scc": scc_summary,
            "communities": community_summary,
            "centrality": {
                "degree": degree_summary,
                "betweenness": betweenness_summary,
            },
        },
    }

    inputs_payload = {"graph_id": graph_id}
    code_meta = _code_metadata()
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / "graph.json").write_text(
            json.dumps(analysis_payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


def run_laplacian(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build and store a Laplacian matrix derived from a GraphArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    laplacian_cfg = _extract_laplacian_cfg(graph_cfg)
    graph_id = _extract_graph_id(laplacian_cfg)

    normalized = _coerce_bool(
        laplacian_cfg.get("normalized"),
        "normalized",
        default=False,
    )
    symmetrize = _coerce_bool(
        laplacian_cfg.get("symmetrize"),
        "symmetrize",
        default=True,
    )
    use_abs_weights = _coerce_bool(
        laplacian_cfg.get("use_abs_weights"),
        "use_abs_weights",
        default=True,
    )
    weight_key = _coerce_optional_str(
        laplacian_cfg.get("weight_key") or laplacian_cfg.get("weight_field"),
        "weight_key",
    )

    store.read_manifest("graphs", graph_id)
    graph_dir = store.artifact_dir("graphs", graph_id)
    graph_payload = _load_graph_payload(graph_dir)
    result = build_laplacian(
        graph_payload,
        normalized=normalized,
        weight_key=weight_key,
        use_abs_weights=use_abs_weights,
        symmetrize=symmetrize,
    )

    inputs_payload = {"graph_id": graph_id}
    code_meta = _code_metadata()
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[graph_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=code_meta,
        provenance=_provenance_metadata(),
    )

    laplacian_meta: dict[str, Any] = {
        "path": "laplacian.npz",
        "format": result.format,
        "shape": [len(result.nodes), len(result.nodes)],
        "degree_format": "vector",
        "laplacian_key": "laplacian",
        "degree_key": "degree",
        "normalized": normalized,
        "weight_key": weight_key or "auto",
        "use_abs_weights": use_abs_weights,
        "symmetrize": symmetrize,
    }
    if result.normalized_laplacian is not None:
        laplacian_meta["normalized_key"] = "laplacian_norm"

    laplacian_payload = {
        "kind": "laplacian",
        "source": {"graph_id": graph_id},
        "nodes": {"count": len(result.nodes), "order": list(result.nodes)},
        "laplacian": laplacian_meta,
    }

    def _writer(base_dir: Path) -> None:
        _write_laplacian_npz(base_dir / "laplacian.npz", result)
        (base_dir / "graph.json").write_text(
            json.dumps(laplacian_payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


def run_from_run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build a minimal bipartite GraphArtifact from a RunArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    run_id = _extract_run_id(graph_cfg)

    store.read_manifest("runs", run_id)
    run_dir = store.artifact_dir("runs", run_id)
    payload = _load_run_dataset_payload(run_dir)
    gas_species = _extract_coord_names(payload, "species")
    surface_species = _extract_coord_names(payload, "surface_species")
    if not gas_species and not surface_species:
        raise ConfigError("run dataset must include species or surface_species coords.")
    reactions = _extract_coord_names(payload, "reaction")

    graph_data = _build_run_bipartite_graph(
        gas_species=gas_species,
        surface_species=surface_species,
        reactions=reactions,
    )
    links_key = "links" if "links" in graph_data else "edges"
    graph_payload = {
        "kind": "run_bipartite",
        "source": {"run_id": run_id},
        "bipartite": {
            "format": "node_link",
            "node_count": len(graph_data.get("nodes", [])),
            "edge_count": len(graph_data.get(links_key, [])),
            "data": graph_data,
        },
    }

    inputs_payload: dict[str, Any] = {"run_id": run_id}
    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[run_id],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    def _writer(base_dir: Path) -> None:
        (base_dir / "graph.json").write_text(
            json.dumps(graph_payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Build and store a stoichiometric matrix GraphArtifact."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, graph_cfg = _extract_graph_cfg(resolved_cfg)
    mechanism, phase = _extract_mechanism(graph_cfg)

    solution = _load_solution(mechanism, phase)
    result = build_stoich(solution)
    species_annotations = annotate_species(solution, phase=phase)
    reaction_annotations = annotate_reactions(solution, phase=phase)
    bipartite_graph = build_bipartite_graph(
        result,
        species_annotations=species_annotations,
        reaction_annotations=reaction_annotations,
    )

    inputs_payload: dict[str, Any] = {"mechanism": mechanism}
    if phase is not None:
        inputs_payload["phase"] = phase

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    manifest = ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=[],
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    metadata = _graph_metadata(
        result,
        mechanism=mechanism,
        phase=phase,
        bipartite_graph=bipartite_graph,
    )

    def _writer(base_dir: Path) -> None:
        _write_stoich_npz(base_dir / "stoich.npz", result)
        (base_dir / "graph.json").write_text(
            json.dumps(metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return store.ensure(manifest, writer=_writer)


register("task", "graphs.stoich", run)
register("task", "graphs.analyze", analyze_graph)
register("task", "graphs.analytics", analyze_graph)
register("task", "graphs.laplacian", run_laplacian)
register("task", "graphs.from_run", run_from_run)

__all__ = [
    "LaplacianResult",
    "StoichResult",
    "annotate_reactions",
    "annotate_species",
    "analyze_graph",
    "build_stoich",
    "build_bipartite_graph",
    "build_laplacian",
    "run_from_run",
    "run_laplacian",
    "run",
]
