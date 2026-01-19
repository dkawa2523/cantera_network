"""Simulation-based inference (SBI) task with optional dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import json
import logging
import math
import platform
from pathlib import Path
import subprocess
from typing import Any, Optional

from rxn_platform import __version__
from rxn_platform.core import ArtifactManifest, make_artifact_id, normalize_reaction_multipliers
from rxn_platform.errors import ConfigError
from rxn_platform.hydra_utils import resolve_config
from rxn_platform.pipelines import PipelineRunner
from rxn_platform.registry import Registry, register
from rxn_platform.store import ArtifactCacheResult, ArtifactStore

try:  # Optional dependency.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:  # Optional dependency.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None

try:  # Optional dependency.
    import torch
    from sbi import utils as sbi_utils
    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    sbi_utils = None
    SNPE = None
    prepare_for_sbi = None
    simulate_for_sbi = None

DEFAULT_NUM_SIMULATIONS = 4
DEFAULT_MAX_EPOCHS = 1
DEFAULT_POSTERIOR_SAMPLES = 4
DEFAULT_MISSING_STRATEGY = "zero"
DEFAULT_METHOD = "snpe"
SUPPORTED_METHODS = ("snpe",)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    key: Any
    low: float
    high: float

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "low": self.low, "high": self.high}
        if self.kind == "reaction_id":
            payload["reaction_id"] = self.key
        elif self.kind == "index":
            payload["index"] = self.key
        else:
            payload["path"] = self.key
        return payload


def _utc_now_iso() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


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


def _resolve_cfg(cfg: Any) -> dict[str, Any]:
    try:
        resolved = resolve_config(cfg)
    except ConfigError:
        if isinstance(cfg, Mapping):
            return dict(cfg)
        raise
    return resolved


def _extract_sbi_cfg(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "sbi" in cfg:
        sbi_cfg = cfg.get("sbi")
        if not isinstance(sbi_cfg, Mapping):
            raise ConfigError("sbi config must be a mapping.")
        return dict(cfg), dict(sbi_cfg)
    return dict(cfg), dict(cfg)


def _extract_params(sbi_cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = sbi_cfg.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("sbi.params must be a mapping.")
    return dict(params)


def _extract_inputs(sbi_cfg: Mapping[str, Any]) -> dict[str, Any]:
    inputs = sbi_cfg.get("inputs", {})
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise ConfigError("sbi.inputs must be a mapping.")
    return dict(inputs)


def _lookup_path(cfg: Mapping[str, Any], path: str) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _extract_seed(cfg: Mapping[str, Any]) -> int:
    for path in ("common.seed", "seed", "sbi.seed"):
        value = _lookup_path(cfg, path)
        if value is None:
            continue
        if isinstance(value, bool):
            raise ConfigError("seed must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("seed must be an integer.") from exc
    return 0


def _extract_sim_cfg(
    resolved_cfg: Mapping[str, Any],
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    sim_cfg: Any = None
    for source in (inputs, params, sbi_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "sim" in source:
            sim_cfg = source.get("sim")
            break
        nested = source.get("inputs")
        if isinstance(nested, Mapping) and "sim" in nested:
            sim_cfg = nested.get("sim")
            break
    if not isinstance(sim_cfg, Mapping):
        raise ConfigError("sbi sim config must be provided as a mapping.")
    return dict(sim_cfg)


def _normalize_features_cfg(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        if "params" in raw:
            params = raw.get("params")
            if not isinstance(params, Mapping):
                raise ConfigError("features.params must be a mapping.")
            config = dict(params)
        else:
            config = dict(raw)
        config.pop("inputs", None)
        if "features" not in config and "features" in raw:
            config["features"] = raw.get("features")
    else:
        config = {"features": raw}
    return config


def _extract_features_cfg(
    resolved_cfg: Mapping[str, Any],
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    raw: Any = None
    for source in (inputs, params, sbi_cfg, resolved_cfg):
        if not isinstance(source, Mapping):
            continue
        if "features" in source:
            raw = source.get("features")
            break
        if "feature" in source:
            raw = source.get("feature")
            break
        nested = source.get("inputs")
        if isinstance(nested, Mapping) and "features" in nested:
            raw = nested.get("features")
            break
    if raw is None:
        raise ConfigError("sbi features config must be provided.")
    return _normalize_features_cfg(raw)


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value


def _coerce_optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer.") from exc


def _coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a float.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a float.") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{label} must be finite.")
    return number


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


def _coerce_float_sequence(value: Any, label: str) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ConfigError(f"{label} must not be empty.")
        values: list[float] = []
        for index, entry in enumerate(value):
            values.append(_coerce_float(entry, f"{label}[{index}]"))
        return values
    raise ConfigError(f"{label} must be a sequence of floats.")


def _extract_bounds(entry: Mapping[str, Any], label: str) -> tuple[float, float]:
    prior = entry.get("prior")
    low: Any = None
    high: Any = None
    if isinstance(prior, Mapping):
        low = prior.get("low", prior.get("min"))
        high = prior.get("high", prior.get("max"))
    if low is None:
        low = entry.get("low", entry.get("min"))
    if high is None:
        high = entry.get("high", entry.get("max"))
    if low is None or high is None:
        raise ConfigError(f"{label} uniform prior requires low/high.")
    low_val = _coerce_float(low, f"{label}.low")
    high_val = _coerce_float(high, f"{label}.high")
    if low_val >= high_val:
        raise ConfigError(f"{label} requires low < high.")
    return low_val, high_val


def _parse_parameter_specs(raw: Any) -> list[ParameterSpec]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ConfigError("parameters must be a sequence of mappings.")
    specs: list[ParameterSpec] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"parameters[{index}] must be a mapping.")
        reaction_id = entry.get("reaction_id") or entry.get("reaction")
        idx = _coerce_optional_int(entry.get("index"), f"parameters[{index}].index")
        path = entry.get("path")
        if reaction_id is not None:
            reaction_id = _require_nonempty_str(
                reaction_id, f"parameters[{index}].reaction_id"
            )
        if path is not None:
            path = _require_nonempty_str(path, f"parameters[{index}].path")
        kinds = [value is not None for value in (reaction_id, idx, path)]
        if sum(1 for flag in kinds if flag) != 1:
            raise ConfigError(
                f"parameters[{index}] must define exactly one of reaction_id, index, or path."
            )
        low, high = _extract_bounds(entry, f"parameters[{index}]")
        if reaction_id is not None:
            kind = "reaction_id"
            key = reaction_id
            default_name = f"reaction_id:{reaction_id}"
        elif idx is not None:
            kind = "index"
            key = idx
            default_name = f"index:{idx}"
        else:
            kind = "path"
            key = path
            default_name = f"path:{path}"
        name = entry.get("name") or default_name
        name = _require_nonempty_str(name, f"parameters[{index}].name")
        specs.append(
            ParameterSpec(name=name, kind=kind, key=key, low=low, high=high)
        )
    if not specs:
        raise ConfigError("parameters must not be empty.")
    return specs


def _extract_parameter_specs(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[ParameterSpec]:
    raw: Any = None
    for source in (params, sbi_cfg):
        if "parameters" in source:
            raw = source.get("parameters")
            break
        if "parameter" in source:
            raw = source.get("parameter")
            break
    if raw is None:
        raise ConfigError("sbi parameters must be provided.")
    return _parse_parameter_specs(raw)


def _extract_method(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    method = None
    for source in (params, sbi_cfg):
        if "method" in source:
            method = source.get("method")
            break
        if "inference" in source:
            method = source.get("inference")
            break
    if method is None:
        return DEFAULT_METHOD
    if not isinstance(method, str):
        raise ConfigError("sbi.method must be a string.")
    key = method.strip().lower()
    if key not in SUPPORTED_METHODS:
        raise ConfigError(f"sbi.method must be one of {SUPPORTED_METHODS}.")
    return key


def _extract_positive_int(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    keys: Sequence[str],
    label: str,
    default: int,
    allow_zero: bool = False,
) -> int:
    value = None
    for source in (params, sbi_cfg):
        for key in keys:
            if key in source:
                value = source.get(key)
                break
        if value is not None:
            break
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer.") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ConfigError(f"{label} must be positive.")
    return number


def _extract_num_simulations(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int:
    return _extract_positive_int(
        sbi_cfg,
        params,
        keys=("num_simulations", "n_simulations", "num_sims", "simulations"),
        label="num_simulations",
        default=DEFAULT_NUM_SIMULATIONS,
        allow_zero=False,
    )


def _extract_max_epochs(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int:
    return _extract_positive_int(
        sbi_cfg,
        params,
        keys=("max_epochs", "max_num_epochs", "epochs"),
        label="max_epochs",
        default=DEFAULT_MAX_EPOCHS,
        allow_zero=False,
    )


def _extract_posterior_samples(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int:
    return _extract_positive_int(
        sbi_cfg,
        params,
        keys=("posterior_samples", "num_posterior_samples", "samples"),
        label="posterior_samples",
        default=DEFAULT_POSTERIOR_SAMPLES,
        allow_zero=True,
    )


def _extract_feature_names(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Optional[list[str]]:
    raw: Any = None
    for source in (params, sbi_cfg):
        for key in ("feature_names", "summary_features", "summary_names"):
            if key in source:
                raw = source.get(key)
                break
        if raw is not None:
            break
    if raw is None:
        return None
    names = _coerce_str_sequence(raw, "feature_names")
    if not names:
        raise ConfigError("feature_names must not be empty.")
    return names


def _extract_missing_strategy(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
) -> str:
    value = None
    for source in (params, sbi_cfg):
        if "missing_strategy" in source:
            value = source.get("missing_strategy")
            break
    if value is None:
        return DEFAULT_MISSING_STRATEGY
    if not isinstance(value, str):
        raise ConfigError("missing_strategy must be a string.")
    key = value.strip().lower()
    if key not in {"zero", "nan", "error"}:
        raise ConfigError("missing_strategy must be zero, nan, or error.")
    return key


def _extract_observed_spec(
    sbi_cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> tuple[Optional[str], Any]:
    for source in (params, inputs, sbi_cfg):
        if not isinstance(source, Mapping):
            continue
        if "observed" in source:
            return "values", source.get("observed")
        if "observed_features_id" in source:
            return "features_id", source.get("observed_features_id")
    return None, None


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


def _normalize_sim_cfg(sim_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, Any], float]]:
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
    return normalized, _build_multiplier_map(multipliers)


def _apply_path_value(payload: dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        if not isinstance(node, Mapping):
            raise ConfigError(f"path {path!r} conflicts with existing value.")
        if not isinstance(node, dict):
            node = dict(node)
            current[part] = node
        current = node
    current[parts[-1]] = value


def _apply_parameter_values(
    base_sim_cfg: Mapping[str, Any],
    base_multipliers: Mapping[tuple[str, Any], float],
    specs: Sequence[ParameterSpec],
    values: Sequence[float],
) -> dict[str, Any]:
    if len(values) != len(specs):
        raise ConfigError("parameter values must match parameter specs length.")
    sim_cfg = copy.deepcopy(base_sim_cfg)
    multipliers = dict(base_multipliers)
    for spec, value in zip(specs, values):
        if spec.kind in {"reaction_id", "index"}:
            multipliers[(spec.kind, spec.key)] = _coerce_float(
                value, f"value[{spec.name}]"
            )
        else:
            _apply_path_value(
                sim_cfg,
                spec.key,
                _coerce_float(value, f"value[{spec.name}]"),
            )
    if multipliers:
        sim_cfg["reaction_multipliers"] = _rebuild_multipliers(multipliers)
        sim_cfg.pop("disabled_reactions", None)
    return sim_cfg


def _run_sim_and_features(
    runner: PipelineRunner,
    sim_cfg: Mapping[str, Any],
    features_cfg: Mapping[str, Any],
) -> str:
    pipeline_cfg = {
        "steps": [
            {"id": "sim", "task": "sim.run", "sim": dict(sim_cfg)},
            {
                "id": "features",
                "task": "features.run",
                "inputs": {"run_id": "@sim"},
                "params": dict(features_cfg),
            },
        ]
    }
    results = runner.run(pipeline_cfg)
    return results["features"]


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


def _load_feature_rows(store: ArtifactStore, features_id: str) -> list[dict[str, Any]]:
    store.read_manifest("features", features_id)
    table_path = store.artifact_dir("features", features_id) / "features.parquet"
    return _read_table_rows(table_path)


def _summarize_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Optional[Sequence[str]],
    missing_strategy: str,
) -> tuple[list[str], list[float]]:
    values_by_name: dict[str, float] = {}
    for row in rows:
        name = row.get("feature")
        if not isinstance(name, str) or not name.strip():
            continue
        value = row.get("value")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        values_by_name[name] = number
    if feature_names is None:
        if not values_by_name:
            raise ConfigError("FeatureArtifact contains no usable features.")
        feature_names = sorted(values_by_name.keys())
    else:
        feature_names = list(feature_names)
    if not feature_names:
        raise ConfigError("feature_names must not be empty.")
    missing: list[str] = []
    values: list[float] = []
    for name in feature_names:
        value = values_by_name.get(name, math.nan)
        if not math.isfinite(value):
            missing.append(name)
            if missing_strategy == "zero":
                value = 0.0
            elif missing_strategy == "nan":
                value = math.nan
        values.append(float(value))
    if missing and missing_strategy == "error":
        raise ConfigError(
            f"Missing or non-finite summary features: {', '.join(missing)}"
        )
    return list(feature_names), values


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    cfg: Mapping[str, Any],
    *,
    store: ArtifactStore,
    registry: Optional[Registry] = None,
) -> ArtifactCacheResult:
    """Run a minimal SBI workflow using FeatureArtifact summaries."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("cfg must be a mapping.")

    resolved_cfg = _resolve_cfg(cfg)
    manifest_cfg, sbi_cfg = _extract_sbi_cfg(resolved_cfg)
    params = _extract_params(sbi_cfg)
    inputs = _extract_inputs(sbi_cfg)
    seed = _extract_seed(resolved_cfg)
    logger = logging.getLogger("rxn_platform.sbi")

    if sbi_utils is None or SNPE is None or torch is None:
        inputs_payload = {"status": "skipped", "reason": "sbi_not_installed"}
        artifact_id = make_artifact_id(
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            exclude_keys=("hydra",),
        )
        manifest = ArtifactManifest(
            schema_version=1,
            kind="sbi",
            id=artifact_id,
            created_at=_utc_now_iso(),
            parents=[],
            inputs=inputs_payload,
            config=manifest_cfg,
            code=_code_metadata(),
            provenance=_provenance_metadata(),
            notes="skipped: sbi not installed",
        )

        def _writer(base_dir: Path) -> None:
            _write_result(
                base_dir / "sbi_result.json",
                {
                    "status": "skipped",
                    "reason": "sbi_not_installed",
                    "todo": "Install sbi to enable SNPE runs.",
                },
            )

        return store.ensure(manifest, writer=_writer)

    sim_cfg = _extract_sim_cfg(resolved_cfg, sbi_cfg, params, inputs)
    features_cfg = _extract_features_cfg(resolved_cfg, sbi_cfg, params, inputs)
    parameter_specs = _extract_parameter_specs(sbi_cfg, params)
    method = _extract_method(sbi_cfg, params)
    num_simulations = _extract_num_simulations(sbi_cfg, params)
    max_epochs = _extract_max_epochs(sbi_cfg, params)
    posterior_samples = _extract_posterior_samples(sbi_cfg, params)
    missing_strategy = _extract_missing_strategy(sbi_cfg, params)
    feature_names = _extract_feature_names(sbi_cfg, params)
    observed_kind, observed_value = _extract_observed_spec(sbi_cfg, params, inputs)

    torch.manual_seed(seed)

    low = torch.tensor([spec.low for spec in parameter_specs], dtype=torch.float32)
    high = torch.tensor([spec.high for spec in parameter_specs], dtype=torch.float32)
    prior = sbi_utils.BoxUniform(low=low, high=high)

    runner = PipelineRunner(store=store, registry=registry, logger=logger)
    base_sim_cfg, base_multipliers = _normalize_sim_cfg(sim_cfg)
    summary_state: dict[str, Any] = {"feature_names": feature_names}
    feature_ids: list[str] = []

    def _simulator(theta: "torch.Tensor") -> "torch.Tensor":
        if theta.dim() == 1:
            batch = [theta]
            squeeze = True
        else:
            batch = list(theta)
            squeeze = False
        outputs: list[list[float]] = []
        for row in batch:
            values = row.detach().cpu().tolist()
            sim_instance = _apply_parameter_values(
                base_sim_cfg,
                base_multipliers,
                parameter_specs,
                values,
            )
            features_id = _run_sim_and_features(runner, sim_instance, features_cfg)
            feature_ids.append(features_id)
            rows = _load_feature_rows(store, features_id)
            names, summary = _summarize_features(
                rows,
                feature_names=summary_state.get("feature_names"),
                missing_strategy=missing_strategy,
            )
            summary_state["feature_names"] = names
            outputs.append(summary)
        tensor = torch.tensor(outputs, dtype=torch.float32)
        if squeeze:
            return tensor[0]
        return tensor

    simulator, prior = prepare_for_sbi(_simulator, prior)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)

    inference = SNPE(prior=prior)
    density_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=max_epochs
    )
    posterior = inference.build_posterior(density_estimator)

    theta_list = theta.detach().cpu().tolist()
    x_list = x.detach().cpu().tolist()

    observed_values: Optional[list[float]] = None
    if observed_kind == "values":
        observed_values = _coerce_float_sequence(observed_value, "observed")
    elif observed_kind == "features_id":
        if not isinstance(observed_value, str) or not observed_value.strip():
            raise ConfigError("observed_features_id must be a non-empty string.")
        rows = _load_feature_rows(store, observed_value)
        _, observed_values = _summarize_features(
            rows,
            feature_names=summary_state.get("feature_names"),
            missing_strategy=missing_strategy,
        )

    if observed_values is None and posterior_samples > 0:
        if not x_list:
            raise ConfigError("No simulation outputs available for posterior sampling.")
        observed_values = list(x_list[0])

    posterior_samples_list: Optional[list[list[float]]] = None
    if posterior_samples > 0 and observed_values is not None:
        if len(observed_values) != len(summary_state.get("feature_names") or []):
            raise ConfigError("observed feature length does not match summary dimension.")
        observed_tensor = torch.tensor(observed_values, dtype=torch.float32)
        samples = posterior.sample((posterior_samples,), x=observed_tensor)
        posterior_samples_list = samples.detach().cpu().tolist()

    feature_names_final = summary_state.get("feature_names") or []
    inputs_payload: dict[str, Any] = {
        "parameters": [spec.to_dict() for spec in parameter_specs],
        "num_simulations": num_simulations,
        "max_epochs": max_epochs,
        "posterior_samples": posterior_samples,
        "method": method,
        "seed": seed,
        "feature_names": feature_names_final,
    }
    if observed_kind == "features_id" and isinstance(observed_value, str):
        inputs_payload["observed_features_id"] = observed_value
    if observed_kind == "values" and observed_values is not None:
        inputs_payload["observed"] = list(observed_values)

    artifact_id = make_artifact_id(
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        exclude_keys=("hydra",),
    )
    parents = _dedupe_preserve(feature_ids)
    if observed_kind == "features_id" and isinstance(observed_value, str):
        parents.append(observed_value)
    manifest = ArtifactManifest(
        schema_version=1,
        kind="sbi",
        id=artifact_id,
        created_at=_utc_now_iso(),
        parents=_dedupe_preserve(parents),
        inputs=inputs_payload,
        config=manifest_cfg,
        code=_code_metadata(),
        provenance=_provenance_metadata(),
    )

    result_payload: dict[str, Any] = {
        "status": "trained",
        "method": method,
        "num_simulations": num_simulations,
        "num_parameters": len(parameter_specs),
        "num_features": len(feature_names_final),
        "seed": seed,
        "parameter_specs": [spec.to_dict() for spec in parameter_specs],
        "feature_names": feature_names_final,
        "theta": theta_list,
        "x": x_list,
    }
    if observed_values is not None:
        result_payload["observed"] = observed_values
    if posterior_samples_list is not None:
        result_payload["posterior_samples"] = posterior_samples_list

    def _writer(base_dir: Path) -> None:
        _write_result(base_dir / "sbi_result.json", result_payload)

    return store.ensure(manifest, writer=_writer)


register("task", "sbi.run", run)

__all__ = ["ParameterSpec", "run"]
