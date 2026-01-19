from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.features  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401
import rxn_platform.tasks.sim  # noqa: F401

from rxn_platform.core import load_config
from rxn_platform.registry import get
from rxn_platform.store import ArtifactStore


def _load_suite_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "golden" / "dummy.yaml"
    cfg = load_config(path)
    if not isinstance(cfg, Mapping):
        raise AssertionError("golden suite config must be a mapping")
    return dict(cfg)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        try:
            frame = pd.read_parquet(path)
            return frame.to_dict(orient="records")
        except Exception:
            pass
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _index_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get(key)
        if isinstance(name, str):
            indexed[name] = row
    return indexed


def _assert_expected(
    *,
    label: str,
    observed: dict[str, dict[str, Any]],
    expected_entries: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> None:
    missing: list[str] = []
    errors: list[str] = []
    for entry in expected_entries:
        if not isinstance(entry, Mapping):
            errors.append(f"{label} expectation must be a mapping")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{label} expectation name is missing")
            continue
        if name not in observed:
            missing.append(name)
            continue
        row = observed[name]
        try:
            actual = float(row.get("value"))
        except (TypeError, ValueError):
            errors.append(f"{label} {name} value is not numeric")
            continue
        try:
            expected = float(entry.get("value"))
        except (TypeError, ValueError):
            errors.append(f"{label} {name} expected value is not numeric")
            continue
        entry_abs_tol = float(entry.get("abs_tol", abs_tol))
        entry_rel_tol = float(entry.get("rel_tol", rel_tol))
        if not math.isclose(
            actual,
            expected,
            rel_tol=entry_rel_tol,
            abs_tol=entry_abs_tol,
        ):
            errors.append(
                f"{label} {name} expected {expected} got {actual} "
                f"(abs_tol={entry_abs_tol}, rel_tol={entry_rel_tol})"
            )
        expected_unit = entry.get("unit")
        if expected_unit is not None:
            actual_unit = row.get("unit")
            if actual_unit != expected_unit:
                errors.append(
                    f"{label} {name} expected unit {expected_unit} got {actual_unit}"
                )
    if missing:
        errors.append(f"missing {label} entries: {', '.join(sorted(missing))}")
    if errors:
        raise AssertionError("; ".join(errors))


def test_golden_suite_dummy(tmp_path: Path) -> None:
    suite_cfg = _load_suite_config()
    tolerances = suite_cfg.get("tolerances", {})
    if tolerances is None:
        tolerances = {}
    if not isinstance(tolerances, Mapping):
        raise AssertionError("golden suite tolerances must be a mapping")
    default_abs = float(tolerances.get("abs", 0.0))
    default_rel = float(tolerances.get("rel", 0.0))

    conditions = suite_cfg.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise AssertionError("golden suite conditions must be a non-empty list")

    sim_task = get("task", "sim.run")
    obs_task = get("task", "observables.run")
    feat_task = get("task", "features.run")

    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise AssertionError("golden suite condition must be a mapping")
        cond_id = condition.get("id")
        if not isinstance(cond_id, str) or not cond_id:
            raise AssertionError("golden suite condition id is required")

        sim_cfg = condition.get("sim")
        if not isinstance(sim_cfg, Mapping):
            raise AssertionError(f"golden suite condition {cond_id} sim is required")
        observables = condition.get("observables")
        if not isinstance(observables, list) or not observables:
            raise AssertionError(
                f"golden suite condition {cond_id} observables are required"
            )
        features = condition.get("features")
        if not isinstance(features, list) or not features:
            raise AssertionError(
                f"golden suite condition {cond_id} features are required"
            )
        expected = condition.get("expected")
        if not isinstance(expected, Mapping):
            raise AssertionError(
                f"golden suite condition {cond_id} expected values are required"
            )

        store = ArtifactStore(tmp_path / cond_id / "artifacts")
        sim_result = sim_task({"sim": dict(sim_cfg)}, store=store)
        run_id = sim_result.manifest.id

        obs_cfg = {
            "inputs": {"run_id": run_id},
            "params": {"observables": observables},
        }
        obs_result = obs_task(obs_cfg, store=store)
        obs_rows = _read_rows(obs_result.path / "values.parquet")
        obs_by_name = _index_rows(obs_rows, "observable")

        feat_cfg = {
            "inputs": {"run_id": run_id},
            "params": {"features": features},
        }
        feat_result = feat_task(feat_cfg, store=store)
        feat_rows = _read_rows(feat_result.path / "features.parquet")
        feat_by_name = _index_rows(feat_rows, "feature")

        expected_obs = expected.get("observables")
        if not isinstance(expected_obs, list) or not expected_obs:
            raise AssertionError(
                f"golden suite condition {cond_id} expected observables are required"
            )
        expected_feat = expected.get("features")
        if not isinstance(expected_feat, list) or not expected_feat:
            raise AssertionError(
                f"golden suite condition {cond_id} expected features are required"
            )

        _assert_expected(
            label=f"{cond_id} observable",
            observed=obs_by_name,
            expected_entries=expected_obs,
            abs_tol=default_abs,
            rel_tol=default_rel,
        )
        _assert_expected(
            label=f"{cond_id} feature",
            observed=feat_by_name,
            expected_entries=expected_feat,
            abs_tol=default_abs,
            rel_tol=default_rel,
        )
