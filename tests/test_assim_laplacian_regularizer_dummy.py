from __future__ import annotations

import copy
import json
from typing import Any

import pytest

np = pytest.importorskip("numpy")

import rxn_platform.backends.dummy  # noqa: F401
import rxn_platform.tasks.assimilation  # noqa: F401
import rxn_platform.tasks.graphs  # noqa: F401
import rxn_platform.tasks.observables  # noqa: F401

from rxn_platform.core import ArtifactManifest
from rxn_platform.registry import get, register
from rxn_platform.store import ArtifactStore
from rxn_platform.tasks.observables import Observable


class DummyMultiplierObservable(Observable):
    name = "dummy_multiplier_laplacian"

    def compute(self, run_dataset, cfg):
        multipliers = run_dataset.attrs.get("reaction_multipliers") or []
        total = 0.0
        for entry in multipliers:
            if not isinstance(entry, dict):
                continue
            weight = 1.0
            if "index" in entry:
                try:
                    weight = float(entry["index"]) + 1.0
                except (TypeError, ValueError):
                    weight = 1.0
            try:
                multiplier_value = float(entry.get("multiplier", 0.0))
            except (TypeError, ValueError):
                multiplier_value = 0.0
            total += multiplier_value * weight
        return {
            "observable": "dummy.multiplier_sum",
            "value": total,
            "unit": "1",
        }


def _read_table(path) -> list[dict[str, Any]]:
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


def _make_graph_manifest(graph_id: str) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        kind="graphs",
        id=graph_id,
        created_at="2026-01-18T00:00:00Z",
        parents=[],
        inputs={"graph": "demo"},
        config={"source": "test"},
        code={"version": "0.0.0"},
        provenance={"python": "3.11"},
    )


def _write_reaction_graph(store: ArtifactStore, graph_id: str) -> None:
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {"name": "reaction_demo"},
        "nodes": [
            {"id": "reaction_1", "kind": "reaction", "reaction_id": "R1"},
            {"id": "reaction_2", "kind": "reaction", "reaction_id": "R2"},
        ],
        "links": [
            {"source": "reaction_1", "target": "reaction_2", "weight": 1.0}
        ],
    }
    manifest = _make_graph_manifest(graph_id)

    def _writer(base_dir):
        (base_dir / "graph.json").write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    store.ensure(manifest, writer=_writer)


def _analysis_params(rows: list[dict[str, Any]], iteration: int) -> list[dict[str, float]]:
    analysis_rows = [
        row
        for row in rows
        if row.get("stage") == "analysis" and row.get("iteration") == iteration
    ]
    analysis_rows.sort(key=lambda row: row.get("sample_id", 0))
    return [json.loads(row["params_json"]) for row in analysis_rows]


def _laplacian_penalty(
    params_list: list[dict[str, float]],
    laplacian: Any,
    names: list[str],
) -> float:
    values = []
    for params in params_list:
        vec = np.array([params[name] for name in names], dtype=float)
        lp = laplacian @ vec
        values.append(float(np.sum(lp * lp)))
    return float(np.mean(values))


def test_assim_eki_laplacian_regularizer_dummy(tmp_path) -> None:
    register(
        "observable",
        DummyMultiplierObservable.name,
        DummyMultiplierObservable(),
        overwrite=True,
    )

    store = ArtifactStore(tmp_path / "artifacts")
    graph_id = "graph-reactions"
    _write_reaction_graph(store, graph_id)

    laplacian_task = get("task", "graphs.laplacian")
    laplacian_cfg = {"graphs": {"laplacian": {"graph_id": graph_id}}}
    laplacian_result = laplacian_task(laplacian_cfg, store=store)

    with np.load(laplacian_result.path / "laplacian.npz") as data:
        laplacian = data["laplacian"]

    iterations = 2
    ensemble_size = 4
    base_cfg = {
        "common": {"seed": 123},
        "sim": {
            "name": "dummy",
            "time": {"start": 0.0, "stop": 1.0, "steps": 2},
        },
        "observables": {"observables": [DummyMultiplierObservable.name]},
        "assimilation": {
            "parameters": [
                {
                    "index": 0,
                    "name": "m0",
                    "prior": {"type": "uniform", "low": 0.5, "high": 1.5},
                },
                {
                    "index": 1,
                    "name": "m1",
                    "prior": {"type": "uniform", "low": 0.5, "high": 1.5},
                },
            ],
            "observed": [
                {
                    "observable": "dummy.multiplier_sum",
                    "value": 3.0,
                    "weight": 1.0,
                    "noise": 1.0,
                }
            ],
            "params": {
                "ensemble_size": ensemble_size,
                "iterations": iterations,
                "ridge": 1.0e-6,
            },
        },
    }

    task = get("task", "assimilation.eki")
    base_result = task(base_cfg, store=store)

    cfg_lambda0 = copy.deepcopy(base_cfg)
    cfg_lambda0["assimilation"]["laplacian"] = {
        "graph_id": laplacian_result.manifest.id,
        "lambda": 0.0,
    }
    lambda0_result = task(cfg_lambda0, store=store)

    cfg_lambda = copy.deepcopy(base_cfg)
    cfg_lambda["assimilation"]["laplacian"] = {
        "graph_id": laplacian_result.manifest.id,
        "lambda": 5.0,
    }
    lambda_result = task(cfg_lambda, store=store)

    base_rows = _read_table(base_result.path / "posterior.parquet")
    lambda0_rows = _read_table(lambda0_result.path / "posterior.parquet")
    lambda_rows = _read_table(lambda_result.path / "posterior.parquet")

    base_params = _analysis_params(base_rows, iterations - 1)
    lambda0_params = _analysis_params(lambda0_rows, iterations - 1)
    lambda_params = _analysis_params(lambda_rows, iterations - 1)

    assert len(base_params) == ensemble_size
    assert len(lambda0_params) == ensemble_size
    assert len(lambda_params) == ensemble_size

    base_array = np.array([[row["m0"], row["m1"]] for row in base_params])
    lambda0_array = np.array([[row["m0"], row["m1"]] for row in lambda0_params])
    assert np.allclose(base_array, lambda0_array)

    penalty_base = _laplacian_penalty(base_params, laplacian, ["m0", "m1"])
    penalty_reg = _laplacian_penalty(lambda_params, laplacian, ["m0", "m1"])

    assert penalty_reg < penalty_base
