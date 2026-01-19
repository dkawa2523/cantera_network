from __future__ import annotations

import math

import pytest

from rxn_platform.tasks.assimilation import (
    LogNormalPrior,
    MisfitColumns,
    ParameterSpec,
    ParameterVector,
    UniformPrior,
    compute_misfit,
)


def test_parameter_sampling_reproducible() -> None:
    params = ParameterVector(
        (
            ParameterSpec(
                name="multiplier:index:0",
                key=("index", 0),
                prior=UniformPrior(low=0.5, high=1.5),
            ),
            ParameterSpec(
                name="multiplier:rxn:R1",
                key=("reaction_id", "R1"),
                prior=LogNormalPrior(mean=0.0, sigma=0.2, low=0.8, high=1.2),
            ),
        )
    )

    samples_a = params.sample_ensemble(3, seed=123)
    samples_b = params.sample_ensemble(3, seed=123)

    assert samples_a == samples_b
    for sample in samples_a:
        assert len(sample) == 2
        assert 0.5 <= sample[0] <= 1.5
        assert 0.8 <= sample[1] <= 1.2


def test_misfit_vector_and_scalar() -> None:
    predicted = [
        {"observable": "obs.a", "value": 2.0},
        {"observable": "obs.b", "value": 4.0},
    ]
    observed = [
        {"observable": "obs.a", "value": 1.0, "weight": 2.0, "noise": 1.0},
        {"observable": "obs.b", "value": 1.0, "weight": 1.0, "noise": 2.0},
    ]

    result = compute_misfit(observed, predicted)

    assert result.vector == [2.0, 1.5]
    assert result.scalar == pytest.approx(6.25)
    assert math.isfinite(result.scalar)


def test_misfit_column_override_and_aggregate() -> None:
    predicted = [
        {"name": "obs.c", "pred": 1.0},
        {"name": "obs.c", "pred": 3.0},
    ]
    observed = [
        {
            "name": "obs.c",
            "obs": 1.0,
            "sigma": 1.0,
            "w": 1.0,
            "aggregate": "mean",
        }
    ]

    columns = MisfitColumns(
        observed_target="name",
        observed_value="obs",
        observed_weight="w",
        observed_noise="sigma",
        observed_aggregate="aggregate",
        predicted_target="name",
        predicted_value="pred",
    )
    result = compute_misfit(observed, predicted, columns=columns)

    assert result.vector == [1.0]
    assert result.scalar == pytest.approx(1.0)
