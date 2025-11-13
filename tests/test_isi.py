"""Analytic checks for the invariance spectrum components."""

import math
from typing import Dict

import pytest
import torch

from invariant_hedging.modules.diagnostics import (
    ISINormalizationConfig,
    compute_C1_global_stability,
    compute_C2_mechanistic_stability,
    compute_C3_structural_stability,
    compute_ISI,
)


def test_c1_single_environment_is_perfect(float_tolerance: Dict[str, float]) -> None:
    env2risk = {"calm": torch.tensor([0.1, 0.2, 0.3])}

    score = compute_C1_global_stability(env2risk, None)

    assert score == pytest.approx(1.0, **float_tolerance)


def test_c1_extreme_dispersion_clamps_to_zero(float_tolerance: Dict[str, float]) -> None:
    env2risk = {"calm": torch.tensor([0.0]), "crisis": torch.tensor([1.0])}
    norm_cfg = ISINormalizationConfig(c1_max_dispersion=0.25)

    score = compute_C1_global_stability(env2risk, norm_cfg)

    expected_var = torch.tensor([0.0, 1.0]).var(unbiased=False).item()
    assert math.isclose(expected_var, 0.25, rel_tol=1e-9)
    assert score == pytest.approx(0.0, **float_tolerance)


def test_c2_mechanistic_alignment_handles_opposites(float_tolerance: Dict[str, float]) -> None:
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]

    score = compute_C2_mechanistic_stability(grads)

    assert score == pytest.approx(0.0, **float_tolerance)


def test_c3_structural_stability_extreme_distance(float_tolerance: Dict[str, float]) -> None:
    env2repr = {
        "calm": torch.tensor([[0.0, 0.0]]),
        "crisis": torch.tensor([[3.0, 4.0]]),
    }
    norm_cfg = {"c3_max_distance": 5.0}

    score = compute_C3_structural_stability(env2repr, norm_cfg)

    assert score == pytest.approx(0.0, **float_tolerance)


def test_isi_weighted_average_respects_bounds(float_tolerance: Dict[str, float]) -> None:
    C1, C2, C3 = 1.0, 0.25, 0.0
    weights = (0.2, 0.3, 0.5)

    isi = compute_ISI(C1, C2, C3, weights)
    expected = 0.2 / sum(weights) * C1 + 0.3 / sum(weights) * C2

    assert isi == pytest.approx(expected, **float_tolerance)


def test_isi_rejects_invalid_weight_lengths() -> None:
    with pytest.raises(ValueError):
        compute_ISI(0.1, 0.2, 0.3, weights=(1.0, 0.5))
