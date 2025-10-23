"""Variance of risk diagnostic checks."""

from typing import Dict

import pytest
import torch

from src.modules.diagnostics import compute_VR


def test_vr_empty_mapping_returns_zero(float_tolerance: Dict[str, float]) -> None:
    assert compute_VR({}) == pytest.approx(0.0, **float_tolerance)


def test_vr_single_environment_returns_zero(float_tolerance: Dict[str, float]) -> None:
    env2risk = {"calm": torch.tensor([1.0, 2.0])}

    assert compute_VR(env2risk) == pytest.approx(0.0, **float_tolerance)


def test_vr_matches_population_variance(float_tolerance: Dict[str, float]) -> None:
    env2risk = {
        "calm": torch.tensor([1.0, 2.0, 3.0]),
        "stress": torch.tensor([4.0]),
        "crisis": 7.0,
    }

    expected = torch.tensor([2.0, 4.0, 7.0]).var(unbiased=False).item()
    result = compute_VR(env2risk)

    assert result == pytest.approx(expected, **float_tolerance)


def test_vr_handles_identical_risks(float_tolerance: Dict[str, float]) -> None:
    env2risk = {
        "calm": torch.tensor([5.0, 5.0]),
        "crisis": torch.tensor([5.0]),
        "stress": 5.0,
    }

    assert compute_VR(env2risk) == pytest.approx(0.0, **float_tolerance)
