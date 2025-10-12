"""Unit tests for invariance gap diagnostic."""

from typing import Dict

import pytest
import torch

from src.diagnostics.external import compute_IG


def test_ig_empty_mapping_returns_zero(float_tolerance: Dict[str, float]) -> None:
    assert compute_IG({}) == pytest.approx(0.0, **float_tolerance)


def test_ig_single_environment_returns_zero(float_tolerance: Dict[str, float]) -> None:
    env2outcome = {"calm": torch.tensor([1.0, 2.0, 3.0])}

    result = compute_IG(env2outcome)

    assert result == pytest.approx(0.0, **float_tolerance)


def test_ig_mixed_scalars_and_tensors(float_tolerance: Dict[str, float]) -> None:
    env2outcome = {
        "calm": torch.tensor([1.0, 2.0, 3.0]),
        "stress": 4.0,
        "crisis": torch.tensor([7.0]),
    }

    expected = torch.tensor([2.0, 4.0, 7.0]).std(unbiased=False).item()
    result = compute_IG(env2outcome)

    assert result == pytest.approx(expected, **float_tolerance)


def test_ig_ignores_empty_tensors(float_tolerance: Dict[str, float]) -> None:
    env2outcome = {
        "calm": torch.tensor([]),
        "crisis": torch.tensor([5.0]),
        "stress": torch.tensor([5.0]),
    }

    result = compute_IG(env2outcome)

    assert result == pytest.approx(0.0, **float_tolerance)
