"""Behavioral checks for diagnostics metrics."""

import pytest
import torch

from src.modules.diagnostics import compute_IG, compute_TR, compute_VR, compute_WG


def test_worst_group_matches_max():
    risks = {"env_a": torch.tensor([0.1]), "env_b": torch.tensor([0.4])}
    result = compute_WG(risks)
    assert result["supported"]
    assert result["WG"] == pytest.approx(0.4)


def test_variance_increases_with_dispersion():
    low_disp = {"env_a": torch.tensor([0.2]), "env_b": torch.tensor([0.21])}
    high_disp = {"env_a": torch.tensor([0.2]), "env_b": torch.tensor([0.5])}
    low = compute_VR(low_disp)
    high = compute_VR(high_disp)
    assert low["supported"] and high["supported"]
    assert high["VR"] > low["VR"]


def test_ig_non_negative():
    outcomes = {"env_a": torch.tensor([0.1, 0.1]), "env_b": torch.tensor([0.2, 0.3])}
    result = compute_IG(outcomes)
    assert result["supported"]
    assert result["IG"] >= 0.0


def test_turnover_non_negative_and_zero_when_constant():
    constant_positions = torch.zeros((3, 4))
    varying_positions = torch.tensor([[0.0, 0.1, 0.2, 0.3], [0.0, -0.1, -0.2, -0.3]])
    static_tr = compute_TR(constant_positions)
    varying_tr = compute_TR(varying_positions)
    assert static_tr["TR"] == pytest.approx(0.0)
    assert varying_tr["TR"] >= 0.0
