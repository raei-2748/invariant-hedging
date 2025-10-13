"""Deterministic tests for the ISI invariance components."""

import pytest
import torch

from hirm.diagnostics.isi import (
    compute_C1_global_stability,
    compute_C2_mechanistic_stability,
    compute_C3_structural_stability,
    compute_ISI,
)


def test_c1_global_stability_monotonic():
    norm_cfg = {"c1_max_dispersion": 2.0, "c3_max_distance": 1.0}
    uniform_risk = {"env_a": torch.tensor([1.0, 1.0]), "env_b": torch.tensor([1.0, 1.0])}
    skewed_risk = {"env_a": torch.tensor([1.0]), "env_b": torch.tensor([2.0])}

    stable = compute_C1_global_stability(uniform_risk, norm_cfg)
    less_stable = compute_C1_global_stability(skewed_risk, norm_cfg)

    assert stable == pytest.approx(1.0)
    assert 0.0 <= less_stable < stable


def test_c2_mechanistic_alignment_cases():
    aligned = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]
    opposite = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    orthogonal = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

    assert compute_C2_mechanistic_stability(aligned) == pytest.approx(1.0)
    assert compute_C2_mechanistic_stability(opposite) == pytest.approx(0.0)
    assert compute_C2_mechanistic_stability(orthogonal) == pytest.approx(0.5)


def test_c3_structural_stability_distance():
    norm_cfg = {"c3_max_distance": 2.0}
    identical = {
        "env_a": torch.zeros((4, 2)),
        "env_b": torch.zeros((4, 2)),
    }
    separated = {
        "env_a": torch.zeros((4, 2)),
        "env_b": torch.ones((4, 2)),
    }

    stable = compute_C3_structural_stability(identical, norm_cfg)
    less_stable = compute_C3_structural_stability(separated, norm_cfg)

    assert stable == pytest.approx(1.0)
    assert 0.0 <= less_stable < stable


def test_isi_aggregation_matches_weights():
    C1, C2, C3 = 0.9, 0.6, 0.3
    weights = (0.2, 0.3, 0.5)
    isi = compute_ISI(C1, C2, C3, weights)
    expected = sum(w * c for w, c in zip(weights, (C1, C2, C3)))
    assert isi == pytest.approx(expected)

