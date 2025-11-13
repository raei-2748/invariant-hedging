import pytest
import torch

from invariant_hedging.modules.head_invariance import cosine_alignment_penalty, varnorm_penalty


def test_cosine_penalty_zero_for_identical_grads():
    g = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    penalty = cosine_alignment_penalty([g, g.clone()])
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_cosine_penalty_large_for_opposite_grads():
    g = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    penalty = cosine_alignment_penalty([g, -g])
    assert penalty.item() == pytest.approx(2.0, abs=1e-6)


def test_cosine_penalty_matches_pairwise_average():
    g1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    g3 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    penalty = cosine_alignment_penalty([g1, g2, g3])
    assert penalty.item() == pytest.approx(1.0, abs=1e-6)


def test_cosine_penalty_invariant_to_scaling():
    g1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    g2 = torch.tensor([-1.0, 4.0, 0.5], dtype=torch.float64)
    scaled = [g1 * 5.0, g2 / 7.0]
    base = cosine_alignment_penalty([g1, g2])
    scaled_penalty = cosine_alignment_penalty(scaled)
    assert scaled_penalty.item() == pytest.approx(base.item(), rel=1e-6, abs=1e-6)


def test_cosine_penalty_handles_zero_gradients():
    g1 = torch.zeros(3, dtype=torch.float64)
    g2 = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    penalty = cosine_alignment_penalty([g1, g2])
    assert not torch.isnan(penalty)
    assert penalty.item() == pytest.approx(1.0, abs=1e-6)


def test_varnorm_zero_for_identical_normalised_grads():
    g1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    g2 = 5.0 * g1
    penalty = varnorm_penalty([g1, g2])
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_varnorm_increases_with_gradient_noise():
    base = torch.tensor([0.1, -0.4, 0.7], dtype=torch.float64)
    noisy = base + torch.tensor([0.5, 0.0, -0.2], dtype=torch.float64)
    penalty = varnorm_penalty([base, noisy])
    assert penalty.item() > 0.0


def test_varnorm_invariant_to_scaling():
    g1 = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    g2 = torch.tensor([-0.5, 4.0, 0.0], dtype=torch.float64)
    scaled = [g1 * 9.0, g2 * 0.1]
    base = varnorm_penalty([g1, g2])
    scaled_penalty = varnorm_penalty(scaled)
    assert scaled_penalty.item() == pytest.approx(base.item(), rel=1e-6, abs=1e-6)


def test_varnorm_handles_zero_gradients():
    g1 = torch.zeros(4, dtype=torch.float64)
    g2 = torch.ones(4, dtype=torch.float64)
    penalty = varnorm_penalty([g1, g2])
    assert not torch.isnan(penalty)
    assert penalty.item() >= 0.0
