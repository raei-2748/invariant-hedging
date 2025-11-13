"""Closed-form checks for distributional penalty helpers."""

from typing import Dict, Sequence

import pytest
import torch

from invariant_hedging.core.losses import compute_hirm_penalty, irm_penalty, vrex_penalty


def test_irm_penalty_matches_polynomial_reference(float_tolerance: Dict[str, float]) -> None:
    w = torch.tensor(0.5, requires_grad=True)
    dummy = torch.tensor(1.0, requires_grad=True)
    env_losses = [
        (w * dummy) ** 2,
        (2.0 * w * dummy) ** 2,
    ]

    penalty = irm_penalty(env_losses, dummy)
    penalty.backward()

    expected_value = 34.0 * (w.detach().item() ** 4)
    expected_grad = 136.0 * (w.detach().item() ** 3)

    assert penalty.item() == pytest.approx(expected_value, **float_tolerance)
    assert w.grad.item() == pytest.approx(expected_grad, **float_tolerance)


def test_vrex_penalty_variance_gradient(float_tolerance: Dict[str, float]) -> None:
    w = torch.tensor(0.8, requires_grad=True)
    env_losses = [w, 2.0 * w]

    penalty = vrex_penalty(env_losses)
    penalty.backward()

    expected_value = 0.25 * (w.detach().item() ** 2)
    expected_grad = 0.5 * w.detach().item()

    assert penalty.item() == pytest.approx(expected_value, **float_tolerance)
    assert w.grad.item() == pytest.approx(expected_grad, **float_tolerance)


def test_hirm_penalty_alignment_and_variance(float_tolerance: Dict[str, float]) -> None:
    w = torch.tensor(0.5, requires_grad=True)
    env_losses: Sequence[torch.Tensor] = [w**2, (2.0 * w) ** 2]
    head_grads = [
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([0.0, 1.0]),),
    ]

    penalty, diagnostics = compute_hirm_penalty(
        env_losses,
        head_grads,
        alignment_weight=0.5,
        variance_weight=0.2,
    )
    penalty.backward()

    w_value = w.detach().item()
    variance_raw = 2.25 * (w_value**4)
    expected_penalty = 0.5 + 0.2 * variance_raw
    expected_grad = 1.8 * (w_value**3)

    assert penalty.item() == pytest.approx(expected_penalty, **float_tolerance)
    assert w.grad.item() == pytest.approx(expected_grad, **float_tolerance)

    assert diagnostics["alignment"].item() == pytest.approx(0.5, **float_tolerance)
    assert diagnostics["variance_raw"].item() == pytest.approx(variance_raw, **float_tolerance)
    assert diagnostics["variance"].item() == pytest.approx(0.2 * variance_raw, **float_tolerance)
    assert diagnostics["cosine"].item() == pytest.approx(0.0, **float_tolerance)


def test_hirm_penalty_single_environment_degenerates(float_tolerance: Dict[str, float]) -> None:
    w = torch.tensor(1.2, requires_grad=True)
    env_losses: Sequence[torch.Tensor] = [w**2]
    head_grads = [(torch.tensor([1.0, 1.0]),)]

    penalty, diagnostics = compute_hirm_penalty(env_losses, head_grads)

    assert penalty.item() == pytest.approx(0.0, **float_tolerance)
    assert not penalty.requires_grad
    assert diagnostics["cosine"].item() == pytest.approx(1.0, **float_tolerance)
