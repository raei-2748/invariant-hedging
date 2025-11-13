"""Tests ensuring diagnostics operate on detached tensors only."""
from __future__ import annotations

import torch
from torch import nn

from invariant_hedging.training.losses import compute_hirm_penalty, hirm_loss
from invariant_hedging.diagnostics.robustness import detach_diagnostics, invariant_gap, safe_eval_metric, worst_group


def _build_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _generate_batch(batch_size: int = 12, *, n_envs: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    features = torch.randn(batch_size, 4)
    targets = torch.randn(batch_size, 1)
    return features, targets


def _environment_losses(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, *, n_envs: int) -> list[torch.Tensor]:
    preds = model(features)
    per_env = preds.reshape(n_envs, -1, preds.shape[-1])
    per_env_targets = targets.reshape(n_envs, -1, targets.shape[-1])
    losses: list[torch.Tensor] = []
    for idx in range(n_envs):
        loss = torch.nn.functional.mse_loss(per_env[idx], per_env_targets[idx], reduction="mean")
        losses.append(loss)
    return losses


def _collect_gradients(losses: list[torch.Tensor], params: list[torch.Tensor]) -> list[list[torch.Tensor]]:
    grad_list: list[list[torch.Tensor]] = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
        grad_list.append([grad.detach() for grad in grads])
    return grad_list


def _run_training_step(log_diagnostics: bool) -> list[torch.Tensor]:
    model = _build_model()
    features, targets = _generate_batch()
    n_envs = 3
    params = [p for p in model.parameters() if p.requires_grad]

    losses = _environment_losses(model, features, targets, n_envs=n_envs)
    total_loss = torch.stack(losses).mean()

    if log_diagnostics:
        grad_payload = _collect_gradients(losses, params)
        penalty, diag_payload = compute_hirm_penalty(losses, grad_payload)
        assert isinstance(penalty, torch.Tensor)
        detached_diag = detach_diagnostics(diag_payload)
        for value in detached_diag.values():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
        total_loss = total_loss + 0.1 * penalty.detach()

        ig_value = safe_eval_metric(invariant_gap, losses)
        wg_value = safe_eval_metric(worst_group, losses)
        assert isinstance(ig_value, float)
        assert isinstance(wg_value, float)

    model.zero_grad()
    total_loss.backward()

    grads = [param.grad.detach().clone() for param in params]
    return grads


def test_gradients_ignore_diagnostics() -> None:
    grads_without = _run_training_step(log_diagnostics=False)
    grads_with = _run_training_step(log_diagnostics=True)

    for grad_a, grad_b in zip(grads_without, grads_with):
        assert torch.allclose(grad_a, grad_b, atol=1e-6)


def test_hirm_loss_diagnostics_detached() -> None:
    env_losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
    env_grads = [[torch.tensor([1.0, -1.0])], [torch.tensor([1.0, -1.0])]]

    total_loss, diagnostics = hirm_loss(
        env_losses,
        env_grads,
        lambda_weight=0.5,
        alignment_weight=1.0,
        variance_weight=0.3,
    )

    assert isinstance(total_loss, torch.Tensor)
    for value in diagnostics.values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad
