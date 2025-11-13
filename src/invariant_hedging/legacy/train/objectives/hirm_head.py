"""Head-only gradient alignment objective for HIRM."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from invariant_hedging.modules.models import Policy


GradClosure = Callable[[bool], torch.Tensor]


@dataclass
class EnvLossPayload:
    """Container holding per-environment losses and gradient closures."""

    name: str
    loss: torch.Tensor
    grad_closure: GradClosure | None = None


@dataclass
class HIRMHeadConfig:
    lambda_align: float = 1.0
    normalize_grad: bool = True
    pairwise_mode: str = "cosine"
    detach_features: bool = False
    eps: float = 1e-12


@dataclass
class HIRMHeadLoss:
    total: torch.Tensor
    avg_risk: torch.Tensor
    penalty: torch.Tensor
    pairwise: List[torch.Tensor]
    var_total: torch.Tensor | None
    gradients: List[torch.Tensor]


def _flatten_gradients(
    params: Sequence[nn.Parameter],
    grads: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(param.new_zeros(param.numel()))
        else:
            pieces.append(grad.reshape(-1))
    if not pieces:
        raise ValueError("No gradient components provided for flattening.")
    return torch.cat(pieces)


def _normalise(vec: torch.Tensor, eps: float) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if norm <= eps:
        return torch.zeros_like(vec)
    return vec / norm


def _pairwise_cosine(stacked: torch.Tensor, eps: float) -> torch.Tensor:
    if stacked.ndim != 2:
        raise ValueError("Expected gradients stacked along dim=0.")
    if stacked.shape[0] < 2:
        return stacked.new_zeros(0)
    normalised = F.normalize(stacked, p=2, dim=1, eps=eps)
    cosines: List[torch.Tensor] = []
    for i, j in combinations(range(normalised.shape[0]), 2):
        value = torch.dot(normalised[i], normalised[j])
        cosines.append(value.clamp(min=-1.0, max=1.0))
    return torch.stack(cosines)


def hirm_head_loss(
    model: Policy,
    env_payloads: Sequence[EnvLossPayload],
    config: HIRMHeadConfig,
) -> HIRMHeadLoss:
    """Compute the HIRM head-only objective for the provided environments."""

    if not env_payloads:
        raise ValueError("At least one environment payload is required.")

    psi_params = model.psi_params()
    if not psi_params:
        raise ValueError("Model exposes no trainable head (psi) parameters.")

    losses = [payload.loss.reshape(-1).mean() for payload in env_payloads]
    loss_tensor = torch.stack(losses)
    avg_risk = loss_tensor.mean()

    penalty = avg_risk.new_zeros(())
    gradients: List[torch.Tensor] = []
    pairwise: List[torch.Tensor] = []
    var_total: torch.Tensor | None = None

    should_compute = (
        config.lambda_align > 0.0 and len(env_payloads) >= 2 and loss_tensor.requires_grad
    )

    if should_compute:
        for payload in env_payloads:
            closure = payload.grad_closure
            if closure is not None:
                loss_for_grad = closure(config.detach_features)
            else:
                loss_for_grad = payload.loss
            grads = torch.autograd.grad(
                loss_for_grad,
                psi_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            vector = _flatten_gradients(psi_params, grads)
            if config.normalize_grad:
                vector = _normalise(vector, config.eps)
            gradients.append(vector)

        if gradients:
            stacked = torch.stack(gradients, dim=0)
            mode = config.pairwise_mode.lower()
            if mode == "cosine":
                pairwise_tensor = _pairwise_cosine(stacked, config.eps)
                if pairwise_tensor.numel() == 0:
                    penalty = stacked.new_zeros(())
                else:
                    penalty = 1.0 - pairwise_tensor.mean()
                pairwise = [c for c in pairwise_tensor]
            elif mode == "var":
                if stacked.shape[0] < 2:
                    penalty = stacked.new_zeros(())
                    var_total = penalty
                else:
                    mean = stacked.mean(dim=0, keepdim=True)
                    centred = stacked - mean
                    cov = centred.T @ centred / stacked.shape[0]
                    var_total = torch.trace(cov)
                    penalty = var_total
            else:
                raise ValueError(f"Unsupported pairwise mode '{config.pairwise_mode}'.")

    total = avg_risk + config.lambda_align * penalty
    return HIRMHeadLoss(
        total=total,
        avg_risk=avg_risk,
        penalty=penalty,
        pairwise=pairwise,
        var_total=var_total,
        gradients=gradients,
    )
