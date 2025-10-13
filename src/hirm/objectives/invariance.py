"""Invariance penalties specialized for head-only IRM training."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch import nn

HeadLossFn = Callable[[nn.Module, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]


def collect_head_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Collect learnable parameters belonging to a head module."""

    params: List[nn.Parameter] = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Head module has no trainable parameters for IRM penalty.")
    return params


def irm_penalty(
    head_module: nn.Module,
    env_features: Iterable[torch.Tensor],
    env_targets: Optional[Iterable[Optional[torch.Tensor]]],
    loss_fn: HeadLossFn,
    *,
    scope: str = "head",
) -> torch.Tensor:
    """Compute an IRMv1-style penalty focusing on head parameters.

    Parameters
    ----------
    head_module:
        Module whose parameters should be penalized.
    env_features:
        Iterable of tensors (one per environment) fed into ``loss_fn``.
    env_targets:
        Iterable of optional targets associated with each environment. Can be ``None``.
    loss_fn:
        Callable returning a scalar loss for a single environment. It must retain
        gradients with respect to ``head_module`` parameters.
    scope:
        Either ``"head"`` or ``"full"``. ``"head"`` restricts gradients to the
        head parameters. ``"full"`` allows gradients to flow through the entire
        module hierarchy (useful for ablations).
    """

    if scope not in {"head", "full"}:
        raise ValueError(f"Unsupported IRM scope: {scope}")

    features_list = list(env_features)
    targets_list: List[Optional[torch.Tensor]]
    if env_targets is None:
        targets_list = [None for _ in features_list]
    else:
        targets_list = list(env_targets)
        if len(targets_list) != len(features_list):
            raise ValueError("env_features and env_targets must have the same length")

    penalties: List[torch.Tensor] = []
    for features, target in zip(features_list, targets_list):
        dummy = torch.tensor(1.0, device=features.device, requires_grad=True)
        env_loss = loss_fn(head_module, features, target, dummy)
        grad = torch.autograd.grad(env_loss, dummy, create_graph=True)[0]
        penalties.append(grad.pow(2))

    if not penalties:
        try:
            device = next(head_module.parameters()).device
        except StopIteration:
            device = features_list[0].device if features_list else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(penalties).mean()


__all__ = ["irm_penalty", "collect_head_parameters"]
