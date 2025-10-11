"""Utilities for manipulating head/backbone parameters for IRM penalties."""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Sequence

import torch
from torch import nn


GradFn = Callable[[nn.Module, Any], torch.Tensor]


def _iter_modules(model: nn.Module, names: Sequence[str]) -> Iterable[nn.Module]:
    for name in names:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            yield module


def select_head_params(model: nn.Module) -> List[nn.Parameter]:
    """Return parameters associated with the decision head (ψ)."""

    preferred_attrs = ("head", "decision_head", "risk_head", "env_adapters", "head_module")
    params: List[nn.Parameter] = []
    seen: set[int] = set()

    for module in _iter_modules(model, preferred_attrs):
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))

    if params:
        return params

    excluded: set[int] = set()
    for module in _iter_modules(model, ("backbone", "representation", "encoder", "feature_extractor")):
        for param in module.parameters():
            excluded.add(id(param))

    for param in model.parameters():
        if param.requires_grad and id(param) not in excluded and id(param) not in seen:
            params.append(param)
            seen.add(id(param))

    if not params:
        raise ValueError("Unable to locate trainable head parameters on the provided model.")
    return params


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone/representation parameters (φ) in-place."""

    head_ids = {id(param) for param in select_head_params(model)}
    frozen: List[nn.Parameter] = []
    for param in model.parameters():
        if id(param) in head_ids:
            continue
        if param.requires_grad:
            param.requires_grad_(False)
            frozen.append(param)
    setattr(model, "_irm_frozen_parameters", frozen)


def unfreeze_backbone(model: nn.Module) -> None:
    """Restore backbone parameters that were frozen via :func:`freeze_backbone`."""

    frozen = getattr(model, "_irm_frozen_parameters", None)
    if not frozen:
        return
    for param in frozen:
        param.requires_grad_(True)
    setattr(model, "_irm_frozen_parameters", [])


def compute_env_head_grads(
    model: nn.Module,
    risk_fn: GradFn,
    env_batches: Iterable[Any],
    *,
    create_graph: bool = False,
) -> List[torch.Tensor]:
    """Compute flattened head gradients for each environment batch."""

    head_params = select_head_params(model)
    grad_vectors: List[torch.Tensor] = []
    for batch in env_batches:
        risk = risk_fn(model, batch)
        if not isinstance(risk, torch.Tensor):
            raise TypeError("risk_fn must return a torch.Tensor scalar loss.")
        if risk.ndim != 0:
            risk = risk.reshape(-1).mean()
        grads = torch.autograd.grad(
            risk,
            head_params,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        flattened: List[torch.Tensor] = []
        for param, grad in zip(head_params, grads):
            if grad is None:
                flattened.append(param.new_zeros(param.numel()))
            else:
                flattened.append(grad.reshape(-1))
        grad_vectors.append(torch.cat(flattened))
    return grad_vectors


__all__ = [
    "select_head_params",
    "freeze_backbone",
    "unfreeze_backbone",
    "compute_env_head_grads",
]

