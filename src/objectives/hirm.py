"""HIRM objective built on gradient alignment."""
from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence, Tuple

import torch

from .grad_align import env_variance, normalized_head_grads, pairwise_cosine


def compute_hirm_penalty(
    env_losses: Sequence[torch.Tensor],
    head_grads: Sequence[Sequence[torch.Tensor]],
    *,
    alignment_weight: float = 1.0,
    variance_weight: float = 0.0,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Return the combined HIRM penalty and diagnostic terms."""

    norm_grads = normalized_head_grads(head_grads)
    cosines = pairwise_cosine(norm_grads)
    if cosines.numel() == 0:
        mean_cosine = norm_grads.new_tensor(1.0)
    else:
        mean_cosine = cosines.mean()
    alignment = alignment_weight * (1.0 - mean_cosine)

    variance_raw = env_variance(env_losses)
    variance = variance_weight * variance_raw

    penalty = alignment + variance
    diagnostics: MutableMapping[str, torch.Tensor] = {
        "alignment": alignment,
        "variance": variance,
        "cosine": mean_cosine,
        "variance_raw": variance_raw,
    }
    return penalty, diagnostics


def hirm_loss(
    env_losses: Sequence[torch.Tensor],
    head_grads: Sequence[Sequence[torch.Tensor]],
    *,
    lambda_weight: float,
    alignment_weight: float = 1.0,
    variance_weight: float = 0.0,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Compose the per-environment losses with the HIRM penalty."""

    if not env_losses:
        raise ValueError("HIRM loss requires at least one environment loss.")

    loss_tensor = torch.stack([loss.reshape(-1).mean() for loss in env_losses])
    base_loss = loss_tensor.mean()

    penalty, diagnostics = compute_hirm_penalty(
        env_losses,
        head_grads,
        alignment_weight=alignment_weight,
        variance_weight=variance_weight,
    )
    total_loss = base_loss + lambda_weight * penalty

    diagnostics = {
        **diagnostics,
        "base": base_loss,
        "penalty": penalty,
        "lambda": base_loss.new_tensor(lambda_weight),
    }
    return total_loss, diagnostics


__all__ = ["compute_hirm_penalty", "hirm_loss"]
