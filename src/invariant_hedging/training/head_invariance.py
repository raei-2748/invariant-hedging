"""Utilities for invariance-enforcing heads and penalties."""
from __future__ import annotations

from invariant_hedging.training.irm.configs import IRMConfig
from invariant_hedging.training.irm.head_grads import compute_env_head_grads, freeze_backbone
from invariant_hedging.training.irm.penalties import cosine_alignment_penalty, varnorm_penalty
from invariant_hedging.training.objectives.hirm_head import (
    EnvLossPayload,
    HIRMHeadConfig,
    HIRMHeadLoss,
    hirm_head_loss,
)

__all__ = [
    "EnvLossPayload",
    "HIRMHeadConfig",
    "HIRMHeadLoss",
    "IRMConfig",
    "compute_env_head_grads",
    "freeze_backbone",
    "hirm_head_loss",
    "cosine_alignment_penalty",
    "varnorm_penalty",
]
