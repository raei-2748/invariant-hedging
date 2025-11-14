"""Loss and penalty utilities used by the invariant hedging engine."""
from __future__ import annotations

from invariant_hedging.training.objectives.cvar import CVaRResult, bootstrap_cvar_ci, cvar_from_pnl, cvar_loss, differentiable_cvar
from invariant_hedging.training.objectives.grad_align import env_variance, normalized_head_grads, pairwise_cosine
from invariant_hedging.training.objectives.hirm import compute_hirm_penalty, hirm_loss
from invariant_hedging.training.objectives.penalties import (
    groupdro_objective,
    irm_penalty,
    update_groupdro_weights,
    vrex_penalty,
)

__all__ = [
    "CVaRResult",
    "bootstrap_cvar_ci",
    "cvar_from_pnl",
    "cvar_loss",
    "differentiable_cvar",
    "env_variance",
    "normalized_head_grads",
    "pairwise_cosine",
    "compute_hirm_penalty",
    "hirm_loss",
    "groupdro_objective",
    "irm_penalty",
    "update_groupdro_weights",
    "vrex_penalty",
]
