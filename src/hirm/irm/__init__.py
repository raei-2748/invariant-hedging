"""Head-only IRM utilities."""

from .configs import IRMConfig, IRMLoggingConfig
from .head_grads import compute_env_head_grads, freeze_backbone, select_head_params, unfreeze_backbone
from .penalties import cosine_alignment_penalty, varnorm_penalty

__all__ = [
    "IRMConfig",
    "IRMLoggingConfig",
    "compute_env_head_grads",
    "freeze_backbone",
    "select_head_params",
    "unfreeze_backbone",
    "cosine_alignment_penalty",
    "varnorm_penalty",
]
