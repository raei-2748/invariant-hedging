"""Head invariant risk minimisation (HIRM) training objective."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..diagnostics import metrics as diag_metrics
from ..diagnostics import safe_eval_metric
from ..irm.configs import IRMConfig
from ..irm.head_grads import compute_env_head_grads
from ..irm.penalties import cosine_alignment_penalty, varnorm_penalty
from ._base import OptimizerStepMixin, stack_losses
from .common import TrainBatch


class HIRMAlgorithm(OptimizerStepMixin):
    """Implements the gradient alignment penalty used by HIRM."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        grad_clip: Optional[float],
        irm_config: IRMConfig,
        detach_features: bool,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
        )
        self._model = model
        self._irm = irm_config
        self._detach_features = bool(detach_features)

    @property
    def representation_scale(self) -> torch.Tensor | None:
        return None

    @property
    def irm_config(self) -> IRMConfig:
        return self._irm

    @property
    def detach_features(self) -> bool:
        return self._detach_features

    def step(self, batch: TrainBatch) -> Dict[str, float]:
        self._optimizer.zero_grad(set_to_none=True)
        losses = stack_losses(batch.env_losses)
        risk = losses.mean()
        logs: Dict[str, float] = {"train/loss": float(risk.detach().item()), "train/penalty": 0.0}

        penalty = None
        grad_list: List[torch.Tensor] = []
        if self._irm.enabled and len(batch.env_losses) >= self._irm.env_min:
            with self._model.alignment_context(detach_features=self._detach_features):
                grad_list = compute_env_head_grads(
                    self._model,
                    lambda _model, payload: payload.get("irm_loss", payload["loss"]),
                    (env.payload for env in batch.env_losses),
                    create_graph=True,
                )
            if grad_list:
                if self._irm.type == "cosine":
                    penalty = cosine_alignment_penalty(grad_list, eps=self._irm.eps)
                else:
                    penalty = varnorm_penalty(grad_list, eps=self._irm.eps)
                weighted = self._irm.lambda_weight * penalty
                logs["train/penalty"] = float(weighted.detach().item())
                logs["train/irm_penalty"] = float(penalty.detach().item())
                logs["train/irm_penalty_weighted"] = float(weighted.detach().item())
                logs["train/lambda"] = float(self._irm.lambda_weight)
            else:
                logs["train/irm_penalty"] = 0.0
                logs["train/irm_penalty_weighted"] = 0.0
                logs["train/lambda"] = float(self._irm.lambda_weight)
        elif self._irm.enabled:
            logs["train/lambda"] = float(self._irm.lambda_weight)
            logs["train/irm_penalty"] = 0.0
            logs["train/irm_penalty_weighted"] = 0.0

        if penalty is not None:
            total = risk + self._irm.lambda_weight * penalty
        else:
            total = risk

        self._backward_and_step(total)

        loss_values = [float(env.loss.detach().item()) for env in batch.env_losses]
        logs["train/ig"] = safe_eval_metric(diag_metrics.invariant_gap, loss_values)
        logs["train/wg"] = safe_eval_metric(diag_metrics.worst_group, loss_values)

        logs.setdefault("train/irm_penalty", 0.0)
        logs.setdefault("train/irm_penalty_weighted", 0.0)
        if self._irm.enabled:
            logs.setdefault("train/lambda", float(self._irm.lambda_weight))

        if self._irm.logging.log_irm_grads and grad_list:
            with torch.no_grad():
                norms = [grad.norm() for grad in grad_list]
                if norms:
                    stacked = torch.stack(norms)
                    logs["train/irm_grad_norm_min"] = float(stacked.min().item())
                    logs["train/irm_grad_norm_mean"] = float(stacked.mean().item())
                    logs["train/irm_grad_norm_max"] = float(stacked.max().item())
                if len(grad_list) >= 2:
                    normalised = []
                    for grad in grad_list:
                        norm = grad.norm()
                        if norm <= self._irm.eps:
                            normalised.append(torch.zeros_like(grad))
                        else:
                            normalised.append(grad / norm.clamp_min(self._irm.eps))
                    stacked = torch.stack(normalised)
                    cos_matrix = stacked @ stacked.T
                    idx = torch.triu_indices(cos_matrix.shape[0], cos_matrix.shape[1], offset=1)
                    pairwise_cos = cos_matrix[idx[0], idx[1]]
                    logs["train/irm_grad_cosine_mean"] = float(pairwise_cos.mean().item())
                    logs["train/irm_grad_cosine_min"] = float(pairwise_cos.min().item())
                    logs["train/irm_grad_cosine_max"] = float(pairwise_cos.max().item())

        return logs


__all__ = ["HIRMAlgorithm"]
