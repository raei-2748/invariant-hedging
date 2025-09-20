from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .base import Algorithm, AlgorithmOutput
from hirm_experiment.data.dataset import EpisodeBatch
from hirm_experiment.training.pnl import rollout_policy


class IRMAlgorithm(Algorithm):
    def __init__(
        self,
        feature_builder,
        device: torch.device,
        lambda_initial: float,
        lambda_target: float,
        pretrain_steps: int,
        ramp_steps: int,
        penalty_anneal_iters: int = 0,
    ) -> None:
        super().__init__("irm_v1", feature_builder, device)
        self.lambda_initial = lambda_initial
        self.lambda_target = lambda_target
        self.pretrain_steps = pretrain_steps
        self.ramp_steps = ramp_steps
        self.penalty_anneal_iters = penalty_anneal_iters

    def _lambda(self, step: int) -> float:
        if step < self.pretrain_steps:
            return self.lambda_initial
        progress = min(max(step - self.pretrain_steps, 0), self.ramp_steps)
        if self.ramp_steps == 0:
            frac = 1.0
        else:
            frac = progress / self.ramp_steps
        return self.lambda_initial + frac * (self.lambda_target - self.lambda_initial)

    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        outputs = self._rollout_all(model, batches)
        env_losses = {env: self._risk_loss(out) for env, out in outputs.items()}
        mean_loss = torch.stack(list(env_losses.values())).mean()
        penalties: Dict[str, torch.Tensor] = {}
        penalty_terms = []
        if step >= self.penalty_anneal_iters:
            for env, batch in batches.items():
                scale = torch.tensor(1.0, device=self.device, requires_grad=True)
                scaled = rollout_policy(model, batch, self.feature_builder, action_scale=scale)
                scaled_loss = self._risk_loss(scaled)
                grad = torch.autograd.grad(scaled_loss, [scale], create_graph=True)[0]
                penalty = grad.pow(2)
                penalties[env] = penalty.detach()
                penalty_terms.append(penalty)
        if penalty_terms:
            penalty_value = torch.stack(penalty_terms).mean()
        else:
            penalty_value = torch.tensor(0.0, device=self.device)
        lam = torch.tensor(self._lambda(step), device=self.device)
        total_loss = mean_loss + lam * penalty_value
        logs = self._collect_logs(outputs)
        logs["loss"] = total_loss.detach()
        logs["irm_penalty"] = penalty_value.detach()
        logs["irm_lambda"] = lam.detach()
        return AlgorithmOutput(total_loss, env_losses, penalties, logs)
