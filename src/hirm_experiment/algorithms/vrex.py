from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .base import Algorithm, AlgorithmOutput
from hirm_experiment.data.dataset import EpisodeBatch


class VRExAlgorithm(Algorithm):
    def __init__(
        self,
        feature_builder,
        device: torch.device,
        penalty_weight: float,
        penalty_anneal_iters: int = 0,
    ) -> None:
        super().__init__("vrex", feature_builder, device)
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters

    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        outputs = self._rollout_all(model, batches)
        env_losses = {env: self._risk_loss(out) for env, out in outputs.items()}
        losses = torch.stack(list(env_losses.values()))
        mean_loss = losses.mean()
        variance = ((losses - mean_loss) ** 2).mean()
        weight = self.penalty_weight if step >= self.penalty_anneal_iters else 0.0
        total_loss = mean_loss + weight * variance
        penalties = {env: torch.tensor(0.0, device=self.device) for env in batches}
        penalties["vrex_variance"] = variance.detach()
        logs = self._collect_logs(outputs)
        logs["loss"] = total_loss.detach()
        logs["variance"] = variance.detach()
        return AlgorithmOutput(total_loss, env_losses, penalties, logs)
