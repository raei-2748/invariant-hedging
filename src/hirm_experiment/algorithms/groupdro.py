from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .base import Algorithm, AlgorithmOutput
from hirm_experiment.data.dataset import EpisodeBatch


class GroupDROAlgorithm(Algorithm):
    def __init__(
        self,
        feature_builder,
        device: torch.device,
        env_names: List[str],
        eta: float = 0.1,
    ) -> None:
        super().__init__("groupdro", feature_builder, device)
        self.eta = eta
        self.env_order = env_names
        weights = torch.ones(len(env_names), device=device) / len(env_names)
        self.weights = weights

    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        outputs = self._rollout_all(model, batches)
        losses = torch.stack([self._risk_loss(outputs[env]) for env in self.env_order])
        with torch.no_grad():
            scaled = self.weights * torch.exp(self.eta * (losses - losses.mean()))
            self.weights = scaled / scaled.sum()
        total_loss = torch.sum(self.weights * losses)
        env_losses = {env: loss for env, loss in zip(self.env_order, losses)}
        penalties = {env: torch.tensor(0.0, device=self.device) for env in batches}
        logs = self._collect_logs(outputs)
        for env, weight in zip(self.env_order, self.weights):
            logs[f"{env}_weight"] = weight.detach()
        logs["loss"] = total_loss.detach()
        return AlgorithmOutput(total_loss, env_losses, penalties, logs)
