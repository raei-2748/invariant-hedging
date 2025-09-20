from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .base import Algorithm, AlgorithmOutput
from hirm_experiment.data.dataset import EpisodeBatch


class ERMAlgorithm(Algorithm):
    def __init__(self, feature_builder, device: torch.device) -> None:
        super().__init__("erm", feature_builder, device)

    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        outputs = self._rollout_all(model, batches)
        env_losses = {env: self._risk_loss(out) for env, out in outputs.items()}
        stacked = torch.stack(list(env_losses.values()))
        total_loss = stacked.mean()
        penalties = {env: torch.tensor(0.0, device=self.device) for env in batches}
        logs = self._collect_logs(outputs)
        logs["loss"] = total_loss.detach()
        return AlgorithmOutput(loss=total_loss, env_losses=env_losses, penalties=penalties, logs=logs)


class ERMRegularizedAlgorithm(Algorithm):
    def __init__(self, feature_builder, device: torch.device, label_smoothing: float = 0.0) -> None:
        super().__init__("erm_reg", feature_builder, device)
        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        outputs = self._rollout_all(model, batches, label_smoothing=self.label_smoothing)
        env_losses = {env: self._risk_loss(out) for env, out in outputs.items()}
        total_loss = torch.stack(list(env_losses.values())).mean()
        penalties = {env: torch.tensor(0.0, device=self.device) for env in batches}
        logs = self._collect_logs(outputs)
        logs["loss"] = total_loss.detach()
        return AlgorithmOutput(loss=total_loss, env_losses=env_losses, penalties=penalties, logs=logs)
