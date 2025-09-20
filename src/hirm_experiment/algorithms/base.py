from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from hirm_experiment.data.dataset import EpisodeBatch
from hirm_experiment.training.features import FeatureBuilder
from hirm_experiment.training.pnl import HedgingOutputs, rollout_policy


@dataclass
class AlgorithmOutput:
    loss: torch.Tensor
    env_losses: Dict[str, torch.Tensor]
    penalties: Dict[str, torch.Tensor]
    logs: Dict[str, torch.Tensor]


class Algorithm(ABC):
    def __init__(self, name: str, feature_builder: FeatureBuilder, device: torch.device) -> None:
        self.name = name
        self.feature_builder = feature_builder
        self.device = device

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        step: int,
    ) -> AlgorithmOutput:
        raise NotImplementedError

    def _rollout_all(
        self,
        model: nn.Module,
        batches: Dict[str, EpisodeBatch],
        label_smoothing: float = 0.0,
        action_scale: torch.Tensor | None = None,
    ) -> Dict[str, HedgingOutputs]:
        outputs: Dict[str, HedgingOutputs] = {}
        scale_tensor = None
        if action_scale is not None:
            scale_tensor = action_scale.to(self.device)
        for env, batch in batches.items():
            outputs[env] = rollout_policy(
                model,
                batch,
                self.feature_builder,
                label_smoothing=label_smoothing,
                action_scale=scale_tensor,
            )
        return outputs

    def _risk_loss(self, outputs: HedgingOutputs) -> torch.Tensor:
        return -outputs.episode_pnl.mean()

    def _collect_logs(self, outputs: Dict[str, HedgingOutputs]) -> Dict[str, torch.Tensor]:
        logs: Dict[str, torch.Tensor] = {}
        for env, out in outputs.items():
            logs[f"{env}_pnl_mean"] = out.episode_pnl.mean().detach()
            logs[f"{env}_pnl_std"] = out.episode_pnl.std(unbiased=False).detach()
            logs[f"{env}_turnover_mean"] = out.turnover.mean().detach()
            logs[f"{env}_hedge_error"] = out.hedge_error.mean().detach()
        return logs
