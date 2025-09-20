from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .features import FeatureBuilder
from hirm_experiment.data.dataset import EpisodeBatch


@dataclass
class HedgingOutputs:
    actions: torch.Tensor
    episode_pnl: torch.Tensor
    step_pnl: torch.Tensor
    turnover: torch.Tensor
    hedge_error: torch.Tensor


def rollout_policy(
    model: nn.Module,
    batch: EpisodeBatch,
    feature_builder: FeatureBuilder,
    label_smoothing: float = 0.0,
    action_scale: Optional[torch.Tensor] = None,
) -> HedgingOutputs:
    device = batch.delta.device
    batch_size, horizon = batch.delta.shape
    inventory = torch.zeros(batch_size, device=device)
    actions = []
    for t in range(horizon):
        features = feature_builder.step_features(batch, t, inventory)
        action = model(features).squeeze(-1)
        if label_smoothing > 0.0:
            action = (1.0 - label_smoothing) * action + label_smoothing * batch.delta[:, t]
        if action_scale is not None:
            action = action * action_scale
        actions.append(action)
        inventory = action
    action_tensor = torch.stack(actions, dim=1)
    return compute_pnl(action_tensor, batch)


def compute_pnl(actions: torch.Tensor, batch: EpisodeBatch) -> HedgingOutputs:
    device = actions.device
    batch_size, horizon = actions.shape
    dspot = batch.spot[:, 1:] - batch.spot[:, :-1]
    prev = torch.zeros(batch_size, 1, device=device)
    prev = torch.cat([prev, actions[:, :-1]], dim=1)
    rebalance = actions - prev
    linear_cost = batch.tx_linear.to(device) * rebalance.abs()
    quadratic_cost = batch.tx_quadratic.to(device) * rebalance.pow(2)
    hedge_error = (actions - batch.delta) * dspot
    step_pnl = -hedge_error - linear_cost - quadratic_cost
    episode_pnl = step_pnl.sum(dim=1)
    turnover = rebalance.abs().sum(dim=1)
    squared_error = hedge_error.pow(2).sum(dim=1)
    return HedgingOutputs(
        actions=actions,
        episode_pnl=episode_pnl,
        step_pnl=step_pnl,
        turnover=turnover,
        hedge_error=squared_error,
    )
