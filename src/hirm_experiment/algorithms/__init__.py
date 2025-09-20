from __future__ import annotations

from typing import Dict, List

import torch

from .erm import ERMAlgorithm, ERMRegularizedAlgorithm
from .groupdro import GroupDROAlgorithm
from .irm import IRMAlgorithm
from .vrex import VRExAlgorithm
from .base import Algorithm
from hirm_experiment.training.features import FeatureBuilder


def build_algorithm(
    config: Dict,
    feature_builder: FeatureBuilder,
    device: torch.device,
    train_envs: List[str],
) -> Algorithm:
    name = config.get("name", "erm")
    if name == "erm":
        return ERMAlgorithm(feature_builder, device)
    if name == "erm_reg":
        smoothing = config.get("regularization", {}).get("label_smoothing", 0.0)
        return ERMRegularizedAlgorithm(feature_builder, device, label_smoothing=smoothing)
    if name == "groupdro":
        eta = float(config.get("eta", 0.1))
        return GroupDROAlgorithm(feature_builder, device, env_names=train_envs, eta=eta)
    if name == "vrex":
        weight = float(config.get("penalty_weight", 1.0))
        anneal = int(config.get("penalty_anneal_iters", 0))
        return VRExAlgorithm(feature_builder, device, penalty_weight=weight, penalty_anneal_iters=anneal)
    if name == "irm_v1":
        return IRMAlgorithm(
            feature_builder,
            device,
            lambda_initial=float(config.get("lambda_initial", 0.0)),
            lambda_target=float(config.get("lambda_target", 1.0)),
            pretrain_steps=int(config.get("pretrain_steps", 0)),
            ramp_steps=int(config.get("ramp_steps", 0)),
            penalty_anneal_iters=int(config.get("penalty_anneal_iters", 0)),
        )
    raise ValueError(f"Unsupported algorithm: {name}")
