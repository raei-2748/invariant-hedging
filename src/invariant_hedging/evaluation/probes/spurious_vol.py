"""Evaluation probes targeting the realized volatility features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SpuriousVolConfig:
    enabled: bool = False
    mode: str = "randomize"
    k: float = 2.0


def _locate_realized_vol_indices(feature_names: Sequence[str]) -> list[int]:
    return [i for i, name in enumerate(feature_names) if "realized_vol" in name]


def apply_spurious_vol_probe(
    features: torch.Tensor,
    feature_names: Sequence[str],
    config: SpuriousVolConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply spurious feature perturbations to realized volatility columns.

    Parameters
    ----------
    features:
        Tensor of shape ``(..., num_features)`` containing engineered features.
    feature_names:
        Sequence of feature names aligned with the last dimension of ``features``.
    config:
        Probe configuration specifying perturbation type.
    generator:
        Optional PRNG generator for reproducible randomization.
    """

    if not config.enabled:
        return features

    idxs = _locate_realized_vol_indices(feature_names)
    if not idxs:
        return features

    perturbed = features.clone()
    if config.mode == "randomize":
        noise = torch.randn_like(perturbed[..., idxs], generator=generator)
        perturbed[..., idxs] = noise
    elif config.mode == "amplify":
        perturbed[..., idxs] = perturbed[..., idxs] * config.k
    else:
        raise ValueError(f"Unknown spurious vol probe mode: {config.mode}")
    return perturbed


def compute_msi_delta(base_msi: float, probed_msi: float) -> float:
    return probed_msi - base_msi


__all__ = ["SpuriousVolConfig", "apply_spurious_vol_probe", "compute_msi_delta"]
