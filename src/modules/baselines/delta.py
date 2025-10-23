"""Delta hedging baseline policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch

from ..data.features import FeatureScaler


@dataclass
class _FeatureStatistics:
    """Lightweight view over feature scaling statistics.

    The evaluation pipeline standardises features using :class:`FeatureScaler`.  We rely on
    the stored mean and standard deviation tensors to recover the original feature values
    before translating them into hedge ratios.
    """

    mean: torch.Tensor
    std: torch.Tensor
    feature_index: Mapping[str, int]

    def to(self, device: torch.device) -> "_FeatureStatistics":
        return _FeatureStatistics(
            mean=self.mean.to(device),
            std=self.std.to(device),
            feature_index=self.feature_index,
        )

    def value(self, features: torch.Tensor, key: str) -> torch.Tensor:
        idx = self.feature_index[key]
        scale = torch.clamp(self.std[idx], min=1e-6)
        return features[:, idx] * scale + self.mean[idx]


class DeltaBaselinePolicy:
    """Hold the option's Black--Scholes delta as the hedge position.

    The policy assumes that ``feature_names`` includes an entry named ``"delta"`` whose
    scaling statistics were computed from the same :class:`FeatureScaler` used during model
    training.  The option delta is recovered by undoing the standardisation before clamping
    the resulting hedge ratio to the configured position limit.
    """

    def __init__(
        self,
        scaler: FeatureScaler,
        feature_names: Sequence[str],
        max_position: float,
        *,
        delta_key: str = "delta",
    ) -> None:
        if len(feature_names) != scaler.mean.shape[0]:
            raise ValueError("Number of feature names must match scaler dimensionality")
        if delta_key not in feature_names:
            raise ValueError(f"Feature '{delta_key}' not present in feature list: {feature_names}")
        index_map = {name: idx for idx, name in enumerate(feature_names)}
        self._stats = _FeatureStatistics(
            mean=scaler.mean.clone().detach(),
            std=torch.clamp(scaler.std.clone().detach(), min=1e-6),
            feature_index=index_map,
        )
        self._delta_key = delta_key
        self.max_position = float(max_position)
        self._device: torch.device | None = None

    def to(self, device: torch.device) -> "DeltaBaselinePolicy":
        self._stats = self._stats.to(device)
        self._device = device
        return self

    def eval(self) -> "DeltaBaselinePolicy":  # pragma: no cover - simple delegation
        return self

    def reset(self, env_index: int | None = None) -> None:  # pragma: no cover - stateless
        del env_index  # no-op hook for interface parity

    def __call__(
        self,
        features: torch.Tensor,
        env_index: int,
        representation_scale=None,
    ) -> Dict[str, torch.Tensor]:
        del env_index, representation_scale  # unused
        device = features.device if self._device is None else self._device
        delta = self._stats.value(features, self._delta_key)
        action = delta.unsqueeze(-1).to(device)
        action = action.clamp(-self.max_position, self.max_position)
        return {"action": action, "raw_action": action}
