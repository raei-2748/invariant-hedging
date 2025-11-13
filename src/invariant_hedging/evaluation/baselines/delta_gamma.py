"""Delta-gamma hedging baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch.distributions import Normal

from invariant_hedging.data.features import FeatureScaler
from invariant_hedging.data.markets import pricing


@dataclass
class _FeatureView:
    mean: torch.Tensor
    std: torch.Tensor
    index: Mapping[str, int]

    def to(self, device: torch.device) -> "_FeatureView":
        return _FeatureView(
            mean=self.mean.to(device),
            std=self.std.to(device),
            index=self.index,
        )

    def get(self, features: torch.Tensor, key: str) -> torch.Tensor:
        idx = self.index[key]
        scale = torch.clamp(self.std[idx], min=1e-6)
        return features[:, idx] * scale + self.mean[idx]


class DeltaGammaBaselinePolicy:
    """Predict the next-period delta using a Black--Scholes delta-gamma update.

    The policy extrapolates one rebalancing step ahead using the current option delta and
    gamma together with an estimate of the underlying state implied by the normalised
    features.  The following assumptions are made and documented explicitly:

    * ``feature_names`` contains the canonical entries produced by
      :class:`~invariant_hedging.data.features.FeatureEngineer` (``delta``, ``gamma``,
      ``time_to_maturity``, ``realized_vol`` and ``inventory``).
    * The realised volatility feature is treated as a proxy for the Black--Scholes implied
      volatility used by :mod:`invariant_hedging.data.markets.pricing`.
    * Risk-free rates are assumed negligible during the short evaluation horizon and are
      therefore set to zero when applying the pricing formulas.

    Under these assumptions the next-period delta is evaluated by reducing the time to
    maturity by one business day (``1/252`` years) before clamping the hedge ratio to the
    configured position limits.
    """

    def __init__(
        self,
        scaler: FeatureScaler,
        feature_names: Sequence[str],
        max_position: float,
        *,
        delta_key: str = "delta",
        gamma_key: str = "gamma",
        tau_key: str = "time_to_maturity",
        vol_key: str = "realized_vol",
        rebalance_dt: float = 1.0 / 252.0,
    ) -> None:
        if len(feature_names) != scaler.mean.shape[0]:
            raise ValueError("Number of feature names must match scaler dimensionality")
        required = {delta_key, gamma_key, tau_key}
        missing = required.difference(feature_names)
        if missing:
            raise ValueError(f"Missing required features for delta-gamma baseline: {sorted(missing)}")
        index_map = {name: idx for idx, name in enumerate(feature_names)}
        self._features = _FeatureView(
            mean=scaler.mean.clone().detach(),
            std=torch.clamp(scaler.std.clone().detach(), min=1e-6),
            index=index_map,
        )
        self._delta_key = delta_key
        self._gamma_key = gamma_key
        self._tau_key = tau_key
        self._vol_key = vol_key if vol_key in index_map else None
        self.max_position = float(max_position)
        self._rebalance_dt = float(rebalance_dt)
        self._normal = Normal(0.0, 1.0)
        self._device: torch.device | None = None

    def to(self, device: torch.device) -> "DeltaGammaBaselinePolicy":
        self._features = self._features.to(device)
        self._normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        self._device = device
        return self

    def eval(self) -> "DeltaGammaBaselinePolicy":  # pragma: no cover - simple delegation
        return self

    def reset(self, env_index: int | None = None) -> None:  # pragma: no cover - stateless
        del env_index

    def _estimate_spot(
        self,
        delta: torch.Tensor,
        gamma: torch.Tensor,
        sigma: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        delta_clamped = torch.clamp(delta, 1e-5, 1.0 - 1e-5)
        d1 = self._normal.icdf(delta_clamped)
        pdf = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        denom = torch.clamp(gamma * sigma * torch.sqrt(torch.clamp(tau, min=1e-6)), min=1e-6)
        return pdf / denom

    def _estimate_strike(
        self,
        spot: torch.Tensor,
        delta: torch.Tensor,
        sigma: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        delta_clamped = torch.clamp(delta, 1e-5, 1.0 - 1e-5)
        d1 = self._normal.icdf(delta_clamped)
        term = d1 * sigma * torch.sqrt(torch.clamp(tau, min=1e-6)) - 0.5 * sigma ** 2 * tau
        return torch.exp(torch.log(torch.clamp(spot, min=1e-6)) - term)

    def __call__(
        self,
        features: torch.Tensor,
        env_index: int,
        representation_scale=None,
    ) -> Dict[str, torch.Tensor]:
        del env_index, representation_scale
        device = features.device if self._device is None else self._device

        delta = self._features.get(features, self._delta_key)
        gamma = self._features.get(features, self._gamma_key)
        tau = torch.clamp(self._features.get(features, self._tau_key), min=1e-6)
        if self._vol_key is not None:
            sigma = torch.clamp(self._features.get(features, self._vol_key), min=1e-4)
        else:
            sigma = torch.full_like(delta, 0.2)

        spot = self._estimate_spot(delta, gamma, sigma, tau)
        strike = self._estimate_strike(spot, delta, sigma, tau)

        tau_next = torch.clamp(tau - self._rebalance_dt, min=1e-6)
        delta_next = pricing.black_scholes_delta(spot, strike, 0.0, sigma, tau_next)

        action = torch.where(tau <= self._rebalance_dt, delta, delta_next)
        action = action.unsqueeze(-1).to(device)
        action = action.clamp(-self.max_position, self.max_position)
        return {"action": action, "raw_action": action}
