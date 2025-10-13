"""Feature engineering for hedging policies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

import torch

from ..markets import pricing
from .types import EpisodeBatch


@dataclass
class FeatureScaler:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> "FeatureScaler":
        return FeatureScaler(mean=self.mean.to(device), std=self.std.to(device))

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / torch.clamp(self.std, min=1e-6)


class FeatureEngineer:
    """Compute invariant and diagnostic feature sets."""

    def __init__(self, realized_vol_window: int = 20):
        self.realized_vol_window = realized_vol_window
        self.scaler: FeatureScaler | None = None
        self.feature_names: List[str] = [
            "delta",
            "gamma",
            "time_to_maturity",
            "realized_vol",
            "inventory",
        ]

    def base_features(self, batch: EpisodeBatch) -> torch.Tensor:
        spot = batch.spot[:, :-1]
        vol = batch.implied_vol[:, :-1]
        tau = batch.time_to_maturity[:, :-1]
        strike = batch.spot[:, [0]].expand_as(spot)
        delta = pricing.black_scholes_delta(spot, strike, batch.rate, vol, tau)
        gamma = pricing.black_scholes_gamma(spot, strike, batch.rate, vol, tau)
        realized_vol = self._trailing_realized_vol(batch)
        return torch.stack([delta, gamma, tau, realized_vol], dim=-1)

    def fit(self, batches: Iterable[EpisodeBatch]) -> FeatureScaler:
        feats = []
        for batch in batches:
            base = self.base_features(batch)
            zeros = torch.zeros_like(base[..., :1])
            combined = torch.cat([base, zeros], dim=-1)
            feats.append(combined.reshape(-1, combined.shape[-1]))
        stacked = torch.cat(feats, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        self.scaler = FeatureScaler(mean=mean, std=std)
        return self.scaler

    def transform(self, batch: EpisodeBatch, inventory: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            raise RuntimeError("Feature scaler has not been fit yet.")
        base = self.base_features(batch)
        inv = inventory.unsqueeze(-1)
        features = torch.cat([base, inv], dim=-1)
        return self.scaler.transform(features)

    def describe(self) -> Mapping[str, object]:
        """Return metadata describing the engineered feature set."""

        if self.scaler is None:
            raise RuntimeError("Feature scaler has not been fit yet.")

        mean = self.scaler.mean.detach().cpu().tolist()
        std = self.scaler.std.detach().cpu().tolist()
        names = list(self.feature_names)
        scaler_stats = {
            "mean": {name: float(value) for name, value in zip(names, mean)},
            "std": {name: float(value) for name, value in zip(names, std)},
            "vector": {
                "mean": mean,
                "std": std,
            },
        }

        return {
            "feature_names": names,
            "windows": {"realized_vol": int(self.realized_vol_window)},
            "scaler": scaler_stats,
        }

    def _trailing_realized_vol(self, batch: EpisodeBatch) -> torch.Tensor:
        returns = torch.log(batch.spot[:, 1:] / batch.spot[:, :-1]).clamp(min=-10, max=10)
        window = self.realized_vol_window
        vols = []
        for t in range(returns.shape[1]):
            start = max(0, t - window + 1)
            segment = returns[:, start : t + 1]
            vol_t = segment.std(dim=1, unbiased=False)
            vols.append(vol_t)
        realized = torch.stack(vols, dim=1)
        return torch.clamp(realized, min=1e-4)


def save_feature_metadata(feature_engineer: FeatureEngineer, run_dir: Path | str) -> Path:
    """Persist feature metadata alongside training/eval artifacts."""

    metadata = feature_engineer.describe()
    run_path = Path(run_dir)
    destination = run_path / "features.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
