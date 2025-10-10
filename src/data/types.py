"""Shared data structures used by synthetic and real datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import torch


@dataclass
class EpisodeBatch:
    """Container for a batch of simulated or real market episodes."""

    spot: torch.Tensor
    option_price: torch.Tensor
    implied_vol: torch.Tensor
    time_to_maturity: torch.Tensor
    rate: float
    env_name: str
    meta: Dict[str, Union[float, str]]

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            spot=self.spot.to(device),
            option_price=self.option_price.to(device),
            implied_vol=self.implied_vol.to(device),
            time_to_maturity=self.time_to_maturity.to(device),
            rate=self.rate,
            env_name=self.env_name,
            meta=self.meta,
        )

    @property
    def steps(self) -> int:
        return self.spot.shape[1] - 1

    def subset(self, indices) -> "EpisodeBatch":
        return EpisodeBatch(
            spot=self.spot[indices],
            option_price=self.option_price[indices],
            implied_vol=self.implied_vol[indices],
            time_to_maturity=self.time_to_maturity[indices],
            rate=self.rate,
            env_name=self.env_name,
            meta=self.meta,
        )
