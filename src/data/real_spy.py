"""Load a thin SPY options slice for the real-data anchor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .synthetic import EpisodeBatch


@dataclass
class RealDataConfig:
    file: str
    slippage_bps: float
    rate: float
    min_volume: int
    episode_length: int = 60


class RealSPYDataset:
    """Very small SPY options dataset for deterministic smoke tests."""

    def __init__(self, cfg: RealDataConfig):
        self.cfg = cfg

    def load(self) -> EpisodeBatch:
        path = Path(self.cfg.file)
        if not path.exists():
            raise FileNotFoundError(f"Real data file not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        df = df[df["volume"] >= self.cfg.min_volume]
        if df.empty:
            raise ValueError("No data passed the liquidity filters")
        series_len = len(df)
        episode_len = self.cfg.episode_length
        repeats = int(np.ceil((episode_len + 1) / series_len))
        spot_series = np.tile(df["spot"].to_numpy(), repeats)[: episode_len + 1]
        option_series = np.tile(df["mid_price"].to_numpy(), repeats)[: episode_len + 1]
        vol_series = np.tile(df["implied_vol"].to_numpy(), repeats)[: episode_len + 1]
        tau = np.maximum(episode_len - np.arange(episode_len + 1), 1) / 252.0
        batch = EpisodeBatch(
            spot=torch.from_numpy(spot_series).float().unsqueeze(0),
            option_price=torch.from_numpy(option_series).float().unsqueeze(0),
            implied_vol=torch.from_numpy(vol_series).float().unsqueeze(0),
            time_to_maturity=torch.from_numpy(tau).float().unsqueeze(0),
            rate=self.cfg.rate,
            env_name="real_spy",
            meta={
                "linear_bps": float(self.cfg.slippage_bps),
                "quadratic": 0.0,
                "slippage_multiplier": 1.0,
                "notional": 1.0,
            },
        )
        return batch


def load_real_dataset(config: Dict) -> EpisodeBatch:
    cfg = RealDataConfig(
        file=config.get("file"),
        slippage_bps=float(config.get("slippage_bps", 5)),
        rate=float(config.get("rate", 0.01)),
        min_volume=int(config.get("min_volume", 1000)),
        episode_length=int(config.get("episode_length", 60)),
    )
    return RealSPYDataset(cfg).load()
