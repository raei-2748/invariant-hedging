"""SPY options + E-mini futures real-data loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from ..types import EpisodeBatch
from ..real_spy import RealSPYDataset, RealDataConfig as SmokeConfig


WINDOW_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "volmageddon": {"start": "2018-02-01", "end": "2018-04-30"},
    "q4_2018": {"start": "2018-10-01", "end": "2018-12-31"},
    "covid_2020": {"start": "2020-02-01", "end": "2020-06-30"},
    "bear_2022": {"start": "2022-01-03", "end": "2022-12-30"},
    "gfc": {"start": "2008-09-01", "end": "2009-06-30"},
}


@dataclass
class WindowConfig:
    name: str
    start: str
    end: str
    linear_bps: float
    quadratic: float
    slippage_multiplier: float


@dataclass
class SpyEminiConfig:
    spy_path: str
    rate: float = 0.01
    mode: str = "full"
    include_gfc: bool = False
    base_linear_bps: float = 5.0
    base_quadratic: float = 0.0
    base_slippage_multiplier: float = 1.0

    def resolve_windows(self) -> List[WindowConfig]:
        windows: List[WindowConfig] = []
        for key, bounds in WINDOW_DEFINITIONS.items():
            if key == "gfc" and not self.include_gfc:
                continue
            windows.append(
                WindowConfig(
                    name=key,
                    start=bounds["start"],
                    end=bounds["end"],
                    linear_bps=self.base_linear_bps,
                    quadratic=self.base_quadratic,
                    slippage_multiplier=self.base_slippage_multiplier,
                )
            )
        return windows


class SpyEminiDataModule:
    """Loads full SPY/E-mini data and exposes named market windows."""

    def __init__(self, config: Dict) -> None:
        self.config = SpyEminiConfig(**config)
        if self.config.mode == "smoke":
            self._windows = []
            self._env_order = ["smoke"]
        else:
            self._windows = self.config.resolve_windows()
            self._env_order = [window.name for window in self._windows]
        self._batches: Dict[str, EpisodeBatch] | None = None

    @property
    def env_order(self) -> List[str]:
        return list(self._env_order)

    def _load_full_dataset(self) -> Dict[str, EpisodeBatch]:
        path = Path(self.config.spy_path)
        if not path.exists():
            raise FileNotFoundError(f"SPY options dataset missing: {path}")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        if {"spot", "mid_price", "implied_vol"} - set(df.columns):
            raise ValueError("Dataset must contain 'spot', 'mid_price', and 'implied_vol' columns")
        batches: Dict[str, EpisodeBatch] = {}
        for window in self._windows:
            mask = (df["date"] >= window.start) & (df["date"] <= window.end)
            window_df = df.loc[mask]
            if window_df.empty:
                continue
            series_len = len(window_df)
            tau = np.maximum(series_len - np.arange(series_len), 1) / 252.0
            batch = EpisodeBatch(
                spot=torch.from_numpy(window_df["spot"].to_numpy(dtype=np.float32)).unsqueeze(0),
                option_price=torch.from_numpy(window_df["mid_price"].to_numpy(dtype=np.float32)).unsqueeze(0),
                implied_vol=torch.from_numpy(window_df["implied_vol"].to_numpy(dtype=np.float32)).unsqueeze(0),
                time_to_maturity=torch.from_numpy(tau.astype(np.float32)).unsqueeze(0),
                rate=self.config.rate,
                env_name=window.name,
                meta={
                    "linear_bps": window.linear_bps,
                    "quadratic": window.quadratic,
                    "slippage_multiplier": window.slippage_multiplier,
                    "notional": 1.0,
                    "start_date": pd.Timestamp(window.start).strftime("%Y-%m-%d"),
                    "end_date": pd.Timestamp(window.end).strftime("%Y-%m-%d"),
                },
            )
            batches[window.name] = batch
        return batches

    def _load_smoke_dataset(self) -> Dict[str, EpisodeBatch]:
        cfg = SmokeConfig(
            file=self.config.spy_path,
            slippage_bps=self.config.base_linear_bps,
            rate=self.config.rate,
            min_volume=0,
            episode_length=60,
        )
        batch = RealSPYDataset(cfg).load()
        return {"smoke": batch}

    def prepare(self, split: str, env_names: Iterable[str]) -> Dict[str, EpisodeBatch]:
        if self._batches is None:
            if self.config.mode == "smoke":
                self._batches = self._load_smoke_dataset()
            else:
                self._batches = self._load_full_dataset()
        requested = list(env_names)
        missing = [name for name in requested if name not in self._batches]
        if missing:
            raise KeyError(f"Requested windows not available: {missing}")
        return {name: self._batches[name] for name in requested}

    def hourly_dataset(self, env_name: str) -> EpisodeBatch:
        batches = self.prepare("train", [env_name])
        if env_name not in batches:
            raise KeyError(f"Window '{env_name}' not available")
        return batches[env_name]
