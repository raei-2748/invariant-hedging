"""Loader for deterministic real-market anchors with episode tagging."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
import warnings

import numpy as np
import pandas as pd
import torch

from .anchors import AnchorSpec, EpisodeWindow, generate_episode_windows, parse_anchor_specs
from ..types import EpisodeBatch
from invariant_hedging.data.tags import EpisodeTag, attach_tags, build_episode_tag


@dataclass
class EpisodeConfig:
    days: int = 60
    stride_days: int = 5
    tz: str = "America/New_York"


@dataclass
class VendorConfig:
    path_csv_root: str = "data/real"


@dataclass
class SymbolConfig:
    underlying: str = "SPY"
    futures: Optional[str] = None
    options: Optional[str] = None


@dataclass
class LoaderConfig:
    source: str
    anchors: List[Mapping[str, object]]
    episode: Mapping[str, object]
    symbols: Mapping[str, object]
    vendor: Mapping[str, object]
    seed: int = 0


@dataclass
class LoadedAnchor:
    """Concrete data produced for a single anchor."""

    anchor: AnchorSpec
    batch: EpisodeBatch
    tags: List[EpisodeTag]


class RealAnchorLoader:
    """Create episode batches for configured real-market anchors."""

    def __init__(self, config: Mapping[str, object]):
        self.config = self._build_config(config)
        self.anchors = parse_anchor_specs(self.config.anchors)
        self.episode_cfg = EpisodeConfig(**self.config.episode)
        self.symbols = SymbolConfig(**self.config.symbols)
        self.vendor = VendorConfig(**self.config.vendor)
        self.seed = int(getattr(self.config, "seed", 0))
        self.root = Path(self.vendor.path_csv_root)
        self._underlying_cache: Optional[pd.DataFrame] = None
        self._missing_option_warned = False
        self._has_option_prices = True
        self._exclusion_windows = self._build_exclusion_windows()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, LoadedAnchor]:
        underlying = self._load_underlying()
        trading_days = underlying.index
        results: Dict[str, LoadedAnchor] = {}
        for anchor in self.anchors:
            windows = generate_episode_windows(
                anchor,
                trading_days,
                self.episode_cfg.days,
                self.episode_cfg.stride_days,
            )
            if not windows:
                continue
            batch, tags = self._build_batch(anchor, underlying, windows)
            results[anchor.name] = LoadedAnchor(anchor=anchor, batch=batch, tags=tags)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(config: Mapping[str, object]) -> LoaderConfig:
        anchors = list(config.get("anchors", []))
        episode = dict(config.get("episode", {}))
        symbols = dict(config.get("symbols", {}))
        vendor = dict(config.get("vendor", {}))
        seed = int(config.get("seed", config.get("train", {}).get("seed", 0)))
        source = str(config.get("source", "real"))
        return LoaderConfig(
            source=source,
            anchors=anchors,
            episode=episode,
            symbols=symbols,
            vendor=vendor,
            seed=seed,
        )

    def _load_underlying(self) -> pd.DataFrame:
        if self._underlying_cache is not None:
            return self._underlying_cache
        path = self.root / f"{self.symbols.underlying}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Underlying CSV not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        if "spot" not in df.columns:
            if "close" in df.columns:
                df = df.rename(columns={"close": "spot"})
            else:
                raise ValueError("Underlying CSV must contain 'spot' or 'close' column")
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(self.episode_cfg.tz)
        df = df.set_index("date")
        option_cols = [c for c in ("option_price", "mid_price") if c in df.columns]
        if "option_price" not in df.columns and option_cols:
            df["option_price"] = df[option_cols[0]]
        if "implied_vol" not in df.columns:
            df["implied_vol"] = np.nan
        self._has_option_prices = "option_price" in df.columns
        if not self._has_option_prices and not self._missing_option_warned:
            warnings.warn(
                "Option price column missing; falling back to underlying-only episodes.",
                UserWarning,
                stacklevel=2,
            )
            self._missing_option_warned = True
        self._underlying_cache = df
        return df

    def _build_batch(
        self,
        anchor: AnchorSpec,
        data: pd.DataFrame,
        windows: Iterable[EpisodeWindow],
    ) -> tuple[EpisodeBatch, List[EpisodeTag]]:
        episodes: List[np.ndarray] = []
        option_values: List[np.ndarray] = []
        vol_values: List[np.ndarray] = []
        tags: List[EpisodeTag] = []
        for window in windows:
            if anchor.split == "train" and self._window_overlaps_exclusion(anchor, window):
                continue
            segment = data.iloc[window.start_index : window.end_index + 1]
            prices = segment["spot"].to_numpy(dtype=np.float32)
            option = segment.get(
                "option_price",
                pd.Series(np.zeros(len(segment), dtype=np.float32), index=segment.index),
            )
            implied_vol = segment.get(
                "implied_vol",
                pd.Series(np.full(len(segment), np.nan, dtype=np.float32), index=segment.index),
            )
            option_arr = option.to_numpy(dtype=np.float32)
            vol_arr = np.nan_to_num(implied_vol.to_numpy(dtype=np.float32))
            episodes.append(prices)
            option_values.append(option_arr)
            vol_values.append(vol_arr)
            tags.append(
                build_episode_tag(
                    episode_id=window.episode_id,
                    regime_name=anchor.name,
                    split=anchor.split,
                    source=self.config.source,
                    start_date=window.start.strftime("%Y-%m-%d"),
                    end_date=window.end.strftime("%Y-%m-%d"),
                    symbol_root=self.symbols.underlying,
                    seed=self.seed,
                    extra={
                        "anchor_start": anchor.start.strftime("%Y-%m-%d"),
                        "anchor_end": anchor.end.strftime("%Y-%m-%d"),
                    },
                )
            )
        if not episodes:
            raise ValueError(f"No episodes generated for anchor '{anchor.name}'")
        spot_tensor = torch.from_numpy(np.stack(episodes, axis=0))
        option_tensor = torch.from_numpy(np.stack(option_values, axis=0))
        vol_tensor = torch.from_numpy(np.stack(vol_values, axis=0))
        length = spot_tensor.shape[1]
        tau = torch.linspace(length, 1, steps=length, dtype=torch.float32) / 252.0
        tau = tau.unsqueeze(0).repeat(spot_tensor.shape[0], 1)
        batch = EpisodeBatch(
            spot=spot_tensor,
            option_price=option_tensor,
            implied_vol=vol_tensor,
            time_to_maturity=tau,
            rate=0.0,
            env_name=anchor.name,
            meta={
                "split": anchor.split,
                "regime_name": anchor.name,
                "source": self.config.source,
                "linear_bps": float(anchor.metadata.get("linear_bps", 0.0)),
                "quadratic": float(anchor.metadata.get("quadratic", 0.0)),
                "slippage_multiplier": float(anchor.metadata.get("slippage_multiplier", 1.0)),
                "notional": float(anchor.metadata.get("notional", 1.0)),
                "anchor_start": anchor.start.strftime("%Y-%m-%d"),
                "anchor_end": anchor.end.strftime("%Y-%m-%d"),
            },
        )
        batch = attach_tags(batch, tags)
        return batch, tags

    def _build_exclusion_windows(self) -> Dict[str, List[tuple[datetime, datetime]]]:
        exclusions: Dict[str, List[tuple[datetime, datetime]]] = {}
        for anchor in self.anchors:
            other_windows = [
                (other.start, other.end)
                for other in self.anchors
                if other.name != anchor.name and other.split != anchor.split
            ]
            exclusions[anchor.name] = other_windows
        return exclusions

    def _window_overlaps_exclusion(self, anchor: AnchorSpec, window: EpisodeWindow) -> bool:
        bounds = self._exclusion_windows.get(anchor.name, [])
        for start, end in bounds:
            if window.end >= start and window.start <= end:
                return True
        return False


def load_real_anchors(config: Mapping[str, object]) -> Dict[str, LoadedAnchor]:
    """Convenience wrapper returning the loader outputs."""

    loader = RealAnchorLoader(config)
    return loader.load()


class RealAnchorDataModule:
    """Adapter exposing the loader via the SyntheticDataModule-style API."""

    def __init__(self, config: Mapping[str, object]):
        self.loader = RealAnchorLoader(config)
        self._loaded: Optional[Dict[str, LoadedAnchor]] = None

    @property
    def env_order(self) -> List[str]:
        return [anchor.name for anchor in self.loader.anchors]

    def _ensure_loaded(self) -> Dict[str, LoadedAnchor]:
        if self._loaded is None:
            self._loaded = self.loader.load()
        return self._loaded

    def prepare(self, split: str, env_names: Iterable[str]) -> Dict[str, EpisodeBatch]:
        loaded = self._ensure_loaded()
        missing = [name for name in env_names if name not in loaded]
        if missing:
            raise KeyError(f"Requested anchors not available: {missing}")
        return {name: loaded[name].batch for name in env_names}
