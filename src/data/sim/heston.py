"""Deterministic Heston regime simulator utilities."""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from ...markets.pricing import black_scholes_price
from ..types import EpisodeBatch


REGIME_ORDER: Sequence[str] = ("low", "med", "high", "crisis")
SPLIT_OFFSETS: Dict[str, int] = {"train": 0, "val": 10_000, "test": 20_000, "hourly": 30_000}


def _stable_seed_offset(label: str) -> int:
    """Return a deterministic 32-bit offset for the provided label."""

    return zlib.adler32(label.encode("utf-8")) & 0xFFFFFFFF


@dataclass(frozen=True)
class RegimeParams:
    """Configuration describing the Heston parameters for a volatility regime."""

    mu: float
    long_term_var: float
    mean_rev: float
    vol_of_vol: float
    rho: float
    v0: float
    dt: float = 1.0


REGIME_DEFAULTS: Dict[str, RegimeParams] = {
    "low": RegimeParams(mu=0.05, long_term_var=0.03, mean_rev=1.6, vol_of_vol=0.2, rho=-0.1, v0=0.03),
    "med": RegimeParams(mu=0.03, long_term_var=0.06, mean_rev=1.2, vol_of_vol=0.25, rho=-0.25, v0=0.06),
    "high": RegimeParams(mu=0.0, long_term_var=0.16, mean_rev=0.9, vol_of_vol=0.35, rho=-0.35, v0=0.18),
    "crisis": RegimeParams(mu=-0.05, long_term_var=0.36, mean_rev=0.5, vol_of_vol=0.55, rho=-0.7, v0=0.5),
}


class HestonGenerator:
    """Euler-Maruyama simulator for the Heston stochastic volatility model."""

    def __init__(self, params: RegimeParams):
        self.mu = params.mu
        self.long_term_var = params.long_term_var
        self.mean_rev = params.mean_rev
        self.vol_of_vol = params.vol_of_vol
        self.rho = params.rho
        self.v0 = max(params.v0, 1e-6)
        self.dt = params.dt / 252.0

    def sample_paths(self, n_paths: int, steps: int, spot0: float, rng: np.random.Generator) -> np.ndarray:
        spot = np.zeros((n_paths, steps + 1), dtype=np.float64)
        var = np.zeros((n_paths, steps + 1), dtype=np.float64)
        spot[:, 0] = spot0
        var[:, 0] = self.v0
        sqrt_dt = math.sqrt(self.dt)
        for t in range(steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(max(1.0 - self.rho ** 2, 1e-8)) * z2
            vt = np.clip(var[:, t], 1e-8, None)
            var[:, t + 1] = np.clip(
                vt + self.mean_rev * (self.long_term_var - vt) * self.dt + self.vol_of_vol * np.sqrt(vt) * sqrt_dt * z2,
                1e-8,
                None,
            )
            spot[:, t + 1] = spot[:, t] * np.exp((self.mu - 0.5 * vt) * self.dt + np.sqrt(vt) * sqrt_dt * z1)
        return spot


def _apply_jump_diffusion(
    spot_paths: np.ndarray,
    rng: np.random.Generator,
    intensity: float = 0.10,
    mean: float = -0.04,
    std: float = 0.08,
    dt: float = 1.0 / 252.0,
) -> np.ndarray:
    """Inject jump diffusion shocks into the provided spot paths.

    The ``intensity`` parameter represents the probability of a jump per step,
    so setting it to ``0.10`` produces jumps on roughly 10% of days.
    """

    if intensity <= 0.0:
        return spot_paths
    log_paths = np.log(np.clip(spot_paths, 1e-12, None))
    log_returns = log_paths[:, 1:] - log_paths[:, :-1]
    jump_mask = rng.uniform(size=log_returns.shape) < intensity
    jump_sizes = rng.normal(loc=mean, scale=std, size=log_returns.shape) * jump_mask
    log_returns = log_returns + jump_sizes
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(log_returns, axis=1)
    return np.exp(log_paths)


def _implied_volatility_series(spot_paths: np.ndarray, dt: float) -> np.ndarray:
    init = max(float(np.std(np.log(spot_paths[:, 1] / spot_paths[:, 0])) / math.sqrt(dt)), 1e-6)
    vols = np.zeros_like(spot_paths)
    vols[:, 0] = init
    inv_sqrt_dt = 1.0 / math.sqrt(max(dt, 1e-12))
    for t in range(1, spot_paths.shape[1]):
        ret = np.log(spot_paths[:, t] / spot_paths[:, t - 1])
        vols[:, t] = np.clip(np.std(ret, ddof=0) * inv_sqrt_dt, 1e-6, None)
    return np.where(np.isfinite(vols), vols, init)


def _time_grid(steps: int, maturity_days: int) -> np.ndarray:
    tau = np.maximum(maturity_days - np.arange(steps + 1), 1) / 252.0
    return np.broadcast_to(tau, (1, steps + 1))


class HestonRegimeSimulator:
    """Synthetic generator that produces deterministic volatility regimes."""

    def __init__(
        self,
        *,
        base_seed: int,
        spot0: float,
        rate: float,
        use_jumps: bool = False,
        use_liquidity_spread: bool = False,
        regime_params: Dict[str, RegimeParams] | None = None,
    ) -> None:
        self.base_seed = int(base_seed)
        self.spot0 = float(spot0)
        self.rate = float(rate)
        self.use_jumps = bool(use_jumps)
        self.use_liquidity_spread = bool(use_liquidity_spread)
        self.regime_params = dict(REGIME_DEFAULTS if regime_params is None else regime_params)

    def regime_schedule(self, split: str, total: int) -> List[str]:
        offset_seed = (self.base_seed + SPLIT_OFFSETS.get(split, 0)) & 0xFFFFFFFF
        rng = np.random.default_rng(offset_seed)
        schedule: List[str] = []
        order = list(REGIME_ORDER)
        while len(schedule) < total:
            rng.shuffle(order)
            schedule.extend(order)
        return schedule[:total]

    def _rng(self, split: str, regime: str, index: int = 0) -> np.random.Generator:
        seed = (
            int(self.base_seed)
            + SPLIT_OFFSETS.get(split, 0)
            + _stable_seed_offset(regime)
            + index
        ) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def generate_batch(
        self,
        *,
        split: str,
        regime: str,
        num_episodes: int,
        steps: int,
        maturity_days: int,
        cost_meta: Dict[str, float],
        strike: float | None = None,
    ) -> EpisodeBatch:
        if regime not in self.regime_params:
            raise KeyError(f"Unknown regime '{regime}'")
        params = self.regime_params[regime]
        rng = self._rng(split, regime)
        generator = HestonGenerator(params)
        spots = generator.sample_paths(num_episodes, steps, self.spot0 if strike is None else strike, rng)
        if self.use_jumps:
            spots = _apply_jump_diffusion(spots, rng, dt=generator.dt)
        implied_vol = _implied_volatility_series(spots, generator.dt)
        strike_vals = np.full((num_episodes, 1), self.spot0 if strike is None else strike)
        time_grid = np.broadcast_to(_time_grid(steps, maturity_days), (num_episodes, steps + 1)).copy()
        option_prices = np.zeros_like(spots)
        for t in range(steps + 1):
            tau = np.clip(time_grid[:, t], 1e-6, None)
            sigma_t = np.clip(implied_vol[:, t], 1e-4, None)
            option_prices[:, t] = black_scholes_price(
                spots[:, t], strike_vals[:, 0], self.rate, sigma_t, tau, option_type="call"
            )
        meta = {
            "linear_bps": float(cost_meta.get("linear_bps", cost_meta.get("linear", 0.0))),
            "quadratic": float(cost_meta.get("quadratic", 0.0)),
            "slippage_multiplier": float(cost_meta.get("slippage_multiplier", 1.0)),
            "notional": float(cost_meta.get("notional", 1.0)),
            "regime": regime,
        }
        if self.use_liquidity_spread:
            spread_bps = {
                "low": 5.0,
                "med": 7.5,
                "high": 15.0,
                "crisis": 30.0,
            }.get(regime, 10.0)
            meta["liquidity_spread_bps"] = spread_bps
        else:
            meta["liquidity_spread_bps"] = 0.0
        return EpisodeBatch(
            spot=torch.from_numpy(spots).float(),
            option_price=torch.from_numpy(option_prices).float(),
            implied_vol=torch.from_numpy(implied_vol).float(),
            time_to_maturity=torch.from_numpy(time_grid).float(),
            rate=self.rate,
            env_name=regime,
            meta=meta,
        )


def environment_cost_meta(cost_cfg: Dict[str, float]) -> Dict[str, float]:
    """Normalize the cost metadata dictionary for downstream environments."""

    return {
        "linear_bps": float(cost_cfg.get("linear_bps", cost_cfg.get("linear", 0.0))),
        "quadratic": float(cost_cfg.get("quadratic", 0.0)),
        "slippage_multiplier": float(cost_cfg.get("slippage_multiplier", 1.0)),
    }


def steps_from_env(env_cfg: Dict[str, Dict]) -> int:
    episode = env_cfg.get("episode", {})
    return int(episode.get("length_days", 60))


def maturity_from_env(env_cfg: Dict[str, Dict]) -> int:
    options = env_cfg.get("options", {})
    return int(options.get("maturity_days", steps_from_env(env_cfg)))

