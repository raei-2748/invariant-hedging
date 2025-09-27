"""Synthetic path generation utilities for single-asset hedging experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from ..markets.pricing import black_scholes_price

ArrayLike = np.ndarray


@dataclass
class EpisodeBatch:
    """Container for a batch of simulated episodes."""

    spot: torch.Tensor  # shape: [batch, steps + 1]
    option_price: torch.Tensor  # shape: [batch, steps + 1]
    implied_vol: torch.Tensor  # shape: [batch, steps + 1]
    time_to_maturity: torch.Tensor  # shape: [batch, steps + 1]
    rate: float
    env_name: str
    meta: Dict[str, float]

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


class GBMGenerator:
    """Generate geometric Brownian motion price paths."""

    def __init__(self, params: Dict[str, float]):
        self.mu = params.get("mu", 0.0)
        self.sigma = params.get("sigma", 0.2)
        self.dt = params.get("dt", 1.0) / 252.0

    def sample_paths(self, n_paths: int, steps: int, spot0: float, seed: Optional[int] = None) -> ArrayLike:
        rng = np.random.default_rng(seed)
        normals = rng.standard_normal(size=(n_paths, steps))
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * normals
        log_returns = drift + diffusion
        log_paths = np.log(spot0) + np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
        return np.exp(log_paths)


class HestonGenerator:
    """Rough Euler-Maruyama scheme for the Heston stochastic volatility model."""

    def __init__(self, params: Dict[str, float]):
        self.mu = params.get("mu", 0.0)
        self.long_term_var = params.get("long_term_var", 0.2)
        self.mean_rev = params.get("mean_rev", 1.0)
        self.vol_of_vol = params.get("vol_of_vol", params.get("vov", 0.3))
        self.rho = params.get("rho", -0.3)
        self.v0 = max(params.get("v0", self.long_term_var), 1e-6)
        self.dt = params.get("dt", 1.0) / 252.0

    def sample_paths(self, n_paths: int, steps: int, spot0: float, seed: Optional[int] = None) -> ArrayLike:
        rng = np.random.default_rng(seed)
        spot = np.zeros((n_paths, steps + 1), dtype=np.float64)
        var = np.zeros((n_paths, steps + 1), dtype=np.float64)
        spot[:, 0] = spot0
        var[:, 0] = self.v0
        sqrt_dt = math.sqrt(self.dt)
        for t in range(steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + math.sqrt(1.0 - self.rho ** 2) * z2
            vt = np.clip(var[:, t], 1e-8, None)
            var[:, t + 1] = np.clip(
                vt
                + self.mean_rev * (self.long_term_var - vt) * self.dt
                + self.vol_of_vol * np.sqrt(vt) * sqrt_dt * z2,
                1e-8,
                None,
            )
            spot[:, t + 1] = spot[:, t] * np.exp(
                (self.mu - 0.5 * vt) * self.dt + np.sqrt(vt) * sqrt_dt * z1
            )
        return spot


def _make_generator(model: str, params: Dict[str, float]):
    if model.lower() == "gbm":
        return GBMGenerator(params)
    if model.lower() == "heston":
        return HestonGenerator(params)
    raise ValueError(f"Unsupported dynamics model: {model}")


def generate_episode_batch(
    env_name: str,
    env_cfg: Dict[str, Dict],
    costs_cfg: Dict[str, float],
    num_episodes: int,
    spot0: float,
    rate: float,
    seed: int,
) -> EpisodeBatch:
    episode_cfg = env_cfg["episode"]
    steps = int(episode_cfg["length_days"])
    dynamics_cfg = env_cfg["dynamics"]
    generator = _make_generator(dynamics_cfg["model"], dynamics_cfg["params"])
    spot_paths = generator.sample_paths(num_episodes, steps, spot0, seed=seed)

    maturity_days = env_cfg.get("options", {}).get("maturity_days", steps)
    rate_env = env_cfg.get("options", {}).get("rate", rate)

    time_to_maturity = np.maximum(
        maturity_days - np.arange(steps + 1), 1
    ) / 252.0
    time_grid = np.broadcast_to(time_to_maturity, (num_episodes, steps + 1))

    # Use instantaneous variance from log returns for GBM, or from Heston generator if available.
    implied_vol = np.zeros_like(spot_paths)
    model_name = dynamics_cfg["model"].lower()
    params = dynamics_cfg.get("params", {})
    if model_name == "gbm":
        init_vol = float(params.get("sigma", 0.2))
    elif model_name == "heston":
        init_var = float(params.get("v0", params.get("long_term_var", 0.2)))
        init_vol = math.sqrt(max(init_var, 1e-8))
    else:
        init_vol = 0.2
    implied_vol[:, 0] = max(init_vol, 1e-6)

    dt = getattr(generator, "dt", 1.0 / 252.0)
    inv_sqrt_dt = 1.0 / math.sqrt(max(dt, 1e-12))
    for t in range(1, steps + 1):
        # Cross-sectional realized volatility scaled to annualized units.
        ret = np.log(spot_paths[:, t] / spot_paths[:, t - 1])
        realized = np.std(ret, ddof=0) * inv_sqrt_dt
        implied_vol[:, t] = np.clip(realized, 1e-6, None)

    implied_vol = np.where(np.isfinite(implied_vol), implied_vol, init_vol)

    strike_mode = env_cfg.get("options", {}).get("strike_mode", "atm")
    if strike_mode == "atm":
        strikes = spot_paths[:, [0]]
    else:
        strikes = np.full((num_episodes, 1), env_cfg["options"].get("strike", spot0))

    option_prices = np.zeros_like(spot_paths)
    for t in range(steps + 1):
        tau = np.clip(time_grid[:, t], 1e-6, None)
        sigma_t = np.clip(implied_vol[:, t], 1e-4, None)
        option_prices[:, t] = black_scholes_price(
            spot_paths[:, t], strikes[:, 0], rate_env, sigma_t, tau, option_type="call"
        )

    return EpisodeBatch(
        spot=torch.from_numpy(spot_paths).float(),
        option_price=torch.from_numpy(option_prices).float(),
        implied_vol=torch.from_numpy(implied_vol).float(),
        time_to_maturity=torch.from_numpy(np.ascontiguousarray(time_grid)).float(),
        rate=rate_env,
        env_name=env_name,
        meta={
            "linear_bps": float(costs_cfg.get("linear_bps", 0.0)),
            "quadratic": float(costs_cfg.get("quadratic", 0.0)),
            "slippage_multiplier": float(costs_cfg.get("slippage_multiplier", 0.0)),
            "notional": float(env_cfg.get("episode", {}).get("notional", 1.0)),
        },
    )


class SyntheticDataModule:
    """Utility to prepare synthetic datasets for each environment split."""

    def __init__(self, config: Dict, env_cfgs: Dict[str, Dict], cost_cfgs: Dict[str, Dict]):
        self.config = config
        self.env_cfgs = env_cfgs
        self.cost_cfgs = cost_cfgs

    def prepare(self, split: str, env_names: Iterable[str]) -> Dict[str, EpisodeBatch]:
        episodes_per_env = {
            "train": self.config.get("train_episodes", 1024),
            "val": self.config.get("val_episodes", 256),
            "test": self.config.get("test_episodes", 256),
            "hourly": self.config.get("hourly_episodes", 128),
        }
        num = episodes_per_env.get(split, episodes_per_env["train"])
        batches: Dict[str, EpisodeBatch] = {}
        split_offsets = {"train": 0, "val": 1, "test": 2, "hourly": 3}
        base_seed = int(self.config.get("seed", 0)) + split_offsets.get(split, 0) * 10_000
        for idx, name in enumerate(env_names):
            env_cfg = self.env_cfgs[name]
            cost_cfg = self.cost_cfgs[env_cfg["costs"]["file"]]
            batch = generate_episode_batch(
                env_name=name,
                env_cfg=env_cfg,
                costs_cfg=cost_cfg,
                num_episodes=num,
                spot0=float(self.config.get("spot_init", 100.0)),
                rate=float(self.config.get("rate", 0.01)),
                seed=base_seed + idx,
            )
            batches[name] = batch
        return batches

    def hourly_dataset(self, env_name: str) -> EpisodeBatch:
        env_cfg = self.env_cfgs[env_name]
        cost_cfg = self.cost_cfgs[env_cfg["costs"]["file"]]
        return generate_episode_batch(
            env_name=env_name,
            env_cfg=env_cfg,
            costs_cfg=cost_cfg,
            num_episodes=int(self.config.get("hourly_episodes", 128)),
            spot0=float(self.config.get("spot_init", 100.0)),
            rate=float(self.config.get("rate", 0.01)),
            seed=int(self.config.get("seed", 0)) + 999,
        )
