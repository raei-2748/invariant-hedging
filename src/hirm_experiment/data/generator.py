from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .black_scholes import call_price, greeks


@dataclass
class RegimeSpec:
    name: str
    sigma: float
    vrp: float
    skew: float


@dataclass
class TransactionCostSpec:
    linear: float
    quadratic: float


@dataclass
class SimulationResult:
    env: str
    spot: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    realized_vol: np.ndarray
    implied_vol: np.ndarray
    option_price: np.ndarray
    tx_cost: TransactionCostSpec
    metadata: Optional[Dict[str, Any]] = None


class VolatilityRegimeSimulator:
    def __init__(
        self,
        regime_specs: Dict[str, RegimeSpec],
        costs: Dict[str, TransactionCostSpec],
        episode_length: int,
        dt: float = 1.0 / 252.0,
        jump_config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.regime_specs = regime_specs
        self.costs = costs
        self.episode_length = episode_length
        self.dt = dt
        self.jump_config = jump_config or {}

    def generate(
        self,
        env: str,
        num_episodes: int,
        seed: Optional[int] = None,
        with_jump: bool = False,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed)
        spec = self.regime_specs[env]
        cost = self.costs[env]
        sigma = spec.sigma
        spot_paths = self._simulate_spot(rng, num_episodes, sigma, with_jump)
        tau = np.full((num_episodes, self.episode_length), 60.0 / 252.0)
        sigma_imp = np.full_like(tau, sigma + spec.vrp)
        prices, delta, gamma, theta = self._compute_option_terms(spot_paths[:, :-1], tau, sigma_imp)
        realized_vol = self._compute_realized_vol(spot_paths)
        metadata: Dict[str, Any] = {
            "source": "synthetic",
            "regime": env,
            "sigma": sigma,
            "vrp": spec.vrp,
            "skew": spec.skew,
            "seed": seed,
            "with_jump": with_jump,
            "episode_length": self.episode_length,
        }
        if with_jump and self.jump_config:
            metadata["jump_config"] = dict(self.jump_config)
        return SimulationResult(
            env=env,
            spot=spot_paths,
            delta=delta,
            gamma=gamma,
            theta=theta,
            realized_vol=realized_vol,
            implied_vol=sigma_imp,
            option_price=prices,
            tx_cost=cost,
            metadata=metadata,
        )

    def _simulate_spot(self, rng: np.random.Generator, num_episodes: int, sigma: float, with_jump: bool) -> np.ndarray:
        steps = self.episode_length
        drift = -0.5 * sigma ** 2
        gaussian = rng.standard_normal(size=(num_episodes, steps))
        increments = (drift * self.dt) + sigma * np.sqrt(self.dt) * gaussian
        if with_jump and self.jump_config:
            intensity = self.jump_config.get("intensity", 0.0)
            mean_jump = self.jump_config.get("mean_jump", 0.0)
            std_jump = self.jump_config.get("std_jump", 0.0)
            jump_mask = rng.uniform(size=(num_episodes, steps)) < intensity * self.dt
            jumps = rng.normal(mean_jump, std_jump, size=(num_episodes, steps)) * jump_mask
            increments = increments + jumps
        log_prices = np.zeros((num_episodes, steps + 1))
        log_prices[:, 1:] = np.cumsum(increments, axis=1)
        spot = np.exp(log_prices)
        return spot

    def _compute_option_terms(
        self,
        spot: np.ndarray,
        tau: np.ndarray,
        sigma_imp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        strike = spot
        price = call_price(spot, strike, tau, sigma_imp)
        delta, gamma, theta = greeks(spot, tau, sigma_imp)
        return price, delta, gamma, theta

    def _compute_realized_vol(self, spot: np.ndarray, window: int = 20) -> np.ndarray:
        log_returns = np.diff(np.log(spot), axis=1)
        realized = np.zeros_like(log_returns)
        for t in range(log_returns.shape[1]):
            start = max(0, t - window + 1)
            window_slice = log_returns[:, start : t + 1]
            ddof = 1 if window_slice.shape[1] > 1 else 0
            std = np.std(window_slice, axis=1, ddof=ddof)
            realized[:, t] = std
        annualized = realized * np.sqrt(252.0)
        return annualized


def build_simulator_from_config(config: Dict, episode_length: int) -> VolatilityRegimeSimulator:
    regimes = {
        name: RegimeSpec(name=name, sigma=float(params["sigma"]), vrp=float(params["vrp"]), skew=float(params.get("skew", 0.0)))
        for name, params in config["volatility_bands"].items()
    }
    costs = {
        name: TransactionCostSpec(linear=float(params["linear"]), quadratic=float(params["quadratic"]))
        for name, params in config["tx_costs"].items()
    }
    jump_cfg = config.get("jump_environments", None)
    return VolatilityRegimeSimulator(regimes, costs, episode_length, jump_config=jump_cfg)
