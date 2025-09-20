from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .generator import SimulationResult, TransactionCostSpec, VolatilityRegimeSimulator, build_simulator_from_config


@dataclass
class EpisodeBatch:
    env: str
    spot: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor
    theta: torch.Tensor
    realized_vol: torch.Tensor
    implied_vol: torch.Tensor
    option_price: torch.Tensor
    tx_linear: torch.Tensor
    tx_quadratic: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            env=self.env,
            spot=self.spot.to(device),
            delta=self.delta.to(device),
            gamma=self.gamma.to(device),
            theta=self.theta.to(device),
            realized_vol=self.realized_vol.to(device),
            implied_vol=self.implied_vol.to(device),
            option_price=self.option_price.to(device),
            tx_linear=self.tx_linear.to(device),
            tx_quadratic=self.tx_quadratic.to(device),
        )


class RegimeDataset:
    def __init__(self, result: SimulationResult) -> None:
        self.env = result.env
        self.spot = torch.from_numpy(result.spot).float()
        self.delta = torch.from_numpy(result.delta).float()
        self.gamma = torch.from_numpy(result.gamma).float()
        self.theta = torch.from_numpy(result.theta).float()
        self.realized_vol = torch.from_numpy(result.realized_vol).float()
        self.implied_vol = torch.from_numpy(result.implied_vol).float()
        self.option_price = torch.from_numpy(result.option_price).float()
        horizon = self.spot.shape[1] - 1
        self.tx_linear = torch.full((self.spot.shape[0], horizon), result.tx_cost.linear, dtype=torch.float32)
        self.tx_quadratic = torch.full((self.spot.shape[0], horizon), result.tx_cost.quadratic, dtype=torch.float32)

    @property
    def episodes(self) -> int:
        return self.spot.shape[0]

    @property
    def horizon(self) -> int:
        return self.spot.shape[1] - 1

    def sample(self, batch_size: int, device: torch.device) -> EpisodeBatch:
        idx = torch.randint(0, self.episodes, (batch_size,), dtype=torch.long)
        batch = EpisodeBatch(
            env=self.env,
            spot=self.spot.index_select(0, idx),
            delta=self.delta.index_select(0, idx),
            gamma=self.gamma.index_select(0, idx),
            theta=self.theta.index_select(0, idx),
            realized_vol=self.realized_vol.index_select(0, idx),
            implied_vol=self.implied_vol.index_select(0, idx),
            option_price=self.option_price.index_select(0, idx),
            tx_linear=self.tx_linear.index_select(0, idx),
            tx_quadratic=self.tx_quadratic.index_select(0, idx),
        )
        return batch.to(device)

    def full(self, device: torch.device) -> EpisodeBatch:
        indices = torch.arange(self.episodes, dtype=torch.long)
        batch = EpisodeBatch(
            env=self.env,
            spot=self.spot.index_select(0, indices),
            delta=self.delta.index_select(0, indices),
            gamma=self.gamma.index_select(0, indices),
            theta=self.theta.index_select(0, indices),
            realized_vol=self.realized_vol.index_select(0, indices),
            implied_vol=self.implied_vol.index_select(0, indices),
            option_price=self.option_price.index_select(0, indices),
            tx_linear=self.tx_linear.index_select(0, indices),
            tx_quadratic=self.tx_quadratic.index_select(0, indices),
        )
        return batch.to(device)


@dataclass
class FeatureStats:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(value.device)
        std = torch.clamp(self.std.to(value.device), min=1e-6)
        return (value - mean) / std


class VolatilityRegimeDataModule:
    def __init__(
        self,
        config: Dict,
        seed: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.simulator = build_simulator_from_config(config, config["episode_length"])
        rng = np.random.default_rng(seed)

        train_envs = list(config.get("train_environments", []))
        if not train_envs:
            raise ValueError("VolatilityRegimeDataModule requires at least one training environment.")
        self.train_envs = train_envs

        val_env = config.get("val_environment")
        self.val_envs = [val_env] if val_env else []

        test_env = config.get("test_environment")
        self.test_envs = [test_env] if test_env else []
        self.additional_stress = list(config.get("additional_stress", []))
        self.datasets: Dict[str, RegimeDataset] = {}
        self.val_datasets: Dict[str, RegimeDataset] = {}
        self.test_datasets: Dict[str, RegimeDataset] = {}
        self._build_splits(rng)
        self.feature_stats = self._compute_feature_stats()

    def _build_splits(self, rng: np.random.Generator) -> None:
        for env in self.train_envs:
            result = self.simulator.generate(env, self.config["train_episodes"], seed=int(rng.integers(0, 1_000_000)))
            self.datasets[env] = RegimeDataset(result)
        for env in self.val_envs:
            result = self.simulator.generate(env, self.config["val_episodes"], seed=int(rng.integers(0, 1_000_000)))
            self.val_datasets[env] = RegimeDataset(result)
        stress_targets = self.test_envs + self.additional_stress
        for env in stress_targets:
            result, name = self._generate_test_variant(env, rng)
            self.test_datasets[name] = RegimeDataset(result)

    def _generate_test_variant(self, env: str, rng: np.random.Generator) -> Tuple[SimulationResult, str]:
        if env == "jump":
            base = self.config["test_environment"]
            result = self.simulator.generate(base, self.config["test_episodes"], seed=int(rng.integers(0, 1_000_000)), with_jump=True)
            result.env = f"{base}_jump"
            return result, result.env
        if env == "liquidity":
            base = self.config["test_environment"]
            result = self.simulator.generate(base, self.config["test_episodes"], seed=int(rng.integers(0, 1_000_000)))
            stress_cost = self.config.get("liquidity_environments", {})
            linear = stress_cost.get("crisis_spread", result.tx_cost.linear)
            quadratic = max(result.tx_cost.quadratic, stress_cost.get("stress_quadratic", 0.2))
            result.tx_cost = TransactionCostSpec(linear=float(linear), quadratic=float(quadratic))
            result.env = f"{base}_liquidity"
            return result, result.env
        result = self.simulator.generate(env, self.config["test_episodes"], seed=int(rng.integers(0, 1_000_000)))
        return result, env

    def _compute_feature_stats(self) -> Dict[str, FeatureStats]:
        delta_list, gamma_list, tau_list, vol_list = [], [], [], []
        for env in self.train_envs:
            ds = self.datasets[env]
            delta_list.append(ds.delta)
            gamma_list.append(ds.gamma)
            tau_list.append(torch.full_like(ds.delta, 60.0 / 252.0))
            vol_list.append(ds.realized_vol)
        delta_tensor = torch.cat(delta_list, dim=0)
        gamma_tensor = torch.cat(gamma_list, dim=0)
        tau_tensor = torch.cat(tau_list, dim=0)
        vol_tensor = torch.cat(vol_list, dim=0)
        stats = {
            "delta": FeatureStats(delta_tensor.mean(), delta_tensor.std(unbiased=False)),
            "gamma": FeatureStats(gamma_tensor.mean(), gamma_tensor.std(unbiased=False)),
            "tau": FeatureStats(tau_tensor.mean(), tau_tensor.std(unbiased=False)),
            "realized_vol": FeatureStats(vol_tensor.mean(), vol_tensor.std(unbiased=False)),
            "inventory": FeatureStats(torch.tensor(0.0), torch.tensor(1.0)),
        }
        return stats

    def sample_train_batches(self, batch_size: int) -> Dict[str, EpisodeBatch]:
        if not self.train_envs:
            raise ValueError("No training environments are available for sampling.")
        env_count = len(self.train_envs)
        base = batch_size // env_count
        remainder = batch_size % env_count
        batches: Dict[str, EpisodeBatch] = {}
        for idx, env in enumerate(self.train_envs):
            extra = 1 if idx < remainder else 0
            env_batch = base + extra
            if env_batch <= 0:
                env_batch = 1
            batches[env] = self.datasets[env].sample(env_batch, self.device)
        return batches

    def sample_validation(self, env: Optional[str] = None) -> Dict[str, EpisodeBatch]:
        targets = [env] if env else self.val_envs
        return {name: self.val_datasets[name].full(self.device) for name in targets}

    def test_sets(self) -> Dict[str, RegimeDataset]:
        return self.test_datasets

    def get_feature_stats(self) -> Dict[str, FeatureStats]:
        return self.feature_stats


def create_data_module(config: Dict, seed: int, device: torch.device) -> VolatilityRegimeDataModule:
    return VolatilityRegimeDataModule(config, seed, device)
