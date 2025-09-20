from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import torch

from .generator import SimulationResult, TransactionCostSpec, build_simulator_from_config
from .real_market import RealMarketLoader


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
        self.metadata: Dict[str, Any] = dict(result.metadata or {})
        self.metadata.setdefault("env", self.env)

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


def _compute_stats_from_datasets(datasets: Iterable[RegimeDataset]) -> Dict[str, FeatureStats]:
    items = list(datasets)
    if not items:
        raise ValueError("No datasets available to compute feature statistics.")
    delta_tensor = torch.cat([ds.delta for ds in items], dim=0)
    gamma_tensor = torch.cat([ds.gamma for ds in items], dim=0)
    tau_tensor = torch.cat([torch.full_like(ds.delta, 60.0 / 252.0) for ds in items], dim=0)
    vol_tensor = torch.cat([ds.realized_vol for ds in items], dim=0)
    stats = {
        "delta": FeatureStats(delta_tensor.mean(), delta_tensor.std(unbiased=False)),
        "gamma": FeatureStats(gamma_tensor.mean(), gamma_tensor.std(unbiased=False)),
        "tau": FeatureStats(tau_tensor.mean(), tau_tensor.std(unbiased=False)),
        "realized_vol": FeatureStats(vol_tensor.mean(), vol_tensor.std(unbiased=False)),
        "inventory": FeatureStats(torch.tensor(0.0), torch.tensor(1.0)),
    }
    return stats


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
            metadata = dict(result.metadata or {})
            metadata.update({
                "env": result.env,
                "variant": "jump",
                "base_env": base,
                "source": metadata.get("source", "synthetic"),
            })
            result.metadata = metadata
            return result, result.env
        if env == "liquidity":
            base = self.config["test_environment"]
            result = self.simulator.generate(base, self.config["test_episodes"], seed=int(rng.integers(0, 1_000_000)))
            stress_cost = self.config.get("liquidity_environments", {})
            linear = stress_cost.get("crisis_spread", result.tx_cost.linear)
            quadratic = max(result.tx_cost.quadratic, stress_cost.get("stress_quadratic", 0.2))
            result.tx_cost = TransactionCostSpec(linear=float(linear), quadratic=float(quadratic))
            result.env = f"{base}_liquidity"
            metadata = dict(result.metadata or {})
            metadata.update({
                "env": result.env,
                "variant": "liquidity",
                "base_env": base,
                "tx_cost_override": {"linear": float(linear), "quadratic": float(quadratic)},
                "source": metadata.get("source", "synthetic"),
            })
            result.metadata = metadata
            return result, result.env
        result = self.simulator.generate(env, self.config["test_episodes"], seed=int(rng.integers(0, 1_000_000)))
        metadata = dict(result.metadata or {})
        metadata.setdefault("env", env)
        result.metadata = metadata
        return result, env

    def _compute_feature_stats(self) -> Dict[str, FeatureStats]:
        datasets = [self.datasets[env] for env in self.train_envs]
        return _compute_stats_from_datasets(datasets)

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


class RealMarketDataModule:
    def __init__(
        self,
        config: Dict,
        seed: int,
        device: torch.device,
    ) -> None:
        del seed  # Real data loading is deterministic
        self.config = config
        self.device = device
        self.real_config = config.get("real_data")
        if not self.real_config:
            raise ValueError("data.real_data must be provided when dataset_type='real' or 'combined'.")
        episode_length = int(config.get("episode_length", 60))
        self.loader = RealMarketLoader(self.real_config, episode_length)
        self.train_envs: List[str] = []
        self.val_envs: List[str] = []
        self.test_envs: List[str] = []
        self.datasets: Dict[str, RegimeDataset] = {}
        self.val_datasets: Dict[str, RegimeDataset] = {}
        self.test_datasets: Dict[str, RegimeDataset] = {}
        self._build_splits()
        self.feature_stats = self._compute_feature_stats()

    def _window_definitions(self, key: str) -> List[Dict[str, Any]]:
        windows = self.real_config.get(key, [])
        return [dict(window) for window in windows]

    def _build_group(
        self,
        key: str,
        tag: str,
        target: Dict[str, RegimeDataset],
        collector: List[str],
    ) -> None:
        for window in self._window_definitions(key):
            result = self.loader.build_window(window, tag)
            dataset = RegimeDataset(result)
            target[result.env] = dataset
            collector.append(result.env)

    def _build_splits(self) -> None:
        self._build_group("train_windows", "train", self.datasets, self.train_envs)
        self._build_group("val_windows", "validation", self.val_datasets, self.val_envs)
        self._build_group("test_windows", "test", self.test_datasets, self.test_envs)
        self._build_group("crisis_windows", "crisis", self.test_datasets, self.test_envs)

    def _compute_feature_stats(self) -> Dict[str, FeatureStats]:
        datasets: List[RegimeDataset] = [self.datasets[name] for name in self.train_envs]
        if not datasets:
            datasets = list(self.datasets.values())
        if not datasets:
            datasets = list(self.val_datasets.values())
        if not datasets:
            datasets = list(self.test_datasets.values())
        if not datasets:
            raise ValueError("RealMarketDataModule could not find any datasets to compute statistics.")
        return _compute_stats_from_datasets(datasets)

    def sample_train_batches(self, batch_size: int) -> Dict[str, EpisodeBatch]:
        if not self.train_envs:
            return {}
        per_env = max(batch_size // len(self.train_envs), 1)
        return {env: self.datasets[env].sample(per_env, self.device) for env in self.train_envs}

    def sample_validation(self, env: Optional[str] = None) -> Dict[str, EpisodeBatch]:
        if not self.val_envs:
            return {}
        if env is not None:
            if env not in self.val_datasets:
                raise KeyError(f"Unknown validation environment: {env}")
            return {env: self.val_datasets[env].full(self.device)}
        return {name: ds.full(self.device) for name, ds in self.val_datasets.items()}

    def test_sets(self) -> Dict[str, RegimeDataset]:
        return self.test_datasets

    def get_feature_stats(self) -> Dict[str, FeatureStats]:
        return self.feature_stats


class CombinedDataModule:
    def __init__(
        self,
        config: Dict,
        seed: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.synthetic = VolatilityRegimeDataModule(config, seed, device)
        self.real = RealMarketDataModule(config, seed, device)
        self.combined_config = config.get("combined", {})
        self.train_envs = list(self.synthetic.train_envs) + list(self.real.train_envs)
        self.feature_stats = self._compute_feature_stats()

    def _compute_feature_stats(self) -> Dict[str, FeatureStats]:
        datasets: List[RegimeDataset] = [self.synthetic.datasets[env] for env in self.synthetic.train_envs]
        datasets.extend(self.real.datasets[env] for env in self.real.train_envs)
        if not datasets:
            if self.synthetic.datasets:
                datasets = list(self.synthetic.datasets.values())
            elif self.real.datasets or self.real.val_datasets or self.real.test_datasets:
                datasets = list(self.real.datasets.values())
                if not datasets:
                    datasets = list(self.real.val_datasets.values())
                if not datasets:
                    datasets = list(self.real.test_datasets.values())
        if not datasets:
            raise ValueError("CombinedDataModule has no datasets available for feature statistics.")
        return _compute_stats_from_datasets(datasets)

    def _split_batch_size(self, batch_size: int) -> Tuple[int, int, bool, bool]:
        syn_ratio_raw = max(float(self.combined_config.get("synthetic_batch_ratio", 0.5)), 0.0)
        real_ratio_raw = max(float(self.combined_config.get("real_batch_ratio", 0.5)), 0.0)
        include_synthetic = bool(self.synthetic.train_envs) and (
            syn_ratio_raw > 0.0 or not self.real.train_envs or (syn_ratio_raw == 0.0 and real_ratio_raw == 0.0)
        )
        include_real = bool(self.real.train_envs) and (
            real_ratio_raw > 0.0 or not self.synthetic.train_envs or (syn_ratio_raw == 0.0 and real_ratio_raw == 0.0)
        )
        total = syn_ratio_raw + real_ratio_raw
        if total == 0.0:
            syn_share = 0.5 if include_synthetic else 0.0
            real_share = 0.5 if include_real else 0.0
        else:
            syn_share = syn_ratio_raw / total if include_synthetic else 0.0
            real_share = real_ratio_raw / total if include_real else 0.0
        synthetic_size = int(round(batch_size * syn_share)) if include_synthetic else 0
        real_size = int(round(batch_size * real_share)) if include_real else 0
        return synthetic_size, real_size, include_synthetic, include_real

    def sample_train_batches(self, batch_size: int) -> Dict[str, EpisodeBatch]:
        if not self.train_envs:
            return {}
        synthetic_size, real_size, include_synthetic, include_real = self._split_batch_size(batch_size)
        batches: Dict[str, EpisodeBatch] = {}
        if include_synthetic:
            size = max(synthetic_size, len(self.synthetic.train_envs))
            batches.update(self.synthetic.sample_train_batches(size))
        if include_real:
            size = max(real_size, len(self.real.train_envs))
            batches.update(self.real.sample_train_batches(size))
        return batches

    def sample_validation(self, env: Optional[str] = None) -> Dict[str, EpisodeBatch]:
        if env is not None:
            if env in self.synthetic.val_datasets:
                return self.synthetic.sample_validation(env)
            if env in self.real.val_datasets:
                return self.real.sample_validation(env)
            raise KeyError(f"Unknown validation environment: {env}")
        batches: Dict[str, EpisodeBatch] = {}
        if self.synthetic.val_envs:
            batches.update(self.synthetic.sample_validation())
        if self.real.val_envs:
            batches.update(self.real.sample_validation())
        return batches

    def test_sets(self) -> Dict[str, RegimeDataset]:
        datasets: Dict[str, RegimeDataset] = {}
        datasets.update(self.synthetic.test_sets())
        datasets.update(self.real.test_sets())
        return datasets

    def get_feature_stats(self) -> Dict[str, FeatureStats]:
        return self.feature_stats


class DataModule(Protocol):
    train_envs: List[str]

    def sample_train_batches(self, batch_size: int) -> Dict[str, EpisodeBatch]:
        ...

    def sample_validation(self, env: Optional[str] = None) -> Dict[str, EpisodeBatch]:
        ...

    def test_sets(self) -> Dict[str, RegimeDataset]:
        ...

    def get_feature_stats(self) -> Dict[str, FeatureStats]:
        ...


def create_data_module(config: Dict, seed: int, device: torch.device) -> DataModule:
    dataset_type = config.get("dataset_type", "synthetic")
    if dataset_type == "synthetic":
        return VolatilityRegimeDataModule(config, seed, device)
    if dataset_type == "real":
        return RealMarketDataModule(config, seed, device)
    if dataset_type == "combined":
        return CombinedDataModule(config, seed, device)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")
