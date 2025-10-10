"""Synthetic path generation utilities for single-asset hedging experiments."""
from __future__ import annotations

from typing import Dict, Iterable

from .sim.heston import (
    HestonRegimeSimulator,
    environment_cost_meta,
    maturity_from_env,
    steps_from_env,
)
from .types import EpisodeBatch


class SyntheticDataModule:
    """Utility to prepare synthetic datasets for each environment split."""

    def __init__(self, config: Dict, env_cfgs: Dict[str, Dict], cost_cfgs: Dict[str, Dict]):
        self.config = config
        self.env_cfgs = env_cfgs
        self.cost_cfgs = cost_cfgs
        self.pipeline = HestonRegimeSimulator(
            base_seed=int(config.get("seed", 0)),
            spot0=float(config.get("spot_init", 100.0)),
            rate=float(config.get("rate", 0.01)),
            use_jumps=bool(config.get("use_jumps", False)),
            use_liquidity_spread=bool(config.get("use_liquidity_spread", False)),
        )

    def prepare(self, split: str, env_names: Iterable[str]) -> Dict[str, EpisodeBatch]:
        episodes_per_env = {
            "train": self.config.get("train_episodes", 1024),
            "val": self.config.get("val_episodes", 256),
            "test": self.config.get("test_episodes", 256),
            "hourly": self.config.get("hourly_episodes", 128),
        }
        num = episodes_per_env.get(split, episodes_per_env["train"])
        batches: Dict[str, EpisodeBatch] = {}
        env_list = list(env_names)
        for name in env_list:
            env_cfg = self.env_cfgs[name]
            cost_cfg = self.cost_cfgs[env_cfg["costs"]["file"]]
            cost_meta = environment_cost_meta(cost_cfg)
            cost_meta["notional"] = float(env_cfg.get("episode", {}).get("notional", 1.0))
            maturity_days = maturity_from_env(env_cfg)
            steps = steps_from_env(env_cfg)
            batch = self.pipeline.generate_batch(
                split=split,
                regime=name,
                num_episodes=int(num),
                steps=steps,
                maturity_days=maturity_days,
                cost_meta=cost_meta,
            )
            batches[name] = batch
        return batches

    def hourly_dataset(self, env_name: str) -> EpisodeBatch:
        env_cfg = self.env_cfgs[env_name]
        cost_cfg = self.cost_cfgs[env_cfg["costs"]["file"]]
        cost_meta = environment_cost_meta(cost_cfg)
        cost_meta["notional"] = float(env_cfg.get("episode", {}).get("notional", 1.0))
        return self.pipeline.generate_batch(
            split="hourly",
            regime=env_name,
            num_episodes=int(self.config.get("hourly_episodes", 128)),
            steps=steps_from_env(env_cfg),
            maturity_days=maturity_from_env(env_cfg),
            cost_meta=cost_meta,
        )
