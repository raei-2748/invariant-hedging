"""Hydra configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.modules.data.features import FeatureEngineer
from src.modules.data.real.loader import RealAnchorDataModule
from src.modules.data.real.spy_emini import SpyEminiDataModule
from src.modules.data.real_spy_loader import RealSpyDataModule
from src.modules.data.synthetic import SyntheticDataModule
from src.modules.data.types import EpisodeBatch
from src.modules.environment import SingleAssetHedgingEnv


@dataclass(frozen=True)
class DataModuleContext:
    data_module: Any
    env_order: List[str]
    name_to_index: Dict[str, int]
    env_configs: Dict[str, Dict]
    cost_configs: Dict[str, Dict]


def load_yaml_config(path: str) -> Dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def resolve_env_configs(env_group: DictConfig) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[str]]:
    all_names: List[str] = []
    for split in (env_group.train, env_group.val, env_group.test):
        for name in split:
            if name not in all_names:
                all_names.append(name)
    env_cfgs: Dict[str, Dict] = {}
    cost_cfgs: Dict[str, Dict] = {}
    for name in all_names:
        env_cfg = load_yaml_config(to_absolute_path(f"configs/envs/{name}.yaml"))
        env_cfgs[name] = env_cfg
        cost_name = env_cfg["costs"]["file"]
        if cost_name not in cost_cfgs:
            cost_cfgs[cost_name] = load_yaml_config(to_absolute_path(f"configs/costs/{cost_name}.yaml"))
    return env_cfgs, cost_cfgs, all_names


def environment_order(env_group: DictConfig) -> List[str]:
    order: List[str] = []
    for split in (env_group.train, env_group.val, env_group.test):
        for name in split:
            if name not in order:
                order.append(name)
    return order


def unwrap_experiment_config(cfg: DictConfig) -> DictConfig:
    if "train" in cfg and "data" not in cfg:
        return cfg.train
    return cfg


def prepare_data_module(cfg: DictConfig, seed: Optional[int] = None) -> DataModuleContext:
    data_cfg_raw = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_cfg_raw, dict):
        raise TypeError("cfg.data must resolve to a mapping")
    data_cfg: Dict[str, Any] = dict(data_cfg_raw)
    if seed is not None and "seed" not in data_cfg:
        data_cfg["seed"] = int(seed)
    dataset_name = str(data_cfg.get("name", "synthetic"))
    if dataset_name == "synthetic":
        env_cfgs, cost_cfgs, env_order = resolve_env_configs(cfg.envs)
        data_module = SyntheticDataModule(
            config=data_cfg,
            env_cfgs=env_cfgs,
            cost_cfgs=cost_cfgs,
        )
        name_to_index = {name: idx for idx, name in enumerate(env_order)}
        return DataModuleContext(
            data_module=data_module,
            env_order=env_order,
            name_to_index=name_to_index,
            env_configs=env_cfgs,
            cost_configs=cost_cfgs,
        )
    source = str(data_cfg.get("source", "")).lower()
    if source == "real":
        data_module = RealAnchorDataModule(data_cfg)
        env_order = data_module.env_order
        name_to_index = {name: idx for idx, name in enumerate(env_order)}
        return DataModuleContext(
            data_module=data_module,
            env_order=env_order,
            name_to_index=name_to_index,
            env_configs={},
            cost_configs={},
        )
    if dataset_name in {"real", "real_spy"}:
        data_module = SpyEminiDataModule(data_cfg.get("real", data_cfg))
        env_order = data_module.env_order
        name_to_index = {name: idx for idx, name in enumerate(env_order)}
        return DataModuleContext(
            data_module=data_module,
            env_order=env_order,
            name_to_index=name_to_index,
            env_configs={},
            cost_configs={},
        )
    if dataset_name == "real_spy_paper":
        data_module = RealSpyDataModule(data_cfg)
        env_order = data_module.env_order
        name_to_index = {name: idx for idx, name in enumerate(env_order)}
        return DataModuleContext(
            data_module=data_module,
            env_order=env_order,
            name_to_index=name_to_index,
            env_configs={},
            cost_configs={},
        )
    raise ValueError(f"Unsupported dataset name '{dataset_name}'")


def build_envs(
    batches: Dict[str, EpisodeBatch],
    feature_engineer: FeatureEngineer,
    name_to_index: Dict[str, int],
) -> Dict[str, SingleAssetHedgingEnv]:
    envs: Dict[str, SingleAssetHedgingEnv] = {}
    for name, batch in batches.items():
        envs[name] = SingleAssetHedgingEnv(name_to_index[name], batch, feature_engineer)
    return envs
