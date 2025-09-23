"""Hydra configuration helpers."""
from __future__ import annotations

from typing import Dict, List, Tuple

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


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
