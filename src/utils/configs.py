"""Hydra configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..data.features import FeatureEngineer
from ..data.synthetic import EpisodeBatch, SyntheticDataModule
from ..envs.single_asset import SingleAssetHedgingEnv


@dataclass(frozen=True)
class DataModuleContext:
    data_module: SyntheticDataModule
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


_METHOD_PRESETS = {
    "erm_reg": "erm_reg",
    "erm": "erm",
    "irm_head": "hirm_head",
    "groupdro": "groupdro",
    "vrex": "vrex",
}

_PHASE_PRESETS = {
    "smoke": "smoke",
    "phase2": "phase2",
    "2": "phase2",
}


def _apply_train_preset(cfg: DictConfig, preset: str) -> None:
    preset_cfg = OmegaConf.load(to_absolute_path(f"configs/train/{preset}.yaml"))
    for key, value in preset_cfg.items():
        if key == "defaults":
            continue
        cfg[key] = value


def _load_model_config(name: str) -> DictConfig:
    return OmegaConf.load(to_absolute_path(f"configs/model/{name}.yaml"))


def _latest_checkpoint(run_root: Path) -> Optional[Path]:
    if not run_root.exists():
        return None
    candidates = [p for p in run_root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        checkpoint_dir = run_dir / "checkpoints"
        manifest = checkpoint_dir / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                data = []
            if isinstance(data, list) and data:
                rel = data[0].get("path") if isinstance(data[0], dict) else None
                if isinstance(rel, str):
                    candidate = checkpoint_dir / rel
                    if candidate.exists():
                        return candidate
        ckpt_files = sorted(checkpoint_dir.glob("*.pt"))
        if ckpt_files:
            return ckpt_files[-1]
    return None


def _load_run_config(checkpoint: Path) -> Optional[DictConfig]:
    run_dir = checkpoint.parent.parent
    config_file = (run_dir / "config.yaml").resolve()
    if not config_file.exists():
        return None
    try:
        return OmegaConf.load(str(config_file))
    except Exception:
        return None


def unwrap_experiment_config(cfg: DictConfig) -> DictConfig:
    if "train" in cfg and "data" not in cfg:
        return cfg.train

    method = cfg.get("method")
    phase_cfg = cfg.get("phase")
    seed_override = cfg.get("seed")

    method_name = str(method) if method else None
    phase_name: str | None
    if isinstance(phase_cfg, DictConfig):
        if phase_cfg.get("name") is not None:
            phase_name = phase_cfg.get("name")
        elif phase_cfg.get("phase") is not None:
            inner = phase_cfg.get("phase")
            if isinstance(inner, DictConfig):
                phase_name = inner.get("name")
            else:
                phase_name = str(inner)
        else:
            phase_name = None
    else:
        phase_name = str(phase_cfg) if phase_cfg else None

    if seed_override is not None:
        cfg.runtime.seed = int(seed_override)

    if method_name and method_name in _METHOD_PRESETS:
        preset = _METHOD_PRESETS[method_name]
        _apply_train_preset(cfg, preset)

    if phase_name and phase_name in _PHASE_PRESETS:
        _apply_train_preset(cfg, _PHASE_PRESETS[phase_name])

    if method_name and method_name in _METHOD_PRESETS:
        cfg.model = _load_model_config(_METHOD_PRESETS[method_name])

    if (
        phase_name == "smoke"
        and cfg.get("eval") is not None
        and cfg.eval.get("report") is not None
        and not cfg.eval.report.get("checkpoint_path")
    ):
        run_root = Path(cfg.runtime.get("output_dir", "runs")).expanduser()
        checkpoint = _latest_checkpoint(run_root)
        if checkpoint is not None:
            cfg.eval.report.checkpoint_path = str(checkpoint)
            if not method_name:
                run_cfg = _load_run_config(checkpoint)
                if run_cfg is not None and run_cfg.get("model") is not None:
                    cfg.model = run_cfg.model

    return cfg


def prepare_data_module(cfg: DictConfig) -> DataModuleContext:
    env_cfgs, cost_cfgs, env_order = resolve_env_configs(cfg.envs)
    data_module = SyntheticDataModule(
        config=OmegaConf.to_container(cfg.data, resolve=True),
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


def build_envs(
    batches: Dict[str, EpisodeBatch],
    feature_engineer: FeatureEngineer,
    name_to_index: Dict[str, int],
) -> Dict[str, SingleAssetHedgingEnv]:
    envs: Dict[str, SingleAssetHedgingEnv] = {}
    for name, batch in batches.items():
        envs[name] = SingleAssetHedgingEnv(name_to_index[name], batch, feature_engineer)
    return envs
