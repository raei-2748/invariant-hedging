"""Evaluation pipeline for crisis regime robustness."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer, FeatureScaler
from .data.synthetic import SyntheticDataModule
from .envs.single_asset import SingleAssetHedgingEnv
from .models.policy_mlp import PolicyMLP
from .objectives import cvar as cvar_obj
from .utils import logging as log_utils, stats
from .utils.configs import resolve_env_configs


def _device(runtime_cfg: DictConfig) -> torch.device:
    device_str = runtime_cfg.get("device", "auto") if runtime_cfg else "auto"
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _policy_from_cfg(cfg: DictConfig, feature_dim: int, num_envs: int) -> PolicyMLP:
    policy = PolicyMLP(
        feature_dim=feature_dim,
        num_envs=num_envs,
        hidden_width=cfg.model.hidden_width,
        hidden_depth=cfg.model.hidden_depth,
        dropout=cfg.model.dropout,
        layer_norm=cfg.model.layer_norm,
        representation_dim=cfg.model.representation_dim,
        adapter_hidden=cfg.model.adapter_hidden,
        max_position=cfg.model.max_position,
    )
    return policy


def _evaluate_env(policy: PolicyMLP, env: SingleAssetHedgingEnv, device: torch.device, alpha: float) -> Dict:
    indices = torch.arange(env.batch.spot.shape[0])
    sim = env.simulate(policy, indices, device)
    pnl = sim.pnl
    step_pnl = sim.step_pnl
    turnover = sim.turnover
    cvar = cvar_obj.cvar_from_pnl(pnl, alpha)
    cvar_ci = cvar_obj.bootstrap_cvar_ci(pnl, alpha)
    record = {
        "cvar": float(cvar.item()),
        "cvar_lower": cvar_ci.lower,
        "cvar_upper": cvar_ci.upper,
        "mean_pnl": float(pnl.mean().item()),
        "turnover": float(turnover.mean().item()),
        "max_drawdown": float(stats.max_drawdown(step_pnl).mean().item()),
        "sharpe": float(stats.sharpe_ratio(step_pnl).mean().item()),
        "pnl": pnl.detach().cpu(),
        "step_pnl": step_pnl.detach().cpu(),
    }
    return record


def _plot_qq(record: Dict, output_path: Path) -> None:
    pnl = record["pnl"].numpy()
    ref = torch.from_numpy(pnl).mean().item()
    quantiles = torch.linspace(0, 1, len(pnl))
    sorted_pnl = torch.sort(torch.from_numpy(pnl))[0]
    normal = torch.distributions.Normal(sorted_pnl.mean(), sorted_pnl.std(unbiased=False))
    ref_quant = normal.icdf(quantiles.clamp(1e-3, 1 - 1e-3))
    plt.figure(figsize=(6, 6))
    plt.plot(ref_quant.numpy(), sorted_pnl.numpy(), marker="o", linestyle="")
    plt.plot(ref_quant.numpy(), ref_quant.numpy(), color="black", linestyle="--", label="45°")
    plt.xlabel("Normal quantiles")
    plt.ylabel("P&L quantiles")
    plt.title("QQ plot vs Normal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    if "train" in cfg:
        cfg = cfg.train
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = _device(cfg.get("runtime", {}))
    env_cfgs, cost_cfgs, env_order = resolve_env_configs(cfg.envs)
    data_module = SyntheticDataModule(
        config=OmegaConf.to_container(cfg.data, resolve=True),
        env_cfgs=env_cfgs,
        cost_cfgs=cost_cfgs,
    )
    test_batches = data_module.prepare("test", cfg.envs.test)
    feature_engineer = FeatureEngineer()
    feature_engineer.scaler = FeatureScaler(
        mean=torch.ones(len(feature_engineer.feature_names)),
        std=torch.ones(len(feature_engineer.feature_names)),
    )

    name_to_index = {name: idx for idx, name in enumerate(env_order)}
    envs = {
        name: SingleAssetHedgingEnv(name_to_index[name], batch, feature_engineer)
        for name, batch in test_batches.items()
    }

    policy = _policy_from_cfg(cfg, feature_dim=len(feature_engineer.feature_names), num_envs=len(env_order))
    policy.to(device)

    checkpoint_path = cfg.eval.report.get("checkpoint_path")
    if checkpoint_path in (None, ""):
        raise ValueError("evaluation.report.checkpoint_path must be provided")
    state = torch.load(Path(checkpoint_path), map_location=device)
    policy.load_state_dict(state["model"])
    scaler_state = state.get("scaler")
    if scaler_state is None:
        raise ValueError("Checkpoint missing feature scaler state")
    feature_engineer.scaler = FeatureScaler(
        mean=torch.as_tensor(scaler_state["mean"]),
        std=torch.as_tensor(scaler_state["std"]),
    )

    policy.eval()
    run_logger = log_utils.RunLogger(OmegaConf.to_container(cfg.logging, resolve=True), resolved_cfg)
    records: List[Dict] = []
    alpha = cfg.eval.report.alpha
    for name, env in envs.items():
        result = _evaluate_env(policy, env, device, alpha)
        result["env"] = name
        records.append(result)
        run_logger.log_metrics({f"test/{name}_cvar": result["cvar"], f"test/{name}_mean": result["mean_pnl"]})

    summary = [
        {k: v for k, v in record.items() if k not in {"pnl", "step_pnl"}}
        for record in records
    ]
    df = pd.DataFrame(summary)
    table_path = Path(run_logger.artifacts_dir) / cfg.eval.report.table_path
    df.to_csv(table_path, index=False)

    plot_dir = Path(run_logger.artifacts_dir) / cfg.eval.report.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        qq_path = plot_dir / f"qq_{record['env']}.png"
        _plot_qq(record, qq_path)

    run_logger.log_final({f"test/{row['env']}_cvar": row["cvar"] for _, row in df.iterrows()})
    run_logger.close()


if __name__ == "__main__":
    main()
