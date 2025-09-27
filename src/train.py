"""Training loop for invariant hedging experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer
from .data.synthetic import EpisodeBatch, SyntheticDataModule
from .envs.single_asset import SingleAssetHedgingEnv
from .models.policy_mlp import PolicyMLP
from .objectives import cvar as cvar_obj
from .objectives import penalties
from .utils import checkpoints, logging as log_utils, seed as seed_utils, stats
from .utils.configs import resolve_env_configs




_TORCH_THREAD_ENV = os.environ.get("HIRM_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")

if _TORCH_THREAD_ENV:
    try:
        torch.set_num_threads(max(1, int(_TORCH_THREAD_ENV)))
    except (TypeError, ValueError):
        pass

try:
    torch.set_num_interop_threads(1)
except (TypeError, RuntimeError, AttributeError):
    pass


def _device(runtime_cfg: DictConfig) -> torch.device:
    device_str = runtime_cfg.get("device", "auto") if runtime_cfg else "auto"
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _init_policy(cfg: DictConfig, feature_dim: int, num_envs: int) -> PolicyMLP:
    return PolicyMLP(
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


def _build_envs(
    batches: Dict[str, EpisodeBatch],
    feature_engineer: FeatureEngineer,
    name_to_index: Dict[str, int],
) -> Dict[str, SingleAssetHedgingEnv]:
    envs: Dict[str, SingleAssetHedgingEnv] = {}
    for name, batch in batches.items():
        envs[name] = SingleAssetHedgingEnv(name_to_index[name], batch, feature_engineer)
    return envs


def _evaluate(
    policy: PolicyMLP,
    envs: Dict[str, SingleAssetHedgingEnv],
    device: torch.device,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    policy.eval()
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for name, env in envs.items():
            indices = torch.arange(env.batch.spot.shape[0])
            sim = env.simulate(policy, indices, device)
            pnl = sim.pnl
            step_pnl = sim.step_pnl
            turnover = sim.turnover
            cvar = cvar_obj.cvar_from_pnl(pnl, alpha)
            results[name] = {
                "cvar": float(cvar.item()),
                "mean_pnl": float(pnl.mean().item()),
                "turnover": float(turnover.mean().item()),
                "max_drawdown": float(stats.max_drawdown(step_pnl).mean().item()),
                "sharpe": float(stats.sharpe_ratio(step_pnl).mean().item()),
            }
    policy.train()
    return results


def _lambda_schedule(step: int, cfg: DictConfig) -> float:
    pretrain = cfg.train.pretrain_steps
    ramp = cfg.irm.get("ramp_steps", cfg.train.irm_ramp_steps)
    if step < pretrain:
        return cfg.irm.get("lambda_init", 0.0)
    if step < pretrain + ramp:
        progress = (step - pretrain) / max(ramp, 1)
        return cfg.irm.get("lambda_init", 0.0) + progress * (
            cfg.irm.get("lambda_target", cfg.model.irm.penalty_weight)
        )
    return cfg.irm.get("lambda_target", cfg.model.irm.penalty_weight)


def _label_smoothing(pnl: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return pnl
    mean = pnl.mean()
    return (1 - smoothing) * pnl + smoothing * mean


def _setup_optimizer(policy: PolicyMLP, cfg: DictConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.name.lower()
    params = policy.parameters()
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay
    if cfg.model.name == "erm_reg":
        weight_decay = cfg.model.regularization.get("weight_decay", weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def _setup_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    name = cfg.scheduler.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.optimizer.lr * 0.1
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    if "train" in cfg and "data" not in cfg:
        cfg = cfg.train
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = _device(cfg.get("runtime", {}))
    generator = seed_utils.seed_everything(cfg.train.seed)
    env_cfgs, cost_cfgs, env_order = resolve_env_configs(cfg.envs)
    data_module = SyntheticDataModule(
        config=OmegaConf.to_container(cfg.data, resolve=True),
        env_cfgs=env_cfgs,
        cost_cfgs=cost_cfgs,
    )
    train_batches = data_module.prepare("train", cfg.envs.train)
    val_batches = data_module.prepare("val", cfg.envs.val)
    test_batches = data_module.prepare("test", cfg.envs.test)
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_batches.values())

    name_to_index = {name: idx for idx, name in enumerate(env_order)}
    train_envs = _build_envs(train_batches, feature_engineer, name_to_index)
    val_envs = _build_envs(val_batches, feature_engineer, name_to_index)
    test_envs = _build_envs(test_batches, feature_engineer, name_to_index)

    policy = _init_policy(cfg, feature_dim=len(feature_engineer.feature_names), num_envs=len(env_order))
    policy.to(device)

    optimizer = _setup_optimizer(policy, cfg)
    scheduler = _setup_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("mixed_precision", True)))

    run_logger = log_utils.RunLogger(OmegaConf.to_container(cfg.logging, resolve=True), resolved_cfg)
    checkpoint_dir = Path(run_logger.checkpoint_dir)
    ckpt_manager = checkpoints.CheckpointManager(checkpoint_dir, top_k=cfg.train.checkpoint_topk)

    group_weights = torch.ones(len(cfg.envs.train), device=device) / max(len(cfg.envs.train), 1)

    per_env_batch = max(1, cfg.train.batch_size // max(len(cfg.envs.train), 1))
    alpha = cfg.loss.cvar_alpha
    log_interval = cfg.logging.log_interval
    eval_interval = cfg.logging.eval_interval

    warning_factor = float(cfg.train.get("max_trade_warning_factor", 0.0))
    max_position = float(cfg.model.max_position)
    max_trade_warn = warning_factor * max_position if warning_factor > 0 else 0.0
    spike_alerted_envs: set[str] = set()

    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        env_losses: List[torch.Tensor] = []
        metrics_to_log = {}
        dummy = torch.tensor(1.0, requires_grad=(cfg.model.objective == "irm"), device=device)

        for env_idx, env_name in enumerate(cfg.envs.train):
            env = train_envs[env_name]
            indices = env.sample_indices(per_env_batch, generator)
            rep_scale = dummy if cfg.model.objective == "irm" else None
            sim = env.simulate(policy, indices, device, representation_scale=rep_scale)
            if sim.probe:
                run_logger.log_probe(env_name, step, sim.probe)
            pnl = sim.pnl
            if cfg.model.name == "erm_reg":
                smoothing = cfg.model.regularization.get("label_smoothing", 0.0)
                pnl = _label_smoothing(pnl, smoothing)
            loss = cvar_obj.differentiable_cvar(-pnl, alpha)
            env_losses.append(loss)
            metrics_to_log[f"train/{env_name}_mean_pnl"] = float(pnl.mean().item())
            metrics_to_log[f"train/{env_name}_cvar"] = float(cvar_obj.cvar_from_pnl(pnl, alpha).item())
            metrics_to_log[f"train/{env_name}_turnover"] = float(sim.turnover.mean().item())
            trades = sim.positions[:, 1:] - sim.positions[:, :-1]
            final_trade = -sim.positions[:, -1:]
            all_trades = torch.cat([trades, final_trade], dim=1)
            mean_abs_trade = float(torch.abs(all_trades).mean().item())
            metrics_to_log[f"train/{env_name}_mean_abs_trade"] = mean_abs_trade
            max_trade_val = float(sim.max_trade.max().item())
            metrics_to_log[f"train/{env_name}_max_trade"] = max_trade_val
            if max_trade_warn > 0 and max_trade_val > max_trade_warn and env_name not in spike_alerted_envs:
                print(
                    f"[warn] large trade magnitude detected: {max_trade_val:.2f} "
                    f"(threshold {max_trade_warn:.2f}) in env '{env_name}' at step {step}"
                )
                spike_alerted_envs.add(env_name)

        loss_tensor = torch.stack(env_losses)
        if cfg.model.objective == "groupdro":
            total_loss = penalties.groupdro_objective(group_weights, env_losses)
        else:
            total_loss = loss_tensor.mean()

        penalty_value = 0.0
        if cfg.model.objective == "irm":
            lambda_weight = _lambda_schedule(step, cfg)
            irm_pen = penalties.irm_penalty(env_losses, dummy)
            total_loss = total_loss + lambda_weight * irm_pen
            penalty_value = float(irm_pen.item())
        elif cfg.model.objective == "vrex":
            vrex_pen = penalties.vrex_penalty(env_losses)
            weight = cfg.model.vrex.penalty_weight
            total_loss = total_loss + weight * vrex_pen
            penalty_value = float(vrex_pen.item())

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if cfg.model.objective == "groupdro":
            group_weights = penalties.update_groupdro_weights(
                group_weights, env_losses, cfg.model.groupdro.step_size
            )

        if step % log_interval == 0 or step == 1:
            metrics_to_log["train/loss"] = float(total_loss.item())
            metrics_to_log["train/penalty"] = penalty_value
            metrics_to_log["train/lr"] = float(optimizer.param_groups[0]["lr"])
            run_logger.log_metrics(metrics_to_log, step=step)

        if step % eval_interval == 0 or step == cfg.train.steps:
            val_metrics = _evaluate(policy, val_envs, device, alpha)
            log_payload = {f"val/{k}_{m}": v for k, metrics in val_metrics.items() for m, v in metrics.items()}
            run_logger.log_metrics(log_payload, step=step)
            primary_env = cfg.envs.val[0]
            score = -val_metrics[primary_env]["cvar"]
            ckpt_manager.save(
                step,
                score,
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "score": score,
                    "scaler": {
                        "mean": feature_engineer.scaler.mean.cpu(),
                        "std": feature_engineer.scaler.std.cpu(),
                    }
                },
            )

    test_metrics = _evaluate(policy, test_envs, device, alpha)
    run_logger.log_final({f"test/{k}_{m}": v for k, metrics in test_metrics.items() for m, v in metrics.items()})
    run_logger.close()


if __name__ == "__main__":
    main()
