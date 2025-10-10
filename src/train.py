"""Training loop for invariant hedging experiments."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer
from .diagnostics import external as diag_external
from .models import (
    HIRMHead,
    HIRMHybrid,
    PolicyMLP,
    RiskHead,
    hirm_head_penalty,
)
from .objectives import cvar as cvar_obj
from .objectives import penalties
from .utils import checkpoints, logging as log_utils, seed as seed_utils, stats
from .utils.configs import build_envs, prepare_data_module, unwrap_experiment_config




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


def _init_policy(cfg: DictConfig, feature_dim: int, num_envs: int):
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


def _init_risk_head(cfg: DictConfig, representation_dim: int):
    hidden = cfg.model.get("risk_hidden", cfg.model.adapter_hidden)
    return RiskHead(representation_dim, hidden)


def _evaluate(
    policy,
    envs: Dict[str, "SingleAssetHedgingEnv"],
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


def lambda_at_step(step: int, *, target: float, schedule: str, warmup_steps: int) -> float:
    schedule = schedule.lower()
    if schedule not in {"none", "linear", "cosine"}:
        raise ValueError(f"Unsupported IRM schedule: {schedule}")
    if schedule == "none" or warmup_steps <= 0:
        return target
    progress = min(1.0, float(step) / float(max(1, warmup_steps)))
    if schedule == "linear":
        return target * progress
    # cosine warmup
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return target * cosine


def _label_smoothing(pnl: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return pnl
    mean = pnl.mean()
    return (1 - smoothing) * pnl + smoothing * mean


def _setup_optimizer(policy: torch.nn.Module, cfg: DictConfig, extra_params=None) -> torch.optim.Optimizer:
    name = cfg.optimizer.name.lower()
    params = list(policy.parameters())
    if extra_params is not None:
        for param in extra_params:
            params.append(param)
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
    cfg = unwrap_experiment_config(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = _device(cfg.get("runtime", {}))
    generator = seed_utils.seed_everything(cfg.train.seed)
    data_ctx = prepare_data_module(cfg)
    train_batches = data_ctx.data_module.prepare("train", cfg.envs.train)
    val_batches = data_ctx.data_module.prepare("val", cfg.envs.val)
    test_batches = data_ctx.data_module.prepare("test", cfg.envs.test)
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_batches.values())

    train_envs = build_envs(train_batches, feature_engineer, data_ctx.name_to_index)
    val_envs = build_envs(val_batches, feature_engineer, data_ctx.name_to_index)
    test_envs = build_envs(test_batches, feature_engineer, data_ctx.name_to_index)

    objective = cfg.model.get("objective", cfg.model.get("name", "erm"))
    policy = _init_policy(
        cfg,
        feature_dim=len(feature_engineer.feature_names),
        num_envs=len(data_ctx.env_order),
    )
    policy.to(device)

    is_hirm_head = objective == "hirm_head"
    is_hirm_hybrid = objective == "hirm_hybrid"
    risk_model = None
    if is_hirm_head:
        head_module = _init_risk_head(cfg, cfg.model.representation_dim)
        risk_model = HIRMHead(nn.Identity(), head_module)
    elif is_hirm_hybrid:
        inv_head = _init_risk_head(cfg, cfg.model.representation_dim)
        adapt_head = _init_risk_head(cfg, cfg.model.representation_dim)
        hybrid_cfg = cfg.get("hybrid", {})
        risk_model = HIRMHybrid(
            nn.Identity(),
            inv_head,
            adapt_head,
            alpha_init=hybrid_cfg.get("init_alpha", cfg.model.get("alpha_init", 0.0)),
            freeze_alpha=hybrid_cfg.get("freeze_alpha", cfg.model.get("freeze_alpha", False)),
        )
    if risk_model is not None:
        risk_model.to(device)

    extra_params = None
    if risk_model is not None:
        extra_params = list(risk_model.parameters())

    optimizer = _setup_optimizer(policy, cfg, extra_params=extra_params)
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
        dummy = torch.tensor(1.0, requires_grad=(objective == "irm"), device=device)
        risk_losses: List[torch.Tensor] = []
        irm_penalties_env: List[torch.Tensor] = []
        msi_values: List[float] = []
        gate_snapshot: Optional[float] = None

        for env_idx, env_name in enumerate(cfg.envs.train):
            env = train_envs[env_name]
            indices = env.sample_indices(per_env_batch, generator)
            collect_rep = is_hirm_head or is_hirm_hybrid
            rep_scale = dummy if objective == "irm" else None
            sim = env.simulate(
                policy,
                indices,
                device,
                representation_scale=rep_scale,
                collect_representation=collect_rep,
            )
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

            if collect_rep:
                if sim.representations is None:
                    raise RuntimeError("Representations were not collected for HIRM objective.")
                rep_summary = sim.representations.mean(dim=1)
                rep_detached = rep_summary.detach()
                rep_for_msi = rep_detached.clone().requires_grad_(True)
                target = (-pnl).detach()

                if risk_model is None:
                    diag_pred = None
                    mse_loss = torch.tensor(0.0, device=device)
                elif is_hirm_head:
                    risk_pred = risk_model(rep_detached)
                    mse_loss = F.mse_loss(risk_pred.squeeze(-1), target)
                    risk_losses.append(mse_loss)
                    scaled_pred = (risk_pred * risk_model.w)
                    inv_loss = F.mse_loss(scaled_pred.squeeze(-1), target)
                    irm_penalties_env.append(hirm_head_penalty(inv_loss, risk_model.w))
                    diag_pred = risk_model(rep_for_msi)
                else:  # hirm_hybrid
                    r_hat, r_inv, _, gate_val = risk_model(rep_detached)
                    mse_loss = F.mse_loss(r_hat.squeeze(-1), target)
                    risk_losses.append(mse_loss)
                    inv_scaled = r_inv * risk_model.w_inv
                    inv_loss = F.mse_loss(inv_scaled.squeeze(-1), target)
                    irm_penalties_env.append(hirm_head_penalty(inv_loss, risk_model.w_inv))
                    diag_outputs = risk_model(rep_for_msi)
                    diag_pred = diag_outputs[0]
                    gate_snapshot = float(gate_val.item())

                if diag_pred is not None:
                    grad = torch.autograd.grad(
                        diag_pred.mean(),
                        rep_for_msi,
                        retain_graph=False,
                        allow_unused=True,
                    )[0]
                    if grad is not None:
                        msi_values.append(float(grad.abs().mean().item()))
                metrics_to_log[f"train/{env_name}_risk_mse"] = float(mse_loss.item())

        loss_tensor = torch.stack(env_losses)
        if objective == "groupdro":
            total_loss = penalties.groupdro_objective(group_weights, env_losses)
        else:
            total_loss = loss_tensor.mean()

        penalty_value = 0.0
        lambda_logged = 0.0
        if objective == "irm":
            lambda_weight = _lambda_schedule(step, cfg)
            irm_pen = penalties.irm_penalty(env_losses, dummy)
            total_loss = total_loss + lambda_weight * irm_pen
            penalty_value = float(irm_pen.item())
            lambda_logged = lambda_weight
        elif objective == "vrex":
            vrex_pen = penalties.vrex_penalty(env_losses)
            weight = cfg.model.vrex.penalty_weight
            total_loss = total_loss + weight * vrex_pen
            penalty_value = float(vrex_pen.item())
        elif objective in {"hirm_head", "hirm_hybrid"}:
            irm_cfg = cfg.get("irm", {})
            lambda_weight = lambda_at_step(
                step,
                target=float(irm_cfg.get("lambda_target", 0.0)),
                schedule=irm_cfg.get("schedule", "none"),
                warmup_steps=int(irm_cfg.get("warmup_steps", 0)),
            )
            if irm_penalties_env:
                head_penalty = torch.stack(irm_penalties_env).mean()
                total_loss = total_loss + lambda_weight * head_penalty
                penalty_value = float(head_penalty.item())
            lambda_logged = lambda_weight

        if (is_hirm_head or is_hirm_hybrid) and risk_losses:
            risk_weight = cfg.model.get("risk_loss_weight", 1.0)
            risk_loss = torch.stack(risk_losses).mean()
            total_loss = total_loss + risk_weight * risk_loss
        else:
            risk_loss = None

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

        if objective in {"hirm_head", "hirm_hybrid", "irm"}:
            metrics_to_log["train/lambda"] = lambda_logged
        if (is_hirm_head or is_hirm_hybrid) and env_losses:
            env_risk_vals = [float(loss.detach().item()) for loss in env_losses]
            ig_val = diag_external.compute_ig(env_risk_vals) or 0.0
            metrics_to_log["train/ig"] = ig_val
            metrics_to_log["train/wg"] = max(env_risk_vals)
            if risk_loss is not None:
                metrics_to_log["train/risk_loss"] = float(risk_loss.item())
            if is_hirm_hybrid and gate_snapshot is not None:
                metrics_to_log["train/gate"] = gate_snapshot

        if step % log_interval == 0 or step == 1:
            metrics_to_log["train/loss"] = float(total_loss.item())
            metrics_to_log["train/penalty"] = penalty_value
            metrics_to_log["train/lr"] = float(optimizer.param_groups[0]["lr"])
            run_logger.log_metrics(metrics_to_log, step=step)

        if step % eval_interval == 0 or step == cfg.train.steps:
            if risk_model is not None:
                risk_model.eval()
            val_metrics = _evaluate(policy, val_envs, device, alpha)
            log_payload = {f"val/{k}_{m}": v for k, metrics in val_metrics.items() for m, v in metrics.items()}
            run_logger.log_metrics(log_payload, step=step)
            primary_env = cfg.envs.val[0]
            score = -val_metrics[primary_env]["cvar"]
            ckpt_payload = {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "score": score,
                "scaler": {
                    "mean": feature_engineer.scaler.mean.cpu(),
                    "std": feature_engineer.scaler.std.cpu(),
                },
            }
            if risk_model is not None:
                ckpt_payload["risk_model"] = risk_model.state_dict()
            ckpt_manager.save(
                step,
                score,
                ckpt_payload,
            )
            if risk_model is not None:
                risk_model.train()

    if risk_model is not None:
        risk_model.eval()
    test_metrics = _evaluate(policy, test_envs, device, alpha)
    run_logger.log_final({f"test/{k}_{m}": v for k, metrics in test_metrics.items() for m, v in metrics.items()})
    if risk_model is not None:
        risk_model.train()
    run_logger.close()


if __name__ == "__main__":
    main()
