"""Training loop for invariant hedging experiments."""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer
from .diagnostics import metrics as diag_metrics
from .models import PolicyMLP
from .objectives import cvar as cvar_obj
from .objectives import penalties
from .objectives.hirm import hirm_loss
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
        target = cfg.irm.get("lambda_I", cfg.irm.get("lambda_target", 0.0))
        init = cfg.irm.get("lambda_init", 0.0)
        return init + progress * (target - init)
    return cfg.irm.get("lambda_I", cfg.irm.get("lambda_target", 0.0))


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
    params = [p for p in policy.parameters() if p.requires_grad]
    if extra_params is not None:
        params.extend(param for param in extra_params if param.requires_grad)
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


def _canonical_objective(cfg: DictConfig) -> str:
    raw_obj = cfg.model.get("objective", cfg.model.get("name", "erm"))
    objective = str(raw_obj).lower()
    if objective == "hirm_hybrid":
        raise RuntimeError("The hybrid HIRM objective has been removed; use 'hirm'.")
    if objective == "hirm_head":
        warnings.warn(
            "Objective 'hirm_head' is deprecated; please switch to 'hirm'.",
            DeprecationWarning,
            stacklevel=3,
        )
        objective = "hirm"
    return objective


def _enforce_legacy_flags(cfg: DictConfig) -> None:
    legacy_cfg = cfg.get("legacy")
    if legacy_cfg is not None and bool(getattr(legacy_cfg, "hybrid_enabled", False)):
        raise RuntimeError("Legacy hybrid support has been removed; disable legacy.hybrid_enabled.")


def _maybe_patch_method_env() -> None:
    method_env = os.environ.get("METHOD")
    if method_env and method_env.lower() == "hirm_head":
        warnings.warn(
            "METHOD=hirm_head is deprecated; redirecting to METHOD=hirm.",
            DeprecationWarning,
            stacklevel=2,
        )
        os.environ["METHOD"] = "hirm"


def _freeze_policy_components(policy: PolicyMLP, cfg: DictConfig) -> None:
    irm_cfg = cfg.get("irm")
    freeze_cfg = None if irm_cfg is None else irm_cfg.get("freeze")
    if not freeze_cfg:
        return

    def _flag(name: str) -> bool:
        if freeze_cfg is None:
            return False
        if isinstance(freeze_cfg, dict):
            return bool(freeze_cfg.get(name, False))
        return bool(getattr(freeze_cfg, name, False))

    def _freeze(module: torch.nn.Module, enabled: bool) -> None:
        if not enabled or module is None:
            return
        for param in module.parameters():
            param.requires_grad_(False)

    _freeze(policy.backbone, _flag("backbone"))
    _freeze(policy.representation, _flag("representation"))
    if _flag("adapters"):
        for adapter in policy.env_adapters:
            _freeze(adapter, True)


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_experiment_config(cfg)
    if cfg.get("legacy") and cfg.legacy.get("hybrid_enabled"):
        raise RuntimeError(
            "Legacy HIRM hybrid support has been removed. Set irm.mode='head' and rerun."
        )

    method_env = os.environ.get("METHOD")
    if method_env and method_env.lower() == "hirm_head":
        warnings.warn(
            "METHOD=hirm_head is deprecated; falling back to METHOD=hirm.",
            DeprecationWarning,
            stacklevel=2,
        )

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    _maybe_patch_method_env()
    _enforce_legacy_flags(cfg)
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

    objective = _canonical_objective(cfg)
    policy = _init_policy(
        cfg,
        feature_dim=len(feature_engineer.feature_names),
        num_envs=len(data_ctx.env_order),
    )
    policy.to(device)
    _freeze_policy_components(policy, cfg)

    optimizer = _setup_optimizer(policy, cfg)
    scheduler = _setup_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("mixed_precision", True)))

    freeze_cfg = cfg.irm.get("freeze", {}) if cfg.get("irm") else {}
    if freeze_cfg.get("policy"):
        for param in policy.parameters():
            param.requires_grad_(False)
    if risk_model is not None and freeze_cfg.get("risk"):
        for param in risk_model.parameters():
            param.requires_grad_(False)
    if risk_model is not None and freeze_cfg.get("alpha") and hasattr(risk_model, "alpha"):
        risk_model.alpha.requires_grad_(False)

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
    is_hirm = objective == "hirm"

    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        env_losses: List[torch.Tensor] = []
        metrics_to_log = {}
        dummy = torch.tensor(1.0, requires_grad=(objective == "irm"), device=device)
        head_grads: List[List[torch.Tensor]] = []

        for env_idx, env_name in enumerate(cfg.envs.train):
            env = train_envs[env_name]
            indices = env.sample_indices(per_env_batch, generator)
            rep_scale = dummy if objective == "irm" else None
            sim = env.simulate(
                policy,
                indices,
                device,
                representation_scale=rep_scale,
                collect_representation=False,
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

            if is_hirm:
                adapter = policy.env_adapters[env.env_index]
                params = [p for p in adapter.parameters() if p.requires_grad]
                if params:
                    grads_raw = torch.autograd.grad(
                        loss,
                        params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    grads = [
                        g if g is not None else torch.zeros_like(param)
                        for g, param in zip(grads_raw, params)
                    ]
                else:
                    grads = [loss.new_zeros(1)]
                head_grads.append(grads)

        loss_tensor = torch.stack(env_losses)
        if objective == "groupdro":
            total_loss = penalties.groupdro_objective(group_weights, env_losses)
        else:
            total_loss = loss_tensor.mean()

        penalty_value = 0.0
        lambda_logged = 0.0
        hirm_diag_metrics: Optional[Dict[str, torch.Tensor]] = None
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
        elif objective == "hirm":
            irm_cfg = cfg.get("irm", {})
            lambda_weight = lambda_at_step(
                step,
                target=float(irm_cfg.get("lambda_target", 0.0)),
                schedule=irm_cfg.get("schedule", "none"),
                warmup_steps=int(irm_cfg.get("warmup_steps", 0)),
            )
            alignment_weight = float(irm_cfg.get("alignment_weight", 1.0))
            variance_weight = float(irm_cfg.get("variance_weight", 0.0))
            total_loss, hirm_diag = hirm_loss(
                env_losses,
                head_grads,
                lambda_weight=lambda_weight,
                alignment_weight=alignment_weight,
                variance_weight=variance_weight,
            )
            penalty_value = float(hirm_diag["penalty"].item())
            lambda_logged = lambda_weight
            hirm_diag_metrics = hirm_diag

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if objective == "groupdro":
            group_weights = penalties.update_groupdro_weights(
                group_weights, env_losses, cfg.model.groupdro.step_size
            )

        if objective in {"irm", "hirm"}:
            metrics_to_log["train/lambda"] = lambda_logged
        if objective == "hirm" and env_losses:
            env_risk_vals = [float(loss.detach().item()) for loss in env_losses]
            metrics_to_log["train/ig"] = diag_metrics.invariant_gap(env_risk_vals)
            metrics_to_log["train/wg"] = diag_metrics.worst_group(env_risk_vals)
            if hirm_diag_metrics is not None:
                metrics_to_log["train/hirm_alignment"] = float(hirm_diag_metrics["alignment"].item())
                metrics_to_log["train/hirm_variance"] = float(hirm_diag_metrics["variance"].item())
                metrics_to_log["train/hirm_cosine"] = float(hirm_diag_metrics["cosine"].item())
                metrics_to_log["train/hirm_variance_raw"] = float(hirm_diag_metrics["variance_raw"].item())
                metrics_to_log["train/hirm_base"] = float(hirm_diag_metrics["base"].item())

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
            ckpt_manager.save(
                step,
                score,
                ckpt_payload,
            )
    test_metrics = _evaluate(policy, test_envs, device, alpha)
    run_logger.log_final({f"test/{k}_{m}": v for k, metrics in test_metrics.items() for m, v in metrics.items()})
    run_logger.close()


if __name__ == "__main__":
    main()
