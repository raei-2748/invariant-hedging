"""Training loop for invariant hedging experiments."""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.core.losses import (
    cvar_from_pnl,
    differentiable_cvar,
    groupdro_objective,
    irm_penalty,
    update_groupdro_weights,
    vrex_penalty,
)
from src.core.optimizers import (
    lambda_at_step,
    label_smoothing,
    setup_optimizer,
    setup_scheduler,
)
from src.core.utils import checkpoints, logging as log_utils, seed as seed_utils, stats
from src.modules.data_pipeline import (
    FeatureEngineer,
    build_envs,
    prepare_data_module,
    unwrap_experiment_config,
)
from src.modules.diagnostics import invariant_gap, safe_eval_metric, worst_group
from src.modules.environment import SingleAssetHedgingEnv
from src.modules.head_invariance import (
    IRMConfig,
    compute_env_head_grads,
    cosine_alignment_penalty,
    freeze_backbone,
    varnorm_penalty,
)
from src.modules.models import PolicyMLP




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
    """Resolve the torch device according to §4.1 training setup."""

    device_str = runtime_cfg.get("device", "auto") if runtime_cfg else "auto"
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _init_policy(cfg: DictConfig, feature_dim: int, num_envs: int):
    """Instantiate the policy network described in §4 (Eq. 1)."""

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
        head_name=cfg.model.get("head_name"),
    )


def _evaluate(
    policy,
    envs: Dict[str, "SingleAssetHedgingEnv"],
    device: torch.device,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate CVaR, mean PnL, and diagnostics across validation environments."""

    policy.eval()
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for name, env in envs.items():
            indices = torch.arange(env.batch.spot.shape[0])
            sim = env.simulate(policy, indices, device)
            pnl = sim.pnl
            step_pnl = sim.step_pnl
            turnover = sim.turnover
            cvar = cvar_from_pnl(pnl, alpha)
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
    """Schedule the IRM penalty weight following §4.2."""

    pretrain = cfg.train.pretrain_steps
    ramp = cfg.irm.get("ramp_steps", cfg.train.irm_ramp_steps)
    init_weight = cfg.irm.get("lambda_init", 0.0)
    target_weight = cfg.irm.get("lambda_target", cfg.model.irm.penalty_weight)
    if step < pretrain:
        return init_weight
    remaining = step - pretrain
    warmup = max(ramp, 1)
    schedule_name = str(cfg.irm.get("schedule", "linear")) if cfg.irm else "linear"
    delta = target_weight - init_weight
    warmed = lambda_at_step(
        remaining,
        target=delta,
        schedule=schedule_name,
        warmup_steps=warmup,
    )
    return init_weight + warmed


def _canonical_objective(cfg: DictConfig) -> str:
    """Map config aliases onto the canonical objective names used in the paper."""

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
    """Guard rails for deprecated legacy toggles retained in the archive."""

    legacy_cfg = cfg.get("legacy")
    if legacy_cfg is not None and bool(getattr(legacy_cfg, "hybrid_enabled", False)):
        raise RuntimeError("Legacy hybrid support has been removed; disable legacy.hybrid_enabled.")


def _maybe_patch_method_env() -> None:
    """Redirect legacy METHOD env vars to the unified HIRM objective."""

    method_env = os.environ.get("METHOD")
    if method_env and method_env.lower() == "hirm_head":
        warnings.warn(
            "METHOD=hirm_head is deprecated; redirecting to METHOD=hirm.",
            DeprecationWarning,
            stacklevel=2,
        )
        os.environ["METHOD"] = "hirm"


def _freeze_policy_components(policy: PolicyMLP, cfg: DictConfig) -> None:
    """Freeze optional policy submodules when reproducing ablations."""

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


def run(cfg: DictConfig) -> Path:
    """Train HIRM using the experiment recipe specified in §5 of the paper."""

    cfg = unwrap_experiment_config(cfg)
    generator = seed_utils.seed_everything(cfg.train.seed)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    _maybe_patch_method_env()
    _enforce_legacy_flags(cfg)
    device = _device(cfg.get("runtime", {}))
    data_ctx = prepare_data_module(cfg, seed=cfg.train.seed)
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

    irm_settings = IRMConfig.from_config(cfg.get("irm"))

    if objective == "hirm" and irm_settings.enabled and irm_settings.freeze_backbone:
        freeze_backbone(policy)

    optimizer = setup_optimizer(policy, cfg)
    scheduler = setup_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("mixed_precision", True)))

    run_logger = log_utils.RunLogger(OmegaConf.to_container(cfg.logging, resolve=True), resolved_cfg)
    final_metrics_path = Path(run_logger.final_metrics_path)
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
    try:
        for step in range(1, cfg.train.steps + 1):
            optimizer.zero_grad()
            env_losses: List[torch.Tensor] = []
            metrics_to_log = {}
            dummy = torch.tensor(1.0, requires_grad=(objective == "irm"), device=device)
            irm_batches: List[dict] = []

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
                    pnl = label_smoothing(pnl, smoothing)
                loss = differentiable_cvar(-pnl, alpha)
                env_losses.append(loss)
                metrics_to_log[f"train/{env_name}_mean_pnl"] = float(pnl.mean().item())
                metrics_to_log[f"train/{env_name}_cvar"] = float(cvar_from_pnl(pnl, alpha).item())
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

                if objective == "hirm" and irm_settings.enabled:
                    irm_batches.append({"loss": loss, "name": env_name})

        loss_tensor = torch.stack(env_losses)
        if objective == "groupdro":
            total_loss = groupdro_objective(group_weights, env_losses)
        else:
            total_loss = loss_tensor.mean()

        penalty_value = 0.0
        lambda_logged = 0.0
        if objective == "irm":
            lambda_weight = _lambda_schedule(step, cfg)
            irm_pen = irm_penalty(env_losses, dummy)
            total_loss = total_loss + lambda_weight * irm_pen
            penalty_value = float(irm_pen.item())
            lambda_logged = lambda_weight
        elif objective == "vrex":
            vrex_pen = vrex_penalty(env_losses)
            weight = cfg.model.vrex.penalty_weight
            total_loss = total_loss + weight * vrex_pen
            penalty_value = float(vrex_pen.item())
        elif objective == "hirm":
            lambda_logged = float(irm_settings.lambda_weight if irm_settings.enabled else 0.0)
            irm_pen = loss_tensor.new_zeros(())
            grad_list: List[torch.Tensor] = []
            compute_penalty = (
                irm_settings.enabled
                and len(irm_batches) >= irm_settings.env_min
                and (
                    irm_settings.lambda_weight > 0.0
                    or irm_settings.logging.log_irm_grads
                )
            )
            if compute_penalty:
                grad_list = compute_env_head_grads(
                    policy,
                    lambda _model, payload: payload["loss"],
                    irm_batches,
                    create_graph=True,
                )
                if irm_settings.type == "cosine":
                    irm_pen = cosine_alignment_penalty(grad_list, eps=irm_settings.eps)
                else:
                    irm_pen = varnorm_penalty(grad_list, eps=irm_settings.eps)
                total_loss = total_loss + irm_settings.lambda_weight * irm_pen
                penalty_value = float(irm_pen.detach().item())
                metrics_to_log["train/irm_penalty"] = penalty_value
                metrics_to_log["train/irm_penalty_weighted"] = float(
                    (irm_settings.lambda_weight * irm_pen).detach().item()
                )
            if not compute_penalty:
                metrics_to_log.setdefault("train/irm_penalty", 0.0)
                metrics_to_log.setdefault("train/irm_penalty_weighted", 0.0)
            if irm_settings.logging.log_irm_grads and grad_list:
                detached_grads = [g.detach() for g in grad_list]
                with torch.no_grad():
                    norms = [grad.norm() for grad in detached_grads]
                    if norms:
                        stacked_norms = torch.stack(norms)
                        metrics_to_log["train/irm_grad_norm_min"] = float(stacked_norms.min().item())
                        metrics_to_log["train/irm_grad_norm_mean"] = float(stacked_norms.mean().item())
                        metrics_to_log["train/irm_grad_norm_max"] = float(stacked_norms.max().item())
                    if len(detached_grads) >= 2:
                        normalised = []
                        for grad in detached_grads:
                            norm = grad.norm()
                            if norm <= irm_settings.eps:
                                normalised.append(torch.zeros_like(grad))
                            else:
                                normalised.append(grad / norm.clamp_min(irm_settings.eps))
                        stacked = torch.stack(normalised)
                        cos_matrix = stacked @ stacked.T
                        idx = torch.triu_indices(cos_matrix.shape[0], cos_matrix.shape[1], offset=1)
                        pairwise_cos = cos_matrix[idx[0], idx[1]]
                        metrics_to_log["train/irm_grad_cosine_mean"] = float(pairwise_cos.mean().item())
                        metrics_to_log["train/irm_grad_cosine_min"] = float(pairwise_cos.min().item())
                        metrics_to_log["train/irm_grad_cosine_max"] = float(pairwise_cos.max().item())

        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if objective == "groupdro":
            group_weights = update_groupdro_weights(
                group_weights, env_losses, cfg.model.groupdro.step_size
            )

        if objective in {"irm", "hirm"}:
            metrics_to_log["train/lambda"] = lambda_logged
        if objective == "hirm" and env_losses:
            # Implements Eq. (4): ISI diagnostic = |IG_in - IG_out| linking to §5.1.
            metrics_to_log["train/ig"] = safe_eval_metric(invariant_gap, env_losses)
            metrics_to_log["train/wg"] = safe_eval_metric(worst_group, env_losses)

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
        return final_metrics_path
    finally:
        run_logger.close()


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
