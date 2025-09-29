"""Evaluation pipeline for crisis regime robustness."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer, FeatureScaler
from .data.synthetic import EpisodeBatch, SyntheticDataModule
from .envs.single_asset import SingleAssetHedgingEnv
from .models.policy_mlp import PolicyMLP
from .objectives import cvar as cvar_obj
from .utils import logging as log_utils, stats
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


_RISK_KEYS = ("ES95", "Mean", "SharpeRisk", "Turnover")
_MSI_EPS = 1e-8


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


def _evaluate_env(
    policy: PolicyMLP,
    env: SingleAssetHedgingEnv,
    device: torch.device,
    alpha: float,
) -> Dict:
    indices = torch.arange(env.batch.spot.shape[0])
    with torch.no_grad():
        sim = env.simulate(policy, indices, device)
    pnl = sim.pnl
    step_pnl = sim.step_pnl
    turnover = sim.turnover
    losses = -pnl
    es95 = cvar_obj.cvar_from_pnl(pnl, alpha)
    mean_loss = losses.mean()
    sharpe = stats.sharpe_ratio(step_pnl).mean()
    turnover_mean = turnover.mean()
    cvar_ci = cvar_obj.bootstrap_cvar_ci(pnl, alpha)
    record = {
        "cvar": float(es95.item()),
        "cvar_lower": cvar_ci.lower,
        "cvar_upper": cvar_ci.upper,
        "mean_pnl": float(pnl.mean().item()),
        "max_drawdown": float(stats.max_drawdown(step_pnl).mean().item()),
        "sharpe": float(sharpe.item()),
        "turnover": float(turnover_mean.item()),
        "risks": {
            "ES95": float(es95.item()),
            "Mean": float(mean_loss.item()),
            "SharpeRisk": float((-sharpe).item()),
            "Turnover": float(turnover_mean.item()),
        },
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


class _ZeroPolicy:
    def __init__(self) -> None:
        self._device: torch.device | None = None

    def to(self, device: torch.device) -> "_ZeroPolicy":
        self._device = device
        return self

    def eval(self) -> "_ZeroPolicy":  # pragma: no cover - trivial
        return self

    def __call__(self, features: torch.Tensor, env_index: int, representation_scale=None) -> Dict[str, torch.Tensor]:
        device = features.device if self._device is None else self._device
        zeros = torch.zeros(features.shape[0], 1, device=device)
        return {"action": zeros, "raw_action": zeros}


class _DeltaBaselinePolicy:
    def __init__(self, scaler: FeatureScaler, max_position: float) -> None:
        self.mean = scaler.mean.clone().detach()
        self.std = torch.clamp(scaler.std.clone().detach(), min=1e-6)
        self.max_position = max_position

    def to(self, device: torch.device) -> "_DeltaBaselinePolicy":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def eval(self) -> "_DeltaBaselinePolicy":  # pragma: no cover - trivial
        return self

    def __call__(self, features: torch.Tensor, env_index: int, representation_scale=None) -> Dict[str, torch.Tensor]:
        delta = features[:, 0] * self.std[0] + self.mean[0]
        action = delta.unsqueeze(-1).clamp(-self.max_position, self.max_position)
        return {"action": action, "raw_action": action}


def _baseline_policy(name: str, scaler: FeatureScaler, max_position: float):
    key = name.lower()
    if key == "no_hedge":
        return _ZeroPolicy()
    if key == "delta":
        return _DeltaBaselinePolicy(scaler, max_position)
    raise ValueError(f"Unsupported baseline '{name}'")


def _evaluate_baselines(
    names: List[str],
    envs: Dict[str, SingleAssetHedgingEnv],
    device: torch.device,
    alpha: float,
    scaler: FeatureScaler,
    max_position: float,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for name in names:
        policy = _baseline_policy(name, scaler, max_position).to(device)
        for env_name, env in envs.items():
            with torch.no_grad():
                result = _evaluate_env(policy, env, device, alpha)
            summary = {
                "baseline": name,
                "env": env_name,
                "env_id": env.env_index,
                **{risk: result["risks"].get(risk) for risk in _RISK_KEYS},
                "mean_pnl": result["mean_pnl"],
                "sharpe": result["sharpe"],
                "turnover": result["turnover"],
                "cvar": result["cvar"],
                "cvar_lower": result["cvar_lower"],
                "cvar_upper": result["cvar_upper"],
            }
            records.append(summary)
    return records


def _flatten_env_records(records: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in records:
        env_name = record["env"]
        split = record["split"]
        env_id = record["env_id"]
        risks = record["risks"]
        row = {
            "env": env_name,
            "split": split,
            "env_id": env_id,
            "mean_pnl": record["mean_pnl"],
            "max_drawdown": record["max_drawdown"],
            "sharpe": record["sharpe"],
            "turnover": record["turnover"],
            "cvar": record["cvar"],
            "cvar_lower": record["cvar_lower"],
            "cvar_upper": record["cvar_upper"],
        }
        for key in _RISK_KEYS:
            row[key] = risks.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _gap(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return max(vals) - min(vals)


def _compute_ig(train_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
    ig: Dict[str, Optional[float]] = {}
    for risk_key in _RISK_KEYS:
        ig[risk_key] = _gap(m[risk_key] for m in train_metrics.values())
    return ig


def _compute_wg(
    train_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Optional[float]]:
    wg: Dict[str, Optional[float]] = {}
    for risk_key in _RISK_KEYS:
        train_vals = [m[risk_key] for m in train_metrics.values() if m.get(risk_key) is not None]
        test_vals = [m[risk_key] for m in test_metrics.values() if m.get(risk_key) is not None]
        if not train_vals or not test_vals:
            wg[risk_key] = None
            continue
        wg[risk_key] = max(test_vals) - max(train_vals)
    return wg


def _build_envs(
    batches: Dict[str, EpisodeBatch],
    feature_engineer: FeatureEngineer,
    name_to_index: Dict[str, int],
) -> Dict[str, SingleAssetHedgingEnv]:
    envs: Dict[str, SingleAssetHedgingEnv] = {}
    for name, batch in batches.items():
        envs[name] = SingleAssetHedgingEnv(name_to_index[name], batch, feature_engineer)
    return envs


def _feature_groups(feature_names: List[str]) -> tuple[List[int], List[int]]:
    invariants: List[int] = []
    spurious: List[int] = []
    for idx, name in enumerate(feature_names):
        key = name.lower()
        if key in {"delta", "gamma", "vega", "theta", "tau", "time_to_maturity", "inventory"}:
            invariants.append(idx)
        elif key in {"realized_vol", "realised_vol"} or "regime" in key or "vol" in key:
            spurious.append(idx)
        else:
            # Default to invariant if we cannot categorise – conservative choice.
            invariants.append(idx)
    return invariants, spurious


def _collect_policy_inputs(
    policy: PolicyMLP,
    env: SingleAssetHedgingEnv,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(env.batch.spot.shape[0])
    with torch.no_grad():
        sim = env.simulate(policy, indices, device)
    batch = env.batch.to(device)
    base = env.feature_engineer.base_features(batch)
    inventory = (sim.positions[:, :-1] / max(env.notional, 1e-8)).to(device)
    features = torch.cat([base, inventory.unsqueeze(-1)], dim=-1)
    scaler = env.feature_engineer.scaler
    if scaler is None:
        raise RuntimeError("Feature scaler must be available to compute MSI")
    mean = scaler.mean.to(device)
    std = torch.clamp(scaler.std.to(device), min=1e-6)
    features = (features - mean) / std
    return features.reshape(-1, features.shape[-1]).detach().cpu()


def _compute_msi(
    policy: PolicyMLP,
    envs: Dict[str, SingleAssetHedgingEnv],
    feature_names: List[str],
    device: torch.device,
    batch_size: int,
) -> Optional[Dict[str, float]]:
    if not envs:
        return None
    invariants, spurious = _feature_groups(feature_names)
    if not invariants:
        return None
    per_env_features: Dict[str, torch.Tensor] = {}
    for name, env in envs.items():
        per_env_features[name] = _collect_policy_inputs(policy, env, device)

    policy.eval()
    total_phi = 0.0
    total_r = 0.0
    total_count = 0
    for name, env in envs.items():
        feats = per_env_features.get(name)
        if feats is None or feats.numel() == 0:
            continue
        num = min(batch_size // max(len(envs), 1), feats.shape[0])
        if num == 0:
            num = min(feats.shape[0], batch_size)
        if num == 0:
            continue
        perm = torch.randperm(feats.shape[0])[:num]
        sampled = feats[perm].to(device)
        sampled.requires_grad_(True)
        output = policy(sampled, env.env_index)
        raw = output.get("raw_action")
        if raw is None:
            raw = output["action"]
        grad = torch.autograd.grad(
            raw.sum(),
            sampled,
            create_graph=False,
            retain_graph=False,
        )[0]
        if grad is None:
            continue
        if invariants:
            phi_grad = grad[:, invariants]
            phi_norm = torch.linalg.vector_norm(phi_grad, dim=1)
            total_phi += float(phi_norm.mean().item()) * num
        if spurious:
            r_grad = grad[:, spurious]
            r_norm = torch.linalg.vector_norm(r_grad, dim=1)
            total_r += float(r_norm.mean().item()) * num
        total_count += num
    if total_count == 0:
        return None
    s_phi = total_phi / total_count
    s_r = total_r / max(total_count, 1)
    msi = s_phi / (s_r + _MSI_EPS)
    return {"value": float(msi), "S_phi": float(s_phi), "S_r": float(s_r)}


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "std": None}
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return {"mean": mean, "std": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "std": math.sqrt(var)}


def _aggregate_records(records: List[Dict]) -> Dict[str, Dict[str, object]]:
    if not records:
        return {}
    env_acc: Dict[str, Dict[str, List[float]]] = {}
    env_meta: Dict[str, Dict[str, object]] = {}
    for record in records:
        env_metrics = record.get("env_metrics", {})
        for env_name, metrics in env_metrics.items():
            env_meta.setdefault(env_name, {})
            if "env_id" in metrics:
                env_meta[env_name]["env_id"] = metrics["env_id"]
            if "split" in metrics:
                env_meta[env_name]["split"] = metrics["split"]
            env_entry = env_acc.setdefault(env_name, {risk: [] for risk in _RISK_KEYS})
            for risk in _RISK_KEYS:
                value = metrics.get(risk)
                if value is not None and not math.isnan(value):
                    env_entry[risk].append(float(value))

    aggregated: Dict[str, Dict[str, object]] = {}
    for env_name, risk_values in env_acc.items():
        agg_metrics = {risk: _mean_std(vals) for risk, vals in risk_values.items()}
        aggregated[env_name] = {**env_meta.get(env_name, {}), **agg_metrics}

    def _aggregate_gap(key: str) -> Dict[str, Dict[str, float]]:
        gap_acc: Dict[str, List[float]] = {risk: [] for risk in _RISK_KEYS}
        for record in records:
            metrics = record.get(key, {})
            for risk in _RISK_KEYS:
                value = metrics.get(risk)
                if value is not None and not math.isnan(value):
                    gap_acc[risk].append(float(value))
        return {risk: _mean_std(vals) for risk, vals in gap_acc.items()}

    ig_summary = _aggregate_gap("IG")
    wg_summary = _aggregate_gap("WG")

    msi_values: List[float] = []
    s_phi_values: List[float] = []
    s_r_values: List[float] = []
    for record in records:
        msi = record.get("MSI") or {}
        value = msi.get("value")
        if value is not None and not math.isnan(value):
            msi_values.append(float(value))
        phi = msi.get("S_phi")
        if phi is not None and not math.isnan(phi):
            s_phi_values.append(float(phi))
        s_r = msi.get("S_r")
        if s_r is not None and not math.isnan(s_r):
            s_r_values.append(float(s_r))

    return {
        "env_metrics": aggregated,
        "IG": ig_summary,
        "WG": wg_summary,
        "MSI": {
            "value": _mean_std(msi_values),
            "S_phi": _mean_std(s_phi_values),
            "S_r": _mean_std(s_r_values),
        },
    }


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    if "train" in cfg and "data" not in cfg:
        cfg = cfg.train
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = _device(cfg.get("runtime", {}))
    env_cfgs, cost_cfgs, env_order = resolve_env_configs(cfg.envs)
    data_module = SyntheticDataModule(
        config=OmegaConf.to_container(cfg.data, resolve=True),
        env_cfgs=env_cfgs,
        cost_cfgs=cost_cfgs,
    )
    feature_engineer = FeatureEngineer()

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
        mean=torch.as_tensor(scaler_state["mean"], dtype=torch.float32),
        std=torch.as_tensor(scaler_state["std"], dtype=torch.float32),
    )

    policy.eval()
    run_logger = log_utils.RunLogger(OmegaConf.to_container(cfg.logging, resolve=True), resolved_cfg)
    name_to_index = {name: idx for idx, name in enumerate(env_order)}

    compute_per_env = bool(cfg.eval.get("compute_per_env_metrics", True))
    env_batches: Dict[str, Dict[str, EpisodeBatch]] = {}
    env_batches["test"] = data_module.prepare("test", cfg.envs.test)
    if compute_per_env and cfg.envs.train:
        env_batches["train"] = data_module.prepare("train", cfg.envs.train)
    if compute_per_env and cfg.envs.get("val"):
        env_batches["val"] = data_module.prepare("val", cfg.envs.val)

    env_splits = {
        split: _build_envs(batches, feature_engineer, name_to_index)
        for split, batches in env_batches.items()
    }

    alpha = cfg.eval.report.alpha
    evaluation_records: List[Dict] = []
    env_metric_payload: Dict[str, Dict[str, object]] = {}
    train_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics: Dict[str, Dict[str, float]] = {}

    for split, split_envs in env_splits.items():
        for env_name, env in split_envs.items():
            result = _evaluate_env(policy, env, device, alpha)
            result.update({"env": env_name, "split": split, "env_id": env.env_index})
            evaluation_records.append(result)
            risk_metrics = {k: float(v) for k, v in result["risks"].items()}
            env_entry = {"env_id": env.env_index, "split": split, **risk_metrics}
            env_metric_payload[env_name] = env_entry
            if split == "train":
                train_metrics[env_name] = risk_metrics
            if split == "test":
                test_metrics[env_name] = risk_metrics
            metrics_to_log = {
                **{f"{split}/{env_name}/{k}": v for k, v in risk_metrics.items()},
                **{f"{split}/env_{env.env_index}/{k}": v for k, v in risk_metrics.items()},
                f"{split}/{env_name}/mean_pnl": result["mean_pnl"],
                f"{split}/{env_name}/sharpe": result["sharpe"],
                f"{split}/{env_name}/turnover": result["turnover"],
            }
            run_logger.log_metrics(metrics_to_log)

    df = _flatten_env_records(evaluation_records)
    table_path = Path(run_logger.artifacts_dir) / cfg.eval.report.table_path
    df.to_csv(table_path, index=False)

    baseline_cfg = cfg.get("baselines")
    if baseline_cfg is None and "eval" in cfg:
        baseline_cfg = cfg.eval.get("baselines")
    if baseline_cfg is not None and feature_engineer.scaler is not None:
        baseline_names = list(baseline_cfg.get("include", []))
        if baseline_names:
            baseline_records = _evaluate_baselines(
                baseline_names,
                env_splits.get("test", {}),
                device,
                alpha,
                feature_engineer.scaler,
                float(cfg.model.max_position),
            )
            baseline_df = pd.DataFrame(baseline_records)
            baseline_path = Path(run_logger.artifacts_dir) / baseline_cfg.get(
                "table_path", "baseline_metrics.csv"
            )
            baseline_df.to_csv(baseline_path, index=False)
            for row in baseline_records:
                metrics = {
                    f"baseline/{row['baseline']}_{row['env']}/{key}": row.get(key)
                    for key in ["ES95", "Mean", "SharpeRisk", "Turnover", "mean_pnl", "sharpe"]
                    if key in row
                }
                run_logger.log_metrics(metrics)

    plot_dir = Path(run_logger.artifacts_dir) / cfg.eval.report.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    for record in evaluation_records:
        qq_path = plot_dir / f"qq_{record['env']}.png"
        _plot_qq(record, qq_path)

    ig = _compute_ig(train_metrics) if train_metrics else {risk: None for risk in _RISK_KEYS}
    wg = (
        _compute_wg(train_metrics, test_metrics)
        if train_metrics and test_metrics
        else {risk: None for risk in _RISK_KEYS}
    )

    compute_msi_flag = bool(cfg.eval.get("compute_msi", False))
    msi_batch_size = int(cfg.eval.get("msi_batch_size", 512))
    msi_envs = env_splits.get("train") or env_splits.get("test")
    msi = (
        _compute_msi(policy, msi_envs, feature_engineer.feature_names, device, msi_batch_size)
        if compute_msi_flag
        else None
    )
    if msi is None:
        msi = {"value": None, "S_phi": None, "S_r": None}

    diagnostics_record = {
        "seed": int(cfg.get("runtime", {}).get("seed", 0)),
        "env_metrics": env_metric_payload,
        "IG": {risk: ig.get(risk) if ig is not None else None for risk in _RISK_KEYS},
        "WG": {risk: wg.get(risk) if wg is not None else None for risk in _RISK_KEYS},
        "MSI": msi,
    }

    diagnostics_path = Path(run_logger.artifacts_dir) / "diagnostics.jsonl"
    existing_records: List[Dict] = []
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    with diagnostics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(diagnostics_record) + "\n")
    existing_records.append(diagnostics_record)
    summary_path = diagnostics_path.with_name("diagnostics_summary.json")
    aggregated = _aggregate_records(existing_records)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregated, handle, indent=2)

    final_metrics: Dict[str, float] = {}
    for env_name, metrics in test_metrics.items():
        for key, value in metrics.items():
            if value is not None:
                final_metrics[f"test/{env_name}/{key}"] = value
    for key, value in diagnostics_record["IG"].items():
        if value is not None:
            final_metrics[f"diagnostics/IG/{key}"] = value
    for key, value in diagnostics_record["WG"].items():
        if value is not None:
            final_metrics[f"diagnostics/WG/{key}"] = value
    for comp_key, comp_value in diagnostics_record["MSI"].items():
        if comp_value is not None:
            final_metrics[f"diagnostics/MSI/{comp_key}"] = comp_value

    run_logger.log_final(final_metrics)
    run_logger.close()


if __name__ == "__main__":
    main()
