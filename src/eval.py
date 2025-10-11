"""Evaluation pipeline for crisis regime robustness."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .data.features import FeatureEngineer, FeatureScaler
from .data.synthetic import EpisodeBatch
from .diagnostics import external as diag_external
from .diagnostics import isi as isi_metrics
from .models.policy_mlp import PolicyMLP
from .objectives import cvar as cvar_obj
from .utils import logging as log_utils, stats
from .utils.configs import build_envs, prepare_data_module, unwrap_experiment_config

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .envs.single_asset import SingleAssetHedgingEnv


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


LOGGER = logging.getLogger(__name__)

_RISK_KEYS: Tuple[str, ...] = ("ES95", "Mean", "SharpeRisk", "Turnover")
_MSI_EPS = 1e-8


_METHOD_CANONICAL: Mapping[str, str] = {
    "erm": "ERM",
    "erm_reg": "ERM_reg",
    "irm": "IRM",
    "hirm": "HIRM",
    "hirm_head": "HIRM_Head",
    "hirm_head_highlite": "HIRM_Head_HighLite",
    "hirm_highlite": "HIRM_Head_HighLite",
    "groupdro": "GroupDRO",
    "group_dro": "GroupDRO",
    "vrex": "V_REx",
    "v-rex": "V_REx",
}


def _es_key(alpha: float) -> str:
    pct = int(round(float(alpha) * 100))
    return f"ES{pct:02d}"


def _resolve_es_keys(alphas: Sequence[float]) -> Tuple[str, ...]:
    unique_keys = []
    seen = set()
    for alpha in alphas:
        key = _es_key(alpha)
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)
    return tuple(unique_keys)


def _canonical_method_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lowered = str(name).strip().lower()
    return _METHOD_CANONICAL.get(lowered, name if isinstance(name, str) else None)


def _method_from_cfg(cfg: DictConfig) -> Optional[str]:
    model_cfg = cfg.get("model") if cfg is not None else None
    candidates: List[Optional[str]] = []
    if model_cfg is not None:
        candidates.extend([getattr(model_cfg, "name", None), getattr(model_cfg, "objective", None)])
    algo_cfg = cfg.get("algorithm") if cfg is not None else None
    if algo_cfg is not None:
        candidates.append(getattr(algo_cfg, "name", None))
    for candidate in candidates:
        resolved = _canonical_method_name(candidate)
        if resolved:
            return str(resolved)
    direct = cfg.get("method") if cfg is not None else None
    if direct:
        resolved = _canonical_method_name(direct)
        if resolved:
            return str(resolved)
    return None


def _config_tag_from_cfg(cfg: DictConfig) -> Optional[str]:
    tags = None
    experiment_cfg = cfg.get("experiment") if cfg is not None else None
    if experiment_cfg is not None and hasattr(experiment_cfg, "tags"):
        tags = getattr(experiment_cfg, "tags")
    if tags is None and cfg is not None and hasattr(cfg, "tags"):
        tags = getattr(cfg, "tags")
    if tags is None:
        return None
    if isinstance(tags, str):
        return tags
    if isinstance(tags, Sequence):
        joined = ",".join(str(item) for item in tags)
        return joined or None
    return None


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


def _slice_horizon(tensor: torch.Tensor, horizon: Optional[int]) -> torch.Tensor:
    if horizon is None:
        return tensor.clone()
    steps = tensor.shape[1] - 1
    if horizon >= steps:
        return tensor.clone()
    end = max(int(horizon) + 1, 1)
    return tensor[:, :end].clone()


def _scaled_meta(meta: Mapping[str, float], multiplier: Optional[float]) -> Dict[str, float]:
    updated = dict(meta)
    if multiplier is None:
        return updated
    scale = float(multiplier)
    for key in ("linear_bps", "quadratic", "slippage_multiplier"):
        if key in updated:
            try:
                updated[key] = float(updated[key]) * scale
            except (TypeError, ValueError):
                continue
    return updated


def _env_with_overrides(
    env: "SingleAssetHedgingEnv",
    horizon: Optional[int],
    cost_multiplier: Optional[float],
) -> "SingleAssetHedgingEnv":
    batch = env.batch
    new_batch = EpisodeBatch(
        spot=_slice_horizon(batch.spot, horizon),
        option_price=_slice_horizon(batch.option_price, horizon),
        implied_vol=_slice_horizon(batch.implied_vol, horizon),
        time_to_maturity=_slice_horizon(batch.time_to_maturity, horizon),
        rate=batch.rate,
        env_name=batch.env_name,
        meta=_scaled_meta(batch.meta, cost_multiplier),
    )
    return SingleAssetHedgingEnv(env.env_index, new_batch, env.feature_engineer)


def _evaluate_env(
    policy: PolicyMLP,
    env: SingleAssetHedgingEnv,
    device: torch.device,
    alphas: Sequence[float],
    primary_alpha: float,
    es_keys: Sequence[str],
) -> Dict:
    indices = torch.arange(env.batch.spot.shape[0])
    with torch.no_grad():
        sim = env.simulate(policy, indices, device)
    pnl = sim.pnl
    step_pnl = sim.step_pnl
    turnover = sim.turnover
    losses = -pnl
    es_values: Dict[str, float] = {}
    for alpha in alphas:
        key = _es_key(alpha)
        if key in es_values:
            continue
        es_val = cvar_obj.cvar_from_pnl(pnl, alpha)
        es_values[key] = float(es_val.item())
    mean_loss = losses.mean()
    sharpe = stats.sharpe_ratio(step_pnl).mean()
    sortino = stats.sortino_ratio(step_pnl).mean()
    turnover_mean = turnover.mean()
    cvar_ci = cvar_obj.bootstrap_cvar_ci(pnl, primary_alpha)
    primary_key = _es_key(primary_alpha)
    record = {
        "cvar": es_values.get(primary_key),
        "cvar_lower": cvar_ci.lower,
        "cvar_upper": cvar_ci.upper,
        "mean_pnl": float(pnl.mean().item()),
        "max_drawdown": float(stats.max_drawdown(step_pnl).mean().item()),
        "sharpe": float(sharpe.item()),
        "sortino": float(sortino.item()),
        "turnover": float(turnover_mean.item()),
        "risks": {
            **{key: es_values.get(key) for key in es_keys},
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
    alphas: Sequence[float],
    primary_alpha: float,
    es_keys: Sequence[str],
    scaler: FeatureScaler,
    max_position: float,
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for name in names:
        policy = _baseline_policy(name, scaler, max_position).to(device)
        for env_name, env in envs.items():
            with torch.no_grad():
                result = _evaluate_env(policy, env, device, alphas, primary_alpha, es_keys)
            summary = {
                "baseline": name,
                "env": env_name,
                "env_id": env.env_index,
                **{risk: result["risks"].get(risk) for risk in _RISK_KEYS},
                "mean_pnl": result["mean_pnl"],
                "sharpe": result["sharpe"],
                "sortino": result["sortino"],
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


def _aggregate_split_metrics(records: Sequence[Dict[str, object]], es_keys: Sequence[str]) -> Dict[str, Optional[float]]:
    acc: Dict[str, List[float]] = {key: [] for key in es_keys}
    acc["mean_pnl"] = []
    acc["turnover"] = []
    for record in records:
        risks = record.get("risks", {}) if isinstance(record, Mapping) else {}
        for key in es_keys:
            value = None
            if isinstance(risks, Mapping):
                value = risks.get(key)
            if value is not None and not math.isnan(value):
                acc[key].append(float(value))
        mean_val = record.get("mean_pnl")
        if mean_val is not None and not math.isnan(mean_val):
            acc["mean_pnl"].append(float(mean_val))
        turnover = record.get("turnover")
        if turnover is not None and not math.isnan(turnover):
            acc["turnover"].append(float(turnover))

    summary: Dict[str, Optional[float]] = {}
    for key, values in acc.items():
        if values:
            summary[key] = sum(values) / len(values)
        else:
            summary[key] = None
    return summary


def _gap(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return max(vals) - min(vals)


def _append_eval_matrix_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _feature_groups(
    feature_names: List[str],
    grouping_cfg: Optional[Dict[str, Iterable[str]]] = None,
) -> tuple[List[int], List[int]]:
    invariants: List[int] = []
    spurious: List[int] = []

    if grouping_cfg:
        explicit_invariants = {name.lower() for name in grouping_cfg.get("invariants", [])}
        explicit_spurious = {name.lower() for name in grouping_cfg.get("spurious", [])}
        invariant_patterns = [pat.lower() for pat in grouping_cfg.get("invariant_patterns", [])]
        spurious_patterns = [pat.lower() for pat in grouping_cfg.get("spurious_patterns", [])]
        default_group = str(grouping_cfg.get("default", "invariant")).lower()

        def _matches(key: str, patterns: Iterable[str]) -> bool:
            return any(pat in key for pat in patterns)

        for idx, name in enumerate(feature_names):
            key = name.lower()
            if key in explicit_invariants or _matches(key, invariant_patterns):
                invariants.append(idx)
            elif key in explicit_spurious or _matches(key, spurious_patterns):
                spurious.append(idx)
            elif default_group == "spurious":
                spurious.append(idx)
            else:
                invariants.append(idx)
        return invariants, spurious

    for idx, name in enumerate(feature_names):
        key = name.lower()
        if key in {"delta", "gamma", "vega", "theta", "tau", "time_to_maturity", "inventory"}:
            invariants.append(idx)
        elif key in {"realized_vol", "realised_vol"} or "regime" in key or "vol" in key:
            spurious.append(idx)
        else:
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
    envs: Dict[str, "SingleAssetHedgingEnv"],
    feature_names: List[str],
    device: torch.device,
    batch_size: int,
    grouping_cfg: Optional[Dict[str, Iterable[str]]] = None,
) -> Optional[Dict[str, float]]:
    if not envs:
        return None
    invariants, spurious = _feature_groups(feature_names, grouping_cfg)
    if not invariants:
        return None
    per_env_features: Dict[str, torch.Tensor] = {}
    for name, env in envs.items():
        per_env_features[name] = _collect_policy_inputs(policy, env, device)

    policy.eval()

    def _compute_autograd() -> Optional[Dict[str, float]]:
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
        msi_val = s_phi / (s_r + _MSI_EPS)
        return {"value": float(msi_val), "S_phi": float(s_phi), "S_r": float(s_r), "method": "autograd"}

    def _compute_finite_difference() -> Optional[Dict[str, float]]:
        eps = 1e-3
        total_phi = 0.0
        total_r = 0.0
        total_count = 0
        with torch.no_grad():
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
                if sampled.numel() == 0:
                    continue
                def _eval(inputs: torch.Tensor) -> torch.Tensor:
                    out = policy(inputs, env.env_index)
                    raw = out.get("raw_action")
                    if raw is None:
                        raw = out["action"]
                    return raw.squeeze(-1)

                grads = torch.zeros_like(sampled)
                for idx in list(invariants) + list(spurious):
                    basis = torch.zeros_like(sampled)
                    basis[:, idx] = eps
                    pos = _eval(sampled + basis)
                    neg = _eval(sampled - basis)
                    grads[:, idx] = (pos - neg) / (2.0 * eps)
                if invariants:
                    phi_grad = grads[:, invariants]
                    phi_norm = torch.linalg.vector_norm(phi_grad, dim=1)
                    total_phi += float(phi_norm.mean().item()) * num
                if spurious:
                    r_grad = grads[:, spurious]
                    r_norm = torch.linalg.vector_norm(r_grad, dim=1)
                    total_r += float(r_norm.mean().item()) * num
                total_count += num
        if total_count == 0:
            return None
        s_phi = total_phi / total_count
        s_r = total_r / max(total_count, 1)
        msi_val = s_phi / (s_r + _MSI_EPS)
        return {
            "value": float(msi_val),
            "S_phi": float(s_phi),
            "S_r": float(s_r),
            "method": "finite_difference",
        }

    try:
        autograd_res = _compute_autograd()
        if autograd_res is not None:
            LOGGER.debug("Computed MSI via autograd")
            return autograd_res
    except RuntimeError as err:
        LOGGER.warning("Autograd MSI failed (%s); falling back to finite differences", err)

    fd_res = _compute_finite_difference()
    if fd_res is not None:
        LOGGER.info("Computed MSI via finite-difference fallback")
    return fd_res


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
        scalar_values: List[float] = []
        for record in records:
            metrics = record.get(key)
            if isinstance(metrics, Mapping):
                for risk in _RISK_KEYS:
                    value = metrics.get(risk)
                    if value is not None and not math.isnan(value):
                        gap_acc[risk].append(float(value))
            elif metrics is not None and not math.isnan(metrics):
                scalar_values.append(float(metrics))
        summary = {risk: _mean_std(vals) for risk, vals in gap_acc.items()}
        summary["value"] = _mean_std(scalar_values)
        return summary

    ig_summary = _aggregate_gap("IG")
    wg_summary = _aggregate_gap("WG")

    vr_values: List[float] = []
    er_values: List[float] = []
    tr_values: List[float] = []
    isi_components: Dict[str, List[float]] = {"C1": [], "C2": [], "C3": [], "ISI": []}
    msi_values: List[float] = []
    s_phi_values: List[float] = []
    s_r_values: List[float] = []
    for record in records:
        vr = record.get("VR")
        if vr is not None and not math.isnan(vr):
            vr_values.append(float(vr))
        er = record.get("ER")
        if er is not None and not math.isnan(er):
            er_values.append(float(er))
        tr = record.get("TR")
        if tr is not None and not math.isnan(tr):
            tr_values.append(float(tr))
        isi_metrics_record = record.get("ISI") or {}
        for key in ("C1", "C2", "C3", "ISI"):
            value = isi_metrics_record.get(key) if isinstance(isi_metrics_record, Mapping) else None
            if value is not None and not math.isnan(value):
                isi_components[key].append(float(value))
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
        "VR": _mean_std(vr_values),
        "ER": _mean_std(er_values),
        "TR": _mean_std(tr_values),
        "ISI": {key: _mean_std(vals) for key, vals in isi_components.items()},
        "MSI": {
            "value": _mean_std(msi_values),
            "S_phi": _mean_std(s_phi_values),
            "S_r": _mean_std(s_r_values),
        },
    }


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_experiment_config(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    device = _device(cfg.get("runtime", {}))
    data_ctx = prepare_data_module(cfg)
    feature_engineer = FeatureEngineer()

    policy = _policy_from_cfg(
        cfg,
        feature_dim=len(feature_engineer.feature_names),
        num_envs=len(data_ctx.env_order),
    )
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

    compute_per_env = bool(cfg.eval.get("compute_per_env_metrics", True))
    env_batches: Dict[str, Dict[str, EpisodeBatch]] = {}
    env_batches["test"] = data_ctx.data_module.prepare("test", cfg.envs.test)
    if compute_per_env and cfg.envs.train:
        env_batches["train"] = data_ctx.data_module.prepare("train", cfg.envs.train)
    if compute_per_env and cfg.envs.get("val"):
        env_batches["val"] = data_ctx.data_module.prepare("val", cfg.envs.val)

    env_splits = {
        split: build_envs(batches, feature_engineer, data_ctx.name_to_index)
        for split, batches in env_batches.items()
    }

    primary_alpha = float(cfg.eval.report.alpha)
    primary_key = _es_key(primary_alpha)
    alpha_candidates = []
    cfg_alphas = cfg.eval.get("es_alpha_list") if cfg.eval else None
    if cfg_alphas is not None:
        alpha_candidates.extend(float(a) for a in cfg_alphas)
    alpha_candidates.append(primary_alpha)
    sanitized_alphas: List[float] = []
    seen_alphas: set[float] = set()
    for alpha_val in alpha_candidates:
        if not (0.0 < float(alpha_val) < 1.0):
            continue
        rounded = float(alpha_val)
        if rounded in seen_alphas:
            continue
        sanitized_alphas.append(rounded)
        seen_alphas.add(rounded)
    es_alpha_list: List[float] = sanitized_alphas or [primary_alpha]
    es_keys = _resolve_es_keys(es_alpha_list)
    global _RISK_KEYS
    _RISK_KEYS = tuple([*es_keys, "Mean", "SharpeRisk", "Turnover"])

    method_name = _method_from_cfg(cfg) or str(getattr(cfg.model, "name", "unknown"))
    config_tag = _config_tag_from_cfg(cfg)
    seed_value = int(cfg.get("runtime", {}).get("seed", 0))
    commit_full = log_utils._get_git_commit()
    commit_hash = commit_full[:8] if commit_full not in {"unknown", ""} else commit_full

    evaluation_records: List[Dict] = []
    env_metric_payload: Dict[str, Dict[str, object]] = {}
    train_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics: Dict[str, Dict[str, float]] = {}

    for split, split_envs in env_splits.items():
        for env_name, env in split_envs.items():
            result = _evaluate_env(policy, env, device, es_alpha_list, primary_alpha, es_keys)
            result.update({"env": env_name, "split": split, "env_id": env.env_index})
            evaluation_records.append(result)
            risk_metrics = {k: float(v) for k, v in result["risks"].items() if v is not None}
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
                f"{split}/{env_name}/sortino": result["sortino"],
                f"{split}/{env_name}/turnover": result["turnover"],
            }
            run_logger.log_metrics(metrics_to_log)

    test_primary_values: List[float] = []
    test_mean_pnls: List[float] = []
    test_sortinos: List[float] = []
    test_turnovers: List[float] = []
    test_env_vectors: List[List[float]] = []
    for record in evaluation_records:
        if record.get("split") != "test":
            continue
        risks = record.get("risks", {})
        primary_val = risks.get(primary_key) if isinstance(risks, Mapping) else None
        if primary_val is None or math.isnan(primary_val):
            continue
        primary_float = float(primary_val)
        mean_val = float(record.get("mean_pnl", 0.0))
        sortino_val = float(record.get("sortino", 0.0))
        turnover_val = float(record.get("turnover", 0.0))
        test_primary_values.append(primary_float)
        test_mean_pnls.append(mean_val)
        test_sortinos.append(sortino_val)
        test_turnovers.append(turnover_val)
        test_env_vectors.append(
            [
                primary_float,
                mean_val,
                sortino_val,
                turnover_val,
            ]
        )

    horizons_cfg = cfg.eval.get("horizons") if cfg.eval else None
    cost_cfg = cfg.eval.get("cost_multipliers") if cfg.eval else None
    save_eval_matrix = bool(cfg.eval.get("save_eval_matrix", False)) if cfg.eval else False
    horizons = [int(h) for h in horizons_cfg] if horizons_cfg else []
    cost_multipliers = [float(c) for c in cost_cfg] if cost_cfg else []
    if save_eval_matrix and horizons and cost_multipliers:
        timestamp = datetime.now(timezone.utc).isoformat()
        eval_rows: List[Dict[str, object]] = []
        for split, split_envs in env_splits.items():
            if not split_envs:
                continue
            for horizon in horizons:
                for cost_mult in cost_multipliers:
                    split_records: List[Dict[str, object]] = []
                    for env in split_envs.values():
                        sweep_env = _env_with_overrides(env, horizon, cost_mult)
                        sweep_res = _evaluate_env(
                            policy,
                            sweep_env,
                            device,
                            es_alpha_list,
                            primary_alpha,
                            es_keys,
                        )
                        split_records.append(sweep_res)
                    if not split_records:
                        continue
                    agg = _aggregate_split_metrics(split_records, es_keys)
                    row: Dict[str, object] = {
                        "method": method_name,
                        "seed": seed_value,
                        "split": split,
                        "horizon": int(horizon),
                        "cost_mult": float(cost_mult),
                        "commit": commit_hash,
                        "config_tag": config_tag,
                        "timestamp": timestamp,
                    }
                    for key in es_keys:
                        row[key.lower()] = agg.get(key)
                    row["mean_pnl"] = agg.get("mean_pnl")
                    row["turnover"] = agg.get("turnover")
                    eval_rows.append(row)
        if eval_rows:
            eval_matrix_path = Path(run_logger.base_dir) / "eval_summaries" / "eval_matrix.csv"
            fieldnames = [
                "method",
                "seed",
                "split",
                "horizon",
                "cost_mult",
                *[key.lower() for key in es_keys],
                "mean_pnl",
                "turnover",
                "commit",
                "config_tag",
                "timestamp",
            ]
            _append_eval_matrix_rows(eval_matrix_path, fieldnames, eval_rows)

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
                es_alpha_list,
                primary_alpha,
                es_keys,
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
                    for key in [*es_keys, "Mean", "SharpeRisk", "Turnover", "mean_pnl", "sharpe"]
                    if key in row
                }
                metrics[f"baseline/{row['baseline']}_{row['env']}/sortino"] = row.get("sortino")
                run_logger.log_metrics(metrics)

    plot_dir = Path(run_logger.artifacts_dir) / cfg.eval.report.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    for record in evaluation_records:
        qq_path = plot_dir / f"qq_{record['env']}.png"
        _plot_qq(record, qq_path)

    compute_msi_flag = bool(cfg.eval.get("compute_msi", False))
    msi_batch_size = int(cfg.eval.get("msi_batch_size", 512))
    msi_groups_cfg = cfg.eval.get("msi_feature_groups")
    if msi_groups_cfg is not None:
        msi_groups_cfg = OmegaConf.to_container(msi_groups_cfg, resolve=True)
    msi_envs = env_splits.get("train") or env_splits.get("test")
    msi = (
        _compute_msi(
            policy,
            msi_envs,
            feature_engineer.feature_names,
            device,
            msi_batch_size,
            grouping_cfg=msi_groups_cfg,
        )
        if compute_msi_flag
        else None
    )
    if msi is None:
        msi = {"value": None, "S_phi": None, "S_r": None, "method": None}

    diagnostics_cfg = cfg.eval.get("diagnostics") if cfg.eval else None
    if diagnostics_cfg is not None:
        diagnostics_cfg = OmegaConf.to_container(diagnostics_cfg, resolve=True)
    diag_cfg: Mapping[str, object] = diagnostics_cfg or {}
    trim_fraction = float(diag_cfg.get("trim_fraction", 0.0))
    clamp_cfg = diag_cfg.get("clamp") if isinstance(diag_cfg, Mapping) else None
    clamp_tuple: Optional[Tuple[Optional[float], Optional[float]]] = None
    if isinstance(clamp_cfg, Mapping):
        clamp_min = clamp_cfg.get("min")
        clamp_max = clamp_cfg.get("max")
        clamp_tuple = (
            float(clamp_min) if clamp_min is not None else None,
            float(clamp_max) if clamp_max is not None else None,
        )
    elif isinstance(clamp_cfg, Sequence) and len(clamp_cfg) == 2:
        clamp_tuple = (
            float(clamp_cfg[0]) if clamp_cfg[0] is not None else None,
            float(clamp_cfg[1]) if clamp_cfg[1] is not None else None,
        )
    alignment_mode = str(diag_cfg.get("alignment", "cosine"))
    covariance_mode = str(diag_cfg.get("covariance_dispersion", "trace"))
    tail_quantile = float(diag_cfg.get("tail_quantile", 0.95))
    compute_isi_flag = bool(diag_cfg.get("compute_isi", False))

    train_primary_values = [
        float(metrics.get(primary_key))
        for metrics in train_metrics.values()
        if metrics.get(primary_key) is not None and not math.isnan(metrics.get(primary_key))
    ]

    ig_source = train_primary_values if train_primary_values else test_primary_values
    ig_value = (
        diag_external.compute_ig(ig_source, trim_fraction=trim_fraction, clamp=clamp_tuple)
        if ig_source
        else None
    )
    wg_value = (
        diag_external.compute_wg(
            train_primary_values,
            test_primary_values,
            trim_fraction=trim_fraction,
            clamp=clamp_tuple,
        )
        if train_primary_values and test_primary_values
        else None
    )
    vr_value = (
        diag_external.compute_variation_ratio(
            test_env_vectors,
            trim_fraction=trim_fraction,
            clamp=clamp_tuple,
            alignment=alignment_mode,
            covariance_dispersion=covariance_mode,
        )
        if test_env_vectors
        else None
    )
    er_value = (
        diag_external.compute_expected_risk(
            test_primary_values,
            trim_fraction=trim_fraction,
            clamp=clamp_tuple,
        )
        if test_primary_values
        else None
    )
    tr_value = (
        diag_external.compute_tail_risk(
            test_primary_values,
            trim_fraction=trim_fraction,
            clamp=clamp_tuple,
            covariance_dispersion=covariance_mode,
            quantile=tail_quantile,
        )
        if test_primary_values
        else None
    )
    isi_values = (
        isi_metrics.compute_ISI(
            test_env_vectors,
            trim_fraction=trim_fraction,
            clamp=clamp_tuple,
            alignment=alignment_mode,
            covariance_dispersion=covariance_mode,
        )
        if compute_isi_flag and test_env_vectors
        else {"C1": None, "C2": None, "C3": None, "ISI": None}
    )

    diagnostics_record = {
        "seed": seed_value,
        "env_metrics": env_metric_payload,
        "IG": ig_value,
        "WG": wg_value,
        "VR": vr_value,
        "ER": er_value,
        "TR": tr_value,
        "ISI": isi_values,
        "MSI": msi,
        "method": method_name,
        "config_tag": config_tag,
    }

    def _safe_mean(values: Sequence[float]) -> Optional[float]:
        valid = [float(v) for v in values if v is not None and not math.isnan(v)]
        if not valid:
            return None
        return sum(valid) / len(valid)

    avg_cvar = _safe_mean(test_primary_values)
    avg_mean_pnl = _safe_mean(test_mean_pnls)
    avg_sortino = _safe_mean(test_sortinos)
    avg_turnover = _safe_mean(test_turnovers)
    env_label = str(cfg.eval.get("name", "eval")) if cfg.eval else "eval"
    report_cfg = cfg.eval.report if cfg.eval and hasattr(cfg.eval, "report") else None
    window_label = "test"
    if report_cfg is not None and hasattr(report_cfg, "get"):
        window_label = str(report_cfg.get("window", "test"))
    per_seed_row = {
        "seed": seed_value,
        "method": method_name,
        "env": env_label,
        "window": window_label,
        "CVaR95": avg_cvar,
        "mean_pnl": avg_mean_pnl,
        "sortino": avg_sortino,
        "turnover": avg_turnover,
        "IG": ig_value,
        "WG": wg_value,
        "VR": vr_value,
        "ER": er_value,
        "TR": tr_value,
        "C1": isi_values.get("C1") if isinstance(isi_values, Mapping) else None,
        "C2": isi_values.get("C2") if isinstance(isi_values, Mapping) else None,
        "C3": isi_values.get("C3") if isinstance(isi_values, Mapping) else None,
        "ISI": isi_values.get("ISI") if isinstance(isi_values, Mapping) else None,
    }
    run_logger.log_diagnostics_row(per_seed_row)

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
    ig_value = diagnostics_record.get("IG")
    if isinstance(ig_value, (int, float)) and ig_value is not None and not math.isnan(ig_value):
        final_metrics["diagnostics/IG"] = float(ig_value)
    wg_value = diagnostics_record.get("WG")
    if isinstance(wg_value, (int, float)) and wg_value is not None and not math.isnan(wg_value):
        final_metrics["diagnostics/WG"] = float(wg_value)
    vr_value = diagnostics_record.get("VR")
    if isinstance(vr_value, (int, float)) and vr_value is not None and not math.isnan(vr_value):
        final_metrics["diagnostics/VR"] = float(vr_value)
    er_value = diagnostics_record.get("ER")
    if isinstance(er_value, (int, float)) and er_value is not None and not math.isnan(er_value):
        final_metrics["diagnostics/ER"] = float(er_value)
    tr_value = diagnostics_record.get("TR")
    if isinstance(tr_value, (int, float)) and tr_value is not None and not math.isnan(tr_value):
        final_metrics["diagnostics/TR"] = float(tr_value)
    isi_comp = diagnostics_record.get("ISI")
    if isinstance(isi_comp, Mapping):
        for comp_key, comp_value in isi_comp.items():
            if isinstance(comp_value, (int, float)) and comp_value is not None and not math.isnan(comp_value):
                final_metrics[f"diagnostics/ISI/{comp_key}"] = float(comp_value)
    for comp_key, comp_value in diagnostics_record.get("MSI", {}).items():
        if isinstance(comp_value, (int, float)) and comp_value is not None and not math.isnan(comp_value):
            final_metrics[f"diagnostics/MSI/{comp_key}"] = float(comp_value)

    run_logger.log_final(final_metrics)
    run_logger.close()


if __name__ == "__main__":
    main()
