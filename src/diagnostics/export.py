"""Diagnostics export orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping

import pandas as pd
import torch

from .efficiency import compute_ER, compute_TR
from .external import compute_IG, compute_VR, compute_WG
from .isi import (
    ISINormalizationConfig,
    compute_C1_global_stability,
    compute_C2_mechanistic_stability,
    compute_C3_structural_stability,
    compute_ISI,
)
from .probe import ProbeConfig, get_diagnostic_batches
from .schema import (
    CANONICAL_COLUMNS,
    SCHEMA_VERSION,
    normalize_diagnostics_frame,
    validate_diagnostics_table,
)

ProbeBatch = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class DiagnosticsRunContext:
    output_dir: Path
    seed: int
    git_hash: str
    exp_id: str
    split_name: str
    regime_tag: str
    is_eval_split: bool
    config_hash: str
    instrument: str
    metric_basis: str
    units: Mapping[str, str] | None = None


def _stack(values: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [value.reshape(-1).float() for value in values if value.numel()]
    if not tensors:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(tensors)


def gather_and_export(
    run_ctx: DiagnosticsRunContext,
    model: object,
    probe_cfg: Mapping[str, object],
    isi_cfg: Mapping[str, object] | None,
    risk_fn: Callable[[object, ProbeBatch], torch.Tensor],
    outcome_fn: Callable[[object, ProbeBatch], torch.Tensor],
    position_fn: Callable[[object, ProbeBatch], torch.Tensor],
    *,
    head_gradient_fn: Callable[[object, ProbeBatch], torch.Tensor] | None = None,
    representation_fn: Callable[[object, ProbeBatch], torch.Tensor] | None = None,
) -> Path:
    """Run the diagnostics pipeline and export a canonical parquet + manifest."""

    output_dir = Path(run_ctx.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "batches" in probe_cfg:
        env_batches: Dict[str, List[ProbeBatch]] = {
            env: list(batches) for env, batches in probe_cfg["batches"].items()
        }
    else:
        probe_config = ProbeConfig(
            batch_size=int(probe_cfg.get("batch_size", 1)),
            n_batches=int(probe_cfg.get("n_batches", 1)),
            envs=tuple(probe_cfg.get("envs", [])),
            seed=int(probe_cfg.get("seed", 0)),
        )
        env_sources = probe_cfg.get("env_sources", {})
        env_batches = get_diagnostic_batches(probe_config, env_sources)

    risk_by_env: Dict[str, List[torch.Tensor]] = {}
    outcome_by_env: Dict[str, List[torch.Tensor]] = {}
    positions_by_env: Dict[str, List[torch.Tensor]] = {}
    grads_by_env: Dict[str, List[torch.Tensor]] = {}
    repr_by_env: Dict[str, List[torch.Tensor]] = {}
    n_obs_by_env: Dict[str, int] = {}

    with torch.no_grad():
        for env_id, batches in env_batches.items():
            risk_by_env.setdefault(env_id, [])
            outcome_by_env.setdefault(env_id, [])
            positions_by_env.setdefault(env_id, [])
            if head_gradient_fn is not None:
                grads_by_env.setdefault(env_id, [])
            if representation_fn is not None:
                repr_by_env.setdefault(env_id, [])

            for batch in batches:
                risk_tensor = risk_fn(model, batch).detach()
                outcome_tensor = outcome_fn(model, batch).detach()
                position_tensor = position_fn(model, batch).detach()

                risk_by_env[env_id].append(risk_tensor)
                outcome_by_env[env_id].append(outcome_tensor)
                positions_by_env[env_id].append(position_tensor)
                n_obs_by_env[env_id] = n_obs_by_env.get(env_id, 0) + int(risk_tensor.numel())

    if head_gradient_fn is not None or representation_fn is not None:
        previous_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            for env_id, batches in env_batches.items():
                if head_gradient_fn is not None:
                    grads_by_env.setdefault(env_id, [])
                if representation_fn is not None:
                    repr_by_env.setdefault(env_id, [])

                for batch in batches:
                    if head_gradient_fn is not None:
                        grads_by_env[env_id].append(head_gradient_fn(model, batch).detach())
                    if representation_fn is not None:
                        repr_by_env[env_id].append(representation_fn(model, batch).detach())
        finally:
            torch.set_grad_enabled(previous_grad_state)

    norm_cfg = ISINormalizationConfig(
        c1_max_dispersion=float(
            (isi_cfg or {}).get("c1_max_dispersion", 1.0)
        ),
        c3_max_distance=float((isi_cfg or {}).get("c3_max_distance", 1.0)),
    )

    weights_cfg = isi_cfg.get("weights") if isi_cfg else None
    if isinstance(weights_cfg, Mapping):
        weights = (
            float(weights_cfg.get("C1", 1.0 / 3.0)),
            float(weights_cfg.get("C2", 1.0 / 3.0)),
            float(weights_cfg.get("C3", 1.0 / 3.0)),
        )
    else:
        weights = weights_cfg

    env_risk_means = {env: _stack(values) for env, values in risk_by_env.items() if values}
    env_repr_means = {
        env: torch.stack(values) if values else torch.zeros(1)
        for env, values in repr_by_env.items()
    }
    grad_vectors: List[torch.Tensor] = []
    for values in grads_by_env.values():
        for tensor in values:
            grad_vectors.append(tensor.reshape(-1))

    C1 = compute_C1_global_stability(env_risk_means, norm_cfg)
    C2 = compute_C2_mechanistic_stability(grad_vectors)
    C3 = compute_C3_structural_stability(env_repr_means, norm_cfg)
    ISI_value = compute_ISI(C1, C2, C3, weights)

    env_outcomes = {env: _stack(values) for env, values in outcome_by_env.items() if values}
    IG = compute_IG(env_outcomes)
    WG = compute_WG(env_risk_means)
    VR = compute_VR(env_risk_means)

    records: List[Dict[str, object]] = []
    all_positions: List[torch.Tensor] = []
    all_outcomes: List[torch.Tensor] = []

    def _append_records(env_id: str, metrics: Mapping[str, float]) -> None:
        base = {
            "env": env_id,
            "split": run_ctx.split_name,
            "seed": run_ctx.seed,
            "algo": run_ctx.exp_id,
        }
        for metric_name, metric_value in metrics.items():
            records.append(
                {
                    **base,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    for env_id, risks in env_risk_means.items():
        er = compute_ER(env_outcomes.get(env_id, torch.tensor([])))
        env_positions_list = positions_by_env.get(env_id, [])
        if env_positions_list:
            stacked_positions = torch.cat(env_positions_list, dim=0)
            tr = compute_TR(stacked_positions)
        else:
            tr = 0.0
        metrics = {
            "C1_global_stability": float(C1),
            "C2_mechanistic_stability": float(C2),
            "C3_structural_stability": float(C3),
            "ISI": float(ISI_value),
            "IG": float(IG),
            "WG_risk": float(WG),
            "VR_risk": float(VR),
            "ER_mean_pnl": float(er),
            "TR_turnover": float(tr),
            "n_obs": float(n_obs_by_env.get(env_id, 0)),
        }
        _append_records(env_id, metrics)
        if env_outcomes.get(env_id) is not None:
            all_outcomes.append(env_outcomes[env_id])
        env_positions = positions_by_env.get(env_id, [])
        if env_positions:
            all_positions.extend(env_positions)

    overall_metrics = {
        "C1_global_stability": float(C1),
        "C2_mechanistic_stability": float(C2),
        "C3_structural_stability": float(C3),
        "ISI": float(ISI_value),
        "IG": float(IG),
        "WG_risk": float(WG),
        "VR_risk": float(VR),
        "ER_mean_pnl": float(compute_ER(_stack(all_outcomes)) if all_outcomes else 0.0),
        "TR_turnover": float(
            compute_TR(torch.cat(all_positions, dim=0)) if all_positions else 0.0
        ),
        "n_obs": float(sum(n_obs_by_env.values())),
    }
    _append_records("__overall__", overall_metrics)

    table = pd.DataFrame.from_records(records, columns=CANONICAL_COLUMNS)
    table = normalize_diagnostics_frame(table)
    validate_diagnostics_table(table)
    parquet_path = output_dir / "diagnostics.parquet"
    try:
        table.to_parquet(parquet_path, index=False)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Writing diagnostics parquet requires 'pyarrow' or 'fastparquet' to be installed"
        ) from exc

    manifest = {
        "seed": run_ctx.seed,
        "git_hash": run_ctx.git_hash,
        "config_hash": run_ctx.config_hash,
        "instrument": run_ctx.instrument,
        "metric_basis": run_ctx.metric_basis,
        "isi_weights": list(weights) if weights is not None else [1 / 3, 1 / 3, 1 / 3],
        "units": run_ctx.units or {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "diagnostics_table": parquet_path.name,
        "columns": list(CANONICAL_COLUMNS),
        "split": run_ctx.split_name,
        "algo": run_ctx.exp_id,
        "regime_tag": run_ctx.regime_tag,
        "rows": int(len(table)),
    }

    manifest_path = output_dir / "diagnostics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return parquet_path


__all__ = ["DiagnosticsRunContext", "gather_and_export"]

