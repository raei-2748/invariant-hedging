"""Aggregate diagnostics metrics from per-seed experiment folders."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from invariant_hedging.modules.diagnostics import (
    compute_ER,
    compute_IG,
    compute_TR,
    compute_VR,
    compute_WG,
    compute_invariance_spectrum,
)

DEFAULT_CONFIG: Mapping[str, Mapping[str, object]] = {
    "invariance": {
        "tau_risk": 1.0,
        "tau_cov": 1.0,
        "trim_ratio": 0.1,
        "weights": {"C1": 1.0 / 3.0, "C2": 1.0 / 3.0, "C3": 1.0 / 3.0},
    },
    "ig": {"tau_norm": 1.0},
    "wg": {"alpha": 0.25},
    "vr": {"epsilon": 1e-8},
    "er": {"cvar_alpha": 0.05},
    "tr": {"epsilon": 1e-8},
}


@dataclass(frozen=True)
class LeafContext:
    seed: int
    regime: str
    split: str
    path: Path


def _parse_seed(name: str) -> int:
    match = re.search(r"\d+", name)
    if match is not None:
        return int(match.group())
    return int(name)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_leaf_dirs(run_dir: Path) -> Iterable[LeafContext]:
    seeds_root = run_dir / "seeds"
    if not seeds_root.exists():
        return []
    for seed_dir in sorted(p for p in seeds_root.iterdir() if p.is_dir()):
        try:
            seed = _parse_seed(seed_dir.name)
        except ValueError:
            continue
        for regime_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            regime = regime_dir.name
            for split_dir in sorted(p for p in regime_dir.iterdir() if p.is_dir()):
                split = split_dir.name
                diag_dir = split_dir / "diagnostics"
                yield LeafContext(seed=seed, regime=regime, split=split, path=diag_dir if diag_dir.exists() else split_dir)


def _extract_positions(pnl_df: pd.DataFrame) -> pd.DataFrame:
    if pnl_df.empty:
        return pd.DataFrame()
    action_cols = [col for col in pnl_df.columns if col not in {"step", "pnl"}]
    if not action_cols:
        return pd.DataFrame()
    return pnl_df[["step", *action_cols]].copy()


def aggregate_leaf(ctx: LeafContext, config: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    risk_path = ctx.path / "risk.csv"
    grad_path = ctx.path / "alignment_head.csv"
    rep_path = ctx.path / "feature_dispersion.csv"
    risk_series_path = ctx.path / "risk_series.csv"
    pnl_path = ctx.path / "pnl.csv"
    cvar_path = ctx.path / "cvar95.json"

    risk_df = _load_csv(risk_path)
    if risk_df.empty and {"probe_id", "env", "risk"} - set(risk_df.columns):
        risk_df = pd.DataFrame(columns=["probe_id", "env", "risk"])
    grad_df = _load_csv(grad_path)
    if grad_df.empty and {"probe_id", "env_i", "env_j", "cosine"} - set(grad_df.columns):
        grad_df = pd.DataFrame(columns=["probe_id", "env_i", "env_j", "cosine"])
    rep_df = _load_csv(rep_path)
    if rep_df.empty and {"probe_id", "dispersion"} - set(rep_df.columns):
        rep_df = pd.DataFrame(columns=["probe_id", "dispersion"])
    risk_series_df = _load_csv(risk_series_path)
    if risk_series_df.empty and {"step", "risk"} - set(risk_series_df.columns):
        risk_series_df = pd.DataFrame(columns=["step", "risk"])
    pnl_df = _load_csv(pnl_path)
    if pnl_df.empty and {"step", "pnl"} - set(pnl_df.columns):
        pnl_df = pd.DataFrame(columns=["step", "pnl"])
    cvar_info = _load_json(cvar_path)

    invariance = compute_invariance_spectrum(rep_df, grad_df, risk_df, config.get("invariance"))
    if not risk_df.empty:
        env_summary = (
            risk_df.groupby("env")["risk"].mean().reset_index().rename(columns={"risk": "value"})
        )
    else:
        env_summary = pd.DataFrame(columns=["env", "value"])
    ig = compute_IG(env_summary, config.get("ig"))

    wg_input = risk_df[["env", "risk"]] if {"env", "risk"}.issubset(risk_df.columns) else pd.DataFrame(columns=["env", "risk"])
    wg = compute_WG(wg_input, config.get("wg"))
    vr_input = risk_series_df if not risk_series_df.empty else pd.DataFrame(columns=["step", "risk"])
    vr = compute_VR(vr_input, config.get("vr"))

    er = compute_ER(pnl_df[["step", "pnl"]] if not pnl_df.empty else pnl_df, config.get("er"))
    positions_df = _extract_positions(pnl_df)
    tr = compute_TR(positions_df, config.get("tr"))

    mean_pnl = er.get("mean_pnl", float("nan"))
    cvar95 = cvar_info.get("cvar95", er.get("cvar"))
    turnover = tr.get("mean_turnover")

    return {
        "ctx": ctx,
        "risk_df": risk_df,
        "invariance": invariance,
        "ig": ig,
        "wg": wg,
        "vr": vr,
        "er": er,
        "tr": tr,
        "mean_pnl": mean_pnl,
        "cvar95": cvar95,
        "turnover": turnover,
    }


def aggregate_run(run_dir: Path | str, config: Mapping[str, Mapping[str, object]] | None = None) -> Dict[str, pd.DataFrame]:
    run_path = Path(run_dir)
    cfg = DEFAULT_CONFIG if config is None else config

    records: List[Dict[str, object]] = []
    invariance_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []

    for ctx in _iter_leaf_dirs(run_path):
        result = aggregate_leaf(ctx, cfg)
        invariance = result["invariance"]
        ig = result["ig"]
        wg = result["wg"]
        vr = result["vr"]
        er = result["er"]
        tr = result["tr"]

        record = {
            "seed": ctx.seed,
            "regime_name": ctx.regime,
            "split": ctx.split,
            "ISI": invariance.get("ISI"),
            "IG": ig.get("IG"),
            "IG_norm": ig.get("IG_norm"),
            "WG": wg.get("WG"),
            "VR": vr.get("VR"),
            "ER": er.get("ER"),
            "TR": tr.get("TR"),
            "C1": invariance.get("C1"),
            "C2": invariance.get("C2"),
            "C3": invariance.get("C3"),
            "mean_pnl": result.get("mean_pnl"),
            "cvar95": result.get("cvar95"),
            "turnover": result.get("turnover"),
        }
        records.append(record)

        invariance_rows.append(
            {
                "seed": ctx.seed,
                "regime_name": ctx.regime,
                "split": ctx.split,
                "type": "aggregate",
                "probe_id": "",
                "ISI": invariance.get("ISI"),
                "C1": invariance.get("C1"),
                "C2": invariance.get("C2"),
                "C3": invariance.get("C3"),
                "IG": ig.get("IG"),
                "IG_norm": ig.get("IG_norm"),
            }
        )
        for probe_id, comps in invariance.get("per_probe", {}).items():
            invariance_rows.append(
                {
                    "seed": ctx.seed,
                    "regime_name": ctx.regime,
                    "split": ctx.split,
                    "type": "probe",
                    "probe_id": probe_id,
                    "C1": comps.get("C1"),
                    "C2": comps.get("C2"),
                    "C3": comps.get("C3"),
                }
            )

        robustness_rows.append(
            {
                "seed": ctx.seed,
                "regime_name": ctx.regime,
                "split": ctx.split,
                "WG": wg.get("WG"),
                "tau": wg.get("tau"),
                "alpha": wg.get("alpha"),
                "VR": vr.get("VR"),
                "risk_mean": vr.get("mean"),
                "risk_std": vr.get("std"),
            }
        )

        efficiency_rows.append(
            {
                "seed": ctx.seed,
                "regime_name": ctx.regime,
                "split": ctx.split,
                "ER": er.get("ER"),
                "mean_pnl": er.get("mean_pnl"),
                "cvar": er.get("cvar"),
                "TR": tr.get("TR"),
                "turnover": tr.get("mean_turnover"),
                "position_norm": tr.get("mean_position"),
            }
        )

    summary_df = pd.DataFrame.from_records(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["seed", "regime_name", "split"]).reset_index(drop=True)

    invariance_df = pd.DataFrame.from_records(invariance_rows)
    robustness_df = pd.DataFrame.from_records(robustness_rows)
    efficiency_df = pd.DataFrame.from_records(efficiency_rows)

    frontier_df = summary_df[[
        "seed",
        "regime_name",
        "split",
        "mean_pnl",
        "cvar95",
        "ER",
        "TR",
    ]].copy() if not summary_df.empty else pd.DataFrame(columns=["seed", "regime_name", "split", "mean_pnl", "cvar95", "ER", "TR"])

    tables = {
        "diagnostics_summary": summary_df,
        "invariance_diagnostics": invariance_df,
        "robustness_diagnostics": robustness_df,
        "efficiency_diagnostics": efficiency_df,
        "capital_efficiency_frontier": frontier_df,
    }

    tables_dir = run_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        output_path = tables_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)

    return tables


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate diagnostics tables")
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing seed folders")
    return parser.parse_args(args=args)


def main(args: Iterable[str] | None = None) -> Dict[str, pd.DataFrame]:
    namespace = parse_args(args)
    return aggregate_run(namespace.run_dir)


if __name__ == "__main__":
    main()
