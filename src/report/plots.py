"""Plotting utilities for the reporting pipeline."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig: plt.Figure, base_path: Path, dpi: int = 160) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _build_matrix(stats_frame: pd.DataFrame, metrics: Sequence[str], regimes_order: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(metrics), len(regimes_order)))
    matrix[:] = np.nan
    for i, metric in enumerate(metrics):
        for j, regime in enumerate(regimes_order):
            subset = stats_frame[(stats_frame["metric"] == metric) & (stats_frame["regime"] == regime)]
            if subset.empty:
                continue
            matrix[i, j] = float(subset.iloc[0]["mean"])
    return matrix


def plot_heatmap(
    stats_frame: pd.DataFrame,
    metrics: Sequence[str],
    regimes_order: Sequence[str],
    *,
    title: str,
    output_path: Path,
    dpi: int = 160,
) -> None:
    if not metrics:
        return
    matrix = _build_matrix(stats_frame, metrics, regimes_order)
    fig, ax = plt.subplots(figsize=(1.6 + len(regimes_order), 1.0 + len(metrics) * 0.4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(regimes_order)))
    ax.set_xticklabels(regimes_order, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title(title)
    for i in range(len(metrics)):
        for j in range(len(regimes_order)):
            value = matrix[i, j]
            if math.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, output_path, dpi=dpi)


def _prepare_metric_samples(seed_frame: pd.DataFrame, metric: str, regimes_order: Sequence[str]) -> list[np.ndarray]:
    samples = []
    for regime in regimes_order:
        subset = seed_frame[(seed_frame["metric"] == metric) & (seed_frame["regime"] == regime)]
        samples.append(subset["value"].astype(float).to_numpy())
    return samples


def plot_seed_distribution(
    seed_frame: pd.DataFrame,
    metric: str,
    regimes_order: Sequence[str],
    *,
    output_path: Path,
    dpi: int = 160,
) -> None:
    data = _prepare_metric_samples(seed_frame, metric, regimes_order)
    if all(len(d) == 0 for d in data):
        return
    fig, ax = plt.subplots(figsize=(1.6 + len(regimes_order), 3))
    ax.boxplot(data, labels=regimes_order)
    ax.set_ylabel(metric)
    ax.set_title(f"Seed distribution — {metric}")
    _save_figure(fig, output_path, dpi=dpi)


def plot_efficiency_frontier(
    seed_frame: pd.DataFrame,
    regimes_order: Sequence[str],
    *,
    er_metric: str = "ER_mean_pnl",
    tr_metric: str = "TR_turnover",
    output_path: Path,
    dpi: int = 160,
) -> None:
    subset = seed_frame[seed_frame["metric"].isin([er_metric, tr_metric])]
    if subset.empty:
        return

    pivot = subset.pivot_table(index=["seed", "regime"], columns="metric", values="value")
    if er_metric not in pivot.columns or tr_metric not in pivot.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(regimes_order)))
    color_map = dict(zip(regimes_order, colors))
    for regime, group in pivot.groupby(level="regime"):
        ax.scatter(group[tr_metric].astype(float), group[er_metric].astype(float), alpha=0.4, label=regime, color=color_map.get(regime, "grey"))

    mean_points = pivot.groupby(level="regime").mean()
    std_points = pivot.groupby(level="regime").std()
    for regime in regimes_order:
        if regime not in mean_points.index:
            continue
        mean_tr = float(mean_points.loc[regime, tr_metric])
        mean_er = float(mean_points.loc[regime, er_metric])
        std_tr = float(std_points.loc[regime, tr_metric]) if regime in std_points.index else 0.0
        std_er = float(std_points.loc[regime, er_metric]) if regime in std_points.index else 0.0
        ax.errorbar(mean_tr, mean_er, xerr=std_tr, yerr=std_er, fmt="o", color=color_map.get(regime, "black"), capsize=3, linewidth=2)

    ax.set_xlabel(tr_metric)
    ax.set_ylabel(er_metric)
    ax.set_title("Efficiency frontier")
    ax.legend(loc="best")
    _save_figure(fig, output_path, dpi=dpi)


def plot_qq(
    seed_frame: pd.DataFrame,
    regime: str,
    *,
    metric: str = "ER_mean_pnl",
    reference: str = "gaussian",
    output_path: Path,
    dpi: int = 160,
) -> None:
    subset = seed_frame[(seed_frame["regime"] == regime) & (seed_frame["metric"] == metric)]
    if subset.empty:
        return
    data = np.sort(subset["value"].astype(float).to_numpy())
    n = len(data)
    probs = (np.arange(1, n + 1) - 0.5) / n

    if reference == "calm_regime":
        reference_data = np.sort(seed_frame[(seed_frame["regime"] == "train_main") & (seed_frame["metric"] == metric)]["value"].astype(float).to_numpy())
        if len(reference_data) != n:
            reference_data = np.interp(probs, np.linspace(0, 1, max(len(reference_data), 2)), reference_data)
        theoretical = reference_data
    else:
        mu = float(np.mean(data))
        sigma = float(np.std(data))
        sigma = sigma if sigma > 0 else 1.0
        from statistics import NormalDist

        nd = NormalDist(mu=mu, sigma=sigma)
        theoretical = np.array([nd.inv_cdf(p) for p in probs])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(theoretical, data, alpha=0.7)
    min_val = float(min(theoretical.min(), data.min()))
    max_val = float(max(theoretical.max(), data.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="grey")
    ax.set_title(f"QQ plot — {regime}")
    ax.set_xlabel("Reference quantiles")
    ax.set_ylabel("Empirical quantiles")
    _save_figure(fig, output_path, dpi=dpi)


def generate_all_plots(
    seed_frame: pd.DataFrame,
    stats_frame: pd.DataFrame,
    *,
    metrics_config: Mapping[str, Sequence[str]],
    regimes_order: Sequence[str],
    figure_config: Mapping[str, object],
    output_dir: Path,
    lite: bool = False,
) -> None:
    dpi = int(figure_config.get("dpi_preview", 160))
    output_dir.mkdir(parents=True, exist_ok=True)

    for block_name, metrics in metrics_config.items():
        plot_heatmap(
            stats_frame,
            metrics,
            regimes_order,
            title=f"{block_name.title()} heatmap",
            output_path=output_dir / f"heatmap_{block_name}",
            dpi=dpi,
        )

    distribution_metrics = [m for m in ["WG_risk", "VR_risk", "ISI"] if m in seed_frame["metric"].unique()]
    for metric in distribution_metrics:
        plot_seed_distribution(
            seed_frame,
            metric,
            regimes_order,
            output_path=output_dir / f"seed_distribution_{metric}",
            dpi=dpi,
        )

    plot_efficiency_frontier(
        seed_frame,
        regimes_order,
        output_path=output_dir / "efficiency_frontier",
        dpi=dpi,
    )

    if lite:
        return

    qq_metric = "ER_mean_pnl"
    reference = str(figure_config.get("qq_reference", "gaussian"))
    for regime in regimes_order:
        plot_qq(
            seed_frame,
            regime,
            metric=qq_metric,
            reference=reference,
            output_path=output_dir / f"qq_{regime}",
            dpi=dpi,
        )
