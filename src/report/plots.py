"""Plotting utilities for the reporting pipeline."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence, Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _apply_style(fig_cfg: Mapping[str, Any]) -> None:
    matplotlib.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def plot_heatmaps(summary: pd.DataFrame, config: Mapping[str, Any], output_dir: Path) -> None:
    fig_cfg = config.get("figures", {})
    _apply_style(fig_cfg)
    dpi = int(fig_cfg.get("dpi_preview", 160))
    regimes = config.get("regimes_order", [])
    metrics_cfg = config.get("metrics", {})
    _ensure_dir(output_dir)

    for block, metrics in metrics_cfg.items():
        block_df = summary[summary["metric"].isin(metrics)]
        if block_df.empty:
            continue
        pivot = block_df.pivot(index="metric", columns="regime", values="mean")
        ci = block_df.pivot(index="metric", columns="regime", values="ci_half_width")
        pivot = pivot.reindex(metrics)
        pivot = pivot[[reg for reg in regimes if reg in pivot.columns]]
        ci = ci.reindex_like(pivot)

        fig, ax = plt.subplots(figsize=(1.8 * len(regimes), 0.5 * len(metrics) + 1))
        im = ax.imshow(pivot.values, cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{block.title()} heatmap")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ci_val = ci.iloc[i, j]
                if math.isnan(value):
                    text = "--"
                else:
                    text = f"{value:.3f}\n±{ci_val:.3f}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        pdf_path = output_dir / f"heatmap_{block}.pdf"
        png_path = output_dir / f"heatmap_{block}.png"
        fig.savefig(pdf_path, dpi=max(dpi, 200))
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)


def plot_scorecard(summary: pd.DataFrame, config: Mapping[str, Any], output_dir: Path) -> None:
    fig_cfg = config.get("figures", {})
    _apply_style(fig_cfg)
    dpi = int(fig_cfg.get("dpi_preview", 160))
    regimes = config.get("regimes_order", [])
    metrics_cfg = config.get("metrics", {})
    _ensure_dir(output_dir)

    metrics = [metric for block in metrics_cfg.values() for metric in block]
    score_df = summary[summary["metric"].isin(metrics)]
    if score_df.empty:
        return
    pivot = score_df.pivot(index="metric", columns="regime", values="mean")
    ci = score_df.pivot(index="metric", columns="regime", values="ci_half_width")
    pivot = pivot.reindex(metrics)
    pivot = pivot[[reg for reg in regimes if reg in pivot.columns]]
    ci = ci.reindex_like(pivot)

    fig, ax = plt.subplots(figsize=(2 + len(regimes) * 1.2, 0.3 * len(metrics) + 1.5))
    ax.axis("off")
    table_data = []
    for metric in pivot.index:
        row = []
        for regime in pivot.columns:
            value = pivot.loc[metric, regime]
            ci_val = ci.loc[metric, regime]
            if math.isnan(value):
                row.append("--")
            else:
                row.append(f"{value:.3f} ± {ci_val:.3f}")
        table_data.append(row)
    table = ax.table(
        cellText=table_data,
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Scorecard (mean ± CI)", pad=20)
    fig.tight_layout()
    pdf_path = output_dir / "scorecard.pdf"
    png_path = output_dir / "scorecard.png"
    fig.savefig(pdf_path, dpi=max(dpi, 200))
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def plot_qq(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
    metric: str,
) -> None:
    fig_cfg = config.get("figures", {})
    _apply_style(fig_cfg)
    _ensure_dir(output_dir)
    regimes = config.get("regimes_order", [])
    bins = int(config.get("qq", {}).get("bins", 100))
    reference = config.get("qq", {}).get("reference", "gaussian")

    fig, axes = plt.subplots(1, len(regimes), figsize=(4 * len(regimes), 3), squeeze=False)
    axes = axes[0]
    reference_values = None
    if reference == "calm_regime" and regimes:
        calm_regime = regimes[0]
        calm_vals = raw[(raw["regime"] == calm_regime) & (raw["metric"] == metric)]["value"].astype(float)
        if len(calm_vals) >= 2:
            reference_values = np.sort(calm_vals)
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        data = raw[(raw["regime"] == regime) & (raw["metric"] == metric)]["value"].astype(float)
        if data.empty:
            ax.set_visible(False)
            continue
        quantiles = np.linspace(0, 1, min(len(data), bins), endpoint=False)[1:]
        if quantiles.size == 0:
            ax.set_visible(False)
            continue
        empirical = np.quantile(data, quantiles)
        if reference == "gaussian" or reference_values is None:
            mu = float(np.mean(data))
            sigma = float(np.std(data)) or 1.0
            dist = NormalDist(mu, sigma)
            theoretical = np.array([dist.inv_cdf(q) for q in quantiles])
        else:
            theoretical = np.quantile(reference_values, quantiles)
        ax.scatter(theoretical, empirical, s=12, alpha=0.8)
        lo = min(theoretical.min(), empirical.min())
        hi = max(theoretical.max(), empirical.max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(regime)
        ax.set_xlabel("Reference quantile")
        ax.set_ylabel("Empirical quantile")
    fig.tight_layout()
    fig.savefig(output_dir / f"qq_{metric}.pdf", dpi=200)
    fig.savefig(output_dir / f"qq_{metric}.png", dpi=fig_cfg.get("dpi_preview", 160))
    plt.close(fig)


def plot_seed_distributions(
    raw: pd.DataFrame,
    metrics: Sequence[str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    fig_cfg = config.get("figures", {})
    _apply_style(fig_cfg)
    _ensure_dir(output_dir)
    regimes = config.get("regimes_order", [])

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(1.5 * len(regimes) + 1, 3))
        data = [
            raw[(raw["regime"] == regime) & (raw["metric"] == metric)]["value"].astype(float).values
            for regime in regimes
        ]
        ax.boxplot(data, labels=regimes, showmeans=True)
        ax.set_title(f"Seed distribution: {metric}")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(output_dir / f"distribution_{metric}.pdf", dpi=200)
        fig.savefig(output_dir / f"distribution_{metric}.png", dpi=fig_cfg.get("dpi_preview", 160))
        plt.close(fig)


def plot_efficiency_frontier(
    raw: pd.DataFrame,
    efficiency_metric: str,
    turnover_metric: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    fig_cfg = config.get("figures", {})
    _apply_style(fig_cfg)
    _ensure_dir(output_dir)

    index_cols = [col for col in ("algo", "seed", "regime") if col in raw.columns]
    pivot = (
        raw[raw["metric"].isin([efficiency_metric, turnover_metric])]
        .pivot_table(index=index_cols, columns="metric", values="value", aggfunc="mean")
        .reset_index()
    )
    if pivot.empty:
        return

    regimes = config.get("regimes_order", [])
    fig, ax = plt.subplots(figsize=(5, 4))
    for regime in regimes:
        regime_df = pivot[pivot["regime"] == regime]
        if regime_df.empty:
            continue
        ax.scatter(
            regime_df[turnover_metric],
            regime_df[efficiency_metric],
            label=regime,
            alpha=0.4,
            s=30,
        )
        mean_x = regime_df[turnover_metric].mean()
        mean_y = regime_df[efficiency_metric].mean()
        std_x = regime_df[turnover_metric].std(ddof=1)
        std_y = regime_df[efficiency_metric].std(ddof=1)
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=std_x,
            yerr=std_y,
            fmt="o",
            color="black",
            capsize=4,
        )
    ax.set_xlabel(turnover_metric)
    ax.set_ylabel(efficiency_metric)
    ax.set_title("Efficiency frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency_frontier.pdf", dpi=200)
    fig.savefig(output_dir / "efficiency_frontier.png", dpi=fig_cfg.get("dpi_preview", 160))
    plt.close(fig)


__all__ = [
    "plot_heatmaps",
    "plot_scorecard",
    "plot_qq",
    "plot_seed_distributions",
    "plot_efficiency_frontier",
]
