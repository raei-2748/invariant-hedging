from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import (
    ReportInputs,
    attach_deltas,
    ensure_assets_dir,
    load_report_inputs,
    resolve_metric_columns,
)
from . import confidence_interval as compute_ci

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _preferred_metric(per_seed: pd.DataFrame) -> str | None:
    metrics = resolve_metric_columns(per_seed)
    for key in ("es95", "es90", "es99", "meanpnl"):
        column = metrics.get(key)
        if column:
            return column
    if metrics:
        return next(iter(metrics.values()))
    return None


def _penalty_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "lambda",
        "penalty",
        "beta",
        "coef",
        "alpha",
    ]
    lowered = {col.lower(): col for col in df.columns}
    for token in candidates:
        if token in lowered:
            return lowered[token]
    for token in candidates:
        for col_lower, col in lowered.items():
            if token in col_lower:
                return col
    return None


def _prepare_penalty_stats(df: pd.DataFrame, penalty_col: str, metric_col: str) -> pd.DataFrame:
    subset = df[["method", penalty_col, metric_col]].dropna()
    if subset.empty:
        return pd.DataFrame()
    subset = subset.copy()
    subset[penalty_col] = pd.to_numeric(subset[penalty_col], errors="coerce")
    subset[metric_col] = pd.to_numeric(subset[metric_col], errors="coerce")
    subset = subset.dropna(subset=[penalty_col, metric_col])
    if subset.empty:
        return pd.DataFrame()
    records: list[dict[str, float]] = []
    for (method, penalty), values in subset.groupby(["method", penalty_col])[metric_col]:
        stats = {
            "method": method,
            "penalty": float(penalty),
        }
        mean, low, high = compute_ci(values.to_numpy(dtype=float))
        stats.update({"mean": mean, "low": low, "high": high})
        records.append(stats)
    return pd.DataFrame.from_records(records)


def plot_penalty_sweep(inputs: ReportInputs, out_path: Path, dpi: int) -> bool:
    df = inputs.per_seed
    if df.empty:
        LOGGER.info("Skipping penalty sweep: no per-seed records")
        return False
    penalty_col = _penalty_column(df)
    metric_col = _preferred_metric(df)
    if not penalty_col or not metric_col:
        LOGGER.info("Skipping penalty sweep: required columns not found")
        return False
    stats = _prepare_penalty_stats(df, penalty_col, metric_col)
    if stats.empty:
        LOGGER.info("Skipping penalty sweep: no usable penalty records")
        return False

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")

    for idx, method in enumerate(sorted(stats["method"].unique())):
        subset = stats[stats["method"] == method].sort_values("penalty")
        if subset.empty:
            continue
        color = cmap(idx % cmap.N)
        penalties = subset["penalty"].to_numpy(dtype=float)
        means = subset["mean"].to_numpy(dtype=float)
        lows = subset["low"].to_numpy(dtype=float)
        highs = subset["high"].to_numpy(dtype=float)
        ax.plot(penalties, means, marker="o", color=color, label=method, linewidth=1.6)
        ax.fill_between(penalties, lows, highs, color=color, alpha=0.2)

    if (stats["penalty"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(penalty_col.replace("_", " ").title())
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title("Penalty Sweep")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved penalty sweep to %s", out_path)
    return True


def _method_category(method: str) -> str | None:
    lowered = method.lower()
    if "head" in lowered:
        return "Head-only"
    if any(token in lowered for token in ("feature", "repr", "encoder")):
        return "Feature"
    if any(token in lowered for token in ("joint", "full", "baseline")):
        return "Joint"
    return None


def plot_head_vs_feature_ablation(inputs: ReportInputs, out_path: Path, dpi: int) -> bool:
    df = inputs.per_seed
    if df.empty or "method" not in df.columns:
        LOGGER.info("Skipping head vs feature ablation: no per-seed data")
        return False
    metric_col = _preferred_metric(df)
    if not metric_col:
        LOGGER.info("Skipping head vs feature ablation: metric unavailable")
        return False
    df = df.copy()
    df["category"] = df["method"].apply(_method_category)
    df = df.dropna(subset=["category"])
    if df.empty:
        LOGGER.info("Skipping head vs feature ablation: categories not detected")
        return False

    records: list[dict[str, float]] = []
    for category, group in df.groupby("category"):
        values = pd.to_numeric(group[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean, low, high = compute_ci(values)
        records.append({"category": category, "mean": mean, "low": low, "high": high})
    if len(records) < 2:
        LOGGER.info("Skipping head vs feature ablation: insufficient categories")
        return False
    stats = pd.DataFrame.from_records(records).sort_values("category")

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(stats))
    colors = plt.get_cmap("tab10")
    ax.bar(positions, stats["mean"], color=[colors(i % colors.N) for i in range(len(stats))], alpha=0.8)
    ax.errorbar(positions, stats["mean"], yerr=[stats["mean"] - stats["low"], stats["high"] - stats["mean"]], fmt="none", color="black", capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(stats["category"], rotation=15, ha="right")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title("Head vs Feature Ablation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved head vs feature ablation to %s", out_path)
    return True


def plot_isi_decomposition(inputs: ReportInputs, out_path: Path, dpi: int) -> bool:
    df = inputs.diagnostics
    if df.empty:
        LOGGER.info("Skipping ISI decomposition: diagnostics unavailable")
        return False
    components = [col for col in ("ig", "wg", "msi") if col in df.columns]
    if not components:
        LOGGER.info("Skipping ISI decomposition: no diagnostic columns found")
        return False
    stats = df.groupby("method")[components].mean(numeric_only=True)
    if stats.empty:
        LOGGER.info("Skipping ISI decomposition: aggregation empty")
        return False
    stats = stats.sort_index()

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = stats.index.tolist()
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    cmap = plt.get_cmap("tab10")
    for idx, component in enumerate(components):
        values = stats[component].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=component.upper(), color=cmap(idx % cmap.N), alpha=0.85)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Average value")
    ax.set_title("ISI Decomposition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved ISI decomposition to %s", out_path)
    return True


def plot_cross_regime_heatmap(inputs: ReportInputs, out_path: Path, dpi: int) -> bool:
    df = inputs.per_seed
    if df.empty or "method" not in df.columns:
        LOGGER.info("Skipping cross-regime heatmap: no per-seed data")
        return False

    env_columns: dict[str, list[str]] = {}
    for col in df.columns:
        lower = col.lower()
        if not any(token in lower for token in ("es", "cvar")):
            continue
        env = None
        for token in ("train", "val", "validation", "test", "crisis", "ood"):
            if token in lower:
                env = "val" if token == "validation" else token
                break
        if env is None:
            continue
        env_columns.setdefault(env, []).append(col)
    if not env_columns:
        LOGGER.info("Skipping cross-regime heatmap: no regime-specific columns detected")
        return False

    records: list[dict[str, float]] = []
    for method, group in df.groupby("method"):
        row: dict[str, float] = {"method": method}
        for env, columns in env_columns.items():
            values: list[np.ndarray] = []
            for column in columns:
                arr = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
                if arr.size:
                    values.append(arr)
            if not values:
                row[env] = math.nan
                continue
            combined = np.concatenate([v[~np.isnan(v)] for v in values if v.size])
            row[env] = float(np.nanmean(combined)) if combined.size else math.nan
        records.append(row)
    heatmap = pd.DataFrame.from_records(records).set_index("method")
    if heatmap.empty:
        LOGGER.info("Skipping cross-regime heatmap: no aggregated data")
        return False

    preferred_order = ["train", "val", "test", "crisis", "ood"]
    ordered_cols = [col for col in preferred_order if col in heatmap.columns]
    ordered_cols.extend([col for col in heatmap.columns if col not in ordered_cols])
    heatmap = heatmap[ordered_cols]

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(1.5 + len(ordered_cols) * 1.2, 1.5 + len(heatmap) * 0.4))
    matrix = heatmap.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels([col.upper() for col in ordered_cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title("Cross-Regime Tail Risk")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Metric", rotation=-90, va="bottom")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved cross-regime heatmap to %s", out_path)
    return True


def plot_capital_frontier(scorecard: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if scorecard is None or scorecard.empty:
        LOGGER.info("Skipping capital frontier: scorecard unavailable")
        return False
    required = {"method", "es95_mean", "meanpnl_mean", "turnover_mean", "n_seeds"}
    if not required.issubset(scorecard.columns):
        LOGGER.info("Skipping capital frontier: required scorecard columns missing")
        return False
    df = scorecard.copy()
    df = df[df["n_seeds"] > 0]
    if df.empty:
        LOGGER.info("Skipping capital frontier: no populated methods")
        return False
    df["abs_es95"] = df["es95_mean"].abs()

    methods = df["method"].tolist()
    risks = df["abs_es95"].to_numpy(dtype=float)
    pnls = df["meanpnl_mean"].to_numpy(dtype=float)
    turns = df["turnover_mean"].to_numpy(dtype=float)

    def _pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
        dominant: list[str] = []
        for name_i, risk_i, pnl_i in points:
            if math.isnan(risk_i) or math.isnan(pnl_i):
                continue
            dominated = False
            for name_j, risk_j, pnl_j in points:
                if name_i == name_j:
                    continue
                if math.isnan(risk_j) or math.isnan(pnl_j):
                    continue
                better_or_equal_risk = risk_j <= risk_i + 1e-9
                better_or_equal_pnl = pnl_j >= pnl_i - 1e-9
                strictly_better = (risk_j < risk_i - 1e-9) or (pnl_j > pnl_i + 1e-9)
                if better_or_equal_risk and better_or_equal_pnl and strictly_better:
                    dominated = True
                    break
            if not dominated:
                dominant.append(name_i)
        return dominant

    pareto = _pareto_front(list(zip(methods, risks, pnls)))

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    min_turn = np.nanmin(turns)
    max_turn = np.nanmax(turns)
    turn_range = max(max_turn - min_turn, 1e-6)

    for idx, method in enumerate(methods):
        color = cmap(idx % cmap.N)
        risk = risks[idx]
        pnl = pnls[idx]
        turn = turns[idx]
        if math.isnan(risk) or math.isnan(pnl):
            continue
        size = 120 * (1 + (0 if math.isnan(turn) else (turn - min_turn) / turn_range))
        edgecolor = "black" if method in pareto else color
        facecolor = color if method in pareto else (*color[:3], 0.5)
        ax.scatter(risk, pnl, s=size, color=facecolor, edgecolor=edgecolor, linewidths=1.0, alpha=0.85, zorder=3)
        ax.text(risk, pnl, f" {method}", ha="left", va="center", fontsize=9)

    ax.set_xlabel("|ES95 Mean|")
    ax.set_ylabel("Mean PnL")
    ax.set_title("Capital-Efficiency Frontier")
    ax.grid(alpha=0.3)
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="black", label="Pareto-dominant", markerfacecolor="none", markersize=8),
        plt.Line2D([0], [0], marker="o", color="gray", label="Non-dominant", markerfacecolor="gray", alpha=0.5, markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved capital frontier to %s", out_path)
    return True


def plot_cvar_violin(diagnostics: pd.DataFrame, scorecard: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if diagnostics is None or diagnostics.empty:
        LOGGER.info("Skipping CVaR violin: diagnostics unavailable")
        return False
    if "es95_crisis" in diagnostics.columns:
        metric_col = "es95_crisis"
    elif "es95" in diagnostics.columns:
        metric_col = "es95"
    else:
        LOGGER.info("Skipping CVaR violin: no crisis ES95 metric")
        return False
    df = diagnostics.dropna(subset=[metric_col])
    if df.empty:
        LOGGER.info("Skipping CVaR violin: no finite observations")
        return False

    ordered: list[str] = []
    if scorecard is not None and not scorecard.empty and "method" in scorecard.columns:
        ordered.extend(list(dict.fromkeys(scorecard["method"].tolist())))
    diag_methods = list(dict.fromkeys(df["method"].tolist()))
    for method in diag_methods:
        if method not in ordered:
            ordered.append(method)
    if not ordered:
        LOGGER.info("Skipping CVaR violin: no methods")
        return False

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    positions = np.arange(len(ordered))
    rng = np.random.default_rng(seed=42)

    for idx, method in enumerate(ordered):
        subset = df[df["method"] == method][metric_col].dropna()
        values = subset.to_numpy(dtype=float)
        if values.size == 0:
            continue
        color = cmap(idx % cmap.N)
        parts = ax.violinplot(values, positions=[positions[idx]], widths=0.8, showmeans=False, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        jitter = (rng.random(values.size) - 0.5) * 0.12
        ax.scatter(np.full(values.size, positions[idx]) + jitter, values, color=color, edgecolor="black", linewidths=0.4, alpha=0.8, zorder=3)
        mean, low, high = compute_ci(values)
        if not math.isnan(mean) and not math.isnan(low) and not math.isnan(high):
            ax.errorbar(positions[idx], mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black", ecolor="black", elinewidth=1.2, capsize=4, markersize=5, zorder=4)
            ax.text(positions[idx], mean, f" {mean:.2f}", va="center", ha="left", fontsize=9, color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered, rotation=20, ha="right")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_xlabel("Method")
    ax.set_title("Crisis ES95 Dispersion by Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved CVaR violin to %s", out_path)
    return True


def plot_ig_vs_cvar(diagnostics: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if diagnostics is None or diagnostics.empty:
        LOGGER.info("Skipping IG vs CVaR: diagnostics unavailable")
        return False
    if not {"ig", "es95_crisis"}.issubset(diagnostics.columns):
        LOGGER.info("Skipping IG vs CVaR: required columns missing")
        return False
    df = diagnostics.dropna(subset=["ig", "es95_crisis"])
    if df.empty:
        LOGGER.info("Skipping IG vs CVaR: no finite observations")
        return False

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    method_colors = {method: cmap(idx % cmap.N) for idx, method in enumerate(sorted(df["method"].unique()))}

    for method, group in df.groupby("method"):
        color = method_colors.get(method, (0.2, 0.2, 0.2, 1.0))
        ax.scatter(group["ig"].to_numpy(dtype=float), group["es95_crisis"].to_numpy(dtype=float), label=method, color=color, edgecolor="black", linewidths=0.4, alpha=0.8)

    x = df["ig"].to_numpy(dtype=float)
    y = df["es95_crisis"].to_numpy(dtype=float)
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), num=200)
        ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1.5, label="Least-squares fit")
        try:  # pragma: no cover - SciPy optional
            from scipy import stats

            r, p = stats.pearsonr(x, y)
        except Exception:
            xm = x - x.mean()
            ym = y - y.mean()
            denom = math.sqrt(float(np.sum(xm ** 2)) * float(np.sum(ym ** 2)))
            r = float(np.sum(xm * ym) / denom) if denom else math.nan
            p = math.nan
        caption = f"r = {r:.3f}, p = {p:.3g}" if not math.isnan(r) else "r unavailable"
        ax.text(0.02, 0.95, caption, transform=ax.transAxes, ha="left", va="top", fontsize=11)
    ax.set_xlabel("IG (Train ES95 gap)")
    ax.set_ylabel("Crisis ES95")
    ax.set_title("IG vs. Crisis ES95")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Saved IG vs CVaR scatter to %s", out_path)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase-2 report figures")
    parser.add_argument("--per-seed", action="append", type=Path, help="Explicit per-seed CSV paths")
    parser.add_argument("--scorecard", type=Path, default=None, help="Optional precomputed scorecard CSV")
    parser.add_argument("--assets-dir", type=Path, default=None, help="Directory for generated figures")
    parser.add_argument("--search-root", action="append", type=Path, help="Additional directories to scan for CSVs")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    assets_dir = ensure_assets_dir(args.assets_dir)
    search_roots = tuple(args.search_root) if args.search_root else None
    inputs = load_report_inputs(
        per_seed_paths=args.per_seed or None,
        scorecard_path=args.scorecard,
        search_roots=search_roots,
    )
    scorecard = attach_deltas(inputs.scorecard) if inputs.scorecard is not None else None

    generators = [
        ("fig_penalty_sweep.png", lambda path: plot_penalty_sweep(inputs, path, args.dpi)),
        ("fig_head_vs_feature.png", lambda path: plot_head_vs_feature_ablation(inputs, path, args.dpi)),
        ("fig_isi_decomposition.png", lambda path: plot_isi_decomposition(inputs, path, args.dpi)),
        ("fig_cross_regime_heatmap.png", lambda path: plot_cross_regime_heatmap(inputs, path, args.dpi)),
        ("fig_capital_frontier.png", lambda path: plot_capital_frontier(scorecard, path, args.dpi)),
        ("fig_cvar_violin.png", lambda path: plot_cvar_violin(inputs.diagnostics, scorecard, path, args.dpi)),
        ("fig_ig_vs_cvar.png", lambda path: plot_ig_vs_cvar(inputs.diagnostics, path, args.dpi)),
    ]

    successes = 0
    for filename, fn in generators:
        out_path = assets_dir / filename
        try:
            if fn(out_path):
                successes += 1
        except Exception as exc:  # pragma: no cover - keep pipeline running
            LOGGER.exception("Failed to generate %s: %s", filename, exc)
    if successes == 0:
        LOGGER.warning("No figures generated; verify input CSVs")
        return 0
    LOGGER.info("Generated %s figures in %s", successes, assets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
