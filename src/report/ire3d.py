"""I–R–E 3D projection helpers."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclasses.dataclass
class IRE3DResult:
    points: pd.DataFrame
    metadata: Mapping[str, object]


def _winsorize(series: pd.Series, low_pct: float, high_pct: float) -> tuple[pd.Series, Dict[str, float]]:
    lower = float(np.percentile(series, low_pct))
    upper = float(np.percentile(series, high_pct))
    clipped = series.clip(lower, upper)
    return clipped, {"lower": lower, "upper": upper}


def _minmax(series: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
    min_v = float(series.min())
    max_v = float(series.max())
    if math.isclose(max_v, min_v):
        return pd.Series(np.zeros_like(series) + 0.5, index=series.index), {"min": min_v, "max": max_v}
    normalized = (series - min_v) / (max_v - min_v)
    return normalized, {"min": min_v, "max": max_v}


def _confidence_interval(series: pd.Series, level: float = 0.95) -> tuple[float, float]:
    n = len(series)
    if n == 0:
        return (math.nan, math.nan)
    mean = float(series.mean())
    if n == 1:
        return (mean, mean)
    std = float(series.std(ddof=1))
    stderr = std / math.sqrt(n)
    alpha = 1 - level
    prob = 1 - alpha / 2
    from statistics import NormalDist

    quantile = NormalDist().inv_cdf(prob)
    half = quantile * stderr
    return (mean - half, mean + half)


def _spearman_rho(x: pd.Series, y: pd.Series) -> float:
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def _compute_alpha(er: pd.Series, tr: pd.Series, mode: str, config: Mapping[str, object]) -> float:
    if mode == "cost_model":
        alpha = float(config.get("alpha", 1.0))
        return alpha
    std_er = float(er.std(ddof=1))
    std_tr = float(tr.std(ddof=1))
    if std_tr == 0:
        return 1.0
    return std_er / std_tr if std_tr else 1.0


def _compute_points(seed_frame: pd.DataFrame, config: Mapping[str, object], regimes_order: Sequence[str]) -> tuple[pd.DataFrame, Mapping[str, object]]:
    axis_I = str(config.get("axis_I", "IG"))
    axis_R = str(config.get("axis_R_source", "CVaR_95"))
    winsor_pct = config.get("winsor_pct", (2.5, 97.5))
    low_pct, high_pct = float(winsor_pct[0]), float(winsor_pct[1])

    er_metric = str(config.get("er_metric", "ER_mean_pnl"))
    tr_metric = str(config.get("tr_metric", "TR_turnover"))

    pivot = seed_frame.pivot_table(index=["seed", "regime"], columns="metric", values="value")
    required_metrics = [axis_I, axis_R, er_metric, tr_metric]
    for metric in required_metrics:
        if metric not in pivot.columns:
            raise KeyError(f"Metric '{metric}' required for I–R–E projection is missing from diagnostics")

    I_raw = pivot[axis_I].astype(float)
    R_raw = pivot[axis_R].astype(float)
    ER_raw = pivot[er_metric].astype(float)
    TR_raw = pivot[tr_metric].astype(float)

    I_wins, I_bounds = _winsorize(I_raw, low_pct, high_pct)
    R_wins, R_bounds = _winsorize(R_raw, low_pct, high_pct)

    I_norm, I_minmax = _minmax(I_wins)
    R_norm, R_minmax = _minmax(R_wins)

    I_star = 1 - I_norm
    R_star = 1 - R_norm

    alpha_mode = str(config.get("E_alpha_mode", "sd_equalize"))
    alpha = _compute_alpha(ER_raw, TR_raw, alpha_mode, config)
    E_value = ER_raw - alpha * TR_raw
    E_wins, E_bounds = _winsorize(E_value, low_pct, high_pct)
    E_norm, E_minmax = _minmax(E_wins)
    E_star = E_norm

    result = pivot.copy()
    result["I_star"] = I_star
    result["R_star"] = R_star
    result["E_star"] = E_star
    result["alpha"] = alpha

    result = result.reset_index()

    rho: Dict[str, float] = {}
    for regime, group in result.groupby("regime"):
        rho_val = _spearman_rho(group["R_star"], group[axis_R])
        if rho_val >= 0:
            raise ValueError(f"Spearman rho check failed for regime '{regime}': {rho_val:.3f} >= 0")
        rho[regime] = rho_val

    metadata = {
        "winsor_pct": [low_pct, high_pct],
        "axis_I": axis_I,
        "axis_R_source": axis_R,
        "er_metric": er_metric,
        "tr_metric": tr_metric,
        "alpha": alpha,
        "winsor_bounds": {axis_I: I_bounds, axis_R: R_bounds, "efficiency": E_bounds},
        "minmax": {axis_I: I_minmax, axis_R: R_minmax, "efficiency": E_minmax},
        "spearman_rho": rho,
    }

    return result, metadata


def _plot_interactive(points: pd.DataFrame, regimes_order: Sequence[str], output_path: Path) -> None:
    colors = plt_colors(len(regimes_order))
    fig = go.Figure()
    color_map = dict(zip(regimes_order, colors))

    for regime in regimes_order:
        subset = points[points["regime"] == regime]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=subset["I_star"],
                y=subset["R_star"],
                z=subset["E_star"],
                mode="markers",
                marker=dict(size=4, color=color_map.get(regime, "grey"), opacity=0.5),
                name=regime,
                hovertext=subset["seed"],
            )
        )

    centroids = points.groupby("regime")[['I_star', 'R_star', 'E_star']].mean()
    cis = {}
    for regime in regimes_order:
        subset = points[points["regime"] == regime]
        if subset.empty:
            continue
        cis[regime] = {
            axis: _confidence_interval(subset[axis])[0:2] for axis in ["I_star", "R_star", "E_star"]
        }
        mean_point = centroids.loc[regime]
        fig.add_trace(
            go.Scatter3d(
                x=[mean_point["I_star"]],
                y=[mean_point["R_star"]],
                z=[mean_point["E_star"]],
                mode="markers",
                marker=dict(size=8, color=color_map.get(regime, "grey"), opacity=1.0, symbol="diamond"),
                name=f"{regime} mean",
            )
        )
        ci = cis[regime]
        for axis, axis_idx in zip(["I_star", "R_star", "E_star"], [0, 1, 2]):
            lo, hi = ci[axis]
            coords = {
                "x": [mean_point["I_star"], mean_point["I_star"]],
                "y": [mean_point["R_star"], mean_point["R_star"]],
                "z": [mean_point["E_star"], mean_point["E_star"]],
            }
            coords[["x", "y", "z"][axis_idx]] = [lo, hi]
            fig.add_trace(
                go.Scatter3d(
                    x=coords["x"],
                    y=coords["y"],
                    z=coords["z"],
                    mode="lines",
                    line=dict(color=color_map.get(regime, "grey"), width=3),
                    showlegend=False,
                )
            )

    fig.update_layout(scene=dict(xaxis_title="I*", yaxis_title="R*", zaxis_title="E*"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def plt_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("tab10")
    return ["rgba({},{},{},{})".format(*(int(c * 255) for c in cmap(i)[:3]), 0.8) for i in np.linspace(0, 1, n)]


def _plot_projection(points: pd.DataFrame, projection: str, regimes_order: Sequence[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    axes = {
        "top": ("I_star", "R_star"),
        "front": ("I_star", "E_star"),
        "side": ("R_star", "E_star"),
    }
    if projection not in axes:
        return
    x_axis, y_axis = axes[projection]
    fig, ax = plt.subplots(figsize=(4, 4))
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("tab10")
    colors = [cmap(x) for x in np.linspace(0, 1, len(regimes_order))]
    color_map = dict(zip(regimes_order, colors))
    for regime in regimes_order:
        subset = points[points["regime"] == regime]
        if subset.empty:
            continue
        ax.scatter(subset[x_axis], subset[y_axis], alpha=0.4, color=color_map.get(regime), label=regime)
        centroid = subset[[x_axis, y_axis]].mean()
        ax.scatter(centroid[x_axis], centroid[y_axis], color=color_map.get(regime), marker="D", s=50)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"I–R–E projection ({projection})")
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    seed_frame: pd.DataFrame,
    *,
    config: Mapping[str, object],
    regimes_order: Sequence[str],
    output_dirs: Mapping[str, Path],
) -> IRE3DResult:
    points, metadata = _compute_points(seed_frame, config, regimes_order)
    tables_dir = output_dirs.get("tables", Path("tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(tables_dir / "ire_points.csv", index=False)

    projections = config.get("projections", ["top", "front", "side"])
    figures_dir = output_dirs.get("figures", Path("figures"))
    for projection in projections:
        _plot_projection(points, str(projection), regimes_order, figures_dir / f"ire_3d_{projection}")

    interactive_path = output_dirs.get("interactive", Path("interactive")) / "ire_3d.html"
    _plot_interactive(points, regimes_order, interactive_path)

    metadata = dict(metadata)
    metadata["interactive"] = str(interactive_path)
    return IRE3DResult(points=points, metadata=metadata)
