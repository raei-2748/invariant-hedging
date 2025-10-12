"""I–R–E 3D projection utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregate import DEFAULT_CONFIDENCE_LEVEL


@dataclass
class IRE3DResult:
    points: pd.DataFrame
    winsor_bounds: Dict[str, Tuple[float, float]]
    minmax_bounds: Dict[str, Tuple[float, float]]
    alpha: float


def _winsorize(series: pd.Series, lower_pct: float, upper_pct: float) -> Tuple[pd.Series, Tuple[float, float]]:
    if series.empty:
        return series.copy(), (float("nan"), float("nan"))
    lower = float(np.percentile(series, lower_pct))
    upper = float(np.percentile(series, upper_pct))
    clipped = series.clip(lower, upper)
    return clipped, (lower, upper)


def _minmax_scale(series: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    if series.empty:
        return series.copy(), (float("nan"), float("nan"))
    min_val = float(series.min())
    max_val = float(series.max())
    if math.isclose(min_val, max_val):
        return pd.Series(np.zeros(len(series))), (min_val, max_val)
    scaled = (series - min_val) / (max_val - min_val)
    return scaled, (min_val, max_val)


def _t_critical(df: int, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
    if df <= 0:
        return 0.0
    alpha = 1 - confidence
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(1 - alpha / 2, df))
    except Exception:  # pragma: no cover
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(1 - alpha / 2))


def _spearman(series_a: pd.Series, series_b: pd.Series) -> float:
    if len(series_a) < 2 or len(series_b) < 2:
        return 0.0
    rank_a = series_a.rank(method="average")
    rank_b = series_b.rank(method="average")
    return float(rank_a.corr(rank_b))


def build_ire_coordinates(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
) -> IRE3DResult:
    report_cfg = config.get("report", {})
    ire_cfg = report_cfg.get("ire3d", {})
    winsor_pct = ire_cfg.get("winsor_pct", [2.5, 97.5])
    lower_pct, upper_pct = float(winsor_pct[0]), float(winsor_pct[1])
    regimes = report_cfg.get("regimes_order", sorted(raw["regime"].unique()))

    index_cols = [col for col in ("algo", "seed", "regime") if col in raw.columns]
    pivot = raw.pivot_table(index=index_cols, columns="metric", values="value", aggfunc="mean").reset_index()
    if pivot.empty:
        raise RuntimeError("Cannot build IRE coordinates without data")

    axis_i = ire_cfg.get("axis_I")
    axis_r = ire_cfg.get("axis_R_source")
    eff_metric = report_cfg.get("metrics", {}).get("efficiency", ["ER_mean_pnl"])[0]
    turnover_metric = report_cfg.get("metrics", {}).get("efficiency", [None])[-1]
    if turnover_metric is None:
        turnover_metric = "TR_turnover"

    for required in [axis_i, axis_r, eff_metric, turnover_metric]:
        if required not in pivot.columns:
            raise KeyError(f"Metric '{required}' required for IRE projection not found")

    winsor_bounds: Dict[str, Tuple[float, float]] = {}
    minmax_bounds: Dict[str, Tuple[float, float]] = {}

    i_series, bounds = _winsorize(pivot[axis_i], lower_pct, upper_pct)
    winsor_bounds[axis_i] = bounds
    i_scaled, mm_bounds = _minmax_scale(i_series)
    minmax_bounds[axis_i] = mm_bounds
    i_star = 1 - i_scaled.clip(0, 1)

    r_series, bounds = _winsorize(pivot[axis_r], lower_pct, upper_pct)
    winsor_bounds[axis_r] = bounds
    r_scaled, mm_bounds = _minmax_scale(r_series)
    minmax_bounds[axis_r] = mm_bounds
    r_star = 1 - r_scaled.clip(0, 1)

    er_series = pivot[eff_metric]
    tr_series = pivot[turnover_metric]
    if ire_cfg.get("E_alpha_mode", "sd_equalize") == "sd_equalize":
        std_er = float(er_series.std(ddof=1) or 1.0)
        std_tr = float(tr_series.std(ddof=1) or 1.0)
        alpha = std_er / std_tr if std_tr else 1.0
    else:
        alpha = float(ire_cfg.get("alpha", 1.0))
    e_raw = er_series - alpha * tr_series
    e_series, bounds = _winsorize(e_raw, lower_pct, upper_pct)
    winsor_bounds["E_composite"] = bounds
    e_scaled, mm_bounds = _minmax_scale(e_series)
    minmax_bounds["E_composite"] = mm_bounds
    e_star = e_scaled.clip(0, 1)

    points = pivot.copy()
    points["I_star"] = i_star
    points["R_star"] = r_star
    points["E_star"] = e_star
    points["alpha"] = alpha

    for regime in regimes:
        subset = points[points["regime"] == regime]
        if subset.empty:
            continue
        rho = _spearman(subset["R_star"], subset[axis_r])
        if rho >= 0:
            raise AssertionError(
                f"Spearman correlation between R* and raw {axis_r} must be negative, got {rho:.3f}"
            )
    return IRE3DResult(points=points, winsor_bounds=winsor_bounds, minmax_bounds=minmax_bounds, alpha=alpha)


def _plot_projection(points: pd.DataFrame, projection: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    if projection == "top":
        x, y = "R_star", "I_star"
    elif projection == "front":
        x, y = "R_star", "E_star"
    else:
        x, y = "I_star", "E_star"
    for regime, group in points.groupby("regime"):
        ax.scatter(group[x], group[y], alpha=0.6, label=regime)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    pdf_path = output_dir / f"ire_3d_{projection}.pdf"
    png_path = output_dir / f"ire_3d_{projection}.png"
    fig.savefig(pdf_path, dpi=200)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def write_ire_assets(points: pd.DataFrame, config: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("plotly is required for 3D IRE visualisations") from exc

    color_map = {
        regime: color
        for regime, color in zip(points["regime"].unique(), ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"])
    }
    fig = go.Figure()
    for regime, group in points.groupby("regime"):
        color = color_map.get(regime, "#4e79a7")
        fig.add_trace(
            go.Scatter3d(
                x=group["I_star"],
                y=group["R_star"],
                z=group["E_star"],
                mode="markers",
                name=str(regime),
                marker=dict(size=5, opacity=0.55, color=color),
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="I*",
            yaxis_title="R*",
            zaxis_title="E*",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    html_path = output_dir / "interactive" / "ire_3d.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))

    fig_path = output_dir / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)
    for projection in config.get("report", {}).get("ire3d", {}).get("projections", ["top", "front", "side"]):
        _plot_projection(points, projection, fig_path)


__all__ = [
    "IRE3DResult",
    "build_ire_coordinates",
    "write_ire_assets",
]
