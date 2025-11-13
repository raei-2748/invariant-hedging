"""CLI for the invariance-robustness-efficiency scatter figure."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover
    from ._cli import (
        bootstrap_cli_environment,
        env_override,
        parse_regime_filter,
        parse_seed_filter,
    )
except ImportError:  # pragma: no cover
    from _cli import (  # type: ignore
        bootstrap_cli_environment,
        env_override,
        parse_regime_filter,
        parse_seed_filter,
    )

bootstrap_cli_environment()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects

from invariant_hedging.reporting.infra.plot_io import append_manifest, apply_style, ensure_out_dir, parse_formats, save_figure
from invariant_hedging.reporting.infra.tables import (
    maybe_filter_regimes,
    maybe_filter_seeds,
    read_diagnostics_summary,
)

LOGGER = logging.getLogger(__name__)


def _resolve_robustness(frame: pd.DataFrame) -> np.ndarray:
    column_names = list(frame.columns)
    if "WG" in column_names:
        return frame["WG"].to_numpy()
    if "IG_norm" in column_names:
        values = frame["IG_norm"].replace(0.0, np.nan)
        return 1.0 / values.to_numpy()
    raise ValueError("diagnostics summary does not contain WG or IG_norm for robustness axis")


def _efficiency_axis(frame: pd.DataFrame, mode: str, composite_alpha: float) -> np.ndarray:
    if mode == "ER":
        if "ER" not in frame.columns:
            raise ValueError("diagnostics summary missing ER column")
        return frame["ER"].to_numpy()
    if mode == "composite":
        if "mean_pnl" not in frame.columns or "TR" not in frame.columns:
            raise ValueError("diagnostics summary missing mean_pnl/TR for composite efficiency")
        return (frame["mean_pnl"] - composite_alpha * frame["TR"]).to_numpy()
    raise ValueError(f"Unknown efficiency axis mode '{mode}'")


def _size_from_turnover(frame: pd.DataFrame) -> np.ndarray:
    if "TR" not in frame.columns:
        return np.full(len(frame), 40.0)
    tr = frame["TR"].fillna(0.0).to_numpy()
    max_tr = np.nanmax(tr) if np.any(tr) else 1.0
    scale = tr / max_tr if max_tr > 0 else np.zeros_like(tr)
    return 30.0 + 170.0 * scale


def create_figure(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
    dpi: int = 300,
    formats: Iterable[str] = ("png", "pdf"),
    style: str = "journal",
    seed_filter: Iterable[int] | None = None,
    regime_filter: Iterable[str] | None = None,
    eff_axis: str = "ER",
    composite_alpha: float = 0.5,
    separate_by_regime: bool = False,
) -> bool:
    tables_dir = run_dir / "tables"
    csv_path = tables_dir / "diagnostics_summary.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing diagnostics summary at %s; skipping I-R-E figure", csv_path)
        return False

    try:
        frame = read_diagnostics_summary(csv_path)
    except ValueError as exc:
        LOGGER.warning(
            "Invalid diagnostics summary schema at %s: %s; skipping I-R-E figure",
            csv_path,
            exc,
        )
        return False
    frame = maybe_filter_seeds(frame, seed_filter)
    frame = maybe_filter_regimes(frame, regime_filter)
    if frame.empty:
        LOGGER.warning("No rows available for I-R-E scatter after filtering; skipping figure")
        return False

    apply_style(style)
    target_dir = ensure_out_dir(run_dir, out_dir)

    base_saved = _render_single(
        frame,
        target_dir,
        formats,
        dpi,
        eff_axis,
        composite_alpha,
        base_name="fig_ire_scatter_3d",
    )

    append_manifest(
        target_dir,
        {
            "name": "fig_ire_scatter_3d",
            "files": [path.name for path in base_saved],
            "sources": ["tables/diagnostics_summary.csv"],
        },
    )

    if separate_by_regime:
        saved = _render_by_regime(
            frame,
            target_dir,
            formats,
            dpi,
            eff_axis,
            composite_alpha,
        )
        append_manifest(
            target_dir,
            {
                "name": "fig_ire_scatter_3d_byregime",
                "files": [path.name for path in saved],
                "sources": ["tables/diagnostics_summary.csv"],
            },
        )
    return True


def _render_single(frame, target_dir, formats, dpi, eff_axis, composite_alpha, base_name: str):
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    color_map = plt.get_cmap("tab10")
    regimes = sorted(frame["regime_name"].unique())
    sizes = pd.Series(_size_from_turnover(frame), index=frame.index)

    for idx, regime in enumerate(regimes):
        subset = frame[frame["regime_name"] == regime]
        color = color_map(idx % color_map.N)
        subset_sizes = sizes.loc[subset.index].to_numpy()
        ax.scatter(
            subset["ISI"],
            _resolve_robustness(subset),
            _efficiency_axis(subset, eff_axis, composite_alpha),
            color=color,
            s=subset_sizes,
            edgecolors="black",
            linewidths=0.4,
            label=regime,
            alpha=0.85,
        )

    ax.set_xlabel("I: ISI")
    ax.set_ylabel("R: Robustness proxy")
    axis_label = "E: ER" if eff_axis == "ER" else f"E: mean_pnl - {composite_alpha} * TR"
    ax.set_zlabel(axis_label)
    ax.set_title("I-R-E Geometry")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    saved_paths = save_figure(
        fig,
        target_dir,
        base_name,
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return saved_paths


def _render_by_regime(frame, target_dir, formats, dpi, eff_axis, composite_alpha):
    regimes = sorted(frame["regime_name"].unique())
    n_regimes = len(regimes)
    cols = min(3, n_regimes)
    rows = math.ceil(n_regimes / cols)
    fig = plt.figure(figsize=(cols * 4.0, rows * 4.0))

    for idx, regime in enumerate(regimes, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        subset = frame[frame["regime_name"] == regime]
        ax.scatter(
            subset["ISI"],
            _resolve_robustness(subset),
            _efficiency_axis(subset, eff_axis, composite_alpha),
            s=_size_from_turnover(subset),
            color="#1f77b4",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_title(regime)
        ax.set_xlabel("I")
        ax.set_ylabel("R")
        ax.set_zlabel("E")

    fig.suptitle("I-R-E Geometry by Regime")
    saved_paths = save_figure(
        fig,
        target_dir,
        "fig_ire_scatter_3d_byregime",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", default="png,pdf")
    parser.add_argument("--style", choices=("journal", "poster"), default="journal")
    parser.add_argument("--seed_filter", default="all")
    parser.add_argument("--regime_filter", default="all")
    parser.add_argument("--eff_axis", choices=("ER", "composite"), default="ER")
    parser.add_argument("--composite_alpha", type=float, default=0.5)
    parser.add_argument("--separate_by_regime", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    format_spec = env_override(args.format, "FIGURE_FORMATS")
    style = env_override(args.style, "FIGURE_STYLE")
    seed_spec = env_override(args.seed_filter, "FIGURE_SEED_FILTER")
    regime_spec = env_override(args.regime_filter, "FIGURE_REGIME_FILTER")

    formats = parse_formats(format_spec)
    seeds = parse_seed_filter(seed_spec)
    regimes = parse_regime_filter(regime_spec)

    create_figure(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
        eff_axis=args.eff_axis,
        composite_alpha=args.composite_alpha,
        separate_by_regime=args.separate_by_regime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

