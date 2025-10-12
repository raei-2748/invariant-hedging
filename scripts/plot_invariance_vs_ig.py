"""CLI for invariance vs influence-gap scatter figure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from infra.plot_io import append_manifest, apply_style, ensure_out_dir, parse_formats, save_figure
from infra.tables import (
    maybe_filter_regimes,
    maybe_filter_seeds,
    read_invariance_diagnostics,
)

LOGGER = logging.getLogger(__name__)


MARKERS = {
    "train": "o",
    "val": "s",
    "validation": "s",
    "test": "^",
}


def _parse_seed_filter(value: str | None) -> list[int] | None:
    if value is None or value.lower() == "all":
        return None
    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    return seeds or None


def _parse_regime_filter(value: str | None) -> list[str] | None:
    if value is None or value.lower() == "all":
        return None
    regimes = [token.strip() for token in value.split(",") if token.strip()]
    return regimes or None


def create_figure(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
    dpi: int = 300,
    formats: Iterable[str] = ("png", "pdf"),
    style: str = "journal",
    seed_filter: Iterable[int] | None = None,
    regime_filter: Iterable[str] | None = None,
) -> bool:
    """Generate the invariance vs IG scatter figure."""

    tables_dir = run_dir / "tables"
    csv_path = tables_dir / "invariance_diagnostics.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing invariance diagnostics at %s; skipping figure", csv_path)
        return False

    frame = read_invariance_diagnostics(csv_path)
    frame = maybe_filter_seeds(frame, seed_filter)
    frame = maybe_filter_regimes(frame, regime_filter)
    if frame.empty:
        LOGGER.warning("No rows available for invariance vs IG after filtering; skipping figure")
        return False

    apply_style(style)
    target_dir = ensure_out_dir(run_dir, out_dir)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    splits = sorted(frame["split"].unique())
    for split in splits:
        subset = frame[frame["split"] == split]
        marker = MARKERS.get(split, "o")
        ax.scatter(
            subset["ISI"],
            subset["IG"],
            label=split.capitalize(),
            marker=marker,
            alpha=0.85,
            edgecolor="black",
            linewidths=0.4,
        )

    x_values = frame["ISI"].to_numpy()
    y_values = frame["IG"].to_numpy()
    if len(x_values) >= 3:
        regression = stats.linregress(x_values, y_values)
        x_sorted = np.linspace(x_values.min(), x_values.max(), 200)
        y_pred = regression.intercept + regression.slope * x_sorted
        residuals = y_values - (regression.intercept + regression.slope * x_values)
        if len(x_values) > 2:
            dof = len(x_values) - 2
            sigma2 = np.sum(residuals**2) / dof
            mean_x = np.mean(x_values)
            se_line = np.sqrt(
                sigma2
                * (
                    1 / len(x_values)
                    + (x_sorted - mean_x) ** 2 / np.sum((x_values - mean_x) ** 2)
                )
            )
            interval = stats.t.ppf(0.975, dof) * se_line
            ax.fill_between(x_sorted, y_pred - interval, y_pred + interval, color="#c6dbee", alpha=0.5)
        ax.plot(x_sorted, y_pred, color="#1f77b4", linestyle="--", linewidth=1.5, label="Fit")
        ax.text(
            0.02,
            0.95,
            f"Pearson r = {regression.rvalue:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

    ax.set_xlabel("ISI (invariance score)")
    ax.set_ylabel("IG (influence gap)")
    ax.set_title("Invariance vs Influence Gap")
    ax.legend()

    saved_paths = save_figure(
        fig,
        target_dir,
        "fig_invariance_vs_ig",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)

    append_manifest(
        target_dir,
        {
            "name": "fig_invariance_vs_ig",
            "files": [path.name for path in saved_paths],
            "sources": ["tables/invariance_diagnostics.csv"],
        },
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Run directory containing tables")
    parser.add_argument("--out_dir", type=Path, default=None, help="Optional output directory")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", default="png,pdf")
    parser.add_argument("--style", choices=("journal", "poster"), default="journal")
    parser.add_argument("--seed_filter", default="all")
    parser.add_argument("--regime_filter", default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    formats = parse_formats(args.format)
    seeds = _parse_seed_filter(args.seed_filter)
    regimes = _parse_regime_filter(args.regime_filter)

    created = create_figure(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        formats=formats,
        style=args.style,
        seed_filter=seeds,
        regime_filter=regimes,
    )
    if not created:
        LOGGER.warning("Figure 'fig_invariance_vs_ig' was not generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

