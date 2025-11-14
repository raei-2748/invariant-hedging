"""CLI for regime panel bar charts."""

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

from invariant_hedging.reporting.plot_io import append_manifest, apply_style, ensure_out_dir, parse_formats, save_figure
from invariant_hedging.reporting.tables import (
    maybe_filter_regimes,
    maybe_filter_seeds,
    read_diagnostics_summary,
)

LOGGER = logging.getLogger(__name__)


METRIC_COLUMNS = {
    "CVaR-95 (loss)": "CVaR95",
    "IG": "IG",
    "ER": "ER",
    "TR": "TR",
}


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
    tables_dir = run_dir / "tables"
    csv_path = tables_dir / "diagnostics_summary.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing diagnostics summary at %s; skipping regime panels", csv_path)
        return False

    try:
        frame = read_diagnostics_summary(csv_path)
    except ValueError as exc:
        LOGGER.warning(
            "Invalid diagnostics summary schema at %s: %s; skipping regime panels",
            csv_path,
            exc,
        )
        return False
    frame = frame[frame["split"].str.lower() == "test"].copy()
    frame = maybe_filter_seeds(frame, seed_filter)
    frame = maybe_filter_regimes(frame, regime_filter)
    if frame.empty:
        LOGGER.warning("No rows available for regime panels after filtering; skipping figure")
        return False

    apply_style(style)
    target_dir = ensure_out_dir(run_dir, out_dir)

    regimes = sorted(frame["regime_name"].unique())
    n_regimes = len(regimes)
    cols = min(3, n_regimes)
    rows = math.ceil(n_regimes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.flatten()

    model_palette = _model_palette(frame["model"].unique())

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        subset = frame[frame["regime_name"] == regime]
        models = list(subset["model"].unique())
        positions = np.arange(len(models))
        width = 0.18

        for offset, (label, column) in enumerate(METRIC_COLUMNS.items()):
            if column not in subset.columns:
                continue
            values = [subset[subset["model"] == model][column].mean() for model in models]
            bar_positions = positions + (offset - 1.5) * width
            ax.bar(
                bar_positions,
                values,
                width=width,
                label=label,
                color=[model_palette[model] for model in models]
                if label == "ER"
                else None,
                alpha=0.85,
            )

        ax.set_title(regime)
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle="-")

    for ax in axes[n_regimes:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(METRIC_COLUMNS))
    fig.suptitle("Regime Diagnostics (test split)", y=1.02)
    fig.tight_layout()

    saved_paths = save_figure(
        fig,
        target_dir,
        "fig_regime_panels",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)

    append_manifest(
        target_dir,
        {
            "name": "fig_regime_panels",
            "files": [path.name for path in saved_paths],
            "sources": ["tables/diagnostics_summary.csv"],
        },
    )
    return True


def _model_palette(models: Iterable[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    mapping: dict[str, str] = {}
    for idx, model in enumerate(sorted(models)):
        mapping[model] = cmap(idx % cmap.N)
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

