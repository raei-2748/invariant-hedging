"""Generate the complete suite of paper figures for a run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.visualization._cli import (
    bootstrap_cli_environment,
    env_override,
    parse_regime_filter,
    parse_seed_filter,
)

bootstrap_cli_environment()

from src.core.infra.plot_io import ensure_out_dir, parse_formats
from src.visualization import (
    plot_alignment_curves,
    plot_capital_efficiency_frontier,
    plot_invariance_vs_ig,
    plot_ire_scatter_3d,
    plot_regime_panels,
)

LOGGER = logging.getLogger(__name__)

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

    run_dir = args.run_dir
    out_dir = ensure_out_dir(run_dir, args.out_dir)

    LOGGER.info("Generating figures into %s", out_dir)

    def _safe_generate(label: str, func, **kwargs) -> None:
        try:
            created = func(**kwargs)
            if not created:
                LOGGER.info("Figure '%s' skipped", label)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to generate '%s': %s", label, exc)

    _safe_generate(
        "fig_invariance_vs_ig",
        plot_invariance_vs_ig.create_figure,
        run_dir=run_dir,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
    )

    _safe_generate(
        "fig_capital_efficiency_frontier",
        plot_capital_efficiency_frontier.create_figure,
        run_dir=run_dir,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
    )

    _safe_generate(
        "fig_ire_scatter_3d",
        plot_ire_scatter_3d.create_figure,
        run_dir=run_dir,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
        eff_axis=args.eff_axis,
        composite_alpha=args.composite_alpha,
        separate_by_regime=args.separate_by_regime,
    )

    _safe_generate(
        "fig_regime_panels",
        plot_regime_panels.create_figure,
        run_dir=run_dir,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
    )

    _safe_generate(
        "fig_alignment_curves",
        plot_alignment_curves.create_figure,
        run_dir=run_dir,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
    )

    LOGGER.info("Figure generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

