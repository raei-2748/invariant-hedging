"""CLI for capital-efficiency frontier figure."""

from __future__ import annotations

import argparse
import logging
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

from infra.plot_io import append_manifest, apply_style, ensure_out_dir, parse_formats, save_figure
from infra.tables import (
    maybe_filter_regimes,
    maybe_filter_seeds,
    read_capital_efficiency_frontier,
)

LOGGER = logging.getLogger(__name__)


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
    csv_path = tables_dir / "capital_efficiency_frontier.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing capital-efficiency frontier at %s; skipping figure", csv_path)
        return False

    try:
        frame = read_capital_efficiency_frontier(csv_path)
    except ValueError as exc:
        LOGGER.warning(
            "Invalid capital-efficiency frontier schema at %s: %s; skipping figure",
            csv_path,
            exc,
        )
        return False
    frame = maybe_filter_seeds(frame, seed_filter)
    frame = maybe_filter_regimes(frame, regime_filter)
    if frame.empty:
        LOGGER.warning("No rows available for capital-efficiency frontier after filtering; skipping figure")
        return False

    apply_style(style)
    target_dir = ensure_out_dir(run_dir, out_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    color_map = plt.get_cmap("tab10")
    models = sorted(frame["model"].unique())
    for idx, model in enumerate(models):
        group = frame[frame["model"] == model].sort_values("cvar95")
        color = color_map(idx % color_map.N)
        ax.plot(
            group["cvar95"],
            group["mean_pnl"],
            marker="o",
            linewidth=1.6,
            markersize=5,
            label=model,
            color=color,
        )
        if not group.empty:
            terminal = group.iloc[-1]
            ax.annotate(
                f"ER={terminal['ER']:.2f}",
                xy=(terminal["cvar95"], terminal["mean_pnl"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
            ax.scatter(
                group["cvar95"],
                group["mean_pnl"],
                s=35,
                color=color,
                edgecolors="black",
                linewidths=0.4,
            )

    ax.set_xlabel("CVaR-95 (loss)")
    ax.set_ylabel("Mean PnL")
    ax.set_title("Capital-Efficiency Frontier")
    ax.legend(title="Model", loc="best")

    saved_paths = save_figure(
        fig,
        target_dir,
        "fig_capital_efficiency_frontier",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)

    append_manifest(
        target_dir,
        {
            "name": "fig_capital_efficiency_frontier",
            "files": [path.name for path in saved_paths],
            "sources": ["tables/capital_efficiency_frontier.csv"],
        },
    )
    return True


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

    created = create_figure(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        formats=formats,
        style=style,
        seed_filter=seeds,
        regime_filter=regimes,
    )
    if not created:
        LOGGER.warning("Figure 'fig_capital_efficiency_frontier' was not generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

