"""Plot alignment trajectories and penalty schedules."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from infra.plot_io import append_manifest, apply_style, ensure_out_dir, parse_formats, save_figure
from infra.tables import maybe_filter_seeds, read_alignment_head

LOGGER = logging.getLogger(__name__)


def _parse_seed_filter(value: str | None) -> list[int] | None:
    if value is None or value.lower() == "all":
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def create_figure(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
    dpi: int = 300,
    formats: Iterable[str] = ("png", "pdf"),
    style: str = "journal",
    seed_filter: Iterable[int] | None = None,
) -> bool:
    tables_dir = run_dir / "tables"
    csv_path = tables_dir / "alignment_head.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing alignment diagnostics at %s; skipping alignment curves", csv_path)
        return False

    try:
        frame = read_alignment_head(csv_path)
    except ValueError as exc:
        LOGGER.warning(
            "Invalid alignment diagnostics schema at %s: %s; skipping alignment curves",
            csv_path,
            exc,
        )
        return False
    if "seed" in frame.columns:
        frame = maybe_filter_seeds(frame, seed_filter)
    elif seed_filter is not None:
        LOGGER.warning("Seed filter requested but alignment table has no seed column")
    if frame.empty:
        LOGGER.warning("No rows available for alignment curves after filtering; skipping figure")
        return False

    apply_style(style)
    target_dir = ensure_out_dir(run_dir, out_dir)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax2 = ax1.twinx()

    cmap = plt.get_cmap("tab10")
    pairs = sorted(frame["pair"].unique())
    frame = frame.sort_values("step")

    for idx, pair in enumerate(pairs):
        subset = frame[frame["pair"] == pair]
        color = cmap(idx % cmap.N)
        ax1.plot(
            subset["step"],
            subset["cosine_alignment"],
            label=f"Pair {pair}",
            color=color,
            linewidth=1.6,
        )

    penalty_curve = frame.groupby("step")["penalty_value"].mean().reset_index()
    ax2.plot(
        penalty_curve["step"],
        penalty_curve["penalty_value"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        label="Penalty",
    )

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cosine alignment")
    ax2.set_ylabel("Penalty value")
    ax1.set_title("Alignment Trajectories")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=2)
    fig.tight_layout()

    saved_paths = save_figure(
        fig,
        target_dir,
        "fig_alignment_curves",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)

    append_manifest(
        target_dir,
        {
            "name": "fig_alignment_curves",
            "files": [path.name for path in saved_paths],
            "sources": ["tables/alignment_head.csv"],
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    formats = parse_formats(args.format)
    seeds = _parse_seed_filter(args.seed_filter)

    create_figure(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        formats=formats,
        style=args.style,
        seed_filter=seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

