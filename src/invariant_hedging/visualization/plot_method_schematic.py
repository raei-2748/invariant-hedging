#!/usr/bin/env python3
"""Create a simple schematic comparing ERM, IRM, and HIRM."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    return parser.parse_args(argv)


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _add_box(ax, xy, text, color):
    box = FancyBboxPatch(
        xy,
        width=2.5,
        height=1.2,
        boxstyle="round,pad=0.2",
        linewidth=2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 1.25,
        xy[1] + 0.6,
        text,
        ha="center",
        va="center",
        fontsize=12,
        color=color,
        fontweight="bold",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axis_off()

    colors = {
        "ERM": "#1f77b4",
        "IRM": "#ff7f0e",
        "HIRM": "#2ca02c",
    }

    _add_box(ax, (0.5, 1.8), "ERM\nSingle environment loss", colors["ERM"])
    _add_box(ax, (3.2, 1.8), "IRM\nInvariance penalty", colors["IRM"])
    _add_box(ax, (5.9, 1.8), "HIRM\nHead-only adaptation", colors["HIRM"])

    arrowprops = dict(arrowstyle="-|>", color="#555555", linewidth=1.6, mutation_scale=15)
    ax.annotate("", xy=(3.0, 2.4), xytext=(2.8, 2.4), arrowprops=arrowprops)
    ax.annotate("", xy=(5.7, 2.4), xytext=(5.5, 2.4), arrowprops=arrowprops)
    ax.text(3.4, 2.25, "Enforce invariance", ha="center", fontsize=10)
    ax.text(6.5, 2.25, "Freeze backbone\nadapt head", ha="center", fontsize=10)

    fig.tight_layout()
    out_path = Path(args.out)
    _ensure_outdir(out_path)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
