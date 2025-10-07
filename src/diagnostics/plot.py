"""Generate diagnostic plots from aggregated CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ig_vs_cvar(df: pd.DataFrame, out: str = "figures/ig_vs_cvar.png") -> None:
    if df.empty or "IG" not in df.columns or "cvar95" not in df.columns:
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax = df.plot.scatter(x="IG", y="cvar95")
    ax.set_title("IG vs Crisis CVaR")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def capital_efficiency(df: pd.DataFrame, out: str = "figures/capital_efficiency.png") -> None:
    if df.empty or "cvar95" not in df.columns or "mean" not in df.columns:
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax = df.plot.scatter(x="cvar95", y="mean")
    ax.set_title("Capital Efficiency Frontier")
    ax.set_xlabel("CVaR-95 (lower is better)")
    ax.set_ylabel("Mean PnL (higher is better)")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to diagnostics CSV")
    parser.add_argument("--outdir", default="figures", help="Directory for generated plots")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    ig_vs_cvar(df, str(Path(args.outdir) / "ig_vs_cvar.png"))
    capital_efficiency(df, str(Path(args.outdir) / "capital_efficiency.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
