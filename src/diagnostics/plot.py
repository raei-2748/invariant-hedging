"""Plot utilities for Phase-2 diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("figures")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ig_vs_cvar(df: pd.DataFrame, out: Path | str = FIG_DIR / "ig_vs_cvar.png") -> None:
    path = Path(out)
    _ensure_parent(path)
    df.plot.scatter(x="IG", y="cvar95")
    plt.title("IG vs Crisis CVaR")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def capital_efficiency(df: pd.DataFrame, out: Path | str = FIG_DIR / "capital_efficiency.png") -> None:
    path = Path(out)
    _ensure_parent(path)
    df.plot.scatter(x="cvar95", y="mean")
    plt.title("Capital Efficiency Frontier")
    plt.xlabel("CVaR-95 (lower is better)")
    plt.ylabel("Mean PnL (higher is better)")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Aggregated diagnostics CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    ig_vs_cvar(df)
    capital_efficiency(df)


if __name__ == "__main__":
    main()
