import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NUMERIC_COLUMNS = ["IG", "cvar95", "mean"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared.dropna(subset=["cvar95", "mean"])


def ig_vs_cvar(df: pd.DataFrame, out: str = "figures/ig_vs_cvar.png") -> None:
    df.plot.scatter(x="IG", y="cvar95")
    plt.title("IG vs Crisis CVaR")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def capital_efficiency(df: pd.DataFrame, out: str = "figures/capital_efficiency.png") -> None:
    df.plot.scatter(x="cvar95", y="mean")
    plt.title("Capital Efficiency Frontier")
    plt.xlabel("CVaR-95 (lower is better)")
    plt.ylabel("Mean PnL (higher is better)")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostics summary figures")
    parser.add_argument("--csv", required=True, help="Path to diagnostics CSV file")
    parser.add_argument(
        "--outdir",
        default="figures",
        help="Directory to store generated figures (default: figures)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = _prepare(df)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        capital_efficiency(df, out=str(output_dir / "capital_efficiency.png"))
        ig_vs_cvar(df, out=str(output_dir / "ig_vs_cvar.png"))


if __name__ == "__main__":
    main()
