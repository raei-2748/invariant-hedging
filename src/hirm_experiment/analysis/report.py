from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def coverage_table(coverage: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if not coverage:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(coverage, orient="index")
    df.index.name = "environment"
    return df


def plot_spread_sensitivity(curve: Dict[str, float], ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    spreads = sorted((float(k), v) for k, v in curve.items())
    xs = [item[0] for item in spreads]
    ys = [item[1] for item in spreads]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Allowed spread cap")
    ax.set_ylabel("CVaR-95")
    ax.set_title("Spread Sensitivity")
    return ax
