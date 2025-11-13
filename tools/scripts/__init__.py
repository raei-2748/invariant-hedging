"""Collection of command-line entrypoints and helpers."""

from invariant_hedging.visualization import (
    plot_alignment_curves,
    plot_capital_efficiency_frontier,
    plot_invariance_vs_ig,
    plot_ire_scatter_3d,
    plot_regime_panels,
)

__all__ = [
    "plot_alignment_curves",
    "plot_capital_efficiency_frontier",
    "plot_invariance_vs_ig",
    "plot_ire_scatter_3d",
    "plot_regime_panels",
]
