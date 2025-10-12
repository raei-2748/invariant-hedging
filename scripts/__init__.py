"""Collection of command-line entrypoints and helpers."""

from . import plot_alignment_curves
from . import plot_capital_efficiency_frontier
from . import plot_invariance_vs_ig
from . import plot_ire_scatter_3d
from . import plot_regime_panels

__all__ = [
    "plot_alignment_curves",
    "plot_capital_efficiency_frontier",
    "plot_invariance_vs_ig",
    "plot_ire_scatter_3d",
    "plot_regime_panels",
]
