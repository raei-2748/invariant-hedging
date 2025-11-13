"""Visualisation toolkit for the invariant hedging paper."""

from . import (
    plot_alignment_curves,
    plot_capital_efficiency_frontier,
    plot_capital_frontier,
    plot_cvar_by_method,
    plot_cvar_violin,
    plot_diag_correlations,
    plot_ig_vs_cvar,
    plot_invariance_vs_ig,
    plot_ire_scatter_3d,
    plot_method_schematic,
    plot_qq_tails,
    plot_regime_panels,
    plot_sweeps,
)
from .plot_cvar_curves import plot_cvar_violin as violin_chart

__all__ = [
    "plot_alignment_curves",
    "plot_capital_efficiency_frontier",
    "plot_capital_frontier",
    "plot_cvar_by_method",
    "plot_cvar_violin",
    "plot_diag_correlations",
    "plot_ig_vs_cvar",
    "plot_invariance_vs_ig",
    "plot_ire_scatter_3d",
    "plot_method_schematic",
    "plot_qq_tails",
    "plot_regime_panels",
    "plot_sweeps",
    "violin_chart",
]
