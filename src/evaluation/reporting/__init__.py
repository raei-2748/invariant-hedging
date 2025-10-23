"""Compatibility wrappers for legacy reporting modules."""

from legacy.report_core import aggregate, latex, lite, paper_provenance, plots, provenance, schema

__all__ = [
    "aggregate",
    "latex",
    "lite",
    "paper_provenance",
    "plots",
    "provenance",
    "schema",
]
