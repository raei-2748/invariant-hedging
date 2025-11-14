"""Diagnostics shared between training and evaluation flows."""

from .invariance import compute_ER, compute_TR
from . import metrics

__all__ = ["compute_ER", "compute_TR", "metrics"]
