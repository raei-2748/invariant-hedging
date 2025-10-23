"""Diagnostic helpers aligned with the paper's robustness analysis."""
from __future__ import annotations

from legacy.diagnostics.efficiency import compute_ER, compute_TR
from legacy.diagnostics.external import compute_IG, compute_VR, compute_WG
from legacy.diagnostics.helpers import detach_diagnostics, safe_eval_metric
from legacy.diagnostics.isi import (
    ISINormalizationConfig,
    compute_C1_global_stability,
    compute_C2_mechanistic_stability,
    compute_C3_structural_stability,
    compute_ISI,
)
from legacy.diagnostics.metrics import invariant_gap, mechanistic_sensitivity, worst_group

__all__ = [
    "ISINormalizationConfig",
    "compute_C1_global_stability",
    "compute_C2_mechanistic_stability",
    "compute_C3_structural_stability",
    "compute_ER",
    "compute_IG",
    "compute_ISI",
    "compute_TR",
    "compute_VR",
    "compute_WG",
    "detach_diagnostics",
    "invariant_gap",
    "mechanistic_sensitivity",
    "safe_eval_metric",
    "worst_group",
]
