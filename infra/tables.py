"""CSV table readers with schema validation for figure generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _read_with_aliases(
    path: Path,
    required: dict[str, Iterable[str]],
    table_name: str,
    optional: dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """Load a CSV and normalise column names using alias definitions."""

    if not path.exists():
        raise FileNotFoundError(f"{table_name} not found at {path}")

    frame = pd.read_csv(path)

    rename_map: dict[str, str] = {}
    missing: list[str] = []

    def _resolve_alias(target: str, aliases: Iterable[str], *, required: bool) -> None:
        if target in frame.columns:
            return
        for alias in aliases:
            if alias in frame.columns:
                rename_map[alias] = target
                return
        if required:
            missing.append(target)

    for canonical, aliases in required.items():
        _resolve_alias(canonical, aliases, required=True)

    if optional:
        for canonical, aliases in optional.items():
            _resolve_alias(canonical, aliases, required=False)

    if missing:
        raise ValueError(
            f"{table_name} at {path} is missing required columns: {', '.join(sorted(missing))}"
        )

    if rename_map:
        frame = frame.rename(columns=rename_map)

    return frame


def read_invariance_diagnostics(path: Path) -> pd.DataFrame:
    required = {
        "seed": ("seed",),
        "regime_name": ("regime_name", "regime"),
        "split": ("split",),
        "ISI": ("ISI", "isi"),
        "IG": ("IG", "ig"),
        "IG_norm": ("IG_norm", "ig_norm", "IGnorm"),
        "C1": ("C1", "c1"),
        "C2": ("C2", "c2"),
        "C3": ("C3", "c3"),
    }
    return _read_with_aliases(path, required, "invariance diagnostics")


def read_capital_efficiency_frontier(path: Path) -> pd.DataFrame:
    required = {
        "model": ("model", "method"),
        "seed": ("seed",),
        "regime_name": ("regime_name", "regime"),
        "mean_pnl": ("mean_pnl", "meanpnl", "meanpnl_crisis"),
        "cvar95": ("cvar95", "CVaR95", "cvar_95", "CVaR_95", "es95", "es95_crisis"),
        "ER": ("ER", "expected_return", "efficiency", "ER_crisis"),
        "TR": ("TR", "turnover", "turnover_crisis"),
    }
    return _read_with_aliases(path, required, "capital efficiency frontier")


def read_diagnostics_summary(path: Path) -> pd.DataFrame:
    required = {
        "model": ("model", "method"),
        "seed": ("seed",),
        "regime_name": ("regime_name", "regime"),
        "split": ("split",),
        "ISI": ("ISI", "isi"),
        "IG": ("IG", "ig"),
        "IG_norm": ("IG_norm", "ig_norm", "IGnorm"),
        "CVaR95": ("CVaR95", "cvar95", "CVaR_95", "cvar_95", "es95", "es95_crisis"),
        "mean_pnl": ("mean_pnl", "meanpnl", "meanpnl_crisis"),
        "TR": ("TR", "turnover", "turnover_crisis"),
        "ER": ("ER", "expected_return", "efficiency", "ER_crisis"),
    }
    optional = {
        "WG": ("WG", "wg"),
        "WG_norm": ("WG_norm", "wg_norm"),
    }
    return _read_with_aliases(path, required, "diagnostics summary", optional=optional)


def read_alignment_head(path: Path) -> pd.DataFrame:
    required = {
        "step": ("step",),
        "pair": ("pair", "pair_id", "pair_index"),
        "penalty_value": ("penalty_value", "penalty", "lambda"),
        "cosine_alignment": ("cosine_alignment", "alignment", "cosine"),
    }
    optional = {
        "epoch": ("epoch",),
        "avg_risk": ("avg_risk", "average_risk"),
        "seed": ("seed",),
    }
    return _read_with_aliases(
        path,
        required,
        "alignment head diagnostics",
        optional=optional,
    )


def maybe_filter_seeds(frame: pd.DataFrame, seeds: Iterable[int] | None) -> pd.DataFrame:
    if seeds is None:
        return frame
    return frame[frame["seed"].isin(list(seeds))].copy()


def maybe_filter_regimes(frame: pd.DataFrame, regimes: Iterable[str] | None) -> pd.DataFrame:
    if regimes is None:
        return frame
    return frame[frame["regime_name"].isin(list(regimes))].copy()

