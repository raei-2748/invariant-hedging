"""CSV table readers with schema validation for figure generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _read_with_schema(path: Path, required: Iterable[str], table_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{table_name} not found at {path}")

    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} at {path} is missing required columns: {', '.join(missing)}"
        )
    return frame


def read_invariance_diagnostics(path: Path) -> pd.DataFrame:
    required = ("seed", "regime_name", "split", "ISI", "IG", "IG_norm", "C1", "C2", "C3")
    return _read_with_schema(path, required, "invariance diagnostics")


def read_capital_efficiency_frontier(path: Path) -> pd.DataFrame:
    required = ("model", "seed", "regime_name", "mean_pnl", "cvar95", "ER", "TR")
    return _read_with_schema(path, required, "capital efficiency frontier")


def read_diagnostics_summary(path: Path) -> pd.DataFrame:
    required = (
        "model",
        "seed",
        "regime_name",
        "split",
        "ISI",
        "IG",
        "IG_norm",
        "CVaR95",
        "mean_pnl",
        "TR",
        "ER",
    )
    return _read_with_schema(path, required, "diagnostics summary")


def read_alignment_head(path: Path) -> pd.DataFrame:
    required = (
        "epoch",
        "step",
        "pair",
        "penalty_value",
        "avg_risk",
        "cosine_alignment",
    )
    return _read_with_schema(path, required, "alignment head diagnostics")


def maybe_filter_seeds(frame: pd.DataFrame, seeds: Iterable[int] | None) -> pd.DataFrame:
    if seeds is None:
        return frame
    return frame[frame["seed"].isin(list(seeds))].copy()


def maybe_filter_regimes(frame: pd.DataFrame, regimes: Iterable[str] | None) -> pd.DataFrame:
    if regimes is None:
        return frame
    return frame[frame["regime_name"].isin(list(regimes))].copy()

