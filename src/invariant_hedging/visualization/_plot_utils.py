#!/usr/bin/env python3
"""Shared helpers for Phase 2 plotting scripts."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:  # Optional dependency used only when available.
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional in light setups.
    _scipy_stats = None

LOGGER = logging.getLogger("phase2_plots")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, raising a readable error if missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Scoreboard CSV not found: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - passthrough for pandas errors.
        raise RuntimeError(f"Failed to read CSV at {csv_path}: {exc}") from exc


def ensure_cols(df: pd.DataFrame, columns: Iterable[str], *, soft: bool = False) -> None:
    """Ensure the DataFrame contains the requested columns.

    Parameters
    ----------
    df:
        DataFrame to validate.
    columns:
        Column names that must be present.
    soft:
        When True, missing columns trigger a warning instead of an exception.
    """
    missing = [col for col in columns if col not in df.columns]
    if not missing:
        return
    if soft:
        LOGGER.warning("Missing optional columns: %s", ", ".join(missing))
        return
    raise KeyError(f"CSV missing required columns: {', '.join(missing)}")


def save_png_with_meta(fig, out_path: str | Path, meta: Mapping[str, object], *, dpi: int = 200) -> None:
    """Persist a Matplotlib figure and accompanying metadata JSON."""
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    meta_path = output_path.with_name(output_path.stem + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def filter_frame(
    df: pd.DataFrame,
    *,
    reg: str | None,
    models: Sequence[str] | None,
) -> pd.DataFrame:
    """Apply common filters for region and model selectors."""
    filtered = df.copy()
    if reg:
        if "reg" not in filtered.columns:
            raise KeyError("CSV missing required column: reg")
        filtered = filtered[filtered["reg"] == reg]
    if models:
        if "model" not in filtered.columns:
            raise KeyError("CSV missing required column: model")
        filtered = filtered[filtered["model"].isin(models)]
    return filtered


@dataclass
class SummaryStats:
    mean: float
    lower: float
    upper: float
    count: int


def compute_ci(series: pd.Series) -> SummaryStats:
    """Compute mean and two-sided 95% confidence interval for a sample."""
    cleaned = series.dropna().astype(float)
    n = int(cleaned.shape[0])
    if n == 0:
        raise ValueError("Cannot compute confidence interval on empty data")
    mean = float(cleaned.mean())
    if n == 1:
        return SummaryStats(mean=mean, lower=mean, upper=mean, count=1)
    std = float(cleaned.std(ddof=1))
    if std == 0.0:
        return SummaryStats(mean=mean, lower=mean, upper=mean, count=n)
    critical = _t_multiplier(n)
    margin = critical * std / math.sqrt(n)
    return SummaryStats(mean=mean, lower=mean - margin, upper=mean + margin, count=n)


def _t_multiplier(n_samples: int) -> float:
    df = max(n_samples - 1, 1)
    if _scipy_stats is not None:  # pragma: no cover - SciPy availability is environment-specific.
        try:
            return float(_scipy_stats.t.ppf(0.975, df))
        except Exception:
            pass
    # Fallback to normal approximation if SciPy is unavailable.
    return float(NormalDist().inv_cdf(0.975))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute Spearman's rho without relying on SciPy."""
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        raise ValueError("Cannot compute correlation on empty data")
    rx = aligned.iloc[:, 0].rank(method="average")
    ry = aligned.iloc[:, 1].rank(method="average")
    return float(rx.corr(ry))


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        raise ValueError("Cannot compute correlation on empty data")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
