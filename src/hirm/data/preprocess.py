"""Pre-processing utilities for the real SPY dataset.

The paper reproduction pipeline downloads raw SPY equity quotes (Yahoo Finance),
licensed OptionMetrics option surfaces, and public CBOE indices. The helpers
below clean those raw inputs, stitch them into a consolidated surface, and write
cached artifacts that the training and evaluation loaders can reuse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

from .spy_loader import SplitConfig


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

REQUIRED_OHLCV_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Adj Close"}
REQUIRED_OPTIONMETRICS_COLUMNS = {"trade_date", "mid", "implied_vol"}


def clean_equity_history(csv_path: Path) -> pd.DataFrame:
    """Load and sanitise the SPY OHLCV history.

    Parameters
    ----------
    csv_path: Path
        Location of a Yahoo-style historical CSV.

    Returns
    -------
    pd.DataFrame
        Sorted dataframe with a normalised ``date`` column and ``spot`` values.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"SPY OHLCV history missing: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_OHLCV_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Equity CSV is missing required columns: {missing}".format(
                missing=", ".join(sorted(missing))
            )
        )
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Encountered unparsable dates in equity history.")
    df["date"] = df["Date"].dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date")
    df["spot"] = df["Adj Close"].astype(float)
    return df[["date", "spot"]]


def load_optionmetrics(source: Path) -> pd.DataFrame:
    """Read OptionMetrics-style option quotes.

    Parameters
    ----------
    source: Path
        Either a CSV file or a directory containing CSV files exported from the
        IvyDB US dataset. Only a tiny subset of columns is required for the
        calibration routines below.
    """

    if not source.exists():
        raise FileNotFoundError(f"OptionMetrics source not found: {source}")

    paths: List[Path]
    if source.is_dir():
        paths = sorted(p for p in source.glob("*.csv") if p.is_file())
    else:
        paths = [source]
    if not paths:
        raise FileNotFoundError(f"No OptionMetrics CSV files discovered in {source}")

    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_OPTIONMETRICS_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "OptionMetrics CSV missing columns: {cols}".format(
                cols=", ".join(sorted(missing))
            )
        )

    data = data.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    if data["trade_date"].isna().any():
        raise ValueError("OptionMetrics quotes contain invalid trade_date entries")
    data["trade_date"] = data["trade_date"].dt.tz_localize(None).dt.normalize()

    if "mid" not in data:
        if {"bid", "ask"}.issubset(data.columns):
            data["mid"] = (data["bid"] + data["ask"]) / 2.0
        else:
            raise ValueError("OptionMetrics data must contain either 'mid' or bid/ask quotes")
    data["mid"] = data["mid"].astype(float)
    data["implied_vol"] = data["implied_vol"].astype(float)
    high_vol_mask = data["implied_vol"] > 1.5
    data.loc[high_vol_mask, "implied_vol"] = data.loc[high_vol_mask, "implied_vol"] / 100.0
    return data


def load_cboe_series(source: Path) -> pd.DataFrame:
    """Load auxiliary CBOE public series (e.g. VIX closes)."""

    if not source.exists():
        raise FileNotFoundError(f"CBOE source not found: {source}")
    if source.is_dir():
        candidates = sorted(p for p in source.glob("*.csv") if p.is_file())
        if not candidates:
            raise FileNotFoundError(f"No CSV files located under {source}")
        path = candidates[0]
    else:
        path = source
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CBOE CSV {path} contains no rows")

    normalised = {col.lower().replace(" ", "_"): col for col in df.columns}
    date_col = normalised.get("date")
    vix_close_col = normalised.get("vix_close") or normalised.get("close")
    if date_col is None or vix_close_col is None:
        raise ValueError(
            "CBOE CSV must contain a date column and a VIX close column;"
            " expected headers like 'DATE' and 'VIX Close'."
        )

    df = df.copy()
    df.rename(columns={date_col: "date", vix_close_col: "vix_close"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Invalid dates in CBOE series")
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df["vix_close"] = df["vix_close"].astype(float)
    df = df.sort_values("date").drop_duplicates("date")
    return df[["date", "vix_close"]]


# ---------------------------------------------------------------------------
# Calibration and caching
# ---------------------------------------------------------------------------


def calibrate_option_surface(
    equity: pd.DataFrame,
    optionmetrics: pd.DataFrame,
    cboe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine the cleaned data into a surface expected by the loaders.

    The calibration step is intentionally lightweight – for the purposes of unit
    tests and smoke runs we aggregate the OptionMetrics quotes into a single
    synthetic at-the-money option per trade date. Real experiments can replace
    this with richer calibration without touching the loaders.
    """

    if equity.empty:
        raise ValueError("Equity dataframe is empty")
    if optionmetrics.empty:
        raise ValueError("OptionMetrics dataframe is empty")

    optionmetrics = optionmetrics.copy()
    optionmetrics["mid_price"] = optionmetrics["mid"]
    optionmetrics["implied_vol"] = optionmetrics["implied_vol"].clip(lower=1e-6)
    grouped = (
        optionmetrics.groupby("trade_date")[["mid_price", "implied_vol"]]
        .mean()
        .reset_index()
        .rename(columns={"trade_date": "date"})
    )

    if cboe is not None and not cboe.empty:
        cboe = cboe.copy()
        cboe["implied_vol_proxy"] = cboe["vix_close"] / 100.0
        grouped = grouped.merge(cboe[["date", "implied_vol_proxy"]], on="date", how="outer")
        grouped = grouped.sort_values("date")
        grouped["implied_vol"] = grouped["implied_vol"].fillna(grouped["implied_vol_proxy"])
        grouped = grouped.drop(columns=["implied_vol_proxy"], errors="ignore")

    surface = equity.merge(grouped, on="date", how="inner")
    surface = surface.dropna(subset=["spot", "mid_price", "implied_vol"])
    surface = surface.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    surface["mid_price"] = surface["mid_price"].astype(float)
    surface["implied_vol"] = surface["implied_vol"].astype(float)
    surface["spot"] = surface["spot"].astype(float)
    return surface[["date", "spot", "mid_price", "implied_vol"]]


@dataclass
class SurfaceCache:
    """Pointers to cached CSV/Parquet outputs."""

    surface_csv: Path
    surface_parquet: Optional[Path]
    split_csv: Dict[str, Path]
    split_parquet: Dict[str, Path]


def _write_parquet(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except ImportError:
        return None
    return path


def stage_surface_cache(
    surface: pd.DataFrame,
    cache_dir: Path,
    *,
    prefer_parquet: bool = True,
) -> SurfaceCache:
    """Write consolidated surface + per-split caches.

    Parameters
    ----------
    surface: pd.DataFrame
        Aggregated time series with columns ``date``, ``spot``, ``mid_price``,
        and ``implied_vol``.
    cache_dir: Path
        Directory where caches should be materialised.
    prefer_parquet: bool
        Attempt to write parquet outputs when optional dependencies are
        available. CSV artefacts are always written for portability.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    surface_csv = cache_dir / "spy_surface.csv"
    surface.to_csv(surface_csv, index=False)
    surface_parquet: Optional[Path] = None
    if prefer_parquet:
        surface_parquet = _write_parquet(surface, cache_dir / "spy_surface.parquet")
    return SurfaceCache(
        surface_csv=surface_csv,
        surface_parquet=surface_parquet,
        split_csv={},
        split_parquet={},
    )


def emit_split_artifacts(
    surface: pd.DataFrame,
    split_configs: Mapping[str, SplitConfig],
    cache_dir: Path,
    *,
    prefer_parquet: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Materialise per-split CSV/Parquet datasets.

    Returns a mapping ``{split_name: {"csv": Path, "parquet": Optional[Path]}}``.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    surface = surface.copy()
    surface["date"] = pd.to_datetime(surface["date"])

    outputs: Dict[str, Dict[str, Path]] = {}
    for name, split_cfg in split_configs.items():
        mask = (surface["date"] >= split_cfg.start_date) & (surface["date"] <= split_cfg.end_date)
        sliced = surface.loc[mask].copy()
        if sliced.empty:
            raise ValueError(
                f"Split '{name}' produced an empty dataset. Check raw data coverage and YAML bounds."
            )
        sliced_csv = cache_dir / f"{name}.csv"
        sliced.to_csv(sliced_csv, index=False)
        sliced_parquet: Optional[Path] = None
        if prefer_parquet:
            sliced_parquet = _write_parquet(sliced, cache_dir / f"{name}.parquet")
        outputs[name] = {"csv": sliced_csv}
        if sliced_parquet is not None:
            outputs[name]["parquet"] = sliced_parquet
    return outputs


def export_metadata(cache_dir: Path, split_outputs: Mapping[str, Mapping[str, Path]]) -> Path:
    """Write a tiny manifest describing the cached artefacts."""

    manifest = {
        "splits": {
            name: {
                kind: str(Path(path).resolve().relative_to(cache_dir.resolve()))
                for kind, path in outputs.items()
            }
            for name, outputs in split_outputs.items()
        }
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path
