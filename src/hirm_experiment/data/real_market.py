from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .black_scholes import greeks
from .generator import SimulationResult, TransactionCostSpec


def _rolling_realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 2)
    vol = log_returns.rolling(window, min_periods=2).std(ddof=0)
    return vol * np.sqrt(252.0)


def _rolling_skew(log_returns: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 3)

    def _skew(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        if valid.size < 3:
            return float("nan")
        centered = valid - valid.mean()
        variance = np.mean(centered ** 2)
        if variance <= 1e-12:
            return 0.0
        return float(np.mean(centered ** 3) / (variance ** 1.5))

    return log_returns.rolling(window, min_periods=3).apply(_skew, raw=True)


def _sanitize_name(name: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


class RealMarketLoader:
    def __init__(self, config: Dict[str, Any], episode_length: int) -> None:
        self.config = config
        self.episode_length = int(episode_length)
        self.options_path = Path(config["spy_options_path"])
        self.futures_path = Path(config["es_futures_path"])
        self.target_dte = int(config.get("target_dte", 60))
        self.start_date = pd.Timestamp(config.get("start_date", "2016-01-01")).tz_localize(None)
        self.end_date = pd.Timestamp(config.get("end_date", "2023-12-31")).tz_localize(None)
        self.interpolation_method = config.get("interpolation_method", "time")
        filters = config.get("filters", {})
        self.filters = {
            "min_open_interest": int(filters.get("min_open_interest", 500)),
            "min_volume": int(filters.get("min_volume", 50)),
            "max_rel_spread": float(filters.get("max_rel_spread", 0.05)),
            "max_abs_spread": float(filters.get("max_abs_spread", 0.15)),
        }
        self.realized_vol_window = int(config.get("realized_vol_window", 20))
        self.realized_skew_window = int(config.get("realized_skew_window", 60))
        self.options_columns = self._resolve_option_columns(config.get("options_columns", {}))
        self.futures_columns = self._resolve_futures_columns(config.get("futures_columns", {}))
        self.tx_costs = config.get("tx_costs", {})
        self._cache: Optional[pd.DataFrame] = None

    def _resolve_option_columns(self, overrides: Dict[str, str]) -> Dict[str, str]:
        defaults = {
            "date": "quote_date",
            "expiration": "expiration",
            "strike": "strike",
            "bid": "bid",
            "ask": "ask",
            "volume": "volume",
            "open_interest": "open_interest",
            "underlying": "underlying_price",
            "implied_vol": "implied_volatility",
        }
        defaults.update({k: v for k, v in overrides.items() if v})
        return defaults

    def _resolve_futures_columns(self, overrides: Dict[str, str]) -> Dict[str, str]:
        defaults = {
            "date": "date",
            "close": "settle",
        }
        defaults.update({k: v for k, v in overrides.items() if v})
        return defaults

    def combined_frame(self) -> pd.DataFrame:
        if self._cache is None:
            futures = self._load_futures()
            options = self._load_options(futures.index)
            combined = self._merge_frames(options, futures)
            mask = (combined.index >= self.start_date) & (combined.index <= self.end_date)
            self._cache = combined.loc[mask].copy()
        return self._cache.copy()

    def build_window(self, window: Dict[str, Any], tag: str) -> SimulationResult:
        frame = self.combined_frame()
        start = pd.Timestamp(window["start"]).tz_localize(None)
        end = pd.Timestamp(window["end"]).tz_localize(None)
        window_frame = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        if window_frame.empty:
            name = window.get("name", f"{start.date()}_{end.date()}")
            raise ValueError(f"Window '{name}' returned no observations in the real data loader.")
        if window_frame.shape[0] <= self.episode_length:
            name = window.get("name", f"{start.date()}_{end.date()}")
            raise ValueError(
                f"Window '{name}' with {window_frame.shape[0]} rows is shorter than the episode length {self.episode_length}."
            )
        arrays = self._frame_to_arrays(window_frame)
        env_name = self._resolve_env_name(window, tag)
        tx_cost = self._resolve_cost(window, tag)
        metadata = self._window_metadata(window, tag, window_frame, tx_cost, env_name)
        metadata["env"] = env_name
        metadata["episode_count"] = int(arrays["spot"].shape[0])
        return SimulationResult(
            env=env_name,
            spot=arrays["spot"],
            delta=arrays["delta"],
            gamma=arrays["gamma"],
            theta=arrays["theta"],
            realized_vol=arrays["realized_vol"],
            implied_vol=arrays["implied_vol"],
            option_price=arrays["option_price"],
            tx_cost=tx_cost,
            metadata=metadata,
        )

    def _read_table(self, path: Path, parse_dates: Iterable[str]) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        for column in parse_dates:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])
        return df

    def _load_futures(self) -> pd.DataFrame:
        date_col = self.futures_columns["date"]
        close_col = self.futures_columns["close"]
        df = self._read_table(self.futures_path, [date_col])
        if date_col not in df.columns or close_col not in df.columns:
            raise ValueError("Futures data must contain date and close columns for preprocessing.")
        df = df[[date_col, close_col]].copy()
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=[close_col])
        df = df.sort_values(date_col).drop_duplicates(date_col, keep="last")
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df[(df[date_col] >= self.start_date) & (df[date_col] <= self.end_date)]
        df = df.set_index(date_col)
        business_index = pd.date_range(self.start_date, self.end_date, freq="B")
        df = df.reindex(business_index)
        df[close_col] = df[close_col].interpolate(method=self.interpolation_method, limit_direction="both")
        log_returns = np.log(df[close_col]).diff()
        df["realized_vol"] = _rolling_realized_vol(log_returns, self.realized_vol_window)
        df["realized_skew"] = _rolling_skew(log_returns, self.realized_skew_window)
        df[["realized_vol", "realized_skew"]] = df[["realized_vol", "realized_skew"]].interpolate(
            method=self.interpolation_method, limit_direction="both"
        )
        df = df.dropna(subset=[close_col])
        df.rename(columns={close_col: "futures_close"}, inplace=True)
        return df[["futures_close", "realized_vol", "realized_skew"]]

    def _load_options(self, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        cols = self.options_columns
        parse_cols = [cols["date"], cols["expiration"]]
        df = self._read_table(self.options_path, parse_cols)
        rename_map = {
            cols["date"]: "date",
            cols["expiration"]: "expiration",
            cols["strike"]: "strike",
            cols["bid"]: "bid",
            cols["ask"]: "ask",
            cols["volume"]: "volume",
            cols["open_interest"]: "open_interest",
            cols["underlying"]: "underlying",
        }
        implied_col = cols.get("implied_vol")
        if implied_col:
            rename_map[implied_col] = "implied_vol"
        df = df.rename(columns=rename_map)
        required = ["date", "expiration", "strike", "bid", "ask", "volume", "open_interest", "underlying"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Options data is missing required columns: {missing}")
        numeric_cols = ["strike", "bid", "ask", "volume", "open_interest", "underlying"]
        for column in numeric_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if "implied_vol" in df.columns:
            df["implied_vol"] = pd.to_numeric(df["implied_vol"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.tz_localize(None)
        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df = df.dropna(subset=["mid", "underlying", "strike"])
        df = df[df["open_interest"] >= self.filters["min_open_interest"]]
        df = df[df["volume"] >= self.filters["min_volume"]]
        df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid"] > 0)]
        df["spread"] = df["ask"] - df["bid"]
        spread_cap = np.maximum(self.filters["max_abs_spread"], self.filters["max_rel_spread"] * df["mid"])
        df = df[df["spread"] <= spread_cap]
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        df = df[df["dte"] > 0]
        df["atm_distance"] = (df["strike"] - df["underlying"]).abs() / df["underlying"].replace(0, np.nan)
        df["dte_distance"] = (df["dte"] - self.target_dte).abs()
        df = df.sort_values(["date", "dte_distance", "atm_distance", "volume"], ascending=[True, True, True, False])
        atm = df.groupby("date").head(1).copy()
        atm = atm.set_index("date").reindex(target_index)
        atm["underlying"] = atm["underlying"].interpolate(method=self.interpolation_method, limit_direction="both")
        atm["mid"] = atm["mid"].interpolate(method=self.interpolation_method, limit_direction="both")
        atm["dte"] = atm["dte"].fillna(self.target_dte)
        if "implied_vol" in atm.columns:
            atm["implied_vol"] = atm["implied_vol"].replace(0.0, np.nan)
            atm["implied_vol"] = atm["implied_vol"].interpolate(method=self.interpolation_method, limit_direction="both")
            atm["implied_vol"] = atm["implied_vol"].ffill().bfill()
        else:
            atm["implied_vol"] = np.nan
        if atm["implied_vol"].isna().all():
            atm["implied_vol"] = 0.2
        atm["implied_vol"] = atm["implied_vol"].fillna(atm["implied_vol"].mean()).fillna(0.2)
        atm["tau"] = atm["dte"].astype(float) / 252.0
        atm["tau"] = atm["tau"].clip(lower=1e-6)
        atm["spot"] = atm["underlying"]
        spot = atm["spot"].to_numpy(dtype=np.float64)
        tau = atm["tau"].to_numpy(dtype=np.float64)
        sigma = np.clip(atm["implied_vol"].to_numpy(dtype=np.float64), 1e-4, None)
        delta, gamma, theta = greeks(spot, tau, sigma)
        atm["delta"] = delta.astype(np.float32)
        atm["gamma"] = gamma.astype(np.float32)
        atm["theta"] = theta.astype(np.float32)
        atm["implied_vol"] = sigma.astype(np.float32)
        atm["option_price"] = atm["mid"].astype(np.float32)
        return atm

    def _merge_frames(self, options: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
        combined = options.join(futures, how="inner")
        combined = combined.interpolate(method=self.interpolation_method, limit_direction="both")
        required = ["spot", "option_price", "delta", "gamma", "theta", "implied_vol", "realized_vol"]
        combined = combined.dropna(subset=required)
        combined = combined.sort_index()
        return combined

    def _resolve_env_name(self, window: Dict[str, Any], tag: str) -> str:
        base_name = window.get("name")
        if base_name:
            sanitized = _sanitize_name(str(base_name))
        else:
            start = pd.Timestamp(window["start"]).strftime("%Y%m%d")
            end = pd.Timestamp(window["end"]).strftime("%Y%m%d")
            sanitized = f"{start}_{end}"
        return f"real_{tag}_{sanitized}"

    def _resolve_cost(self, window: Dict[str, Any], tag: str) -> TransactionCostSpec:
        profile = window.get("cost_profile")
        if not profile:
            profile = "crisis" if tag == "crisis" else "default"
        default_cost = self.tx_costs.get("default", {"linear": 0.001, "quadratic": 0.05})
        profile_cost = self.tx_costs.get(profile, default_cost)
        linear = float(window.get("tx_linear", profile_cost.get("linear", default_cost.get("linear", 0.001))))
        quadratic = float(window.get("tx_quadratic", profile_cost.get("quadratic", default_cost.get("quadratic", 0.05))))
        return TransactionCostSpec(linear=linear, quadratic=quadratic)

    def _window_metadata(
        self,
        window: Dict[str, Any],
        tag: str,
        frame: pd.DataFrame,
        tx_cost: TransactionCostSpec,
        env_name: str,
    ) -> Dict[str, Any]:
        start = pd.Timestamp(window["start"]).tz_localize(None)
        end = pd.Timestamp(window["end"]).tz_localize(None)
        metadata = {
            "source": "real",
            "window_name": window.get("name", env_name),
            "window_type": tag,
            "window_start": start.strftime("%Y-%m-%d"),
            "window_end": end.strftime("%Y-%m-%d"),
            "options_path": str(self.options_path),
            "futures_path": str(self.futures_path),
            "target_dte": self.target_dte,
            "episode_length": self.episode_length,
            "realized_vol_window": self.realized_vol_window,
            "realized_skew_window": self.realized_skew_window,
            "episode_sampling": {"method": "sliding_window", "horizon": self.episode_length},
            "tx_cost": {"linear": tx_cost.linear, "quadratic": tx_cost.quadratic},
            "total_days": int(frame.shape[0]),
        }
        if tag == "crisis" or window.get("crisis", False):
            metadata["crisis"] = True
        if "realized_vol" in frame.columns:
            metadata["avg_realized_vol"] = float(frame["realized_vol"].mean())
        if "realized_skew" in frame.columns:
            metadata["avg_realized_skew"] = float(frame["realized_skew"].mean())
        return metadata

    def _frame_to_arrays(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        required = ["spot", "delta", "gamma", "theta", "realized_vol", "implied_vol", "option_price"]
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Real market frame is missing required columns: {missing_columns}")
        for column in required:
            if frame[column].isna().any():
                frame[column] = frame[column].interpolate(method=self.interpolation_method, limit_direction="both")
                frame[column] = frame[column].ffill().bfill()
        horizon = self.episode_length
        spot_series = frame["spot"].to_numpy(dtype=np.float32)
        delta_series = frame["delta"].to_numpy(dtype=np.float32)
        gamma_series = frame["gamma"].to_numpy(dtype=np.float32)
        theta_series = frame["theta"].to_numpy(dtype=np.float32)
        realized_vol_series = frame["realized_vol"].to_numpy(dtype=np.float32)
        implied_vol_series = frame["implied_vol"].to_numpy(dtype=np.float32)
        option_price_series = frame["option_price"].to_numpy(dtype=np.float32)
        spot_windows = sliding_window_view(spot_series, horizon + 1).copy()
        delta_windows = sliding_window_view(delta_series, horizon).copy()
        gamma_windows = sliding_window_view(gamma_series, horizon).copy()
        theta_windows = sliding_window_view(theta_series, horizon).copy()
        realized_vol_windows = sliding_window_view(realized_vol_series, horizon).copy()
        implied_vol_windows = sliding_window_view(implied_vol_series, horizon).copy()
        option_price_windows = sliding_window_view(option_price_series, horizon).copy()
        return {
            "spot": spot_windows,
            "delta": delta_windows,
            "gamma": gamma_windows,
            "theta": theta_windows,
            "realized_vol": realized_vol_windows,
            "implied_vol": implied_vol_windows,
            "option_price": option_price_windows,
        }
