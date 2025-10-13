"""Hydra-integrated loader for the real SPY dataset used in the paper runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from .preprocess import (
    calibrate_option_surface,
    clean_equity_history,
    emit_split_artifacts,
    export_metadata,
    load_cboe_series,
    load_optionmetrics,
    stage_surface_cache,
)
from .real.spy_emini import SpyEminiDataModule, WindowConfig
from .spy_loader import SplitConfig, load_split_config


@dataclass
class RawSourcesConfig:
    """Paths to the raw OptionMetrics/CBOE/Yahoo inputs."""

    spy_ohlcv: str = "data/raw/spy_ohlcv.csv"
    optionmetrics: str = "data/raw/optionmetrics"
    cboe: str = "data/raw/cboe"


@dataclass
class SplitGroupConfig:
    """Split names mapped to YAML definitions under ``configs/splits``."""

    directory: str = "configs/splits"
    train: List[str] = field(default_factory=lambda: ["spy_train"])
    val: List[str] = field(default_factory=lambda: ["spy_val"])
    test: List[str] = field(
        default_factory=lambda: ["spy_test_2018", "spy_test_2020", "spy_test_2022"]
    )
    extra: List[str] = field(default_factory=list)


@dataclass
class RealSpyLoaderConfig:
    """Configuration marshalled via Hydra."""

    name: str = "real_spy_paper"
    cache_dir: str = "data/cache/real_spy"
    raw: RawSourcesConfig = field(default_factory=RawSourcesConfig)
    splits: SplitGroupConfig = field(default_factory=SplitGroupConfig)
    rate: float = 0.01
    base_linear_bps: float = 5.0
    base_quadratic: float = 0.0
    base_slippage_multiplier: float = 1.0
    require_fresh_cache: bool = False
    prefer_parquet: bool = True


cs = ConfigStore.instance()
try:
    cs.store(group="data", name="real_spy_paper", node=RealSpyLoaderConfig)
except ValueError:
    # Config already registered by another import site; safe to ignore.
    pass


class RealSpyDataModule:
    """Wrapper around :class:`SpyEminiDataModule` with caching + split resolution."""

    def __init__(self, config: Dict | RealSpyLoaderConfig):
        self.config = self._build_config(config)
        self.cache_dir = Path(to_absolute_path(self.config.cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.surface_path = self.cache_dir / "spy_surface.csv"
        self.split_cache_dir = self.cache_dir / "splits"
        self.raw_paths = RawSourcesConfig(
            spy_ohlcv=str(Path(to_absolute_path(self.config.raw.spy_ohlcv))),
            optionmetrics=str(Path(to_absolute_path(self.config.raw.optionmetrics))),
            cboe=str(Path(to_absolute_path(self.config.raw.cboe))),
        )
        self.split_dir = Path(to_absolute_path(self.config.splits.directory))
        self.split_stems: Dict[str, List[str]] = {
            "train": list(self.config.splits.train),
            "val": list(self.config.splits.val),
            "test": list(self.config.splits.test),
            "extra": list(self.config.splits.extra),
        }
        self._stem_to_config = self._load_split_configs()
        self.split_groups = self._resolve_split_groups()
        self._split_configs = {cfg.name: cfg for cfg in self._stem_to_config.values()}
        self._env_order = self._build_env_order()
        self._data_module: Optional[SpyEminiDataModule] = None
        self._ensure_cache()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(config: Dict | RealSpyLoaderConfig) -> RealSpyLoaderConfig:
        if isinstance(config, RealSpyLoaderConfig):
            return config
        raw_defaults = RawSourcesConfig()
        raw_cfg = config.get("raw", {})
        splits_defaults = SplitGroupConfig()
        splits_cfg = config.get("splits", {})
        return RealSpyLoaderConfig(
            name=config.get("name", "real_spy_paper"),
            cache_dir=config.get("cache_dir", "data/cache/real_spy"),
            raw=RawSourcesConfig(
                spy_ohlcv=raw_cfg.get("spy_ohlcv", raw_defaults.spy_ohlcv),
                optionmetrics=raw_cfg.get("optionmetrics", raw_defaults.optionmetrics),
                cboe=raw_cfg.get("cboe", raw_defaults.cboe),
            ),
            splits=SplitGroupConfig(
                directory=splits_cfg.get("directory", splits_defaults.directory),
                train=list(splits_cfg.get("train", splits_defaults.train)),
                val=list(splits_cfg.get("val", splits_defaults.val)),
                test=list(splits_cfg.get("test", splits_defaults.test)),
                extra=list(splits_cfg.get("extra", splits_defaults.extra)),
            ),
            rate=float(config.get("rate", 0.01)),
            base_linear_bps=float(config.get("base_linear_bps", 5.0)),
            base_quadratic=float(config.get("base_quadratic", 0.0)),
            base_slippage_multiplier=float(
                config.get("base_slippage_multiplier", 1.0)
            ),
            require_fresh_cache=bool(config.get("require_fresh_cache", False)),
            prefer_parquet=bool(config.get("prefer_parquet", True)),
        )

    def _build_env_order(self) -> List[str]:
        order: List[str] = []
        for split in ("train", "val", "test", "extra"):
            for name in self.split_groups.get(split, []):
                if name not in order:
                    order.append(name)
        return order

    def _load_split_configs(self) -> Dict[str, SplitConfig]:
        configs: Dict[str, SplitConfig] = {}
        stems: List[str] = []
        for values in self.split_stems.values():
            stems.extend(values)
        for stem in sorted(set(stems)):
            split_yaml = self.split_dir / f"{stem}.yaml"
            configs[stem] = load_split_config(split_yaml)
        return configs

    def _resolve_split_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for split, stems in self.split_stems.items():
            names: List[str] = []
            for stem in stems:
                cfg = self._stem_to_config.get(stem)
                if cfg is None:
                    raise KeyError(f"Split YAML '{stem}.yaml' not found in {self.split_dir}")
                names.append(cfg.name)
            groups[split] = names
        return groups

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_available(self) -> bool:
        if not self.surface_path.exists():
            return False
        for name in self._env_order:
            split_csv = self.split_cache_dir / f"{name}.csv"
            if not split_csv.exists():
                return False
        return True

    def _ensure_cache(self) -> None:
        if self.config.require_fresh_cache or not self._cache_available():
            equity = clean_equity_history(Path(self.raw_paths.spy_ohlcv))
            optionmetrics = load_optionmetrics(Path(self.raw_paths.optionmetrics))
            cboe = load_cboe_series(Path(self.raw_paths.cboe))
            surface = calibrate_option_surface(equity, optionmetrics, cboe)
            stage_surface_cache(surface, self.cache_dir, prefer_parquet=self.config.prefer_parquet)
            split_outputs = emit_split_artifacts(
                surface,
                self._split_configs,
                self.split_cache_dir,
                prefer_parquet=self.config.prefer_parquet,
            )
            export_metadata(self.cache_dir, split_outputs)
        else:
            # nothing to do – existing caches are reused
            pass

    # ------------------------------------------------------------------
    # Public API mirroring ``SpyEminiDataModule``
    # ------------------------------------------------------------------

    @property
    def env_order(self) -> List[str]:
        return list(self._env_order)

    def split_envs(self, split: str) -> List[str]:
        if split not in self.split_groups:
            raise KeyError(f"Unknown split '{split}'")
        return list(self.split_groups[split])

    def _setup_module(self) -> None:
        if self._data_module is not None:
            return
        module = SpyEminiDataModule(
            {
                "spy_path": str(self.surface_path),
                "mode": "full",
                "rate": self.config.rate,
                "include_gfc": True,
                "base_linear_bps": self.config.base_linear_bps,
                "base_quadratic": self.config.base_quadratic,
                "base_slippage_multiplier": self.config.base_slippage_multiplier,
            }
        )
        windows: List[WindowConfig] = []
        for name in self._env_order:
            split_cfg = self._split_configs.get(name)
            if split_cfg is None:
                raise KeyError(f"Split '{name}' not resolved; check configuration")
            windows.append(
                WindowConfig(
                    name=name,
                    start=split_cfg.start_date.date().isoformat(),
                    end=split_cfg.end_date.date().isoformat(),
                    linear_bps=self.config.base_linear_bps,
                    quadratic=self.config.base_quadratic,
                    slippage_multiplier=self.config.base_slippage_multiplier,
                )
            )
        module._windows = windows
        module._env_order = self.env_order
        self._data_module = module

    def prepare(self, split: str, env_names: Optional[Iterable[str]] = None):
        if split not in self.split_groups and split != "extra":
            raise KeyError(f"Unsupported split '{split}'")
        self._setup_module()
        allowed = set(self.split_groups.get(split, [])) | set(self.split_groups.get("extra", []))
        names = list(env_names) if env_names is not None else list(self.split_groups.get(split, []))
        missing = [name for name in names if name not in allowed]
        if missing:
            raise KeyError(f"Requested environments {missing} are not registered for split '{split}'")
        return self._data_module.prepare(split, names)

    def hourly_dataset(self, env_name: str):
        self._setup_module()
        return self._data_module.hourly_dataset(env_name)
