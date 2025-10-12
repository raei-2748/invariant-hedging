import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data import spy_loader
from src.data.real_spy_loader import RealSpyDataModule


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-02", periods=7, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(100, 105, num=len(dates)),
            "High": np.linspace(101, 106, num=len(dates)),
            "Low": np.linspace(99, 104, num=len(dates)),
            "Close": np.linspace(100, 105, num=len(dates)),
            "Adj Close": np.linspace(100, 105, num=len(dates)),
            "Volume": np.arange(1, len(dates) + 1) * 100,
        }
    )
    csv_path = tmp_path / "spy.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def split_yaml(tmp_path: Path) -> Path:
    contents = """
name: test_split
instrument: SPY
start_date: "2020-01-02"
end_date: "2020-01-10"
regime_tag: unit_test
"""
    path = tmp_path / "split.yaml"
    path.write_text(contents)
    return path


def test_slice_inclusive_bounds(synthetic_csv: Path, split_yaml: Path) -> None:
    config = spy_loader.load_split_config(split_yaml)
    df = spy_loader.load_raw_csv(synthetic_csv)
    sliced = spy_loader.slice_for_split(df, config)
    assert sliced["Date"].iloc[0].date().isoformat() == "2020-01-02"
    assert sliced["Date"].iloc[-1].date().isoformat() == "2020-01-10"
    assert len(sliced) == 7

    with_returns = spy_loader.compute_log_returns(sliced)
    assert len(with_returns) == 6  # one business day dropped for log return
    assert "log_ret" in with_returns.columns
    assert not with_returns["log_ret"].isna().any()

    enriched = spy_loader.attach_metadata_columns(with_returns, config)
    assert set(["regime_tag", "split_name"]).issubset(enriched.columns)
    assert (enriched["regime_tag"] == "unit_test").all()
    assert (enriched["split_name"] == "test_split").all()


def test_validation_overlap_detection(tmp_path: Path) -> None:
    val_yaml = tmp_path / "spy_val.yaml"
    test_yaml = tmp_path / "spy_test_overlap.yaml"
    val_yaml.write_text(
        """
name: spy_val_overlap
instrument: SPY
start_date: "2018-02-10"
end_date: "2018-03-10"
regime_tag: val
no_overlap_with_tests: true
"""
    )
    test_yaml.write_text(
        """
name: spy_test_overlap
instrument: SPY
start_date: "2018-02-01"
end_date: "2018-02-28"
regime_tag: crisis
"""
    )

    with pytest.raises(ValueError) as excinfo:
        spy_loader.load_split_config(val_yaml)
    assert "overlaps test split" in str(excinfo.value)


def test_cli_smoke(tmp_path: Path, synthetic_csv: Path) -> None:
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        """
name: cli_split
instrument: SPY
start_date: "2020-01-02"
end_date: "2020-01-10"
regime_tag: cli_regime
notes: "Synthetic CLI validation"
"""
    )

    runs_dir = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.spy_loader",
            "--split",
            str(split_path),
            "--csv",
            str(synthetic_csv),
            "--runs_dir",
            str(runs_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cli_split" in result.stdout
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    metadata_path = run_dirs[0] / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["split_name"] == "cli_split"
    assert metadata["regime_tag"] == "cli_regime"
    assert metadata["row_count"] > 0
    assert Path(metadata["split_yaml_path"]).resolve() == split_path.resolve()


def test_real_spy_module_train_val_test_splits(tmp_path: Path) -> None:
    cfg = OmegaConf.load("configs/paper/data.yaml")
    base_data_cfg = OmegaConf.load("configs/data/real_spy_paper.yaml")
    data_cfg = OmegaConf.merge(base_data_cfg, cfg.data)
    data_cfg.cache_dir = str(tmp_path / "cache")
    data_cfg.raw.spy_ohlcv = "data/sample/raw/spy_ohlcv.csv"
    data_cfg.raw.optionmetrics = "data/sample/raw/optionmetrics_spy.csv"
    data_cfg.raw.cboe = "data/sample/raw/cboe_vix.csv"
    data_cfg.require_fresh_cache = True
    data_cfg.prefer_parquet = False

    module = RealSpyDataModule(OmegaConf.to_container(data_cfg, resolve=True))

    expected_order = [
        "spy_train",
        "spy_val_2018q4",
        "spy_test_2018",
        "spy_test_2020",
        "spy_test_2022",
        "spy_test_2008",
    ]
    assert module.env_order == expected_order

    train_envs = module.split_envs("train")
    assert train_envs == ["spy_train"]
    train_batches = module.prepare("train", train_envs)
    assert set(train_batches.keys()) == {"spy_train"}
    assert train_batches["spy_train"].spot.shape[1] > 1

    val_envs = module.split_envs("val")
    assert val_envs == ["spy_val_2018q4"]
    val_batches = module.prepare("val", val_envs)
    assert set(val_batches.keys()) == {"spy_val_2018q4"}

    test_envs = module.split_envs("test")
    assert test_envs == ["spy_test_2018", "spy_test_2020", "spy_test_2022"]
    test_batches = module.prepare("test", test_envs)
    assert set(test_batches.keys()) == set(test_envs)
    for batch in test_batches.values():
        assert batch.option_price.shape[1] == batch.spot.shape[1]

    # Optional split can be requested via the "extra" group or by name when testing
    extra_envs = module.split_envs("extra")
    assert extra_envs == ["spy_test_2008"]
    extra_batch = module.prepare("test", extra_envs)["spy_test_2008"]
    assert extra_batch.meta["start_date"] == "2008-09-01"
    assert extra_batch.meta["end_date"] == "2009-03-31"

    # Caches were written to the temporary directory
    splits_dir = tmp_path / "cache" / "splits"
    assert (splits_dir / "spy_train.csv").exists()
    assert (splits_dir / "spy_val_2018q4.csv").exists()
