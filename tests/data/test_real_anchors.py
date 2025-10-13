from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from hirm.data.real.loader import RealAnchorLoader
from hirm.infra.paths import canonical_run_dir, episode_file_path


@pytest.fixture
def anchor_config(tmp_path: Path) -> Dict:
    dates = pd.bdate_range("2017-01-03", "2022-12-30")
    steps = np.arange(len(dates), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "spot": 100 + 0.1 * steps,
            "option_price": 10 + 0.01 * steps,
            "implied_vol": 0.2,
        }
    )
    data_root = tmp_path / "vendor"
    data_root.mkdir()
    df.to_csv(data_root / "SPY.csv", index=False)
    return {
        "source": "real",
        "anchors": [
            {"name": "train_2017", "split": "train", "start": "2017-01-03", "end": "2017-12-29"},
            {"name": "train_2018", "split": "train", "start": "2018-01-02", "end": "2018-09-28"},
            {"name": "val_2018_late_spike", "split": "val", "start": "2018-10-10", "end": "2018-11-30"},
            {"name": "test_2018_volmageddon", "split": "test", "start": "2018-02-01", "end": "2018-02-28"},
        ],
        "symbols": {"underlying": "SPY"},
        "vendor": {"path_csv_root": str(data_root)},
        "episode": {"days": 20, "stride_days": 5, "tz": "America/New_York"},
        "seed": 7,
    }


def test_episode_determinism(anchor_config: Dict) -> None:
    loader_one = RealAnchorLoader(anchor_config)
    loader_two = RealAnchorLoader(anchor_config)
    anchors_one = loader_one.load()
    anchors_two = loader_two.load()
    for name in anchors_one:
        batch_one = anchors_one[name].batch
        batch_two = anchors_two[name].batch
        assert torch.equal(batch_one.spot, batch_two.spot)
        assert torch.equal(batch_one.option_price, batch_two.option_price)
        assert torch.equal(batch_one.implied_vol, batch_two.implied_vol)


def test_no_overlap_across_splits(anchor_config: Dict) -> None:
    loader = RealAnchorLoader(anchor_config)
    anchors = loader.load()
    train_ranges: List[range] = []
    other_ranges: List[range] = []
    for item in anchors.values():
        tags = item.batch.meta["episode_tags"]
        for tag in tags:
            start = pd.Timestamp(tag["start_date"]).date()
            end = pd.Timestamp(tag["end_date"]).date()
            base = pd.Timestamp("1970-01-01").date()
            days = set(range((start - base).days, (end - base).days + 1))
            if tag["split"] == "train":
                train_ranges.append(days)
            else:
                other_ranges.append(days)
    for train_window in train_ranges:
        for other in other_ranges:
            assert train_window.isdisjoint(other)


def test_tagged_output_directories(anchor_config: Dict, tmp_path: Path) -> None:
    loader = RealAnchorLoader(anchor_config)
    anchors = loader.load()
    first = anchors["train_2017"].batch.meta["episode_tags"][0]
    run_dir = canonical_run_dir("20240101", "real_test", root=tmp_path)
    path = episode_file_path(run_dir, first, "pnl.csv")
    path.write_text(
        "episode_id,start_date,end_date,split,regime_name,source,seed,mean_pnl\n"
        "0,2017-01-03,2017-01-24,train,train_2017,real,7,0.0\n"
    )
    assert path.exists()
    assert "train/train_2017" in str(path)
    header = path.read_text().splitlines()[0]
    assert "regime_name" in header


def test_anchor_boundaries_respected(anchor_config: Dict) -> None:
    loader = RealAnchorLoader(anchor_config)
    anchors = loader.load()
    anchor = anchors["train_2017"].anchor
    tags = anchors["train_2017"].batch.meta["episode_tags"]
    first = tags[0]
    last = tags[-1]
    anchor_start = pd.Timestamp(anchor.start).tz_localize(None)
    anchor_end = pd.Timestamp(anchor.end).tz_localize(None)
    assert pd.Timestamp(first["start_date"]) >= anchor_start
    assert pd.Timestamp(last["end_date"]) <= anchor_end


def test_missing_options_graceful(tmp_path: Path) -> None:
    dates = pd.bdate_range("2017-01-03", "2017-06-30")
    df = pd.DataFrame({"date": dates, "spot": 100 + 0.1 * np.arange(len(dates))})
    root = tmp_path / "vendor"
    root.mkdir()
    df.to_csv(root / "SPY.csv", index=False)
    config = {
        "source": "real",
        "anchors": [
            {"name": "train_smoke", "split": "train", "start": "2017-01-03", "end": "2017-06-30"}
        ],
        "symbols": {"underlying": "SPY"},
        "vendor": {"path_csv_root": str(root)},
        "episode": {"days": 20, "stride_days": 10, "tz": "America/New_York"},
        "seed": 3,
    }
    with pytest.warns(UserWarning):
        loader = RealAnchorLoader(config)
        anchors = loader.load()
    batch = anchors["train_smoke"].batch
    assert torch.allclose(batch.option_price, torch.zeros_like(batch.option_price))
