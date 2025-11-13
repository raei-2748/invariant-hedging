from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from invariant_hedging.data.environment import register_real_anchors


@pytest.fixture
def registry_config(tmp_path: Path) -> Dict:
    dates = pd.bdate_range("2017-01-03", "2019-12-31")
    steps = np.arange(len(dates), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "spot": 100 + 0.1 * steps,
            "option_price": 5 + 0.01 * steps,
            "implied_vol": 0.2,
        }
    )
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    df.to_csv(vendor / "SPY.csv", index=False)
    return {
        "source": "real",
        "anchors": [
            {"name": "train_2017", "split": "train", "start": "2017-01-03", "end": "2017-12-29"},
            {"name": "train_2018", "split": "train", "start": "2018-01-02", "end": "2018-09-28"},
            {"name": "val_2018_late_spike", "split": "val", "start": "2018-10-10", "end": "2018-11-30"},
        ],
        "symbols": {"underlying": "SPY"},
        "vendor": {"path_csv_root": str(vendor)},
        "episode": {"days": 20, "stride_days": 5, "tz": "America/New_York"},
        "seed": 5,
    }


def test_registry_returns_expected_names(registry_config: Dict) -> None:
    specs = register_real_anchors(registry_config)
    names = [spec.name for spec in specs]
    assert names == [anchor["name"] for anchor in registry_config["anchors"]]
    splits = {spec.name: spec.split for spec in specs}
    assert splits["train_2017"] == "train"
    assert splits["val_2018_late_spike"] == "val"


def test_registry_unknown_name(registry_config: Dict) -> None:
    with pytest.raises(KeyError):
        register_real_anchors(registry_config, include=["unknown_anchor"])
