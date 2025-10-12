# Data guide

The real-data experiments marry three sources:

1. **SPY equity quotes** from Yahoo Finance (public).
2. **CBOE volatility indices** (public, used as a volatility proxy when calibrating).
3. **OptionMetrics IvyDB US** option surfaces (licensed; not bundled).

The training code expects OptionMetrics CSV exports to live alongside public
inputs so the preprocessing stage can synthesise at-the-money options for each
business day. All scripts resolve paths relative to the repository root unless
otherwise noted.

## Directory layout

```
data/
├── raw/
│   ├── spy_ohlcv.csv             # Yahoo Finance SPY OHLCV download
│   ├── cboe/
│   │   └── VIX_History.csv       # CBOE daily VIX closes
│   └── optionmetrics/            # IvyDB CSV exports (manual)
│       ├── ivydb_spy_2010.csv
│       └── ...
├── cache/
│   └── real_spy/                 # Auto-generated calibration artefacts
│       ├── spy_surface.csv
│       ├── manifest.json
│       └── splits/
│           ├── spy_train.csv
│           ├── spy_val_2018q4.csv
│           └── ...
└── sample/                       # Lightweight fixtures used in CI
    ├── raw/                      # Synthetic subset mirroring the layout above
    └── cache/                    # Pre-baked caches for tests
```

The `data/sample/` subtree ships with the repository to power smoke tests. The
real dataset must be staged under `data/raw/` by the user.

### Miniature dataset for releases

Tagged paper releases bundle a tarball named `data-mini.tar.gz` that mirrors the
contents of `data/sample/`. The packaging helper (`scripts/package_release.py`)
produces the tarball and an accompanying SHA256 checksum so downstream users can
extract the lightweight fixtures without cloning the full repository history.
Supply the `--data-tar` flag to re-use an existing archive when the miniature
dataset is unchanged, keeping uploads small during release iterations. See
[RELEASE.md](RELEASE.md#4-collect-release-assets) for the full packaging workflow
and manifest verification steps.

## Fetching public ingredients

`scripts/data/fetch_spy.sh` downloads Yahoo Finance OHLCV data and the CBOE VIX
series. It never fetches OptionMetrics files (which are licensed) but prints the
expected directory structure and cache locations. Run the script in dry-run mode
first to review the actions:

```bash
scripts/data/fetch_spy.sh --dry-run
```

To execute the downloads (requires internet access):

```bash
scripts/data/fetch_spy.sh
```

The script places CSVs under `data/raw/` and reminds you to copy IvyDB exports
into `data/raw/optionmetrics/`.

## OptionMetrics requirements

The loaders expect OptionMetrics CSVs with at least the columns
`trade_date`, `expiration`, `option_type`, `bid`, `ask`, `mid`, and
`implied_vol`. Export a thin slice (e.g. ATM calls) covering 2008–2022 and drop
it into `data/raw/optionmetrics/`. Multiple CSVs are supported; every `.csv`
file in that directory is concatenated during preprocessing.

Because IvyDB is licensed, commit hooks and scripts never attempt to download it
for you. Ensure you comply with your institution’s licence when copying data.

## Preprocessing and caches

`src/data/preprocess.py` provides helpers used by
`src/data/real_spy_loader.RealSpyDataModule` to clean inputs and cache
calibrated surfaces. When `require_fresh_cache` is true (or when no cache exists
at `data/cache/real_spy/`), the module:

1. Reads SPY OHLCV quotes and normalises the `date` and `spot` columns.
2. Ingests all IvyDB CSVs, computing mid prices and implied vols per business
   day.
3. Loads the CBOE VIX close to backfill missing implied vols.
4. Emits a consolidated `spy_surface.csv` plus one CSV per split under
   `data/cache/real_spy/splits/`.
5. Records a JSON manifest describing the cached artefacts.

The splits are sourced from `configs/splits/*.yaml` and aligned with the paper’s
train/validation/test windows. Cached CSVs are reused on subsequent runs to
avoid recomputing calibration.

To force a refresh you can either delete `data/cache/real_spy/` or launch the
paper config with `data.require_fresh_cache=true`:

```bash
python scripts/train.py --config-name paper/data data.require_fresh_cache=true
```

`RealSpyDataModule` automatically rebuilds caches on first access. It also backs
the CI-friendly sample dataset in `data/sample/`, which covers the same date
ranges with synthetic values. If release packaging ever reports missing files,
confirm this directory mirrors the expected layout described above.

## Using the loader in configs

`configs/paper/data.yaml` composes the Hydra experiment with the real SPY loader
(`name: real_spy_paper`). The config points to the sample dataset by default so
unit tests and smoke trainings work out of the box. Override the `data.raw.*`
paths and `data.cache_dir` to point at your staged raw files to run against the
full dataset.
