# Real Market Anchors

The `configs/examples/real_anchors.yaml` configuration defines the deterministic market
regimes used throughout the real-data experiments. Each anchor names a
contiguous window of historical SPY data and is expanded into multiple rolling
episodes using the `episode.days` length and `episode.stride_days` stride. All
episodes are tagged with metadata identifying their split (`train`, `val`, or
`test`), regime name, and original anchor boundaries.

## Expected CSV inputs

The default vendor configuration expects daily CSV files under
`data/real/<SYMBOL>.csv`. At a minimum the underlying file must expose the
following columns:

| column         | description                                   |
|----------------|-----------------------------------------------|
| `date`         | trading day in ISO format                      |
| `spot`         | SPY close price (renamed from `close` if set)  |
| `option_price` | Mid option price (optional; defaults to 0.0)   |
| `implied_vol`  | Implied volatility (optional)                  |

Missing option metrics trigger a warning and the loader falls back to
underlying-only features, allowing smoke tests to run without OptionMetrics
feeds.

## Output structure

Tagged evaluation artifacts are written under the canonical structure:

```
reports/artifacts/<timestamp>_<experiment>/
  seeds/<seed>/<split>/<regime_name>/
    pnl.csv
    cvar95.json
```

Every `pnl.csv` includes the canonical tag columns in the header so that later
analysis can group by split or regime without additional bookkeeping.
