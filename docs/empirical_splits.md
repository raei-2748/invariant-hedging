# Empirical Data Splits

This project uses two real-data anchors to complement the synthetic regimes.

## Training + Validation Anchor (2017–2019)

- Path: `data/real/spy/spy_options_2017_2019.csv`
- Description: Daily SPY ATM 60D call quotes with accompanying bid/ask implied vols and trade metadata.
- Usage: Referenced via `real_data_anchor.file_train` and `run/plan_run1.yaml` as `real_2017_2019`.

## COVID Crisis Anchor (2020)

- Path: `data/real/spy/spy_options_2020_covid.csv`
- Description: Same schema as the training anchor but restricted to the COVID crisis window.
- Usage: Exposed to evaluation-only sweeps through `real_data_anchor.file_test_covid` and `run/plan_run2.yaml` as `real_covid`.

## Reference Futures

- Paths:
  - `data/real/es_futures/es_2017_2019.csv`
  - `data/real/es_futures/es_2020_covid.csv`
- Description: S&P E-mini references for narrative linkage and potential basis diagnostics.

## Schema

Both SPY option files must contain the following columns:

```
trade_date,spot,strike,type,iv_bid,iv_ask,iv_mid,bid,ask,mid,volume,ttm_days
```

Run `tools/scripts/check_schema.py` to verify the schema and capture summary statistics in `reports/data_ingest_report.md` once the full datasets are in place.
