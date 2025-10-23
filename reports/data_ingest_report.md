# Real Data Ingest Report

_Updated: 2025-09-30

## Schema Validation

| file | exists | rows | date_range | missing_columns | notes |
| --- | --- | --- | --- | --- | --- |
| data/real/spy/spy_options_2017_2019.csv | ✅ | 0 | — | — | Header staged; populate with daily SPY ATM 60D calls |
| data/real/spy/spy_options_2020_covid.csv | ✅ | 0 | — | — | Header staged; awaiting COVID window fills |

## Outstanding Items

- Replace the placeholder CSVs with full datasets before rerunning baselines.
- After populating data, rerun:
  ```bash
  python3 scripts/check_schema.py \
    --files data/real/spy/spy_options_2017_2019.csv data/real/spy/spy_options_2020_covid.csv \
    --required spot,strike,type,iv_bid,iv_ask,iv_mid,bid,ask,mid,volume,trade_date,ttm_days \
    --out reports/data_ingest_report.md
  ```
  to append fresh statistics with row counts and date ranges.
- Stage matching S&P E-mini references in `data/real/es_futures/` once sourced (current files contain only headers).

