#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: fetch_spy.sh [--dry-run]

Downloads the public ingredients for the SPY real-data pipeline and documents
where licensed OptionMetrics exports must be staged.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
CACHE_DIR="$ROOT_DIR/data/cache/real_spy"
PUBLIC_SPY_URL="https://query1.finance.yahoo.com/v7/finance/download/SPY"
PUBLIC_CBOE_URL="https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

period_start="315576000" # 1980-01-01 UTC
date_now="$(date +%s)"

announce_downloads() {
  echo "== Public data downloads =="
  echo "SPY OHLCV (Yahoo Finance) -> $RAW_DIR/spy_ohlcv.csv"
  echo "  curl -L \"$PUBLIC_SPY_URL?period1=$period_start&period2=$date_now&interval=1d&events=history&includeAdjustedClose=true\""
  echo "CBOE VIX daily close -> $RAW_DIR/cboe/VIX_History.csv"
  echo "  curl -L '$PUBLIC_CBOE_URL'"
  echo
  echo "== Licensed data =="
  echo "OptionMetrics IvyDB US exports must be placed under $RAW_DIR/optionmetrics/"
  echo "Expected columns: trade_date, expiration, option_type, bid, ask, mid, implied_vol"
  echo "Place CSVs named like ivydb_spy_*.csv (any .csv in the directory will be ingested)."
  echo
  echo "== Cache bootstrap =="
  echo "After staging files trigger preprocessing by launching the paper config:" 
  echo "  python tools/scripts/train.py --config-name paper/data --multirun data.require_fresh_cache=true"
  echo "(the RealSpyDataModule rebuilds caches automatically on first access)."
  echo
  echo "Caches will be written to $CACHE_DIR"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  announce_downloads
  exit 0
fi

announce_downloads

echo "Creating directories under $RAW_DIR"
mkdir -p "$RAW_DIR" "$RAW_DIR/cboe" "$RAW_DIR/optionmetrics" "$CACHE_DIR"

spy_target="$RAW_DIR/spy_ohlcv.csv"
query="$PUBLIC_SPY_URL?period1=$period_start&period2=$date_now&interval=1d&events=history&includeAdjustedClose=true"
echo "Downloading SPY OHLCV to $spy_target"
if ! curl -L -f "$query" -o "$spy_target"; then
  echo "Warning: failed to download SPY OHLCV from Yahoo Finance." >&2
fi

echo "Downloading CBOE VIX series to $RAW_DIR/cboe/VIX_History.csv"
if ! curl -L -f "$PUBLIC_CBOE_URL" -o "$RAW_DIR/cboe/VIX_History.csv"; then
  echo "Warning: failed to download CBOE VIX series." >&2
fi

echo "Manual step required: copy OptionMetrics IvyDB exports into $RAW_DIR/optionmetrics"
echo "Example: cp /path/to/ivydb/SPY_2010.csv $RAW_DIR/optionmetrics/"

echo "Fetch complete. Run the preprocessing pipeline once files are available."
