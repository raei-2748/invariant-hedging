#!/usr/bin/env bash
set -euo pipefail
SEEDS_FILE="seeds/seed_list.txt"
METHODS=("erm_reg" "irm_head" "groupdro" "vrex")
for M in "${METHODS[@]}"; do
  while read -r S || [[ -n "$S" ]]; do
    [[ -z "$S" ]] && continue
    python -m src.train method="$M" seed="$S" +phase=2
  done < "$SEEDS_FILE"
done
python -m src.eval +phase=2
python -m src.diagnostics.collect --runs runs --out tables/diag.csv
python -m src.diagnostics.plot --csv tables/diag.csv --outdir figures
echo "Done. See tables/diag.csv and figures/."
