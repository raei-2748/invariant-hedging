#!/usr/bin/env bash
set -euo pipefail
SEEDS_FILE="seeds/seed_list.txt"
declare -A CONFIGS=(
  [erm_reg]="train/erm_reg"
  [irm_head]="train/hirm_head"
  [groupdro]="train/groupdro"
  [vrex]="train/vrex"
)
METHODS=("erm_reg" "irm_head" "groupdro" "vrex")
for M in "${METHODS[@]}"; do
  CONFIG_NAME=${CONFIGS[$M]}
  while read -r S; do
    [[ -z "$S" || "$S" =~ ^# ]] && continue
    python -m src.train --config-name="$CONFIG_NAME" train.seed="$S" +phase=2
  done < "$SEEDS_FILE"
done
python -m src.eval --config-name=eval/daily +phase=2
python -m src.diagnostics.collect --runs runs --out tables/diag.csv
python -m src.diagnostics.plot --csv tables/diag.csv --outdir figures
echo "Done. See tables/diag.csv and figures/."
