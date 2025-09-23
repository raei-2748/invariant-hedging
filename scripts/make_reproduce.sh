#!/usr/bin/env bash
set -euo pipefail

MODELS=("train/erm" "train/erm_reg" "train/irm" "train/groupdro" "train/vrex")

export WANDB_MODE=${WANDB_MODE:-offline}

for cfg in "${MODELS[@]}"; do
  echo "[make_reproduce] Training ${cfg}"
  scripts/run_train.sh "${cfg}" "$@"
  latest_run=$(ls -td runs/20* 2>/dev/null | head -n1)
  if [ -z "${latest_run}" ]; then
    echo "No run directory found after training ${cfg}" >&2
    exit 1
  fi
  latest_ckpt=$(ls -t "${latest_run}"/checkpoints/*.pt 2>/dev/null | head -n1)
  if [ -z "${latest_ckpt}" ]; then
    echo "No checkpoint found in ${latest_run}" >&2
    exit 1
  fi
  echo "[make_reproduce] Evaluating ${cfg} using ${latest_ckpt}"
  scripts/run_eval.sh "${cfg}" eval.report.checkpoint_path="${latest_ckpt}" "$@"
 done
