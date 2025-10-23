#!/usr/bin/env bash
set -euo pipefail

MODELS=("train/erm" "train/erm_reg" "train/irm" "train/groupdro" "train/vrex")

PHASE=${PHASE:-phase1}

case "${PHASE}" in
  phase1)
    export WANDB_MODE=${WANDB_MODE:-offline}

    for cfg in "${MODELS[@]}"; do
      echo "[make_reproduce] Training ${cfg}"
      tools/run_train.sh "${cfg}" "$@"
      latest_run=$(ls -td reports/artifacts/20* 2>/dev/null | head -n1)
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
      tools/run_eval.sh "${cfg}" eval.report.checkpoint_path="${latest_ckpt}" "$@"
    done
    ;;
  phase2)
    echo "Running Phase-2 reproduce: see src/legacy/experiments_notes/phase2_plan.md for configs and objectives."
    ;;
  *)
    echo "Unknown PHASE='${PHASE}'. Supported values: phase1, phase2." >&2
    exit 1
    ;;
esac
