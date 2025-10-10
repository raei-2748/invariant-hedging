#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-train/erm}
shift || true

case "${CONFIG}" in
  train/hirm_head)
    echo "[warn] train/hirm_head is deprecated. Redirecting to train/hirm." >&2
    CONFIG="train/hirm"
    ;;
  train/hirm_hybrid)
    echo "[error] train/hirm_hybrid has been removed. Use train/hirm with irm.mode=hybrid." >&2
    exit 1
    ;;
esac

if [ "${METHOD:-}" = "hirm_head" ]; then
  echo "[warn] METHOD=hirm_head is deprecated; please set METHOD=hirm." >&2
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-SEQUENTIAL}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export HIRM_TORCH_NUM_THREADS=${HIRM_TORCH_NUM_THREADS:-$OMP_NUM_THREADS}

WANDB_MODE=${WANDB_MODE:-offline} python -m src.train --config-name="${CONFIG}" "$@"
