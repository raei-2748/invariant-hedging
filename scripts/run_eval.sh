#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-train/erm}
shift || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-SEQUENTIAL}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export HIRM_TORCH_NUM_THREADS=${HIRM_TORCH_NUM_THREADS:-$OMP_NUM_THREADS}

WANDB_MODE=${WANDB_MODE:-offline} python -m hirm.eval --config-name="${CONFIG}" "$@"
