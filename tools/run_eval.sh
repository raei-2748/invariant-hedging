#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
SRC_PATH="${REPO_ROOT}/src"
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SRC_PATH}"
else
  export PYTHONPATH="${SRC_PATH}:${PYTHONPATH}";
fi

CONFIG=${1:-train/erm}
shift || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-SEQUENTIAL}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export KMP_INIT_AT_FORK=${KMP_INIT_AT_FORK:-FALSE}
export HIRM_TORCH_NUM_THREADS=${HIRM_TORCH_NUM_THREADS:-$OMP_NUM_THREADS}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}

python3 experiments/run_diagnostics.py --config-name="${CONFIG}" "$@"
