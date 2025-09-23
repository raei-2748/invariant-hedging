#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-train/erm}
shift || true

WANDB_MODE=${WANDB_MODE:-offline} python -m src.eval --config-name="${CONFIG}" "$@"
