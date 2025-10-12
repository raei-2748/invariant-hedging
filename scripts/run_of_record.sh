#!/usr/bin/env bash
# Run the compact training + evaluation pipeline used for the paper snapshot.
set -euo pipefail

TRAIN_CONFIG="train/paper"
EVAL_CONFIG="eval/paper"
TRAIN_OVERRIDES=()
EVAL_OVERRIDES=()
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/run_of_record.sh [options]

Options:
  --dry-run                 Print the commands without executing them.
  --train-config NAME       Hydra config name for training (default: train/paper).
  --eval-config NAME        Hydra config name for evaluation (default: eval/paper).
  --train-override ARG      Additional override passed to the train command (repeatable).
  --eval-override ARG       Additional override passed to the eval command (repeatable).
  -h, --help                Show this help message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        --train-override)
            TRAIN_OVERRIDES+=("$2")
            shift 2
            ;;
        --eval-override)
            EVAL_OVERRIDES+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

TRAIN_CMD=("python3" "-m" "src.train" "--config-name=${TRAIN_CONFIG}")
for override in "${TRAIN_OVERRIDES[@]}"; do
    TRAIN_CMD+=("${override}")
done

EVAL_CMD=("python3" "-m" "src.eval" "--config-name=${EVAL_CONFIG}")
for override in "${EVAL_OVERRIDES[@]}"; do
    EVAL_CMD+=("${override}")
done

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[DRY-RUN] ${TRAIN_CMD[*]}"
    echo "[DRY-RUN] latest_run=\$(ls -td runs/paper/*/ 2>/dev/null | head -1)"
    echo "[DRY-RUN] checkpoint=\$(python3 scripts/find_latest_checkpoint.py \"\$latest_run\")"
    echo "[DRY-RUN] ${EVAL_CMD[*]} eval.report.checkpoint_path=\$checkpoint"
    exit 0
fi

"${TRAIN_CMD[@]}"
LATEST_RUN=$(ls -td runs/paper/*/ 2>/dev/null | head -1 || true)
if [[ -z "${LATEST_RUN:-}" ]]; then
    echo "Could not locate the latest training run under runs/paper" >&2
    exit 1
fi
CHECKPOINT=$(python3 scripts/find_latest_checkpoint.py "${LATEST_RUN}")
"${EVAL_CMD[@]}" "eval.report.checkpoint_path=${CHECKPOINT}"
LATEST_EVAL=$(ls -td runs/paper_eval/*/ 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_EVAL:-}" ]]; then
    python3 scripts/build_paper_diagnostics.py "${LATEST_EVAL%/}"
fi
