#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SMOKE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
RUN_ROOT="${REPO_ROOT}/runs/paper"

if [[ ! -f "${REPO_ROOT}/tools/run_train.sh" ]] || [[ ! -f "${REPO_ROOT}/tools/run_eval.sh" ]]; then
  echo "Expected run scripts not found under tools/." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/data/spy_sample.csv" ]]; then
  echo "Required data slice missing: data/spy_sample.csv" >&2
  echo "Please populate data prerequisites before running the paper harness." >&2
  exit 1
fi

if (( SMOKE )); then
  METHODS=("train/erm")
  SEEDS=("0")
  EVAL_WINDOWS=("smoke")
  TRAIN_OVERRIDES=(
    "train.steps=100"
    "train.pretrain_steps=20"
    "train.irm_ramp_steps=20"
    "train.eval_interval=20"
    "logging.eval_interval=20"
    "train.checkpoint_topk=1"
  )
  EVAL_OVERRIDES=()
else
  METHODS=(
    "train/erm"
    "train/erm_reg"
    "train/irm"
    "train/groupdro"
    "train/vrex"
  )
  SEEDS=("0" "1" "2" "3" "4")
  EVAL_WINDOWS=("daily" "robustness")
  TRAIN_OVERRIDES=()
  EVAL_OVERRIDES=()
fi

for method in "${METHODS[@]}"; do
  cfg_path="${REPO_ROOT}/configs/${method}.yaml"
  if [[ ! -f "${cfg_path}" ]]; then
    echo "Config not found for method '${method}' (expected ${cfg_path})." >&2
    exit 1
  fi
done

for window in "${EVAL_WINDOWS[@]}"; do
  eval_path="${REPO_ROOT}/configs/eval/${window}.yaml"
  if [[ ! -f "${eval_path}" ]]; then
    echo "Evaluation window config missing: ${eval_path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}"

for method in "${METHODS[@]}"; do
  slug="${method//\//__}"
  for seed in "${SEEDS[@]}"; do
    run_dir="${RUN_ROOT}/${slug}/seed_${seed}"
    train_cmd=("${REPO_ROOT}/tools/run_train.sh" "${method}" "train.seed=${seed}")
    if [[ ${#TRAIN_OVERRIDES[@]-0} -gt 0 ]]; then
      for override in "${TRAIN_OVERRIDES[@]}"; do
        train_cmd+=("${override}")
      done
    fi
    if [[ ${#EXTRA_ARGS[@]-0} -gt 0 ]]; then
      for extra in "${EXTRA_ARGS[@]}"; do
        train_cmd+=("${extra}")
      done
    fi

    echo "[run_of_record] Training ${method} (seed=${seed})"
    echo "  → ${train_cmd[*]}"
    if (( ! DRY_RUN )); then
      if [[ -d "${run_dir}" ]]; then
        echo "  Cleaning existing run directory ${run_dir}" >&2
        rm -rf "${run_dir}"
      fi
      before_latest=$(ls -td "${REPO_ROOT}/runs"/20*/ 2>/dev/null | head -n1 || true)
      "${train_cmd[@]}"
      after_latest=$(ls -td "${REPO_ROOT}/runs"/20*/ 2>/dev/null | head -n1 || true)
      if [[ -z "${after_latest}" ]]; then
        echo "No training run directory created under runs/." >&2
        exit 1
      fi
      if [[ "${after_latest}" == "${before_latest}" ]]; then
        echo "Unable to identify the new training run directory." >&2
        exit 1
      fi
      mkdir -p "${run_dir}"
      cp -a "${after_latest%/}/." "${run_dir}/"
      metrics_path="${run_dir}/final_metrics.json"
      if [[ ! -f "${metrics_path}" ]]; then
        echo "Expected metrics file missing after training: ${metrics_path}" >&2
        exit 1
      fi

      checkpoint=$(python3 "${REPO_ROOT}/tools/scripts/find_latest_checkpoint.py" "${run_dir}" 2>/dev/null || true)
      if [[ -z "${checkpoint}" ]] || [[ ! -f "${checkpoint}" ]]; then
        echo "Unable to resolve checkpoint for ${run_dir}" >&2
        exit 1
      fi

      for window in "${EVAL_WINDOWS[@]}"; do
        eval_dir="${run_dir}/eval/${window}"
        eval_config="${method}"
        eval_args=("eval.report.checkpoint_path=${checkpoint}")
        if (( SMOKE )) && [[ "${window}" == "smoke" ]]; then
          eval_config="eval/${window}"
        else
          eval_args=("eval=${window}" "${eval_args[@]}")
        fi
        eval_cmd=("${REPO_ROOT}/tools/run_eval.sh" "${eval_config}" "${eval_args[@]}")
        if [[ ${#EVAL_OVERRIDES[@]-0} -gt 0 ]]; then
          for override in "${EVAL_OVERRIDES[@]}"; do
            eval_cmd+=("${override}")
          done
        fi
        if [[ ${#EXTRA_ARGS[@]-0} -gt 0 ]]; then
          for extra in "${EXTRA_ARGS[@]}"; do
            eval_cmd+=("${extra}")
          done
        fi
        echo "[run_of_record] Evaluating ${method} (seed=${seed}, window=${window})"
        echo "  → ${eval_cmd[*]}"
        if [[ -d "${eval_dir}" ]]; then
          rm -rf "${eval_dir}"
        fi
        before_eval=$(ls -td "${REPO_ROOT}/runs"/20*/ 2>/dev/null | head -n1 || true)
        "${eval_cmd[@]}"
        after_eval=$(ls -td "${REPO_ROOT}/runs"/20*/ 2>/dev/null | head -n1 || true)
        if [[ -z "${after_eval}" ]]; then
          echo "No evaluation directory created under runs/." >&2
          exit 1
        fi
        if [[ "${after_eval}" == "${before_eval}" ]]; then
          echo "Unable to identify the new evaluation directory." >&2
          exit 1
        fi
        mkdir -p "${eval_dir}"
        cp -a "${after_eval%/}/." "${eval_dir}/"
        eval_metrics="${eval_dir}/final_metrics.json"
        if [[ ! -f "${eval_metrics}" ]]; then
          echo "Expected evaluation metrics missing: ${eval_metrics}" >&2
          exit 1
        fi
      done
    fi
  done
done

if (( DRY_RUN )); then
  echo "[run_of_record] Dry run complete — no commands executed."
  exit 0
fi

methods_csv=$(printf "%s," "${METHODS[@]}")
methods_csv=${methods_csv%,}
seeds_csv=$(printf "%s," "${SEEDS[@]}")
seeds_csv=${seeds_csv%,}
windows_csv=$(printf "%s," "${EVAL_WINDOWS[@]}")
windows_csv=${windows_csv%,}

python3 "${REPO_ROOT}/tools/scripts/paper_provenance.py" \
  --methods "${methods_csv}" \
  --seeds "${seeds_csv}" \
  --eval-windows "${windows_csv}" \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/paper_provenance.json" \
  --metrics-output "${RUN_ROOT}/final_metrics.json"

echo "[run_of_record] Artifacts stored under ${RUN_ROOT}"
