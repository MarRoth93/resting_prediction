#!/bin/bash
#SBATCH --job-name=rp_optuna_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/resting_prediction/slurm_logs/%x_%j.out
#SBATCH --error=/home/rothermm/resting_prediction/slurm_logs/%x_%j.err
#SBATCH --chdir=/home/rothermm/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
DATA_ROOT="${DATA_ROOT:-processed_data}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-.}"
MODEL_DIR="${MODEL_DIR:-outputs/shared_space}"

FEATURE_TYPE="${FEATURE_TYPE:-}"
STUDY_NAME="${STUDY_NAME:-}"
N_TRIALS="${N_TRIALS:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-}"
STORAGE_URL="${STORAGE_URL:-}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-}"
SEED="${SEED:-}"

ALPHA_MIN="${ALPHA_MIN:-}"
ALPHA_MAX="${ALPHA_MAX:-}"
NCOMP_MIN="${NCOMP_MIN:-}"
NCOMP_MAX="${NCOMP_MAX:-}"
NCOMP_STEP="${NCOMP_STEP:-}"
FIXED_EVAL_SIZE="${FIXED_EVAL_SIZE:-}"
EVAL_SPLIT_SEED="${EVAL_SPLIT_SEED:-}"

DRY_RUN="${DRY_RUN:-0}"                            # 1 => --dry-run
RETRAIN_BEST="${RETRAIN_BEST:-1}"                  # 0 => --no-retrain-best
CLEANUP_TRIAL_ARTIFACTS="${CLEANUP_TRIAL_ARTIFACTS:-1}"  # 0 => --no-cleanup-trial-artifacts

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "RAW_DATA_ROOT: ${RAW_DATA_ROOT}"
echo "MODEL_DIR (best-model base): ${MODEL_DIR}"
echo "FEATURE_TYPE: ${FEATURE_TYPE:-<config default>}"
echo "STUDY_NAME: ${STUDY_NAME:-<config default>}"
echo "N_TRIALS: ${N_TRIALS:-<config default>}"
echo "TIMEOUT_SEC: ${TIMEOUT_SEC:-<config default>}"
echo "SWEEP_OUTPUT_ROOT: ${SWEEP_OUTPUT_ROOT:-<config default>}"
echo "DRY_RUN: ${DRY_RUN}"
echo "RETRAIN_BEST: ${RETRAIN_BEST}"
echo "CLEANUP_TRIAL_ARTIFACTS: ${CLEANUP_TRIAL_ARTIFACTS}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

if [[ -n "${FEATURE_TYPE}" ]]; then
  FEATURE_TYPES=("${FEATURE_TYPE}")
else
  readarray -t FEATURE_TYPES < <(
    CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
import yaml

cfg_path = os.environ["CONFIG_PATH"]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

default_feature = str(cfg.get("features", {}).get("type", "clip")).strip()
eval_cfg = cfg.get("evaluation", {}) or {}
run_feature_sweep = bool(eval_cfg.get("run_feature_sweep", True))
feature_backbones = eval_cfg.get("feature_backbones", [])

features = []
if default_feature:
    features.append(default_feature)
if run_feature_sweep and isinstance(feature_backbones, list):
    for feature in feature_backbones:
        feature = str(feature).strip()
        if feature and feature not in features:
            features.append(feature)

for feature in features:
    print(feature)
PY
  )
fi

if [[ "${#FEATURE_TYPES[@]}" -eq 0 ]]; then
  echo "No feature backbones resolved from ${CONFIG_PATH}; aborting."
  exit 1
fi

PRIMARY_FEATURE="${FEATURE_TYPES[0]}"

if [[ -n "${STUDY_NAME}" ]]; then
  BASE_STUDY_NAME="${STUDY_NAME}"
else
  BASE_STUDY_NAME=$(
    CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
import yaml

cfg_path = os.environ["CONFIG_PATH"]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
print(str(cfg.get("sweep", {}).get("shared_space", {}).get("study_name", "shared_space_optuna")).strip())
PY
  )
fi

if [[ -z "${BASE_STUDY_NAME}" ]]; then
  BASE_STUDY_NAME="shared_space_optuna"
fi

echo "Running Optuna sweep for feature backbones: ${FEATURE_TYPES[*]}"
echo "Base study name: ${BASE_STUDY_NAME}"

status=0
for feature in "${FEATURE_TYPES[@]}"; do
  feature_model_dir="${MODEL_DIR}"
  feature_study_name="${BASE_STUDY_NAME}"
  if [[ "${feature}" != "${PRIMARY_FEATURE}" ]]; then
    feature_model_dir="${MODEL_DIR}_${feature}"
    feature_study_name="${BASE_STUDY_NAME}_${feature}"
  fi

  CMD=(
    python -u -m src.pipelines.sweep_shared_space_optuna
    --config "${CONFIG_PATH}"
    --data-root "${DATA_ROOT}"
    --raw-data-root "${RAW_DATA_ROOT}"
    --feature-type "${feature}"
    --study-name "${feature_study_name}"
    --best-model-dir "${feature_model_dir}"
  )

  if [[ -n "${N_TRIALS}" ]]; then
    CMD+=(--n-trials "${N_TRIALS}")
  fi
  if [[ -n "${TIMEOUT_SEC}" ]]; then
    CMD+=(--timeout "${TIMEOUT_SEC}")
  fi
  if [[ -n "${STORAGE_URL}" ]]; then
    CMD+=(--storage "${STORAGE_URL}")
  fi
  if [[ -n "${SWEEP_OUTPUT_ROOT}" ]]; then
    CMD+=(--output-root "${SWEEP_OUTPUT_ROOT}")
  fi
  if [[ -n "${SEED}" ]]; then
    CMD+=(--seed "${SEED}")
  fi
  if [[ -n "${ALPHA_MIN}" ]]; then
    CMD+=(--alpha-min "${ALPHA_MIN}")
  fi
  if [[ -n "${ALPHA_MAX}" ]]; then
    CMD+=(--alpha-max "${ALPHA_MAX}")
  fi
  if [[ -n "${NCOMP_MIN}" ]]; then
    CMD+=(--ncomp-min "${NCOMP_MIN}")
  fi
  if [[ -n "${NCOMP_MAX}" ]]; then
    CMD+=(--ncomp-max "${NCOMP_MAX}")
  fi
  if [[ -n "${NCOMP_STEP}" ]]; then
    CMD+=(--ncomp-step "${NCOMP_STEP}")
  fi
  if [[ -n "${FIXED_EVAL_SIZE}" ]]; then
    CMD+=(--fixed-eval-size "${FIXED_EVAL_SIZE}")
  fi
  if [[ -n "${EVAL_SPLIT_SEED}" ]]; then
    CMD+=(--eval-split-seed "${EVAL_SPLIT_SEED}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    CMD+=(--dry-run)
  fi
  if [[ "${RETRAIN_BEST}" == "0" ]]; then
    CMD+=(--no-retrain-best)
  fi
  if [[ "${CLEANUP_TRIAL_ARTIFACTS}" == "0" ]]; then
    CMD+=(--no-cleanup-trial-artifacts)
  fi

  echo "---- Running Optuna sweep for feature=${feature} study=${feature_study_name} best_model_dir=${feature_model_dir}"
  echo "Command: ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "${LOG_DIR}/optuna_sweep_${feature}_${SLURM_JOB_ID}.debug.log"
  status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    echo "Optuna sweep failed for feature=${feature} with status ${status}"
    exit "${status}"
  fi
done

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
