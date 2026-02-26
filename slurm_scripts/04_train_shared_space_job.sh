#!/bin/bash
#SBATCH --job-name=rp_train_shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "RAW_DATA_ROOT: ${RAW_DATA_ROOT}"
echo "MODEL_DIR (primary): ${MODEL_DIR}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

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

if [[ "${#FEATURE_TYPES[@]}" -eq 0 ]]; then
  echo "No feature backbones resolved from ${CONFIG_PATH}; aborting."
  exit 1
fi

echo "Training feature backbones: ${FEATURE_TYPES[*]}"
PRIMARY_FEATURE="${FEATURE_TYPES[0]}"

for feature in "${FEATURE_TYPES[@]}"; do
  feature_model_dir="${MODEL_DIR}"
  if [[ "${feature}" != "${PRIMARY_FEATURE}" ]]; then
    feature_model_dir="${MODEL_DIR}_${feature}"
  fi

  echo "---- Training shared space for feature=${feature} -> output=${feature_model_dir}"
  python -u -m src.pipelines.train_shared_space \
    --config "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --raw-data-root "${RAW_DATA_ROOT}" \
    --output-dir "${feature_model_dir}" \
    --feature-type "${feature}" \
    2>&1 | tee "${LOG_DIR}/train_shared_space_${feature}_${SLURM_JOB_ID}.debug.log"
  status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    echo "Training failed for feature=${feature} with status ${status}"
    exit "${status}"
  fi
done

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
