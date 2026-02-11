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
echo "MODEL_DIR: ${MODEL_DIR}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"
python -u -m src.pipelines.train_shared_space \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --output-dir "${MODEL_DIR}" \
  2>&1 | tee "${LOG_DIR}/train_shared_space_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
