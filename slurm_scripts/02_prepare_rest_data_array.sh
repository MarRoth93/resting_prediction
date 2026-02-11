#!/bin/bash
#SBATCH --job-name=rp_prepare_rest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --output=/home/rothermm/resting_prediction/slurm_logs/%x_sub%a_%j.out
#SBATCH --error=/home/rothermm/resting_prediction/slurm_logs/%x_sub%a_%j.err
#SBATCH --chdir=/home/rothermm/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"
DATA_ROOT="${DATA_ROOT:-.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"

subjects=(1 2 5 7)
SUBJECT_ID="${subjects[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SUBJECT_ID: ${SUBJECT_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "CONFIG_PATH: ${CONFIG_PATH}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"
python -u -m src.data.prepare_rest_data \
  -sub "${SUBJECT_ID}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_DIR}/prepare_rest_sub${SUBJECT_ID}_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
