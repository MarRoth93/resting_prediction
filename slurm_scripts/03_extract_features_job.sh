#!/bin/bash
#SBATCH --job-name=rp_extract_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/marco/Marco/resting_prediction/slurm_logs/%x_%j.out
#SBATCH --error=/home/marco/Marco/resting_prediction/slurm_logs/%x_%j.err
#SBATCH --chdir=/home/marco/Marco/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/marco/Marco/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"
STIMULI_PATH="${STIMULI_PATH:-nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"
FEATURE_OUTPUT_DIR="${FEATURE_OUTPUT_DIR:-processed_data/features}"
MODELS="${MODELS:-clip dinov2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "STIMULI_PATH: ${STIMULI_PATH}"
echo "FEATURE_OUTPUT_DIR: ${FEATURE_OUTPUT_DIR}"
echo "MODELS: ${MODELS}"
echo "DEVICE: ${DEVICE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

read -r -a models_arr <<< "${MODELS}"
python -u -m src.data.prepare_features \
  --stimuli "${STIMULI_PATH}" \
  --output-dir "${FEATURE_OUTPUT_DIR}" \
  --models "${models_arr[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/extract_features_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
