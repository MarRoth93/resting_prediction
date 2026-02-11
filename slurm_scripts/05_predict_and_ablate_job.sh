#!/bin/bash
#SBATCH --job-name=rp_predict_ablate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/marco/Marco/resting_prediction/slurm_logs/%x_%j.out
#SBATCH --error=/home/marco/Marco/resting_prediction/slurm_logs/%x_%j.err
#SBATCH --chdir=/home/marco/Marco/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/marco/Marco/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
MODEL_DIR="${MODEL_DIR:-outputs/shared_space}"
DATA_ROOT="${DATA_ROOT:-processed_data}"
PREDICTION_DIR="${PREDICTION_DIR:-outputs/predictions}"
ABLATION_DIR="${ABLATION_DIR:-outputs/ablations}"
TEST_SUBJECT="${TEST_SUBJECT:-7}"
FEWSHOT_N_SHOTS="${FEWSHOT_N_SHOTS:-100}"
FEATURE_TYPE="${FEATURE_TYPE:-clip}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "PREDICTION_DIR: ${PREDICTION_DIR}"
echo "ABLATION_DIR: ${ABLATION_DIR}"
echo "TEST_SUBJECT: ${TEST_SUBJECT}"
echo "FEWSHOT_N_SHOTS: ${FEWSHOT_N_SHOTS}"
echo "FEATURE_TYPE: ${FEATURE_TYPE}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

python -u -m src.pipelines.predict_subject \
  --mode zero_shot \
  --test-sub "${TEST_SUBJECT}" \
  --model-dir "${MODEL_DIR}" \
  --data-root "${DATA_ROOT}" \
  --feature-type "${FEATURE_TYPE}" \
  --output-dir "${PREDICTION_DIR}" \
  2>&1 | tee "${LOG_DIR}/predict_zero_shot_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}
if [[ "${status}" -ne 0 ]]; then
  echo "Zero-shot prediction failed with status ${status}"
  exit "${status}"
fi

python -u -m src.pipelines.predict_subject \
  --mode few_shot \
  --test-sub "${TEST_SUBJECT}" \
  --n-shots "${FEWSHOT_N_SHOTS}" \
  --model-dir "${MODEL_DIR}" \
  --data-root "${DATA_ROOT}" \
  --feature-type "${FEATURE_TYPE}" \
  --output-dir "${PREDICTION_DIR}" \
  2>&1 | tee "${LOG_DIR}/predict_few_shot_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}
if [[ "${status}" -ne 0 ]]; then
  echo "Few-shot prediction failed with status ${status}"
  exit "${status}"
fi

python -u -m src.pipelines.run_ablations \
  --config "${CONFIG_PATH}" \
  --model-dir "${MODEL_DIR}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${ABLATION_DIR}" \
  2>&1 | tee "${LOG_DIR}/run_ablations_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
