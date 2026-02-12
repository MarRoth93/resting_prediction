#!/bin/bash
#SBATCH --job-name=rp_visualize_maps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/resting_prediction/slurm_logs/%x_%j.out
#SBATCH --error=/home/rothermm/resting_prediction/slurm_logs/%x_%j.err
#SBATCH --chdir=/home/rothermm/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"
DATA_ROOT="${DATA_ROOT:-processed_data}"
PREDICTION_DIR="${PREDICTION_DIR:-outputs/predictions}"
ABLATION_DIR="${ABLATION_DIR:-outputs/ablations/fewshot}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-outputs/visualizations/prediction_maps}"
TEST_SUBJECT="${TEST_SUBJECT:-7}"
N_EXAMPLES="${N_EXAMPLES:-12}"
EXAMPLE_MODE="${EXAMPLE_MODE:-top}"
EXAMPLE_SEED="${EXAMPLE_SEED:-42}"
FEWSHOT_N_SHOTS="${FEWSHOT_N_SHOTS:-}"
FEWSHOT_SEED="${FEWSHOT_SEED:-}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "PREDICTION_DIR: ${PREDICTION_DIR}"
echo "ABLATION_DIR: ${ABLATION_DIR}"
echo "VIS_OUTPUT_DIR: ${VIS_OUTPUT_DIR}"
echo "TEST_SUBJECT: ${TEST_SUBJECT}"
echo "N_EXAMPLES: ${N_EXAMPLES}"
echo "EXAMPLE_MODE: ${EXAMPLE_MODE}"
echo "EXAMPLE_SEED: ${EXAMPLE_SEED}"
echo "FEWSHOT_N_SHOTS: ${FEWSHOT_N_SHOTS:-<auto>}"
echo "FEWSHOT_SEED: ${FEWSHOT_SEED:-<auto>}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

cmd=(
  python -u -m src.pipelines.visualize_prediction_maps
  --test-sub "${TEST_SUBJECT}"
  --data-root "${DATA_ROOT}"
  --predictions-dir "${PREDICTION_DIR}"
  --ablation-dir "${ABLATION_DIR}"
  --output-dir "${VIS_OUTPUT_DIR}"
  --n-examples "${N_EXAMPLES}"
  --example-mode "${EXAMPLE_MODE}"
  --example-seed "${EXAMPLE_SEED}"
)

if [[ -n "${FEWSHOT_N_SHOTS}" ]]; then
  cmd+=(--fewshot-n-shots "${FEWSHOT_N_SHOTS}")
fi

if [[ -n "${FEWSHOT_SEED}" ]]; then
  cmd+=(--fewshot-seed "${FEWSHOT_SEED}")
fi

"${cmd[@]}" \
  2>&1 | tee "${LOG_DIR}/visualize_prediction_maps_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
