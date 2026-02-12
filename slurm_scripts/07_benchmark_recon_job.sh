#!/bin/bash
#SBATCH --job-name=rp_benchmark_recon
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/rothermm/resting_prediction/slurm_logs/%x_%j.out
#SBATCH --error=/home/rothermm/resting_prediction/slurm_logs/%x_%j.err
#SBATCH --chdir=/home/rothermm/resting_prediction

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/slurm_logs}"
CONDA_ENV="${CONDA_ENV:-resting-prediction}"

TEST_SUBJECT="${TEST_SUBJECT:-7}"
DATA_ROOT="${DATA_ROOT:-processed_data}"
PREDICTION_DIR="${PREDICTION_DIR:-outputs/predictions}"
ABLATION_DIR="${ABLATION_DIR:-outputs/ablations/fewshot}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-outputs/reconstruction_benchmark/subj07}"

SDXL_FEATURE_NPZ="${SDXL_FEATURE_NPZ:-/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_sdxl_vae_features.npz}"
SDXL_REF_NPZ="${SDXL_REF_NPZ:-/home/rothermm/brain-diffuser/data/extracted_features/subj07/sdxl_vae_ref_latents.npz}"
TEST_IMAGES_NPY="${TEST_IMAGES_NPY:-}"
TEST_IMAGES_DIR="${TEST_IMAGES_DIR:-}"

FEWSHOT_N_SHOTS="${FEWSHOT_N_SHOTS:-}"
FEWSHOT_SEED="${FEWSHOT_SEED:-}"

ALPHA_MIN="${ALPHA_MIN:-1e4}"
ALPHA_MAX="${ALPHA_MAX:-1e7}"
ALPHA_COUNT="${ALPHA_COUNT:-8}"
CV_FOLDS="${CV_FOLDS:-5}"
FMRI_SCALE="${FMRI_SCALE:-300.0}"

DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp32}"
N_PANELS="${N_PANELS:-20}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "TEST_SUBJECT: ${TEST_SUBJECT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "PREDICTION_DIR: ${PREDICTION_DIR}"
echo "ABLATION_DIR: ${ABLATION_DIR}"
echo "BENCHMARK_OUTPUT_DIR: ${BENCHMARK_OUTPUT_DIR}"
echo "SDXL_FEATURE_NPZ: ${SDXL_FEATURE_NPZ}"
echo "SDXL_REF_NPZ: ${SDXL_REF_NPZ}"
echo "TEST_IMAGES_NPY: ${TEST_IMAGES_NPY:-<none>}"
echo "TEST_IMAGES_DIR: ${TEST_IMAGES_DIR:-<none>}"
echo "FEWSHOT_N_SHOTS: ${FEWSHOT_N_SHOTS:-<auto>}"
echo "FEWSHOT_SEED: ${FEWSHOT_SEED:-<auto>}"
echo "DEVICE: ${DEVICE}"
echo "PRECISION: ${PRECISION}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

cmd=(
  python -u -m src.pipelines.benchmark_reconstructions
  --test-sub "${TEST_SUBJECT}"
  --data-root "${DATA_ROOT}"
  --predictions-dir "${PREDICTION_DIR}"
  --ablation-dir "${ABLATION_DIR}"
  --output-dir "${BENCHMARK_OUTPUT_DIR}"
  --sdxl-feature-npz "${SDXL_FEATURE_NPZ}"
  --sdxl-ref-npz "${SDXL_REF_NPZ}"
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --alpha-count "${ALPHA_COUNT}"
  --cv-folds "${CV_FOLDS}"
  --fmri-scale "${FMRI_SCALE}"
  --decode-batch-size "${DECODE_BATCH_SIZE}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --n-panels "${N_PANELS}"
)

if [[ -n "${TEST_IMAGES_NPY}" ]]; then
  cmd+=(--test-images-npy "${TEST_IMAGES_NPY}")
fi

if [[ -n "${TEST_IMAGES_DIR}" ]]; then
  cmd+=(--test-images-dir "${TEST_IMAGES_DIR}")
fi

if [[ -n "${FEWSHOT_N_SHOTS}" ]]; then
  cmd+=(--fewshot-n-shots "${FEWSHOT_N_SHOTS}")
fi

if [[ -n "${FEWSHOT_SEED}" ]]; then
  cmd+=(--fewshot-seed "${FEWSHOT_SEED}")
fi

"${cmd[@]}" \
  2>&1 | tee "${LOG_DIR}/benchmark_recon_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
