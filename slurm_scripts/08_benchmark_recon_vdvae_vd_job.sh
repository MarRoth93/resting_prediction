#!/bin/bash
#SBATCH --job-name=rp_benchmark_vdvae_vd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
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
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-outputs/reconstruction_benchmark_vdvae_vd/subj07}"

BRAIN_DIFFUSER_ROOT="${BRAIN_DIFFUSER_ROOT:-/home/rothermm/brain-diffuser}"
SUBJ_PADDED=$(printf "%02d" "${TEST_SUBJECT}")

VDVAE_FEATURE_NPZ="${VDVAE_FEATURE_NPZ:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/nsd_vdvae_features_31l.npz}"
VDVAE_REF_NPZ="${VDVAE_REF_NPZ:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/ref_latents.npz}"
CLIPTEXT_TRAIN_NPY="${CLIPTEXT_TRAIN_NPY:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/nsd_cliptext_train.npy}"
CLIPTEXT_TEST_NPY="${CLIPTEXT_TEST_NPY:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/nsd_cliptext_test.npy}"
CLIPVISION_TRAIN_NPY="${CLIPVISION_TRAIN_NPY:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/nsd_clipvision_train.npy}"
CLIPVISION_TEST_NPY="${CLIPVISION_TEST_NPY:-${BRAIN_DIFFUSER_ROOT}/data/extracted_features/subj${SUBJ_PADDED}/nsd_clipvision_test.npy}"
VD_WEIGHTS_PATH="${VD_WEIGHTS_PATH:-${BRAIN_DIFFUSER_ROOT}/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth}"

TEST_IMAGES_NPY="${TEST_IMAGES_NPY:-}"
TEST_IMAGES_DIR="${TEST_IMAGES_DIR:-}"

FEWSHOT_N_SHOTS="${FEWSHOT_N_SHOTS:-}"
FEWSHOT_SEED="${FEWSHOT_SEED:-}"

FMRI_SCALE="${FMRI_SCALE:-300.0}"
VDVAE_ALPHA="${VDVAE_ALPHA:-50000.0}"
CLIPTEXT_ALPHA="${CLIPTEXT_ALPHA:-100000.0}"
CLIPVISION_ALPHA="${CLIPVISION_ALPHA:-60000.0}"
RIDGE_MAX_ITER="${RIDGE_MAX_ITER:-50000}"
VDVAE_CHUNK_SIZE="${VDVAE_CHUNK_SIZE:-2048}"
VDVAE_BATCH_SIZE="${VDVAE_BATCH_SIZE:-8}"

DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp16}"
VD_STRENGTH="${VD_STRENGTH:-0.5}"
VD_MIXING="${VD_MIXING:-0.2}"
VD_GUIDANCE_SCALE="${VD_GUIDANCE_SCALE:-20.0}"
VD_DDIM_STEPS="${VD_DDIM_STEPS:-50}"
VD_DDIM_ETA="${VD_DDIM_ETA:-0.0}"
N_PANELS="${N_PANELS:-20}"
REUSE_PREDICTED_FEATURES="${REUSE_PREDICTED_FEATURES:-1}"

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
echo "BRAIN_DIFFUSER_ROOT: ${BRAIN_DIFFUSER_ROOT}"
echo "VDVAE_FEATURE_NPZ: ${VDVAE_FEATURE_NPZ}"
echo "VDVAE_REF_NPZ: ${VDVAE_REF_NPZ}"
echo "CLIPTEXT_TRAIN_NPY: ${CLIPTEXT_TRAIN_NPY}"
echo "CLIPTEXT_TEST_NPY: ${CLIPTEXT_TEST_NPY}"
echo "CLIPVISION_TRAIN_NPY: ${CLIPVISION_TRAIN_NPY}"
echo "CLIPVISION_TEST_NPY: ${CLIPVISION_TEST_NPY}"
echo "VD_WEIGHTS_PATH: ${VD_WEIGHTS_PATH}"
echo "TEST_IMAGES_NPY: ${TEST_IMAGES_NPY:-<none>}"
echo "TEST_IMAGES_DIR: ${TEST_IMAGES_DIR:-<none>}"
echo "FEWSHOT_N_SHOTS: ${FEWSHOT_N_SHOTS:-<auto>}"
echo "FEWSHOT_SEED: ${FEWSHOT_SEED:-<auto>}"
echo "DEVICE: ${DEVICE}"
echo "PRECISION: ${PRECISION}"
echo "REUSE_PREDICTED_FEATURES: ${REUSE_PREDICTED_FEATURES}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

cmd=(
  python -u -m src.pipelines.benchmark_reconstructions_vdvae_vd
  --test-sub "${TEST_SUBJECT}"
  --data-root "${DATA_ROOT}"
  --predictions-dir "${PREDICTION_DIR}"
  --ablation-dir "${ABLATION_DIR}"
  --output-dir "${BENCHMARK_OUTPUT_DIR}"
  --brain-diffuser-root "${BRAIN_DIFFUSER_ROOT}"
  --vdvae-feature-npz "${VDVAE_FEATURE_NPZ}"
  --vdvae-ref-npz "${VDVAE_REF_NPZ}"
  --cliptext-train-npy "${CLIPTEXT_TRAIN_NPY}"
  --cliptext-test-npy "${CLIPTEXT_TEST_NPY}"
  --clipvision-train-npy "${CLIPVISION_TRAIN_NPY}"
  --clipvision-test-npy "${CLIPVISION_TEST_NPY}"
  --vd-weights-path "${VD_WEIGHTS_PATH}"
  --fmri-scale "${FMRI_SCALE}"
  --vdvae-alpha "${VDVAE_ALPHA}"
  --cliptext-alpha "${CLIPTEXT_ALPHA}"
  --clipvision-alpha "${CLIPVISION_ALPHA}"
  --ridge-max-iter "${RIDGE_MAX_ITER}"
  --vdvae-chunk-size "${VDVAE_CHUNK_SIZE}"
  --vdvae-batch-size "${VDVAE_BATCH_SIZE}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --vd-strength "${VD_STRENGTH}"
  --vd-mixing "${VD_MIXING}"
  --vd-guidance-scale "${VD_GUIDANCE_SCALE}"
  --vd-ddim-steps "${VD_DDIM_STEPS}"
  --vd-ddim-eta "${VD_DDIM_ETA}"
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

if [[ "${REUSE_PREDICTED_FEATURES}" == "1" ]]; then
  cmd+=(--reuse-predicted-features)
fi

"${cmd[@]}" \
  2>&1 | tee "${LOG_DIR}/benchmark_recon_vdvae_vd_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
