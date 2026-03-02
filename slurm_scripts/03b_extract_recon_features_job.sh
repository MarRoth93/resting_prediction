#!/bin/bash
#SBATCH --job-name=rp_extract_recon_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

TEST_SUBJECT="${TEST_SUBJECT:-7}"
SUBJ_PADDED=$(printf "%02d" "${TEST_SUBJECT}")
DATA_ROOT="${DATA_ROOT:-processed_data}"
RECON_FEATURE_DIR="${RECON_FEATURE_DIR:-${DATA_ROOT}/reconstruction_features/subj${SUBJ_PADDED}}"
RECON_MODEL_ROOT="${RECON_MODEL_ROOT:-/home/rothermm/brain-diffuser}"
STIMULI_HDF5="${STIMULI_HDF5:-nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"
VD_WEIGHTS_PATH="${VD_WEIGHTS_PATH:-${RECON_MODEL_ROOT}/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth}"

ANNOTS_NPY="${ANNOTS_NPY:-}"
VDVAE_BATCH_SIZE="${VDVAE_BATCH_SIZE:-8}"
CLIPVISION_BATCH_SIZE="${CLIPVISION_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

if [[ -z "${ANNOTS_NPY}" ]]; then
  if [[ -f "${PROJECT_DIR}/data/annots/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="${PROJECT_DIR}/data/annots/COCO_73k_annots_curated.npy"
  elif [[ -f "${PROJECT_DIR}/nsddata/experiments/nsd/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="${PROJECT_DIR}/nsddata/experiments/nsd/COCO_73k_annots_curated.npy"
  elif [[ -f "/home/rothermm/brain-diffuser/data/annots/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="/home/rothermm/brain-diffuser/data/annots/COCO_73k_annots_curated.npy"
  fi
fi

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "TEST_SUBJECT: ${TEST_SUBJECT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "RECON_FEATURE_DIR: ${RECON_FEATURE_DIR}"
echo "RECON_MODEL_ROOT: ${RECON_MODEL_ROOT}"
echo "STIMULI_HDF5: ${STIMULI_HDF5}"
echo "ANNOTS_NPY: ${ANNOTS_NPY:-<auto resolver in python>}"
echo "VD_WEIGHTS_PATH: ${VD_WEIGHTS_PATH}"
echo "VDVAE_BATCH_SIZE: ${VDVAE_BATCH_SIZE}"
echo "CLIPVISION_BATCH_SIZE: ${CLIPVISION_BATCH_SIZE}"
echo "DEVICE: ${DEVICE}"
echo "SKIP_IF_EXISTS: ${SKIP_IF_EXISTS}"

module purge
module load miniconda

source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

cmd=(
  python -u -m src.data.prepare_reconstruction_features
  --subject "${TEST_SUBJECT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${RECON_FEATURE_DIR}"
  --stimuli-hdf5 "${STIMULI_HDF5}"
  --recon-model-root "${RECON_MODEL_ROOT}"
  --vd-weights-path "${VD_WEIGHTS_PATH}"
  --vdvae-batch-size "${VDVAE_BATCH_SIZE}"
  --clipvision-batch-size "${CLIPVISION_BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${ANNOTS_NPY}" ]]; then
  cmd+=(--annots-npy "${ANNOTS_NPY}")
fi

if [[ "${SKIP_IF_EXISTS}" == "1" ]]; then
  cmd+=(--skip-if-exists)
fi

"${cmd[@]}" \
  2>&1 | tee "${LOG_DIR}/extract_recon_features_sub${SUBJ_PADDED}_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
