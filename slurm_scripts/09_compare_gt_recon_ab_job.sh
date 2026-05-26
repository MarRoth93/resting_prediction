#!/bin/bash
#SBATCH --job-name=09_rp_gt_recon_ab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=36:00:00
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
DATA_ROOT="${DATA_ROOT:-/scratch_shared/rothermm/brain-diffuser/data}"
RECON_MODEL_ROOT="${RECON_MODEL_ROOT:-/home/rothermm/brain-diffuser}"
SHARED_RAW_ROOT="${SHARED_RAW_ROOT:-/scratch_shared/rothermm/brain-diffuser/data}"
STIMULI_HDF5="${STIMULI_HDF5:-${SHARED_RAW_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"
VD_WEIGHTS_PATH="${VD_WEIGHTS_PATH:-${RECON_MODEL_ROOT}/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth}"
ANNOTS_NPY="${ANNOTS_NPY:-}"

AB_ROOT="${AB_ROOT:-outputs/gt_recon_ab_compare/subj${SUBJ_PADDED}}"
LABEL_A="${LABEL_A:-current}"
LABEL_B="${LABEL_B:-brain_diffuser_compat}"

DATA_ROOT_A="${DATA_ROOT_A:-${AB_ROOT}/data_${LABEL_A}}"
DATA_ROOT_B="${DATA_ROOT_B:-${AB_ROOT}/data_${LABEL_B}}"
RECON_FEATURE_DIR_A="${RECON_FEATURE_DIR_A:-${AB_ROOT}/reconstruction_features_${LABEL_A}/subj${SUBJ_PADDED}}"
RECON_FEATURE_DIR_B="${RECON_FEATURE_DIR_B:-${AB_ROOT}/reconstruction_features_${LABEL_B}/subj${SUBJ_PADDED}}"
GT_RECON_DIR_A="${GT_RECON_DIR_A:-${AB_ROOT}/recon_${LABEL_A}}"
GT_RECON_DIR_B="${GT_RECON_DIR_B:-${AB_ROOT}/recon_${LABEL_B}}"
COMPARE_DIR="${COMPARE_DIR:-${AB_ROOT}/comparison}"

VDVAE_BATCH_SIZE="${VDVAE_BATCH_SIZE:-8}"
CLIPVISION_BATCH_SIZE="${CLIPVISION_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp16}"
FMRI_SCALE="${FMRI_SCALE:-300.0}"
VDVAE_ALPHA="${VDVAE_ALPHA:-50000.0}"
CLIPTEXT_ALPHA="${CLIPTEXT_ALPHA:-100000.0}"
CLIPVISION_ALPHA="${CLIPVISION_ALPHA:-60000.0}"
RIDGE_MAX_ITER="${RIDGE_MAX_ITER:-50000}"
VDVAE_CHUNK_SIZE="${VDVAE_CHUNK_SIZE:-2048}"
VD_STRENGTH="${VD_STRENGTH:-0.5}"
VD_MIXING="${VD_MIXING:-0.2}"
VD_GUIDANCE_SCALE="${VD_GUIDANCE_SCALE:-20.0}"
VD_DDIM_STEPS="${VD_DDIM_STEPS:-50}"
VD_DDIM_ETA="${VD_DDIM_ETA:-0.0}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

mkdir -p "${LOG_DIR}"

echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "TEST_SUBJECT: ${TEST_SUBJECT}"
echo "AB_ROOT: ${AB_ROOT}"
echo "LABEL_A: ${LABEL_A}"
echo "LABEL_B: ${LABEL_B}"
echo "DATA_ROOT_A: ${DATA_ROOT_A}"
echo "DATA_ROOT_B: ${DATA_ROOT_B}"
echo "RECON_FEATURE_DIR_A: ${RECON_FEATURE_DIR_A}"
echo "RECON_FEATURE_DIR_B: ${RECON_FEATURE_DIR_B}"
echo "GT_RECON_DIR_A: ${GT_RECON_DIR_A}"
echo "GT_RECON_DIR_B: ${GT_RECON_DIR_B}"
echo "COMPARE_DIR: ${COMPARE_DIR}"

module purge
module load miniconda
source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated Conda environment: $(which python)"

cd "${PROJECT_DIR}"

if [[ -z "${ANNOTS_NPY}" ]]; then
  if [[ -f "${PROJECT_DIR}/data/annots/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="${PROJECT_DIR}/data/annots/COCO_73k_annots_curated.npy"
  elif [[ -f "${SHARED_RAW_ROOT}/annots/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="${SHARED_RAW_ROOT}/annots/COCO_73k_annots_curated.npy"
  elif [[ -f "${SHARED_RAW_ROOT}/nsddata/experiments/nsd/COCO_73k_annots_curated.npy" ]]; then
    ANNOTS_NPY="${SHARED_RAW_ROOT}/nsddata/experiments/nsd/COCO_73k_annots_curated.npy"
  fi
fi

prepare_task() {
  local label="$1"
  local output_root="$2"
  shift 2
  echo "==== Preparing task data: ${label} ===="
  python -u -m src.data.prepare_task_data \
    -sub "${TEST_SUBJECT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${output_root}" \
    "$@"
}

extract_recon_features() {
  local label="$1"
  local variant_data_root="$2"
  local feature_dir="$3"
  echo "==== Extracting reconstruction features: ${label} ===="
  cmd=(
    python -u -m src.data.prepare_reconstruction_features
    --subject "${TEST_SUBJECT}"
    --data-root "${variant_data_root}"
    --output-dir "${feature_dir}"
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
  "${cmd[@]}"
}

run_gt_recon() {
  local label="$1"
  local variant_data_root="$2"
  local feature_dir="$3"
  local out_dir="$4"
  echo "==== Reconstructing GT fMRI: ${label} ===="
  python -u -m src.pipelines.reconstruct_gt_vdvae_vd \
    --test-sub "${TEST_SUBJECT}" \
    --data-root "${variant_data_root}" \
    --output-dir "${out_dir}" \
    --recon-model-root "${RECON_MODEL_ROOT}" \
    --vdvae-feature-npz "${feature_dir}/vdvae_features.npz" \
    --vdvae-ref-npz "${feature_dir}/ref_latents.npz" \
    --cliptext-train-npy "${feature_dir}/cliptext_train.npy" \
    --cliptext-test-npy "${feature_dir}/cliptext_test.npy" \
    --clipvision-train-npy "${feature_dir}/clipvision_train.npy" \
    --clipvision-test-npy "${feature_dir}/clipvision_test.npy" \
    --cliptext-train-stim-idx "${feature_dir}/cliptext_train_stim_idx.npy" \
    --cliptext-test-stim-idx "${feature_dir}/cliptext_test_stim_idx.npy" \
    --clipvision-train-stim-idx "${feature_dir}/clipvision_train_stim_idx.npy" \
    --clipvision-test-stim-idx "${feature_dir}/clipvision_test_stim_idx.npy" \
    --vd-weights-path "${VD_WEIGHTS_PATH}" \
    --fmri-scale "${FMRI_SCALE}" \
    --vdvae-alpha "${VDVAE_ALPHA}" \
    --cliptext-alpha "${CLIPTEXT_ALPHA}" \
    --clipvision-alpha "${CLIPVISION_ALPHA}" \
    --ridge-max-iter "${RIDGE_MAX_ITER}" \
    --vdvae-chunk-size "${VDVAE_CHUNK_SIZE}" \
    --vdvae-batch-size "${VDVAE_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --precision "${PRECISION}" \
    --vd-strength "${VD_STRENGTH}" \
    --vd-mixing "${VD_MIXING}" \
    --vd-guidance-scale "${VD_GUIDANCE_SCALE}" \
    --vd-ddim-steps "${VD_DDIM_STEPS}" \
    --vd-ddim-eta "${VD_DDIM_ETA}"
}

prepare_task "${LABEL_A}" "${DATA_ROOT_A}"
prepare_task "${LABEL_B}" "${DATA_ROOT_B}" --max-sessions 37 --stimulus-order insertion

extract_recon_features "${LABEL_A}" "${DATA_ROOT_A}" "${RECON_FEATURE_DIR_A}"
extract_recon_features "${LABEL_B}" "${DATA_ROOT_B}" "${RECON_FEATURE_DIR_B}"

run_gt_recon "${LABEL_A}" "${DATA_ROOT_A}" "${RECON_FEATURE_DIR_A}" "${GT_RECON_DIR_A}"
run_gt_recon "${LABEL_B}" "${DATA_ROOT_B}" "${RECON_FEATURE_DIR_B}" "${GT_RECON_DIR_B}"

echo "==== Comparing runs ===="
python -u -m src.pipelines.compare_gt_recon_ab \
  --subject "${TEST_SUBJECT}" \
  --label-a "${LABEL_A}" \
  --label-b "${LABEL_B}" \
  --data-root-a "${DATA_ROOT_A}" \
  --data-root-b "${DATA_ROOT_B}" \
  --recon-dir-a "${GT_RECON_DIR_A}" \
  --recon-dir-b "${GT_RECON_DIR_B}" \
  --output-dir "${COMPARE_DIR}" \
  2>&1 | tee "${LOG_DIR}/compare_gt_recon_ab_sub${SUBJ_PADDED}_${SLURM_JOB_ID}.debug.log"
status=${PIPESTATUS[0]}

echo "==== Job finished at $(date) with exit code ${status} ===="
exit "${status}"
