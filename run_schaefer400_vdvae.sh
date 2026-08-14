#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/psycontrol/miniforge3/envs/resting-prediction/bin/python}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${ROOT}/artifacts/schaefer400_multiexpert/for_predictions/random_unseen_500_seed42}"
SELECTION_DIR="${SELECTION_DIR:-${ROOT}/artifacts/schaefer400_multiexpert/prediction_inputs/random_unseen_500_seed42}"
NSD_DATA_ROOT="${NSD_DATA_ROOT:-${ROOT}/data/processed_schaefer400}"
RECONSTRUCTION_FEATURE_DIR="${RECONSTRUCTION_FEATURE_DIR:-${ROOT}/data/processed/reconstruction_features/subj07}"
DECODER_DIR="${DECODER_DIR:-${ROOT}/artifacts/schaefer400_multiexpert/vdvae_decoder}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/schaefer400_multiexpert/reconstructions_vdvae/random_unseen_500_seed42}"
RECON_MODEL_ROOT="${RECON_MODEL_ROOT:-${ROOT}/third_party}"
CONDITION="${CONDITION:-learned_fusion}"
DEVICE="${DEVICE:-cuda}"
ALPHA="${ALPHA:-50000.0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/resting_prediction-matplotlib}"

usage() {
  cat <<'EOF'
Usage: ./run_schaefer400_vdvae.sh COMMAND

Commands:
  check         Validate the 50 FOR predictions, shared NSD decoder data, and VDVAE assets.
  fit-decoder   Fit the shared Schaefer-400 -> VDVAE latent ridge decoder only.
  reconstruct   Reconstruct images using an already fitted decoder; resumes partial subjects.
  all           Fit the decoder if needed, then reconstruct all 50 subjects.

The script never starts unless one of these commands is supplied. Defaults use
the random_unseen_500_seed42 prediction batch and learned_fusion.npy. Restrict a
manual smoke run with SUBJECTS="sub-0258". Override PYTHON, PREDICTION_ROOT,
SELECTION_DIR, NSD_DATA_ROOT, RECONSTRUCTION_FEATURE_DIR, DECODER_DIR,
OUTPUT_ROOT, RECON_MODEL_ROOT, CONDITION, DEVICE, ALPHA, or MPLCONFIGDIR as needed.
EOF
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
case "${command}" in
  check|fit-decoder|reconstruct|all) ;;
  *)
    usage
    exit 2
    ;;
esac

subject_args=()
if [[ -n "${SUBJECTS:-}" ]]; then
  read -r -a requested_subjects <<< "${SUBJECTS}"
  subject_args=(--subjects "${requested_subjects[@]}")
fi

mkdir -p "${MPLCONFIGDIR}"
MPLCONFIGDIR="${MPLCONFIGDIR}" PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.reconstruct_for_schaefer400_vdvae \
  "${command}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --selection-dir "${SELECTION_DIR}" \
  --nsd-data-root "${NSD_DATA_ROOT}" \
  --reconstruction-feature-dir "${RECONSTRUCTION_FEATURE_DIR}" \
  --decoder-dir "${DECODER_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --recon-model-root "${RECON_MODEL_ROOT}" \
  --condition "${CONDITION}" \
  --device "${DEVICE}" \
  --alpha "${ALPHA}" \
  "${subject_args[@]}"
