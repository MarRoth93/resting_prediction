#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/psycontrol/miniforge3/envs/resting-prediction/bin/python}"
CONFIG="${CONFIG:-${ROOT}/config_for_assessor_study.yaml}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/resting_prediction-matplotlib}"
export PYTHON CONFIG MPLCONFIGDIR

study() {
  MPLCONFIGDIR="${MPLCONFIGDIR}" PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.for_assessor_study \
    "$@" --config "${CONFIG}"
}

decoder_validation() {
  MPLCONFIGDIR="${MPLCONFIGDIR}" PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.validate_schaefer400_vdvae_decoder \
    "$@" --study-config "${CONFIG}"
}

usage() {
  cat <<'EOF'
Usage: ./run_for_assessor_study.sh COMMAND

Commands:
  check                 Validate frozen inputs and the completed FOR group CSV.
  validate-nsd          Tune/test the decoder on held-out NSD, reconstruct those
                        validation images, and score them with frozen assessors.
  approve-validation    Record explicit approval after reviewing NSD reports.
  reconstruct-for       Fit the selected shared decoder and reconstruct all FOR subjects.
  score-for             Extract originals and score original/reconstructed FOR images.
  analyze               Run subject-level healthy/depressed and NSD-reference comparisons.
  all-after-approval    Run reconstruct-for, score-for, and analyze sequentially.

Nothing runs without an explicit command. Before `check`, copy
for_groups.example.csv to for_groups.csv and replace the example rows with all
50 real subject/group assignments. `validate-nsd` must be reviewed and approved
before FOR reconstruction can start.
EOF
}

mkdir -p "${MPLCONFIGDIR}"
command="${1:-}"
case "${command}" in
  check)
    study check
    decoder_validation check
    PYTHON="${PYTHON}" ./run_schaefer400_vdvae.sh check
    ;;
  validate-nsd)
    study check
    decoder_validation check
    PYTHON="${PYTHON}" ./run_schaefer400_vdvae.sh check
    decoder_validation validate
    decoder_validation decode
    study extract-originals --sets nsd_original
    study score --sets nsd_original nsd_reconstruction
    study validate-assessors
    ;;
  approve-validation)
    study approve-validation
    ;;
  reconstruct-for)
    study check-approval
    alpha="$(decoder_validation selected-alpha)"
    PYTHON="${PYTHON}" ALPHA="${alpha}" ./run_schaefer400_vdvae.sh all
    ;;
  score-for)
    study check-approval
    study check
    study extract-originals --sets for_original
    study score --sets for_original for_reconstruction
    ;;
  analyze)
    study analyze
    ;;
  all-after-approval)
    "${BASH_SOURCE[0]}" reconstruct-for
    "${BASH_SOURCE[0]}" score-for
    "${BASH_SOURCE[0]}" analyze
    ;;
  *)
    usage
    exit 2
    ;;
esac
