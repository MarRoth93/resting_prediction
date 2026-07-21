#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/psycontrol/miniforge3/envs/resting-prediction/bin/python}"
CONFIG="${CONFIG:-config_schaefer400.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/media/psycontrol/HDD/Datasets/brain-diffuser/data}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/processed_schaefer400}"
FOR_DATA_ROOT="${FOR_DATA_ROOT:-/media/psycontrol/HDD/Datasets/FOR/subjects_400_rest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/schaefer400_multiexpert}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/model}"

usage() {
  cat <<'EOF'
Usage: ./run_schaefer400.sh COMMAND

Commands:
  check-for       Validate all existing FOR parcel exports; no accuracy is computed.
  check-raw       Validate raw NSD inputs needed before atlas/data preparation.
  prepare-atlas   Download minimal official assets and register Schaefer-400 for NSD 1-6.
  prepare-nsd     Prepare parcel-level NSD task and REST arrays for subjects 1-6.
  check-train     Validate all prepared training arrays and CLIP features.
  train           Train the two-expert modal-dropout model in Schaefer parcel space.
  predict-for     REST-only inference for FOR_SUBJECT using FEATURES (.npy).

Overrides: PYTHON, CONFIG, RAW_DATA_ROOT, DATA_ROOT, FOR_DATA_ROOT, OUTPUT_ROOT,
MODEL_DIR, SUBJECTS (space-separated NSD IDs), FOR_SUBJECT, FEATURES, PREDICTION_DIR.
EOF
}

read -r -a subject_args <<< "${SUBJECTS:-1 2 3 4 5 6}"

case "${1:-}" in
  check-for)
    PYTHONPATH=. "${PYTHON}" -m src.pipelines.check_schaefer400 \
      --config "${CONFIG}" --for-data-root "${FOR_DATA_ROOT}" --require for
    ;;
  check-raw)
    PYTHONPATH=. "${PYTHON}" -m src.pipelines.check_schaefer400 \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --for-data-root "${FOR_DATA_ROOT}" \
      --require raw
    ;;
  prepare-atlas)
    PYTHONPATH=. "${PYTHON}" -u -m src.data.prepare_schaefer400_atlas \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --subjects "${subject_args[@]}"
    ;;
  prepare-nsd)
    PYTHONPATH=. "${PYTHON}" -u -m src.data.prepare_schaefer400_nsd \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --subjects "${subject_args[@]}"
    ;;
  check-train)
    PYTHONPATH=. "${PYTHON}" -m src.pipelines.check_schaefer400 \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --for-data-root "${FOR_DATA_ROOT}" \
      --require train
    ;;
  train)
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.train_schaefer400 \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --output-dir "${MODEL_DIR}"
    ;;
  predict-for)
    if [[ -z "${FOR_SUBJECT:-}" || -z "${FEATURES:-}" ]]; then
      echo "predict-for requires FOR_SUBJECT=sub-XXXX and FEATURES=/path/to/clip_features.npy" >&2
      exit 2
    fi
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.predict_for_schaefer400 \
      --config "${CONFIG}" --model-dir "${MODEL_DIR}" \
      --for-data-root "${FOR_DATA_ROOT}" --for-subject "${FOR_SUBJECT}" \
      --features "${FEATURES}" \
      --output-dir "${PREDICTION_DIR:-${OUTPUT_ROOT}/for_predictions/${FOR_SUBJECT}}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
