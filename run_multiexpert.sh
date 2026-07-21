#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/psycontrol/miniforge3/envs/resting-prediction/bin/python}"
CONFIG="${CONFIG:-config_multiexpert.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/media/psycontrol/HDD/Datasets/brain-diffuser/data}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/multiexpert}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/model}"
SUBJECT="${SUBJECT:-7}"
SHOTS="${SHOTS:-100}"
SEED="${SEED:-42}"
CHECK_SCOPE="${CHECK_SCOPE:-train}"

usage() {
  cat <<'EOF'
Usage: ./run_multiexpert.sh COMMAND

Commands:
  check                Validate config, real data, atlases, seeds, and any model.
  prepare-reliability  Create trial-level reliability files for LOSO subjects 1-6.
  train                Train the Stage-1 dropout-fusion model on subjects 1-6.
  loso                 Run the six folds and five robustness seeds, then evaluate the gate.
  predict              Run zero-shot and 100-shot subject prediction (subject 7 stays gated).

Useful overrides: CHECK_SCOPE=train|loso|predict, FOLD=1, LOSO_SEED=42,
SUBJECT=7, SHOTS=100, SEED=42, CONFIG, DATA_ROOT, RAW_DATA_ROOT,
OUTPUT_ROOT, MODEL_DIR, or PYTHON.
EOF
}

case "${1:-}" in
  check)
    PYTHONPATH=. "${PYTHON}" -m src.pipelines.check_multiexpert \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --model-dir "${MODEL_DIR}" \
      --gate-path "${OUTPUT_ROOT}/loso/gate.json" \
      --require "${CHECK_SCOPE}"
    ;;
  prepare-reliability)
    for subject in 1 2 3 4 5 6; do
      PYTHONPATH=. "${PYTHON}" -u -m src.data.prepare_reliability_data \
        --sub "${subject}" --raw-data-root "${RAW_DATA_ROOT}" \
        --processed-root "${DATA_ROOT}"
    done
    ;;
  train)
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.train_multiexpert \
      --config "${CONFIG}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --output-dir "${MODEL_DIR}"
    ;;
  loso)
    args=(
      --config "${CONFIG}"
      --data-root "${DATA_ROOT}"
      --raw-data-root "${RAW_DATA_ROOT}"
      --output-dir "${OUTPUT_ROOT}/loso"
      --frozen-config config.yaml
    )
    if [[ -n "${FOLD:-}" ]]; then
      args+=(--fold "${FOLD}")
    fi
    if [[ -n "${LOSO_SEED:-}" ]]; then
      args+=(--seed "${LOSO_SEED}")
    fi
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.loso_multiexpert "${args[@]}"
    ;;
  predict)
    prediction_dir="${OUTPUT_ROOT}/predictions"
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.predict_multiexpert \
      --mode zero_shot --test-sub "${SUBJECT}" --config "${CONFIG}" \
      --model-dir "${MODEL_DIR}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --output-dir "${prediction_dir}" \
      --gate-path "${OUTPUT_ROOT}/loso/gate.json"
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.predict_multiexpert \
      --mode few_shot --test-sub "${SUBJECT}" --n-shots "${SHOTS}" --seed "${SEED}" \
      --config "${CONFIG}" --model-dir "${MODEL_DIR}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" --output-dir "${prediction_dir}" \
      --gate-path "${OUTPUT_ROOT}/loso/gate.json"
    ;;
  *)
    usage
    exit 2
    ;;
esac
