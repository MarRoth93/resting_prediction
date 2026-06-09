#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-config.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/media/psycontrol/HDD/Datasets/brain-diffuser/data}"
DATA_ROOT="${DATA_ROOT:-processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/local_logs}"
PYTHON="${PYTHON:-python}"

MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/shared_space}"
PREDICTION_DIR="${PREDICTION_DIR:-${OUTPUT_ROOT}/predictions}"
ABLATION_DIR="${ABLATION_DIR:-${OUTPUT_ROOT}/ablations}"
FEATURE_OUTPUT_DIR="${FEATURE_OUTPUT_DIR:-${DATA_ROOT}/features}"
STIMULI_PATH="${STIMULI_PATH:-${RAW_DATA_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5}"

TEST_SUBJECT="${TEST_SUBJECT:-7}"
FEWSHOT_N_SHOTS="${FEWSHOT_N_SHOTS:-100}"
FEATURE_TYPE="${FEATURE_TYPE:-}"
MODELS="${MODELS:-clip dinov2}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"

RUN_DOWNLOAD="${RUN_DOWNLOAD:-0}"
DOWNLOAD_REST_MOTION="${DOWNLOAD_REST_MOTION:-0}"
RUN_PREP_TASK="${RUN_PREP_TASK:-1}"
RUN_PREP_REST="${RUN_PREP_REST:-1}"
RUN_FEATURES="${RUN_FEATURES:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OPTUNA_SWEEP="${RUN_OPTUNA_SWEEP:-0}"
RUN_PREDICT_ABLATE="${RUN_PREDICT_ABLATE:-1}"
RUN_VISUALIZE="${RUN_VISUALIZE:-1}"
RUN_EXTRACT_RECON_FEATURES="${RUN_EXTRACT_RECON_FEATURES:-0}"
RUN_BENCHMARK_SDXL="${RUN_BENCHMARK_SDXL:-0}"
RUN_BENCHMARK_VDVAE_VD="${RUN_BENCHMARK_VDVAE_VD:-0}"

SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7}"
DRY_RUN="${DRY_RUN:-0}"
START_AT="${START_AT:-}"
STOP_AFTER="${STOP_AFTER:-}"

N_TRIALS="${N_TRIALS:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-}"
STUDY_NAME="${STUDY_NAME:-}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-}"
RETRAIN_BEST="${RETRAIN_BEST:-1}"
CLEANUP_TRIAL_ARTIFACTS="${CLEANUP_TRIAL_ARTIFACTS:-1}"

VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-${OUTPUT_ROOT}/visualizations/prediction_maps}"
N_EXAMPLES="${N_EXAMPLES:-12}"
EXAMPLE_MODE="${EXAMPLE_MODE:-top}"
EXAMPLE_SEED="${EXAMPLE_SEED:-42}"

RECON_MODEL_ROOT="${RECON_MODEL_ROOT:-${PROJECT_DIR}}"
SUBJ_PADDED=$(printf "%02d" "${TEST_SUBJECT}")
RECON_FEATURE_DIR="${RECON_FEATURE_DIR:-${DATA_ROOT}/reconstruction_features/subj${SUBJ_PADDED}}"
SDXL_BENCHMARK_OUTPUT_DIR="${SDXL_BENCHMARK_OUTPUT_DIR:-${OUTPUT_ROOT}/reconstruction_benchmark/subj${SUBJ_PADDED}}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${OUTPUT_ROOT}/reconstruction_benchmark_vdvae_vd/subj${SUBJ_PADDED}}"
ANNOTS_NPY="${ANNOTS_NPY:-${RAW_DATA_ROOT}/annots/COCO_73k_annots_curated.npy}"
SDXL_FEATURE_NPZ="${SDXL_FEATURE_NPZ:-${RAW_DATA_ROOT}/extracted_features/subj${SUBJ_PADDED}/nsd_sdxl_vae_features.npz}"
SDXL_REF_NPZ="${SDXL_REF_NPZ:-${RAW_DATA_ROOT}/extracted_features/subj${SUBJ_PADDED}/sdxl_vae_ref_latents.npz}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  log "Running ${name}"
  printf 'Command:' | tee "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    log "${name} failed with exit code ${status}; see ${log_file}"
    exit "${status}"
  fi
}

should_run() {
  local step="$1"
  if [[ -n "${START_AT}" && "${step}" < "${START_AT}" ]]; then
    return 1
  fi
  return 0
}

maybe_stop() {
  local step="$1"
  if [[ -n "${STOP_AFTER}" && "${step}" == "${STOP_AFTER}" ]]; then
    log "Stopping after ${step} because STOP_AFTER=${STOP_AFTER}"
    exit 0
  fi
}

resolve_feature_types() {
  if [[ -n "${FEATURE_TYPE}" ]]; then
    printf '%s\n' "${FEATURE_TYPE}"
    return
  fi
  CONFIG_PATH="${CONFIG_PATH}" "${PYTHON}" - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_PATH"]) as f:
    cfg = yaml.safe_load(f) or {}

features = []
default_feature = str(cfg.get("features", {}).get("type", "clip")).strip()
if default_feature:
    features.append(default_feature)

eval_cfg = cfg.get("evaluation", {}) or {}
if bool(eval_cfg.get("run_feature_sweep", True)):
    for feature in eval_cfg.get("feature_backbones", []) or []:
        feature = str(feature).strip()
        if feature and feature not in features:
            features.append(feature)

for feature in features:
    print(feature)
PY
}

resolve_base_study_name() {
  if [[ -n "${STUDY_NAME}" ]]; then
    printf '%s\n' "${STUDY_NAME}"
    return
  fi
  CONFIG_PATH="${CONFIG_PATH}" "${PYTHON}" - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_PATH"]) as f:
    cfg = yaml.safe_load(f) or {}
print(str(cfg.get("sweep", {}).get("shared_space", {}).get("study_name", "shared_space_optuna")).strip())
PY
}

run_download() {
  local cmd=(
    "${PYTHON}" -u download_nsddata.py \
      --output-root "${RAW_DATA_ROOT}"
  )
  [[ "${DOWNLOAD_REST_MOTION}" != "1" ]] && cmd+=(--skip-rest-motion)
  run_cmd "00_download_nsddata" "${cmd[@]}"
}

run_prepare_task() {
  local sub="$1"
  local sub_padded
  sub_padded=$(printf "%02d" "${sub}")
  run_cmd "01_prepare_task_sub${sub_padded}" \
    "${PYTHON}" -u -m src.data.prepare_task_data \
      --sub "${sub}" \
      --data-root "${RAW_DATA_ROOT}" \
      --output-root "${DATA_ROOT}" \
      --config "${CONFIG_PATH}"
}

run_prepare_rest() {
  local sub="$1"
  local sub_padded
  sub_padded=$(printf "%02d" "${sub}")
  run_cmd "02_prepare_rest_sub${sub_padded}" \
    "${PYTHON}" -u -m src.data.prepare_rest_data \
      --sub "${sub}" \
      --data-root "${RAW_DATA_ROOT}" \
      --output-root "${DATA_ROOT}" \
      --config "${CONFIG_PATH}"
}

run_features() {
  read -r -a models_arr <<< "${MODELS}"
  run_cmd "03_extract_features" \
    "${PYTHON}" -u -m src.data.prepare_features \
      --stimuli "${STIMULI_PATH}" \
      --output-dir "${FEATURE_OUTPUT_DIR}" \
      --models "${models_arr[@]}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}"
}

run_standard_training() {
  mapfile -t feature_types < <(resolve_feature_types)
  if [[ "${#feature_types[@]}" -eq 0 ]]; then
    log "No feature backbones resolved from ${CONFIG_PATH}"
    exit 1
  fi
  local primary="${feature_types[0]}"
  for feature in "${feature_types[@]}"; do
    local feature_model_dir="${MODEL_DIR}"
    if [[ "${feature}" != "${primary}" ]]; then
      feature_model_dir="${MODEL_DIR}_${feature}"
    fi
    run_cmd "04_train_shared_space_${feature}" \
      "${PYTHON}" -u -m src.pipelines.train_shared_space \
        --config "${CONFIG_PATH}" \
        --data-root "${DATA_ROOT}" \
        --raw-data-root "${RAW_DATA_ROOT}" \
        --output-dir "${feature_model_dir}" \
        --feature-type "${feature}"
  done
}

run_optuna_training() {
  mapfile -t feature_types < <(resolve_feature_types)
  if [[ "${#feature_types[@]}" -eq 0 ]]; then
    log "No feature backbones resolved from ${CONFIG_PATH}"
    exit 1
  fi
  local primary="${feature_types[0]}"
  local base_study
  base_study=$(resolve_base_study_name)

  for feature in "${feature_types[@]}"; do
    local feature_model_dir="${MODEL_DIR}"
    local feature_study_name="${base_study}"
    if [[ "${feature}" != "${primary}" ]]; then
      feature_model_dir="${MODEL_DIR}_${feature}"
      feature_study_name="${base_study}_${feature}"
    fi

    local cmd=(
      "${PYTHON}" -u -m src.pipelines.sweep_shared_space_optuna
      --config "${CONFIG_PATH}"
      --data-root "${DATA_ROOT}"
      --raw-data-root "${RAW_DATA_ROOT}"
      --feature-type "${feature}"
      --study-name "${feature_study_name}"
      --best-model-dir "${feature_model_dir}"
    )
    [[ -n "${N_TRIALS}" ]] && cmd+=(--n-trials "${N_TRIALS}")
    [[ -n "${TIMEOUT_SEC}" ]] && cmd+=(--timeout "${TIMEOUT_SEC}")
    [[ -n "${SWEEP_OUTPUT_ROOT}" ]] && cmd+=(--output-root "${SWEEP_OUTPUT_ROOT}")
    [[ "${RETRAIN_BEST}" == "0" ]] && cmd+=(--no-retrain-best)
    [[ "${CLEANUP_TRIAL_ARTIFACTS}" == "0" ]] && cmd+=(--no-cleanup-trial-artifacts)

    run_cmd "04b_optuna_sweep_${feature}" "${cmd[@]}"
  done
}

run_predict_ablate() {
  local feature="${FEATURE_TYPE}"
  if [[ -z "${feature}" ]]; then
    feature=$(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON}" - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_PATH"]) as f:
    cfg = yaml.safe_load(f) or {}
print(str(cfg.get("features", {}).get("type", "clip")).strip())
PY
)
  fi

  run_cmd "05_predict_zero_shot" \
    "${PYTHON}" -u -m src.pipelines.predict_subject \
      --mode zero_shot \
      --test-sub "${TEST_SUBJECT}" \
      --model-dir "${MODEL_DIR}" \
      --data-root "${DATA_ROOT}" \
      --feature-type "${feature}" \
      --output-dir "${PREDICTION_DIR}"

  run_cmd "05_predict_few_shot" \
    "${PYTHON}" -u -m src.pipelines.predict_subject \
      --mode few_shot \
      --test-sub "${TEST_SUBJECT}" \
      --n-shots "${FEWSHOT_N_SHOTS}" \
      --model-dir "${MODEL_DIR}" \
      --data-root "${DATA_ROOT}" \
      --feature-type "${feature}" \
      --output-dir "${PREDICTION_DIR}"

  run_cmd "05_run_ablations" \
    "${PYTHON}" -u -m src.pipelines.run_ablations \
      --config "${CONFIG_PATH}" \
      --model-dir "${MODEL_DIR}" \
      --data-root "${DATA_ROOT}" \
      --feature-types "${feature}" \
      --output-dir "${ABLATION_DIR}"
}

run_visualize() {
  run_cmd "06_visualize_prediction_maps" \
    "${PYTHON}" -u -m src.pipelines.visualize_prediction_maps \
      --test-sub "${TEST_SUBJECT}" \
      --data-root "${DATA_ROOT}" \
      --predictions-dir "${PREDICTION_DIR}" \
      --ablation-dir "${ABLATION_DIR}/fewshot" \
      --output-dir "${VIS_OUTPUT_DIR}" \
      --n-examples "${N_EXAMPLES}" \
      --example-mode "${EXAMPLE_MODE}" \
      --example-seed "${EXAMPLE_SEED}"
}

run_recon_features() {
  local cmd=(
    "${PYTHON}" -u -m src.data.prepare_reconstruction_features
    --subject "${TEST_SUBJECT}"
    --data-root "${DATA_ROOT}"
    --output-dir "${RECON_FEATURE_DIR}"
    --stimuli-hdf5 "${STIMULI_PATH}"
    --recon-model-root "${RECON_MODEL_ROOT}"
    --vd-weights-path "${RECON_MODEL_ROOT}/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth"
    --annots-npy "${ANNOTS_NPY}"
    --vdvae-batch-size "${VDVAE_BATCH_SIZE:-8}"
    --clipvision-batch-size "${CLIPVISION_BATCH_SIZE:-8}"
    --device "${DEVICE}"
  )
  [[ "${SKIP_IF_EXISTS:-1}" == "1" ]] && cmd+=(--skip-if-exists)
  run_cmd "03b_extract_recon_features_sub${SUBJ_PADDED}" "${cmd[@]}"
}

run_sdxl_benchmark() {
  local cmd=(
    "${PYTHON}" -u -m src.pipelines.benchmark_reconstructions
    --test-sub "${TEST_SUBJECT}"
    --data-root "${DATA_ROOT}"
    --predictions-dir "${PREDICTION_DIR}"
    --ablation-dir "${ABLATION_DIR}/fewshot"
    --output-dir "${SDXL_BENCHMARK_OUTPUT_DIR}"
    --sdxl-feature-npz "${SDXL_FEATURE_NPZ}"
    --sdxl-ref-npz "${SDXL_REF_NPZ}"
    --alpha-min "${ALPHA_MIN:-1e4}"
    --alpha-max "${ALPHA_MAX:-1e7}"
    --alpha-count "${ALPHA_COUNT:-8}"
    --cv-folds "${CV_FOLDS:-5}"
    --fmri-scale "${FMRI_SCALE:-300.0}"
    --decode-batch-size "${DECODE_BATCH_SIZE:-8}"
    --device "${DEVICE}"
    --precision "${PRECISION:-fp32}"
    --n-panels "${N_PANELS:-20}"
  )
  [[ -n "${TEST_IMAGES_NPY:-}" ]] && cmd+=(--test-images-npy "${TEST_IMAGES_NPY}")
  [[ -n "${TEST_IMAGES_DIR:-}" ]] && cmd+=(--test-images-dir "${TEST_IMAGES_DIR}")
  [[ -n "${FEWSHOT_N_SHOTS_OVERRIDE:-}" ]] && cmd+=(--fewshot-n-shots "${FEWSHOT_N_SHOTS_OVERRIDE}")
  [[ -n "${FEWSHOT_SEED:-}" ]] && cmd+=(--fewshot-seed "${FEWSHOT_SEED}")
  run_cmd "07_benchmark_recon_sdxl" "${cmd[@]}"
}

run_vdvae_vd_benchmark() {
  run_cmd "08_benchmark_recon_vdvae_vd" \
    "${PYTHON}" -u -m src.pipelines.benchmark_reconstructions_vdvae_vd \
      --test-sub "${TEST_SUBJECT}" \
      --data-root "${DATA_ROOT}" \
      --predictions-dir "${PREDICTION_DIR}" \
      --ablation-dir "${ABLATION_DIR}/fewshot" \
      --output-dir "${BENCHMARK_OUTPUT_DIR}" \
      --recon-model-root "${RECON_MODEL_ROOT}" \
      --vdvae-feature-npz "${RECON_FEATURE_DIR}/vdvae_features.npz" \
      --vdvae-ref-npz "${RECON_FEATURE_DIR}/ref_latents.npz" \
      --cliptext-train-npy "${RECON_FEATURE_DIR}/cliptext_train.npy" \
      --cliptext-test-npy "${RECON_FEATURE_DIR}/cliptext_test.npy" \
      --clipvision-train-npy "${RECON_FEATURE_DIR}/clipvision_train.npy" \
      --clipvision-test-npy "${RECON_FEATURE_DIR}/clipvision_test.npy" \
      --vd-weights-path "${RECON_MODEL_ROOT}/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth" \
      --device "${DEVICE}"
}

log "Local pipeline from ${PROJECT_DIR}"
log "RAW_DATA_ROOT=${RAW_DATA_ROOT}"
log "DATA_ROOT=${DATA_ROOT}"
log "OUTPUT_ROOT=${OUTPUT_ROOT}"
log "LOG_DIR=${LOG_DIR}"
log "DRY_RUN=${DRY_RUN}"

if should_run "00_download"; then
  [[ "${RUN_DOWNLOAD}" == "1" ]] && run_download || log "Skipping 00_download"
  maybe_stop "00_download"
fi

if should_run "01_prepare_task"; then
  if [[ "${RUN_PREP_TASK}" == "1" ]]; then
    for sub in ${SUBJECTS}; do
      run_prepare_task "${sub}"
    done
  else
    log "Skipping 01_prepare_task"
  fi
  maybe_stop "01_prepare_task"
fi

if should_run "02_prepare_rest"; then
  if [[ "${RUN_PREP_REST}" == "1" ]]; then
    for sub in ${SUBJECTS}; do
      run_prepare_rest "${sub}"
    done
  else
    log "Skipping 02_prepare_rest"
  fi
  maybe_stop "02_prepare_rest"
fi

if should_run "03_extract_features"; then
  [[ "${RUN_FEATURES}" == "1" ]] && run_features || log "Skipping 03_extract_features"
  maybe_stop "03_extract_features"
fi

if should_run "03b_extract_recon_features"; then
  [[ "${RUN_EXTRACT_RECON_FEATURES}" == "1" ]] && run_recon_features || log "Skipping 03b_extract_recon_features"
  maybe_stop "03b_extract_recon_features"
fi

if should_run "04_train"; then
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ "${RUN_OPTUNA_SWEEP}" == "1" ]]; then
      run_optuna_training
    else
      run_standard_training
    fi
  else
    log "Skipping 04_train"
  fi
  maybe_stop "04_train"
fi

if should_run "05_predict_ablate"; then
  [[ "${RUN_PREDICT_ABLATE}" == "1" ]] && run_predict_ablate || log "Skipping 05_predict_ablate"
  maybe_stop "05_predict_ablate"
fi

if should_run "06_visualize"; then
  [[ "${RUN_VISUALIZE}" == "1" ]] && run_visualize || log "Skipping 06_visualize"
  maybe_stop "06_visualize"
fi

if should_run "07_benchmark_sdxl"; then
  [[ "${RUN_BENCHMARK_SDXL}" == "1" ]] && run_sdxl_benchmark || log "Skipping 07_benchmark_sdxl"
  maybe_stop "07_benchmark_sdxl"
fi

if should_run "08_benchmark_vdvae_vd"; then
  [[ "${RUN_BENCHMARK_VDVAE_VD}" == "1" ]] && run_vdvae_vd_benchmark || log "Skipping 08_benchmark_vdvae_vd"
  maybe_stop "08_benchmark_vdvae_vd"
fi

log "Local pipeline complete."
