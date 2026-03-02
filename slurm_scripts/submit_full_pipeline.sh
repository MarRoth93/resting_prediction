#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
SCRIPT_DIR="${PROJECT_DIR}/slurm_scripts"
LOG_DIR="${PROJECT_DIR}/slurm_logs"
RUN_VISUALIZE="${RUN_VISUALIZE:-1}"
RUN_BENCHMARK_VDVAE_VD="${RUN_BENCHMARK_VDVAE_VD:-1}"
RUN_EXTRACT_RECON_FEATURES="${RUN_EXTRACT_RECON_FEATURES:-${RUN_BENCHMARK_VDVAE_VD}}"
RUN_OPTUNA_SWEEP="${RUN_OPTUNA_SWEEP:-1}"
# SDXL reconstruction benchmark is disabled by default.
RUN_BENCHMARK_SDXL="${RUN_BENCHMARK_SDXL:-0}"
MODEL_DIR="${MODEL_DIR:-outputs/shared_space}"

mkdir -p "${LOG_DIR}"

echo "Submitting resting_prediction pipeline from ${PROJECT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "RUN_VISUALIZE: ${RUN_VISUALIZE}"
echo "RUN_BENCHMARK_VDVAE_VD: ${RUN_BENCHMARK_VDVAE_VD}"
echo "RUN_EXTRACT_RECON_FEATURES: ${RUN_EXTRACT_RECON_FEATURES}"
echo "RUN_OPTUNA_SWEEP: ${RUN_OPTUNA_SWEEP}"
echo "RUN_BENCHMARK_SDXL: ${RUN_BENCHMARK_SDXL}"
echo "MODEL_DIR: ${MODEL_DIR}"

# 1) Preprocessing (parallel)
task_job_id=$(sbatch --parsable "${SCRIPT_DIR}/01_prepare_task_data_array.sh")
rest_job_id=$(sbatch --parsable "${SCRIPT_DIR}/02_prepare_rest_data_array.sh")
feat_job_id=$(sbatch --parsable "${SCRIPT_DIR}/03_extract_features_job.sh")

echo "Submitted task preparation array: ${task_job_id}"
echo "Submitted rest preparation array: ${rest_job_id}"
echo "Submitted feature extraction job: ${feat_job_id}"

recon_feat_job_id=""
if [[ "${RUN_EXTRACT_RECON_FEATURES}" == "1" ]]; then
  recon_feat_job_id=$(sbatch --parsable "${SCRIPT_DIR}/03b_extract_recon_features_job.sh")
  echo "Submitted reconstruction-feature extraction job: ${recon_feat_job_id}"
else
  echo "Skipping reconstruction-feature extraction job (RUN_EXTRACT_RECON_FEATURES=${RUN_EXTRACT_RECON_FEATURES})"
fi

# 2) Train/sweep after preprocessing is complete
if [[ "${RUN_OPTUNA_SWEEP}" == "1" ]]; then
  train_job_id=$(MODEL_DIR="${MODEL_DIR}" sbatch --parsable \
    --dependency="afterok:${task_job_id}:${rest_job_id}:${feat_job_id}" \
    "${SCRIPT_DIR}/04b_optuna_sweep_job.sh")
  echo "Submitted Optuna sweep job: ${train_job_id} (depends on preprocessing)"
else
  train_job_id=$(MODEL_DIR="${MODEL_DIR}" sbatch --parsable \
    --dependency="afterok:${task_job_id}:${rest_job_id}:${feat_job_id}" \
    "${SCRIPT_DIR}/04_train_shared_space_job.sh")
  echo "Submitted standard training job: ${train_job_id} (depends on preprocessing)"
fi

# 3) Predict + ablations after training
eval_job_id=$(MODEL_DIR="${MODEL_DIR}" sbatch --parsable \
  --dependency="afterok:${train_job_id}" \
  "${SCRIPT_DIR}/05_predict_and_ablate_job.sh")
echo "Submitted prediction/ablation job: ${eval_job_id} (depends on training)"

# 4) Optional downstream steps after prediction/ablation
if [[ "${RUN_VISUALIZE}" == "1" ]]; then
  vis_job_id=$(sbatch --parsable \
    --dependency="afterok:${eval_job_id}" \
    "${SCRIPT_DIR}/06_visualize_prediction_maps_job.sh")
  echo "Submitted visualization job: ${vis_job_id} (depends on prediction/ablation)"
else
  echo "Skipping visualization job (RUN_VISUALIZE=${RUN_VISUALIZE})"
fi

if [[ "${RUN_BENCHMARK_SDXL}" == "1" ]]; then
  sdxl_job_id=$(sbatch --parsable \
    --dependency="afterok:${eval_job_id}" \
    "${SCRIPT_DIR}/07_benchmark_recon_job.sh")
  echo "Submitted SDXL benchmark job: ${sdxl_job_id} (depends on prediction/ablation)"
else
  echo "Skipping SDXL benchmark job (RUN_BENCHMARK_SDXL=${RUN_BENCHMARK_SDXL})"
fi

if [[ "${RUN_BENCHMARK_VDVAE_VD}" == "1" ]]; then
  bench_dependency="afterok:${eval_job_id}"
  if [[ -n "${recon_feat_job_id}" ]]; then
    bench_dependency="afterok:${eval_job_id}:${recon_feat_job_id}"
  fi
  vdvae_vd_job_id=$(sbatch --parsable \
    --dependency="${bench_dependency}" \
    "${SCRIPT_DIR}/08_benchmark_recon_vdvae_vd_job.sh")
  echo "Submitted VDVAE+VD benchmark job: ${vdvae_vd_job_id} (depends on ${bench_dependency})"
else
  echo "Skipping VDVAE+VD benchmark job (RUN_BENCHMARK_VDVAE_VD=${RUN_BENCHMARK_VDVAE_VD})"
fi

echo "Pipeline submission complete."
