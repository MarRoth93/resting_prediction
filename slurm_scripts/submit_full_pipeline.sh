#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/rothermm/resting_prediction}"
SCRIPT_DIR="${PROJECT_DIR}/slurm_scripts"
LOG_DIR="${PROJECT_DIR}/slurm_logs"

mkdir -p "${LOG_DIR}"

echo "Submitting resting_prediction pipeline from ${PROJECT_DIR}"
echo "Logs: ${LOG_DIR}"

# 1) Preprocessing (parallel)
task_job_id=$(sbatch --parsable "${SCRIPT_DIR}/01_prepare_task_data_array.sh")
rest_job_id=$(sbatch --parsable "${SCRIPT_DIR}/02_prepare_rest_data_array.sh")
feat_job_id=$(sbatch --parsable "${SCRIPT_DIR}/03_extract_features_job.sh")

echo "Submitted task preparation array: ${task_job_id}"
echo "Submitted rest preparation array: ${rest_job_id}"
echo "Submitted feature extraction job: ${feat_job_id}"

# 2) Train after preprocessing is complete
train_job_id=$(sbatch --parsable \
  --dependency="afterok:${task_job_id}:${rest_job_id}:${feat_job_id}" \
  "${SCRIPT_DIR}/04_train_shared_space_job.sh")
echo "Submitted training job: ${train_job_id} (depends on preprocessing)"

# 3) Predict + ablations after training
eval_job_id=$(sbatch --parsable \
  --dependency="afterok:${train_job_id}" \
  "${SCRIPT_DIR}/05_predict_and_ablate_job.sh")
echo "Submitted prediction/ablation job: ${eval_job_id} (depends on training)"

echo "Pipeline submission complete."
