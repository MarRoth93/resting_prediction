# SLURM Scripts for `resting_prediction`

These scripts follow your existing server pattern:
- `partition=normal`
- `module purge && module load miniconda`
- `conda activate <env>`
- detailed debug prints and `tee` logs

## Scripts

- `00_download_nsddata_job.sh` (optional): download NSD data
- `01_prepare_task_data_array.sh`: task-data preprocessing for subjects `[1,2,3,4,5,6,7]` (array job)
- `02_prepare_rest_data_array.sh`: REST preprocessing for subjects `[1,2,3,4,5,6,7]` (array job)
- `03_extract_features_job.sh`: feature extraction (GPU job)
- `04_train_shared_space_job.sh`: train shared-space model
- `05_predict_and_ablate_job.sh`: zero-shot + few-shot + ablation run
- `06_visualize_prediction_maps_job.sh`: render side-by-side GT vs zero-shot vs best few-shot maps
- `07_benchmark_recon_job.sh`: reconstruct GT/zero/few-shot conditions via SDXL-VAE latent pipeline
- `08_benchmark_recon_vdvae_vd_job.sh`: reconstruct GT/zero/few-shot conditions via VDVAE + Versatile Diffusion pipeline
- `submit_full_pipeline.sh`: submits the full dependency chain
  - default chain includes `01 -> 05`, then `06` and `08`
  - `07` (SDXL benchmark) is skipped by default; enable with `RUN_BENCHMARK_SDXL=1`

## Default Paths

All scripts assume project path:
- `/home/rothermm/resting_prediction`

Logs are written to:
- `/home/rothermm/resting_prediction/slurm_logs`

## One-command submission

```bash
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

## Environment overrides

You can override environment variables at submit time, for example:

```bash
CONDA_ENV=myenv \
PROJECT_DIR=/home/rothermm/resting_prediction \
sbatch /home/rothermm/resting_prediction/slurm_scripts/04_train_shared_space_job.sh
```

Common overrides:
- `CONDA_ENV` (default: `resting-prediction`)
- `DATA_ROOT`, `OUTPUT_ROOT`, `CONFIG_PATH`
- `MODEL_DIR`, `PREDICTION_DIR`, `ABLATION_DIR`
- `VIS_OUTPUT_DIR`, `TEST_SUBJECT`, `N_EXAMPLES`, `EXAMPLE_MODE`, `EXAMPLE_SEED`
- `BENCHMARK_OUTPUT_DIR`, `SDXL_FEATURE_NPZ`, `SDXL_REF_NPZ`, `TEST_IMAGES_NPY`, `TEST_IMAGES_DIR`
- `BRAIN_DIFFUSER_ROOT`, `VDVAE_FEATURE_NPZ`, `VDVAE_REF_NPZ`, `CLIPTEXT_TRAIN_NPY`, `CLIPTEXT_TEST_NPY`, `CLIPVISION_TRAIN_NPY`, `CLIPVISION_TEST_NPY`, `VD_WEIGHTS_PATH`
- `MODELS`, `DEVICE`, `BATCH_SIZE`

## Notes

- `submit_full_pipeline.sh` does **not** include `00_download_nsddata_job.sh` by default.
- If your conda env has a different name, set `CONDA_ENV=<name>` when submitting.
- If your SLURM account/partition differs, edit the `#SBATCH` headers accordingly.
