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
- `03b_extract_recon_features_job.sh`: build local reconstruction feature bundle for one subject (VDVAE + CLIP-text + CLIP-vision)
- `04_train_shared_space_job.sh`: train shared-space model
- `04b_optuna_sweep_job.sh`: Optuna LOSO sweep for shared-space hyperparameters and retrain best model(s)
- `05_predict_and_ablate_job.sh`: zero-shot + few-shot + ablation run
- `06_visualize_prediction_maps_job.sh`: render side-by-side GT vs zero-shot vs best few-shot maps
- `07_benchmark_recon_job.sh`: reconstruct GT/zero/few-shot conditions via SDXL-VAE latent pipeline
- `08_benchmark_recon_vdvae_vd_job.sh`: reconstruct GT/zero/few-shot conditions via VDVAE + Versatile Diffusion pipeline
- `submit_full_pipeline.sh`: submits the full dependency chain
  - default chain includes `01/02/03` (parallel) + `03b`, then `04b -> 05`, then `06` and `08`
  - `07` (SDXL benchmark) is skipped by default; enable with `RUN_BENCHMARK_SDXL=1`
  - `08` depends on both prediction outputs and `03b` outputs when `RUN_EXTRACT_RECON_FEATURES=1`
  - set `RUN_OPTUNA_SWEEP=0` to use `04_train_shared_space_job.sh` instead of `04b`

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
- `RECON_MODEL_ROOT`, `RECON_FEATURE_DIR`, `VDVAE_FEATURE_NPZ`, `VDVAE_REF_NPZ`, `CLIPTEXT_TRAIN_NPY`, `CLIPTEXT_TEST_NPY`, `CLIPVISION_TRAIN_NPY`, `CLIPVISION_TEST_NPY`, `CLIPTEXT_TRAIN_STIM_IDX_NPY`, `CLIPTEXT_TEST_STIM_IDX_NPY`, `CLIPVISION_TRAIN_STIM_IDX_NPY`, `CLIPVISION_TEST_STIM_IDX_NPY`, `VD_WEIGHTS_PATH`, `ALLOW_PARTIAL_EVAL_FEATURE_COVERAGE`
- `RUN_BENCHMARK_VDVAE_VD`, `RUN_EXTRACT_RECON_FEATURES`, `RUN_BENCHMARK_SDXL`, `RUN_VISUALIZE`, `RUN_OPTUNA_SWEEP`
- `MODEL_DIR`
- `STIMULI_HDF5`, `ANNOTS_NPY`, `VDVAE_BATCH_SIZE`, `CLIPVISION_BATCH_SIZE`, `SKIP_IF_EXISTS`
- `MODELS`, `DEVICE`, `BATCH_SIZE`

For `08_benchmark_recon_vdvae_vd_job.sh`, the default local feature contract is:
- `${RECON_FEATURE_DIR}/vdvae_features.npz` with `train_latents`, `test_latents`, and train/test stimulus-index vectors
- `${RECON_FEATURE_DIR}/ref_latents.npz`
- `${RECON_FEATURE_DIR}/cliptext_train.npy`, `cliptext_test.npy`
- `${RECON_FEATURE_DIR}/clipvision_train.npy`, `clipvision_test.npy`
- optional: `${RECON_FEATURE_DIR}/cliptext_*_stim_idx.npy`, `${RECON_FEATURE_DIR}/clipvision_*_stim_idx.npy`

`03b_extract_recon_features_job.sh` produces the above contract under:
- `${DATA_ROOT}/reconstruction_features/subjXX`

`04b_optuna_sweep_job.sh` now sweeps all configured feature backbones (unless `FEATURE_TYPE` is set),
and retrains each best model directly into:
- primary backbone: `${MODEL_DIR}`
- additional backbones: `${MODEL_DIR}_<feature_type>`

This keeps downstream `05_predict_and_ablate_job.sh` compatible with feature-backbone sweep expectations.

For CLIP-text captions, `03b` resolves annotations in this order:
- `${ANNOTS_NPY}` (if explicitly set)
- `${PROJECT_DIR}/data/annots/COCO_73k_annots_curated.npy`
- `${PROJECT_DIR}/nsddata/experiments/nsd/COCO_73k_annots_curated.npy`
- `/home/rothermm/brain-diffuser/data/annots/COCO_73k_annots_curated.npy`

## Notes

- `submit_full_pipeline.sh` does **not** include `00_download_nsddata_job.sh` by default.
- If your conda env has a different name, set `CONDA_ENV=<name>` when submitting.
- If your SLURM account/partition differs, edit the `#SBATCH` headers accordingly.
