# Run The Pipeline

This file is the practical guide for running `resting_prediction`.

If you want the conceptual overview first, read [HOW_PIPELINE_WORKS.md](/home/rothermm/resting_prediction/HOW_PIPELINE_WORKS.md).

## What This Project Expects

The project is built around SLURM jobs in [slurm_scripts](/home/rothermm/resting_prediction/slurm_scripts).

Default assumptions:
- project directory: `/home/rothermm/resting_prediction`
- Conda environment: `resting-prediction`
- raw NSD-style data root: `/scratch_shared/rothermm/brain-diffuser/data`
- processed outputs: `processed_data/`
- model and evaluation outputs: `outputs/`
- logs: `slurm_logs/`

The default train/test split in [config.yaml](/home/rothermm/resting_prediction/config.yaml:4) is:
- training subjects: `1, 2, 3, 4, 5, 6`
- test subject: `7`

## Before You Start

Make sure these are available:
- the Conda environment can import the project dependencies
- raw NSD data exists under the expected shared root
- you can submit jobs with `sbatch`
- the reconstruction benchmark dependencies exist if you plan to run step `03b` or `08`
  - default external model root: `/home/rothermm/brain-diffuser`

Useful directories to check:
- raw data root: `/scratch_shared/rothermm/brain-diffuser/data`
- local logs after submission: `slurm_logs/`
- processed subject data: `processed_data/subjXX/`

## Fastest Way To Run Everything

Submit the whole pipeline:

```bash
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

This submits the default chain:
1. task preprocessing
2. REST preprocessing
3. stimulus feature extraction
4. optional reconstruction-feature extraction
5. shared-space training or Optuna sweep
6. zero-shot and few-shot prediction
7. ablations
8. optional visualization
9. optional reconstruction benchmark

The submission logic lives in [slurm_scripts/submit_full_pipeline.sh](/home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh:1).

## What Gets Submitted

Core jobs:
- `01_prepare_task_data_array.sh`: prepares task fMRI arrays for subjects `1-7`
- `02_prepare_rest_data_array.sh`: prepares REST arrays for subjects `1-7`
- `03_extract_features_job.sh`: extracts CLIP and DINOv2 stimulus features
- `04_train_shared_space_job.sh`: standard shared-space training
- `04b_optuna_sweep_job.sh`: Optuna sweep plus retraining of best model
- `05_predict_and_ablate_job.sh`: zero-shot, few-shot, and ablation runs

Optional jobs:
- `03b_extract_recon_features_job.sh`: prepares local feature bundles for reconstruction benchmarking
- `06_visualize_prediction_maps_job.sh`: renders qualitative prediction examples
- `07_benchmark_recon_job.sh`: SDXL-based reconstruction benchmark
- `08_benchmark_recon_vdvae_vd_job.sh`: VDVAE + Versatile Diffusion benchmark
- `09_compare_gt_recon_ab_job.sh`: GT-only comparison between two task-data definitions

## Recommended First Run

If you are new to the project, use the default pipeline first:

```bash
RUN_OPTUNA_SWEEP=1 \
RUN_VISUALIZE=1 \
RUN_BENCHMARK_VDVAE_VD=0 \
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

Why:
- you get the main preprocessing, training, prediction, and ablation outputs
- you avoid the heaviest reconstruction benchmark on the first pass

## Minimal Main-Pipeline Run

If you want the main prediction pipeline without extra visualization or reconstruction:

```bash
RUN_VISUALIZE=0 \
RUN_BENCHMARK_VDVAE_VD=0 \
RUN_BENCHMARK_SDXL=0 \
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

## Run Jobs One By One

Use this when you want tighter control.

### 1. Prepare task data

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/01_prepare_task_data_array.sh
```

Outputs:
- `processed_data/subjXX/train_fmri.npy`
- `processed_data/subjXX/test_fmri.npy`
- `processed_data/subjXX/train_stim_idx.npy`
- `processed_data/subjXX/test_stim_idx.npy`
- `processed_data/subjXX/mask.npy`

### 2. Prepare REST data

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/02_prepare_rest_data_array.sh
```

Outputs:
- `processed_data/subjXX/rest_run*.npy`
- `processed_data/subjXX/rest_run_manifest.json`

### 3. Extract stimulus features

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/03_extract_features_job.sh
```

Outputs:
- `processed_data/features/clip_features.npy`
- `processed_data/features/dinov2_features.npy`

### 4. Train the shared-space model

Standard training:

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/04_train_shared_space_job.sh
```

Optuna sweep plus retraining of best model:

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/04b_optuna_sweep_job.sh
```

Outputs:
- `outputs/shared_space/`
- optionally `outputs/shared_space_dinov2/`
- optionally `outputs/shared_space_clip_dinov2/`

### 5. Run prediction and ablations

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/05_predict_and_ablate_job.sh
```

Outputs:
- `outputs/predictions/`
- `outputs/ablations/`

### 6. Optional downstream jobs

Visualization:

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/06_visualize_prediction_maps_job.sh
```

Reconstruction features:

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/03b_extract_recon_features_job.sh
```

VDVAE + VD benchmark:

```bash
sbatch /home/rothermm/resting_prediction/slurm_scripts/08_benchmark_recon_vdvae_vd_job.sh
```

## Common Overrides

Most jobs accept environment-variable overrides at submit time.

Examples:

```bash
CONDA_ENV=myenv \
CONFIG_PATH=config.yaml \
sbatch /home/rothermm/resting_prediction/slurm_scripts/04_train_shared_space_job.sh
```

```bash
MODEL_DIR=outputs/shared_space \
TEST_SUBJECT=7 \
FEWSHOT_N_SHOTS=100 \
sbatch /home/rothermm/resting_prediction/slurm_scripts/05_predict_and_ablate_job.sh
```

Common variables:
- `CONDA_ENV`
- `CONFIG_PATH`
- `DATA_ROOT`
- `RAW_DATA_ROOT`
- `OUTPUT_ROOT`
- `MODEL_DIR`
- `PREDICTION_DIR`
- `ABLATION_DIR`
- `TEST_SUBJECT`
- `FEATURE_TYPE`

## A/B Analysis Mask Runs

By default, preprocessing keeps all voxels in NSD's `nsdgeneral` mask:

```yaml
analysis_mask:
  mode: "nsdgeneral"
```

To test the stricter parcel-consistent variant, use:

```yaml
analysis_mask:
  mode: "atlas_labeled_only"
  atlas_type: "combined_rois"
  use_common_labels: true
```

With `use_common_labels: true`, preprocessing keeps only `nsdgeneral` voxels whose atlas labels survive the same cross-subject common-label policy used by shared-space training.

Run A/B variants into separate folders, for example:

```bash
OUTPUT_ROOT=processed_data_nsdgeneral \
DATA_ROOT=processed_data_nsdgeneral \
MODEL_DIR=outputs/shared_space_nsdgeneral \
PREDICTION_DIR=outputs/predictions_nsdgeneral \
ABLATION_DIR=outputs/ablations_nsdgeneral \
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

and for the labeled-only config:

```bash
CONFIG_PATH=config_labeled_only.yaml \
OUTPUT_ROOT=processed_data_labeled_only \
DATA_ROOT=processed_data_labeled_only \
MODEL_DIR=outputs/shared_space_labeled_only \
PREDICTION_DIR=outputs/predictions_labeled_only \
ABLATION_DIR=outputs/ablations_labeled_only \
bash /home/rothermm/resting_prediction/slurm_scripts/submit_full_pipeline.sh
```

Prediction metrics include atlas split summaries such as:
- `atlas_labeled_n_voxels`
- `atlas_labeled_median_r`
- `atlas_unlabeled_n_voxels`
- `atlas_unlabeled_median_r`

## Where To Look After A Run

Logs:
- `slurm_logs/*.out`
- `slurm_logs/*.err`
- `slurm_logs/*.debug.log`

Processed data:
- `processed_data/subjXX/`
- `processed_data/features/`
- `processed_data/reconstruction_features/subjXX/`

Model and evaluation outputs:
- `outputs/shared_space/`
- `outputs/shared_space_<feature>/`
- `outputs/predictions/`
- `outputs/ablations/`
- `outputs/visualizations/`
- `outputs/reconstruction_benchmark_vdvae_vd/`

## Good First Checks

After preprocessing:
- each subject folder exists under `processed_data/subjXX/`
- `train_fmri.npy` and `test_fmri.npy` exist
- at least some `rest_run*.npy` files exist

After feature extraction:
- `processed_data/features/clip_features.npy` exists
- `processed_data/features/dinov2_features.npy` exists

After training:
- `outputs/shared_space/builder.npz` exists
- `outputs/shared_space/encoder.npz` exists
- `outputs/shared_space/shared_stim_idx.npy` exists

After prediction:
- `outputs/predictions/zeroshot_sub7_metrics.json` exists
- `outputs/ablations/fewshot/fewshot_ablation.json` exists

## If Something Fails

Check in this order:
1. the `.out` and `.err` SLURM logs
2. the matching `.debug.log` file in `slurm_logs/`
3. whether your data paths match the defaults in the job script
4. whether the required upstream outputs already exist

Typical failure causes:
- wrong `DATA_ROOT` or missing NSD files
- missing Conda environment or missing Python package
- missing GPU for feature extraction or reconstruction jobs
- trying to run prediction before training artifacts exist
- trying to run reconstruction benchmarking before `03b` has generated local reconstruction features

## Related Files

- configuration: [config.yaml](/home/rothermm/resting_prediction/config.yaml)
- job overview: [slurm_scripts/README.md](/home/rothermm/resting_prediction/slurm_scripts/README.md)
- pipeline concept guide: [HOW_PIPELINE_WORKS.md](/home/rothermm/resting_prediction/HOW_PIPELINE_WORKS.md)
