# How To Use

Complete guide for running the **Predicting Task Activation from Resting State Data** pipeline.

Train on NSD subjects 1, 2, 5 and predict fMRI task activation for subject 7 using resting-state hyperalignment.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Step 1 — Download NSD Data](#3-step-1--download-nsd-data)
4. [Step 2 — Prepare Task Data](#4-step-2--prepare-task-data)
5. [Step 3 — Prepare REST Data](#5-step-3--prepare-rest-data)
6. [Step 4 — Extract Stimulus Features](#6-step-4--extract-stimulus-features)
7. [Step 5 — Train Shared Space Model](#7-step-5--train-shared-space-model)
8. [Step 6 — Predict (Zero-Shot / Few-Shot)](#8-step-6--predict)
9. [Step 7 — Run Ablations](#9-step-7--run-ablations)
10. [Step 8 — Run Tests](#10-step-8--run-tests)
11. [Configuration Reference](#11-configuration-reference)
12. [Directory Structure](#12-directory-structure)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

- Python 3.10+
- AWS CLI configured with access to `s3://natural-scenes-dataset` (public, `--no-sign-request`)
- GPU with CUDA (recommended for feature extraction; CPU works but is slow)
- ~100 GB disk space for full NSD data (less if skipping stimuli)

## 2. Installation

```bash
cd /home/rothermm/resting_prediction
pip install -r requirements.txt
```

### requirements.txt contents

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Core numerical operations |
| scipy | >= 1.10 | SVD, filtering, signal processing |
| nibabel | >= 5.0 | NIfTI file I/O |
| h5py | >= 3.8 | HDF5 stimuli file |
| matplotlib | >= 3.7 | Visualization |
| seaborn | >= 0.12 | Statistical plots |
| scikit-learn | >= 1.2 | Randomized SVD |
| pyyaml | >= 6.0 | Config loading |
| tqdm | >= 4.65 | Progress bars |
| torch | >= 2.0 | Feature extraction models |
| torchvision | >= 0.15 | Image transforms |
| open-clip-torch | >= 2.20 | CLIP features |
| transformers | >= 4.30 | Pretrained models |
| optuna | >= 3.0 | Hyperparameter optimization |
| tensorboard | >= 2.0 | Sweep metric and hparam tracking |
| nilearn | >= 0.10 | Optional neuroimaging utilities |
| pytest | >= 7.0 | Testing |

---

## 3. Step 1 — Download NSD Data

**Script:** `download_nsddata.py`

Downloads all required NSD data from AWS S3 for subjects 1, 2, 5, 7.

### What gets downloaded

| # | Data | S3 Path | Local Path | Size |
|---|------|---------|------------|------|
| 1 | Experiment design | `nsddata/experiments/nsd/` | `nsddata/experiments/nsd/` | ~50 MB |
| 2 | Stimuli (73k images) | `nsddata_stimuli/stimuli/nsd/` | `nsddata_stimuli/stimuli/nsd/` | ~26 GB |
| 3 | Task betas per session | `nsddata_betas/ppdata/subj{XX}/...` | `nsddata_betas/ppdata/subj{XX}/...` | ~60 GB total |
| 4 | ROI masks + atlases | `nsddata/ppdata/subj{XX}/func1pt8mm/roi/` | `nsddata/ppdata/subj{XX}/func1pt8mm/roi/` | ~200 MB |
| 5 | REST timeseries | `nsddata_timeseries/ppdata/subj{XX}/...` | `nsddata_timeseries/ppdata/subj{XX}/...` | ~10 GB |
| 6 | Noise ceiling maps (ncsnr) | `nsddata_betas/ppdata/subj{XX}/...` | `nsddata_betas/ppdata/subj{XX}/func1pt8mm/` | ~50 MB |

### Commands

```bash
# Download everything
python download_nsddata.py

# Skip the 26 GB stimuli file (if already downloaded or not needed yet)
python download_nsddata.py --skip-stimuli

# Skip REST timeseries
python download_nsddata.py --skip-rest

# Skip task betas
python download_nsddata.py --skip-betas

# Download only REST data
python download_nsddata.py --only-rest
```

### Flags

| Flag | Description |
|------|-------------|
| `--skip-stimuli` | Skip the ~26 GB `nsd_stimuli.hdf5` download |
| `--skip-rest` | Skip resting-state timeseries download |
| `--skip-betas` | Skip task beta downloads |
| `--only-rest` | Download only REST timeseries (nothing else) |

### Notes

- Already-downloaded files are automatically skipped.
- Beta sessions are discovered dynamically from S3 (no hardcoded count).
- REST runs are discovered by listing S3 and filtering for files with "rest" in the name.
- A REST run manifest is saved to `processed_data/subj{XX}/rest_run_manifest.json` per subject.
- ROI files downloaded: `nsdgeneral.nii.gz`, `prf-visualrois.nii.gz`, `prf-eccrois.nii.gz`, `Kastner2015.nii.gz`, `floc-bodies.nii.gz`, `floc-faces.nii.gz`, `floc-places.nii.gz`, `floc-words.nii.gz`.

---

## 4. Step 2 — Prepare Task Data

**Script:** `src/data/prepare_task_data.py`

Loads raw NSD betas, applies `nsdgeneral` mask, separates train/test by masterordering threshold, averages repeated presentations, and saves processed arrays.

### Command

```bash
# Process one subject at a time
python -m src.data.prepare_task_data -sub 1
python -m src.data.prepare_task_data -sub 2
python -m src.data.prepare_task_data -sub 5
python -m src.data.prepare_task_data -sub 7
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-sub` / `--sub` | int | **required** | Subject ID. Choices: `1`, `2`, `5`, `7` |
| `--data-root` | str | `.` | Root directory where raw NSD data lives |
| `--output-root` | str | `processed_data` | Output directory for processed files |

### Outputs (per subject)

Saved to `processed_data/subj{XX}/`:

| File | Shape | Description |
|------|-------|-------------|
| `train_fmri.npy` | `(N_train, V_sub)` | Averaged betas per training stimulus |
| `test_fmri.npy` | `(N_test, V_sub)` | Averaged betas per test stimulus |
| `train_stim_idx.npy` | `(N_train,)` | NSD image IDs (canonical sorted order) |
| `test_stim_idx.npy` | `(N_test,)` | NSD image IDs (canonical sorted order) |
| `test_fmri_trials.npy` | `(N_trials, V_sub)` | Trial-level test betas (for noise ceiling) |
| `test_trial_labels.npy` | `(N_trials,)` | Stimulus index per trial |
| `mask.npy` | `(X, Y, Z)` | Boolean nsdgeneral mask |

### Notes

- Train/test split: `masterordering > 1000` = train, `<= 1000` = test (NSD convention).
- Sessions are dynamically discovered (not hardcoded to 37).
- Stimulus ordering is canonical (sorted by NSD image ID) for cross-subject alignment.
- `V_sub` differs per subject (this is the variable voxel problem the pipeline solves).

---

## 5. Step 3 — Prepare REST Data

**Script:** `src/data/prepare_rest_data.py`

Preprocesses resting-state timeseries: discard initial TRs, detrend, high-pass filter, motion censoring, z-score.

### Command

```bash
# Process one subject at a time
python -m src.data.prepare_rest_data -sub 1
python -m src.data.prepare_rest_data -sub 2
python -m src.data.prepare_rest_data -sub 5
python -m src.data.prepare_rest_data -sub 7
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-sub` / `--sub` | int | **required** | Subject ID. Choices: `1`, `2`, `5`, `7` |
| `--data-root` | str | `.` | Root directory for raw NSD data |
| `--output-root` | str | `processed_data` | Output directory |
| `--config` | str | `config.yaml` | Path to config file (reads `rest_preprocessing` section) |

### Preprocessing pipeline (per run)

1. Discard first N TRs (T1 equilibration, default: 5)
2. Linear detrending
3. Butterworth high-pass filter (default: 0.01 Hz cutoff)
4. Motion censoring (drop TRs with FD > 0.5 mm; exclude run if > 30% censored)
5. Nuisance regression (optional, disabled by default)
6. Z-score per voxel

### Config options (`rest_preprocessing` in `config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `discard_initial_trs` | `5` | Initial TRs to drop |
| `detrend` | `true` | Apply linear detrending |
| `highpass_cutoff_hz` | `0.01` | High-pass cutoff in Hz |
| `motion_censoring.enabled` | `true` | Enable FD-based censoring |
| `motion_censoring.fd_threshold_mm` | `0.5` | FD threshold in mm |
| `motion_censoring.max_censored_fraction` | `0.3` | Max censored fraction before excluding run |
| `nuisance_regression` | `false` | Regress out nuisance signals |
| `zscore` | `true` | Z-score per voxel |
| `min_usable_trs` | `100` | Minimum total TRs across all runs |

### Outputs (per subject)

Saved to `processed_data/subj{XX}/`:

| File | Shape | Description |
|------|-------|-------------|
| `rest_run1.npy` | `(T_clean, V_sub)` | Preprocessed REST run 1 |
| `rest_run2.npy` | `(T_clean, V_sub)` | Preprocessed REST run 2 |
| ... | ... | Additional REST runs |

### Notes

- REST runs are discovered from the manifest created by `download_nsddata.py`, or by listing the timeseries directory.
- TR is read from the NIfTI header (expected ~1.333 s for NSD).
- Requires at least 2 REST runs per subject.
- Run is excluded if fewer than 30 TRs remain after discarding initial TRs, or if motion censoring exceeds the threshold.

---

## 6. Step 4 — Extract Stimulus Features

**Script:** `src/data/prepare_features.py`

Extracts image features from all 73k NSD stimuli using pretrained vision models.

### Command

```bash
# Extract both CLIP and DINOv2 features (default)
python -m src.data.prepare_features

# Extract only CLIP
python -m src.data.prepare_features --models clip

# Extract only DINOv2
python -m src.data.prepare_features --models dinov2

# Use CPU (slower)
python -m src.data.prepare_features --device cpu

# Custom batch size
python -m src.data.prepare_features --batch-size 32
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stimuli` | str | `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` | Path to stimuli HDF5 file |
| `--output-dir` | str | `processed_data/features` | Output directory |
| `--models` | str (space-separated) | `clip dinov2` | Models to extract. Options: `clip`, `dinov2` |
| `--device` | str | `cuda` | Device for inference. `cuda` or `cpu` |
| `--batch-size` | int | `64` | Images per batch |

### Outputs

Saved to `processed_data/features/`:

| File | Shape | Description |
|------|-------|-------------|
| `clip_features.npy` | `(73000, 768)` | CLIP ViT-L/14 features |
| `dinov2_features.npy` | `(73000, 1024)` | DINOv2 ViT-L/14 features |

### Notes

- Requires `nsd_stimuli.hdf5` (download with step 1, or skip stimuli and use pre-extracted features).
- Already-extracted feature files are automatically skipped.
- CLIP uses `open_clip` library with OpenAI pretrained weights.
- DINOv2 is loaded via `torch.hub` from `facebookresearch/dinov2`.
- GPU strongly recommended (~30 min on GPU vs hours on CPU).

---

## 7. Step 5 — Train Shared Space Model

**Script:** `src/pipelines/train_shared_space.py`

Builds the shared hyperalignment space from training subjects' REST data and trains the encoding model.

### Command

```bash
# Train with default config
python -m src.pipelines.train_shared_space

# Custom config
python -m src.pipelines.train_shared_space --config my_config.yaml

# Custom paths
python -m src.pipelines.train_shared_space \
    --data-root processed_data \
    --raw-data-root . \
    --output-dir outputs/shared_space
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | `config.yaml` | Path to config file |
| `--data-root` | str | `processed_data` | Root for processed `.npy` files |
| `--raw-data-root` | str | `.` | Root for raw NSD data (needed for atlas NIfTI files) |
| `--output-dir` | str | `outputs/shared_space` | Where to save model artifacts |
| `--feature-type` | str | `""` | Optional feature override (`clip`, `dinov2`, `clip_dinov2`) |

### What it does

1. Loads all training subjects' preprocessed data (task + REST)
2. Loads and harmonizes atlas across all subjects (intersection policy)
3. Builds shared space via CHA: REST connectivity -> SVD basis -> Procrustes alignment
4. Pools training data in shared k-dimensional space
5. Trains ridge regression encoder (stimulus features -> shared space)
6. Saves all model artifacts with provenance metadata

### Key config sections used

| Config Key | Default | Description |
|------------|---------|-------------|
| `subjects.train` | `[1, 2, 5]` | Training subjects |
| `subjects.test` | `[7]` | Test subject (for atlas harmonization) |
| `alignment.n_components` | `50` | Target dimensionality k |
| `alignment.min_k` | `10` | Minimum k (fail-fast) |
| `alignment.connectivity_mode` | `parcellation` | `parcellation` or `voxel_correlation` |
| `alignment.experiment_mode` | `hybrid_cha` | `hybrid_cha` or `strict_rest_cha` |
| `alignment.atlas_type` | `combined_rois` | `combined_rois`, `kastner`, `prf_visualrois` |
| `alignment.ensemble_method` | `average` | How to combine multi-run REST connectivity |
| `alignment.max_iters` | `10` | Procrustes iteration limit |
| `alignment.tol` | `1e-5` | Procrustes convergence tolerance |
| `encoding.ridge_alpha` | `100.0` | Ridge regression regularization |
| `features.type` | `clip` | Feature type to train on |
| `random_seed` | `42` | Global seed |

### Outputs

Saved to `outputs/shared_space/`:

| File | Description |
|------|-------------|
| `builder.npz` | SharedSpaceBuilder state (bases, rotations, template) |
| `encoder.npz` | Ridge encoder weights (W, b, standardization params) |
| `atlas_info.npz` | Harmonized atlas labels, parcel count |
| `atlas_masked_{sub}.npy` | Per-subject atlas within mask (for all subjects including test) |
| `metadata.json` | Provenance: timestamp, config hash, k_global, etc. |

---

## 8. Step 6 — Predict

**Script:** `src/pipelines/predict_subject.py`

Predict task activation for the test subject. Supports zero-shot (REST only) and few-shot (REST + some task examples).

### Zero-Shot Prediction

Uses only the test subject's resting-state data to align to the shared space.

```bash
python -m src.pipelines.predict_subject --mode zero_shot

# Custom options
python -m src.pipelines.predict_subject \
    --mode zero_shot \
    --test-sub 7 \
    --model-dir outputs/shared_space \
    --data-root processed_data \
    --feature-type clip \
    --output-dir outputs/predictions
```

### Few-Shot Prediction

Uses N shared-stimuli responses from the test subject for Procrustes alignment.

```bash
# 100-shot prediction
python -m src.pipelines.predict_subject --mode few_shot --n-shots 100

# With encoder fine-tuning
python -m src.pipelines.predict_subject --mode few_shot --n-shots 100 --fine-tune

# Custom seed for reproducible random split
python -m src.pipelines.predict_subject --mode few_shot --n-shots 250 --seed 123
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | str | **required** | `zero_shot` or `few_shot` |
| `--test-sub` | int | `7` | Test subject ID |
| `--n-shots` | int | `100` | Number of shared stimuli for few-shot alignment |
| `--model-dir` | str | `outputs/shared_space` | Path to trained model |
| `--data-root` | str | `processed_data` | Path to processed data |
| `--feature-type` | str | `clip` | Feature type. Options: `clip`, `dinov2` |
| `--fine-tune` | flag | off | Fine-tune encoder on few-shot data (few-shot only) |
| `--seed` | int | `42` | Random seed for few-shot split |
| `--output-dir` | str | `outputs/predictions` | Output directory |

### Outputs

Saved to `outputs/predictions/`:

**Zero-shot:**

| File | Description |
|------|-------------|
| `zeroshot_sub7_pred.npy` | Predicted voxel responses `(N_test, V_sub)` |
| `zeroshot_sub7_corrs.npy` | Per-voxel correlation with ground truth `(V_sub,)` |
| `zeroshot_sub7_metrics.json` | Evaluation metrics |

**Few-shot:**

| File | Description |
|------|-------------|
| `fewshot_sub7_N100_seed42_pred.npy` | Predicted voxel responses for held-out stimuli |
| `fewshot_sub7_N100_seed42_metrics.json` | Evaluation metrics |

### Metrics reported

| Metric | Description |
|--------|-------------|
| `median_r` | Median voxelwise Pearson r across all voxels |
| `mean_r` | Mean voxelwise Pearson r |
| `median_pattern_r` | Median pattern correlation (per-stimulus) |
| `two_vs_two` | 2-vs-2 identification accuracy (chance = 50%) |
| `noise_ceiling_median` | Median noise ceiling from split-half (zero-shot only) |
| `n_voxels` | Number of voxels |
| `n_stimuli` | Number of test stimuli |

---

## 9. Step 7 — Run Ablations

**Script:** `src/pipelines/run_ablations.py`

Sweeps over different few-shot sample sizes with repeated random splits.

### Command

```bash
# Run with config defaults
python -m src.pipelines.run_ablations

# Custom paths
python -m src.pipelines.run_ablations \
    --config config.yaml \
    --model-dir outputs/shared_space \
    --data-root processed_data \
    --output-dir outputs/ablations
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | `config.yaml` | Config file |
| `--model-dir` | str | `outputs/shared_space` | Path to trained model |
| `--data-root` | str | `processed_data` | Path to processed data |
| `--output-dir` | str | `outputs/ablations` | Output directory |
| `--n-bootstrap` | int | `5000` | Bootstrap resamples for confidence intervals |
| `--ci-level` | float | `95.0` | Confidence interval level (percent) |
| `--n-permutations` | int | `10000` | Permutations for significance tests |
| `--permutation-metric` | str | `median_r` | Metric used for permutation tests |
| `--permutation-reference-condition` | int | `0` | Preferred reference condition for permutation tests |
| `--fdr-alpha` | float | `0.05` | BH-FDR alpha for multiple-comparison correction |

### Config options used

| Config Key | Default | Description |
|------------|---------|-------------|
| `evaluation.fewshot_shots` | `[0, 10, 25, 50, 100, 250, 500, 750]` | Shot counts to sweep |
| `fewshot_n_repeats` | `5` | Random splits per shot count |
| `random_seed` | `42` | Base seed (incremented per repeat) |
| `evaluation.statistics.ci_level` | `95.0` | Confidence interval level |
| `evaluation.statistics.n_bootstrap` | `5000` | Bootstrap resamples |
| `evaluation.statistics.n_permutations` | `10000` | Permutation count |
| `evaluation.statistics.permutation_metric` | `median_r` | Metric for permutation p-values |
| `evaluation.statistics.permutation_reference_condition` | `0` | Preferred reference condition |
| `evaluation.statistics.fdr_alpha` | `0.05` | BH-FDR threshold |

### How it works

- `N=0` runs zero-shot (single run, no random split).
- For each `N > 0`, runs `n_repeats` random splits with seeds `base_seed, base_seed+1, ...`.
- Reports mean and standard deviation of median voxelwise correlation across repeats.
- Computes bootstrap confidence intervals per condition for key metrics.
- Computes permutation-test p-values for condition improvements vs a reference condition (prefers `N=0`; falls back to first condition with >=2 repeats).
- Applies Benjamini-Hochberg FDR correction across permutation tests and reports q-values/significance flags.

### Outputs

Saved to `outputs/ablations/fewshot/`:

| File | Description |
|------|-------------|
| `fewshot_ablation.json` | Summary: per-N median_r, std, all repeat metrics |
| `fewshot_summary.csv` | Per-condition table with mean/std/CI, permutation p-values, BH q-values, and reject flags |
| `fewshot_statistics.json` | Statistical settings + structured summary rows |
| `fewshot_sub7_N{X}_seed{Y}_pred.npy` | Individual predictions |
| `fewshot_sub7_N{X}_seed{Y}_metrics.json` | Individual metrics |

### Optuna Sweep (LOSO, Ridge + Alignment)

**Script:** `src/pipelines/sweep_shared_space_optuna.py`

This runs an Optuna study over:

- `encoding.ridge_alpha` (log-scale)
- `alignment.n_components` (integer step)

Objective is mean LOSO zero-shot `median_r` across `subjects.train`.

```bash
# Full sweep (uses config defaults under sweep.shared_space)
python -m src.pipelines.sweep_shared_space_optuna

# Dry-run validation only (folds + search-space + settings)
python -m src.pipelines.sweep_shared_space_optuna --dry-run

# Typical custom run
python -m src.pipelines.sweep_shared_space_optuna \
    --study-name ridge_align_loso_v1 \
    --n-trials 40 \
    --alpha-min 1e-2 \
    --alpha-max 1e5 \
    --ncomp-min 20 \
    --ncomp-max 120 \
    --ncomp-step 5 \
    --fixed-eval-size 250 \
    --eval-split-seed 42
```

TensorBoard:

```bash
tensorboard --logdir outputs/hparam_sweeps/shared_space/tensorboard
```

Optional Slurm wrapper:

```bash
sbatch slurm_scripts/04b_optuna_sweep_job.sh
```

Key CLI flags:

| Flag | Type | Description |
|------|------|-------------|
| `--study-name` | str | Study name used for Optuna storage + TensorBoard run grouping |
| `--n-trials` | int | Number of trials |
| `--timeout` | int | Optional timeout (seconds) |
| `--storage` | str | Optuna storage URL or local DB path |
| `--seed` | int | Sampler seed |
| `--alpha-min` / `--alpha-max` | float | Ridge alpha search bounds |
| `--ncomp-min` / `--ncomp-max` / `--ncomp-step` | int | `n_components` integer search grid |
| `--fixed-eval-size` | int | Fixed held-out eval rows per fold subject |
| `--eval-split-seed` | int | Eval split seed |
| `--dry-run` | flag | Validate sweep setup without launching trials |
| `--no-cleanup-trial-artifacts` | flag | Keep per-trial fold artifacts |
| `--no-retrain-best` | flag | Skip final retrain on full train subjects |

---

## 10. Step 8 — Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only unit tests (shapes, safety checks)
python -m pytest tests/test_shapes.py -v

# Run only integration tests (end-to-end with synthetic data)
python -m pytest tests/test_pipeline.py -v

# Run a specific test
python -m pytest tests/test_shapes.py::TestSVDBasis::test_parcellation_shape -v
```

### Test inventory

**`tests/test_shapes.py`** — 16 unit tests:

| Class | Test | What it checks |
|-------|------|----------------|
| `TestSVDBasis` | `test_parcellation_shape` | P shape (V, k) with rank clamping |
| | `test_voxel_correlation_shape` | P shape from V x V connectivity |
| | `test_min_k_raises` | Fail-fast when k < min_k |
| | `test_basis_orthogonal_columns` | P columns orthonormal |
| `TestProcrustes` | `test_rotation_orthogonal` | R @ R.T = I |
| | `test_nan_raises` | NaN input raises ValueError |
| | `test_constant_column_raises` | Near-constant column raises ValueError |
| `TestConnectivity` | `test_parcellation_output_shape` | C shape (R, V) |
| | `test_voxel_correlation_output_shape` | C shape (V, V) |
| `TestEncoder` | `test_fit_predict_shapes` | W, b, Z_pred shapes |
| | `test_predict_voxels_shape` | Y_pred shape (N, V) |
| `TestMetrics` | `test_voxelwise_correlation_shape` | Returns (V,), all finite |
| | `test_voxelwise_perfect_correlation` | Identical input gives r = 1 |
| | `test_zero_variance_voxel` | Zero-variance voxels return 0, not NaN |
| | `test_pattern_correlation_shape` | Returns (N,) |
| | `test_noise_ceiling_shape` | Returns (V,) in [0, 1] |

**`tests/test_pipeline.py`** — 3 integration tests:

| Test | What it checks |
|------|----------------|
| `test_shared_space_alignment` | CHA alignment recovers cross-subject structure |
| `test_encoder_prediction` | Encoder learns in shared space |
| `test_shared_space_builder_roundtrip` | Save/load preserves all state |

**`tests/test_statistics.py`** — 4 unit tests:

| Test | What it checks |
|------|----------------|
| `test_single_value_returns_degenerate_interval` | Bootstrap CI for singleton sample |
| `test_mean_is_inside_interval` | Bootstrap CI sanity on simple sample |
| `test_detects_large_difference` | Permutation test detects large effect |
| `test_builds_rows_and_metric_columns` | Condition summary includes CI + p-value fields |

---

## 11. Configuration Reference

All settings live in `config.yaml`. The full file:

```yaml
subjects:
  train: [1, 2, 5]
  test: [7]
data_root: "."
output_root: "outputs"

features:
  type: "clip"              # clip, dinov2, clip_dinov2
  clip_model: "ViT-L/14"
  dinov2_model: "dinov2_vitl14"
  batch_size: 64
  device: "cuda"

random_seed: 42
fewshot_n_repeats: 5

rest_preprocessing:
  discard_initial_trs: 5
  detrend: true
  highpass_cutoff_hz: 0.01
  motion_censoring:
    enabled: true
    fd_threshold_mm: 0.5
    max_censored_fraction: 0.3
  nuisance_regression: false
  zscore: true
  min_usable_trs: 100

alignment:
  n_components: 50
  min_k: 10
  connectivity_mode: "parcellation"    # parcellation (required for zero-shot) or voxel_correlation
  experiment_mode: "hybrid_cha"        # hybrid_cha or strict_rest_cha
  atlas_type: "combined_rois"          # combined_rois, kastner, prf_visualrois
  atlas_harmonize: "intersection"
  ensemble_method: "average"
  max_iters: 10
  tol: 1.0e-5

compute:
  dtype: "float32"
  use_randomized_svd: true
  randomized_svd_oversampling: 10
  memmap_threshold_gb: 2.0

encoding:
  model_type: "ridge"
  ridge_alpha: 100.0
  hidden_dim: 2048
  dropout: 0.3
  learning_rate: 1.0e-4
  epochs: 100
  batch_size: 256

evaluation:
  metrics: ["voxelwise_correlation", "pattern_correlation", "two_vs_two"]
  noise_ceiling_method: "split_half"
  reliability_thresholds: [0.0, 0.1, 0.3]
  nc_floor: 0.1
  fewshot_shots: [0, 10, 25, 50, 100, 250, 500, 750]

sweep:
  shared_space:
    study_name: "shared_space_optuna"
    output_dir: "outputs/hparam_sweeps/shared_space"
    n_trials: 40
    timeout: null
    sampler_seed: 42
    cleanup_trial_artifacts: true
    retrain_best: true
    pruner:
      type: "median"
      n_startup_trials: 5
      n_warmup_steps: 2
      interval_steps: 1
    search_space:
      ridge_alpha:
        min: 1.0e-2
        max: 1.0e5
        log: true
      n_components:
        min: 20
        max: 120
        step: 5
```

---

## 12. Directory Structure

After running the full pipeline:

```
resting_prediction/
  config.yaml
  requirements.txt
  download_nsddata.py
  how_to_use.md

  src/
    data/
      prepare_task_data.py
      prepare_rest_data.py
      prepare_features.py
      nsd_loader.py
      load_atlas.py
    alignment/
      utils.py
      rest_preprocessing.py
      shared_space.py
      cha_alignment.py
    models/
      encoding.py
      baseline_mni.py
    evaluation/
      metrics.py
      visualize.py
    pipelines/
      train_shared_space.py
      predict_subject.py
      run_ablations.py

  tests/
    conftest.py
    test_shapes.py
    test_pipeline.py

  nsddata/                      # [downloaded] raw NSD metadata + ROIs
  nsddata_betas/                # [downloaded] task betas per subject
  nsddata_stimuli/              # [downloaded] 73k stimulus images
  nsddata_timeseries/           # [downloaded] REST timeseries

  processed_data/               # [generated] preprocessed arrays
    subj01/
      train_fmri.npy
      test_fmri.npy
      train_stim_idx.npy
      test_stim_idx.npy
      test_fmri_trials.npy
      test_trial_labels.npy
      mask.npy
      rest_run1.npy
      rest_run2.npy
      rest_run_manifest.json
    subj02/ ...
    subj05/ ...
    subj07/ ...
    features/
      clip_features.npy
      dinov2_features.npy

  outputs/                      # [generated] model + predictions
    shared_space/
      builder.npz
      encoder.npz
      atlas_info.npz
      atlas_masked_1.npy
      atlas_masked_2.npy
      atlas_masked_5.npy
      atlas_masked_7.npy
      metadata.json
    predictions/
      zeroshot_sub7_pred.npy
      zeroshot_sub7_corrs.npy
      zeroshot_sub7_metrics.json
      fewshot_sub7_N100_seed42_pred.npy
      fewshot_sub7_N100_seed42_metrics.json
    ablations/
      fewshot/
        fewshot_ablation.json
```

---

## 13. Troubleshooting

**"No beta session files found"**
Run `download_nsddata.py` first. Check that `nsddata_betas/ppdata/subj{XX}/func1pt8mm/betas_fithrf_GLMdenoise_RR/` contains `betas_session*.nii.gz`.

**"found N REST runs, need >= 2"**
Subject needs at least 2 REST runs. Check `nsddata_timeseries/ppdata/subj{XX}/func1pt8mm/timeseries/` and the REST manifest at `processed_data/subj{XX}/rest_run_manifest.json`.

**"k_actual < min_k"**
The number of usable SVD components is too low. This happens when the atlas has very few parcels. Try increasing `n_components` or using a richer atlas (`combined_rois` instead of `kastner`).

**Out of memory during beta loading**
Each session loads ~1.8 mm NIfTI volume. Process one subject at a time. Consider closing other applications.

**CUDA out of memory during feature extraction**
Reduce `--batch-size` (e.g., `--batch-size 16`) or use `--device cpu`.

**pytest not found**
```bash
pip install pytest
```

**AWS download fails**
Check AWS CLI is installed and the bucket is accessible:
```bash
aws s3 ls s3://natural-scenes-dataset/ --no-sign-request
```
