# Analysis Mask A/B Runs

This note explains how to compare the current `nsdgeneral` voxel mask against a stricter parcel-labeled mask.

## Pipeline A: Current Behavior

Pipeline A keeps all voxels in NSD's `nsdgeneral` mask. This is the default:

```yaml
analysis_mask:
  mode: "nsdgeneral"
  atlas_type: "combined_rois"
  use_common_labels: true
```

Run it into its own folders:

```bash
CONFIG_PATH=config.yaml \
OUTPUT_ROOT=processed_data_nsdgeneral \
DATA_ROOT=processed_data_nsdgeneral \
MODEL_DIR=outputs/shared_space_nsdgeneral \
PREDICTION_DIR=outputs/predictions_nsdgeneral \
ABLATION_DIR=outputs/ablations_nsdgeneral \
bash slurm_scripts/submit_full_pipeline.sh
```

## Pipeline B: Atlas-Labeled Voxels Only

Pipeline B keeps only voxels that are both:

- inside `nsdgeneral`
- assigned to a common nonzero atlas/ROI label

Create a copy of `config.yaml`, for example `config_labeled_only.yaml`, and set:

```yaml
analysis_mask:
  mode: "atlas_labeled_only"
  atlas_type: "combined_rois"
  use_common_labels: true
```

`atlas_type` must match `alignment.atlas_type`. With `use_common_labels: true`, preprocessing uses the same cross-subject common-label logic as shared-space training.

Run it into separate folders:

```bash
CONFIG_PATH=config_labeled_only.yaml \
OUTPUT_ROOT=processed_data_labeled_only \
DATA_ROOT=processed_data_labeled_only \
MODEL_DIR=outputs/shared_space_labeled_only \
PREDICTION_DIR=outputs/predictions_labeled_only \
ABLATION_DIR=outputs/ablations_labeled_only \
bash slurm_scripts/submit_full_pipeline.sh
```

## What To Compare

The prediction metrics now include atlas split summaries:

- `atlas_labeled_n_voxels`
- `atlas_labeled_median_r`
- `atlas_labeled_mean_r`
- `atlas_unlabeled_n_voxels`
- `atlas_unlabeled_median_r`
- `atlas_unlabeled_mean_r`

For Pipeline B, unlabeled voxel counts should be zero because those voxels are removed during preprocessing.

Preprocessing also writes mask summaries per subject:

- `processed_data*/subjXX/analysis_mask_summary.json`
- `processed_data*/subjXX/rest_analysis_mask_summary.json`

These files report how many `nsdgeneral` voxels were retained and how many unlabeled voxels were dropped.
