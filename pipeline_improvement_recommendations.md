# Pipeline And Activation Prediction Improvements

## Current Status
- Zero-shot `median_r ~= 0.118` with noise ceiling `~= 0.257` (about 46% of ceiling).
- Best few-shot `median_r ~= 0.169` (about 66% of ceiling).
- The pipeline is working; the activation mapping is the main bottleneck.

## Highest-Impact Improvements (General Pipeline)
1. Fix evaluation protocol first: use one fixed held-out eval set across all `N_shots` so ablations are directly comparable.
2. Add ceiling-normalized metrics (`r / noise_ceiling`) per ROI and per voxel group, not only global medians.
3. Add more source subjects to shared-space training if available (current train set is small).
4. Tune shared-space hyperparameters with LOSO CV: `k`, connectivity mode, atlas choice, parcel thresholds.
5. Increase reproducibility: multiple seeds everywhere and report mean/CI for zero-shot too.

## Highest-Impact Improvements (Activation Prediction Model)
1. Reliability-weighted regression: weight voxels/parcels by split-half reliability (or noise ceiling).
2. Better stimulus features: compare `clip`, `dinov2`, and concatenated `clip_dinov2` for the encoder target mapping.
3. Regularization sweep per ROI (not one global alpha): visual areas often need different shrinkage.
4. ROI-expert models: separate mappings for early visual vs higher-level regions, then concatenate outputs.
5. Few-shot adaptation with stronger priors: shrink toward zero-shot weights (instead of independent fit).

## Why Few-Shot N=10 Can Be Worse Than Zero-Shot
- Very small support set + random split variance + strong regularization mismatch.
- This usually improves after fixed eval sets, reliability weighting, and ROI-wise alphas.

## Recommended Experiment Order (No Code Changes Yet)
1. Lock evaluation split and rerun ablation table.
2. Run feature comparison (`clip` vs `dinov2` vs combined).
3. Run ROI-wise alpha tuning.
4. Run reliability-weighted model.
5. Recheck zero-shot/few-shot gains and ceiling-normalized improvement.
