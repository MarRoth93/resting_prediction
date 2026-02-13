# Pipeline Improvement Roadmap

## Scope
This roadmap prioritizes measurable gains in activation prediction and reconstruction quality, while preventing misleading comparisons from data-row mismatches.

## Baseline (Current)
- Zero-shot `median_r ~= 0.118` with noise ceiling `~= 0.257` (about `46%` of ceiling).
- Best few-shot `median_r ~= 0.169` (about `66%` of ceiling).
- Main bottleneck remains fMRI-to-feature prediction quality, not pipeline execution.

## Phase A: Stability And Data-Alignment Guardrails (Do First)
1. Lock evaluation protocol.
- Use one fixed held-out eval row set across all `N_shots` and seeds.
- Store eval row indices in artifacts so reruns are byte-for-byte comparable.

2. Enforce train/test row contracts between `processed_data` and brain-diffuser features.
- Current warning pattern (`train_fmri=9000` vs feature train rows `8859`, test feature rows `982`) shows this is an active risk.
- Prefer exact alignment by stimulus index; only use prefix fallback as temporary degraded mode.

3. Make alignment status explicit in outputs.
- Keep writing alignment metadata (`*_rows_used`, `*_alignment_mode`) in `summary.json`.
- Treat non-identity alignment as a tracked experimental condition, not silent default.

4. Improve run reproducibility.
- Standardize seed handling across prediction, ablation, and reconstruction.
- Report mean and confidence interval across seeds for zero-shot and few-shot.

5. Add atlas-utilization guardrails (within `nsdgeneral` mask).
- Save `atlas_utilization_report.json` during shared-space training with per-subject:
  labeled voxel fraction, parcel-count utilization, and parcel-size statistics.
- Fail fast on atlas/mask/task/REST voxel-count mismatches before model fitting.
- Include atlas coverage stats in prediction metrics (e.g., labeled fraction and parcels present).

## Phase B: Activation Prediction Improvements (Highest Performance Leverage)
1. Reliability-weighted regression.
- Weight voxels/parcels by split-half reliability or noise ceiling.
- Target: improve median correlation in reliable voxels without degrading whole-brain median.

2. Feature backbone comparison.
- Compare `clip`, `dinov2`, and `clip_dinov2` with identical splits and metrics.
- Select one default feature set before further architecture changes.

3. ROI-wise regularization.
- Tune `alpha` per ROI (instead of one global value).
- Prioritize early visual vs higher-order regions separately.

4. ROI-expert mappings.
- Train specialized regressors for ROI groups and concatenate outputs.
- Keep a global baseline model for sanity comparison.

5. Few-shot adaptation with zero-shot prior.
- Replace independent few-shot fitting with shrinkage toward zero-shot weights.
- Especially important for low-`N` regimes where variance is high.

## Phase C: Evaluation And Reporting Upgrades
1. Add ceiling-normalized metrics by ROI and voxel reliability bins.
- Report both raw `r` and normalized `r / ceiling`.

2. Keep condition-level reconstruction diagnostics.
- For each condition (`gt_fmri`, `zero_shot`, `few_shot`), track feature-space `R2` and alignment metadata.
- Separate "model quality" from "data-coverage limitations" in interpretations.

3. Track failure modes explicitly.
- Cases where few-shot underperforms zero-shot (especially low `N`) should be reported with split variance context, not treated as anomalies.

## Why Few-Shot N=10 Can Underperform Zero-Shot
- Small support sets are high-variance and sensitive to split choice.
- Uniform regularization can over-shrink informative regions or under-shrink noisy ones.
- Performance usually stabilizes after fixed eval splits, reliability weighting, and ROI-wise hyperparameters.

## Recommended Execution Order
1. Freeze eval split and alignment contract; rerun baseline table.
2. Run feature backbone comparison (`clip`, `dinov2`, `clip_dinov2`).
3. Add ROI-wise alpha tuning.
4. Add reliability-weighted regression.
5. Add zero-shot-prior few-shot adaptation.
6. Re-run VDVAE+VD recon benchmark and compare feature-space `R2` + panel quality.

## Acceptance Criteria
1. Every run records exact eval rows and alignment mode in outputs.
2. Few-shot and zero-shot comparisons use identical held-out rows.
3. At least one model update improves median `r` and ceiling-normalized `r` over current baseline.
4. Improvements are consistent across multiple seeds, not a single split artifact.
5. Atlas utilization is reported for all subjects and no shape-mismatch guardrail is triggered.
