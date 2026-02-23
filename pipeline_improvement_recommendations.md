# Pipeline Improvement Roadmap

## Scope
This roadmap prioritizes measurable gains in activation prediction and reconstruction quality, while preventing misleading comparisons from data-row mismatches.

## Status Snapshot (2026-02-23)
- ✅ Completed: fixed eval split protocol, eval-index persistence, row-contract validation in core train/predict, atlas utilization reporting, feature-backbone sweep, multi-seed CI/permutation summaries.
- 🟡 Partially completed: reconstruction alignment transparency and row handling (alignment metadata exists; prefix fallback still allowed in VDVAE+VD benchmark).
- 🔜 Next highest leverage: reliability/ROI-aware encoding improvements.

## Baseline (Current)
- Zero-shot `median_r ~= 0.118` with noise ceiling `~= 0.257` (about `46%` of ceiling).
- Best few-shot `median_r ~= 0.169` (about `66%` of ceiling).
- Main bottleneck remains fMRI-to-feature prediction quality, not pipeline execution.

## Phase A: Stability And Data-Alignment Guardrails
1. ✅ Lock evaluation protocol. **Done**
- One fixed held-out eval row set is used across `N_shots` and seeds.
- Eval row indices are persisted in artifacts and written to metrics.

2. 🟡 Enforce train/test row contracts between `processed_data` and brain-diffuser features. **Mostly done**
- Core train/predict paths now fail fast on row/voxel mismatches.
- Remaining gap: reconstruction benchmark still allows `prefix` fallback when external feature rows mismatch.
- Action: remove fallback and require exact stimulus-index alignment for all benchmark feature inputs.

3. ✅ Make alignment status explicit in outputs. **Done (core + benchmark)**
- Alignment metadata (`*_rows_used`, `*_alignment_mode`) is written in reconstruction summaries.
- Non-identity alignment is visible and tracked.

4. ✅ Improve run reproducibility. **Done**
- Seed handling is standardized across prediction/ablation flows.
- Aggregate reporting includes bootstrap CI and permutation statistics across repeats.

5. ✅ Add atlas-utilization guardrails (within `nsdgeneral` mask). **Done**
- `atlas_utilization_report.json` is saved during shared-space training.
- Shape/voxel consistency checks fail fast before fitting.
- Atlas coverage metrics are included in prediction outputs.

## Phase B: Activation Prediction Improvements (Highest Performance Leverage)
1. ⏳ Reliability-weighted regression.
- Weight voxels/parcels by split-half reliability or noise ceiling.
- Target: improve median correlation in reliable voxels without degrading whole-brain median.

2. ✅ Feature backbone comparison. **Implemented; keep running as benchmark gate**
- Compare `clip`, `dinov2`, and `clip_dinov2` with identical splits and metrics.
- Select one default feature set before further architecture changes.

3. ⏳ ROI-wise regularization.
- Tune `alpha` per ROI (instead of one global value).
- Prioritize early visual vs higher-order regions separately.

4. ⏳ ROI-expert mappings.
- Train specialized regressors for ROI groups and concatenate outputs.
- Keep a global baseline model for sanity comparison.

5. ⏳ Few-shot adaptation with zero-shot prior.
- Replace independent few-shot fitting with shrinkage toward zero-shot weights.
- Especially important for low-`N` regimes where variance is high.

## Phase C: Evaluation And Reporting Upgrades
1. 🟡 Add ceiling-normalized metrics by ROI and voxel reliability bins. **Partially done**
- Report both raw `r` and normalized `r / ceiling`.
- Current status: reliability-stratified raw metrics and noise ceiling summaries exist; ROI-wise normalized reporting is still pending.

2. ✅ Keep condition-level reconstruction diagnostics. **Done**
- For each condition (`gt_fmri`, `zero_shot`, `few_shot`), track feature-space `R2` and alignment metadata.
- Separate "model quality" from "data-coverage limitations" in interpretations.

3. ✅ Track failure modes explicitly. **Mostly done**
- Cases where few-shot underperforms zero-shot (especially low `N`) should be reported with split variance context, not treated as anomalies.
- Current reporting includes repeat distributions and inferential statistics; continue highlighting low-`N` instability in summaries.

## Why Few-Shot N=10 Can Underperform Zero-Shot
- Small support sets are high-variance and sensitive to split choice.
- Uniform regularization can over-shrink informative regions or under-shrink noisy ones.
- Performance usually stabilizes after fixed eval splits, reliability weighting, and ROI-wise hyperparameters.

## Recommended Execution Order
1. Remove `prefix` row-alignment fallback in reconstruction benchmarks (require stimulus-index alignment).
2. Add ROI-wise alpha tuning.
3. Add reliability-weighted regression.
4. Add zero-shot-prior few-shot adaptation.
5. Re-run feature-backbone sweep as a regression check (`clip`, `dinov2`, `clip_dinov2`).
6. Re-run VDVAE+VD recon benchmark and compare feature-space `R2` + panel quality.

## Acceptance Criteria
1. Every run records exact eval rows and alignment mode in outputs.
2. Few-shot and zero-shot comparisons use identical held-out rows.
3. At least one model update improves median `r` and ceiling-normalized `r` over current baseline.
4. Improvements are consistent across multiple seeds, not a single split artifact.
5. Atlas utilization is reported for all subjects and no shape-mismatch guardrail is triggered.
6. Reconstruction benchmarks no longer use degraded `prefix` row matching.
