# NSD-first improvement plan (voxel-level), with deferred FOR requirements

**Status:** proposed, nothing implemented.
**Goal:** get genuinely good image→brain→image results on NSD alone, at voxel
resolution. FOR is deferred, but every FOR prerequisite is recorded here so no
NSD decision silently forecloses it.
**Authored:** 2026-07-31, after a Claude↔Codex design review.

---

## 0. Where we actually stand (measured, not assumed)

| Thing | State | Evidence |
|---|---|---|
| Frozen voxel encoder | mean LOSO median voxel r **0.17818** (seeds 42–46) vs ridge 0.17432 | `artifacts/results.json` |
| NSD voxel targets | `nsdgeneral` mask, 12,682–17,907 voxels/subject (mean 14,888) | `data/processed/subj0*/train_fmri.npy` |
| NSD REST budget | 4,398–7,956 TRs/subject (20–36 runs) | counted from `rest_run*.npy` |
| Seed bank | 499 NSD-ROI seeds | `artifacts/model/external_seed_info.json` |
| Parcel (FOR-compat) encoder | best val 0.12621, peaks epoch 9 then decays to 0.083 by 29 | `schaefer400_multiexpert/model/encoder/training_state.json` |
| Parcel→VDVAE decoder | **0.0464 vs 0.0468 mean-latent baseline** — below baseline; alpha 200,000 = grid max | `vdvae_decoder_validation/validation_summary.json` |
| Assessor preservation | original↔recon r ∈ [−0.036, +0.034], OOD pct 100 | `for_assessor_study/validation/assessor_preservation.json` |
| Image-space metrics | **not implemented** — only latent-space r/R² exist | `benchmark_reconstructions_vdvae_vd.py:308,954-956` |

### 0.1 Phase 0 prediction results (measured 2026-07-31, subject 7)

| Metric | Zero-shot | 100-shot |
|---|---:|---:|
| Median voxel r | 0.1730 | 0.1774 |
| Mean voxel r | 0.1838 | 0.1914 |
| Median pattern r | 0.5584 | 0.5750 |
| 2-vs-2 accuracy | 0.967 | 0.984 |

12,682 voxels, fixed 200-image eval split, seed 42. Two findings:

- **100 real task responses add only +0.0044 median r over pure REST alignment.**
  The pipeline's central thesis holds — REST-derived alignment does nearly all the
  work that 100 labeled examples would. This is also what makes a zero-shot-only
  dataset like FOR coherent in principle.
- Stage A is not the broken part. The encoder generalizes to an unseen subject
  consistently with its 0.178 LOSO benchmark.

### 0.2 Noise-ceiling headroom (corrected)

`noise_ceiling_split_half` (`src/evaluation/metrics.py:126-168`) returns a
Spearman-Brown-corrected split-half **reliability** ρ — a variance ratio. The
maximum correlation a *perfect* model can reach against a noisy target is **√ρ**,
not ρ. Comparing r to ρ directly overstates achievement by ~2x.

| Voxel subset | n | median r | √ρ = max achievable | achieved |
|---|---:|---:|---:|---:|
| all | 12,682 | 0.1730 | 0.5065 | **34.2%** |
| nc ≥ 0.1 | 11,138 | 0.1950 | 0.5290 | 36.9% |
| nc ≥ 0.3 | 4,932 | 0.2899 | 0.6228 | 46.5% |
| nc ≥ 0.5 | 761 | 0.4114 | 0.7457 | 55.2% |

**There is ~3x headroom, not 1.5x.** Report the √ρ-normalized figure from now on;
the raw r/ρ ratio is not a meaningful "percent of ceiling".

### 0.3 Phase 0 reconstruction results — **GATE 0 FAILED** (measured 2026-07-31)

`artifacts/reconstructions/subj07/summary.json`. Latent R² and per-dimension
correlation against true latents on the 200-image eval split:

| Condition | VDVAE R² | VDVAE target_r | CLIP-text R² | CLIP-text target_r | CLIP-vis R² | CLIP-vis target_r |
|---|---:|---:|---:|---:|---:|---:|
| **gt_fmri (measured!)** | **−0.0249** | **0.0013** | +0.1127 | 0.3403 | +0.0385 | 0.2061 |
| zero_shot | −0.0425 | 0.0002 | −0.0084 | 0.2081 | −0.0853 | 0.0515 |
| few_shot | −0.0744 | 0.0005 | +0.0036 | 0.2412 | −0.0936 | 0.1071 |

**The VDVAE arm is completely broken, even from measured fMRI.** `target_r` = 0.0013
means essentially no latent dimension is predicted. Predictions are shrunk 6.3x
(pred_std 0.0425 vs true_std 0.2690). Visual confirmation: the VDVAE-only
intermediate for the best-predicted row (`reconstructions_vdvae/gt_fmri/
row00849_stim61797.png`, act corr zero=0.401) is pure high-frequency grid noise. The
original stimulus is a bearded man with glasses in a car; no arm produces anything
face-like.

The CLIP arms partially work (CLIP-text target_r 0.34 from measured fMRI), but
Versatile Diffusion runs img2img at `strength 0.5` off the VDVAE output, so a noise
initialization poisons the whole chain regardless.

**Consequences that reorder this plan:**

1. **The parcel-track VDVAE failure was never mainly an ROI problem.** The identical
   failure occurs at voxel resolution (12,682 voxels), with measured responses, and
   9,000 training rows. Earlier attribution of 0.0464-vs-0.0468 to a 400-feature
   information ceiling was wrong.
2. **Stage A is healthy; stage C is broken upstream of everything.** No encoder,
   alignment, or FOR work can be evaluated end-to-end until this is fixed.
3. **All prior FOR reconstruction work sat on a chain that has never worked.**

Hypotheses to test, cheapest first (none yet confirmed):
- **Decode the TRUE test latents through VDVAE.** If true latents also render as
  noise, the VDVAE decode path / `ref_latents.npz` is broken and the regression is
  irrelevant. If they render correctly, the regression is at fault. This single test
  splits the problem in half — do it first.
- Per-dimension target standardization before ridge (features are z-scored,
  `_standardize_fmri:264-278`; confirm targets are too).
- Latent extraction correctness vs the brain-diffuser reference (layer set,
  ordering, normalization).
- Ruled out: `fmri_scale = 300.0` is a mathematical no-op — `_standardize_fmri`
  z-scores per column after dividing, so the constant cancels.

**Phase 0 outcome: do not proceed to Phases 2–3 until Gate 0 passes.** Phase 1
(image metrics) is still worth doing in parallel, since it is the scoreboard that
will tell you whether a fix actually worked. Add the ORIGINAL stimulus to the
comparison panels — the current panels omit it, which makes visual judgement
impossible.

---

## 1. Scope decision: NSD-only optimum ≠ FOR-compatible optimum

The 400-parcel representation was never a scientific choice. It existed to make NSD
and FOR commensurable, and it cost NSD its voxels. With FOR deferred, **four
FOR-compatibility compromises can be dropped for NSD work** — but each will have to
be re-imposed (at a measured cost) if/when FOR returns.

| Decision | NSD-only optimum | FOR-compatible version | Cost of the FOR version |
|---|---|---|---|
| Seed bank | full 499 NSD-ROI seeds | ≤416 FreeSurfer-derivable, or Schaefer-400 | drops 83 NSD-only localizer seeds (prf-*, floc-*, streams) |
| Target mask | full `nsdgeneral`, ~14,888 voxels | 72-parcel visual intersection | FOR gets only **~2,268** voxels there (measured; range 1,601–3,220) |
| Shared dim `k` | 100 is well-supported (4,400–7,956 TRs) | ~20–30 | FOR split-half subspace stability: **0.615 @ k=20, 0.528 @ k=50, 0.493 @ k=100** |
| Stage C decoder | subject-native ridge, own 9,000 rows | shared decoder in a common space (fsaverage/MNI) | needs surfaces + registrations FOR does not currently ship |

**Rule for this plan:** tag every change `[NSD-FREE]` (optimize freely) or
`[FOR-CRITICAL]` (a decision FOR will constrain). Log divergences in §7 as they
accumulate. Do not silently optimize a `[FOR-CRITICAL]` knob into a corner.

---

## 2. Phase 0 — Establish the real baseline (no new code)

Nothing is redesigned until we know what the existing chain produces.

1. `./run_pipeline.sh check`
2. `./run_pipeline.sh predict` → `artifacts/predictions/subj07/` (zero-shot + 100-shot)
3. `./run_pipeline.sh reconstruct` → `artifacts/reconstructions/subj07/`

Record, for ground-truth / zero-shot / few-shot separately: VDVAE, CLIP-text and
CLIP-vision R² against true latents, plus the image panels.

**Gate 0:** ground-truth-response reconstructions must be recognizable. If decoding
*measured* subject-7 voxels already fails, the problem is the decoder or the VDVAE
assets, not the encoder — and the whole plan re-roots there.

> Subject 7 has already been evaluated once (`artifacts/results.json`:
> `subject_7_already_evaluated: true`). Phase 0 is a **measurement**, not model
> selection. Do not tune anything against subject 7.

---

## 3. Phase 1 — Add image-space metrics `[NSD-FREE]`

Latent R² is not a result anyone will believe. Implement the standard
reconstruction battery in a new `src/evaluation/image_metrics.py`:

- low-level: PixCorr, SSIM
- high-level: AlexNet(2,5), Inception, CLIP, EfficientNet-B1, SwAV
- 2-way identification accuracy per metric

Wire into `benchmark_reconstructions_vdvae_vd.py` alongside the existing feature
metrics. Report ground-truth / zero-shot / few-shot side by side. This is the
scoreboard for every later phase; build it before optimizing against it.

**Gate 1:** ground-truth-response numbers land in the range the brain-diffuser
literature reports for NSD. If not, the decoder or assets are miscalibrated and
Phase 2+ is premature.

---

## 4. Phase 2 — Decoder factorial pilot `[NSD-FREE, informs FOR-CRITICAL]`

This is the decisive cheap experiment. It needs **no new data and no FOR access**,
and it settles which lever actually matters. Run on NSD only, subject- and
image-disjoint throughout.

Factors:

| Factor | Levels |
|---|---|
| Features | 400 parcels · full `nsdgeneral` voxels · 72-parcel-restricted voxels |
| Training rows | 5,767 (shared-1000 only) · ~52,264 (all subjects' own train images) |
| Response source | measured · outer-LOSO **predicted** |
| Regularization | ridge grid **extended past 200,000** · low-rank / reduced-rank regression |
| Baselines | mean-latent · **direct CLIP→VDVAE** (no brain at all) |

Two prerequisites:

1. **Extend the alpha grid.** The parcel decoder selected 200,000, the maximum
   tested. Any conclusion drawn at a grid boundary is unsafe.
2. **Unlock the rows.** VDVAE latents are a function of the *image only*;
   `train_latents (9000, 91168)` already sits unused in
   `data/processed/reconstruction_features/subj07/vdvae_features.npz`. Computing
   latents for subjects 1–6's own train images yields ~52,264 (subject, image)
   rows — >10× more — at ~19 GB and one GPU pass via
   `src/data/prepare_reconstruction_features.py`.

**Gate 2 (hard):** every configuration must beat *both* the mean-latent and the
direct-CLIP→VDVAE baseline. Beating mean-latent alone is not enough — if brain data
cannot outperform decoding the stimulus embedding directly, the brain is decorative.
The parcel track never cleared even the weaker bar.

**Deliverable:** a table attributing gains to features vs rows vs regularization.
This is what tells us whether the FOR-compatible 72-parcel restriction (~2,268
voxels) can ever work, before anyone requests FOR data.

---

## 5. Phase 3 — Stage A/B improvements `[mixed]`

Only after Phases 0–2 give a trustworthy scoreboard.

**3a. Principled `k` selection `[FOR-CRITICAL]`.** `compute_global_k`
(`src/alignment/utils.py:119`) checks only matrix shape — never numerical rank or
subspace stability. Add split-half subspace-similarity diagnostics and select `k`
by nested LOSO on subjects 1–6. NSD will likely support k=100; record the stability
curve anyway, because FOR will need ~20–30 and per-dataset `k` is **incompatible**
with the fixed encoder output dim.

**3b. Encoder capacity audit `[NSD-FREE]`.** With one `clip` stream, `encode` builds
only **2 tokens** (CLS + one stream, `nonlinear_encoding.py:204-214`), so the
8-layer/8-head stack attends over two positions — architecturally near an MLP.
Either add real streams (CLIP-text alongside CLIP-vision; `feature_slices` already
supports multi-stream) or shrink the backbone. Test against the 0.178 benchmark.

**3c. Investigate the parcel encoder's early collapse `[diagnostic]`.** Val peaks
at epoch 9 then decays monotonically to 0.083 by epoch 29 despite dropout 0.20 and
method_dropout 0.25. Determine whether this is parcel-specific (low signal) or a
general optimization fault that also afflicts the voxel encoder.

**Gate 3:** mean LOSO median voxel r must exceed the frozen **0.17818** on subjects
1–6, with all-seed-means-beat-ridge preserved. Do not touch subject 7.

### 5.1 Closing the 3x gap to √ρ — architectural levers

Ordered by expected gain. See §0.2 for why the target is √ρ ≈ 0.51, not ρ.

**A1. Spatial image features — the biggest single lever. `[NSD-FREE]`**
The encoder input is ONE globally pooled 768-d CLIP vector per image
(`encoder/metadata.json`: `input_dim: 768`, `feature_slices: {"clip": [0,768]}`).
Visual cortex is retinotopic: a voxel's response depends on *where* in the image
something appeared. Global pooling has already destroyed that information, so no
amount of model capacity downstream can recover it. `clipvision_train.npy` is
already **(9000, 257, 768)** — 256 patch tokens + CLS — in
`data/processed/reconstruction_features/subj07/`. Would need generating for
subjects 1–6 via `src/data/prepare_reconstruction_features.py`.

**A2. The 2-token degeneracy — fixed for free by A1. `[NSD-FREE]`**
With a single `clip` stream, `encode` builds only CLS + 1 stream token
(`nonlinear_encoding.py:204-214`), so the 8-layer/8-head stack attends over **2
positions** — architecturally an MLP with extra machinery. Patch tokens or extra
streams (CLIP-text, DINOv2) make the TRIBE design do what it was built for.

**A3-prior. FOR voxel data rescues the alignment from near-degeneracy (measured 2026-08-04).**

FOR voxel timeseries arrived (`/media/psycontrol/HDD/Datasets/FOR/voxel_timeseries/`,
50 subjects, validated — parcel-averaging reproduces the old parcel series for all
400 parcels). Split-half basis stability, 5 subjects, mean squared canonical
correlation between bases fit on each half-run:

| k | voxels (V=2,336) | chance | gain | visual parcels (V=72) | chance | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.554 | 0.004 | **129.5x** | 0.702 | 0.139 | 5.1x |
| 20 | 0.452 | 0.009 | **52.8x** | 0.678 | 0.278 | 2.4x |
| 50 | 0.304 | 0.021 | **14.2x** | 0.782 | 0.694 | 1.1x |
| 100 | 0.255 | 0.043 | **6.0x** | 1.000 | 1.000 | **1.0x** |

**The parcel representation was producing an illusion of stability.** Its raw numbers
look *higher* (0.68–1.00 vs 0.26–0.55) but its gain over chance is far *worse*. At
V=72 and k=100 the basis spans the entire space, so stability is trivially 1.000 and
carries zero alignment information. Codex's earlier 400-parcel figures
(0.615/0.528/0.493 at k=20/50/100) correspond to gains of only 12.3x/4.2x/2.0x once
chance (k/400) is divided out.

Consequence: **k=100 is defensible at voxel resolution** (6.0x above chance) where it
was degenerate at parcel level. The earlier conclusion that FOR forces k down to
~20–30 was an artifact of the parcel representation, not a property of the 237-TR
budget. k=20–50 remains stronger (14–53x) and is worth preferring, but the
NSD/FOR divergence on `k` is much smaller than feared.

Caveats: this is split-half (~118 TRs per half), so it *understates* full-run
(237 TR) stability. It measures within-subject replicability, which is necessary but
not sufficient for good cross-subject alignment. n=5 subjects.

**A3. The k=100 bottleneck. `[FOR-CRITICAL]`**
Everything funnels through a 100-D shared space. There are 4,932 voxels at nc≥0.3
whose reliable structure plausibly exceeds 100 dimensions. Sweep
k ∈ {100, 150, 200, 300}. Log in §7 — FOR can only support ~20–30, so a large NSD
optimum is a divergence to record, not a problem to hide.

**A4. Loss/metric mismatch. `[NSD-FREE]`**
Training minimizes MSE on standardized Z (`nonlinear_encoding.py:354`); evaluation
is per-voxel correlation across images. Add a correlation term, and weight voxels
by `ncsnr` so capacity goes where signal exists instead of fitting noise in
unreliable voxels.

**A5. The readout is strictly linear. `[breaks zero-shot — evaluate last]`**
`Z_hat @ R.T @ P.T` is linear from shared space to voxels
(`nonlinear_encoding.py:479-486`). A nonlinear per-voxel readout adds capacity but
requires per-subject fitting, which forfeits the zero-shot property that makes FOR
viable at all. Only pursue if you accept spending shots.

### 5.2 Closing the gap — data levers

**D1. Single-trial betas. `[NSD-FREE]`**
Training uses trial-averaged responses. Single-trial betas give ~3x the rows and are
standard for encoding models — noisier per row, but far more of them.

**D2. Subject 8 is missing. `[NSD-FREE, cheap]`**
Only subj01–07 are downloaded (verified under
`nsddata/ppdata/`). NSD has 8 subjects. Adding subject 8 gives ~9,000 more training
rows *and* a 7th subject for the shared space. Mechanical work via
`src/data/download_nsddata.py` + `prepare_task_data` + `prepare_rest_data`.

**D3. REST is unequal across subjects. `[NSD-FREE]`**
Subjects 1 and 5 have 36 runs (~7,900 TRs); subjects 2, 3, 4, 6, 7 have only 20
(~4,400). More REST directly improves `P` and `R`, lifting stages A and B together.
First test whether the 36-run subjects show better alignment — that quantifies the
payoff before downloading anything.

**D4. Fix / cross-check the noise-ceiling estimator. `[measurement]`**
With 3 trials per shared image, `trials[0::2]` takes 2 and `trials[1::2]` takes 1.
Spearman-Brown assumes EQUAL halves, so this unbalanced split depresses `r_half`
and likely underestimates ρ. NSD's official per-voxel `ncsnr.nii.gz` is already
downloaded and unused — cross-check against it. This changes the denominator for
every result, so do it early (fold into Phase 1).

### 5.3 Suggested order within Phase 3

1. **A1 + A2** — spatial features; biggest expected gain, features partly exist
2. **D4 + A4** — fix the measuring stick and reliability-weight the loss (Phase 1)
3. **A3** — k sweep
4. **D2 + D3** — subject 8 and extra REST; mechanical, parallelizable
5. **D1** — single-trial betas
6. **A5** — only if zero-shot can be sacrificed

Levers that CANNOT move the ceiling: better architecture, more training data, or
any modelling choice. ρ is a property of the measurement. The only genuine
ceiling-raisers would be more trial repeats (NSD has none to give — all 1,000 shared
images have exactly 3) or better beta estimation (already on
`betas_fithrf_GLMdenoise_RR`, NSD's best, and the only variant downloaded).
Restricting to reliable voxels raises the *reported* number without changing physics
— legitimate, but report it as a stratum, not as the headline.

---

## 6. Phase 4 — End-to-end NSD result

Retrain with the Phase 3 winners, regenerate predictions and reconstructions, and
report the Phase 1 battery for ground-truth / zero-shot / few-shot. Freeze as a new
release alongside — never overwriting — `artifacts/model` and `artifacts/results.json`.

**Gate 4:** zero-shot reconstruction beats the direct-CLIP→VDVAE baseline on a
majority of high-level metrics. That is the claim worth publishing: *resting-state
alignment lets an image-trained encoder produce subject-specific responses that
reconstruct better than the stimulus embedding alone.*

---

## 7. Divergence register (fill in as work proceeds)

| Date | Decision | Tag | FOR impact |
|---|---|---|---|
| — | (append each `[FOR-CRITICAL]` choice here) | | |

---

## 8. Deferred: everything FOR will need

Recorded now so nothing is rediscovered later. **None of this blocks Phases 0–4.**

### 8.1 Data to request from `stengerm`

The local FOR export (`/media/psycontrol/HDD/Datasets/FOR/subjects_400_rest/`, 59 MB)
is **parcels-only**. All three of its volume files are 3D atlas label volumes —
including the misleadingly named `schaefer400_bold_raw.nii.gz` (64×64×33, 443 unique
integer labels, verified 3D for all 50 subjects). Source paths are recorded in
`schaefer400_parcel_timeseries.mat` (`bold_file`/`mask_file`/`atlas_file` keys):

| Item | Path pattern | Why |
|---|---|---|
| 4D BOLD | `/net/storage/psycontrol/stengerm/output_all/sub-XXXX_ses-01.ica/filtered_func_data_clean.nii.gz` | the actual voxel data |
| Brain mask | same dir, `mask.nii.gz` | voxel selection |
| **Motion params + FD** | same `.ica` dir | **blocking for any group claim** |
| **FIX provenance** | classification/filtering/censoring records | verify what denoising already happened |
| **Full FreeSurfer subject dirs** | surfaces, `sphere.reg` | required for a surface bridge |
| **BOLD→T1 registration** | `.ica`/FEAT reg dir | required for any common-space bridge |

Reachability: `/net` does not exist on this workstation (no fstab/autofs entry);
`ssh marc3` connects but has no `/net`; no local copy exists anywhere
(searched `filtered_func_data_clean*`, `*.ica`, `*.nii.gz >5 MB`). One unchecked
location: `/dev/nvme1n1`, an unmounted 7.3 TB ext4 volume labeled `SSD_2`.

### 8.2 FOR constraints that are already known

- **Target support must match NSD's.** NSD's `nsdgeneral` spans only 72–74 Schaefer
  parcels (72 common to all six); FOR's cortical coverage spans 394–400. Leading
  components would describe visual cortex in NSD but near-whole cortex in FOR, and
  no rotation reconciles that. Restricting FOR to the 72-parcel intersection yields
  **mean 2,268 voxels** (1,601–3,220) — a 5.7× gain over 400, not 27×. Volume
  coverage is nevertheless comparable (NSD 86.8 cm³ vs FOR 102.1 cm³; FOR voxels are
  45.0 mm³ vs NSD 5.8 mm³) — same extent, coarser sampling.
- **REST budget is the hard limit.** FOR has 237 TRs in one ~7.9-min session vs NSD's
  4,398–7,956. Split-half subspace similarity: 0.615/0.528/0.493 at k=20/50/100.
  More voxels cannot add timepoints. Note: plain shrinkage of `C` toward zero does
  **not** change its singular vectors — use truncation or a properly regularized
  covariance/CCA estimator.
- **Motion is the classic confound** for depressed-vs-control connectivity, and it
  propagates into any reconstruction difference. ICA-FIX alone is insufficient
  justification for skipping nuisance regression. Retain residual-motion QC with
  group-identical thresholds. Current FOR loading only z-scores parcels and records
  the input as precleaned (`src/data/schaefer400.py:346`).
- **Friston-24 must not be re-run blindly** on already-cleaned data;
  `config.yaml`'s `nuisance_regression.require_motion: true` would reject FOR inputs
  outright.
- **The frozen model cannot be reused.** Its `template_fingerprint` is (499, k) and
  83 of those seeds are NSD-only functional localizers. The shared space must be
  refit on a FOR-derivable seed set. (The *method* and the non-localizer seed subset
  remain reusable — only the fitted artifact is not.)
- **Implementation is not "just change the targets."** The Schaefer contract binds
  seeds *and* fusion groups to 400 parcels (`schaefer400_support.py:50`), the loader
  hardcodes 400 target columns (`schaefer400.py:165`), training feeds parcel REST as
  both target and seed runs (`schaefer400_support.py:203`), and the fusion decoder
  needs a group ID per target voxel (`multiexpert_encoding.py:466`). A new
  seed-parcels/voxel-targets contract should be built on the generic variable-voxel
  multi-expert path, not by widening the parcel-only contract.

### 8.3 The structural limit on FOR reconstruction

`Z_hat` depends on the image alone; subject identity enters *only* through the
alignment transforms. For a single orthonormal expert this is exactly degenerate —
`P` has orthonormal columns (`utils.py:60-83`) and `R` is orthogonal, so
`Y_hat @ P @ R = Z_hat` identically. For the active `learned_fusion` output the
degeneracy is **not** exact: two latent heads plus image-dependent per-region
softmax weights are mixed voxelwise (`multiexpert_encoding.py:386-400, 490-505`), so
no single `(P,R)` inverts it. Either way, no measured FOR response enters, so
apparent subject variance may be registration, resolution, or unstable-basis
variance rather than neural signal.

Therefore, when FOR resumes:

- **Primary endpoint = connectivity / alignment-basis features**, preregistered,
  with motion and scanner covariates, nested CV, and subject-label permutation.
- **Stage C stays secondary and explicitly illustrative** until either independent
  FOR task fMRI exists, or an outer-LOSO subject-residual validation anchors
  reconstruction variance to measured responses: hold subject 6 out of A/B *and* the
  decoder, subtract the across-subject image mean, and test whether predicted
  subject residuals recover measured residuals relative to trial reliability —
  with transform-shuffling, per-expert vs fused, and direct-CLIP controls.
- Never promote reconstruction to an inferential arm on the strength of
  "predicted decoding is worse than measured decoding" alone; that does not
  discriminate signal from noise.

---

## 9. Invariants

- `artifacts/model/` and `artifacts/results.json` are frozen. Never modify.
- Subject 7 is evaluation-only and already used once. No model selection on it.
- Session notes stay in `.agent-session-notes/`, untracked. `MEMORY.md` is only
  updated via `$consolidate`.
- FOR clinical group labels (`for_groups.csv`) stay local and uncommitted.
