# Implementation plan v2 — NSD-trained voxel encoder, FOR transfer

**Status:** proposed. Supersedes the sequencing in `NSD_VOXEL_IMPROVEMENT_PLAN.md`
(keep that file for its measured baselines in §0).
**Authored:** 2026-08-04, after a Claude↔Codex design review (`AGREE_WITH_CHANGES`).

---

## 1. Goal

Train an image→brain encoder on NSD at voxel resolution, transfer it to the FOR
cohort using resting-state alignment only, and reconstruct images from predicted
activation. Everything is gated so that no stage is built on an unvalidated one.

## 2. Established facts (measured, with evidence)

**Encoder works.** Subject 7 zero-shot median voxel r = 0.1730, 100-shot 0.1774,
2-vs-2 = 0.967, 12,682 voxels (`artifacts/predictions/subj07/*metrics.json`).
Frozen LOSO 0.17818 (`artifacts/results.json`). 100 real task responses add only
+0.0044 over pure REST alignment — the central thesis holds.

**Headroom is ~3x.** `noise_ceiling_split_half` (`src/evaluation/metrics.py:126`)
returns a Spearman-Brown *reliability* ρ (median 0.2565). Max achievable r is
√ρ = 0.5065, so we sit at **34.2%**. Do not quote r/ρ as "percent of ceiling".

**Reconstruction is broken, but not uniformly.** `gt_fmri` arm from *measured*
fMRI: global VDVAE per-dim r = 0.0013, pred_std 0.0425 vs true_std 0.2690.
But layerwise (Codex measurement): **layer 1 r = 0.2195 / row r = 0.7956;
layer 2 r = 0.1109 / row r = 0.6220**; the large 4,096- and 16,384-dim fine layers
are ≈0 and ~7x under-dispersed, and dominate the global figure. **The ridge did
learn coarse structure.** Low global R² is expected for a 91,168-dim target and is
not by itself a failure signal.

**Root-cause candidate, verified in code:** the VDVAE regression targets are
**stochastic posterior samples, not posterior means**. `DecBlock.sample` computes
`z = draw_gaussian_diag_samples(qm, qv)` (`third_party/vdvae/vae.py:119`) and
`_extract_vdvae_latents` stores that `z`
(`src/data/prepare_reconstruction_features.py:293`). A large share of fine-layer
target variance is therefore irreducible sampling noise that no regressor can
predict. This predicts exactly the observed coarse-good / fine-dead pattern.

**FOR voxel data is in and validated.**
`/media/psycontrol/HDD/Datasets/FOR/voxel_timeseries/`, 50 subjects,
`voxel_timeseries` (n_vox, 237) float32, `voxel_ijk` **1-based**, TR 2.0, grid
64×64×33, all from `filtered_func_data_clean.nii.gz`. Parcel-averaging reproduces
the old 400-parcel series for all 400 parcels. Whole-brain mean 42,601 voxels;
Schaefer cortex 10,901; the 72 NSD-visual parcels **2,269** (1,603–3,220).

**Voxels rescue the alignment from near-degeneracy.** Split-half basis stability
(5 subjects, mean squared canonical correlation vs chance k/V):

| k | voxels V=2,336 | chance | gain | visual parcels V=72 | chance | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.554 | 0.004 | 129.5x | 0.702 | 0.139 | 5.1x |
| 20 | 0.452 | 0.009 | 52.8x | 0.678 | 0.278 | 2.4x |
| 50 | 0.304 | 0.021 | 14.2x | 0.782 | 0.694 | 1.1x |
| 100 | 0.255 | 0.043 | 6.0x | 1.000 | 1.000 | **1.0x** |

At V=72 with k=100 the basis spans the whole space, so stability is trivially 1.000
and carries zero information. **But this does not license k=100 at voxel level
either:** 0.255 × 100 ≈ **25.5 effective overlapping dimensions**, of which chance
contributes ≈4.3 — so only **~21 of 100** dimensions are stable. Each split has
~118 TRs, so the centered connectivity matrix has rank ≤117 and k=100 nearly
saturates the temporal rank. Indicated range is **k ≈ 20–30**, to be selected, not
assumed.

**FOR preprocessing state.** Not demeaned (voxel mean ~9,300), not z-scored
(std ~42) → **z-scoring is mandatory**. Effectively detrended (~0.04% of variance)
and high-pass filtered. No GSR. DVARS max/median 1.46–1.69, 0–4 frames above
1.5× median. TR1 is an outlier (|z|>2) in **34/50** subjects, TR2 in 2/50.
Motion parameters are **not available**.

---

## 3. Phase 1 — Diagnose and fix reconstruction (NSD only)

Order matters: exclude target/decode problems **before** touching prediction
statistics. An affine rescaling cannot change per-dimension correlation, so
calibration can only ever fix *dispersion*, never *information*.

### 1.1 Oracle decode of true latents
Decode the saved `test_latents` from `vdvae_features.npz` directly through VDVAE,
bypassing the ridge entirely.
- Fails → the decode path, layer ordering vs `_VDVAE_LAYER_DIMS`, or
  `ref_latents.npz` is broken. Fix that; everything else is moot.
- Passes → decode is sound; continue to 1.2.

### 1.2 Repeated-extraction reliability audit
Re-extract latents for a fixed image set under independent RNG seeds and report
**test–retest reliability per VDVAE layer**. This measures how much of each layer
is predictable *in principle* given the sampling in `DecBlock.sample`.
Deliverable: a per-layer ceiling. Achieved r can then be read against its own
ceiling instead of against 1.0.
Also evaluate using the posterior **mean** `qm` as the regression target instead of
a sample — strictly more predictable if the sampling noise is material.

### RESULTS of 1.1 and 1.2 (measured 2026-08-04)

Run via `src/pipelines/diagnose_vdvae_latents.py`; outputs in
`artifacts/diagnostics/vdvae_latents/`.

**1.1 Oracle decode: the decode path is CORRECT.** Feeding the saved true
`test_latents` produces clearly recognisable images (row 849 = the bearded man in
the car; row 973 = curly-haired man on the phone), blurry-but-faithful as VDVAE
should be. This rules out: decode path, `ref_latents.npz`, layer ordering vs
`_VDVAE_LAYER_DIMS`, and target validity. **The entire failure is in the
voxels→latents regression step.**

**1.2 Reliability audit: the targets are ~99% sampling noise per dimension.**
50 images, 3 extraction seeds, 3 pairs. Global `mean_dim_reliability` = **0.0125**,
so the single-sample ceiling is √0.0125 = **0.112**; achieved 0.00099 = 0.9% of it.
By layer band:

| Layers | n_dims | share | reliability | ceiling √ρ | achieved | % of ceiling |
|---|---:|---:|---:|---:|---:|---:|
| 0–1 | 32 | 0.04% | 0.43–0.48 | 0.66–0.69 | 0.069–0.204 | 10–29% |
| 2–5 | 1,024 | 1.1% | 0.115–0.137 | 0.34–0.37 | 0.021–0.057 | 6–16% |
| 6–13 | 8,192 | 9.0% | 0.021–0.043 | 0.14–0.21 | ≈0 | 0–3% |
| 14–30 | **81,920** | **89.9%** | **0.003–0.017** | 0.056–0.13 | ≈0 | 0–2% |

Three conclusions:

1. **The regression target is ill-posed.** Because VDVAE is hierarchical and each
   layer conditions on samples from the previous ones, re-extracting the same image
   yields a numerically very different — but equally valid — code. Asking ridge to
   hit one specific draw is asking it to predict noise. The oracle result and the
   near-zero reliability are consistent: a single draw fully specifies the image,
   yet two draws barely correlate dimension-wise.
2. **90% of the target is unpredictable in principle** and dominates the global
   metric. The 0.0013 figure that looked like total failure is mostly layers whose
   own ceiling is ≈0.06.
3. **There is still real headroom where it counts.** Layer 0 achieves only 29% of
   its 0.694 ceiling — roughly 3x improvement available on the layers that
   determine image layout.

Note `mean_row_reliability` is high for coarse layers (0.99, 0.94, 0.68–0.85) while
per-dim reliability is moderate. That gap is an image-independent common mean
inflating the row measure; the per-dimension (across-image centred) figure is the
honest one.

### 1.2b Two fixes these results point to `[new — highest value]`

**(a) Regress the posterior MEAN, not a sample.** `DecBlock.sample` returns
`z = draw_gaussian_diag_samples(qm, qv)` (`third_party/vdvae/vae.py:119`) and
extraction stores that `z` (`prepare_reconstruction_features.py:293`). Storing `qm`
instead makes the target deterministic given the image, raising reliability to 1.0
by construction and the ceiling from 0.112 to ~1.0. This is the single
highest-leverage change available.

**(b) Predict coarse layers only; let the prior fill the fine layers.**
`Decoder.forward_manual_latents` iterates with
`itertools.zip_longest(self.dec_blocks, latents)` (`vae.py:218`), so supplying
**fewer** latent tensors than blocks makes each remaining block take `lvs=None` and
draw from its learned prior via `sample_uncond` (`vae.py:123-134`). No VDVAE change
needed. Predicting layers 0–5 (1,056 dims, all the layout information) and letting
layers 6–30 sample from the prior should give correct structure plus plausible
texture, instead of the 6.3x-shrunk off-manifold values that currently decode to
grid noise.

### 1.2c Review of the two fixes (Claude↔Codex, 2026-08-05) — both valid, ranking corrected

**Verdict: `AGREE_WITH_CHANGES`.** Fix (b) is mechanically valid and is the leading
*production* candidate. Fix (a) is valid as an *experiment*. Renormalization is a
benchmark to test first — **not** established as the primary fix.

**Claims of mine that must be retracted or softened:**

1. **"Scale causes the grid noise" is not established.** Oracle and Ridge latents
   differ simultaneously in scale, information, cross-coordinate covariance, and
   hierarchical consistency. Oracle-passes + raw-fails does not isolate scale.
   *Required control I had missed:* take the **true** latents and artificially
   downscale them 6.3x. If that alone produces grid noise, scale is sufficient.
2. **"Renormalization approximates the prior" is unsound.** It matches only
   coordinate-wise mean and variance — not cross-coordinate covariance, and not the
   hierarchical conditional. When a latent *is* supplied, `sample_uncond` bypasses
   the conditional `pm,pv` entirely (`vae.py:123`). A natural-looking improvement
   after full renormalization could therefore be **unpaired texture hallucination**
   rather than recovered stimulus information.
3. **"99% of the target is noise" and "3x headroom" are over-stated.**
   `pairwise_reliability` averages Pearson r across dimensions and takes sqrt of that
   mean (`diagnose_vdvae_latents.py:82`) — that is a heuristic, not a
   variance-weighted ICC, and 50 images gives noisy per-dimension correlations
   (especially for the two 16-dim layers). Codex's read-only decomposition found the
   coordinate-wise mean pattern accounts for only ~50% and ~31% of within-row
   variance in layers 0–1 — enough to inflate row reliability, **not** enough to
   explain 0.99/0.94 on its own, so my common-mean explanation is incomplete.
   *Required:* recompute row reliability after subtracting a per-coordinate mean
   estimated from **independent training images**; report bootstrap CIs and a proper
   variance-component/ICC estimate. The qualitative fine-layer conclusion will
   probably survive; the specific numbers should not be quoted until then.
4. **Monte-Carlo seed-averaging is not a strictly better fix (a).** It estimates
   `E[z_i | image]` (marginal joint-posterior expectation), a different object from
   the local `qm_i` conditioned on a mean-propagated path, and a vector assembled
   from averaged hierarchical samples need not be a coherent joint sample. Under an
   additive-noise model, `rho_R = R·rho / (1 + (R-1)·rho)`: at global rho = 0.0125,
   even R=8 gives only **0.092** — nowhere near determinism. For the coarse bands
   (rho ≈ 0.12) R=8 gives ≈ **0.52**, so it is genuinely useful there. Treat
   R ∈ {1,2,4,8} as a cheap non-vendored baseline, not a replacement.

**Corrections in my favour:**

5. **Fix (a) is less invasive than I stated.** No vendored edit needed: forward hooks
   on each block's `enc` can record partial-path `qm`, and a temporary project-local
   monkey-patch / context manager around `DecBlock.sample` can propagate means
   end-to-end. Oracle-decode both variants (sampled-path `qm`, fully mean-propagated
   `qm`) on a *small* image set **before** regenerating 9,000 targets.
6. **Do not copy the reference calibration literally.** The official brain-diffuser
   `vdvae_regression.py` standardizes predictions using the **test batch's own**
   per-dimension mean/std — that is *transductive* and invalid for inductive or
   per-subject deployment. Learn the affine calibration from pooled **out-of-fold
   training** predictions, with an epsilon and shrinkage/caps for near-zero predicted
   standard deviations. (§1.3 already specified this; keep it.)
7. **The layer-5 prefix cutoff is unvalidated.** Layers 0–5 are 1×1 and 4×4 latents;
   "all layout information" overstates it, and the 8×8 layers may carry useful
   mid-level geometry. Sweep prefix lengths at resolution boundaries — **2, 6, 14,
   31** — with several fixed prior seeds. 6 is a reasonable first candidate, not an
   established optimum.

**RESULT of battery arm 1 — downscaled-true oracle (measured 2026-08-07):**
Decoding the TRUE test latents multiplied by 0.15806 (the exact measured
pred_std/true_std ratio; decoded latent std 0.0426 vs predictions' 0.0425)
produces **the same high-frequency grid noise as the ridge predictions** — on
latents whose information content, covariance, and hierarchical coherence are
perfect. Outputs in `artifacts/diagnostics/vdvae_latents_scaled/` (`--scale` flag
added to `diagnose_vdvae_latents.py` oracle mode; `latent_scale` recorded in the
summary). **Scale alone is sufficient to break the decoder.** Established:
under-dispersion fully explains the grid-noise failure mode; calibration is
causally motivated, not cosmetic. Still open: whether rescaled *predictions*
(imperfect information) yield recognizable images — that is the renormalization
arm, and the hallucination risk from point 2 stands, so the shuffled control
remains mandatory.

**Agreed experiment order (supersedes my ranking):**

1. **Controlled scale/truncation battery, VDVAE-only, fixed RNG.** Arms: true
   latents · artificially downscaled true latents · raw predictions · OOF-renormalized
   predictions · shuffled renormalized predictions · calibrated coarse prefix + prior.
   Keep Versatile Diffusion **out** initially — its independent CLIP streams and 0.5
   img2img strength confound the result (`benchmark_reconstructions_vdvae_vd.py:968`).
2. OOF full renormalization as the cheapest benchmark.
3. **OOF-calibrated coarse prefix + conditional-prior suffix — the leading production
   candidate.**
4. `qm` and Monte-Carlo target experiments, only if coarse prediction remains limiting.
5. Controls throughout: CLIP-only / mean-or-prior initialization, and shuffled-stream arms.

### 1.2d RESULTS of the calibration battery (measured 2026-08-07) — GATE 1 CORE TEST PASSED

`src/pipelines/vdvae_calibration_battery.py`; outputs in
`artifacts/diagnostics/vdvae_battery/`. 200 eval images, measured-fMRI (gt_fmri)
predictions, VDVAE-only (no Versatile Diffusion), fixed RNG, transductive
diagnostic calibration.

| arm | PixCorr [95% CI] | CLIP 2-way ID | latent std |
|---|---|---:|---:|
| true | 0.922 [0.915, 0.929] | **0.963** | 0.269 |
| raw_pred | 0.230 [0.198, 0.263] | 0.522 | 0.043 |
| **renorm_pred** | **0.269 [0.237, 0.300]** | **0.618** | 0.269 |
| renorm_shuffled | 0.010 [−0.020, 0.043] | 0.497 | 0.269 |
| prefix_0 (pure prior) | 0.043 [0.014, 0.072] | 0.500 | — |
| prefix_2 | 0.082 | 0.576 | — |
| prefix_6 | 0.252 | 0.622 | — |
| prefix_14 | 0.260 | **0.624** | — |

**Conclusions:**

1. **Paired ≫ shuffled — the hallucination hypothesis is dead.** renorm_pred
   identification 0.618 vs shuffled 0.497 (= chance); PixCorr 0.269 vs 0.010 with
   non-overlapping CIs. The calibrated reconstructions carry genuine per-image
   stimulus information from measured fMRI.
2. **Calibration is the production fix.** It takes identification from 0.522 to
   0.618 and produces composition-matched images (visible in the panels) where raw
   predictions gave grid noise.
3. **Prefix truncation is a marginal refinement, not a requirement.** prefix_6
   (0.622) and prefix_14 (0.624) edge out full renorm (0.618) but within noise;
   prefix_2 is clearly worse (0.576) — the 4×4/8×8 bands carry real signal, so
   Codex was right that "layers 0–5 hold all layout" was overstated.
4. **prefix_0 sits exactly at chance (0.500)** — the pure-prior floor behaves, and
   every informative arm clears it.
5. **Metric lesson: PixCorr alone is misleading.** raw_pred grid noise still scores
   PixCorr 0.230 (global luminance structure survives) while its ID is near
   chance. Identification is the discriminating metric; keep both.
6. Gap to true (0.963 vs 0.618) is the information limit of the current ridge from
   measured fMRI — that is Phase-2/3 territory (features, rows, regularization),
   not a decoder problem.

**Production decision:** integrate per-dimension affine calibration (learned from
OUT-OF-FOLD training predictions, per §1.3 — not the battery's transductive
variant) into `benchmark_reconstructions_vdvae_vd.py`, optionally with
prefix 6–14 truncation; then re-run the full gt/zero-shot/few-shot benchmark
including Versatile Diffusion.

### 1.2e End-to-end calibrated benchmark (measured 2026-08-07)

OOF calibration integrated into `benchmark_reconstructions_vdvae_vd.py`
(`src/pipelines/vdvae_calibration.py`; default ON; `--vdvae-calibration none`
reproduces legacy behavior; 117 tests pass). Full rerun with Versatile Diffusion
completed. `summary.json` now carries `vdvae_calibration` + per-condition
`vdvae_calibrated` blocks; panels include the original stimulus.

**Independent confirmation:** the OOF-learned `median_gain = 6.98` (n_capped = 0)
matches the 6.3–7x under-dispersion first measured on eval predictions — two
estimates, no shared data.

**Calibrated std by condition:** gt_fmri **0.2662** (target 0.269 — near-perfect);
zero_shot **0.358** (+33%); few_shot **0.487** (+81%). The shared-calibration
limitation is real and WORSE for few-shot than the ~1.3x estimate. Calibrated R²
is strongly negative (−0.99 to −3.4) as expected from variance inflation — judge
by images/identification, never by calibrated R².

**Visual outcome:** VDVAE inits from measured fMRI are now composition-matched
(row 849 init is a head-and-shoulders figure). But final VD outputs remain
unrecognizable: VD at strength 0.5 with weakly predicted CLIP streams
(zero-shot cliptext r=0.208, clipvision r=0.051) overwrites the init's layout with
confident wrong semantics. Zero-shot finals are washed-out haze; few-shot finals
are abstract (its 81% over-dispersed init plausibly contributes).

**Status: Gate 1 core test passed at the VDVAE stage (battery: paired 0.618 vs
shuffled 0.497); end-to-end recognizability NOT yet achieved.** The bottleneck has
moved from the decoder (fixed) to (a) per-condition calibration overshoot and
(b) CLIP-stream prediction quality steering VD.

**Next steps, ranked:**
1. **Per-condition scale correction, non-transductive:** the encoder can predict
   TRAINING-stimulus responses for the zero/few-shot conditions (frozen transformer
   + P,R on train CLIP features), feed those through the ridge, and learn each
   condition's own calibration from training-side predictions. No eval leakage.
2. **Score the final VD images** with the battery's PixCorr + CLIP 2-way ID
   (functions are importable) including a shuffled control — quantify what survives
   to the end instead of eyeballing panels.
3. VD parameter sweep (strength/mixing) — cheap; strength 0.5 may be too
   init-destructive when CLIP conditioning is weak.
4. CLIP-stream prediction quality is Phase-2/3 territory (features, rows,
   regularization) — the same levers as the encoder improvements.

### 1.2f Per-condition calibration + end-to-end scoring (measured 2026-08-07) — PHASE 1 CORE COMPLETE

Fix history: per-condition mode as first delegated fit fold ridges on the
condition's own inputs → gains exploded (zero 1.25, few 1.70 std). Plan defect,
corrected: folds now fit on MEASURED train fMRI and apply to condition source rows
(`fit_inputs: "measured"` in cache metadata). Separately, VD loading died on a
HuggingFace SSL timeout — weights are cached; run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

Final per-condition calibrated stds: gt 0.2662, zero 0.1909, few 0.1969
(target 0.269 — now ~29% UNDER for the predicted conditions; second-order,
tolerable). Gains: gt 6.99, zero 8.86, few 7.35.

**Scores (200 eval images; `scores_vdvae.csv` = VDVAE inits, `scores_final.csv` =
after Versatile Diffusion):**

| condition | stage | PixCorr [CI] | shuffled | 2-way ID | ID shuffled |
|---|---|---|---:|---:|---:|
| gt_fmri | init | 0.279 [0.245, 0.311] | 0.008 | **0.618** | 0.490 |
| gt_fmri | final | 0.275 [0.243, 0.307] | 0.009 | **0.575** | 0.490 |
| zero_shot | init | 0.057 [0.022, 0.095] | 0.020 | **0.530** | 0.499 |
| zero_shot | final | 0.058 [0.022, 0.095] | 0.019 | **0.524** | 0.495 |
| few_shot | init | 0.056 [0.022, 0.091] | −0.010 | **0.570** | 0.501 |
| few_shot | final | 0.055 [0.023, 0.086] | −0.012 | 0.532 | 0.502 |

**Conclusions:**

1. **The zero-shot thesis survives to pixels — weakly but detectably.** Images
   reconstructed from REST-aligned, image-predicted voxels identify their stimulus
   above chance (ID 0.530 init / 0.524 final; PixCorr CI excludes the shuffled
   mean). First quantitative end-to-end evidence for the pipeline's core premise.
2. **Few-shot > zero-shot at the init stage (0.570 vs 0.530)** — 100 task trials
   buy a real reconstruction gain even though voxel-r barely moves (0.177 vs
   0.173). VD then erodes most of that few-shot advantage (0.532 final).
3. **gt_fmri reproduces the battery (0.618 vs 0.618)** through the integrated
   pipeline — implementation validated.
4. **VD is roughly information-neutral-to-lossy** (gt 0.618→0.575; few
   0.570→0.532): with weakly predicted CLIP streams it does not add semantics,
   it polishes away signal. VD parameter tuning is a real lever but secondary to
   encoder quality.
5. Remaining ~29% under-dispersion for predicted conditions: train-side
   encoder predictions have higher ridge-output variance than eval-side; refine
   only if it matters after encoder improvements.

**Phase-1 verdict: the reconstruction chain is fixed, calibrated, quantified, and
controlled.** The bottleneck is now unambiguously **prediction quality into the
latents** (gt ceiling 0.618 vs zero-shot 0.530) — i.e., Phase 2/3 encoder work.
Remaining Phase-1 nice-to-haves (full metric battery: SSIM/AlexNet/Inception/
EffNet/SwAV; VD strength/mixing sweep) are polish, not blockers.

### 1.3 Calibration ablation
Arms: raw predictions · full reference-style renormalization ·
layerwise/predictability-gated calibration.
**Calibration must be learned from out-of-fold TRAINING predictions**, never from
the evaluation batch's own mean/std (that leaks).
Report every metric **by layer** as well as globally. Expect full renormalization
to *hurt* fine layers by amplifying unpredictable components.

### 1.4 Factorial shuffle controls
A single VDVAE shuffle is insufficient, because the final image also receives
independently-predicted CLIP streams (`benchmark_reconstructions_vdvae_vd.py:868`);
identification could stay above chance from CLIP alone. Run, for both VDVAE-only
and final VD images, with **identical frozen calibration** across arms:

1. true VDVAE latents (oracle)
2. raw predicted VDVAE
3. fully renormalized VDVAE
4. layer-gated / calibrated VDVAE
5. VDVAE shuffled, CLIP paired
6. VDVAE paired, CLIP shuffled
7. all streams shuffled
8. mean/prior VDVAE baseline

### 1.5 Metrics and panels
Implement PixCorr, SSIM, AlexNet(2,5), Inception, CLIP, EfficientNet-B1, SwAV, and
2-way identification in a new `src/evaluation/image_metrics.py`, frozen definitions.
Add the **original stimulus** to panels — the capability already exists in
`src/pipelines/reconstruction_utils.py:172`; `run_pipeline.sh:125` simply never
passes an image-source option. Wiring, not new code.

**Gate 1:** paired calibrated **VDVAE-only** images beat shuffled arms on
identification with bootstrap/permutation uncertainty. Visual recognizability is
*not* sufficient on its own.

---

## 4. Phase 2 — New experimental voxel contract (runs in parallel with Phase 1)

**The frozen release is immutable.** `config.yaml`, `run_pipeline.sh`,
`artifacts/model/`, `artifacts/results.json` are not to be refactored.
`run_pipeline.sh:82` asserts a (499, 100) template and the frozen architecture; that
assertion stays true.

Build the unified voxel route as a **new experimental contract based on
`config_multiexpert.yaml`**. The generic multi-expert decoder already supports
variable voxel counts (`src/models/multiexpert_encoding.py:466`), so it is the right
foundation — not the parcel-bound Schaefer contract, which hardcodes 400 target
columns and one fusion group per target (`src/data/schaefer400.py:165`,
`src/pipelines/schaefer400_support.py:50,203`).

`(seed_set, target_mask, k)` does **not** capture the real differences. The contract
needs configurable:
- subject loaders (NSD betas vs FOR `time_by_voxel.mat`)
- seed providers (400 Schaefer parcel series — available for both datasets already)
- target masks (the 72 NSD-visual parcels; NSD loses nothing, since all 15,724
  `nsdgeneral` voxels already lie inside them)
- fusion-group maps (Schaefer parcel ID per target voxel)
- preprocessing profiles (below)
- coordinate/mask fingerprints for provenance

Retire the old paths only after parity tests and artifact-compatibility checks.

**FOR preprocessing profile:**

| Step | Setting | Reason |
|---|---|---|
| `discard_initial_trs` | **2** | TR1 is an outlier in 34/50; 2 is conservative and costs ~0.8% |
| detrend | **off** | already detrended; see filter note |
| high-pass | **off** | `preprocess_rest_run` uses 5th-order Butterworth + `filtfilt` (`src/data/prepare_rest_data.py:357`); reapplying **squares the response** and adds edge artifacts — reapplication is NOT harmless |
| motion censoring | **off** | needs FD, unavailable |
| nuisance regression | **off** | needs motion params; `require_motion: true` would hard-fail; would double-correct ICA-FIX |
| z-score | **on** | data is raw-scale BOLD |

Assert `voxel_ijk` is 1-based in the loader — 0-based indexing silently yields
garbage (r ≈ 0.7, differences ~1,000).

**Gate 2:** NSD LOSO median voxel r ≥ **0.17818**, all seed means beat ridge.
Do not use subject 7 for selection.

### 4.1 Gate-2 protocol (finalized 2026-08-09, Claude↔Codex review, AGREE_WITH_CHANGES)

**Data layer: DONE.** `data/processed_voxel_contract/` — contract.json (72 common
parcels), NSD bundles subj01–07 (≤10 voxels trimmed per subject), FOR bundles all
50 (mean 2,269 targets, agreement r = 1.000000). Subject-7 parcel REST generated,
run-aligned (0 mismatches).

**Architecture: minimal delta, single expert.** Reuse `SharedSpaceBuilder` +
`StaticTransformerEncoder` UNCHANGED (both verified variable-V and seed-count
agnostic). New `src/pipelines/train_voxel_contract.py` + `config_voxel_contract.yaml`
with `config.yaml` hyperparameters (k=100, same transformer/LR/patience, fixed
200-image eval split). Multi-expert is deferred to a post-gate enhancement
experiment — it answers a different question and would confound seed-bank
attribution.

**Driver requirements (from `loso_multiexpert.py`, reuse the safeguards not the
model):** held-out subject must not influence any registry; per-fold input/config
fingerprinting; stale-result rejection; resumability. Not "just a loop".

**Arms, both through the SAME driver, folds, contracted targets, and eval rows:**
1. **Parity arm:** 499 NSD-ROI seeds, seed 42 — must approximately reproduce the
   frozen 0.18129 or there is protocol drift to fix before interpreting anything.
2. **Gate arm:** 400 Schaefer parcel seeds. Report the absolute score AND the
   paired 400-vs-499 per-fold difference.

**Ridge baseline: refit in the 400-seed space** (changing seeds changes P, R, and
therefore Z; only the scalar 0.17432 survives from the frozen run and it is not
apples-to-apples). Frozen linear-ridge recipe per fold, alpha frozen from training
data only; deterministic, so one 6-fold baseline suffices. 0.17432 becomes a
historical secondary comparator only.

**Aggregation (frozen semantics):** per fold = median voxel r over the exact
config-derived 200 rows; per seed = mean of 6 fold medians; gate = mean of 5 seed
means ≥ 0.1781831592 AND every seed mean > the new 400-space ridge.

**Stage-A futility rule (exact):** seed 42 first. Continue to seeds 43–46 only if
seed-42 mean ≥ **0.1757758409** (the minimum frozen seed result) AND > the new
ridge. Otherwise stop for futility.

**Diagnostics recorded per fold (do not select on them):** singular spectra,
effective ranks, fingerprint residuals — `compute_global_k` checks only shape, and
these make a failed gate interpretable.

**Pre-registered fallback if the 400-seed arm fails: Schaefer-1000 seeds** (same
atlas family, one factor — granularity). NOT the hybrid 400+416 bank (duplicated
coverage reweights fingerprint rows; fingerprints are raw `C·P·R` with no block
normalization). Note: Schaefer-1000 needs a new verified preparation layer.
Re-baselining the gate is not a fallback.

**Scope honesty:** passing Gate 2 yields a model with a FOR-compatible **seed
definition** and acceptable NSD accuracy — cleanup and spatial-resolution transfer
remain unvalidated (Gates 3/3a).

### 4.2 Gate-2 Stage-A RESULTS (measured 2026-08-09)

Driver: `src/pipelines/train_voxel_contract.py`; artifacts under
`artifacts/voxel_contract_loso/`. One driver fix during bring-up: the nsd499 arm
must load the FROZEN seed registry from `artifacts/model/external_seed_info.json`
(the frozen release computed its 499 seed defs over all six subjects; recomputing
over five fold-training subjects yields a different registry and breaks parity —
registry scoping is inherited from the reference protocol). Verified the frozen
defs resolve to the existing cache `182cc676bcc3`.

| fold (held-out) | 499-seed encoder | 400-seed encoder | Δ | 400-space ridge |
|---|---:|---:|---:|---:|
| subj1 | 0.1770 | 0.1723 | −0.0047 | 0.1664 |
| subj2 | 0.1582 | 0.1387 | −0.0195 | 0.1357 |
| subj3 | 0.1681 | 0.1611 | −0.0070 | 0.1514 |
| subj4 | 0.1590 | 0.1443 | −0.0147 | 0.1500 |
| subj5 | 0.2625 | 0.2540 | −0.0085 | 0.2522 |
| subj6 | 0.1628 | 0.1569 | −0.0058 | 0.1478 |
| **mean** | **0.18125** | **0.17122** | **−0.0100** | 0.16726 |

- **Parity arm PASSED:** 0.18125 vs frozen 0.18129 (Δ = 0.00004). The new driver,
  contract targets, and protocol are validated; every number below is
  interpretable.
- **Gate arm Stage-A FAILED (futility):** 0.17122 < 0.17578. The pre-registered
  rule stops seeds 43–46. The Schaefer-400 seed-bank cost is **−0.0100 (−5.5%)**,
  negative in all six folds — real, uniform, and now cleanly attributed.
- Encoder beats its refit ridge on the mean (0.17122 > 0.16726) and in 5/6 folds
  (exception: subj4, 0.1443 < 0.1500).
- Diagnostics show no rank starvation (effective rank 10.5 both arms); fold-1
  fingerprint residual is higher for 400 seeds (0.42 vs 0.35) — consistent with
  modest information loss, not degeneracy.

**DECISION D-08 (resolved 2026-08-13, project owner):** the measured −5.5% cost is
**accepted**. The Schaefer-1000 fallback is deferred indefinitely (would require
FOR FreeSurfer surfaces for deployment in any case). Consequences: (i) the Gate-2
threshold is consciously waived for the 400-parcel configuration — this is a
documented owner decision, not a silent re-baseline; (ii) the FINAL FOR-facing
model is trained on all six NSD subjects with the schaefer400-arm configuration
(seed 42, k=100, frozen hyperparameters), evaluated ONCE zero-shot on subject 7
for the honest transfer number; (iii) Gates 3a (cleanup shift) and 3
(resolution shift) remain required before FOR inference.

**FINAL MODEL (trained 2026-08-13, per D-08):**
`artifacts/voxel_contract_final/seed42/` — schaefer400 arm, trained on subjects
1–6, evaluated once zero-shot on subject 7 (`subject7_usage:
evaluation_only_once_D08`). Result: **encoder median voxel r = 0.1631** vs ridge
0.1588 (α=10,000), 12,672 target voxels, 200 eval images. Consistency check: the
frozen 499-seed model scored 0.1730 on the same subject/protocol → observed cost
−0.0099, matching the LOSO-measured −0.0100 on a subject that never entered any
fold. The seed-bank cost generalizes. This is the FOR-facing model, pending
Gates 3a and 3.

**Pre-registered fallback = Schaefer-1000 seeds. Practical constraint discovered:**
FOR-side Schaefer-1000 requires FOR FreeSurfer surfaces (Tier-3 data, not held);
NSD-side needs only the CBIG annotations + existing transforms. Sequencing
decision: test Schaefer-1000 on **NSD first** — if it recovers the loss, that
evidence justifies the Tier-3 FOR data request; if it does not, the request is
moot and the measured −5.5% cost becomes a conscious accept/reject decision for
the project owner.

**Gate-3a RESULT (measured 2026-08-14):** `artifacts/gate3a/seed42/gate3a_summary.json`.
Held-out subjects realigned from minimal-cleanup REST (motion regression and
censoring off; detrend/high-pass/z-score unchanged), evaluated against the saved
Gate-2 fold models. Per-fold deltas: subj01 −0.0118, subj02 −0.0094, subj03
−0.0132, subj04 −0.0027, subj05 −0.0042, subj06 −0.0047. **Mean delta −0.0077;
first-order flag NOT raised** (threshold −0.01). Interpretation: the motion-cleanup
shift is a real but second-order effect, smaller than the accepted seed-bank cost.
Caveats: two folds individually exceed 0.01; this bounds the *absence* of motion
cleanup, whereas FOR data is ICA-FIX-cleaned (likely better than nothing), so the
true FOR shift is plausibly smaller than this bound. Proceed to Gate 3.

**Gate-3a (original spec): NSD cleanup stress test** — after Stage A, before seeds 43–46:
keep training folds Friston-24-cleaned, realign each held-out subject from an
independently regenerated MINIMAL-cleanup REST view (BOTH voxel REST and Schaefer
seed REST), evaluate the same 200 images. A sensitivity bound for the NSD→FOR
cleanup shift, not an ICA-FIX simulation. (No cleanup ablation is needed before
the first training run: connectivity re-centers/rescales every column, so
z-scoring differences are irrelevant; motion regression/filtering/FIX can change
covariance and are what this test bounds.)

---

## 5. Phase 3 — k selection and cross-resolution validation

### 3.1 Nested k selection
Sweep k ∈ {10, 20, 30, 50, 75, 100} with **no winner named in advance**. Select on:
- nested NSD LOSO encoding performance
- absolute contiguous-block / bootstrap subspace overlap (not ratio-to-chance)
- fingerprint-to-template residual
- stability of reconstructed voxel patterns for a fixed image set
- singular-spectrum / eigengap diagnostics

Calibrate the FOR stability threshold **empirically**: degrade NSD REST to
FOR-like time budgets (~237 TRs) and measure when held-out voxel prediction fails.
That gives a defensible floor instead of an arbitrary multiple of chance.

**Gate-3 RESULT (measured 2026-08-14):** `artifacts/gate3/seed42/gate3_summary.json`.
NSD REST block-averaged 2×2×2 (3.6 mm ≈ FOR's 45 mm³ voxels; 2,619–3,570 coarse
voxels per fold — FOR-scale). Both arms evaluated on coarse truth. Mean deltas by
fingerprint variant: **raw −0.0050** (flag NOT raised), column_normalized −0.0578,
volume_weighted −0.0667. **Selected variant: raw.** The singular-value weighting in
`F = C·P` is load-bearing, not a defect — normalizing it degrades alignment ~12x.
The cross-resolution concern from the 2026-08-09 review is retired with evidence;
FOR deployment uses the unmodified production fingerprint matching. Per-fold raw
deltas: −0.0033 to −0.0079, uniform across subjects.

### 3.2 NSD resolution-shift gate `[original spec]`
Variable V is algebraically fine, but **cross-resolution fingerprint equivalence is
unproven**. With `C = UΣVᵀ` and `P = V_k`, the fingerprint is `F = C·P = U_kΣ_k` —
it retains singular-value weighting. Voxel density, voxel volume, mask definition
and within-parcel sampling all change `Σ` and therefore the orientation Procrustes
emphasizes. The alignment centers but does **not** whiten fingerprints
(`src/alignment/shared_space.py:148`, `src/alignment/cha_alignment.py:43`).

Test, on NSD only: restrict to the same 72 parcels, subsample to FOR-like spatial
density (7.7x larger effective voxels), recompute transforms, and measure
- fingerprint residual vs native-resolution NSD
- fixed-image prediction stability
- held-out task prediction

Compare raw `UΣ`, column-normalized, parcel-volume-weighted, and whitened
fingerprints. Select using NSD only.

**Gate 3:** the resolution-shift test must show acceptable degradation, and every
FOR subject must clear the empirically calibrated stability floor.

---

**PHASE-4 MILESTONE (2026-08-14): FOR inference COMPLETE.**
`artifacts/for_inference/seed42/` — all 50 subjects, verified: 500×V_t predicted
responses per subject (finite, subject-distinct), connectivity (400×V_t) and
fingerprint (400×100) features saved per subject for the clinical analysis,
provenance with `accuracy_computed: false` throughout, batch manifest complete.
Pipeline provenance chain: final model (subj-7 zero-shot 0.1631) → Gate 3a
(−0.0077) → Gate 3 (−0.0050, raw fingerprints) → this batch. The prediction side
of the project is complete end to end. Module is diagnosis-blind by construction
(no group logic, no clinical-file access). Next: Phase 5 analysis
(connectivity-primary, per D-02; confirmatory only with motion data).

## 6. Phase 4 — FOR inference

Only after Gates 2 and 3. Produces predicted FOR voxel activation
(~2,269 visual voxels × N images) per subject.

FOR prediction accuracy is **permanently unmeasurable** — no FOR task fMRI exists.
`batch_manifest.json` already records `accuracy_computed: false`; keep that honest.

---

## 7. Phase 5 — FOR analysis

### 7.0 FROZEN ANALYSIS DESIGN (2026-08-14, Claude↔Codex review, AGREE_WITH_CHANGES)

Note: this review round ran as a single-critic (Codex-only) assessment. The
critic performed label-blind audits of the real bundles; findings below marked
[audited] carry measured numbers.

**Endpoint structure (frozen before any label access):**
1. **Model-derived primary (1 test):** global raw **NSD-template fingerprint
   alignment error** ||F_s − T||_F/||T||_F, exactly the Gate-3 definition
   (`gate3_resolution_shift.py:260`). Precise naming is mandatory: this is a
   fingerprint alignment error against a 6-subject NSD reference scanned on a
   different protocol — NOT a normative-deviation biomarker, and NOT "the
   quantity the model sees" (the encoder consumes P,R; the residual is
   diagnostic). The estimand is stated literally: which FOR group lies closer
   to the NSD reference. [audited: 74% of residual squared error is the
   fingerprint column-mean offset (r=.97 with it) — the offset/centered-shape
   decomposition is preregistered as non-inferential QC.]
2. **Conventional family primary (1 test):** one permutation **omnibus** over
   the 28 Yeo-7 block values of the 400×400 parcel connectivity (sum of squared
   studentized group effects). Contract: Pearson on the 235×400 contract seeds,
   clip + Fisher-z, exclude self-edges, unique edges only, checksum-pinned
   Schaefer/Yeo ordering. Framed as a MODEL-FREE ANALYSIS OF THE SAME SCAN —
   not independent corroboration (same data, same motion confound).
   [audited: contiguous-half across-subject reliability of the 28 blocks
   r=.73–.85, median .79.]
3. **Exploratory localization (BH within each set):** 7 network fingerprint
   residuals; 28 individual connectivity blocks.
4. **QC / sensitivity only (never inferential):** residual mean-offset vs
   centered-shape decomposition; endpoint split-half reliability (label-blind
   measurement gate BEFORE labels); V_t, standardized DVARS, coverage
   correlations [audited: raw residual vs V_t r=.013; centered residual
   r=.44 — a reason the raw residual is primary]; log(V_t)/DVARS enter only as
   declared sensitivity analyses. Outcome-adaptive demotion rules are
   prohibited.

**Statistics:** subject-label permutation (10,000), two-sided; effect sizes
with bootstrap CIs. When FOR motion data arrives: Freedman–Lane residual
permutation (fit reduced nuisance model, permute residuals, reconstruct,
refit; same schedule across endpoints — Winkler et al. 2014), with group-blind
exclusions, mean FD + censored-fraction covariates, residual motion–endpoint
checks, stricter-censoring sensitivity analyses. Friston-24 is NOT re-run
blindly without upstream FIX provenance.

**Claim policy:** before motion data, NO clinical/neural group claim is
non-exploratory — label permutation protects type-I error of the label
association but cannot separate diagnosis from group-correlated motion (Power
2012; Ciric 2017). Permitted pre-motion claims: artifact completeness, endpoint
reliability, coverage robustness, or "a preregistered pipeline detected an
association in the delivered dataset, unresolved with respect to motion."

**Enforced blindness (not merely procedural):** the analysis module is
developed and frozen against synthetic labels; endpoint manifest, code commit,
QC decisions and output schema are fixed BEFORE label access; the final run is
executed such that `for_groups.csv` is read exactly once — and the module must
never import the existing label loader (`for_assessor_study.py:89` reads labels
even in check mode; that path is off-limits).

**Illustrative reconstructions:** 4 stimulus IDs frozen now; subjects selected
per group by a deterministic hash rule after unblinding; labeled "random
examples", never "representative cases". No statistics.

**Rejected as primary:** ML classification on fingerprints (n=50, 40k-d;
optimism risk); if ever added as clearly-labeled secondary, all preprocessing
and feature selection must sit inside every outer fold and every permutation.

### 7.0a Addendum to the frozen design (2026-08-20, Claude↔Codex implementation review, AGREE_WITH_CHANGES)

1. **Endpoint definition pinned:** the fingerprint primary uses the SAVED
   zero-shot fingerprint `F_s := C_s·P_s·R_s` from
   `artifacts/for_inference/seed42/<sub>/fingerprint.npy`; residual
   = ||F_s − T||_F/||T||_F against the final-model builder template. A second
   fitted rotation is PROHIBITED (Gate 3 refits one only because it starts
   from unrotated C·P; the saved FOR fingerprints are already rotation-aligned
   — `for_inference.py` applies R at inference; label-free audit: double
   rotation changes values ≤ 6.7e-7 but is definitionally wrong).
2. **Honest framing:** `for_groups.csv` was already read by the exploratory
   assessor analysis (2026-08-14). All Phase-5 claims are therefore framed as
   a "prospectively locked endpoint analysis after prior label exposure" —
   endpoints and tests specified before THESE endpoints ever met labels, not
   "before any label access" by the project. This disclosure is mandatory in
   results.json and any write-up.
3. **Reliability is a diagnostic, not a gate:** contiguous 117/118-TR
   split-half reliability (within-half standardization) is reported for the
   28 connectivity blocks and for the fingerprint residual; no post-hoc
   threshold is invented.
4. **Freeze anchor:** endpoint/manifest/calibration/atlas/permutation-schedule
   hashes are recorded in a GIT-TRACKED `analysis_registry/phase5_freeze.json`
   before the label run; `analyze` validates the digest before opening the
   label file and refuses to overwrite existing results.
5. **Motion adjustment is a versioned second analysis:** Freedman–Lane is
   implemented and unit-tested but NOT exposed in the production CLI; when
   motion data arrives, a label-free motion-preparation/freeze stage precedes
   a separate, explicitly versioned motion-adjusted run.
6. **Result semantics:** both primary p-values reported raw (no adjustment
   across the two families; family-wise error across families is explicitly
   uncontrolled); omnibus reports statistic + p only (no invented effect
   size); per-endpoint effects = raw group difference + Cohen's d with
   stratified subject bootstrap (10,000 resamples, percentile CI, named RNG
   streams); omnibus p is upper-tail (direction-agnostic via squaring),
   per-block/per-network tests two-sided via |t|; one precomputed 10,000×50
   permutation schedule shared by all endpoints.

### 7.0b Illustrative reconstruction, assessor, and swap-control RESULTS (measured 2026-08-14–18)

All exploratory / illustrative under D-02; motion-uncorrected.

**Reconstruction from predicted fMRI (2026-08-14,
`reconstruct_from_predictions.py`, `artifacts/recon_from_predictions/seed42/`):**
72-parcel common decoder fitted on measured subj07 responses; go/no-go on
measured inputs passed (paired CLIP 2-way ID 0.546 vs 0.500 shuffled). A
degeneracy on FOR inputs (constant images) was traced to ~4× per-voxel
over-dispersion from small V_t (amplitude ~1/√V_t) and fixed by per-subject
column-moment standardization. VDVAE-stage scores across all 51 subjects:
FOR pixcorr 0.094–0.122, subj07 0.109 (inside the FOR range); all subjects
clear shuffled baseline (+0.02 rule). Zero-shot transfer costs ≈ nothing at
the reconstruction level.

**Assessor study (2026-08-14, `assessor_score_reconstructions.py` label-free →
`assessor_group_analysis.py`, `analysis_exploratory/`):** 6 affective
dimensions; depressed−healthy differences all null (|Δ| ≤ 0.008 rating points,
95% bootstrap CIs within ±0.024, all BH-adjusted p = 0.87). Reconstructions
score systematically below originals on valence/arousal/approach (Δ ≈ −0.15
to −0.47) — a blur effect, uniform across subjects.

**Swap control (2026-08-18, `swap_control_reconstruction.py`,
`artifacts/swap_control/seed42/`):** because Z_hat is image-only, "subject A
with subject B's rest" ≡ subject B's prediction; the swap control is therefore
a seeded cross-subject comparison (fixed `torch.manual_seed`, identical batch
layout, so all subjects consume identical VDVAE prior draws). Same-image
cross-subject correlation by stage (6 subjects, 471 images; 29 NaN-degenerate
rows excluded): decoder inputs 0.613 → predicted latents 0.869 → seeded
pixels 0.859 → unseeded pixels 0.713. Conclusions: (a) ~half of the apparent
between-subject image differences in the unseeded run were prior-sampling
noise; (b) the residual ~0.14 dissimilarity is causally attributable to the
rest-derived transform but is rendering style, not content — the ridge decoder
filters most subject-specific input variance (0.613→0.869) by construction.
This sharpens the D-02 rationale: group signal must be sought in
connectivity/fingerprint features, not reconstructions. State figures:
`artifacts/state_figures/` (fig1–6).

### 7.1 Exploratory Phase-5 RESULTS (measured 2026-08-20, motion-uncorrected)

Pipeline: endpoints → calibrate → freeze (commit cc7a275, digest 0a0b125f…) →
analyze; single label read; all freeze checks passed.
Measurement quality (label-free): block reliability r=.729–.850 (median .794,
reproducing the independent design-review audit), fingerprint residual
reliability r=.868; calibration: fake-label p-values ~uniform, injected d≈1
detected at 96%.

**Both primaries null.** Primary 1 (global fingerprint residual): depressed
0.670 vs healthy 0.637, Δ=+0.034 [−0.033, 0.105], d=0.26 [−0.27, 0.86],
p=0.358. Primary 2 (28-block connectivity omnibus): T=7.07, p=0.854 — the
combined block profile is *more* similar across groups than typical chance
relabelings. Exploratory: 0/7 network residuals and 0/28 blocks BH-significant
(min raw p=.27 SomMot residual; .30 SalVentAttn|Cont block, d=0.30, BH p=.945).
QC note recorded pre-labels: residual vs mean DVARS ρ=.29 — motion adjustment
(Freedman–Lane, dormant) remains the required confirmatory step.

Interpretation under the frozen claim policy: a preregistered pipeline
detected NO group association in the delivered dataset at these endpoints
(exploratory, motion-unresolved). n=50 power caveat: effects below d≈0.8 were
never likely to clear α=.05.

**Primary endpoint: connectivity / alignment-basis features**, per the frozen
§7.0 design (two primaries, subject-label permutation; ML classification /
nested CV rejected as primary — superseded statement corrected 2026-08-20).

**Reconstruction stays secondary and explicitly illustrative.** `Z_hat` depends on
the image alone, so FOR subject identity enters only through `(P_for, R_for)`; any
group difference in reconstructions is a deterministic function of resting-state
connectivity, not independent evidence about visual processing.

**Motion gate.** Low DVARS outlier counts show few gross transients; they do **not**
establish equal residual motion across clinical groups. Connectivity is especially
vulnerable because motion changes the primary endpoint directly. Therefore:
- obtain realignment parameters / FD, or recover motion estimates from the
  preprocessing outputs, **or**
- keep the clinical comparison **exploratory** and attach no confirmatory claim.

Disabling unavailable Friston-24/FD processing is operationally correct but does not
make the clinical comparison valid.

---

## 8. Authoritative sequence

1. Oracle true-latent decode + repeated-extraction reliability audit
2. Calibration ablation (raw / full / layer-gated) with factorial shuffles
3. Image metrics with frozen definitions and original stimuli
4. *In parallel:* new experimental generic voxel contract; frozen release preserved
5. Nested k selection + NSD resolution-shift validation
6. FOR inference — only after absolute stability and cross-domain gates
7. Clinical analysis — confirmatory only with adequate motion control, else exploratory

Keep separate artifact roots per phase. Reconstruction tuning must never touch
subject 7 model selection.

## 9. Invariants

- `artifacts/model/`, `artifacts/results.json`, `config.yaml`, `run_pipeline.sh` frozen.
- Subject 7 is evaluation-only and already used once.
- `for_groups.csv` stays local and uncommitted.
- Session notes in `.agent-session-notes/`; `MEMORY.md` only via `$consolidate`.
