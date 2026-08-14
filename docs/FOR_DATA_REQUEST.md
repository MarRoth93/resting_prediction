# FOR resting-state data request — context document

**From:** Marco Rothermel (`marco.rothermel@uni-marburg.de`), resting_prediction project
**To:** Madleen, and her Claude Code agent
**Date:** 2026-08-03

---

## 0. How to use this document

This is a **context file**, not a task list. It is written so that an AI coding
agent working with Madleen can (a) explain to her what is being asked for and why,
(b) locate the files, (c) verify them before transfer, and (d) recognise valid
substitutes if the exact files named here do not exist.

If something named below is missing or differently organised on your side, the
"Why we need it" column is the thing to reason from — please propose the closest
equivalent rather than assuming the request is impossible.

---

## 1. The request in one table

| # | What | Path pattern (as recorded in the data you already gave us) | Priority |
|---|---|---|---|
| 1 | 4D resting-state BOLD | `/net/storage/psycontrol/stengerm/output_all/sub-XXXX_ses-01.ica/filtered_func_data_clean.nii.gz` | **Essential** |
| 2 | Brain mask | `/net/storage/psycontrol/stengerm/output_all/sub-XXXX_ses-01.ica/mask.nii.gz` | Nice to have |
| 3 | Motion parameters / FD | same `.ica` directory (e.g. `mc/prefiltered_func_data_mcf.par`, `*_rel.rms`, `*_abs.rms`) | **Essential for group analysis** |
| 4 | ICA-FIX provenance | FIX classification / labels / training set used, and what filtering + censoring was applied | **Essential for group analysis** |
| 5 | FreeSurfer subject directories | wherever `recon-all` output lives (needs `surf/`, `label/`, `sphere.reg`) | Later — do not send yet |
| 6 | BOLD→T1 registration | FEAT/`.ica` registration output (e.g. `reg/example_func2highres.mat`, or the FSL/FreeSurfer equivalent) | Later — do not send yet |

For all 50 subjects currently under `subjects_400_rest` (`sub-0027` … `sub-3134`).

**Items 5 and 6 are deliberately parked.** There is an unresolved problem on our
side (§6) that must be fixed before those are useful. Please do not spend effort on
them yet — we will come back if and when they are needed.

---

## 2. What the project does, briefly

We train an image→brain encoding model on the Natural Scenes Dataset (NSD), then
transfer it to other subjects using **resting-state fMRI only**.

```
image → CLIP ViT-L/14 → transformer → 100-D shared response   (depends on image ONLY)
resting-state fMRI → connectivity fingerprint → subject alignment (P, R)
predicted response for subject s = shared response @ R_sᵀ @ P_sᵀ
```

The key property: the resting-state scan is **not** an input to the neural network.
It is used to compute a per-subject change of basis `(P_s, R_s)` that maps the
common 100-D response space onto that individual's own brain. This is what allows
the model to predict responses for a subject who never performed any task in the
scanner — which is exactly the FOR situation.

**FOR has no task fMRI**, so FOR resting-state serves exactly one purpose: computing
`(P_for, R_for)`. It is never training data.

---

## 3. What we already have (please do not re-send)

`/media/psycontrol/HDD/Datasets/FOR/` — 59 MB total, plus `allsubject_info.mat`.

Per subject, under `subjects_400_rest/sub-XXXX/`:

| File | What it actually contains (verified) |
|---|---|
| `schaefer400_parcel_timeseries.mat` | `parcel_timeseries` (400, 237) float32, `TR_seconds` 2.0, `parcel_ids`, `parcel_voxel_count`, and the source paths in §1 |
| `schaefer400_parcel_summary.tsv` | per-parcel `voxel_count`, `usable` flag |
| `schaefer400_bold_raw.nii.gz` | **3D atlas label volume**, 64×64×33 @ 3.281×3.281×4.18 mm, 442 nonzero labels (400 Schaefer cortical + 40 FreeSurfer aseg + medial-wall) |
| `schaefer400_feat_highres.nii.gz` | same atlas, 176×256×170 @ 1 mm |
| `schaefer400_native_t1.mgz` | same atlas, 256³ @ 1 mm |

Two notes for your agent:

- Despite its name, `schaefer400_bold_raw.nii.gz` contains **no BOLD data**. It is
  the atlas resampled into the BOLD grid. We verified all 50 copies are 3D
  single-volume label images. This is why we need item 1 — the export is complete
  and internally consistent, it simply never contained voxel timeseries.
- The atlas files are sufficient for us to define parcel membership ourselves. We
  confirmed the cortical labels account for 99% of the voxel set used in the
  parcellation (sub-0027: 9,636 atlas cortical voxels vs 9,563 in the summary TSV, a
  73-voxel / 1% difference). That is why item 2 is only "nice to have".

---

## 4. Why each item is needed

### Item 1 — 4D BOLD `filtered_func_data_clean.nii.gz` (essential)

The 400-parcel average discards the information the alignment method actually uses.
Three measured reasons:

**(a) Dimensional crowding makes parcel-level alignment partly meaningless.** The
alignment extracts a 100-dimensional basis from the target space. With 400 parcels
that basis spans 25% of the entire space, so two subjects with *no* functional
correspondence still show ~25% subspace agreement by chance. We simulated this:

| Target space | dimension V | chance overlap of two unrelated subjects' 100-D subspaces |
|---|---:|---:|
| 400 parcels | 400 | **0.2495** |
| voxels (visual support) | ~2,268 | 0.0443 |
| voxels (NSD comparison) | 15,724 | 0.0064 |

**(b) Voxels within a parcel do not respond alike.** Measured on NSD subject 1
(1,000 images, 15,724 voxels grouped by Schaefer parcel): mean within-parcel
voxel-to-voxel correlation across images is only **0.216**. Parcel averaging retains
50.7% of per-voxel signal amplitude — but more importantly it collapses ~228 numbers
into 1, deleting the spatial pattern inside each parcel. Visual cortex is
retinotopic, so neighbouring voxels in one parcel encode different visual-field
positions and often respond in opposite directions to the same image; averaging
cancels them.

**(c) The parcellation removes precisely what the method relies on.** Schaefer
parcels are defined *by connectivity homogeneity* — voxels are grouped because their
connectivity fingerprints are similar. Our alignment works by *matching connectivity
fingerprints* to find functionally corresponding locations across brains. So
parcel-averaging deliberately erases the within-parcel fingerprint variation the
method needs.

**Note:** we already have the atlas in BOLD grid, so item 1 alone is sufficient — we
can extract voxels ourselves. No parcellation work is needed on your side.

### Item 2 — `mask.nii.gz` (nice to have)

Only to reproduce the exact voxel set used for the existing parcel timeseries, for
provenance. As noted above the difference is ~1%. Cheap to include; not worth effort
if awkward.

### Items 3 and 4 — motion and FIX provenance (essential for the group analysis)

The eventual scientific question is whether a clinical group difference exists.
Head motion is the classic confound for exactly this kind of comparison: it produces
systematic connectivity differences, and clinical groups frequently differ in
motion. Because our pipeline turns resting-state connectivity into the subject
alignment, any motion-driven connectivity difference propagates directly into every
downstream result.

ICA-FIX cleaning is good but does not by itself let us skip motion QC. To make a
defensible group claim we need to:

- apply residual-motion QC with **thresholds identical across groups**
- report FD distributions per group and include motion as a covariate
- document exclusions

None of that is possible without the motion traces. We also need to know what FIX
already did (item 4) so we do not double-correct: our default pipeline applies
Friston-24 nuisance regression, which would be wrong to re-run on already-cleaned
data. Knowing the FIX configuration lets us disable the right steps deliberately
rather than guessing.

**Specifically useful:** which FIX training set / threshold was used, whether
high-pass filtering was applied and at what cutoff, whether any volume censoring or
spike regression was done, and whether motion parameters were regressed out before
or after ICA.

### Items 5 and 6 — surfaces and registration (parked)

These would be needed to map data into a common space (fsaverage or MNI) for the
image-reconstruction stage. That stage is currently broken on our side for reasons
unrelated to FOR (§6), so requesting them now would be premature. Recorded here only
so the eventual need is not a surprise.

We infer from `schaefer400_native_t1.mgz` (256³, 1 mm) that `recon-all` was run
upstream, and from `schaefer400_feat_highres.nii.gz` that a BOLD→highres
registration exists. We have the *products* of those transforms but not the
transforms themselves, so they cannot be inverted or reused.

---

## 5. Verification checklist before transfer

Suggested checks your agent can run so we catch mismatches early. Expected values
come from the parcel export we already hold.

```python
import nibabel as nib
img = nib.load("filtered_func_data_clean.nii.gz")
print(img.shape)                    # expect (64, 64, 33, 237)
print(img.header.get_zooms())       # expect ~(3.281, 3.281, 4.18, 2.0)
```

- **4 dimensions**, not 3. A 3D result means the wrong file (that is exactly the
  mistake in the current export).
- **237 volumes** for every subject — all 50 are 237 in our parcel copy, so any
  deviation is worth flagging rather than silently sending.
- **TR = 2.0 s**.
- **Grid 64×64×33** matching `schaefer400_bold_raw.nii.gz`, so the atlas we already
  have overlays without resampling. If your BOLD is on a different grid than the
  atlas we were given, please tell us — that would mean the atlas was resampled and
  we need the matching version.
- Sanity: the parcel-average of the 4D data under the atlas should reproduce
  `parcel_timeseries` in the `.mat` file. If your agent can confirm that for one
  subject, it validates the whole transfer.

## Size estimate

135,168 voxels × 237 volumes:

| dtype | per subject | 50 subjects (gzipped) |
|---|---|---|
| int16 | 64 MB raw, ~29–45 MB gzipped | **~1.8 GB** |
| float32 | 128 MB raw, ~58–90 MB gzipped | **~3.5 GB** |

So the essential request is roughly **2–4 GB**. Motion files add a few MB. Any
transfer route is fine — rsync/scp, a share, or a mounted drive. If it helps, a
one-subject test transfer first would let us validate the format before you move all
50.

---

## 6. Honest note on timing

There is currently a blocking problem on **our** side, unrelated to FOR. Our
image-reconstruction stage does not work even with real NSD data: decoding a
subject's *measured* fMRI produces noise rather than a recognisable image (the
VDVAE latent regression has R² = −0.025 and per-dimension correlation of 0.001
against ground truth). We are debugging that.

This does not affect items 1–4. The FOR resting-state data feeds the *alignment*
stage, which is healthy and separately valuable — and the data will keep. But it is
why items 5–6 are parked, and why we are not in a rush.

We would rather be transparent about this than have you invest effort on the
assumption that everything downstream is ready.

---

## 7. Questions for you

1. Are the `.ica` directories still intact, or only the parcellation outputs?
2. Are motion parameters and FD available per subject, and in what format?
3. What FIX configuration was used — training set, threshold, and whether
   high-pass filtering / censoring / motion regression were applied?
4. Was every subject a single session (`ses-01`), or do some have more resting-state
   runs that were not used? More resting-state data per subject would materially
   improve the alignment — 237 volumes is short for estimating a connectivity
   fingerprint, and any additional runs would be valuable.
5. Is there a preferred transfer route, or access we could be granted so we pull it
   ourselves rather than you pushing it?

---

## 8. What we explicitly do NOT need

- Task fMRI — we understand there is none for this cohort.
- Any parcellation or atlas work; we have the atlas files and can parcellate.
- Clinical, demographic, or diagnostic information. Group labels are handled
  separately and are not part of this request.
- Anything derived from `allsubject_info.mat` — we already have that file.
