# FOR VDVAE assessor study

This is a staged, resumable analysis for asking whether frozen assessor scores
of model-generated reconstructions differ between healthy and depressed FOR
participants. It does not treat the reconstructions as measured FOR task-fMRI.

## Fixed study contract

- The same 500 model-unseen NSD images are used for every FOR subject.
- FOR responses are the existing `learned_fusion.npy` predictions produced
  from image CLIP features plus each subject's resting-state alignment.
- One shared Schaefer-400-to-VDVAE decoder is selected using NSD only.
- Diagnosis is never provided to the response model, decoder, VDVAE, or
  assessors.
- The frozen `/home/psycontrol/Marco/assessor_lab` source and model bundles are
  checksum-pinned in `config_for_assessor_study.yaml`.
- The deprecated PHQ-conditioned assessor is deliberately not used.

Valence and arousal come from the assessor's mixed v4 bundle. Approach,
attention, control, and dominance come from its six-dimension v3 bundle.
Dominance remains exploratory because the assessor documentation reports weak
ground-truth reliability for that dimension.

## 1. Supply the blinded FOR group table

The available `allsubject_info.mat` is a MATLAB table that SciPy exposes as an
opaque object. Export or otherwise create a two-column CSV instead of inferring
diagnosis from subject numbers:

```bash
cp for_groups.example.csv for_groups.csv
```

Replace the example rows with all 50 subjects:

```csv
subject,group
sub-0027,healthy
sub-0108,depressed
```

Only `healthy` and `depressed` are accepted. The launcher checks that every
prediction subject appears exactly once. Keep this clinical file local and do
not commit it.

## 2. Run the read-only checks

```bash
./run_for_assessor_study.sh check
```

This validates group membership, all 50 FOR predictions, NSD inputs, VDVAE
assets, and the exact frozen assessor checksums. It does not fit or reconstruct.

## 3. Validate on NSD

```bash
./run_for_assessor_study.sh validate-nsd
```

The command:

1. Reserves 20% of the shared NSD images as a final test set before tuning.
2. Tunes ridge regularization on the remaining images. Every tuning fold holds
   out both one NSD subject and the validation image identities; another
   subject's response to the same image cannot leak into that fold.
3. Tests the selected setting on the untouched images, once for every held-out
   NSD subject, and compares it with a mean-latent baseline.
4. Reconstructs the held-out NSD predictions with the frozen VDVAE.
5. Extracts the matching original images and scores originals and
   reconstructions with both frozen assessors. The primary valence/arousal
   bundle also records calibrated intervals and OOD percentiles.
6. Writes an assessor-preservation report.

Review:

```text
artifacts/schaefer400_multiexpert/vdvae_decoder_validation/validation_summary.json
artifacts/schaefer400_multiexpert/for_assessor_study/validation/assessor_preservation.json
```

The first report must show that the ridge decoder improves over the
mean-latent baseline. The second shows whether each assessor property survives
the reconstruction step. Do not continue with properties that have inadequate
preservation or extreme OOD scores.

## 4. Explicitly approve the frozen validation

After reviewing the reports:

```bash
./run_for_assessor_study.sh approve-validation
```

This records the checksums of both reports and the frozen assessor. Any later
change invalidates the approval and prevents the FOR run.

## 5. Reconstruct and analyze FOR

Run each stage separately:

```bash
./run_for_assessor_study.sh reconstruct-for
./run_for_assessor_study.sh score-for
./run_for_assessor_study.sh analyze
```

Or, after validation approval, run all three sequentially:

```bash
./run_for_assessor_study.sh all-after-approval
```

The reconstruction stage reads the NSD-selected alpha, refits one shared
decoder on all eligible NSD examples, and produces 500 images for each of the
50 FOR subjects. All long stages resume existing valid outputs.

## Analysis outputs

The analysis writes:

```text
artifacts/schaefer400_multiexpert/for_assessor_study/analysis/
  image_level_scores.csv
  subject_level_scores.csv
  group_comparisons.csv
  technical_quality_comparisons.csv
  analysis_summary.json
```

The primary comparison is depressed FOR minus healthy FOR. Each participant,
not each image, is the independent observation. Every subject's reconstruction
score is first expressed relative to the assessor score of the corresponding
original image. P-values use subject-label permutation and primary dimensions
receive within-contrast false-discovery-rate correction.

Healthy FOR versus NSD and depressed FOR versus NSD are secondary reference
comparisons. The NSD reference is leakage-controlled but uses measured task
responses, whereas FOR uses model-predicted responses. It is therefore not a
matched clinical control group and must not replace the within-FOR comparison.
