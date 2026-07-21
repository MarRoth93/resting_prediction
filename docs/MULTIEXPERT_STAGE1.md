# Stage-1 multi-expert resting-state embedding

This is an experimental path beside the frozen pipeline. `config.yaml`,
`run_pipeline.sh`, and `artifacts/model` remain the production reference.

The new model learns from two views of the same resting-state data:

- `hybrid_cha`: the existing connectivity-hyperalignment method.
- `connectivity_srm`: a REST-only shared-response factorization.

Each method predicts through its own shared latent space. The model decodes both
predictions into the current subject's voxels and learns how much to trust each
method in each HCP-MMP region. During training, either method can be hidden at
random, but at least one is always available. This is the TRIBE-v2-style
modality-dropout principle applied to alignment methods rather than sensor types.

## Commands

```bash
./run_multiexpert.sh check
./run_multiexpert.sh prepare-reliability
CHECK_SCOPE=loso ./run_multiexpert.sh check
./run_multiexpert.sh loso
./run_multiexpert.sh train
./run_multiexpert.sh predict
```

`FOLD=1 LOSO_SEED=42 ./run_multiexpert.sh loso` runs one resumable LOSO unit.
Completed fold results are reused when the full command is run later.
Each fold defines its connectivity-seed vocabulary from its five training
subjects only; the held-out subject cannot influence which seeds are retained.
If a retained anatomical label does not exist in the held-out brain, its seed
row is recorded as unavailable, filled with zero, and excluded from the
hyperalignment comparison. This keeps the fold executable without consulting
held-out responses or changing its training-defined vocabulary.

The current processed data already pass the training check. Subjects 1-6 do not
yet contain the trial-level arrays needed for the reliability part of the LOSO
gate, so `prepare-reliability` must be run before LOSO. That command validates
the existing mask, stimulus order, and averaged test responses, then writes only
the two missing trial-level files; it does not rewrite frozen task arrays.

## Artifacts

The full model is written below `artifacts/multiexpert/model/`:

```text
manifest.json
effective_config.json
external_seed_info.json
region_registry.json
shared_stim_idx.npy
experts/
  hybrid_cha/
  connectivity_srm/
encoder/
```

LOSO folds, predictions, and the final gate live below
`artifacts/multiexpert/loso/`. Subject 7 cannot be predicted by the standard
launcher until `gate.json` records a complete pass over subjects 1-6 and seeds
42-46.

Completed folds checksum the processed task, REST, CLIP, reliability, atlas,
and external-seed-cache inputs they consumed. A resumed fold is rejected if
those inputs or the frozen comparison config have changed.

The gate compares the dropout-fusion model with the actual current hybrid-CHA
encoder. It requires all of the following:

- mean zero-shot LOSO median voxel correlation improves by at least `0.005`;
- the new method wins for at least four of six subjects;
- the median result among voxels with noise ceiling at least `0.3` does not fall
  by more than `0.002`.

For NSD's three repeats, the noise ceiling uses the stimuli that have all three
available trials, treats all repeat pairings symmetrically, and then corrects
reliability to the three-trial average. This remains valid for participants who
did not finish every NSD session.

The approved 100-shot calibration is kept unchanged. Because it centers exactly
100 examples in a 100-dimensional latent space, one direction is numerically
ambiguous; this affects all few-shot comparisons equally and does not affect the
zero-shot promotion gate. A 101-shot or regularized variant is a later option.

Connectivity hyperalignment inside regions and a VAE connectivity embedding are
not included in Stage 1. They remain Stage-2 candidates only if this two-expert
version clears the measured gate.
