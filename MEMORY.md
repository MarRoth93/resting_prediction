# Project Memory

## Environment & runtime
- Always use `/home/psycontrol/miniforge3/envs/resting-prediction/bin/python`;
  the base miniforge python lacks `h5py`, which `src/data/schaefer400.py:13`
  imports at module scope.
- Versatile Diffusion fetches from HuggingFace at load time; on this network run
  reconstruction with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (weights are
  cached locally). Evidence: `docs/IMPLEMENTATION_PLAN_V2.md` §1.2f.
- `benchmark_reconstructions_vdvae_vd` requires an ABSOLUTE `--recon-model-root`
  (relative paths break a `sys.path`+chdir interaction in
  `_load_versatile_components`, `benchmark_reconstructions_vdvae_vd.py:583`).
- `gh` CLI is not installed on this workstation.
- `/dev/nvme1n1`: 7.3 TB ext4 labeled `SSD_2`, unmounted, contents never
  inspected (needs sudo).

## FOR dataset
- Clinical mapping: `/media/psycontrol/HDD/Datasets/FOR/allsubject_info.mat` is
  a MATLAB table SciPy cannot parse (`MatlabOpaque`); `Proband` = column 1,
  `Group` = column 69 (1-based). Group 1 = healthy (MDD_Current=0, Lifetime=0);
  Group 2 = MDD-lifetime (18 of 25 current). The 25/25 mapping lives in the
  local, gitignored `for_groups.csv`.
- `voxel_timeseries/*_time_by_voxel.mat`: `voxel_ijk` is **1-based** (MATLAB);
  0-based indexing yields plausible-looking garbage (r ≈ 0.7 vs shipped
  parcels). Loaders (`src/data/voxel_contract.py`) enforce 1-based.
- No motion/FD/confound files exist anywhere under the FOR tree (verified
  2026-08-05). When requesting them upstream: ask ONLY for the raw 6
  rigid-body parameters, one row per retained volume — the repo derives
  motion6/12/friston24 itself (`prepare_rest_data.build_motion_confounds`).
- FSL MCFLIRT `.par` column trap: FSL writes rotations X/Y/Z (radians) first,
  then translations (mm); `compute_framewise_displacement`
  (`src/data/prepare_rest_data.py:46-63`) expects translations first. Any
  MCFLIRT importer must reorder columns before FD computation.
- FOR TR 1 is a T1-equilibration outlier in 34/50 subjects; the voxel contract
  discards the first 2 TRs (`data/processed_voxel_contract/contract.json`
  policies).

## Modeling facts (non-obvious, verified)
- The fingerprint's singular-value weighting (`F = C·P`) is load-bearing:
  column-normalizing it degrades cross-resolution alignment ~12× (Gate 3,
  `artifacts/gate3/seed42/gate3_summary.json`; raw variant selected for FOR).
- VDVAE latent regression targets are stochastic posterior samples
  (`third_party/vdvae/vae.py:119`); ~90% of dimensions have a test–retest
  ceiling of ≈0.06. Low global R²/correlation on the 91,168-d target is
  expected, not a failure signal; judge via identification vs shuffled
  controls.
- VDVAE latent calibration must be fit on MEASURED train fMRI and applied to
  condition inputs; fitting the calibration ridges on predicted-fMRI inputs
  makes gains explode (~4.7× over-dispersion). See `--vdvae-calibration
  per-condition` in `benchmark_reconstructions_vdvae_vd.py`.
- `notebooks/multiexpert_methods_explained.py` section 3: the cSRM objective
  plot is flat by construction (SVD init is already optimal on that synthetic
  data) — known cosmetic issue, not a bug.

## Protocol invariants
- `artifacts/model/`, `artifacts/results.json`, `config.yaml`,
  `run_pipeline.sh` are frozen. NSD subject 7 is evaluation-only; its two uses
  (frozen release; final model, D-08) are deliberate and recorded.
- Parity runs of the 499-seed configuration must load the frozen seed registry
  from `artifacts/model/external_seed_info.json`; recomputing seed defs over
  fold subjects changes the cache id and breaks parity
  (`train_voxel_contract.py`, nsd499 arm).
- The FOR clinical analysis design: connectivity/alignment features are the
  preregistered primary endpoint; reconstructions are illustrative only
  (decision D-02, `docs/IMPLEMENTATION_PLAN_V2.md`). The current group-analysis
  code has no motion covariate — adding mean FD is a code change, not config.
