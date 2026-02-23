# Copilot Instructions for `resting_prediction`

## Big picture (read this first)
- This project predicts subject-specific NSD task fMRI from resting-state fMRI via a shared latent space.
- Main flow is: data prep (`src/data`) → shared-space training (`src/pipelines/train_shared_space.py`) → inference/ablations (`src/pipelines/predict_subject.py`, `src/pipelines/run_ablations.py`).
- Core alignment logic is in `src/alignment/shared_space.py`:
  - REST connectivity → subject basis `P_s` (SVD)
  - Procrustes rotations `R_s` to a shared template
  - zero-shot for new subject uses fingerprint alignment (`src/alignment/cha_alignment.py`)

## Architecture and boundaries
- `src/data/*`: converts raw NSD assets into canonical `.npy` artifacts in `processed_data/subjXX/`.
- `src/alignment/*`: CHA/shared-space math; keep it independent of file I/O when possible.
- `src/models/encoding.py`: feature→shared-space encoder (ridge baseline + few-shot fine-tune helper).
- `src/pipelines/*`: orchestration, metrics, persistence, and CLI entrypoints.
- `src/evaluation/*`: metrics/statistics/visualization utilities consumed by pipeline scripts.

## Project-specific data contracts (critical)
- `test_stim_idx.npy` row order must be identical across subjects; training enforces this and raises on mismatch.
- Subject arrays are strict shape contracts (`train_fmri`, `test_fmri`, `*_stim_idx`, optional trial-level test arrays); validate before adding new logic.
- REST runs are saved as contiguous filenames (`rest_run1.npy`, `rest_run2.npy`, ...). Loader stops at first gap (`NSDSubjectData.rest_runs`).
- `NSDFeatures` expects 0-based NSD stimulus indices and supports `clip`, `dinov2`, and `clip_dinov2`.
- Zero-shot alignment requires `alignment.connectivity_mode: parcellation`; `voxel_correlation` is incompatible for cross-subject zero-shot.

## Configuration and defaults
- Single source of runtime knobs is `config.yaml` (subjects, alignment mode, feature backbone, eval split policy, statistics).
- Current default subjects in config are train `[1,2,3,4,5,6]`, test `[7]` (do not assume older 1/2/5-only setup).
- Prefer reading config values in pipeline code instead of hardcoding constants.

## Developer workflows (local)
- Prepare task data: `python -m src.data.prepare_task_data -sub 7`
- Prepare REST data: `python -m src.data.prepare_rest_data -sub 7 --config config.yaml`
- Train shared space: `python -m src.pipelines.train_shared_space --config config.yaml`
- Predict: `python -m src.pipelines.predict_subject --mode zero_shot --test-sub 7`
- Ablations: `python -m src.pipelines.run_ablations --config config.yaml`
- Tests: `pytest -q`

## HPC workflow (from existing conventions)
- SLURM chain lives in `slurm_scripts/`; use `slurm_scripts/submit_full_pipeline.sh` for dependency-managed execution.
- Script defaults/overrides (e.g., `CONDA_ENV`, `PROJECT_DIR`, benchmark toggles) are documented in `slurm_scripts/README.md`.

## Patterns to preserve when editing
- Keep arrays `float32` for large tensors unless precision-sensitive code requires otherwise.
- Use deterministic seeding (`random`, `numpy`, `torch`) as done in training/ablation pipelines.
- Persist reproducibility artifacts (`metadata.json`, eval splits under `outputs/shared_space/eval_splits/`).
- Prefer extending existing pipeline CLIs rather than adding one-off scripts.
