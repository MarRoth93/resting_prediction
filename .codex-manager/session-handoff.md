## Session Handoff

Date: 2026-02-07 13:15 CET
Task: Implement resting-state prediction project
Plan verdict: APPROVED (after 5 review rounds with all critical issues addressed)
Change verdict: APPROVED (2 Codex code reviews — 8 total issues found and fixed)
Current step: Implementation COMPLETE — all fixes applied, tests pass (19/19)
Files changed: See list below
Verification run: `python -m pytest tests/ -v` → 19/19 passed
Open issues: None — ready for data download and execution
Next command: Download NSD data and run pipeline

## Codex Review Round 1 Fixes (4 issues):
1. **Critical fix:** Few-shot Procrustes now uses correct template rows via `shot_indices` parameter
2. **Critical fix:** strict_rest_cha mode blocked for few-shot with clear error
3. **High fix:** SharedSpaceBuilder save/load now persists all config params (min_k, ensemble_method, max_iters, tol)
4. **Medium fix:** Empty REST runs validation added to compute_rest_connectivity

## Codex Review Round 2 Fixes (4 issues):
5. **High fix:** REST run save uses contiguous indices (no gaps from excluded runs) — prepare_rest_data.py
6. **High fix:** Shared-stimulus row correspondence validated across subjects — train_shared_space.py
7. **Medium fix:** Few-shot sampling guards against negative size and reports actual shot count — predict_subject.py
8. **Medium fix:** `fine_tune_encoder` alpha=0.0 truthy bug fixed with `is not None` check — encoding.py

## Codex Mathematical Validation (confirmed correct):
- `predict_voxels`: Y_hat = Z_hat @ R.T @ P.T is correct inverse mapping
- Hybrid template fitting and rotation usage are internally consistent
- Fingerprint calibration mean(C_s @ P_s @ R_s) and zero-shot Procrustes are coherent
- No train/test data leakage detected

## Files Created:
### Infrastructure
- config.yaml — full configuration
- requirements.txt — all dependencies
- how_to_use.md — complete usage guide

### src/data/
- prepare_task_data.py — task fMRI preprocessing (dynamic sessions, canonical ordering, trial-level test data)
- prepare_rest_data.py — REST preprocessing (detrend, highpass, motion censoring, z-score)
- nsd_loader.py — NSDSubjectData + NSDFeatures lazy loaders
- load_atlas.py — atlas loading, combined ROIs, harmonization, QC
- prepare_features.py — CLIP + DINOv2 extraction

### src/alignment/
- utils.py — Procrustes (with NaN/variance safety), SVD basis, global k
- rest_preprocessing.py — parcellation + voxel connectivity computation
- shared_space.py — SharedSpaceBuilder (hybrid_cha + strict_rest_cha modes)
- cha_alignment.py — zero-shot CHA fingerprint alignment

### src/models/
- encoding.py — SharedSpaceEncoder (ridge with explicit intercept) + fine-tuning
- baseline_mni.py — MNI baseline ridge model

### src/evaluation/
- metrics.py — voxelwise_correlation, pattern_correlation, 2v2, noise ceiling, ROI eval
- visualize.py — histograms, few-shot curves, ROI comparison, pattern plots

### src/pipelines/
- train_shared_space.py — full training pipeline with provenance
- predict_subject.py — zero-shot + few-shot prediction
- run_ablations.py — few-shot sweep experiments

### tests/
- conftest.py — synthetic data fixtures
- test_shapes.py — 16 unit tests (shapes, safety checks)
- test_pipeline.py — 3 integration tests (alignment, encoder, save/load)
