# Predicting Task Activation from Resting State Data — Implementation Plan

## Goal

Build a system that predicts subject-specific fMRI visual activation patterns (NSD task betas) from resting-state fMRI data alone. Train on subjects 1, 2, 5; predict subject 7's voxel-wise responses to images using only subject 7's resting state (zero-shot), with optional few-shot fine-tuning.

---

## 0. Definitions & Scope

- **Zero-shot**: No task data from subject 7 at all. Alignment uses only REST. Training subjects (1, 2, 5) may use task data for template orientation.
- **Few-shot**: Subject 7 provides N responses to shared stimuli for alignment/fine-tuning. The remaining shared stimuli serve as held-out evaluation.
- **Reproducibility**: All randomized operations use fixed seeds (default: 42). Few-shot ablations use 5 repeated random splits and report mean ± std.
- **Global random seed**: At pipeline entry, set all sources of randomness:
  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42)
  torch.manual_seed(42); torch.cuda.manual_seed_all(42)
  ```
  Per-function seeds via `np.random.RandomState(seed)`.
- **Artifact provenance**: Save with each model artifact a `metadata.json` containing: config hash (SHA256 of config.yaml), data manifest hashes (SHA256 of each input .npy file path list), git commit hash (if available), timestamp, and Python/library versions.
- **Connectivity modes**:
  - `parcellation` (required for zero-shot CHA): Uses atlas parcels as common reference → C is (R × V). Enables cross-subject fingerprint alignment because R is the same for all subjects. **Default for zero-shot.**
  - `voxel_correlation`: Uses full V × V correlation matrix. Cannot be used for zero-shot CHA (V differs across subjects). Useful for within-subject encoding or when few-shot alignment is guaranteed.
- **Task-based connectivity**: Computing connectivity from task betas instead of REST. This is **not zero-shot** for subject 7 (uses task data). Classified as a "pseudo-REST fallback" — only applicable if REST timeseries are unavailable AND few-shot mode is accepted.
- **Experiment modes** (explicit, mutually exclusive):
  - `strict_rest_cha`: ALL subjects use REST-only connectivity for basis P_s. Training subjects' task responses are used ONLY for the encoding model (stimulus features → shared space), NOT for alignment. This is the purest zero-shot design.
  - `hybrid_cha` (DEFAULT): Training subjects use REST for P_s AND shared task responses for Procrustes orientation of the template. Subject 7 uses REST-only CHA fingerprint alignment. **Space-bridging requirement**: Training rotations R_s are learned in task-projected space (Z_s = Y_shared @ P_s), but R_7 is learned in fingerprint space (F_7 = C_7 @ P_7). To ensure these spaces are compatible, add a **calibration step**: after learning the task-based template, project it into fingerprint space using training subjects (F_template = mean(C_s @ P_s @ R_s)), then align subject 7's fingerprint to this calibrated template. Both spaces share the same P_s and R_s, so the fingerprint-space template inherits the task-based orientation. This is the practical default.
  - `mixed_connectivity`: Some subjects use REST, others use task-derived connectivity. Not recommended; only as fallback if REST is unavailable for a training subject.

  **Design rationale for `hybrid_cha` as default**: Using training subjects' task responses for template orientation does NOT violate the zero-shot constraint for subject 7 (no subject 7 task data is used). It only means the shared space template is better oriented. This is standard practice in CHA literature.

- **Transductive evaluation note**: The shared1000 stimuli are used for template orientation (training subjects) and also for evaluation (subject 7). The shared1000 image IDs are known during training. However:
  - The encoding model is trained on subject-SPECIFIC images (9k/subject), NOT on shared1000.
  - Template orientation uses training subjects' responses to shared1000, not subject 7's.
  - Subject 7's shared1000 responses are held out entirely.
  - This is NOT transductive w.r.t. subject 7's data, but IS stimulus-aware. Label this as "shared-stimulus evaluation" in reporting.

- **Subject weighting**: When pooling training data across subjects with different session counts/trial numbers:
  - Default: equal sample weight (just concatenate). Simple and standard.
  - Optional: equal subject weight — subsample larger subjects to match smallest. Use when subject data volumes differ by >2x.
- **Atlas harmonization for subject 7**: Subject 7's atlas labels ARE included in the intersection computation during harmonization. This is not a data leak because atlas labels are anatomical (from retinotopy/localizer), not task-derived. Subject 7's atlas is needed to ensure parcels exist in all subjects.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Extraction](#3-feature-extraction)
4. [Handling Variable Voxel Counts](#4-handling-variable-voxel-counts-the-big-problem)
5. [Approach A — Baseline Ridge in Common Space](#5-approach-a--baseline-ridge-in-common-space)
6. [Approach B — Hyperalignment Shared Space](#6-approach-b--hyperalignment-shared-space)
7. [Approach C — Deep Learning Encoder](#7-approach-c--deep-learning-encoder)
8. [Zero-Shot vs Few-Shot Strategies](#8-zero-shot-vs-few-shot-strategies)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Repository Structure](#10-repository-structure)
11. [Implementation Steps (Ordered)](#11-implementation-steps-ordered)
12. [Configuration & Hyperparameters](#12-configuration--hyperparameters)
13. [Risks & Mitigations](#13-risks--mitigations)

---

## 1. Problem Analysis

### Core Challenge

Given:
- **Training subjects** (1, 2, 5): Both resting-state fMRI and task fMRI (image viewing betas)
- **Test subject** (7): Only resting-state fMRI (+ optionally a few task trials)

Predict: Subject 7's voxel-wise fMRI response to any image.

### Key Difficulties

1. **Variable voxel counts**: The `nsdgeneral` mask is subject-specific. Each subject has a different number of voxels in visual cortex (e.g., subj01 ~15,000, subj02 ~14,500, etc.). You cannot directly stack or compare voxel vectors across subjects.

2. **Anatomical variability**: Even if voxel counts matched, the same voxel index across subjects corresponds to different neural populations. Functional alignment is required.

3. **Resting state → Task bridge**: Resting-state connectivity patterns must encode enough subject-specific information to enable cross-subject generalization. The hypothesis is that resting-state functional connectivity captures the subject's representational geometry.

4. **Dimensionality**: Each subject has ~8,000–15,000 nsdgeneral voxels and ~10,000 unique training images. The feature-to-voxel mapping is high-dimensional.

### Scientific Basis

- **Hyperalignment** (Haxby et al., 2011): Aligns individual brains into a common representational space using orthogonal transformations.
- **Connectivity-based Hyperalignment (CHA)** (Guntupalli et al., 2016): Uses resting-state connectivity instead of task data for alignment, enabling zero-shot scenarios.
- **Encoding models**: Ridge regression from stimulus features (e.g., CLIP embeddings) to neural responses is the standard NSD benchmark approach.

---

## 2. Data Pipeline

### 2.1 Data to Download

#### Task Data (already handled by existing scripts)
- **Betas**: `betas_fithrf_GLMdenoise_RR` — up to 40 sessions × 750 trials per subject (session count varies by subject; discover dynamically, do NOT hardcode 37)
- **Experiment design**: `nsd_expdesign.mat` — stimulus ordering, repetition structure
- **Stimuli**: `nsd_stimuli.hdf5` — 73,000 images (425×425×3)
- **ROI masks**: `nsdgeneral.nii.gz` per subject

#### Resting-State Data (NEW — needs download script)
NSD includes resting-state fMRI runs collected during the experiment. The timeseries bucket contains both task and rest runs.

**Important: Metadata-driven REST run discovery.** The timeseries directory contains ALL functional runs (task + rest). REST runs MUST be identified from NSD metadata, NOT by filename pattern matching (`*rest*` is fragile and may not match NSD's naming convention).

**Discovery protocol:**
1. Load `nsd_expdesign.mat` → check for session/run metadata fields (e.g., `isrest`, `runtype`)
2. If metadata fields exist: use them to identify REST run indices per subject
3. If no metadata: list all timeseries files per subject via `aws s3 ls`, then identify REST runs by:
   a. Checking NSD documentation/README for naming convention
   b. Cross-referencing with known task session counts (NSD has up to 40 sessions; non-task runs are REST)
4. **Validation**: After discovery, assert ≥ 2 REST runs per subject. If < 2 runs found, raise error with subject ID.
5. **Store discovered REST run paths** in `processed_data/subj{XX}/rest_run_manifest.json` for reproducibility.

```
s3://natural-scenes-dataset/nsddata_timeseries/ppdata/subj{XX}/func1pt8mm/timeseries/
```

**Estimated storage per subject**: ~2-5 GB for REST runs (4-6 runs × ~300 TRs × 1.8mm resolution). Total for 4 subjects: ~10-20 GB.

**Fallback (pseudo-REST, NOT zero-shot)**: If REST timeseries are unavailable or too large, compute connectivity from the task betas (parcel-voxel correlations from a random subset of stimuli). **WARNING**: For subject 7, this requires task data and therefore violates the zero-shot constraint. This fallback is only valid for:
- Training subjects (1, 2, 5): acceptable, since we already have their task data
- Subject 7 in **few-shot mode only**: if we accept using task responses for connectivity computation
- If REST data is truly unavailable for ANY subject, the zero-shot CHA path for subject 7 is disabled; only few-shot and MNI baseline remain viable.

#### Annotations (optional, for captions)
- `COCO_73k_annots_curated.npy` — already referenced in prepare_nsddata.py

### 2.1b Data Contract (Expected Shapes & Sizes)

| Data | Path | Shape | Size (est.) |
|------|------|-------|-------------|
| Betas session | `nsddata_betas/.../betas_session{NN}.nii.gz` | (X, Y, Z, 750) ~(81, 104, 83, 750) | ~1.5 GB/session |
| nsdgeneral mask | `nsddata/.../roi/nsdgeneral.nii.gz` | (81, 104, 83) | ~1 MB |
| Experiment design | `nsd_expdesign.mat` | masterordering: (27750,), subjectim: (8, 10000) | ~5 MB |
| Stimuli | `nsd_stimuli.hdf5` | imgBrick: (73000, 425, 425, 3) | ~26 GB |
| REST run (timeseries) | `nsddata_timeseries/.../timeseries/` | (X, Y, Z, ~300) per run | ~500 MB/run |
| Visual ROI atlas | `nsddata/.../roi/prf-visualrois.nii.gz` | (81, 104, 83) | ~1 MB |
| Kastner atlas | `nsddata/.../roi/Kastner2015.nii.gz` | (81, 104, 83) | ~1 MB |

**Expected voxel counts** (nsdgeneral, to verify after download):
- Subj01: ~15,000-16,000
- Subj02: ~14,000-15,000
- Subj05: ~13,000-14,000
- Subj07: ~12,000-13,000

**Expected atlas region counts** (within nsdgeneral):
- prf-visualrois: ~7 regions (V1v, V1d, V2v, V2d, V3v, V3d, hV4)
- Kastner2015: ~25 regions (more visual areas)
- Both are too few for `n_components=300`. See Section 12 for adaptive k strategy.

### 2.2 Download Script Updates

**File: `download_nsddata.py`** — Add resting-state downloads with metadata-driven filtering:

```python
# Download resting-state timeseries (metadata-driven, NOT wildcard)
# Step 1: List available timeseries files per subject
# Step 2: Identify REST runs from NSD metadata or documentation
# Step 3: Download only identified REST files
import json

def discover_rest_runs(sub: int) -> list[str]:
    """
    Discover REST run filenames for a subject.
    Uses NSD experiment metadata or falls back to listing + filtering.
    Returns list of S3 keys for REST runs.
    """
    # First try: check nsd_expdesign.mat for run type metadata
    # Second try: aws s3 ls and identify non-task runs
    # Must return >= 2 runs or raise ValueError
    ...

for sub in [1, 2, 5, 7]:
    rest_files = discover_rest_runs(sub)
    for rest_file in rest_files:
        os.system(
            f'aws s3 cp s3://natural-scenes-dataset/nsddata_timeseries/ppdata/'
            f'subj{sub:02d}/func1pt8mm/timeseries/{rest_file} '
            f'nsddata_timeseries/ppdata/subj{sub:02d}/func1pt8mm/timeseries/'
        )
    # Save manifest for reproducibility
    manifest_path = f'processed_data/subj{sub:02d}/rest_run_manifest.json'
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump({'rest_runs': rest_files, 'subject': sub}, f)

# Download additional ROI atlases for parcellation
for sub in [1, 2, 5, 7]:
    for roi_file in ['prf-visualrois.nii.gz', 'Kastner2015.nii.gz',
                     'prf-eccrois.nii.gz', 'floc-bodies.nii.gz',
                     'floc-faces.nii.gz', 'floc-places.nii.gz',
                     'floc-words.nii.gz']:
        os.system(
            'aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/'
            'subj{:02d}/func1pt8mm/roi/{} '
            'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub, roi_file, sub)
        )
```

**Note**: If `discover_rest_runs()` finds < 2 runs for any subject, it raises an error. The pipeline then decides:
- Training subjects: fall back to task-based connectivity (acceptable, but marks this as "mixed-CHA")
- Subject 7 zero-shot: ABORT zero-shot mode entirely; switch to few-shot
The `prepare_rest_data.py` script must handle both REST and task-connectivity scenarios.

### 2.3 Preprocessing Pipeline

**File: `src/data/prepare_task_data.py`**

Refactored version of `prepare_nsddata.py` with these changes:

```python
def prepare_task_data(sub: int, data_root: str, output_root: str) -> dict:
    """
    Prepare averaged task fMRI for one subject.

    Returns:
        dict with keys:
        - 'train_fmri': np.ndarray (N_train, V_sub) — averaged betas per stimulus
        - 'test_fmri': np.ndarray (N_test, V_sub)
        - 'train_stim_idx': np.ndarray (N_train,) — NSD image indices (0-based)
        - 'test_stim_idx': np.ndarray (N_test,)
        - 'mask': np.ndarray (X, Y, Z) — nsdgeneral mask
        - 'num_voxels': int — V_sub
    """
```

Key processing:
1. Load `nsd_expdesign.mat` → get masterordering and subjectim
2. Split by masterordering threshold (>1000 = train, <=1000 = test/shared)
3. Load nsdgeneral mask → count voxels (V_sub varies per subject)
4. **Discover sessions dynamically**: glob `betas_session*.nii.gz` files rather than hardcoding session count (NSD session counts vary by subject: subj01 has 40, others may have fewer). Load all available sessions, apply mask.
5. **Save averaged data for both sets; trial-level ONLY for test/shared (for noise ceiling)**:
   - Train averaged: `train_fmri.npy` (N_train_stim × V) — averaged over repeated presentations
   - Train stimulus indices: `train_stim_idx.npy` (N_train_stim,) — NSD image IDs
   - **Test/shared averaged**: `test_fmri.npy` (N_test_stim × V) — averaged over repeated presentations
   - **Test/shared trial-level**: `test_fmri_trials.npy` (N_test_trials × V) — individual trial betas (REQUIRED for noise ceiling)
   - **Test/shared trial labels**: `test_trial_labels.npy` (N_test_trials,) — which stimulus each trial corresponds to
   - **Note**: Train trial-level data is NOT saved (too large: ~27,750 trials × V_sub × float32 ≈ 1.6 GB/subject). Noise ceiling is computed on test/shared stimuli only. Alternatively, use NSD-provided `ncsnr.nii.gz` maps.
6. Save stimulus indices sorted by NSD image ID for canonical cross-subject ordering
7. Save as `processed_data/subj{XX}/train_fmri.npy`, `test_fmri.npy`, etc.

**Critical**: The shared1000 stimulus indices must be stored in **canonical NSD image ID order** (sorted by nsdId). This ensures that row i of `test_fmri.npy` corresponds to the same image across all subjects, which is required for Procrustes alignment.

**File: `src/data/prepare_rest_data.py`**

```python
def prepare_rest_data(sub: int, data_root: str, output_root: str) -> dict:
    """
    Prepare resting-state data for one subject.

    Returns:
        dict with keys:
        - 'rest_runs': list[np.ndarray] — each (T_run, V_sub)
        - 'mask': np.ndarray — nsdgeneral mask
        - 'num_voxels': int
    """
```

Processing:
1. Load resting-state timeseries NIfTI files
2. Apply nsdgeneral mask
3. **REST preprocessing policy** (applied per run, in order):
   a. Discard first 5 TRs (T1 equilibration)
   b. Linear detrending (remove drift; `scipy.signal.detrend` along time axis)
   c. High-pass filter: Butterworth 0.01 Hz (removes scanner drift). **TR is read from NIfTI header** (`img.header.get_zooms()[-1]`), NOT hardcoded. Expected NSD TR ≈ 1.333s → Nyquist ≈ 0.375 Hz. If TR differs from expected, log a warning.
   d. **Motion censoring/scrubbing** (if motion params available):
      - Compute framewise displacement (FD) from motion parameters
      - Censor TRs with FD > 0.5mm: **drop censored TRs entirely** (do NOT interpolate — interpolation introduces artifacts in connectivity estimates)
      - If > 30% of TRs are censored in a run, exclude the entire run
      - Log censoring statistics per subject per run
   e. Nuisance regression (optional): regress out 6 motion parameters + derivatives if available; skip if not
   f. Z-score within each run (voxel-wise, mean=0, std=1)

   **Minimum retained data**: After preprocessing, require ≥ 100 usable TRs per subject total across all REST runs. If not met, follow failure policy (Section 6.2.1).
4. Save as `processed_data/subj{XX}/rest_run{N}.npy`

**Note**: NSD betas (`betas_fithrf_GLMdenoise_RR`) are already denoised (GLMdenoise + Ridge Regression). REST timeseries are raw, so explicit preprocessing is required.

**File: `src/data/prepare_features.py`**

```python
def extract_clip_features(stimuli_path: str, output_path: str, batch_size: int = 64):
    """
    Extract CLIP (ViT-L/14) features for all 73k NSD stimuli.

    Output: (73000, 768) float32 array saved to output_path.
    """

def extract_dinov2_features(stimuli_path: str, output_path: str, batch_size: int = 64):
    """
    Extract DINOv2 (ViT-L/14) features for all 73k NSD stimuli.

    Output: (73000, 1024) float32 array.
    """
```

### 2.4 Data Loading Utilities

**File: `src/data/nsd_loader.py`**

```python
class NSDSubjectData:
    """Unified data loader for one NSD subject."""

    def __init__(self, sub: int, data_root: str = "processed_data"):
        self.sub = sub
        self.data_root = data_root

    @cached_property
    def train_fmri(self) -> np.ndarray:
        """(N_train, V_sub) averaged task betas."""
        ...

    @cached_property
    def test_fmri(self) -> np.ndarray:
        """(N_test, V_sub) averaged task betas."""
        ...

    @cached_property
    def train_stim_idx(self) -> np.ndarray:
        """NSD image indices for training stimuli."""
        ...

    @cached_property
    def test_stim_idx(self) -> np.ndarray:
        """NSD image indices for test stimuli."""
        ...

    @cached_property
    def rest_runs(self) -> list[np.ndarray]:
        """List of (T_run, V_sub) resting-state arrays."""
        ...

    @cached_property
    def mask(self) -> np.ndarray:
        """3D nsdgeneral mask."""
        ...

    @cached_property
    def num_voxels(self) -> int:
        """Number of masked voxels for this subject."""
        ...

class NSDFeatures:
    """Feature loader for all NSD stimuli."""

    def __init__(self, features_dir: str = "processed_data/features"):
        ...

    def get_features(self, stim_indices: np.ndarray, feature_type: str = "clip") -> np.ndarray:
        """Get features for specific stimuli by NSD index."""
        ...
```

---

## 3. Feature Extraction

### 3.1 Stimulus Feature Types

| Feature | Model | Dims | Use Case |
|---------|-------|------|----------|
| CLIP ViT-L/14 | openai/clip | 768 | Primary — best for semantic encoding |
| DINOv2 ViT-L/14 | facebook/dinov2 | 1024 | Backup — good for visual features |
| CLIP + DINOv2 concat | — | 1792 | Combined semantic + visual |
| VGG-16 fc6 | torchvision | 4096 | Baseline (commonly used in NSD literature) |

### 3.2 Feature Extraction Pipeline

```python
# src/features/extract_features.py

def extract_all_features(
    stimuli_hdf5: str = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
    output_dir: str = "processed_data/features",
    models: list[str] = ["clip", "dinov2"],
    device: str = "cuda",
    batch_size: int = 64,
):
    """
    Extract features from all 73k NSD stimuli using specified models.

    Saves:
    - output_dir/clip_features.npy — (73000, 768)
    - output_dir/dinov2_features.npy — (73000, 1024)
    """
```

### 3.3 Shared Stimuli Across Subjects

NSD shares ~1,000 images across all subjects (the "shared1000" set, masterordering ≤ 1000). These are critical for:
- Cross-subject alignment validation
- Few-shot fine-tuning
- Noise ceiling estimation

The `prepare_nsddata.py` already separates these as the "test" set. We keep this split:
- **Train set**: Subject-specific images (~9,000 per subject, masterordering > 1000)
- **Shared set**: 1,000 images seen by all subjects (masterordering ≤ 1000)

---

## 4. Handling Variable Voxel Counts (The Big Problem)

### The Problem

Subject voxel counts under nsdgeneral:
- Subj01: ~15,724 voxels
- Subj02: ~14,278 voxels
- Subj05: ~13,039 voxels
- Subj07: ~12,682 voxels

(Exact numbers depend on the mask — to be verified after download.)

You cannot directly average, stack, or compare voxel vectors across subjects because:
1. Different vector lengths
2. Even if padded, voxel i in subject 1 ≠ voxel i in subject 7

### Solution Strategy: Project to a Common Low-Dimensional Space

**Core idea**: Each subject's voxels are projected into a **common k-dimensional space** via resting-state derived transformations. The encoding model operates in this shared space, not in voxel space.

```
Subject s voxel space (V_s dims)
        ↓  P_s (V_s × k basis from REST)
Subject s component space (k dims)
        ↓  R_s (k × k rotation to shared)
Shared space (k dims) ← SAME for all subjects
```

**This completely sidesteps the variable voxel problem** because:
- P_s is computed independently per subject from their own REST data
- P_s has shape (V_s, k) — subject-specific number of rows, common k columns
- The shared space has fixed dimensionality k for all subjects
- The encoding model maps: features (F dims) → shared space (k dims)

### Concrete Implementation

1. **Per-subject REST basis** P_s ∈ ℝ^(V_s × k):
   - Compute resting-state connectivity matrix C_s
   - SVD of C_s → take top-k right singular vectors → P_s

2. **Shared space** S ∈ ℝ^(V_common × k) — but V_common doesn't matter because we only use the k-dimensional projections:
   - Actually, the shared space is defined as a k-dimensional abstract space
   - Alignment rotations R_s ∈ ℝ^(k × k) map each subject's component space to the shared space

3. **Encoding model**: X (features) → Z_shared (k dims) via ridge regression

4. **Prediction for new subject u**:
   - Compute P_u from REST_u (V_u × k)
   - Align: R_u = Procrustes(P_u, S)
   - Predict: Z_hat = ridge(X_new) → Y_hat = Z_hat @ R_u.T @ P_u.T → (N × V_u)

### Important Detail: Procrustes Alignment with Different V

The standard Procrustes alignment requires P_s and S to have the same shape. Since V differs across subjects, we **cannot** directly compare P matrices in voxel space.

**Solution approaches:**

#### Option A: Alignment via Shared Stimuli Responses (Recommended)

Instead of aligning P matrices directly, align via the **response space** to shared stimuli:

1. Each subject has responses Y_s to the shared1000 images: (1000, V_s)
2. Project to components: Z_s = Y_s @ P_s → (1000, k)
3. Align Z matrices (which are all 1000 × k) using Procrustes
4. This works because the shared stimuli create a common reference frame

```
For training subjects (1, 2, 5):
    Z_s = Y_shared_s @ P_s    # (1000, k) — all same shape!
    R_s = Procrustes(Z_s, Z_template)    # align in k-space

For test subject 7 (zero-shot):
    # No Y available! Use connectivity structure instead.
    # See Section 6 for CHA-based alignment.
```

#### Option B: CHA-Style Alignment via REST Connectivity

Use resting-state connectivity patterns (which can be computed in a common parcellation space) to derive alignment. This is the approach from the reference project.

1. Use an atlas (e.g., Schaefer parcellation, registered to each subject's space) to define R common target regions
2. Compute parcel-to-voxel connectivity: C_s is (R × V_s)
3. SVD of C_s → P_s (V_s × k)
4. Now define "connectivity fingerprints" in parcel space: F_s = C_s @ P_s → (R × k)
5. Align F matrices (all R × k, same shape!) using Procrustes

**This enables zero-shot because it only requires REST data.**

#### Option C: MNI Resampling (Simplest but lossy)

Resample all subjects to MNI space at the same resolution, apply a common mask, getting identical voxel counts. This loses subject-specific detail but is the simplest approach.

### Recommended Approach

**Use Option B (CHA-style) as the primary method**, with Option A as validation. Option C as the simplest baseline.

---

## 5. Approach A — Baseline Ridge in Common Space

### 5.1 Overview

Simplest viable approach using MNI resampling:

1. Resample all subjects' betas to MNI 2mm space with a common visual cortex mask
2. Average betas across repeated presentations
3. Train ridge regression: CLIP features → MNI voxels (pooling subjects 1,2,5)
4. Predict subject 7's MNI voxels from CLIP features
5. Optionally resample predictions back to subject 7's native space

### 5.2 Implementation

**File: `src/models/baseline_mni.py`**

```python
class MNIBaselineModel:
    """
    Simple baseline: ridge regression in MNI space.

    Pools training subjects in a common MNI mask and trains
    a single encoding model.
    """

    def __init__(self, alpha: float = 100.0, feature_type: str = "clip"):
        self.alpha = alpha
        self.feature_type = feature_type
        self.model = None

    def fit(
        self,
        subjects_data: list[NSDSubjectData],
        features: NSDFeatures,
        mni_mask: np.ndarray,
    ):
        """
        Train ridge from features to MNI-space voxels.
        Concatenates all training subjects.
        """
        X_all, Y_all = [], []
        for subj in subjects_data:
            X = features.get_features(subj.train_stim_idx, self.feature_type)
            Y = resample_to_mni(subj.train_fmri, subj.mask, mni_mask)
            X_all.append(X)
            Y_all.append(Y)

        X_concat = np.concatenate(X_all, axis=0)
        Y_concat = np.concatenate(Y_all, axis=0)

        # Ridge regression
        self.model = fit_ridge(X_concat, Y_concat, self.alpha)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict MNI-space activation from features."""
        return predict_ridge(X_new, self.model)
```

### 5.3 Limitations

- MNI resampling blurs fine-grained patterns
- Ignores individual functional topography
- Does not leverage resting-state data at all
- Subject 7 predictions are in MNI space, not native space

### 5.4 But: Useful as a lower bound and sanity check.

---

## 6. Approach B — Hyperalignment Shared Space (Primary)

### 6.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                             │
│                                                               │
│  For each training subject s ∈ {1, 2, 5}:                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  REST_s → Connectivity C_s → SVD → Basis P_s (V_s×k) │   │
│  │  TASK_s → mask → zscore → Y_s (N_s × V_s)           │   │
│  │  Z_s = Y_s @ P_s  →  (N_s × k) component responses  │   │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  Shared Space Alignment:                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Use shared1000 responses Z_shared_s (1000 × k)     │   │
│  │  Procrustes: R_s = align(Z_s_shared, Z_template)    │   │
│  │  Z_s_aligned = Z_s @ R_s  →  (N_s × k)             │   │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  Global Encoder:                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  X_all = concat(X_1, X_2, X_5)    (N_total × F)    │   │
│  │  Z_all = concat(Z_1_aligned, Z_2_aligned, Z_5_aligned)│  │
│  │  Ridge: W = (X'X + αI)⁻¹ X'Z     (F × k)          │   │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE (Subject 7)                      │
│                                                               │
│  Zero-Shot:                                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  REST_7 → Connectivity C_7 → SVD → P_7 (V_7 × k)   │   │
│  │  Align P_7 to shared space (via connectivity):       │   │
│  │    CHA-style: align connectivity fingerprints        │   │
│  │    → R_7 (k × k)                                    │   │
│  │  For new image x:                                    │   │
│  │    z_hat = W @ features(x)           (k,)           │   │
│  │    y_hat = R_7.T @ z_hat then P_7.T  (V_7,)        │   │
│  │    = z_hat @ R_7.T @ P_7.T                          │   │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  Few-Shot:                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Use shared1000 responses from subj7 for alignment   │   │
│  │  R_7 = Procrustes(Z_7_shared, Z_template)           │   │
│  │  Optionally fine-tune W on subj7 data                │   │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Detailed Implementation

#### 6.2.1 Resting-State Preprocessing

**File: `src/alignment/rest_preprocessing.py`**

```python
def compute_rest_connectivity(
    rest_runs: list[np.ndarray],
    mask: np.ndarray,
    mode: str = "parcellation",  # parcellation required for zero-shot CHA
    atlas: np.ndarray | None = None,  # REQUIRED when mode="parcellation"
    ensemble: str = "average",
) -> np.ndarray:
    """
    Compute resting-state connectivity from multiple runs.

    Args:
        rest_runs: List of (T_run, V_sub) arrays, z-scored
        mask: 3D brain mask (for atlas alignment)
        mode: 'voxel_correlation' or 'parcellation'
        atlas: Required for parcellation mode
        ensemble: 'average' (average connectivity) or 'concat' (concat runs)

    Returns:
        C: Connectivity matrix
           - voxel_correlation: (V_sub, V_sub) — WARNING: cannot be used for
             zero-shot CHA because V differs across subjects
           - parcellation: (R, V_sub) where R = number of common parcels
             (after harmonization). R is the SAME across all subjects.

    Raises:
        ValueError: if mode="parcellation" and atlas is None
        ValueError: if mode="voxel_correlation" used in a zero-shot context
    """

def compute_rest_basis(
    connectivity: np.ndarray,
    n_components: int = 50,
    min_k: int = 10,
) -> np.ndarray:
    """
    SVD of connectivity matrix → basis P (V_sub × k).

    k is clamped: k_actual = min(n_components, min(C.shape) - 1).
    For parcellation mode with R regions: rank(C) <= R, so:
      - visual_rois (~7 regions): k <= 6 (USE VOXEL MODE OR BIGGER ATLAS)
      - Kastner2015 (~25 regions): k <= 24
      - Combined multi-atlas (~50+ regions): k <= 49
      - Schaefer400 (~200 within nsdgeneral): k <= 199
    A warning is logged if k_actual < n_components.

    Fail-fast: if k_actual < min_k, raises ValueError with diagnostic message
    indicating which atlas/mode to try instead.

    **Global k enforcement**: In multi-subject pipelines, compute k_actual per subject
    FIRST, then set k_global = min(k_actual across all subjects). Re-run SVD with
    k_global to ensure all P_s have the same number of columns. This is REQUIRED
    for Procrustes alignment (Z matrices must have matching column counts).

    Uses randomized_svd when min(C.shape) > 1000 for efficiency.

    Returns top-k right singular vectors as P (V_sub × k_actual).
    """
```

**Adaptive k rule**: The default `n_components=300` assumes either:
1. `connectivity_mode=voxel_correlation` (rank = V, always sufficient — but **NOT valid for zero-shot CHA**), or
2. A high-resolution atlas (Schaefer 400+) for parcellation mode.

If using `visual_rois` or `Kastner2015` in parcellation mode, either:
- Use a **combined atlas** that merges multiple NSD-provided ROIs into ~50+ regions, or
- Accept lower k (k=20-25 with Kastner) — still useful but lower capacity.

**For zero-shot, parcellation is mandatory** (voxel_correlation produces fingerprints of different sizes across subjects). The config `n_components` is treated as a ceiling; the actual k is determined at runtime.

**Failure policy for insufficient REST data**: If a subject has < 2 REST runs or < 100 usable TRs total after preprocessing:
1. Log error with subject ID and available data statistics
2. For training subjects: fall back to task-based connectivity (acceptable)
3. For subject 7 in zero-shot mode: ABORT — zero-shot is impossible without REST; switch to few-shot
4. Store a flag in the output metadata indicating which connectivity source was used per subject

#### 6.2.2 Shared Space Alignment

**File: `src/alignment/shared_space.py`**

```python
class SharedSpaceBuilder:
    """
    Builds a shared representational space from multiple subjects.

    Handles variable voxel counts by operating in k-dimensional
    component space rather than voxel space.
    """

    def __init__(
        self,
        n_components: int = 300,
        connectivity_mode: str = "parcellation",  # MUST be parcellation for zero-shot
        max_iters: int = 10,
        tol: float = 1e-5,
        atlas_type: str = "kastner",
    ):
        self.n_components = n_components
        self.connectivity_mode = connectivity_mode
        self.max_iters = max_iters
        self.tol = tol

        # Learned state
        self.subject_bases: dict[int, np.ndarray] = {}  # sub_id → P_s (V_s × k)
        self.subject_rotations: dict[int, np.ndarray] = {}  # sub_id → R_s (k × k)
        self.template_Z: np.ndarray | None = None  # (N_shared × k) template

    def fit(
        self,
        subjects: dict[int, NSDSubjectData],
        features: NSDFeatures,
    ) -> "SharedSpaceBuilder":
        """
        Fit shared space using training subjects' data.

        Mode `hybrid_cha` (default):
        1. For each subject: REST → connectivity → SVD → P_s (REST only!)
        2. For each subject: project shared1000 task responses → Z_s = Y_shared @ P_s
           (task responses used ONLY for Procrustes template orientation, NOT for P_s)
           CRITICAL: shared1000 responses MUST be sorted by canonical NSD image ID
           across all subjects, so row i of Z_s corresponds to the same image for all s.
        3. Initialize template Z as average of Z_s (after sign alignment)
        4. Iterate: Procrustes align each Z_s to template, update template
        5. Store P_s and R_s for each subject

        Mode `strict_rest_cha`:
        - Steps 1 same; steps 2-4 use connectivity fingerprints F_s = C_s @ P_s
          instead of task-projected Z_s. No task data touches alignment.

        No subject 7 data is used in fitting.
        """
        ...

    def align_new_subject_zeroshot(
        self,
        rest_runs: list[np.ndarray],
        mask: np.ndarray,
        atlas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a new subject using only REST data (zero-shot).

        Uses CHA-style connectivity fingerprint alignment.

        Returns:
            P_new: (V_new, k) basis
            R_new: (k, k) rotation to shared space
        """
        ...

    def align_new_subject_fewshot(
        self,
        rest_runs: list[np.ndarray],
        mask: np.ndarray,
        task_fmri_shared: np.ndarray,
        shared_stim_idx: np.ndarray,
        atlas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a new subject using REST + a few shared task responses.

        Returns:
            P_new: (V_new, k) basis
            R_new: (k, k) rotation to shared space
        """
        ...

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Orthogonal Procrustes: find R such that source @ R ≈ target.

    Safety checks:
    - Raises ValueError if source or target contain NaN/Inf
    - Raises ValueError if any column has zero variance (near-constant)
    - Centers both matrices before computing M (removes mean)

    Args:
        source: (N, k)
        target: (N, k)

    Returns:
        R: (k, k) orthogonal rotation matrix
    """
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)):
        raise ValueError("NaN/Inf in Procrustes input")
    # Check for zero-variance columns
    if np.any(source.std(axis=0) < 1e-10) or np.any(target.std(axis=0) < 1e-10):
        raise ValueError("Near-constant column in Procrustes input")
    # Center before alignment
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    M = source_c.T @ target_c  # k × k
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt
```

#### 6.2.3 CHA Zero-Shot Alignment (Key Innovation)

**File: `src/alignment/cha_alignment.py`**

For zero-shot alignment of subject 7 (no task data), we need to align using REST-only information:

```python
def align_via_connectivity_fingerprint(
    P_new: np.ndarray,
    C_new: np.ndarray,
    training_fingerprints: list[np.ndarray],
    training_rotations: list[np.ndarray],
    template_S: np.ndarray,
) -> np.ndarray:
    """
    Zero-shot alignment using connectivity fingerprints.

    The idea: if subjects share the same connectivity structure (measured
    in a common parcellation space), we can align their component spaces
    without any shared task data.

    For parcellation mode:
      C_s is (R × V_s) — parcel-to-voxel connectivity
      P_s is (V_s × k) — basis
      Fingerprint F_s = C_s @ P_s → (R × k) — parcel-space representation

      All fingerprints are (R × k) regardless of V_s!
      So we can Procrustes-align F_new to the average training fingerprint.

    Args:
        P_new: (V_new, k) basis for new subject
        C_new: (R, V_new) parcellation connectivity for new subject
        training_fingerprints: list of (R, k) fingerprints from training subjects
        training_rotations: list of (k, k) rotations from training subjects
        template_S: (R, k) or (k, k) shared template

    Returns:
        R_new: (k, k) rotation to align new subject to shared space
    """
    # Compute new subject's fingerprint
    F_new = C_new @ P_new  # (R, k)

    # Average training fingerprints (already aligned to shared space)
    F_template = np.mean([
        F_s @ R_s for F_s, R_s in zip(training_fingerprints, training_rotations)
    ], axis=0)  # (R, k)

    # Procrustes alignment in fingerprint space
    R_new = procrustes_align(F_new, F_template)  # (k, k)

    return R_new
```

#### 6.2.4 Encoding Model

**File: `src/models/encoding.py`**

```python
class SharedSpaceEncoder:
    """
    Maps stimulus features to shared-space responses via ridge regression.
    """

    def __init__(self, alpha: float = 100.0):
        self.alpha = alpha
        self.W: np.ndarray | None = None  # (F, k)
        self.b: np.ndarray | None = None  # (k,)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
    ):
        """
        Fit ridge: X (N × F) → Z (N × k) with explicit intercept.

        Procedure:
        1. Standardize X: Xs = (X - mean) / std
        2. Center Z: Zc = Z - Z_mean
        3. Ridge on centered data: W = (Xs'Xs + αI)⁻¹ Xs'Zc
        4. Intercept: b = Z_mean (since Xs is zero-mean, no correction needed)
        """
        # Standardize features
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1e-8
        Xs = (X - self.x_mean) / self.x_std

        # Center targets
        self.z_mean = Z.mean(axis=0)
        Zc = Z - self.z_mean

        # Ridge: W = (Xs'Xs + αI)⁻¹ Xs'Zc
        N, F = Xs.shape
        self.W = np.linalg.solve(
            Xs.T @ Xs + self.alpha * np.eye(F),
            Xs.T @ Zc
        )
        # Intercept = target mean (since Xs is zero-mean after standardization)
        self.b = self.z_mean

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict shared-space responses: (N, F) → (N, k)."""
        Xs = (X_new - self.x_mean) / self.x_std
        return Xs @ self.W + self.b

    def predict_voxels(
        self,
        X_new: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """
        Predict voxel-wise responses for a specific subject.

        Args:
            X_new: (N, F) stimulus features
            P: (V_sub, k) subject's REST basis
            R: (k, k) subject's rotation to shared space

        Returns:
            Y_hat: (N, V_sub) predicted voxel responses
        """
        Z_hat = self.predict(X_new)  # (N, k)
        return Z_hat @ R.T @ P.T    # (N, V_sub)
```

#### 6.2.5 Atlas for Parcellation Mode

To use parcellation-based connectivity, we need a common atlas in each subject's native space. Options:

1. **NSD-provided ROIs**: `nsdgeneral.nii.gz` is already a mask but not a parcellation
2. **Kastner atlas / visual ROIs**: NSD provides visual area labels (V1, V2, V3, etc.) per subject
3. **Schaefer parcellation**: Available in fsaverage space, can be projected to volumetric native space
4. **Simple approach**: Use the nsdgeneral voxels and create a parcellation by spatial k-means clustering

**Recommended**: Use NSD's visual ROI parcellations (prf-visualrois, Kastner2015, HCP_MMP1) which are available per subject in native func1pt8mm space.

**File: `src/data/load_atlas.py`**

```python
def load_visual_rois(sub: int, data_root: str = "nsddata") -> np.ndarray:
    """
    Load visual area parcellation for subject.
    Available ROIs: V1v, V1d, V2v, V2d, V3v, V3d, hV4, VO1, VO2,
                    PHC1, PHC2, LO1, LO2, TO1, TO2, V3a, V3b, IPS0-5

    Returns: 3D integer label map (0 = background)
    """

def harmonize_atlas_labels(
    atlas_maps: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], list[int]]:
    """
    Harmonize atlas labels across subjects.

    Problem: Different subjects may have different sets of ROI labels
    present within nsdgeneral (e.g., subject A has labels {1,2,3,5},
    subject B has {1,2,4,5} — label 3 is missing in B, label 4 in A).

    Solution: INTERSECTION policy — keep only labels present in ALL subjects.
    Remap surviving labels to contiguous integers [1, 2, ..., R_common].

    Args:
        atlas_maps: sub_id → 3D integer atlas in native space
        masks: sub_id → 3D nsdgeneral mask

    Returns:
        harmonized_maps: sub_id → 3D atlas with remapped labels (0=background)
        common_labels: sorted list of original label IDs present in all subjects

    Raises:
        ValueError if R_common < min_k + 1 (insufficient parcels for desired k).
        E.g., with min_k=10, we need at least 11 common parcels (rank = R, k <= R-1).
    """
```

#### 6.2.5b Parcel QC (Quality Check)

After atlas harmonization, verify atlas quality per subject:

```python
def parcel_qc(
    atlas_masked: np.ndarray,
    common_labels: list[int],
    sub_id: int,
    min_voxels_per_parcel: int = 10,
) -> dict:
    """
    QC check for parcellation within nsdgeneral mask.

    Checks:
    1. No empty parcels (every label in common_labels has >= min_voxels_per_parcel voxels)
    2. Reports parcel sizes (min, max, mean voxels per parcel)
    3. Warns if any parcel has < min_voxels_per_parcel voxels

    Returns:
        dict with 'n_parcels', 'min_voxels', 'max_voxels', 'mean_voxels',
        'empty_parcels' (list), 'warnings' (list[str])

    If empty parcels are found:
    - Log warning with subject ID and parcel IDs
    - Remove empty parcels from common_labels and re-harmonize
    """
```

### 6.3 Complete Training Pipeline

**File: `src/pipelines/train_shared_space.py`**

```python
def train_pipeline(
    train_subs: list[int] = [1, 2, 5],
    n_components: int = 300,
    ridge_alpha: float = 100.0,
    feature_type: str = "clip",
    connectivity_mode: str = "parcellation",
    output_dir: str = "outputs/shared_space",
):
    """
    Complete training pipeline.

    Steps:
    1. Load all training subjects' data
    2. Load/extract stimulus features
    3. Build shared space from REST + shared stimuli
    4. Train global encoder
    5. Save model artifacts
    """

    # Step 1: Load data
    subjects = {s: NSDSubjectData(s) for s in train_subs}
    features = NSDFeatures()

    # Step 2: Build shared space
    builder = SharedSpaceBuilder(
        n_components=n_components,
        connectivity_mode=connectivity_mode,
    )
    builder.fit(subjects, features)

    # Step 3: Prepare training data in shared space
    X_all, Z_all = [], []
    for sub_id, subj in subjects.items():
        X = features.get_features(subj.train_stim_idx, feature_type)
        Z = subj.train_fmri @ builder.subject_bases[sub_id]  # (N, k)
        Z = Z @ builder.subject_rotations[sub_id]  # align to shared space
        X_all.append(X)
        Z_all.append(Z)

    X_concat = np.concatenate(X_all, axis=0)
    Z_concat = np.concatenate(Z_all, axis=0)

    # Step 4: Train encoder
    encoder = SharedSpaceEncoder(alpha=ridge_alpha)
    encoder.fit(X_concat, Z_concat)

    # Step 5: Save
    save_model(builder, encoder, output_dir)
```

### 6.4 Complete Prediction Pipeline

**File: `src/pipelines/predict_subject.py`**

```python
def predict_zero_shot(
    test_sub: int = 7,
    model_dir: str = "outputs/shared_space",
    output_dir: str = "outputs/predictions",
    feature_type: str = "clip",
):
    """
    Zero-shot prediction for test subject using only REST data.
    """
    # Load model
    builder, encoder = load_model(model_dir)

    # Load test subject data
    test_subj = NSDSubjectData(test_sub)
    features = NSDFeatures()

    # Align test subject (zero-shot via REST only)
    P_new, R_new = builder.align_new_subject_zeroshot(
        rest_runs=test_subj.rest_runs,
        mask=test_subj.mask,
    )

    # Predict
    X_test = features.get_features(test_subj.test_stim_idx, feature_type)
    Y_pred = encoder.predict_voxels(X_test, P_new, R_new)

    # Evaluate
    Y_true = test_subj.test_fmri
    metrics = evaluate(Y_true, Y_pred)

    return Y_pred, metrics


def predict_few_shot(
    test_sub: int = 7,
    n_shots: int = 100,
    model_dir: str = "outputs/shared_space",
    output_dir: str = "outputs/predictions",
    feature_type: str = "clip",
    fine_tune: bool = False,
    seed: int = 42,
):
    """
    Few-shot prediction using N shared-stimuli responses from test subject.

    Args:
        seed: Random seed for shot/eval split. In ablation loop,
              caller passes seed=base_seed + repeat_idx.
    """
    # Load model
    builder, encoder = load_model(model_dir)

    # Load test subject data
    test_subj = NSDSubjectData(test_sub)
    features = NSDFeatures()

    # Random split: sample n_shots for alignment, rest for evaluation
    rng = np.random.RandomState(seed)
    n_shared = len(test_subj.test_stim_idx)
    shot_indices = rng.choice(n_shared, size=n_shots, replace=False)
    eval_indices = np.setdiff1d(np.arange(n_shared), shot_indices)

    shared_fmri = test_subj.test_fmri[shot_indices]
    shared_idx = test_subj.test_stim_idx[shot_indices]

    # Align test subject (few-shot via shared responses)
    P_new, R_new = builder.align_new_subject_fewshot(
        rest_runs=test_subj.rest_runs,
        mask=test_subj.mask,
        task_fmri_shared=shared_fmri,
        shared_stim_idx=shared_idx,
    )

    # Optional: fine-tune encoder on test subject data
    if fine_tune:
        X_shared = features.get_features(shared_idx, feature_type)
        Z_shared = shared_fmri @ P_new @ R_new
        encoder = fine_tune_encoder(encoder, X_shared, Z_shared)

    # Predict held-out test stimuli (complement of shots)
    eval_stim_idx = test_subj.test_stim_idx[eval_indices]
    X_test = features.get_features(eval_stim_idx, feature_type)
    Y_pred = encoder.predict_voxels(X_test, P_new, R_new)

    # Evaluate on held-out only
    Y_true = test_subj.test_fmri[eval_indices]
    metrics = evaluate(Y_true, Y_pred)

    return Y_pred, metrics
```

---

## 7. Approach C — Deep Learning Encoder (Optional Extension)

### 7.1 Overview

Replace ridge regression with a neural network for the encoding model:

```
Features (768-D CLIP) → MLP → Shared Space (k-D) → Voxels via P_s R_s
```

**File: `src/models/neural_encoder.py`**

```python
class NeuralEncoder(nn.Module):
    """
    MLP encoder: features → shared space responses.

    Architecture:
        Input (768) → Linear(768, 2048) → LayerNorm → GELU → Dropout(0.3)
                    → Linear(2048, 2048) → LayerNorm → GELU → Dropout(0.3)
                    → Linear(2048, k)
    """

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 2048,
                 n_components: int = 300, dropout: float = 0.3):
        ...
```

### 7.2 Training

```python
# Use same shared space alignment as Approach B
# Train with MSE loss on Z_shared targets
# Optionally add contrastive loss for better generalization
```

### 7.3 Priority

This is a **stretch goal**. Start with ridge regression (Approach B), which is well-established for NSD encoding models and doesn't require GPU training.

---

## 8. Zero-Shot vs Few-Shot Strategies

### 8.1 Zero-Shot (REST only)

**Alignment method**: CHA-style connectivity fingerprint alignment (Section 6.2.3)

**Requirements**:
- Subject 7 resting-state data (multiple runs preferred)
- A parcellation atlas in subject 7's native space

**Expected performance**: Moderate. CHA alignment captures coarse functional organization but may miss fine-grained individual patterns.

**Improvements**:
- Use multiple REST runs with ensemble averaging
- Higher n_components (300-500)
- Parcellation with more regions (Schaefer 400+)

### 8.2 Few-Shot (REST + N shared stimuli)

**Alignment method**: Direct Procrustes on shared-stimulus responses

**Requirements**:
- Subject 7 resting-state data
- N responses to shared stimuli from subject 7

**Expected performance**: Better than zero-shot, especially with N ≥ 50-100.

**Variations**:
- **Few-shot alignment only**: Use shared responses for alignment, keep global encoder
- **Few-shot fine-tuning**: Additionally fine-tune the encoder weights on subject 7's data
- **Hybrid**: CHA alignment + Procrustes correction using shared responses

### 8.3 Ablation: Number of Shots

The shared1000 set has ~1000 images. For few-shot ablation, N images are used for alignment and the remaining (1000 - N) are used for evaluation. This means N=1000 is invalid (no held-out eval data). Max practical N is ~750, leaving 250 for evaluation.

```python
def run_fewshot_ablation(
    shots_list: list[int] = [0, 10, 25, 50, 100, 250, 500, 750],
    n_repeats: int = 5,
    seed: int = 42,
    ...
):
    """
    Run prediction with varying numbers of shared-stimulus examples.

    For each N in shots_list:
    - Repeat n_repeats times with different random splits (seeded)
    - Each split: randomly sample N images for alignment, evaluate on remaining
    - Report mean ± std across repeats

    N=0 corresponds to zero-shot (uses CHA alignment, no random split needed).
    """
```

**Protocol details:**
- Shots are sampled randomly from the shared1000 set using `np.random.RandomState(seed + repeat_idx)`
- The held-out set for evaluation is always the complement (1000 - N images)
- This avoids protocol leakage from deterministic `[:n_shots]` ordering

---

## 9. Evaluation Framework

### 9.1 Metrics

**File: `src/evaluation/metrics.py`**

```python
def voxelwise_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Per-voxel Pearson r across stimuli. Returns (V,) array.

    Safety: Voxels with zero variance (constant across stimuli) return r=0.0
    (not NaN). Log count of zero-variance voxels if > 0.
    """

def median_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Median of voxelwise correlations."""

def noise_ceiling(
    trial_fmri: np.ndarray,
    trial_labels: np.ndarray,
    method: str = "split_half",
) -> np.ndarray:
    """
    Upper bound from repeated presentations.

    IMPORTANT: Requires TRIAL-LEVEL (not averaged) betas.
    Use the *_trials.npy files from preprocessing, NOT the averaged *_fmri.npy.

    Methods:
    - 'split_half': Compute correlation between odd/even trial averages.
      Spearman-Brown corrected. Returns voxelwise NC in correlation units.
    - 'ncsnr_mask': Use NSD-provided ncsnr maps (nsddata/ppdata/subj{XX}/
      func1pt8mm/ncsnr.nii.gz) ONLY as a reliability mask (threshold on ncsnr
      value), NOT as a direct correlation ceiling. ncsnr is signal-to-noise
      ratio, not a correlation metric. To convert: NC_corr ≈ ncsnr² / (ncsnr² + 1/reps).

    Default 'split_half' avoids ncsnr conversion ambiguity.
    """

def normalized_performance(corrs: np.ndarray, nc: np.ndarray, nc_floor: float = 0.1) -> np.ndarray:
    """
    Fraction of noise ceiling achieved: r / NC.

    Clipping policy: Voxels with NC < nc_floor are excluded from the ratio
    (set to NaN) to avoid division by near-zero values inflating the metric.
    Report the fraction of voxels excluded.
    """

def pattern_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Per-stimulus Pearson r across voxels. Returns (N,) array."""

def two_vs_two_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray, n_pairs: int = 1000) -> float:
    """
    2-vs-2 identification accuracy:
    Given 2 stimuli and 2 predictions, can we match them correctly?
    Chance = 50%.
    """

def roi_evaluation(
    Y_true: np.ndarray, Y_pred: np.ndarray,
    atlas: np.ndarray, mask: np.ndarray,
    roi_names: dict[int, str],
) -> dict[str, dict[str, float]]:
    """Per-ROI evaluation: V1, V2, V3, hV4, etc."""
```

### 9.1b Voxel Reliability Masking

**Problem**: Many nsdgeneral voxels may have very low signal-to-noise, dominating metrics with noise.

**Solution**: Apply voxel reliability masking before computing metrics:

```python
def get_reliable_voxels(
    noise_ceiling: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Return boolean mask of voxels exceeding noise ceiling threshold.

    Default: NC > 0.1 (voxels where signal is detectable).
    Report metrics on: all voxels, reliable voxels (NC > 0.1), and high-SNR voxels (NC > 0.3).
    """
```

All evaluation metrics should be reported in three tiers:
1. **All voxels**: Full nsdgeneral mask
2. **Reliable voxels**: NC > 0.1 (excludes noise-dominated voxels)
3. **High-SNR voxels**: NC > 0.3 (best-quality voxels)

### 9.2 Baselines for Comparison

1. **Chance**: Predict mean training activation for all stimuli → r ≈ 0
2. **Subject mean**: Predict average response per voxel → r ≈ 0
3. **Same-subject ridge**: Train and test on same subject (upper bound for individual model)
4. **Cross-subject average**: Average training subjects' responses (no alignment)
5. **MNI baseline**: Approach A (Section 5)
6. **CHA zero-shot**: Approach B without task data from test subject
7. **Few-shot (varying N)**: Approach B with N shared-stimulus examples
8. **Oracle**: Noise ceiling (theoretical maximum)

### 9.3 Visualization

**File: `src/evaluation/visualize.py`**

```python
def plot_correlation_histogram(corrs: np.ndarray, nc: np.ndarray = None, title: str = ""):
    """Histogram of voxelwise correlations with optional noise ceiling."""

def plot_correlation_brain_map(corrs: np.ndarray, mask: np.ndarray, shape: tuple):
    """3D brain map of prediction quality."""

def plot_fewshot_curve(results: dict[int, float]):
    """Performance vs number of shots."""

def plot_roi_comparison(roi_results: dict, conditions: list[str]):
    """Bar chart comparing ROI performance across conditions."""

def plot_predicted_vs_actual_patterns(Y_true, Y_pred, stim_indices, stimuli):
    """Side-by-side: image, predicted pattern, actual pattern."""
```

---

## 10. Repository Structure

```
resting_prediction/
├── PLAN.md                          # This file
├── README.md                        # Project documentation
├── requirements.txt                 # Dependencies
├── config.yaml                      # Default hyperparameters
│
├── download_nsddata.py              # EXISTING — updated for REST data
├── prepare_nsddata.py               # EXISTING — kept for reference
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_task_data.py     # Refactored prepare_nsddata.py
│   │   ├── prepare_rest_data.py     # REST data preprocessing
│   │   ├── prepare_features.py      # Stimulus feature extraction
│   │   ├── nsd_loader.py            # NSDSubjectData + NSDFeatures classes
│   │   └── load_atlas.py            # Atlas/parcellation loading
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── rest_preprocessing.py    # REST connectivity computation
│   │   ├── shared_space.py          # SharedSpaceBuilder class
│   │   ├── cha_alignment.py         # CHA zero-shot alignment
│   │   └── utils.py                 # Procrustes, SVD helpers
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoding.py              # SharedSpaceEncoder (ridge)
│   │   ├── baseline_mni.py          # MNI baseline model
│   │   └── neural_encoder.py        # Optional MLP encoder
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # All evaluation metrics
│   │   └── visualize.py             # Plotting functions
│   │
│   └── pipelines/
│       ├── __init__.py
│       ├── train_shared_space.py    # Full training pipeline
│       ├── predict_subject.py       # Zero-shot + few-shot prediction
│       └── run_ablations.py         # Ablation experiments
│
├── scripts/
│   ├── 01_download_data.sh          # Data download wrapper
│   ├── 02_prepare_data.sh           # Data preparation wrapper
│   ├── 03_extract_features.sh       # Feature extraction wrapper
│   ├── 04_train.sh                  # Training wrapper
│   ├── 05_predict.sh                # Prediction wrapper
│   └── 06_evaluate.sh               # Evaluation wrapper
│
├── tests/
│   ├── test_shapes.py               # Unit tests: matrix shape contracts
│   ├── test_pipeline.py             # Integration test: synthetic end-to-end
│   └── conftest.py                  # Pytest fixtures (synthetic data generators)
│
├── notebooks/
│   ├── 01_explore_data.ipynb        # Data exploration
│   ├── 02_rest_connectivity.ipynb   # REST analysis
│   ├── 03_alignment_demo.ipynb      # Alignment visualization
│   └── 04_results.ipynb             # Results visualization
│
├── processed_data/                  # Generated data (gitignored)
│   ├── subj01/
│   │   ├── train_fmri.npy
│   │   ├── test_fmri.npy
│   │   ├── train_stim_idx.npy
│   │   ├── test_stim_idx.npy
│   │   ├── rest_run1.npy ... rest_runN.npy
│   │   └── mask.npy
│   ├── subj02/ ...
│   ├── subj05/ ...
│   ├── subj07/ ...
│   └── features/
│       ├── clip_features.npy
│       └── dinov2_features.npy
│
└── outputs/                         # Model outputs (gitignored)
    ├── shared_space/
    │   ├── builder.npz
    │   └── encoder.npz
    ├── predictions/
    │   ├── zeroshot_sub7.npy
    │   └── fewshot_sub7_N100.npy
    └── figures/
```

---

## 11. Implementation Steps (Ordered)

### Phase 1: Data Infrastructure (Days 1-2)

- [ ] **Task 1.1**: Update `download_nsddata.py` to include resting-state timeseries and visual ROIs
- [ ] **Task 1.2**: Create `src/data/prepare_task_data.py` — refactored prepare_nsddata with proper abstraction
- [ ] **Task 1.3**: Create `src/data/prepare_rest_data.py` — REST preprocessing pipeline
- [ ] **Task 1.4**: Create `src/data/nsd_loader.py` — unified data loading classes
- [ ] **Task 1.5**: Create `src/data/load_atlas.py` — visual ROI atlas loading
- [ ] **Task 1.6**: Run data preparation for all 4 subjects, verify voxel counts

### Phase 2: Feature Extraction (Day 2)

- [ ] **Task 2.1**: Create `src/data/prepare_features.py` — CLIP and DINOv2 extraction
- [ ] **Task 2.2**: Extract features for all 73k stimuli
- [ ] **Task 2.3**: Verify feature quality with sanity checks (nearest neighbor retrieval)

### Phase 3: Alignment Infrastructure (Days 3-4)

- [ ] **Task 3.1**: Create `src/alignment/utils.py` — Procrustes, SVD, z-scoring helpers
- [ ] **Task 3.2**: Create `src/alignment/rest_preprocessing.py` — REST connectivity computation
- [ ] **Task 3.3**: Create `src/alignment/shared_space.py` — SharedSpaceBuilder with response-based alignment
- [ ] **Task 3.4**: Create `src/alignment/cha_alignment.py` — CHA zero-shot alignment via connectivity fingerprints
- [ ] **Task 3.5**: Test alignment on training subjects with leave-one-out validation

### Phase 4: Encoding Models (Day 4)

- [ ] **Task 4.1**: Create `src/models/encoding.py` — SharedSpaceEncoder (ridge regression)
- [ ] **Task 4.2**: Create `src/models/baseline_mni.py` — MNI baseline
- [ ] **Task 4.3**: Test encoding on training subjects (within-subject, cross-subject)

### Phase 5: Evaluation Framework (Day 5)

- [ ] **Task 5.1**: Create `src/evaluation/metrics.py` — all metrics
- [ ] **Task 5.2**: Create `src/evaluation/visualize.py` — plotting functions
- [ ] **Task 5.3**: Compute noise ceilings for all subjects

### Phase 6: Full Pipeline (Days 5-6)

- [ ] **Task 6.1**: Create `src/pipelines/train_shared_space.py` — end-to-end training
- [ ] **Task 6.2**: Create `src/pipelines/predict_subject.py` — zero-shot and few-shot prediction
- [ ] **Task 6.3**: Run full pipeline: train on {1,2,5}, predict subject 7

### Phase 7: Experiments & Ablations (Days 6-7)

- [ ] **Task 7.1**: Run MNI baseline
- [ ] **Task 7.2**: Run zero-shot CHA prediction
- [ ] **Task 7.3**: Run few-shot prediction (N = 10, 25, 50, 100, 250, 500, 750) — 5 random splits each
- [ ] **Task 7.4**: Run leave-one-out cross-validation on training subjects
- [ ] **Task 7.5**: ROI-specific analysis (V1, V2, V3, hV4, etc.)
- [ ] **Task 7.6**: Ablation: n_components (50, 100, 200, 300, 500)
- [ ] **Task 7.7**: Ablation: feature type (CLIP, DINOv2, concat)
- [ ] **Task 7.8**: Ablation: ridge alpha grid search

### Phase 8: Documentation & Polish (Day 7)

- [ ] **Task 8.1**: Create results notebook with all figures
- [ ] **Task 8.2**: Write README.md
- [ ] **Task 8.3**: Clean up code, add docstrings

---

## 11b. Compute Policy

### Memory Management

- **float32 everywhere**: All arrays use `np.float32` (not float64) to halve memory. Set at data loading time.
- **V × V connectivity matrix**: For `voxel_correlation` mode with V ≈ 15,000, the matrix is ~900 MB (float32). Use `np.memmap` for subjects with V > 10,000 if available RAM < 16 GB.
- **Randomized SVD**: For large matrices (V × V), use `sklearn.utils.extmath.randomized_svd` instead of full `np.linalg.svd`. This computes only the top-k singular vectors in O(V × k²) instead of O(V³). Use randomized SVD when `min(C.shape) > 1000`.
- **Batch feature extraction**: Extract CLIP/DINOv2 features in batches of 64 images on GPU to avoid OOM.
- **Memmap for betas**: If loading all 37 sessions at once exceeds RAM, use `np.memmap` or process sessions sequentially and accumulate voxel sums for averaging.

### Computation Time Estimates

| Operation | Estimated Time | Notes |
|-----------|---------------|-------|
| CLIP feature extraction (73k images) | ~30 min on GPU | Batched, ViT-L/14 |
| REST connectivity (parcellation) | ~1 min/subject | R × V, small |
| REST connectivity (voxel_correlation) | ~5 min/subject | V × V, use randomized SVD |
| Shared space alignment (3 subjects) | ~2 min | Iterative Procrustes in k-space |
| Ridge regression (30k samples × k) | ~1 min | Small problem in shared space |
| Full pipeline (train + predict) | ~45 min total | Excluding data download |

## 11c. Test Strategy

### Unit Tests (`tests/test_shapes.py`)

Shape-contract tests that verify matrix dimensions through the pipeline without requiring real data:

```python
def test_rest_basis_shape():
    """P_s has shape (V_sub, k_actual) where k_actual <= n_components."""
    V, k = 1000, 50
    C = np.random.randn(25, V).astype(np.float32)  # parcellation: (R, V)
    P = compute_rest_basis(C, n_components=k)
    assert P.shape == (V, min(k, 24))  # R=25 → rank limit = 24

def test_shared_space_alignment_shapes():
    """All Z_s are (N_shared, k) regardless of V_s."""
    ...

def test_fingerprint_common_R():
    """All fingerprints F_s = C_s @ P_s have shape (R, k) with same R."""
    ...

def test_encoding_roundtrip():
    """predict_voxels output has shape (N, V_sub)."""
    ...

def test_procrustes_orthogonal():
    """R @ R.T ≈ I (rotation matrix is orthogonal)."""
    ...
```

### Integration Tests (`tests/test_pipeline.py`)

End-to-end test with synthetic data (small V, small N):

```python
def test_full_pipeline_synthetic():
    """
    Create 3 synthetic 'subjects' with known shared structure,
    run full train + zero-shot predict, verify predictions correlate
    with ground truth above chance.
    """
    ...
```

### Smoke Test (`scripts/smoke_test.sh`)

Quick test with real data (1 subject, 1 session, first 100 stimuli):
```bash
python -m pytest tests/ -v --tb=short
python scripts/run_smoke_test.py  # subset of real data, < 5 min
```

---

## 12. Configuration & Hyperparameters

**File: `config.yaml`**

```yaml
# Data
subjects:
  train: [1, 2, 5]
  test: [7]
data_root: "."
output_root: "outputs"

# Feature extraction
features:
  type: "clip"  # clip, dinov2, clip_dinov2, vgg16
  clip_model: "ViT-L/14"
  dinov2_model: "dinov2_vitl14"
  batch_size: 64
  device: "cuda"

# Reproducibility
random_seed: 42
fewshot_n_repeats: 5

# REST preprocessing
rest_preprocessing:
  discard_initial_trs: 5       # T1 equilibration
  detrend: true                # linear detrend per run
  highpass_cutoff_hz: 0.01     # Butterworth high-pass; set to null to skip
  nuisance_regression: false   # true if motion param files available
  zscore: true                 # z-score per run per voxel

# Alignment
alignment:
  n_components: 50           # k ceiling — actual k = min(this, rank(C)-1)
                             # Default 50 is conservative; with combined_rois (~50+ regions) k_actual ≈ 49
                             # With kastner only (~25 regions) k_actual ≈ 24
                             # Increase to 200+ only with Schaefer400 or voxel_correlation mode
  min_k: 10                  # Fail-fast: abort zero-shot if effective k < min_k after atlas harmonization
  connectivity_mode: "parcellation"  # parcellation (default, REQUIRED for zero-shot) or voxel_correlation (few-shot only)
  atlas_type: "combined_rois"  # combined_rois (~50+ regions, DEFAULT) or kastner (~25) or schaefer400
  atlas_harmonize: "intersection"  # keep only labels present in ALL subjects
  ensemble_method: "average"  # average or concat REST runs
  max_iters: 10
  tol: 1.0e-5
  # NOTE: For parcellation mode, k is clamped to min(n_components, n_regions-1).
  # With kastner (~25 regions), k_actual = 24. For higher k, use combined_rois or voxel_correlation.
  # IMPORTANT: voxel_correlation mode CANNOT be used for zero-shot CHA alignment
  # because V differs across subjects and fingerprints F_s = C_s @ P_s would have
  # shape (V_s, k) — not comparable across subjects. Parcellation gives (R, k) with common R.

# Compute policy
compute:
  dtype: "float32"             # np.float32 everywhere to save memory
  use_randomized_svd: true     # Use sklearn randomized_svd when min(C.shape) > 1000
  randomized_svd_oversampling: 10  # Extra components for numerical stability
  memmap_threshold_gb: 2.0     # Use np.memmap for arrays exceeding this size

# Encoding
encoding:
  model_type: "ridge"         # ridge or neural
  ridge_alpha: 100.0
  # Neural encoder (if model_type == "neural")
  hidden_dim: 2048
  dropout: 0.3
  learning_rate: 1.0e-4
  epochs: 100
  batch_size: 256

# Evaluation
evaluation:
  metrics: ["voxelwise_correlation", "pattern_correlation", "two_vs_two"]
  noise_ceiling_method: "ncsnr"
  fewshot_shots: [0, 10, 25, 50, 100, 250, 500, 750]  # max 750 to leave 250 for eval
```

---

## 13. Risks & Mitigations

### Risk 1: Resting-state data not available or insufficient
**Mitigation**: For **training subjects**: use task-based functional connectivity as a proxy (compute parcel-voxel correlations from task betas). This is acceptable since training subjects' task data is already used. For **subject 7 in zero-shot**: NO fallback — if REST is unavailable, zero-shot is impossible; switch to few-shot. Report which connectivity source was used per subject. Mixed-source experiments (REST for some, task for others) are reported as a separate "mixed-CHA" condition, NOT as zero-shot.

### Risk 2: CHA zero-shot alignment is too poor
**Mitigation**: Fall back to few-shot alignment with shared stimuli. Even N=50 examples should help significantly.

### Risk 3: Variable voxel counts break existing code
**Mitigation**: The entire pipeline operates in k-dimensional shared space. Voxel-space operations are always subject-specific and never cross-subject. Carefully verify all matrix shapes.

### Risk 4: Memory issues with large connectivity matrices
**Mitigation**: Use parcellation mode (R × V instead of V × V). For R=25 (Kastner) and V=15,000, memory is ~1.5 MB vs ~900 MB. If `voxel_correlation` mode is needed (few-shot only), use `np.float32` (~450 MB), `np.memmap` for on-disk storage, and `randomized_svd` (O(V·k²) instead of O(V³)).

### Risk 5: Ridge regression underfits / overfits
**Mitigation**: Cross-validate alpha across a wide range (0.1 to 10,000). Also try neural encoder as alternative.

### Risk 6: Training subjects have too few overlapping stimuli
**Mitigation**: Each NSD subject sees ~10,000 unique images, but the shared1000 set (1,000 images) is seen by all. This is sufficient for alignment. For the encoding model, we pool all subject-specific training images (~30,000 total across 3 subjects).

### Risk 7: Parcellation atlas not available in native space
**Mitigation**: NSD provides visual area ROIs in native func1pt8mm space. Alternatively, use volumetric Schaefer atlas from FreeSurfer outputs (NSD provides FreeSurfer surfaces per subject).

### Risk 8: Metrics dominated by noisy voxels
**Mitigation**: Apply voxel reliability masking using noise ceiling thresholds (NC > 0.1, NC > 0.3). Report metrics at all three tiers. Alternatively, use NSD-provided ncsnr maps.

### Risk 9: MNI baseline requires spatial transforms
**Mitigation**: The MNI baseline is a secondary priority. If native-to-MNI transforms are not readily available, skip Approach A and focus on Approach B (shared space), which operates entirely in native space and does not require spatial registration. If MNI is needed later, use `nilearn.image.resample_to_img` with NSD-provided T1w-to-MNI transforms or FreeSurfer's `mri_vol2vol`.

### Risk 10: REST timeseries download fails or pulls wrong files
**Mitigation**: Use metadata-driven `discover_rest_runs()` (see Section 2.2) — never rely on filename wildcards. Validate discovered files by checking NIfTI headers (correct dimensions, reasonable TR). If discovery fails for any subject, apply the fallback rules from Section 0 (training subjects: task-connectivity acceptable as "mixed-CHA"; subject 7 zero-shot: ABORT). Log file sizes during download and abort if total exceeds 50 GB.

### Risk 11: Atlas labels differ across subjects (empty parcels after harmonization)
**Mitigation**: Use intersection policy — keep only labels present in ALL subjects. Run parcel QC to detect and remove parcels with < 10 voxels. If fewer than 3 parcels remain after harmonization, fall back to a different atlas (Kastner → combined_rois → spatial k-means on nsdgeneral voxels).

### Risk 12: Zero-shot CHA with parcellation yields low k (e.g., k=24 with Kastner)
**Mitigation**: Low k limits representational capacity. Options in priority order:
1. Combine multiple NSD atlases (prf-visualrois + Kastner + floc-*) → ~50+ regions → k up to ~49
2. Use Schaefer400 parcellation (if available in native space) → k up to ~199
3. Accept low k and compensate with more sophisticated encoding (neural encoder)
4. Hybrid: use parcellation for zero-shot alignment, then refine with few-shot Procrustes if k is too low

---

## Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
nibabel>=5.0
h5py>=3.8
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.2
pyyaml>=6.0
tqdm>=4.65

# Feature extraction (GPU recommended)
torch>=2.0
torchvision>=0.15
open-clip-torch>=2.20    # For CLIP features
transformers>=4.30       # For DINOv2

# Optional
nilearn>=0.10           # For brain visualization
jupyter>=1.0
```
