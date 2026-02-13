"""
Atlas/parcellation loading and cross-subject harmonization.

Supports:
- kastner: Kastner2015 atlas (~25 visual regions)
- combined_rois: Merge multiple NSD-provided ROIs (~50+ regions)
- Individual ROIs: prf-visualrois, floc-*, etc.
"""

import logging
import os
from collections import Counter

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# NSD-provided ROI files available per subject in func1pt8mm/roi/
ROI_FILES = {
    "kastner": "Kastner2015.nii.gz",
    "prf_visualrois": "prf-visualrois.nii.gz",
    "prf_eccrois": "prf-eccrois.nii.gz",
    "floc_bodies": "floc-bodies.nii.gz",
    "floc_faces": "floc-faces.nii.gz",
    "floc_places": "floc-places.nii.gz",
    "floc_words": "floc-words.nii.gz",
}


def load_atlas(
    sub: int,
    atlas_type: str = "kastner",
    data_root: str = ".",
) -> np.ndarray:
    """
    Load atlas parcellation for a subject in native func1pt8mm space.

    Args:
        sub: subject number
        atlas_type: 'kastner', 'prf_visualrois', 'combined_rois', or specific ROI key
        data_root: root data directory

    Returns:
        3D integer label map (0 = background)
    """
    roi_dir = os.path.join(data_root, f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/")

    if atlas_type == "combined_rois":
        return _load_combined_rois(roi_dir)

    roi_file = ROI_FILES.get(atlas_type)
    if roi_file is None:
        raise ValueError(f"Unknown atlas_type: {atlas_type}. Available: {list(ROI_FILES.keys()) + ['combined_rois']}")

    path = os.path.join(roi_dir, roi_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Atlas file not found: {path}")

    atlas = nib.load(path).get_fdata().astype(np.int32)
    n_labels = len(np.unique(atlas[atlas > 0]))
    logger.info(f"Subject {sub}: loaded {atlas_type} with {n_labels} labels")
    return atlas


def _load_combined_rois(roi_dir: str) -> np.ndarray:
    """
    Combine multiple ROI files into a single atlas with unique labels.

    Each ROI file's labels are offset so they don't overlap:
    - Kastner labels: 1-25 → 1-25
    - prf-visualrois labels: 1-7 → 101-107
    - floc-bodies labels: 1-N → 201-20N
    - etc.
    """
    offsets = {
        "kastner": 0,
        "prf_visualrois": 100,
        "floc_bodies": 200,
        "floc_faces": 300,
        "floc_places": 400,
        "floc_words": 500,
    }

    combined = None
    total_labels = 0

    for roi_key, offset in offsets.items():
        roi_file = ROI_FILES.get(roi_key)
        if roi_file is None:
            continue
        path = os.path.join(roi_dir, roi_file)
        if not os.path.exists(path):
            logger.info(f"  Skipping {roi_key}: file not found")
            continue

        atlas = nib.load(path).get_fdata().astype(np.int32)
        if combined is None:
            combined = np.zeros_like(atlas)

        # Add offset labels where atlas > 0 and combined == 0 (no overlap priority)
        new_mask = (atlas > 0) & (combined == 0)
        combined[new_mask] = atlas[new_mask] + offset
        n_new = len(np.unique(atlas[atlas > 0]))
        total_labels += n_new
        logger.info(f"  Added {roi_key}: {n_new} labels (offset {offset})")

    if combined is None:
        raise FileNotFoundError(f"No ROI files found in {roi_dir}")

    actual_labels = len(np.unique(combined[combined > 0]))
    logger.info(f"  Combined atlas: {actual_labels} unique labels (from {total_labels} total)")
    return combined


def get_atlas_within_mask(
    atlas: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Extract atlas labels within the nsdgeneral mask.

    Returns 1D array of length V (number of masked voxels), with label IDs.
    Background voxels within the mask get label 0.
    """
    return atlas[mask > 0].astype(np.int32)


def harmonize_atlas_labels(
    atlas_maps: dict[int, np.ndarray],
    masks: dict[int, np.ndarray],
    min_k: int = 10,
    min_voxels_per_parcel: int = 10,
) -> tuple[dict[int, np.ndarray], list[int], dict[int, int]]:
    """
    Harmonize atlas labels across subjects using intersection policy.

    Keeps only labels present in ALL subjects with >= min_voxels_per_parcel
    voxels each. Remaps surviving labels to contiguous integers [1, ..., R].

    Args:
        atlas_maps: sub_id -> 3D atlas
        masks: sub_id -> 3D nsdgeneral mask
        min_k: minimum required parcels (fail-fast)
        min_voxels_per_parcel: minimum voxels per parcel per subject

    Returns:
        harmonized: sub_id -> 1D remapped atlas (V_sub,)
        common_labels: sorted list of original label IDs present in all subjects
        label_remap: original_label -> new_label mapping
    """
    # Get labels per subject within mask
    per_subject_labels = {}
    for sub_id in atlas_maps:
        masked = atlas_maps[sub_id][masks[sub_id] > 0].astype(np.int32)
        # Count voxels per label
        counts = Counter(masked[masked > 0])
        # Keep labels with enough voxels
        valid = {lbl for lbl, cnt in counts.items() if cnt >= min_voxels_per_parcel}
        per_subject_labels[sub_id] = valid
        logger.info(f"Subject {sub_id}: {len(valid)} labels with >= {min_voxels_per_parcel} voxels")

    # Intersection
    # Normalize NumPy scalar labels to native Python ints so they are JSON-safe.
    common_labels = [int(lbl) for lbl in sorted(set.intersection(*per_subject_labels.values()))]
    logger.info(f"Common labels across all subjects: {len(common_labels)}")

    # Fail-fast
    if len(common_labels) < min_k + 1:
        raise ValueError(
            f"Only {len(common_labels)} common parcels (need >= {min_k + 1} for k={min_k}). "
            f"Try a different atlas_type or lower min_k."
        )

    # Remap to contiguous
    label_remap = {int(old): int(new) for new, old in enumerate(common_labels, start=1)}

    # Apply remapping
    harmonized = {}
    for sub_id in atlas_maps:
        masked = atlas_maps[sub_id][masks[sub_id] > 0].astype(np.int32)
        remapped = np.zeros_like(masked)
        for old, new in label_remap.items():
            remapped[masked == old] = new
        harmonized[sub_id] = remapped

    return harmonized, common_labels, label_remap


def atlas_utilization_summary(
    atlas_masked: np.ndarray,
    n_parcels: int,
    sub_id: int | None = None,
    min_voxels_per_parcel: int = 10,
    min_labeled_fraction_warn: float = 0.5,
) -> dict:
    """
    Summarize how well an atlas is utilized within a subject's nsdgeneral mask.

    Args:
        atlas_masked: 1D remapped labels (V_sub,), 0 means unlabeled.
        n_parcels: expected number of parcels after harmonization.
        sub_id: optional subject id for human-readable warnings.
        min_voxels_per_parcel: parcel-size threshold used for "small" warnings.
        min_labeled_fraction_warn: warn if labeled voxels / total masked voxels
            falls below this threshold.

    Returns:
        dict with coverage, parcel-size stats, and warning strings.
    """
    if atlas_masked.ndim != 1:
        raise ValueError(f"atlas_masked must be 1D, got shape {atlas_masked.shape}")
    if n_parcels < 1:
        raise ValueError(f"n_parcels must be >=1, got {n_parcels}")

    counts = Counter(atlas_masked[atlas_masked > 0])
    sizes = [int(counts.get(i, 0)) for i in range(1, n_parcels + 1)]
    nonzero_sizes = [s for s in sizes if s > 0]

    n_voxels = int(atlas_masked.shape[0])
    n_labeled = int((atlas_masked > 0).sum())
    labeled_fraction = float(n_labeled / n_voxels) if n_voxels > 0 else 0.0
    n_present = int(sum(s > 0 for s in sizes))
    empty = [i + 1 for i, s in enumerate(sizes) if s == 0]
    small = [i + 1 for i, s in enumerate(sizes) if 0 < s < min_voxels_per_parcel]

    subject_label = f"Subject {sub_id}" if sub_id is not None else "Subject"
    warnings = []
    if n_voxels == 0:
        warnings.append(f"{subject_label}: atlas_masked has zero voxels.")
    if n_labeled == 0:
        warnings.append(f"{subject_label}: no voxels assigned to atlas parcels.")
    if labeled_fraction < min_labeled_fraction_warn:
        warnings.append(
            f"{subject_label}: only {labeled_fraction:.3f} of nsdgeneral voxels have atlas labels."
        )
    if empty:
        warnings.append(f"{subject_label}: {len(empty)} empty parcels: {empty}")
    if small:
        warnings.append(
            f"{subject_label}: {len(small)} small parcels (<{min_voxels_per_parcel} voxels): {small}"
        )

    return {
        "sub_id": int(sub_id) if sub_id is not None else None,
        "n_voxels_masked": n_voxels,
        "n_voxels_labeled": n_labeled,
        "labeled_fraction": labeled_fraction,
        "n_parcels_expected": int(n_parcels),
        "n_parcels_present": n_present,
        "min_voxels_per_parcel": int(min_voxels_per_parcel),
        "min_voxels": int(min(nonzero_sizes)) if nonzero_sizes else 0,
        "median_voxels": float(np.median(nonzero_sizes)) if nonzero_sizes else 0.0,
        "mean_voxels": float(np.mean(nonzero_sizes)) if nonzero_sizes else 0.0,
        "max_voxels": int(max(nonzero_sizes)) if nonzero_sizes else 0,
        "empty_parcels": empty,
        "small_parcels": small,
        "warnings": warnings,
    }


def build_atlas_utilization_report(
    atlas_masked_by_subject: dict[int, np.ndarray],
    n_parcels: int,
    min_voxels_per_parcel: int = 10,
    min_labeled_fraction_warn: float = 0.5,
) -> dict:
    """
    Build a cross-subject atlas utilization report for saved training artifacts.
    """
    if not atlas_masked_by_subject:
        raise ValueError("atlas_masked_by_subject is empty.")

    summaries = {}
    labeled_fractions = []
    parcels_present = []
    subjects_with_warnings = []

    for sub_id in sorted(atlas_masked_by_subject):
        summary = atlas_utilization_summary(
            atlas_masked=atlas_masked_by_subject[sub_id],
            n_parcels=n_parcels,
            sub_id=sub_id,
            min_voxels_per_parcel=min_voxels_per_parcel,
            min_labeled_fraction_warn=min_labeled_fraction_warn,
        )
        summaries[str(int(sub_id))] = summary
        labeled_fractions.append(summary["labeled_fraction"])
        parcels_present.append(summary["n_parcels_present"])
        if summary["warnings"]:
            subjects_with_warnings.append(int(sub_id))

    return {
        "n_subjects": len(summaries),
        "n_parcels_expected": int(n_parcels),
        "min_voxels_per_parcel": int(min_voxels_per_parcel),
        "min_labeled_fraction_warn": float(min_labeled_fraction_warn),
        "labeled_fraction_min": float(np.min(labeled_fractions)),
        "labeled_fraction_median": float(np.median(labeled_fractions)),
        "labeled_fraction_max": float(np.max(labeled_fractions)),
        "parcels_present_min": int(np.min(parcels_present)),
        "parcels_present_median": float(np.median(parcels_present)),
        "parcels_present_max": int(np.max(parcels_present)),
        "subjects_with_warnings": subjects_with_warnings,
        "per_subject": summaries,
    }


def parcel_qc(
    atlas_masked: np.ndarray,
    n_parcels: int,
    sub_id: int,
    min_voxels_per_parcel: int = 10,
) -> dict:
    """
    QC check for parcellation within nsdgeneral mask.

    Args:
        atlas_masked: 1D array of remapped labels (V_sub,)
        n_parcels: expected number of parcels
        sub_id: subject ID for logging
        min_voxels_per_parcel: minimum voxels per parcel

    Returns:
        dict with QC metrics
    """
    counts = Counter(atlas_masked[atlas_masked > 0])
    sizes = [counts.get(i, 0) for i in range(1, n_parcels + 1)]
    empty = [i + 1 for i, s in enumerate(sizes) if s == 0]
    small = [i + 1 for i, s in enumerate(sizes) if 0 < s < min_voxels_per_parcel]

    result = {
        "sub_id": sub_id,
        "n_parcels": n_parcels,
        "min_voxels": min(sizes) if sizes else 0,
        "max_voxels": max(sizes) if sizes else 0,
        "mean_voxels": np.mean(sizes) if sizes else 0,
        "empty_parcels": empty,
        "small_parcels": small,
        "warnings": [],
    }

    if empty:
        result["warnings"].append(f"Subject {sub_id}: {len(empty)} empty parcels: {empty}")
    if small:
        result["warnings"].append(f"Subject {sub_id}: {len(small)} small parcels (<{min_voxels_per_parcel}): {small}")

    for w in result["warnings"]:
        logger.warning(w)

    return result
