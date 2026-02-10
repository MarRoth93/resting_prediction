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
    common_labels = sorted(set.intersection(*per_subject_labels.values()))
    logger.info(f"Common labels across all subjects: {len(common_labels)}")

    # Fail-fast
    if len(common_labels) < min_k + 1:
        raise ValueError(
            f"Only {len(common_labels)} common parcels (need >= {min_k + 1} for k={min_k}). "
            f"Try a different atlas_type or lower min_k."
        )

    # Remap to contiguous
    label_remap = {old: new for new, old in enumerate(common_labels, start=1)}

    # Apply remapping
    harmonized = {}
    for sub_id in atlas_maps:
        masked = atlas_maps[sub_id][masks[sub_id] > 0].astype(np.int32)
        remapped = np.zeros_like(masked)
        for old, new in label_remap.items():
            remapped[masked == old] = new
        harmonized[sub_id] = remapped

    return harmonized, common_labels, label_remap


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
