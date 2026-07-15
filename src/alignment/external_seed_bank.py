"""
Build cached whole-brain ROI seed time series for external-target CHA.

The prediction target remains the processed subject visual mask. This module
only creates matched REST seed time series from NSD-native ROI atlases so that
connectivity fingerprints can use many shared whole-brain targets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from src.data.prepare_rest_data import (
    array_sha256,
    build_motion_confounds,
    find_motion_file,
    load_motion_params,
    rest_preprocessing_hash,
    preprocess_rest_run,
    read_tr_from_nifti,
    validate_rest_provenance,
)

logger = logging.getLogger(__name__)


SEED_SET = "nsd_roi_seed_bank"
SEED_ATLAS_FILES = [
    "lh.HCP_MMP1.nii.gz",
    "rh.HCP_MMP1.nii.gz",
    "lh.corticalsulc.nii.gz",
    "rh.corticalsulc.nii.gz",
    "lh.streams.nii.gz",
    "rh.streams.nii.gz",
    "lh.MTL.nii.gz",
    "rh.MTL.nii.gz",
    "lh.thalamus.nii.gz",
    "rh.thalamus.nii.gz",
    "lh.prf-visualrois.nii.gz",
    "rh.prf-visualrois.nii.gz",
    "lh.prf-eccrois.nii.gz",
    "rh.prf-eccrois.nii.gz",
    "lh.floc-bodies.nii.gz",
    "rh.floc-bodies.nii.gz",
    "lh.floc-faces.nii.gz",
    "rh.floc-faces.nii.gz",
    "lh.floc-places.nii.gz",
    "rh.floc-places.nii.gz",
    "lh.floc-words.nii.gz",
    "rh.floc-words.nii.gz",
]


@dataclass(frozen=True)
class SeedDef:
    seed_set: str
    atlas_file: str
    label: int
    name: str


def subject_tag(sub: int) -> str:
    return f"subj{sub:02d}"


def roi_dir(raw_data_root: str | Path, sub: int) -> Path:
    return Path(raw_data_root) / "nsddata" / "ppdata" / subject_tag(sub) / "func1pt8mm" / "roi"


def timeseries_dir(raw_data_root: str | Path, sub: int) -> Path:
    return Path(raw_data_root) / "nsddata_timeseries" / "ppdata" / subject_tag(sub) / "func1pt8mm" / "timeseries"


def motion_dir(raw_data_root: str | Path, sub: int) -> Path:
    return Path(raw_data_root) / "nsddata_timeseries" / "ppdata" / subject_tag(sub) / "func1pt8mm" / "motion"


def seed_defs_to_jsonable(seed_defs: list[SeedDef]) -> list[dict]:
    return [asdict(seed) for seed in seed_defs]


def seed_defs_from_jsonable(rows: list[dict]) -> list[SeedDef]:
    return [
        SeedDef(
            seed_set=str(row["seed_set"]),
            atlas_file=str(row["atlas_file"]),
            label=int(row["label"]),
            name=str(row["name"]),
        )
        for row in rows
    ]


def load_rest_manifest(data_root: str | Path, raw_data_root: str | Path, sub: int) -> list[str]:
    manifest_path = Path(data_root) / subject_tag(sub) / "rest_run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            rest_runs = json.load(f).get("rest_runs", [])
        if rest_runs:
            return list(rest_runs)

    ts_dir = timeseries_dir(raw_data_root, sub)
    if not ts_dir.exists():
        raise FileNotFoundError(ts_dir)
    return sorted(p.name for p in ts_dir.glob("*.nii.gz"))


def load_atlas_array(raw_data_root: str | Path, sub: int, atlas_file: str) -> np.ndarray:
    path = roi_dir(raw_data_root, sub) / atlas_file
    if not path.exists():
        raise FileNotFoundError(
            f"Required ROI atlas is missing: {path}\n"
            "Download the full NSD ROI atlas set with:\n"
            f"  python -m src.data.download_nsddata --output-root {raw_data_root} --only-rois"
        )
    return nib.load(str(path)).get_fdata().astype(np.int32)


def build_common_seed_defs(
    seed_set: str,
    raw_data_root: str | Path,
    subjects: list[int],
    pred_masks: dict[int, np.ndarray],
    min_voxels_per_seed: int = 10,
) -> tuple[list[SeedDef], list[dict]]:
    """
    Build the seed definitions present in every requested subject.

    Labels are retained only when they have enough voxels in every subject.
    """
    if seed_set != SEED_SET:
        raise ValueError(f"The frozen pipeline requires seed_set={SEED_SET!r}.")

    per_subject_valid: list[set[tuple[str, int]]] = []
    coverage_rows: list[dict] = []
    for sub in subjects:
        valid_for_subject: set[tuple[str, int]] = set()
        pred_mask = np.asarray(pred_masks[sub], dtype=bool)
        for atlas_file in SEED_ATLAS_FILES:
            atlas = load_atlas_array(raw_data_root, sub, atlas_file)
            if atlas.shape != pred_mask.shape:
                raise ValueError(
                    f"Atlas {atlas_file} shape {atlas.shape} does not match "
                    f"prediction mask {pred_mask.shape} for subject {sub}."
                )
            labels = atlas[atlas > 0]
            counts = Counter(int(v) for v in labels.ravel() if int(v) > 0)
            for label, count in counts.items():
                if int(count) >= int(min_voxels_per_seed):
                    valid_for_subject.add((atlas_file, int(label)))
            coverage_rows.append(
                {
                    "seed_set": seed_set,
                    "subject": int(sub),
                    "atlas_file": atlas_file,
                    "n_labels_present": int(len(counts)),
                    "n_labels_valid_min_voxels": int(
                        sum(count >= int(min_voxels_per_seed) for count in counts.values())
                    ),
                    "n_labeled_voxels": int(np.sum(labels > 0)),
                }
            )
        per_subject_valid.append(valid_for_subject)

    common = sorted(set.intersection(*per_subject_valid), key=lambda x: (x[0], x[1]))
    seed_defs = [
        SeedDef(
            seed_set=seed_set,
            atlas_file=atlas_file,
            label=int(label),
            name=f"{Path(atlas_file).name.replace('.nii.gz', '')}:label{int(label):03d}",
        )
        for atlas_file, label in common
    ]
    if not seed_defs:
        raise ValueError(f"No common seeds found for seed_set={seed_set!r}.")
    return seed_defs, coverage_rows


def seed_bank_cache_id(seed_set: str, seed_defs: list[SeedDef], rest_cfg: dict) -> str:
    payload = {
        "seed_set": seed_set,
        "seed_defs": seed_defs_to_jsonable(seed_defs),
        "rest_preprocessing": rest_cfg,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:12]


def build_seed_masks(
    raw_data_root: str | Path,
    sub: int,
    seed_defs: list[SeedDef],
    pred_mask: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    labels_by_atlas: dict[str, np.ndarray] = {}
    seed_masks: list[np.ndarray] = []
    pred_mask = np.asarray(pred_mask, dtype=bool)

    for seed in seed_defs:
        if seed.atlas_file not in labels_by_atlas:
            labels_by_atlas[seed.atlas_file] = load_atlas_array(raw_data_root, sub, seed.atlas_file)
        atlas = labels_by_atlas[seed.atlas_file]
        mask = atlas == seed.label
        if not np.any(mask):
            raise ValueError(f"Seed has no voxels for subject {sub}: {seed}")
        seed_masks.append(mask)

    union_mask = np.zeros_like(pred_mask, dtype=bool)
    for seed_mask in seed_masks:
        union_mask |= seed_mask
    return union_mask, seed_masks


def _seed_column_indices(seed_union_mask: np.ndarray, seed_masks: list[np.ndarray]) -> list[np.ndarray]:
    seed_union_indices = np.flatnonzero(seed_union_mask.ravel())
    out: list[np.ndarray] = []
    for seed_mask in seed_masks:
        flat_seed = np.flatnonzero(seed_mask.ravel())
        positions = np.searchsorted(seed_union_indices, flat_seed)
        if not np.all(seed_union_indices[positions] == flat_seed):
            raise RuntimeError("Internal seed index mapping failed")
        out.append(positions.astype(np.int32, copy=False))
    return out


def average_seed_timeseries(seed_voxel_ts: np.ndarray, seed_col_indices: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((seed_voxel_ts.shape[0], len(seed_col_indices)), dtype=np.float32)
    for i, cols in enumerate(seed_col_indices):
        if cols.size == 0:
            raise ValueError(f"Seed index {i} has no columns in union labels")
        out[:, i] = seed_voxel_ts[:, cols].mean(axis=1)
    return out


def _build_rest_confounds(
    rest_file: str,
    motion_dir_path: Path,
    n_trs: int,
    rest_cfg: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    nuisance_raw_cfg = rest_cfg.get("nuisance_regression", {}) or {}
    nuisance_enabled = (
        bool(nuisance_raw_cfg.get("enabled", False))
        if isinstance(nuisance_raw_cfg, dict)
        else bool(nuisance_raw_cfg)
    )
    nuisance_cfg = nuisance_raw_cfg if isinstance(nuisance_raw_cfg, dict) else {}
    motion_censoring_enabled = bool((rest_cfg.get("motion_censoring", {}) or {}).get("enabled", False))
    if not (nuisance_enabled or motion_censoring_enabled):
        return None, None

    motion_file = find_motion_file(rest_file, str(motion_dir_path))
    if motion_file is None:
        if nuisance_enabled and bool(nuisance_cfg.get("require_motion", False)):
            raise FileNotFoundError(f"No motion params found for {rest_file} in {motion_dir_path}")
        logger.info("No motion params found for %s; skipping motion-based steps", rest_file)
        return None, None

    motion_params = load_motion_params(motion_file, expected_trs=n_trs)
    nuisance_regressors = None
    if nuisance_enabled:
        motion_model = str(nuisance_cfg.get("motion_model", "friston24"))
        if motion_model.lower() not in {"none", "", "false"}:
            nuisance_regressors = build_motion_confounds(
                motion_params,
                model=motion_model,
                standardize=bool(nuisance_cfg.get("standardize", True)),
            )
    return motion_params, nuisance_regressors


def preprocess_seed_run(
    raw_data_root: str | Path,
    sub: int,
    rest_file: str,
    pred_mask: np.ndarray,
    seed_union_mask: np.ndarray,
    seed_col_indices: list[np.ndarray],
    rest_cfg: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    img_path = timeseries_dir(raw_data_root, sub) / rest_file
    img = nib.load(str(img_path))
    tr = read_tr_from_nifti(img)
    raw = np.asarray(img.dataobj, dtype=np.float32)

    pred_raw = raw[pred_mask].T
    seed_raw = raw[seed_union_mask].T
    combined = np.concatenate([pred_raw, seed_raw], axis=1)
    del raw, pred_raw, seed_raw

    motion_params, nuisance_regressors = _build_rest_confounds(
        rest_file=rest_file,
        motion_dir_path=motion_dir(raw_data_root, sub),
        n_trs=combined.shape[0],
        rest_cfg=rest_cfg,
    )
    motion_cfg = rest_cfg.get("motion_censoring", {}) or {}
    processed = preprocess_rest_run(
        combined,
        tr=tr,
        discard_initial_trs=int(rest_cfg.get("discard_initial_trs", 5)),
        detrend=bool(rest_cfg.get("detrend", True)),
        highpass_cutoff_hz=rest_cfg.get("highpass_cutoff_hz", 0.01),
        motion_params=motion_params,
        motion_censoring_enabled=bool(motion_cfg.get("enabled", False)),
        motion_censoring_strategy=str(motion_cfg.get("strategy", "drop")),
        fd_threshold_mm=float(motion_cfg.get("fd_threshold_mm", 0.5)),
        max_censored_fraction=float(motion_cfg.get("max_censored_fraction", 0.3)),
        nuisance_regressors=nuisance_regressors,
        zscore=bool(rest_cfg.get("zscore", True)),
    )
    if processed is None:
        return None, None

    n_pred_voxels = int(np.asarray(pred_mask, dtype=bool).sum())
    pred_ts = processed[:, :n_pred_voxels]
    seed_voxel_ts = processed[:, n_pred_voxels:]
    seed_ts = average_seed_timeseries(seed_voxel_ts, seed_col_indices)
    return seed_ts, pred_ts


def _load_cached_seed_runs(
    cache_dir: Path,
    n_runs: int,
    n_seeds: int,
    reference_rest_runs: list[np.ndarray] | None,
) -> list[np.ndarray] | None:
    seed_runs = []
    for i in range(n_runs):
        path = cache_dir / f"external_seed_run{i + 1}.npy"
        if not path.exists():
            return None
        arr = np.load(path)
        if arr.ndim != 2 or int(arr.shape[1]) != int(n_seeds):
            return None
        if reference_rest_runs is not None and int(arr.shape[0]) != int(reference_rest_runs[i].shape[0]):
            return None
        seed_runs.append(arr.astype(np.float32, copy=False))
    return seed_runs


def load_or_prepare_external_seed_runs(
    sub: int,
    data_root: str | Path,
    raw_data_root: str | Path,
    pred_mask: np.ndarray,
    seed_defs: list[SeedDef],
    rest_cfg: dict,
    seed_set: str,
    reference_rest_runs: list[np.ndarray] | None = None,
    force_recompute: bool = False,
) -> list[np.ndarray]:
    """
    Load cached external seed time series, or create them from raw REST NIfTIs.
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    reference_provenance = validate_rest_provenance(
        data_root=data_root,
        sub=sub,
        expected_config=rest_cfg,
        expected_mask=pred_mask,
        reference_rest_runs=reference_rest_runs,
    )
    rest_files = load_rest_manifest(data_root, raw_data_root, sub)
    cache_id = seed_bank_cache_id(seed_set, seed_defs, rest_cfg)
    cache_dir = Path(data_root) / subject_tag(sub) / "external_seed_banks" / f"{seed_set}_{cache_id}"
    manifest_path = cache_dir / "manifest.json"

    if reference_rest_runs is not None and len(rest_files) != len(reference_rest_runs):
        raise ValueError(
            f"Subject {sub}: manifest has {len(rest_files)} REST files but processed data has "
            f"{len(reference_rest_runs)} rest_run arrays."
        )

    cache_manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            cache_manifest = json.load(f)
    cache_matches_reference = bool(
        cache_manifest
        and cache_manifest.get("reference_rest_provenance_hash")
        == reference_provenance.get("provenance_hash")
        and cache_manifest.get("prediction_mask_sha256") == array_sha256(pred_mask)
        and cache_manifest.get("rest_preprocessing_hash") == rest_preprocessing_hash(rest_cfg)
    )

    if not force_recompute and cache_matches_reference:
        cached = _load_cached_seed_runs(
            cache_dir=cache_dir,
            n_runs=len(rest_files),
            n_seeds=len(seed_defs),
            reference_rest_runs=reference_rest_runs,
        )
        if cached is not None:
            logger.info(
                "Subject %d: loaded cached external seed bank %s (%d runs, %d seeds)",
                sub,
                cache_dir,
                len(cached),
                len(seed_defs),
            )
            return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_union_mask, seed_masks = build_seed_masks(raw_data_root, sub, seed_defs, pred_mask)
    seed_col_indices = _seed_column_indices(seed_union_mask, seed_masks)

    seed_runs: list[np.ndarray] = []
    kept_rest_files: list[str] = []
    for rest_idx, rest_file in enumerate(rest_files):
        logger.info("Subject %d: preparing external seeds for %s", sub, rest_file)
        seed_ts, pred_ts = preprocess_seed_run(
            raw_data_root=raw_data_root,
            sub=sub,
            rest_file=rest_file,
            pred_mask=pred_mask,
            seed_union_mask=seed_union_mask,
            seed_col_indices=seed_col_indices,
            rest_cfg=rest_cfg,
        )
        if seed_ts is None or pred_ts is None:
            logger.warning("Subject %d: external seed run excluded for %s", sub, rest_file)
            continue
        if reference_rest_runs is not None:
            ref = reference_rest_runs[len(seed_runs)]
            if int(seed_ts.shape[0]) != int(ref.shape[0]):
                raise ValueError(
                    f"Subject {sub} {rest_file}: external seed TRs={seed_ts.shape[0]} "
                    f"but processed rest_run{len(seed_runs) + 1} has {ref.shape[0]} TRs. "
                    "Re-run prepare_rest_data with the same rest_preprocessing config, "
                    "or clear the external seed cache if stale."
                )
            if int(pred_ts.shape[1]) != int(ref.shape[1]):
                raise ValueError(
                    f"Subject {sub} {rest_file}: visual voxel count mismatch between "
                    f"raw-preprocessed seed helper ({pred_ts.shape[1]}) and processed REST ({ref.shape[1]})."
                )
        seed_runs.append(seed_ts.astype(np.float32, copy=False))
        kept_rest_files.append(rest_file)
        np.save(cache_dir / f"external_seed_run{len(seed_runs)}.npy", seed_runs[-1])

    if reference_rest_runs is not None and len(seed_runs) != len(reference_rest_runs):
        raise ValueError(
            f"Subject {sub}: external seed helper kept {len(seed_runs)} runs but "
            f"processed REST has {len(reference_rest_runs)} runs."
        )
    if not seed_runs:
        raise ValueError(f"Subject {sub}: no usable external seed runs.")

    manifest = {
        "subject": int(sub),
        "seed_set": seed_set,
        "cache_id": cache_id,
        "n_seeds": int(len(seed_defs)),
        "n_seed_union_voxels": int(seed_union_mask.sum()),
        "rest_files": kept_rest_files,
        "seed_defs": seed_defs_to_jsonable(seed_defs),
        "rest_preprocessing": rest_cfg,
        "rest_preprocessing_hash": rest_preprocessing_hash(rest_cfg),
        "prediction_mask_sha256": array_sha256(pred_mask),
        "reference_rest_provenance_hash": reference_provenance["provenance_hash"],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Subject %d: saved external seed bank to %s (%d runs, %d seeds)",
        sub,
        cache_dir,
        len(seed_runs),
        len(seed_defs),
    )
    return seed_runs


def save_external_seed_info(
    output_dir: str | Path,
    seed_set: str,
    seed_defs: list[SeedDef],
    coverage_rows: Iterable[dict],
    rest_cfg: dict,
    min_voxels_per_seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    info = {
        "seed_set": seed_set,
        "n_seeds": int(len(seed_defs)),
        "min_voxels_per_seed": int(min_voxels_per_seed),
        "cache_id": seed_bank_cache_id(seed_set, seed_defs, rest_cfg),
        "seed_defs": seed_defs_to_jsonable(seed_defs),
        "coverage": list(coverage_rows),
        "rest_preprocessing": rest_cfg,
    }
    with open(Path(output_dir) / "external_seed_info.json", "w") as f:
        json.dump(info, f, indent=2)
    np.save(
        Path(output_dir) / "external_seed_names.npy",
        np.array([seed.name for seed in seed_defs], dtype=object),
    )


def load_external_seed_info(model_dir: str | Path) -> tuple[str, list[SeedDef], dict]:
    path = Path(model_dir) / "external_seed_info.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing external seed-bank metadata: {path}. "
            "Re-run shared-space training with connectivity_mode='external_seed_bank'."
        )
    with open(path) as f:
        info = json.load(f)
    seed_set = str(info["seed_set"])
    if seed_set != SEED_SET:
        raise ValueError(f"Unsupported seed set in {path}: {seed_set!r}.")
    seed_defs = seed_defs_from_jsonable(info["seed_defs"])
    rest_cfg = info.get("rest_preprocessing", {}) or {}
    return seed_set, seed_defs, rest_cfg
