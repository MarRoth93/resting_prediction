"""
Full training pipeline: build shared space + train encoder.
"""

import hashlib
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import yaml

from src.alignment.shared_space import SharedSpaceBuilder
from src.data.load_atlas import (
    build_atlas_utilization_report,
    harmonize_atlas_labels,
    load_atlas,
    parcel_qc,
)
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.models.encoding import SharedSpaceEncoder

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _validate_subject_row_contract(subject: NSDSubjectData) -> None:
    """Fail fast on mismatched rows/voxels within one processed subject."""
    train_rows = int(subject.train_fmri.shape[0])
    test_rows = int(subject.test_fmri.shape[0])
    if train_rows != int(subject.train_stim_idx.shape[0]):
        raise ValueError(
            f"Subject {subject.sub}: train row mismatch train_fmri={train_rows}, "
            f"train_stim_idx={int(subject.train_stim_idx.shape[0])}."
        )
    if test_rows != int(subject.test_stim_idx.shape[0]):
        raise ValueError(
            f"Subject {subject.sub}: test row mismatch test_fmri={test_rows}, "
            f"test_stim_idx={int(subject.test_stim_idx.shape[0])}."
        )

    task_voxels = int(subject.test_fmri.shape[1])
    if int(subject.train_fmri.shape[1]) != task_voxels:
        raise ValueError(
            f"Subject {subject.sub}: train/test voxel mismatch train_fmri="
            f"{int(subject.train_fmri.shape[1])}, test_fmri={task_voxels}."
        )
    if subject.rest_runs:
        rest_voxels = {int(run.shape[1]) for run in subject.rest_runs}
        if len(rest_voxels) != 1 or next(iter(rest_voxels)) != task_voxels:
            raise ValueError(
                f"Subject {subject.sub}: REST voxel dimensions {sorted(rest_voxels)} "
                f"do not match task voxels {task_voxels}."
            )


def _validate_feature_contract(
    subject: NSDSubjectData,
    features: NSDFeatures,
    feature_type: str,
) -> None:
    """Ensure subject stimulus indices are valid for chosen feature backbone."""
    all_idx = np.concatenate([subject.train_stim_idx, subject.test_stim_idx]).astype(np.int64)
    if all_idx.size == 0:
        raise ValueError(f"Subject {subject.sub}: empty stimulus index arrays.")
    if np.any(all_idx < 0):
        raise ValueError(f"Subject {subject.sub}: negative stimulus indices detected.")

    max_idx = int(all_idx.max())
    try:
        probe = features.get_features(np.array([max_idx], dtype=np.int64), feature_type)
    except Exception as exc:
        raise ValueError(
            f"Subject {subject.sub}: stimulus index {max_idx} is invalid for feature_type={feature_type}."
        ) from exc
    if int(probe.shape[0]) != 1:
        raise ValueError(
            f"Subject {subject.sub}: feature probe returned unexpected shape {probe.shape}."
        )


def _build_shared_stimulus_intersection(
    subjects: dict[int, NSDSubjectData],
    train_subs: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """
    Build canonical shared stimulus rows via intersection of test_stim_idx.

    Returns:
        shared_stim_idx: sorted NSD stimulus IDs used for hybrid alignment rows.
        shared_rows: sub_id -> row indices into subject.test_fmri aligned to shared_stim_idx.
        dropped_counts: sub_id -> number of subject test stimuli excluded by intersection.
    """
    if not train_subs:
        raise ValueError("No training subjects provided.")

    per_subject_sets: dict[int, set[int]] = {}
    for s in train_subs:
        stim_idx = np.asarray(subjects[s].test_stim_idx, dtype=np.int64).ravel()
        if stim_idx.size == 0:
            raise ValueError(f"Subject {s}: empty test_stim_idx.")
        unique_idx = np.unique(stim_idx)
        if unique_idx.size != stim_idx.size:
            raise ValueError(f"Subject {s}: duplicate entries found in test_stim_idx.")
        if not np.array_equal(unique_idx, stim_idx):
            raise ValueError(
                f"Subject {s}: test_stim_idx must be sorted ascending for deterministic alignment."
            )
        per_subject_sets[s] = set(stim_idx.tolist())

    shared_set = set.intersection(*(per_subject_sets[s] for s in train_subs))
    if not shared_set:
        raise ValueError(
            "No overlapping test stimuli found across training subjects; cannot build hybrid shared space."
        )
    shared_stim_idx = np.array(sorted(shared_set), dtype=np.int64)

    shared_rows: dict[int, np.ndarray] = {}
    dropped_counts: dict[int, int] = {}
    for s in train_subs:
        subj_idx = np.asarray(subjects[s].test_stim_idx, dtype=np.int64).ravel()
        rows = np.searchsorted(subj_idx, shared_stim_idx)
        valid = (rows < subj_idx.size) & (subj_idx[rows] == shared_stim_idx)
        if not np.all(valid):
            missing = shared_stim_idx[~valid][:10].tolist()
            raise ValueError(
                f"Subject {s}: failed to map shared stimuli into test_stim_idx. "
                f"Example missing IDs: {missing}"
            )
        shared_rows[s] = rows.astype(np.int64, copy=False)
        dropped_counts[s] = int(subj_idx.size - shared_stim_idx.size)

    return shared_stim_idx, shared_rows, dropped_counts


def train_pipeline(
    config_path: str = "config.yaml",
    data_root: str = "processed_data",
    raw_data_root: str = ".",
    output_dir: str = "outputs/shared_space",
):
    """
    Complete training pipeline.

    1. Load all training subjects' data
    2. Load/harmonize atlas
    3. Build shared space from REST + shared stimuli
    4. Train global encoder
    5. Save model artifacts with provenance
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("random_seed", 42)
    set_seeds(seed)

    train_subs = config["subjects"]["train"]
    n_components = config["alignment"]["n_components"]
    min_k = config["alignment"]["min_k"]
    connectivity_mode = config["alignment"]["connectivity_mode"]
    experiment_mode = config["alignment"]["experiment_mode"]
    atlas_type = config["alignment"]["atlas_type"]
    min_voxels_per_parcel = int(config["alignment"].get("min_voxels_per_parcel", 10))
    min_labeled_fraction_warn = float(config["alignment"].get("min_labeled_fraction_warn", 0.5))
    ridge_alpha = config["encoding"]["ridge_alpha"]
    feature_type = config["features"]["type"]

    logger.info(f"Training pipeline: subjects={train_subs}, mode={experiment_mode}")
    logger.info(f"  connectivity={connectivity_mode}, atlas={atlas_type}, k={n_components}")

    # Step 1: Load data
    subjects = {s: NSDSubjectData(s, data_root) for s in train_subs}
    features = NSDFeatures(os.path.join(data_root, "features"))
    for s in train_subs:
        _validate_subject_row_contract(subjects[s])
        _validate_feature_contract(subjects[s], features, feature_type)

    # Step 2: Load and harmonize atlas (if parcellation mode)
    atlas_masked = None
    n_parcels = None
    atlas_utilization_report = None

    if connectivity_mode == "parcellation":
        # Load atlas for all subjects (including test subject for harmonization)
        all_subs = train_subs + config["subjects"]["test"]
        atlas_maps = {}
        masks = {}
        for s in all_subs:
            atlas_maps[s] = load_atlas(s, atlas_type, raw_data_root)
            if s in subjects:
                masks[s] = subjects[s].mask
            else:
                # Load mask for test subject too (for harmonization only)
                masks[s] = np.load(os.path.join(data_root, f"subj{s:02d}/mask.npy"))

        harmonized, common_labels, label_remap = harmonize_atlas_labels(
            atlas_maps,
            masks,
            min_k=min_k,
            min_voxels_per_parcel=min_voxels_per_parcel,
        )
        n_parcels = len(common_labels)
        logger.info(f"Atlas harmonized: {n_parcels} common parcels")

        # QC
        atlas_masked = {}
        for s in all_subs:
            atlas_masked[s] = harmonized[s]
            expected_v = int(masks[s].sum())
            if int(harmonized[s].shape[0]) != expected_v:
                raise ValueError(
                    f"Subject {s}: harmonized atlas length {harmonized[s].shape[0]} "
                    f"does not match nsdgeneral mask voxels {expected_v}."
                )
            if s in subjects:
                subj_v = int(subjects[s].test_fmri.shape[1])
                if int(harmonized[s].shape[0]) != subj_v:
                    raise ValueError(
                        f"Subject {s}: harmonized atlas length {harmonized[s].shape[0]} "
                        f"does not match task data voxels {subj_v}."
                    )
                if subjects[s].rest_runs:
                    rest_v = int(subjects[s].rest_runs[0].shape[1])
                    if int(harmonized[s].shape[0]) != rest_v:
                        raise ValueError(
                            f"Subject {s}: harmonized atlas length {harmonized[s].shape[0]} "
                            f"does not match REST data voxels {rest_v}."
                        )

            qc = parcel_qc(
                harmonized[s],
                n_parcels,
                s,
                min_voxels_per_parcel=min_voxels_per_parcel,
            )
            for w in qc["warnings"]:
                logger.warning(w)

        atlas_utilization_report = build_atlas_utilization_report(
            atlas_masked,
            n_parcels=n_parcels,
            min_voxels_per_parcel=min_voxels_per_parcel,
            min_labeled_fraction_warn=min_labeled_fraction_warn,
        )
        logger.info(
            "Atlas utilization: labeled fraction min/median/max = %.3f / %.3f / %.3f",
            atlas_utilization_report["labeled_fraction_min"],
            atlas_utilization_report["labeled_fraction_median"],
            atlas_utilization_report["labeled_fraction_max"],
        )
        if atlas_utilization_report["subjects_with_warnings"]:
            logger.warning(
                "Atlas utilization warnings for subjects: %s",
                atlas_utilization_report["subjects_with_warnings"],
            )

    # Step 3: Build shared space
    rest_runs = {s: subjects[s].rest_runs for s in train_subs}
    task_responses_shared = {}

    shared_stim_idx, shared_test_rows, dropped_test_counts = _build_shared_stimulus_intersection(
        subjects=subjects,
        train_subs=train_subs,
    )
    logger.info(
        "Shared-stimulus intersection size for hybrid alignment: %d",
        int(shared_stim_idx.shape[0]),
    )

    for s in train_subs:
        n_total = int(subjects[s].test_stim_idx.shape[0])
        n_shared = int(shared_test_rows[s].shape[0])
        if dropped_test_counts[s] > 0:
            logger.info(
                "Subject %d: using %d/%d test stimuli for shared alignment (dropped %d).",
                s,
                n_shared,
                n_total,
                dropped_test_counts[s],
            )
        task_responses_shared[s] = np.array(
            subjects[s].test_fmri[shared_test_rows[s]],
            dtype=np.float32,
        )

    train_atlas = {s: atlas_masked[s] for s in train_subs} if atlas_masked else None

    builder = SharedSpaceBuilder(
        n_components=n_components,
        min_k=min_k,
        connectivity_mode=connectivity_mode,
        experiment_mode=experiment_mode,
        ensemble_method=config["alignment"]["ensemble_method"],
        max_iters=config["alignment"]["max_iters"],
        tol=config["alignment"]["tol"],
    )
    builder.fit(
        rest_runs=rest_runs,
        task_responses_shared=task_responses_shared,
        atlas_masked=train_atlas,
        n_parcels=n_parcels,
    )

    # Step 4: Prepare training data in shared space
    X_all, Z_all = [], []
    for sub_id in train_subs:
        subj = subjects[sub_id]
        X = features.get_features(subj.train_stim_idx, feature_type)
        # Project to component space and rotate to shared space
        P = builder.subject_bases[sub_id]
        R = builder.subject_rotations[sub_id]
        Z = np.array(subj.train_fmri, dtype=np.float32) @ P @ R  # (N, k)
        X_all.append(X)
        Z_all.append(Z)
        logger.info(f"Subject {sub_id}: X {X.shape}, Z {Z.shape}")

    X_concat = np.concatenate(X_all, axis=0)
    Z_concat = np.concatenate(Z_all, axis=0)
    logger.info(f"Pooled training: X {X_concat.shape}, Z {Z_concat.shape}")

    # Step 5: Train encoder
    encoder = SharedSpaceEncoder(alpha=ridge_alpha)
    encoder.fit(X_concat, Z_concat)

    # Step 6: Save
    os.makedirs(output_dir, exist_ok=True)
    builder.save(output_dir)
    encoder.save(os.path.join(output_dir, "encoder.npz"))
    np.save(os.path.join(output_dir, "shared_stim_idx.npy"), shared_stim_idx)

    # Save atlas info
    if n_parcels:
        serializable_label_remap = {int(old): int(new) for old, new in label_remap.items()}
        np.savez(
            os.path.join(output_dir, "atlas_info.npz"),
            common_labels=common_labels,
            label_remap=json.dumps(serializable_label_remap),
            n_parcels=n_parcels,
            atlas_type=atlas_type,
        )
        # Save per-subject atlas for inference
        for s in atlas_masked:
            np.save(os.path.join(output_dir, f"atlas_masked_{s}.npy"), atlas_masked[s])
        if atlas_utilization_report is not None:
            atlas_utilization_report["atlas_type"] = atlas_type
            with open(os.path.join(output_dir, "atlas_utilization_report.json"), "w") as f:
                json.dump(atlas_utilization_report, f, indent=2)

    # Save provenance metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
        "train_subjects": train_subs,
        "k_global": builder.k_global,
        "n_parcels": n_parcels,
        "n_train_samples": X_concat.shape[0],
        "experiment_mode": experiment_mode,
        "shared_stimulus_strategy": "intersection",
        "n_shared_stimuli": int(shared_stim_idx.shape[0]),
        "per_subject_dropped_test_stimuli": {
            str(s): int(dropped_test_counts[s]) for s in train_subs
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Saved to {output_dir}")
    return builder, encoder


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train shared space model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--raw-data-root", default=".")
    parser.add_argument("--output-dir", default="outputs/shared_space")
    args = parser.parse_args()

    train_pipeline(args.config, args.data_root, args.raw_data_root, args.output_dir)
