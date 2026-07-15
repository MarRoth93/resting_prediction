"""
Full training pipeline: build shared space + train encoder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
from datetime import datetime

import numpy as np

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData, resolve_feature_streams
from src.data.shared_paths import default_raw_data_root
from src.models.encoding_factory import build_encoder, save_encoder

logger = logging.getLogger(__name__)


def _feature_streams_for_training(config: dict, feature_type: str) -> list[str]:
    features_cfg = config.get("features", {}) or {}
    return resolve_feature_streams(feature_type, features_cfg.get("streams"))


def _get_feature_matrix_and_slices(
    features: NSDFeatures,
    stim_idx: np.ndarray,
    feature_type: str,
    streams: list[str] | None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]] | None]:
    if streams is None:
        return features.get_features(stim_idx, feature_type), None
    bundle = features.get_feature_bundle(stim_idx, streams)
    return bundle.array, bundle.slices


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
    data_root: str = "data/processed",
    raw_data_root: str = default_raw_data_root(),
    output_dir: str = "artifacts/retrained_model",
):
    """
    Complete training pipeline.

    1. Load all training subjects' data
    2. Build the common external REST seed bank
    3. Build the shared space from REST + shared stimuli
    4. Train and save the static transformer
    """
    config = load_config(config_path)

    seed = config.get("random_seed", 42)
    set_seeds(seed)

    train_subs = config["subjects"]["train"]
    n_components = config["alignment"]["n_components"]
    min_k = config["alignment"]["min_k"]
    external_seed_cfg = config["alignment"]["external_seed_bank"]
    feature_type = str(config["features"]["type"])
    feature_streams = _feature_streams_for_training(config, feature_type)
    logger.info("Using feature streams for static transformer training: %s", feature_streams)

    logger.info("Training external-seed hybrid alignment for subjects %s, k=%d", train_subs, n_components)

    # Step 1: Load data
    subjects = {s: NSDSubjectData(s, data_root) for s in train_subs}
    features = NSDFeatures(os.path.join(data_root, "features"))
    for s in train_subs:
        _validate_subject_row_contract(subjects[s])
        _validate_feature_contract(subjects[s], features, feature_type)

    # Step 2: Build the common seed definition and matched seed time series.
    from src.alignment.external_seed_bank import (
        build_common_seed_defs,
        load_or_prepare_external_seed_runs,
    )

    all_subs = train_subs + config["subjects"]["test"]
    masks = {
        s: subjects[s].mask
        if s in subjects
        else np.load(os.path.join(data_root, f"subj{s:02d}/mask.npy"))
        for s in all_subs
    }
    external_seed_set = str(external_seed_cfg["seed_set"])
    min_voxels_per_seed = int(external_seed_cfg["min_voxels_per_seed"])
    external_seed_defs, external_seed_coverage = build_common_seed_defs(
        seed_set=external_seed_set,
        raw_data_root=raw_data_root,
        subjects=all_subs,
        pred_masks=masks,
        min_voxels_per_seed=min_voxels_per_seed,
    )
    if len(external_seed_defs) < min_k:
        raise ValueError(
            f"External seed bank has only {len(external_seed_defs)} seeds, below min_k={min_k}."
        )
    logger.info("External seed bank %s: %d common seeds", external_seed_set, len(external_seed_defs))

    rest_cfg = config["rest_preprocessing"]
    external_seed_runs = {
        s: load_or_prepare_external_seed_runs(
            sub=s,
            data_root=data_root,
            raw_data_root=raw_data_root,
            pred_mask=masks[s],
            seed_defs=external_seed_defs,
            rest_cfg=rest_cfg,
            seed_set=external_seed_set,
            reference_rest_runs=subjects[s].rest_runs,
            force_recompute=bool(external_seed_cfg["force_recompute"]),
        )
        for s in train_subs
    }

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

    builder = SharedSpaceBuilder(
        n_components=n_components,
        min_k=min_k,
        ensemble_method=config["alignment"]["ensemble_method"],
        max_iters=config["alignment"]["max_iters"],
        tol=config["alignment"]["tol"],
    )
    builder.fit(
        rest_runs=rest_runs,
        task_responses_shared=task_responses_shared,
        external_seed_runs=external_seed_runs,
    )

    # Step 4: Prepare training data in shared space
    X_all, Z_all, sample_groups_all = [], [], []
    feature_slices = None
    for sub_id in train_subs:
        subj = subjects[sub_id]
        X, slices = _get_feature_matrix_and_slices(
            features=features,
            stim_idx=subj.train_stim_idx,
            feature_type=feature_type,
            streams=feature_streams,
        )
        if feature_slices is None and slices is not None:
            feature_slices = slices
        # Project to component space and rotate to shared space
        P = builder.subject_bases[sub_id]
        R = builder.subject_rotations[sub_id]
        Z = np.array(subj.train_fmri, dtype=np.float32) @ P @ R  # (N, k)
        X_all.append(X)
        Z_all.append(Z)
        sample_groups_all.append(np.asarray(subj.train_stim_idx, dtype=np.int64))
        logger.info(f"Subject {sub_id}: X {X.shape}, Z {Z.shape}")

    X_concat = np.concatenate(X_all, axis=0)
    Z_concat = np.concatenate(Z_all, axis=0)
    sample_groups_concat = np.concatenate(sample_groups_all, axis=0)
    logger.info(f"Pooled training: X {X_concat.shape}, Z {Z_concat.shape}")

    # Step 5: Train encoder
    encoder = build_encoder(
        config=config,
        input_dim=int(X_concat.shape[1]),
        output_dim=int(Z_concat.shape[1]),
        feature_slices=feature_slices,
    )
    encoder.fit(X_concat, Z_concat, sample_groups=sample_groups_concat)

    # Step 6: Save
    os.makedirs(output_dir, exist_ok=True)
    builder.save(output_dir)
    encoder_artifact = save_encoder(encoder, output_dir)
    np.save(os.path.join(output_dir, "shared_stim_idx.npy"), shared_stim_idx)

    from src.alignment.external_seed_bank import save_external_seed_info

    save_external_seed_info(
        output_dir=output_dir,
        seed_set=external_seed_set,
        seed_defs=external_seed_defs,
        coverage_rows=external_seed_coverage,
        rest_cfg=rest_cfg,
        min_voxels_per_seed=min_voxels_per_seed,
    )

    # Save provenance metadata
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()
    effective_config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "effective_config_hash": effective_config_hash,
        "analysis_mask": copy.deepcopy(config.get("analysis_mask", {})),
        "train_subjects": train_subs,
        "feature_type": feature_type,
        "k_global": builder.k_global,
        "n_train_samples": X_concat.shape[0],
        "experiment_mode": builder.experiment_mode,
        "connectivity_mode": builder.connectivity_mode,
        "shared_stimulus_strategy": "intersection",
        "n_shared_stimuli": int(shared_stim_idx.shape[0]),
        "per_subject_dropped_test_stimuli": {
            str(s): int(dropped_test_counts[s]) for s in train_subs
        },
        "encoding": {
            "architecture": str(encoder.architecture),
            "encoder_artifact": encoder_artifact,
            "feature_streams": feature_streams,
            "feature_slices": feature_slices,
        },
    }
    metadata["external_seed_bank"] = {
        "seed_set": external_seed_set,
        "n_seeds": int(len(external_seed_defs)),
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
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--raw-data-root", default=default_raw_data_root())
    parser.add_argument("--output-dir", default="artifacts/retrained_model")
    args = parser.parse_args()

    train_pipeline(
        args.config,
        args.data_root,
        args.raw_data_root,
        args.output_dir,
    )
