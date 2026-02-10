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
    get_atlas_within_mask,
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
    ridge_alpha = config["encoding"]["ridge_alpha"]
    feature_type = config["features"]["type"]

    logger.info(f"Training pipeline: subjects={train_subs}, mode={experiment_mode}")
    logger.info(f"  connectivity={connectivity_mode}, atlas={atlas_type}, k={n_components}")

    # Step 1: Load data
    subjects = {s: NSDSubjectData(s, data_root) for s in train_subs}
    features = NSDFeatures(os.path.join(data_root, "features"))

    # Step 2: Load and harmonize atlas (if parcellation mode)
    atlas_masked = None
    n_parcels = None

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
            atlas_maps, masks, min_k=min_k
        )
        n_parcels = len(common_labels)
        logger.info(f"Atlas harmonized: {n_parcels} common parcels")

        # QC
        atlas_masked = {}
        for s in all_subs:
            atlas_masked[s] = harmonized[s]
            qc = parcel_qc(harmonized[s], n_parcels, s)
            for w in qc["warnings"]:
                logger.warning(w)

    # Step 3: Build shared space
    rest_runs = {s: subjects[s].rest_runs for s in train_subs}
    task_responses_shared = {}

    # Validate shared-stimulus row correspondence across training subjects
    ref_stim_idx = subjects[train_subs[0]].test_stim_idx
    for s in train_subs[1:]:
        if not np.array_equal(subjects[s].test_stim_idx, ref_stim_idx):
            raise ValueError(
                f"Shared stimulus ordering mismatch between subject {train_subs[0]} "
                f"and subject {s}. test_stim_idx must be identical across all training "
                f"subjects for Procrustes alignment row correspondence. "
                f"Sub {train_subs[0]}: {len(ref_stim_idx)} stimuli, Sub {s}: {len(subjects[s].test_stim_idx)} stimuli."
            )

    for s in train_subs:
        task_responses_shared[s] = np.array(subjects[s].test_fmri, dtype=np.float32)

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

    # Save atlas info
    if n_parcels:
        np.savez(
            os.path.join(output_dir, "atlas_info.npz"),
            common_labels=common_labels,
            label_remap=json.dumps(label_remap),
            n_parcels=n_parcels,
            atlas_type=atlas_type,
        )
        # Save per-subject atlas for inference
        for s in atlas_masked:
            np.save(os.path.join(output_dir, f"atlas_masked_{s}.npy"), atlas_masked[s])

    # Save provenance metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
        "train_subjects": train_subs,
        "k_global": builder.k_global,
        "n_parcels": n_parcels,
        "n_train_samples": X_concat.shape[0],
        "experiment_mode": experiment_mode,
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
