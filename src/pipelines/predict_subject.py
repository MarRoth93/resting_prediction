"""
Prediction pipeline: zero-shot and few-shot prediction for test subject.
"""

import json
import logging
import os

import numpy as np
import yaml

from src.alignment.shared_space import SharedSpaceBuilder
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.evaluation.metrics import (
    median_correlation,
    noise_ceiling_split_half,
    pattern_correlation,
    two_vs_two_accuracy,
    voxelwise_correlation,
)
from src.models.encoding import SharedSpaceEncoder, fine_tune_encoder

logger = logging.getLogger(__name__)


def predict_zero_shot(
    test_sub: int = 7,
    model_dir: str = "outputs/shared_space",
    data_root: str = "processed_data",
    feature_type: str = "clip",
    output_dir: str = "outputs/predictions",
) -> dict:
    """
    Zero-shot prediction using only REST data from test subject.

    Returns dict with predictions, metrics, and metadata.
    """
    logger.info(f"Zero-shot prediction for subject {test_sub}")

    # Load model
    builder = SharedSpaceBuilder.load(model_dir)
    encoder = SharedSpaceEncoder.load(os.path.join(model_dir, "encoder.npz"))

    # Load test subject data
    test_subj = NSDSubjectData(test_sub, data_root)
    features = NSDFeatures(os.path.join(data_root, "features"))

    # Load atlas
    atlas_path = os.path.join(model_dir, f"atlas_masked_{test_sub}.npy")
    atlas_masked = np.load(atlas_path) if os.path.exists(atlas_path) else None
    atlas_info_path = os.path.join(model_dir, "atlas_info.npz")
    n_parcels = None
    if os.path.exists(atlas_info_path):
        n_parcels = int(np.load(atlas_info_path)["n_parcels"])

    # Align test subject (zero-shot)
    P_new, R_new = builder.align_new_subject_zeroshot(
        rest_runs=test_subj.rest_runs,
        atlas_masked=atlas_masked,
        n_parcels=n_parcels,
    )

    # Predict
    X_test = features.get_features(test_subj.test_stim_idx, feature_type)
    Y_pred = encoder.predict_voxels(X_test, P_new, R_new)
    Y_true = np.array(test_subj.test_fmri, dtype=np.float32)

    # Evaluate
    voxel_corrs = voxelwise_correlation(Y_true, Y_pred)
    metrics = {
        "median_r": float(np.median(voxel_corrs)),
        "mean_r": float(np.mean(voxel_corrs)),
        "median_pattern_r": float(np.median(pattern_correlation(Y_true, Y_pred))),
        "two_vs_two": two_vs_two_accuracy(Y_true, Y_pred),
        "n_voxels": int(Y_true.shape[1]),
        "n_stimuli": int(Y_true.shape[0]),
        "mode": "zero_shot",
    }

    # Noise ceiling (if trial data available)
    if test_subj.test_fmri_trials is not None:
        nc = noise_ceiling_split_half(
            np.array(test_subj.test_fmri_trials, dtype=np.float32),
            test_subj.test_trial_labels,
        )
        metrics["noise_ceiling_median"] = float(np.median(nc))

    logger.info(f"Zero-shot results: {metrics}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"zeroshot_sub{test_sub}_pred.npy"), Y_pred)
    np.save(os.path.join(output_dir, f"zeroshot_sub{test_sub}_corrs.npy"), voxel_corrs)
    with open(os.path.join(output_dir, f"zeroshot_sub{test_sub}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"Y_pred": Y_pred, "metrics": metrics, "voxel_corrs": voxel_corrs}


def predict_few_shot(
    test_sub: int = 7,
    n_shots: int = 100,
    model_dir: str = "outputs/shared_space",
    data_root: str = "processed_data",
    feature_type: str = "clip",
    fine_tune: bool = False,
    seed: int = 42,
    output_dir: str = "outputs/predictions",
) -> dict:
    """
    Few-shot prediction using N shared-stimuli responses from test subject.

    Args:
        seed: random seed for shot/eval split
    """
    logger.info(f"Few-shot prediction: sub={test_sub}, n_shots={n_shots}, seed={seed}")

    # Load model
    builder = SharedSpaceBuilder.load(model_dir)
    encoder = SharedSpaceEncoder.load(os.path.join(model_dir, "encoder.npz"))

    # Load test subject
    test_subj = NSDSubjectData(test_sub, data_root)
    features = NSDFeatures(os.path.join(data_root, "features"))

    # Load atlas
    atlas_path = os.path.join(model_dir, f"atlas_masked_{test_sub}.npy")
    atlas_masked = np.load(atlas_path) if os.path.exists(atlas_path) else None
    atlas_info_path = os.path.join(model_dir, "atlas_info.npz")
    n_parcels = None
    if os.path.exists(atlas_info_path):
        n_parcels = int(np.load(atlas_info_path)["n_parcels"])

    # Random split: n_shots for alignment, rest for evaluation
    rng = np.random.RandomState(seed)
    n_shared = len(test_subj.test_stim_idx)
    min_eval = 50  # minimum stimuli reserved for evaluation
    max_shots = n_shared - min_eval
    if max_shots < 1:
        raise ValueError(
            f"Not enough shared stimuli for few-shot: n_shared={n_shared}, "
            f"need at least {min_eval + 1} (min_eval={min_eval} + 1 shot)"
        )
    actual_shots = min(n_shots, max_shots)
    shot_indices = rng.choice(n_shared, size=actual_shots, replace=False)
    eval_indices = np.setdiff1d(np.arange(n_shared), shot_indices)

    shared_fmri = np.array(test_subj.test_fmri, dtype=np.float32)[shot_indices]

    # Align (few-shot) — pass shot_indices for correct template row matching
    P_new, R_new = builder.align_new_subject_fewshot(
        rest_runs=test_subj.rest_runs,
        task_fmri_shared=shared_fmri,
        shot_indices=shot_indices,
        atlas_masked=atlas_masked,
        n_parcels=n_parcels,
    )

    # Optionally fine-tune encoder
    enc = encoder
    if fine_tune:
        shot_stim_idx = test_subj.test_stim_idx[shot_indices]
        X_shared = features.get_features(shot_stim_idx, feature_type)
        Z_shared = shared_fmri @ P_new @ R_new
        enc = fine_tune_encoder(encoder, X_shared, Z_shared)

    # Predict on held-out
    eval_stim_idx = test_subj.test_stim_idx[eval_indices]
    X_test = features.get_features(eval_stim_idx, feature_type)
    Y_pred = enc.predict_voxels(X_test, P_new, R_new)
    Y_true = np.array(test_subj.test_fmri, dtype=np.float32)[eval_indices]

    # Evaluate
    voxel_corrs = voxelwise_correlation(Y_true, Y_pred)
    metrics = {
        "median_r": float(np.median(voxel_corrs)),
        "mean_r": float(np.mean(voxel_corrs)),
        "median_pattern_r": float(np.median(pattern_correlation(Y_true, Y_pred))),
        "two_vs_two": two_vs_two_accuracy(Y_true, Y_pred),
        "n_shots": int(actual_shots),
        "n_eval": int(len(eval_indices)),
        "seed": seed,
        "fine_tune": fine_tune,
        "mode": "few_shot",
    }

    logger.info(f"Few-shot N={n_shots}: {metrics}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tag = f"fewshot_sub{test_sub}_N{n_shots}_seed{seed}"
    np.save(os.path.join(output_dir, f"{tag}_pred.npy"), Y_pred)
    with open(os.path.join(output_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"Y_pred": Y_pred, "metrics": metrics, "voxel_corrs": voxel_corrs}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Predict for test subject")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], required=True)
    parser.add_argument("--test-sub", type=int, default=7)
    parser.add_argument("--n-shots", type=int, default=100)
    parser.add_argument("--model-dir", default="outputs/shared_space")
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--feature-type", default="clip")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()

    if args.mode == "zero_shot":
        predict_zero_shot(args.test_sub, args.model_dir, args.data_root,
                          args.feature_type, args.output_dir)
    else:
        predict_few_shot(args.test_sub, args.n_shots, args.model_dir, args.data_root,
                         args.feature_type, args.fine_tune, args.seed, args.output_dir)
