"""
Prediction pipeline: zero-shot and few-shot prediction for test subject.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.data.shared_paths import default_raw_data_root
from src.evaluation.metrics import (
    noise_ceiling_split_half,
    pattern_correlation,
    two_vs_two_accuracy,
    voxelwise_correlation,
)
from src.models.encoding_factory import load_encoder
from src.pipelines.eval_split import fixed_eval_indices

logger = logging.getLogger(__name__)


def _validate_subject_row_contract(subject: NSDSubjectData) -> None:
    """Fail fast on row/voxel mismatches within one subject artifact bundle."""
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

    v_task = int(subject.test_fmri.shape[1])
    if subject.rest_runs:
        rest_voxels = {int(run.shape[1]) for run in subject.rest_runs}
        if len(rest_voxels) != 1 or next(iter(rest_voxels)) != v_task:
            raise ValueError(
                f"Subject {subject.sub}: REST voxel dimensions {sorted(rest_voxels)} "
                f"do not match task voxels {v_task}."
            )

    if subject.test_fmri_trials is not None or subject.test_trial_labels is not None:
        if subject.test_fmri_trials is None or subject.test_trial_labels is None:
            raise ValueError(
                f"Subject {subject.sub}: trial-level fMRI and labels must both be present."
            )
        if int(subject.test_fmri_trials.shape[0]) != int(subject.test_trial_labels.shape[0]):
            raise ValueError(
                f"Subject {subject.sub}: trial rows mismatch test_fmri_trials="
                f"{int(subject.test_fmri_trials.shape[0])}, test_trial_labels="
                f"{int(subject.test_trial_labels.shape[0])}."
            )
        if int(subject.test_fmri_trials.shape[1]) != v_task:
            raise ValueError(
                f"Subject {subject.sub}: test_fmri_trials voxels="
                f"{int(subject.test_fmri_trials.shape[1])} do not match task voxels {v_task}."
            )


def _validate_feature_indices(
    subject: NSDSubjectData,
    features: NSDFeatures,
    feature_type: str,
) -> None:
    all_idx = np.concatenate([subject.train_stim_idx, subject.test_stim_idx]).astype(np.int64)
    if all_idx.size == 0:
        raise ValueError(f"Subject {subject.sub}: empty stimulus indices.")
    if np.any(all_idx < 0):
        raise ValueError(f"Subject {subject.sub}: negative stimulus indices detected.")
    max_idx = int(all_idx.max())
    try:
        probe = features.get_features(np.array([max_idx], dtype=np.int64), feature_type)
    except Exception as exc:
        raise ValueError(
            f"Subject {subject.sub}: stimulus index {max_idx} invalid for feature_type={feature_type}."
        ) from exc
    if int(probe.shape[0]) != 1:
        raise ValueError(
            f"Subject {subject.sub}: feature probe returned unexpected shape {probe.shape}."
        )


def _parse_reliability_thresholds(
    thresholds: list[float] | tuple[float, ...] | None,
) -> list[float]:
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.3]
    clean = sorted({float(t) for t in thresholds})
    if clean and clean[0] < 0:
        raise ValueError(f"Reliability thresholds must be >= 0, got {clean}.")
    return clean


def _threshold_key(threshold: float) -> str:
    token = f"{threshold:.3f}".rstrip("0").rstrip(".")
    if token == "":
        token = "0"
    return token.replace("-", "m").replace(".", "_")


def _compute_noise_ceiling_and_reliability_metrics(
    subject: NSDSubjectData,
    voxel_corrs: np.ndarray,
    reliability_thresholds: list[float],
) -> dict:
    """
    Compute split-half noise ceiling and reliability-stratified summary metrics.
    """
    if subject.test_fmri_trials is None:
        return {}

    nc = noise_ceiling_split_half(
        np.array(subject.test_fmri_trials, dtype=np.float32),
        subject.test_trial_labels,
    )
    out: dict[str, float | int] = {
        "noise_ceiling_median": float(np.median(nc)),
        "noise_ceiling_mean": float(np.mean(nc)),
    }
    for thr in reliability_thresholds:
        mask = nc >= thr
        key = _threshold_key(thr)
        out[f"n_voxels_nc_ge_{key}"] = int(mask.sum())
        if not np.any(mask):
            continue
        out[f"median_r_nc_ge_{key}"] = float(np.median(voxel_corrs[mask]))
        out[f"mean_r_nc_ge_{key}"] = float(np.mean(voxel_corrs[mask]))
    return out


def _load_external_seed_runs(
    test_subj: NSDSubjectData,
    test_sub: int,
    model_dir: str,
    data_root: str,
    raw_data_root: str,
) -> list[np.ndarray]:
    """Load/prepare prediction-time seed runs for external_seed_bank models."""
    from src.alignment.external_seed_bank import (
        load_external_seed_info,
        load_or_prepare_external_seed_runs,
    )

    seed_set, seed_defs, rest_cfg = load_external_seed_info(model_dir)
    seed_runs = load_or_prepare_external_seed_runs(
        sub=test_sub,
        data_root=data_root,
        raw_data_root=raw_data_root,
        pred_mask=test_subj.mask,
        seed_defs=seed_defs,
        rest_cfg=rest_cfg,
        seed_set=seed_set,
        reference_rest_runs=test_subj.rest_runs,
    )
    logger.info(
        "Subject %d external seed bank: seed_set=%s, runs=%d, seeds=%d",
        test_sub,
        seed_set,
        len(seed_runs),
        len(seed_defs),
    )
    return seed_runs


def _resolve_fewshot_support_candidates(
    test_stim_idx: np.ndarray,
    model_dir: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Resolve few-shot support rows and corresponding template rows.

    Returns:
        support_rows: row indices into test_subj.test_fmri/test_stim_idx.
        template_rows: matching row indices into builder.template_Z.
        strategy: string tag for metrics/debugging.
    """
    test_stim_idx = np.asarray(test_stim_idx, dtype=np.int64).ravel()
    if test_stim_idx.size == 0:
        raise ValueError("test_stim_idx is empty.")
    unique_test_idx = np.unique(test_stim_idx)
    if unique_test_idx.size != test_stim_idx.size:
        raise ValueError("test_stim_idx contains duplicates; expected unique canonical stimulus rows.")
    if not np.array_equal(unique_test_idx, test_stim_idx):
        raise ValueError("test_stim_idx must be sorted ascending for deterministic row mapping.")

    shared_idx_path = os.path.join(model_dir, "shared_stim_idx.npy")
    if not os.path.exists(shared_idx_path):
        raise FileNotFoundError(f"Missing required shared stimulus artifact: {shared_idx_path}")

    shared_stim_idx = np.asarray(np.load(shared_idx_path), dtype=np.int64).ravel()
    if shared_stim_idx.size == 0:
        raise ValueError(f"Shared stimulus artifact is empty: {shared_idx_path}")
    unique_shared = np.unique(shared_stim_idx)
    if unique_shared.size != shared_stim_idx.size:
        raise ValueError(f"Shared stimulus artifact contains duplicate IDs: {shared_idx_path}")
    if not np.array_equal(unique_shared, shared_stim_idx):
        logger.warning(
            "shared_stim_idx.npy is not sorted; canonicalizing sorted unique order."
        )
        shared_stim_idx = unique_shared

    rows = np.searchsorted(test_stim_idx, shared_stim_idx)
    valid = (rows < test_stim_idx.size) & (test_stim_idx[rows] == shared_stim_idx)
    if not np.any(valid):
        raise ValueError(
            f"Subject has zero overlap with training shared-stimulus artifact {shared_idx_path}."
        )

    support_rows = rows[valid].astype(np.int64, copy=False)
    template_rows = np.nonzero(valid)[0].astype(np.int64, copy=False)
    missing_count = int(shared_stim_idx.size - support_rows.size)
    if missing_count > 0:
        logger.warning(
            "Test subject is missing %d/%d training shared stimuli; few-shot support "
            "will use the %d overlapping rows.",
            missing_count,
            int(shared_stim_idx.size),
            int(support_rows.size),
        )
    return support_rows, template_rows, "shared_stim_intersection"


def _sample_support_shots(
    support_rows: np.ndarray,
    template_rows: np.ndarray,
    n_shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample few-shot rows from a precomputed support pool."""
    if int(n_shots) < 1:
        raise ValueError(f"n_shots must be >=1, got {n_shots}.")
    if support_rows.size != template_rows.size:
        raise ValueError(
            "support_rows and template_rows must have identical lengths for row correspondence."
        )
    if support_rows.size == 0:
        raise ValueError("Few-shot support pool is empty.")

    actual_shots = int(min(int(n_shots), int(support_rows.size)))
    rng = np.random.RandomState(seed)
    pick = np.sort(rng.choice(support_rows.size, size=actual_shots, replace=False).astype(np.int64))
    shot_rows = support_rows[pick].astype(np.int64, copy=False)
    shot_template_rows = template_rows[pick].astype(np.int64, copy=False)
    return shot_rows, shot_template_rows, actual_shots


def predict_zero_shot(
    test_sub: int = 7,
    config_path: str = "config.yaml",
    model_dir: str = "artifacts/model",
    data_root: str = "data/processed",
    raw_data_root: str = default_raw_data_root(),
    output_dir: str = "artifacts/predictions",
) -> dict:
    """
    Zero-shot prediction using only REST data from test subject.

    Returns dict with predictions, metrics, and metadata.
    """
    logger.info(f"Zero-shot prediction for subject {test_sub}")
    config = load_config(config_path)
    feature_type = str(config["features"]["type"])
    evaluation_cfg = config["evaluation"]
    fixed_eval_size = int(evaluation_cfg["fixed_eval_size"])
    eval_split_seed = int(evaluation_cfg["eval_split_seed"])
    reliability_thresholds = evaluation_cfg["reliability_thresholds"]

    # Load model
    builder = SharedSpaceBuilder.load(model_dir)
    encoder = load_encoder(model_dir)

    # Load test subject data
    test_subj = NSDSubjectData(test_sub, data_root)
    _validate_subject_row_contract(test_subj)
    features = NSDFeatures(os.path.join(data_root, "features"))
    _validate_feature_indices(test_subj, features, feature_type)
    reliability_thresholds = _parse_reliability_thresholds(reliability_thresholds)

    external_seed_runs = _load_external_seed_runs(
        test_subj=test_subj,
        test_sub=test_sub,
        model_dir=model_dir,
        data_root=data_root,
        raw_data_root=raw_data_root,
    )

    # Align test subject (zero-shot)
    P_new, R_new = builder.align_new_subject_zeroshot(
        rest_runs=test_subj.rest_runs,
        external_seed_runs=external_seed_runs,
    )

    # Predict
    n_shared = int(len(test_subj.test_stim_idx))
    eval_indices = fixed_eval_indices(
        n_shared=n_shared,
        eval_size=fixed_eval_size,
        seed=eval_split_seed,
    )

    X_test = features.get_features(test_subj.test_stim_idx, feature_type)
    Y_pred_full = encoder.predict_voxels(X_test, P_new, R_new)
    Y_true_full = np.array(test_subj.test_fmri, dtype=np.float32)
    Y_pred = Y_pred_full[eval_indices]
    Y_true = Y_true_full[eval_indices]

    # Evaluate
    voxel_corrs = voxelwise_correlation(Y_true, Y_pred)
    metrics = {
        "median_r": float(np.median(voxel_corrs)),
        "mean_r": float(np.mean(voxel_corrs)),
        "median_pattern_r": float(np.median(pattern_correlation(Y_true, Y_pred))),
        "two_vs_two": two_vs_two_accuracy(Y_true, Y_pred),
        "n_voxels": int(Y_true.shape[1]),
        "n_stimuli": int(Y_true.shape[0]),
        "n_stimuli_total": int(n_shared),
        "n_eval": int(len(eval_indices)),
        "eval_split_mode": "fixed",
        "eval_indices": eval_indices.astype(np.int64).tolist(),
        "mode": "zero_shot",
        "encoder_type": str(encoder.encoder_type),
    }
    metrics["eval_split_seed"] = eval_split_seed
    metrics["eval_split_size"] = fixed_eval_size
    metrics["eval_split_source"] = "config"

    metrics.update(
        _compute_noise_ceiling_and_reliability_metrics(
            subject=test_subj,
            voxel_corrs=voxel_corrs,
            reliability_thresholds=reliability_thresholds,
        )
    )

    logger.info(f"Zero-shot results: {metrics}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"zeroshot_sub{test_sub}_pred.npy"), Y_pred_full)
    np.save(os.path.join(output_dir, f"zeroshot_sub{test_sub}_corrs.npy"), voxel_corrs)
    with open(os.path.join(output_dir, f"zeroshot_sub{test_sub}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"Y_pred": Y_pred_full, "metrics": metrics, "voxel_corrs": voxel_corrs}


def predict_few_shot(
    test_sub: int = 7,
    n_shots: int = 100,
    config_path: str = "config.yaml",
    model_dir: str = "artifacts/model",
    data_root: str = "data/processed",
    raw_data_root: str = default_raw_data_root(),
    seed: int = 42,
    output_dir: str = "artifacts/predictions",
) -> dict:
    """
    Few-shot prediction using N shared-stimuli responses from test subject.

    Args:
        seed: random seed for shot/eval split
    """
    logger.info(f"Few-shot prediction: sub={test_sub}, n_shots={n_shots}, seed={seed}")
    config = load_config(config_path)
    feature_type = str(config["features"]["type"])
    evaluation_cfg = config["evaluation"]
    fixed_eval_size = int(evaluation_cfg["fixed_eval_size"])
    eval_split_seed = int(evaluation_cfg["eval_split_seed"])
    reliability_thresholds = evaluation_cfg["reliability_thresholds"]

    # Load model
    builder = SharedSpaceBuilder.load(model_dir)
    encoder = load_encoder(model_dir)

    # Load test subject
    test_subj = NSDSubjectData(test_sub, data_root)
    _validate_subject_row_contract(test_subj)
    features = NSDFeatures(os.path.join(data_root, "features"))
    _validate_feature_indices(test_subj, features, feature_type)
    reliability_thresholds = _parse_reliability_thresholds(reliability_thresholds)

    external_seed_runs = _load_external_seed_runs(
        test_subj=test_subj,
        test_sub=test_sub,
        model_dir=model_dir,
        data_root=data_root,
        raw_data_root=raw_data_root,
    )

    # Split: support shots from train pool, evaluation from fixed held-out rows
    n_shared = int(len(test_subj.test_stim_idx))
    support_candidates, support_template_candidates, support_strategy = (
        _resolve_fewshot_support_candidates(
            test_stim_idx=test_subj.test_stim_idx,
            model_dir=model_dir,
        )
    )
    logger.info(
        "Few-shot support pool (%s): %d candidate rows before eval exclusion.",
        support_strategy,
        int(support_candidates.size),
    )

    eval_indices = fixed_eval_indices(
        n_shared=n_shared,
        eval_size=fixed_eval_size,
        seed=eval_split_seed,
    )
    support_mask = ~np.isin(support_candidates, eval_indices, assume_unique=False)
    support_pool_rows = support_candidates[support_mask]
    support_pool_template_rows = support_template_candidates[support_mask]
    if support_pool_rows.size == 0:
        raise ValueError(
            "No few-shot rows remain in the shared-support pool after applying fixed eval split."
        )
    shot_indices, template_shot_indices, actual_shots = _sample_support_shots(
        support_rows=support_pool_rows,
        template_rows=support_pool_template_rows,
        n_shots=n_shots,
        seed=seed,
    )

    shared_fmri = np.array(test_subj.test_fmri, dtype=np.float32)[shot_indices]

    # Align (few-shot) against template rows corresponding to the selected support stimuli.
    P_new, R_new = builder.align_new_subject_fewshot(
        rest_runs=test_subj.rest_runs,
        task_fmri_shared=shared_fmri,
        shot_indices=template_shot_indices,
        external_seed_runs=external_seed_runs,
    )

    # Predict on held-out
    eval_stim_idx = test_subj.test_stim_idx[eval_indices]
    X_test = features.get_features(eval_stim_idx, feature_type)
    Y_pred = encoder.predict_voxels(X_test, P_new, R_new)
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
        "n_stimuli_total": int(n_shared),
        "seed": seed,
        "eval_split_mode": "fixed",
        "eval_indices": eval_indices.astype(np.int64).tolist(),
        "shot_indices": shot_indices.astype(np.int64).tolist(),
        "shot_template_indices": template_shot_indices.astype(np.int64).tolist(),
        "support_strategy": support_strategy,
        "n_support_candidates": int(support_candidates.size),
        "n_support_pool": int(support_pool_rows.size),
        "mode": "few_shot",
        "encoder_type": str(encoder.encoder_type),
    }
    metrics["eval_split_seed"] = eval_split_seed
    metrics["eval_split_size"] = fixed_eval_size
    metrics["eval_split_source"] = "config"
    metrics.update(
        _compute_noise_ceiling_and_reliability_metrics(
            subject=test_subj,
            voxel_corrs=voxel_corrs,
            reliability_thresholds=reliability_thresholds,
        )
    )

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
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--raw-data-root", default=default_raw_data_root())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/predictions")
    args = parser.parse_args()

    if args.mode == "zero_shot":
        predict_zero_shot(
            test_sub=args.test_sub,
            config_path=args.config,
            model_dir=args.model_dir,
            data_root=args.data_root,
            raw_data_root=args.raw_data_root,
            output_dir=args.output_dir,
        )
    else:
        predict_few_shot(
            test_sub=args.test_sub,
            n_shots=args.n_shots,
            config_path=args.config,
            model_dir=args.model_dir,
            data_root=args.data_root,
            raw_data_root=args.raw_data_root,
            seed=args.seed,
            output_dir=args.output_dir,
        )
