"""
Prediction pipeline: zero-shot and few-shot prediction for test subject.
"""

import json
import logging
import os

import numpy as np

from src.alignment.shared_space import SharedSpaceBuilder
from src.data.load_atlas import atlas_utilization_summary
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.evaluation.metrics import (
    noise_ceiling_split_half,
    pattern_correlation,
    two_vs_two_accuracy,
    voxelwise_correlation,
)
from src.models.encoding import SharedSpaceEncoder, fine_tune_encoder
from src.pipelines.eval_split import (
    load_or_create_fixed_eval_split,
)

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


def _load_parcellation_artifacts(
    builder: SharedSpaceBuilder,
    test_subj: NSDSubjectData,
    test_sub: int,
    model_dir: str,
) -> tuple[np.ndarray | None, int | None, dict | None]:
    """
    Load and validate atlas artifacts when parcellation connectivity is used.
    """
    if builder.connectivity_mode != "parcellation":
        return None, None, None

    atlas_path = os.path.join(model_dir, f"atlas_masked_{test_sub}.npy")
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(
            f"Missing atlas artifact for subject {test_sub}: {atlas_path}. "
            "Re-run shared-space training with parcellation mode."
        )
    atlas_masked = np.load(atlas_path)

    atlas_info_path = os.path.join(model_dir, "atlas_info.npz")
    if not os.path.exists(atlas_info_path):
        raise FileNotFoundError(
            f"Missing atlas_info.npz in {model_dir}. "
            "Re-run shared-space training to save harmonized atlas metadata."
        )
    n_parcels = int(np.load(atlas_info_path)["n_parcels"])

    expected_v = int(test_subj.test_fmri.shape[1])
    if atlas_masked.ndim != 1:
        raise ValueError(
            f"atlas_masked for subject {test_sub} must be 1D, got shape {atlas_masked.shape}."
        )
    if int(atlas_masked.shape[0]) != expected_v:
        raise ValueError(
            f"Atlas-mask mismatch for subject {test_sub}: "
            f"atlas length={atlas_masked.shape[0]}, test_fmri voxels={expected_v}."
        )
    if np.any(atlas_masked < 0) or np.any(atlas_masked > n_parcels):
        raise ValueError(
            f"Subject {test_sub}: atlas labels must be in [0, {n_parcels}] after harmonization."
        )

    atlas_summary = atlas_utilization_summary(
        atlas_masked=atlas_masked,
        n_parcels=n_parcels,
        sub_id=test_sub,
        min_labeled_fraction_warn=0.0,  # log structure warnings only at inference
    )
    logger.info(
        "Subject %d atlas utilization: labeled_fraction=%.3f, parcels_present=%d/%d",
        test_sub,
        atlas_summary["labeled_fraction"],
        atlas_summary["n_parcels_present"],
        n_parcels,
    )
    for warning in atlas_summary["warnings"]:
        logger.warning(warning)

    return atlas_masked, n_parcels, atlas_summary


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
        logger.warning(
            "Missing shared_stim_idx.npy in %s; falling back to legacy few-shot row mapping.",
            model_dir,
        )
        legacy_rows = np.arange(test_stim_idx.shape[0], dtype=np.int64)
        return legacy_rows, legacy_rows.copy(), "legacy_all_rows"

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
    model_dir: str = "outputs/shared_space",
    data_root: str = "processed_data",
    feature_type: str = "clip",
    output_dir: str = "outputs/predictions",
    use_fixed_eval_split: bool = True,
    fixed_eval_size: int = 250,
    eval_split_seed: int = 42,
    reliability_thresholds: list[float] | tuple[float, ...] | None = None,
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
    _validate_subject_row_contract(test_subj)
    features = NSDFeatures(os.path.join(data_root, "features"))
    _validate_feature_indices(test_subj, features, feature_type)
    reliability_thresholds = _parse_reliability_thresholds(reliability_thresholds)

    # Load/validate atlas artifacts for parcellation mode
    atlas_masked, n_parcels, atlas_summary = _load_parcellation_artifacts(
        builder=builder,
        test_subj=test_subj,
        test_sub=test_sub,
        model_dir=model_dir,
    )

    # Align test subject (zero-shot)
    P_new, R_new = builder.align_new_subject_zeroshot(
        rest_runs=test_subj.rest_runs,
        atlas_masked=atlas_masked,
        n_parcels=n_parcels,
    )

    # Predict
    n_shared = int(len(test_subj.test_stim_idx))
    eval_split_meta = None
    eval_split_path = None
    if use_fixed_eval_split:
        eval_indices, eval_split_meta, eval_split_path = load_or_create_fixed_eval_split(
            model_dir=model_dir,
            test_sub=test_sub,
            n_shared=n_shared,
            eval_size=fixed_eval_size,
            seed=eval_split_seed,
        )
    else:
        eval_indices = np.arange(n_shared, dtype=np.int64)

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
        "eval_split_mode": "fixed" if use_fixed_eval_split else "all_rows",
        "eval_indices": eval_indices.astype(np.int64).tolist(),
        "mode": "zero_shot",
    }
    if eval_split_meta is not None:
        metrics["eval_split_seed"] = int(eval_split_meta["seed"])
        metrics["eval_split_size"] = int(eval_split_meta["eval_size"])
        metrics["eval_split_source"] = str(eval_split_meta["source"])
    if eval_split_path is not None:
        metrics["eval_split_path"] = eval_split_path
    if atlas_summary is not None:
        metrics.update({
            "atlas_labeled_fraction": float(atlas_summary["labeled_fraction"]),
            "atlas_n_parcels_present": int(atlas_summary["n_parcels_present"]),
            "atlas_n_parcels_expected": int(atlas_summary["n_parcels_expected"]),
        })

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
    model_dir: str = "outputs/shared_space",
    data_root: str = "processed_data",
    feature_type: str = "clip",
    fine_tune: bool = False,
    seed: int = 42,
    output_dir: str = "outputs/predictions",
    use_fixed_eval_split: bool = True,
    fixed_eval_size: int = 250,
    eval_split_seed: int = 42,
    reliability_thresholds: list[float] | tuple[float, ...] | None = None,
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
    _validate_subject_row_contract(test_subj)
    features = NSDFeatures(os.path.join(data_root, "features"))
    _validate_feature_indices(test_subj, features, feature_type)
    reliability_thresholds = _parse_reliability_thresholds(reliability_thresholds)

    # Load/validate atlas artifacts for parcellation mode
    atlas_masked, n_parcels, atlas_summary = _load_parcellation_artifacts(
        builder=builder,
        test_subj=test_subj,
        test_sub=test_sub,
        model_dir=model_dir,
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

    eval_split_meta = None
    eval_split_path = None
    if use_fixed_eval_split:
        eval_indices, eval_split_meta, eval_split_path = load_or_create_fixed_eval_split(
            model_dir=model_dir,
            test_sub=test_sub,
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
    else:
        min_eval = 50  # minimum stimuli reserved for evaluation
        max_shots = n_shared - min_eval
        if max_shots < 1:
            raise ValueError(
                f"Not enough shared stimuli for few-shot: n_shared={n_shared}, "
                f"need at least {min_eval + 1} (min_eval={min_eval} + 1 shot)"
            )
        support_pool_rows = support_candidates
        support_pool_template_rows = support_template_candidates
        if support_pool_rows.size == 0:
            raise ValueError("Few-shot support pool is empty.")
        max_shots = int(min(int(max_shots), int(support_pool_rows.size)))
        if max_shots < 1:
            raise ValueError(
                f"Not enough support rows for few-shot after constraints: "
                f"support_pool={int(support_pool_rows.size)}, max_shots={max_shots}."
            )
        requested_shots = int(min(int(n_shots), max_shots))
        shot_indices, template_shot_indices, actual_shots = _sample_support_shots(
            support_rows=support_pool_rows,
            template_rows=support_pool_template_rows,
            n_shots=requested_shots,
            seed=seed,
        )
        eval_indices = np.setdiff1d(np.arange(n_shared), shot_indices, assume_unique=False)

    shared_fmri = np.array(test_subj.test_fmri, dtype=np.float32)[shot_indices]

    # Align (few-shot) against template rows corresponding to the selected support stimuli.
    P_new, R_new = builder.align_new_subject_fewshot(
        rest_runs=test_subj.rest_runs,
        task_fmri_shared=shared_fmri,
        shot_indices=template_shot_indices,
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
        "n_stimuli_total": int(n_shared),
        "seed": seed,
        "eval_split_mode": "fixed" if use_fixed_eval_split else "complement_random",
        "eval_indices": eval_indices.astype(np.int64).tolist(),
        "shot_indices": shot_indices.astype(np.int64).tolist(),
        "shot_template_indices": template_shot_indices.astype(np.int64).tolist(),
        "support_strategy": support_strategy,
        "n_support_candidates": int(support_candidates.size),
        "n_support_pool": int(support_pool_rows.size),
        "fine_tune": fine_tune,
        "mode": "few_shot",
    }
    if eval_split_meta is not None:
        metrics["eval_split_seed"] = int(eval_split_meta["seed"])
        metrics["eval_split_size"] = int(eval_split_meta["eval_size"])
        metrics["eval_split_source"] = str(eval_split_meta["source"])
    if eval_split_path is not None:
        metrics["eval_split_path"] = eval_split_path
    if atlas_summary is not None:
        metrics.update({
            "atlas_labeled_fraction": float(atlas_summary["labeled_fraction"]),
            "atlas_n_parcels_present": int(atlas_summary["n_parcels_present"]),
            "atlas_n_parcels_expected": int(atlas_summary["n_parcels_expected"]),
        })
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
    parser.add_argument("--model-dir", default="outputs/shared_space")
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--feature-type", default="clip")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split-seed", type=int, default=42)
    parser.add_argument("--fixed-eval-size", type=int, default=250)
    parser.add_argument("--disable-fixed-eval-split", action="store_true")
    parser.add_argument(
        "--reliability-thresholds",
        default="0.0,0.1,0.3",
        help="Comma-separated NC thresholds for stratified metrics.",
    )
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()
    reliability_thresholds = [
        float(x.strip()) for x in args.reliability_thresholds.split(",") if x.strip() != ""
    ]
    use_fixed_eval_split = not args.disable_fixed_eval_split

    if args.mode == "zero_shot":
        predict_zero_shot(
            test_sub=args.test_sub,
            model_dir=args.model_dir,
            data_root=args.data_root,
            feature_type=args.feature_type,
            output_dir=args.output_dir,
            use_fixed_eval_split=use_fixed_eval_split,
            fixed_eval_size=args.fixed_eval_size,
            eval_split_seed=args.eval_split_seed,
            reliability_thresholds=reliability_thresholds,
        )
    else:
        predict_few_shot(
            test_sub=args.test_sub,
            n_shots=args.n_shots,
            model_dir=args.model_dir,
            data_root=args.data_root,
            feature_type=args.feature_type,
            fine_tune=args.fine_tune,
            seed=args.seed,
            output_dir=args.output_dir,
            use_fixed_eval_split=use_fixed_eval_split,
            fixed_eval_size=args.fixed_eval_size,
            eval_split_seed=args.eval_split_seed,
            reliability_thresholds=reliability_thresholds,
        )
