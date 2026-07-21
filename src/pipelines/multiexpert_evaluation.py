"""Evaluation summaries shared by multi-expert prediction and LOSO."""

from __future__ import annotations

import numpy as np

from src.data.nsd_loader import NSDSubjectData
from src.evaluation.metrics import pattern_correlation, two_vs_two_accuracy, voxelwise_correlation


def validate_repeated_trial_contract(
    trial_fmri: np.ndarray,
    trial_labels: np.ndarray,
    *,
    expected_voxels: int | None = None,
) -> tuple[np.ndarray, int]:
    trial_fmri = np.asarray(trial_fmri)
    trial_labels = np.asarray(trial_labels).ravel()
    if trial_fmri.ndim != 2 or trial_labels.shape[0] != trial_fmri.shape[0]:
        raise ValueError("Trial fMRI and labels must have matching rows.")
    if expected_voxels is not None and int(trial_fmri.shape[1]) != int(expected_voxels):
        raise ValueError("Trial-level and averaged fMRI voxel counts differ.")
    labels, counts = np.unique(trial_labels, return_counts=True)
    if labels.size < 10:
        raise ValueError("At least ten repeated stimuli are required for reliability.")
    repeats = int(counts.max())
    if repeats < 2:
        raise ValueError("At least two repeats per stimulus are required for reliability.")
    retained_labels = labels[counts == repeats]
    if retained_labels.size < 10:
        raise ValueError(
            "At least ten stimuli with the maximum repeat count are required for reliability."
        )
    return retained_labels, repeats


def balanced_repeated_noise_ceiling(
    trial_fmri: np.ndarray,
    trial_labels: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Reliability of the averaged response from balanced repeated trials.

    Every pair of repetitions is correlated across stimuli, their correlations
    are averaged in Fisher-z space, and Spearman-Brown then estimates the
    reliability of the mean of all repetitions. Unlike an odd/even split, this
    treats NSD's three repeats symmetrically.
    """
    trial_fmri = np.asarray(trial_fmri, dtype=np.float32)
    trial_labels = np.asarray(trial_labels).ravel()
    labels, repeats = validate_repeated_trial_contract(trial_fmri, trial_labels)
    by_repeat = [
        np.stack(
            [trial_fmri[trial_labels == label][repeat] for label in labels],
            axis=0,
        )
        for repeat in range(repeats)
    ]
    pairwise = []
    for first in range(repeats):
        for second in range(first + 1, repeats):
            pairwise.append(
                voxelwise_correlation(by_repeat[first], by_repeat[second])
            )
    pairwise_array = np.stack(pairwise, axis=0)
    fisher = np.arctanh(np.clip(pairwise_array, -0.999999, 0.999999))
    single_trial_reliability = np.tanh(fisher.mean(axis=0))
    single_trial_reliability = np.clip(single_trial_reliability, 0.0, 0.999999)
    averaged = (
        repeats * single_trial_reliability
        / (1.0 + (repeats - 1) * single_trial_reliability)
    )
    return (
        np.clip(averaged, 0.0, 1.0).astype(np.float32),
        repeats,
        int(labels.size),
    )


def threshold_key(threshold: float) -> str:
    token = f"{float(threshold):.3f}".rstrip("0").rstrip(".") or "0"
    return token.replace("-", "m").replace(".", "_")


def compute_subject_noise_ceiling(
    subject: NSDSubjectData,
) -> tuple[np.ndarray, int, int] | None:
    if subject.test_fmri_trials is None or subject.test_trial_labels is None:
        return None
    return balanced_repeated_noise_ceiling(
        np.asarray(subject.test_fmri_trials, dtype=np.float32),
        subject.test_trial_labels,
    )


def evaluate_voxel_prediction(
    subject: NSDSubjectData,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    reliability_thresholds: list[float] | tuple[float, ...],
    precomputed_noise_ceiling: tuple[np.ndarray, int, int] | None = None,
) -> tuple[dict, np.ndarray]:
    truth = np.asarray(truth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError(
            f"Truth and prediction must be matching 2D arrays, got "
            f"{truth.shape} and {prediction.shape}."
        )
    voxel_corrs = voxelwise_correlation(truth, prediction)
    metrics = {
        "median_r": float(np.median(voxel_corrs)),
        "mean_r": float(np.mean(voxel_corrs)),
        "median_pattern_r": float(np.median(pattern_correlation(truth, prediction))),
        "two_vs_two": (
            float(two_vs_two_accuracy(truth, prediction))
            if truth.shape[0] >= 2
            else 0.5
        ),
        "n_eval": int(truth.shape[0]),
        "n_voxels": int(truth.shape[1]),
    }
    if subject.test_fmri_trials is not None and subject.test_trial_labels is not None:
        if precomputed_noise_ceiling is None:
            ceiling, repeats, reliability_stimuli = compute_subject_noise_ceiling(subject)
        else:
            ceiling, repeats, reliability_stimuli = precomputed_noise_ceiling
        if ceiling.shape != voxel_corrs.shape:
            raise ValueError("Noise-ceiling and prediction voxel dimensions differ.")
        metrics["noise_ceiling_median"] = float(np.median(ceiling))
        metrics["noise_ceiling_mean"] = float(np.mean(ceiling))
        metrics["noise_ceiling_method"] = "balanced_pairwise_spearman_brown"
        metrics["noise_ceiling_repeats"] = int(repeats)
        metrics["noise_ceiling_stimuli"] = int(reliability_stimuli)
        for threshold in sorted({float(value) for value in reliability_thresholds}):
            if threshold < 0:
                raise ValueError("Reliability thresholds must be non-negative.")
            keep = ceiling >= threshold
            key = threshold_key(threshold)
            metrics[f"n_voxels_nc_ge_{key}"] = int(keep.sum())
            if np.any(keep):
                metrics[f"median_r_nc_ge_{key}"] = float(np.median(voxel_corrs[keep]))
                metrics[f"mean_r_nc_ge_{key}"] = float(np.mean(voxel_corrs[keep]))
    return metrics, voxel_corrs
