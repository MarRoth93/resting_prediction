"""
Evaluation metrics for voxel-wise prediction quality.
"""

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def voxelwise_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Per-voxel Pearson r across stimuli.

    Voxels with zero variance return r=0.0 (not NaN).

    Args:
        Y_true: (N, V) ground truth
        Y_pred: (N, V) predictions

    Returns:
        (V,) array of correlations
    """
    N = Y_true.shape[0]
    # Center
    Y_true_c = Y_true - Y_true.mean(axis=0)
    Y_pred_c = Y_pred - Y_pred.mean(axis=0)

    # Standard deviations
    true_std = Y_true_c.std(axis=0)
    pred_std = Y_pred_c.std(axis=0)

    # Handle zero-variance voxels
    zero_var = (true_std < 1e-10) | (pred_std < 1e-10)
    n_zero = zero_var.sum()
    if n_zero > 0:
        logger.info(f"voxelwise_correlation: {n_zero} zero-variance voxels set to r=0")

    true_std[true_std < 1e-10] = 1e-10
    pred_std[pred_std < 1e-10] = 1e-10

    corr = (Y_true_c * Y_pred_c).sum(axis=0) / (N * true_std * pred_std)
    corr[zero_var] = 0.0

    return corr.astype(np.float32)


def median_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Median of voxelwise correlations."""
    return float(np.median(voxelwise_correlation(Y_true, Y_pred)))


def pattern_correlation(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Per-stimulus Pearson r across voxels.

    Args:
        Y_true: (N, V)
        Y_pred: (N, V)

    Returns:
        (N,) array of pattern correlations
    """
    N = Y_true.shape[0]
    corrs = np.zeros(N, dtype=np.float32)
    for i in range(N):
        t = Y_true[i] - Y_true[i].mean()
        p = Y_pred[i] - Y_pred[i].mean()
        t_std = t.std()
        p_std = p.std()
        if t_std < 1e-10 or p_std < 1e-10:
            corrs[i] = 0.0
        else:
            corrs[i] = (t * p).sum() / (len(t) * t_std * p_std)
    return corrs


def two_vs_two_accuracy(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    n_pairs: int = 1000,
    seed: int = 42,
) -> float:
    """
    2-vs-2 identification accuracy.

    Given 2 stimuli and 2 predictions, can we match them correctly?
    Chance = 50%.

    Args:
        Y_true: (N, V)
        Y_pred: (N, V)
        n_pairs: number of random pairs to test
        seed: random seed

    Returns:
        Accuracy (0.5 = chance, 1.0 = perfect)
    """
    N = Y_true.shape[0]
    rng = np.random.RandomState(seed)

    correct = 0
    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)

        # Correct assignment: (true_i, pred_i) + (true_j, pred_j)
        corr_correct = (
            np.corrcoef(Y_true[i], Y_pred[i])[0, 1]
            + np.corrcoef(Y_true[j], Y_pred[j])[0, 1]
        )
        # Swapped assignment
        corr_swapped = (
            np.corrcoef(Y_true[i], Y_pred[j])[0, 1]
            + np.corrcoef(Y_true[j], Y_pred[i])[0, 1]
        )
        if corr_correct > corr_swapped:
            correct += 1

    return correct / n_pairs


def noise_ceiling_split_half(
    trial_fmri: np.ndarray,
    trial_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute noise ceiling via split-half correlation with Spearman-Brown correction.

    For each stimulus with >= 2 trials, split into odd/even,
    compute average per split, correlate across splits.

    Args:
        trial_fmri: (N_trials, V) individual trial betas
        trial_labels: (N_trials,) stimulus index per trial

    Returns:
        (V,) noise ceiling in correlation units
    """
    V = trial_fmri.shape[1]
    label_counts = Counter(trial_labels)

    # Only stimuli with >= 2 trials
    valid_labels = [lbl for lbl, cnt in label_counts.items() if cnt >= 2]
    if len(valid_labels) < 10:
        logger.warning(f"Only {len(valid_labels)} stimuli with >=2 trials for noise ceiling")

    odd_means = []
    even_means = []
    for lbl in valid_labels:
        mask = trial_labels == lbl
        trials = trial_fmri[mask]
        odd_means.append(trials[0::2].mean(axis=0))
        even_means.append(trials[1::2].mean(axis=0))

    odd_arr = np.array(odd_means, dtype=np.float32)  # (N_valid, V)
    even_arr = np.array(even_means, dtype=np.float32)

    # Voxelwise correlation between odd and even averages
    r_half = voxelwise_correlation(odd_arr, even_arr)

    # Spearman-Brown correction: r_full = 2*r_half / (1 + r_half)
    r_half = np.clip(r_half, -0.999, 0.999)
    nc = 2 * r_half / (1 + np.abs(r_half))

    return np.clip(nc, 0, 1).astype(np.float32)


def normalized_performance(
    corrs: np.ndarray,
    nc: np.ndarray,
    nc_floor: float = 0.1,
) -> np.ndarray:
    """
    Fraction of noise ceiling achieved: r / NC.

    Voxels with NC < nc_floor are excluded (set to NaN).

    Returns:
        (V,) array, NaN for excluded voxels
    """
    result = np.full_like(corrs, np.nan)
    valid = nc >= nc_floor
    result[valid] = corrs[valid] / nc[valid]

    n_excluded = (~valid).sum()
    if n_excluded > 0:
        logger.info(f"normalized_performance: {n_excluded} voxels excluded (NC < {nc_floor})")

    return result


def get_reliable_voxels(
    noise_ceiling: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Return boolean mask of voxels exceeding noise ceiling threshold.

    Args:
        noise_ceiling: (V,) NC values
        threshold: minimum NC value

    Returns:
        (V,) boolean mask
    """
    return noise_ceiling >= threshold


def roi_evaluation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    atlas_masked: np.ndarray,
    n_parcels: int,
    roi_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-ROI evaluation.

    Args:
        Y_true: (N, V)
        Y_pred: (N, V)
        atlas_masked: (V,) integer labels
        n_parcels: number of parcels
        roi_names: optional label -> name mapping

    Returns:
        dict of roi_name -> {'median_r': float, 'mean_r': float, 'n_voxels': int}
    """
    voxel_corrs = voxelwise_correlation(Y_true, Y_pred)
    results = {}

    for p in range(1, n_parcels + 1):
        mask = atlas_masked == p
        if mask.sum() == 0:
            continue
        name = roi_names.get(p, f"ROI_{p}") if roi_names else f"ROI_{p}"
        roi_corrs = voxel_corrs[mask]
        results[name] = {
            "median_r": float(np.median(roi_corrs)),
            "mean_r": float(np.mean(roi_corrs)),
            "n_voxels": int(mask.sum()),
        }

    return results
