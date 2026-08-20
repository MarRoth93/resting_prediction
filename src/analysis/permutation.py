"""Label-free permutation and bootstrap statistics for clinical endpoints."""

from __future__ import annotations

import numpy as np


def _values_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("values must be a non-empty one- or two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains NaN/Inf.")
    return array


def _group_mask(group_a_mask: np.ndarray, n_subjects: int) -> np.ndarray:
    raw = np.asarray(group_a_mask)
    if raw.dtype.kind != "b" or raw.shape != (n_subjects,):
        raise ValueError(
            f"group_a_mask must be boolean with shape ({n_subjects},), "
            f"got {raw.shape}."
        )
    mask = raw.astype(bool, copy=False)
    if int(mask.sum()) < 2 or int((~mask).sum()) < 2:
        raise ValueError("Welch statistics require at least two subjects per group.")
    return mask


def _schedule(schedule: np.ndarray, n_subjects: int) -> np.ndarray:
    raw = np.asarray(schedule)
    if raw.ndim != 2 or raw.shape[1:] != (n_subjects,):
        raise ValueError(
            "schedule must have shape (n_permutations, n_subjects); "
            f"got {raw.shape}."
        )
    if raw.dtype.kind not in "iu":
        raise ValueError("schedule must contain integer indices.")
    normalized = raw.astype(np.int64, copy=False)
    expected = np.arange(n_subjects, dtype=np.int64)
    if any(not np.array_equal(np.sort(row), expected) for row in normalized):
        raise ValueError("Every schedule row must be a permutation of subject indices.")
    return normalized


def build_permutation_schedule(
    n_subjects: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Build a deterministic schedule without identity permutations."""
    if int(n_subjects) < 2:
        raise ValueError("n_subjects must be at least two.")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be positive.")
    n_subjects = int(n_subjects)
    n_permutations = int(n_permutations)
    rng = np.random.default_rng(seed)
    identity = np.arange(n_subjects, dtype=np.int64)
    result = np.empty((n_permutations, n_subjects), dtype=np.int64)
    for index in range(n_permutations):
        row = rng.permutation(n_subjects)
        while np.array_equal(row, identity):
            row = rng.permutation(n_subjects)
        result[index] = row
    return result


def welch_t(values: np.ndarray, group_a_mask: np.ndarray) -> np.ndarray:
    """Return Welch's t statistic for group A minus group B, per column."""
    array = _values_2d(values)
    mask = _group_mask(group_a_mask, array.shape[0])
    group_a = array[mask]
    group_b = array[~mask]
    difference = group_a.mean(axis=0) - group_b.mean(axis=0)
    denominator_squared = (
        group_a.var(axis=0, ddof=1) / group_a.shape[0]
        + group_b.var(axis=0, ddof=1) / group_b.shape[0]
    )
    invalid = (~np.isfinite(denominator_squared)) | (denominator_squared <= 0.0)
    if np.any(invalid):
        columns = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"Welch t is undefined for constant endpoint column(s): {columns}."
        )
    result = difference / np.sqrt(denominator_squared)
    if not np.all(np.isfinite(result)):
        raise ValueError("Welch t produced NaN/Inf.")
    return result


def permutation_pvalue_two_sided(
    values: np.ndarray,
    group_a_mask: np.ndarray,
    schedule: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed Welch t and two-sided subject-permutation p-values."""
    array = _values_2d(values)
    mask = _group_mask(group_a_mask, array.shape[0])
    permutations = _schedule(schedule, array.shape[0])
    observed = welch_t(array, mask)
    exceedances = np.zeros(array.shape[1], dtype=np.int64)
    for row in permutations:
        statistic = welch_t(array[row], mask)
        exceedances += np.abs(statistic) >= np.abs(observed)
    pvalues = (1.0 + exceedances) / (permutations.shape[0] + 1.0)
    return observed, pvalues


def omnibus_sum_t2(values: np.ndarray, group_a_mask: np.ndarray) -> float:
    """Return the sum of squared per-column Welch t statistics."""
    statistic = welch_t(values, group_a_mask)
    return float(np.sum(statistic**2))


def permutation_pvalue_omnibus(
    values: np.ndarray,
    group_a_mask: np.ndarray,
    schedule: np.ndarray,
) -> tuple[float, float]:
    """Return the observed sum-t-squared statistic and its upper-tail p-value."""
    array = _values_2d(values)
    mask = _group_mask(group_a_mask, array.shape[0])
    permutations = _schedule(schedule, array.shape[0])
    observed = omnibus_sum_t2(array, mask)
    exceedances = 0
    for row in permutations:
        exceedances += int(omnibus_sum_t2(array[row], mask) >= observed)
    pvalue = (1.0 + exceedances) / (permutations.shape[0] + 1.0)
    return observed, float(pvalue)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in the input order."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("BH p-values must be a finite one-dimensional vector.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("BH p-values must be between zero and one.")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    pooled_variance = (
        (group_a.shape[0] - 1) * group_a.var(axis=0, ddof=1)
        + (group_b.shape[0] - 1) * group_b.var(axis=0, ddof=1)
    ) / (group_a.shape[0] + group_b.shape[0] - 2)
    if np.any((~np.isfinite(pooled_variance)) | (pooled_variance <= 0.0)):
        columns = np.flatnonzero(
            (~np.isfinite(pooled_variance)) | (pooled_variance <= 0.0)
        ).tolist()
        raise ValueError(
            f"Cohen's d is undefined for constant endpoint column(s): {columns}."
        )
    return (group_a.mean(axis=0) - group_b.mean(axis=0)) / np.sqrt(
        pooled_variance
    )


def stratified_bootstrap(
    values: np.ndarray,
    group_a_mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap group A and B separately and return point estimates and CIs."""
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive.")
    array = _values_2d(values)
    mask = _group_mask(group_a_mask, array.shape[0])
    group_a = array[mask]
    group_b = array[~mask]
    point_difference = group_a.mean(axis=0) - group_b.mean(axis=0)
    point_d = _cohens_d(group_a, group_b)

    rng = np.random.default_rng(seed)
    differences = np.empty((int(n_boot), array.shape[1]), dtype=np.float64)
    effect_sizes = np.empty_like(differences)
    for index in range(int(n_boot)):
        sampled_a = group_a[rng.integers(0, group_a.shape[0], group_a.shape[0])]
        sampled_b = group_b[rng.integers(0, group_b.shape[0], group_b.shape[0])]
        differences[index] = sampled_a.mean(axis=0) - sampled_b.mean(axis=0)
        effect_sizes[index] = _cohens_d(sampled_a, sampled_b)

    return {
        "raw_mean_difference": point_difference,
        "raw_mean_difference_ci95": np.percentile(
            differences, [2.5, 97.5], axis=0
        ).T,
        "cohens_d": point_d,
        "cohens_d_ci95": np.percentile(effect_sizes, [2.5, 97.5], axis=0).T,
    }


def _ols_group_t(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    rank = int(np.linalg.matrix_rank(design))
    degrees_of_freedom = design.shape[0] - rank
    if degrees_of_freedom <= 0:
        raise ValueError("Full model has no residual degrees of freedom.")
    coefficients = np.linalg.pinv(design) @ values
    residuals = values - design @ coefficients
    residual_variance = np.sum(residuals**2, axis=0) / degrees_of_freedom
    group_variance_factor = float(np.linalg.pinv(design.T @ design)[1, 1])
    standard_error_squared = residual_variance * group_variance_factor
    invalid = (~np.isfinite(standard_error_squared)) | (
        standard_error_squared <= 0.0
    )
    if np.any(invalid):
        columns = np.flatnonzero(invalid).tolist()
        raise ValueError(
            "Group coefficient t is undefined for constant endpoint column(s): "
            f"{columns}."
        )
    statistic = coefficients[1] / np.sqrt(standard_error_squared)
    if not np.all(np.isfinite(statistic)):
        raise ValueError("Group coefficient t produced NaN/Inf.")
    return statistic


def freedman_lane(
    values: np.ndarray,
    group_a_mask: np.ndarray,
    covariates: np.ndarray,
    schedule: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a two-sided Freedman-Lane residual-permutation group test."""
    array = _values_2d(values)
    mask = _group_mask(group_a_mask, array.shape[0])
    permutations = _schedule(schedule, array.shape[0])
    nuisance = np.asarray(covariates, dtype=np.float64)
    if nuisance.ndim == 1:
        nuisance = nuisance[:, None]
    if nuisance.ndim != 2 or nuisance.shape[0] != array.shape[0]:
        raise ValueError(
            "covariates must have shape (n_subjects, n_covariates); "
            f"got {nuisance.shape}."
        )
    if not np.all(np.isfinite(nuisance)):
        raise ValueError("covariates contains NaN/Inf.")

    intercept = np.ones((array.shape[0], 1), dtype=np.float64)
    reduced_design = np.column_stack((intercept, nuisance))
    full_design = np.column_stack(
        (intercept, mask.astype(np.float64), nuisance)
    )
    if np.linalg.matrix_rank(full_design) != full_design.shape[1]:
        raise ValueError("Full model design is rank deficient.")

    reduced_coefficients = np.linalg.pinv(reduced_design) @ array
    fitted = reduced_design @ reduced_coefficients
    residuals = array - fitted
    observed = _ols_group_t(array, full_design)
    exceedances = np.zeros(array.shape[1], dtype=np.int64)
    for row in permutations:
        pseudo_values = fitted + residuals[row]
        statistic = _ols_group_t(pseudo_values, full_design)
        exceedances += np.abs(statistic) >= np.abs(observed)
    pvalues = (1.0 + exceedances) / (permutations.shape[0] + 1.0)
    return observed, pvalues
