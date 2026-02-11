"""
Statistical utilities for research-grade evaluation.
"""

import csv
from typing import Any

import numpy as np


def bootstrap_confidence_interval(
    values: np.ndarray | list[float],
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    statistic: str = "mean",
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a 1D sample.

    Args:
        values: sample values
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level in percent (0, 100)
        statistic: 'mean' or 'median'
        seed: RNG seed

    Returns:
        (ci_low, ci_high)
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("bootstrap_confidence_interval requires at least one value")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap should be >= 100 for stable intervals")
    if not 0 < ci_level < 100:
        raise ValueError(f"ci_level must be in (0, 100), got {ci_level}")
    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported statistic: {statistic}")

    rng = np.random.RandomState(seed)
    n = arr.size
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    samples = arr[indices]

    if statistic == "mean":
        boot_stats = samples.mean(axis=1)
    else:
        boot_stats = np.median(samples, axis=1)

    alpha = (100.0 - ci_level) / 2.0
    low = float(np.percentile(boot_stats, alpha))
    high = float(np.percentile(boot_stats, 100.0 - alpha))
    return low, high


def permutation_test_mean_difference(
    sample_a: np.ndarray | list[float],
    sample_b: np.ndarray | list[float],
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> float:
    """
    Two-sample permutation test on mean difference.

    Tests H0: mean(sample_a) == mean(sample_b)

    Args:
        sample_a: first sample
        sample_b: second sample
        n_permutations: number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        seed: RNG seed

    Returns:
        p-value
    """
    a = np.asarray(sample_a, dtype=np.float64).ravel()
    b = np.asarray(sample_b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError("permutation_test_mean_difference requires >=2 values per sample")
    if n_permutations < 100:
        raise ValueError("n_permutations should be >= 100")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unsupported alternative: {alternative}")

    observed = float(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.RandomState(seed)

    exceed = 0
    for _ in range(n_permutations):
        perm = combined[rng.permutation(combined.size)]
        diff = float(perm[:n_a].mean() - perm[n_a:].mean())
        if alternative == "greater":
            is_extreme = diff >= observed
        elif alternative == "less":
            is_extreme = diff <= observed
        else:
            is_extreme = abs(diff) >= abs(observed)
        if is_extreme:
            exceed += 1

    return float((exceed + 1) / (n_permutations + 1))


def benjamini_hochberg(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: list/array of raw p-values
        alpha: target FDR level

    Returns:
        q_values: BH-adjusted p-values (same order as input)
        reject: boolean array indicating q <= alpha
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=bool)
    if np.any(~np.isfinite(p)):
        raise ValueError("p_values contain NaN/Inf")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    m = p.size
    order = np.argsort(p)
    ranked_p = p[order]

    # Raw BH adjusted values
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = ranked_p * m / ranks
    adjusted = np.minimum(adjusted, 1.0)

    # Enforce monotonicity from largest rank to smallest
    adjusted_monotone = np.minimum.accumulate(adjusted[::-1])[::-1]

    q = np.empty_like(adjusted_monotone)
    q[order] = adjusted_monotone
    reject = q <= alpha
    return q.astype(np.float64), reject.astype(bool)


def build_condition_summary(
    condition_results: dict[int, dict[str, Any]],
    metric_names: list[str] | None = None,
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    permutation_metric: str = "median_r",
    n_permutations: int = 10000,
    permutation_reference_condition: int = 0,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build a summary table with CIs and permutation p-values vs reference.

    Args:
        condition_results: condition -> {'repeats': [metric dicts], ...}
        metric_names: metrics to aggregate; inferred from first condition if None
        n_bootstrap: bootstrap samples for CI
        ci_level: confidence level
        permutation_metric: metric used for permutation test
        n_permutations: permutation count
        permutation_reference_condition: preferred reference condition
        fdr_alpha: FDR threshold for BH correction
        seed: RNG seed

    Returns:
        List of row dicts sorted by condition key.
    """
    if not condition_results:
        return []

    sorted_conditions = sorted(condition_results.keys())
    first_repeats = condition_results[sorted_conditions[0]].get("repeats", [])
    if metric_names is None:
        if not first_repeats:
            return []
        candidate = [
            "median_r",
            "mean_r",
            "median_pattern_r",
            "two_vs_two",
            "noise_ceiling_median",
        ]
        metric_names = [m for m in candidate if m in first_repeats[0]]

    rows: list[dict[str, Any]] = []

    reference_condition = None
    reference_values = None

    if permutation_reference_condition in condition_results:
        reps = condition_results[permutation_reference_condition].get("repeats", [])
        vals = [r[permutation_metric] for r in reps if permutation_metric in r]
        if len(vals) >= 2:
            reference_condition = permutation_reference_condition
            reference_values = np.asarray(vals, dtype=np.float64)

    if reference_values is None:
        for condition in sorted_conditions:
            reps = condition_results[condition].get("repeats", [])
            vals = [r[permutation_metric] for r in reps if permutation_metric in r]
            if len(vals) >= 2:
                reference_condition = condition
                reference_values = np.asarray(vals, dtype=np.float64)
                break

    p_rows: list[int] = []
    p_vals: list[float] = []
    p_key = f"{permutation_metric}_p_perm_vs_reference"

    for idx, condition in enumerate(sorted_conditions):
        repeats = condition_results[condition].get("repeats", [])
        row: dict[str, Any] = {
            "condition": int(condition),
            "n_repeats": int(len(repeats)),
        }

        for metric_idx, metric in enumerate(metric_names):
            values = np.asarray(
                [r[metric] for r in repeats if metric in r],
                dtype=np.float64,
            )
            if values.size == 0:
                continue

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
            ci_low, ci_high = bootstrap_confidence_interval(
                values=values,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                statistic="mean",
                seed=seed + idx * 1000 + metric_idx,
            )
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high

        if (
            reference_values is not None
            and condition != reference_condition
            and row["n_repeats"] >= 2
        ):
            cond_vals = np.asarray(
                [r[permutation_metric] for r in repeats if permutation_metric in r],
                dtype=np.float64,
            )
            if cond_vals.size >= 2:
                row["permutation_reference_condition"] = int(reference_condition)
                row[f"{permutation_metric}_delta_vs_reference"] = float(
                    cond_vals.mean() - reference_values.mean()
                )
                row[p_key] = permutation_test_mean_difference(
                    sample_a=cond_vals,
                    sample_b=reference_values,
                    n_permutations=n_permutations,
                    alternative="greater",
                    seed=seed + idx * 2000 + 17,
                )
                p_rows.append(len(rows))
                p_vals.append(float(row[p_key]))

        rows.append(row)

    if p_vals:
        q_vals, reject = benjamini_hochberg(p_vals, alpha=fdr_alpha)
        q_key = f"{permutation_metric}_q_bh_vs_reference"
        r_key = f"{permutation_metric}_reject_bh_vs_reference"
        for i, row_idx in enumerate(p_rows):
            rows[row_idx][q_key] = float(q_vals[i])
            rows[row_idx][r_key] = bool(reject[i])
            rows[row_idx]["fdr_alpha"] = float(fdr_alpha)

    return rows


def save_summary_csv(rows: list[dict[str, Any]], path: str):
    """Write summary rows to CSV."""
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
