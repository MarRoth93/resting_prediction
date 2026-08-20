import itertools

import numpy as np
import pytest
from scipy.stats import ttest_ind

from src.analysis.permutation import (
    bh_adjust,
    build_permutation_schedule,
    freedman_lane,
    omnibus_sum_t2,
    permutation_pvalue_omnibus,
    permutation_pvalue_two_sided,
    stratified_bootstrap,
    welch_t,
)


def _all_group_relabelings(n_subjects: int, n_group_a: int) -> np.ndarray:
    identity_mask = np.arange(n_subjects) < n_group_a
    rows = []
    for selected in itertools.combinations(range(n_subjects), n_group_a):
        mask = np.zeros(n_subjects, dtype=bool)
        mask[list(selected)] = True
        if np.array_equal(mask, identity_mask):
            continue
        group_a = np.flatnonzero(mask)
        group_b = np.flatnonzero(~mask)
        rows.append(np.concatenate((group_a, group_b)))
    return np.asarray(rows, dtype=np.int64)


def test_welch_t_matches_scipy():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(19, 4))
    mask = np.arange(19) < 8
    expected = ttest_ind(
        values[mask], values[~mask], axis=0, equal_var=False
    ).statistic
    np.testing.assert_allclose(welch_t(values, mask), expected, rtol=1e-12)


def test_exhaustive_4v4_two_sided_permutation_pvalue():
    values = np.asarray([0.0, 0.3, 0.9, 1.2, 1.1, 1.8, 2.0, 2.5])[:, None]
    mask = np.arange(8) < 4
    schedule = _all_group_relabelings(8, 4)
    observed, pvalue = permutation_pvalue_two_sided(values, mask, schedule)

    all_statistics = [abs(float(observed[0]))]
    for selected in itertools.combinations(range(8), 4):
        relabeled = np.zeros(8, dtype=bool)
        relabeled[list(selected)] = True
        if np.array_equal(relabeled, mask):
            continue
        all_statistics.append(abs(float(welch_t(values, relabeled)[0])))
    expected = np.mean(np.asarray(all_statistics) >= abs(observed[0]))
    assert pvalue[0] == pytest.approx(expected)


def test_omnibus_upper_tail_detects_injected_effect_and_null_is_sane():
    rng = np.random.default_rng(5)
    mask = np.arange(40) < 20
    schedule = build_permutation_schedule(40, 999, 42)
    null_values = rng.normal(size=(40, 5))
    effect_values = null_values.copy()
    effect_values[mask] += 1.5

    observed, effect_p = permutation_pvalue_omnibus(
        effect_values, mask, schedule
    )
    _, null_p = permutation_pvalue_omnibus(null_values, mask, schedule)
    assert observed == pytest.approx(omnibus_sum_t2(effect_values, mask))
    assert effect_p < 0.01
    assert 0.05 < null_p < 0.95


def test_bh_adjust_matches_hand_computed_example():
    np.testing.assert_allclose(
        bh_adjust(np.asarray([0.01, 0.04, 0.03, 0.002])),
        np.asarray([0.02, 0.04, 0.04, 0.008]),
    )


def test_freedman_lane_handles_confound_and_detects_true_group_effect():
    rng = np.random.default_rng(9)
    n_subjects = 60
    mask = np.arange(n_subjects) < 30
    covariate = mask.astype(float) + rng.normal(scale=0.35, size=n_subjects)
    noise = rng.normal(scale=0.7, size=n_subjects)
    schedule = build_permutation_schedule(n_subjects, 999, 73)

    no_effect = (2.0 * covariate + noise)[:, None]
    _, no_effect_p = freedman_lane(no_effect, mask, covariate, schedule)
    with_effect = (2.0 * covariate + 1.5 * mask + noise)[:, None]
    _, effect_p = freedman_lane(with_effect, mask, covariate, schedule)
    assert no_effect_p[0] > 0.05
    assert effect_p[0] < 0.05


def test_schedule_is_deterministic_and_excludes_identity():
    first = build_permutation_schedule(8, 100, 42)
    second = build_permutation_schedule(8, 100, 42)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 8)
    assert first.dtype == np.int64
    identity = np.arange(8)
    assert not any(np.array_equal(row, identity) for row in first)
    assert all(np.array_equal(np.sort(row), identity) for row in first)


def test_stratified_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(12)
    mask = np.arange(200) < 100
    values = rng.normal(size=200)
    values[mask] += 0.75
    result = stratified_bootstrap(values, mask, n_boot=2_000, seed=1042)
    low, high = result["raw_mean_difference_ci95"][0]
    assert low < 0.75 < high
    assert result["raw_mean_difference"][0] == pytest.approx(
        values[mask].mean() - values[~mask].mean()
    )


def test_constant_endpoint_raises_instead_of_returning_nan_pvalue():
    values = np.ones((8, 1))
    mask = np.arange(8) < 4
    schedule = build_permutation_schedule(8, 20, 1)
    with pytest.raises(ValueError, match="constant endpoint"):
        permutation_pvalue_two_sided(values, mask, schedule)
