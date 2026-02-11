"""
Unit tests for statistical evaluation utilities.
"""

import numpy as np

from src.evaluation.statistics import (
    benjamini_hochberg,
    bootstrap_confidence_interval,
    build_condition_summary,
    permutation_test_mean_difference,
)


class TestBootstrapConfidenceInterval:
    def test_single_value_returns_degenerate_interval(self):
        low, high = bootstrap_confidence_interval([0.25], n_bootstrap=500, ci_level=95.0)
        assert low == 0.25
        assert high == 0.25

    def test_mean_is_inside_interval(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        low, high = bootstrap_confidence_interval(values, n_bootstrap=2000, ci_level=95.0)
        mean_val = float(values.mean())
        assert low <= mean_val <= high


class TestPermutationTestMeanDifference:
    def test_detects_large_difference(self):
        a = np.array([0.8, 0.9, 1.0, 0.85, 0.95], dtype=np.float32)
        b = np.array([0.1, 0.2, 0.15, 0.05, 0.25], dtype=np.float32)
        p = permutation_test_mean_difference(
            sample_a=a,
            sample_b=b,
            n_permutations=5000,
            alternative="greater",
            seed=42,
        )
        assert p < 0.05


class TestBenjaminiHochberg:
    def test_bh_adjustment_shape_and_bounds(self):
        p = np.array([0.001, 0.01, 0.03, 0.2], dtype=np.float64)
        q, reject = benjamini_hochberg(p, alpha=0.05)
        assert q.shape == p.shape
        assert reject.shape == p.shape
        assert np.all(q >= 0)
        assert np.all(q <= 1)

    def test_bh_rejects_small_p_values(self):
        p = np.array([0.001, 0.01, 0.02, 0.6], dtype=np.float64)
        q, reject = benjamini_hochberg(p, alpha=0.05)
        assert reject[0]
        assert reject[1]
        assert not reject[3]


class TestBuildConditionSummary:
    def test_builds_rows_and_metric_columns(self):
        condition_results = {
            0: {
                "repeats": [
                    {"median_r": 0.1, "mean_r": 0.1, "median_pattern_r": 0.1, "two_vs_two": 0.55},
                    {"median_r": 0.12, "mean_r": 0.11, "median_pattern_r": 0.1, "two_vs_two": 0.56},
                ]
            },
            50: {
                "repeats": [
                    {"median_r": 0.2, "mean_r": 0.19, "median_pattern_r": 0.18, "two_vs_two": 0.62},
                    {"median_r": 0.22, "mean_r": 0.21, "median_pattern_r": 0.2, "two_vs_two": 0.64},
                ]
            },
        }

        rows = build_condition_summary(
            condition_results=condition_results,
            metric_names=["median_r", "mean_r", "median_pattern_r", "two_vs_two"],
            n_bootstrap=1000,
            ci_level=95.0,
            permutation_metric="median_r",
            n_permutations=3000,
            permutation_reference_condition=0,
            seed=42,
        )
        assert len(rows) == 2
        by_condition = {row["condition"]: row for row in rows}
        assert 0 in by_condition
        assert 50 in by_condition
        assert "median_r_mean" in by_condition[0]
        assert "median_r_ci_low" in by_condition[50]
        assert "median_r_p_perm_vs_reference" in by_condition[50]
        assert "median_r_q_bh_vs_reference" in by_condition[50]
        assert "median_r_reject_bh_vs_reference" in by_condition[50]

    def test_falls_back_when_reference_has_insufficient_repeats(self):
        condition_results = {
            0: {"repeats": [{"median_r": 0.1}]},  # insufficient for permutation
            10: {"repeats": [{"median_r": 0.12}, {"median_r": 0.14}, {"median_r": 0.13}]},
            50: {"repeats": [{"median_r": 0.2}, {"median_r": 0.21}, {"median_r": 0.22}]},
        }

        rows = build_condition_summary(
            condition_results=condition_results,
            metric_names=["median_r"],
            n_bootstrap=500,
            ci_level=95.0,
            permutation_metric="median_r",
            n_permutations=1000,
            permutation_reference_condition=0,
            seed=42,
        )
        by_condition = {row["condition"]: row for row in rows}
        assert by_condition[50]["permutation_reference_condition"] == 10
        assert "median_r_p_perm_vs_reference" in by_condition[50]
        assert "median_r_q_bh_vs_reference" in by_condition[50]
