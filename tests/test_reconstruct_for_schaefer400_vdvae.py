import numpy as np
from sklearn.linear_model import Ridge

from src.pipelines.reconstruct_for_schaefer400_vdvae import (
    fit_chunked_ridge,
    standardize_parcel_patterns,
)


def test_standardize_parcel_patterns_zero_fills_missing_columns():
    rng = np.random.RandomState(4)
    values = rng.randn(12, 400).astype(np.float32)
    available = np.ones(400, dtype=bool)
    available[[3, 98]] = False
    values[:, ~available] = np.nan

    standardized = standardize_parcel_patterns(
        values,
        available_parcels=available,
    )

    np.testing.assert_allclose(standardized[:, ~available], 0.0)
    np.testing.assert_allclose(standardized[:, available].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized[:, available].std(axis=0, ddof=1), 1.0, atol=1e-6)


def test_chunked_ridge_matches_sklearn(tmp_path):
    rng = np.random.RandomState(9)
    design = rng.randn(25, 400).astype(np.float32)
    targets = rng.randn(18, 13).astype(np.float32)
    target_rows = rng.randint(0, targets.shape[0], size=design.shape[0])
    alpha = 7.0
    coefficient_path = tmp_path / "coefficients.npy"
    intercept_path = tmp_path / "intercept.npy"

    fit_chunked_ridge(
        design=design,
        targets=targets,
        target_rows=target_rows,
        alpha=alpha,
        chunk_size=4,
        coefficient_path=coefficient_path,
        intercept_path=intercept_path,
    )

    expected = Ridge(alpha=alpha, fit_intercept=True).fit(design, targets[target_rows])
    actual = design @ np.load(coefficient_path) + np.load(intercept_path)
    np.testing.assert_allclose(actual, expected.predict(design), atol=2e-5, rtol=2e-5)
