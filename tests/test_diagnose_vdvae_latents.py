import numpy as np

from src.data.prepare_reconstruction_features import _VDVAE_LAYER_DIMS
from src.pipelines.diagnose_vdvae_latents import (
    layer_bounds,
    pairwise_reliability,
    per_dim_correlation,
    per_row_correlation,
)


def test_layer_bounds_cover_vdvae_latents_contiguously():
    bounds = layer_bounds(_VDVAE_LAYER_DIMS)

    assert bounds[0][0] == 0
    assert all(first[1] == second[0] for first, second in zip(bounds, bounds[1:]))
    assert sum(end - start for start, end in bounds) == 91168


def test_per_dim_correlation_handles_perfect_sign_flip_and_constant_columns():
    values = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 3.0, 2.0],
            [2.0, 2.0, 8.0],
            [4.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(per_dim_correlation(values, values), 1.0)
    np.testing.assert_allclose(per_dim_correlation(values, -values), -1.0)

    with_constant = values.copy()
    with_constant[:, 1] = 7.0
    correlations = per_dim_correlation(with_constant, with_constant)
    assert correlations[1] == 0.0
    assert not np.isnan(correlations).any()


def test_per_row_correlation_handles_perfect_and_constant_rows():
    values = np.array(
        [
            [0.0, 1.0, 4.0, 2.0],
            [1.0, 3.0, 2.0, 7.0],
            [2.0, 2.0, 8.0, 5.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(per_row_correlation(values, values), 1.0)

    with_constant = values.copy()
    with_constant[1] = 6.0
    correlations = per_row_correlation(with_constant, with_constant)
    assert correlations[1] == 0.0


def test_pairwise_reliability_decreases_with_noise():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(40, 128)).astype(np.float32)
    low_noise = [
        signal + rng.normal(scale=0.05, size=signal.shape).astype(np.float32)
        for _ in range(3)
    ]
    high_noise = [
        signal + rng.normal(scale=3.0, size=signal.shape).astype(np.float32)
        for _ in range(3)
    ]

    low_reliability = pairwise_reliability(low_noise)
    high_reliability = pairwise_reliability(high_noise)

    assert low_reliability["mean_dim_reliability"] > high_reliability[
        "mean_dim_reliability"
    ]
    assert low_reliability["n_pairs"] == 3
    assert high_reliability["n_pairs"] == 3
