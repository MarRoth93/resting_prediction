import numpy as np

from src.pipelines.vdvae_calibration_battery import (
    bootstrap_ci,
    calibrate_affine,
    pixcorr,
    two_way_identification,
)


def test_calibrate_affine_restores_target_moments_and_guards_zero_variance():
    rng = np.random.default_rng(42)
    pred = rng.normal(size=(128, 5)).astype(np.float32)
    target_mean = np.array([-1.0, 0.5, 2.0, 4.0, -3.0], dtype=np.float32)
    target_std = np.array([0.2, 0.4, 0.8, 1.5, 2.0], dtype=np.float32)

    calibrated = calibrate_affine(pred, target_mean, target_std)

    assert calibrated.dtype == np.float32
    np.testing.assert_allclose(calibrated.mean(axis=0), target_mean, atol=1e-6)
    np.testing.assert_allclose(calibrated.std(axis=0), target_std, atol=1e-6)

    pred[:, 2] = 7.0
    guarded = calibrate_affine(pred, target_mean, target_std)
    assert np.isfinite(guarded).all()
    np.testing.assert_allclose(guarded[:, 2], target_mean[2])


def test_pixcorr_identical_noise_and_constant_images():
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, size=(32, 32, 32, 3), dtype=np.uint8)
    independent = rng.integers(0, 256, size=images.shape, dtype=np.uint8)

    np.testing.assert_allclose(pixcorr(images, images), 1.0)
    assert abs(float(pixcorr(images, independent).mean())) < 0.02

    constant = np.full((4, 8, 8, 3), 11, dtype=np.uint8)
    correlations = pixcorr(constant, constant)
    np.testing.assert_array_equal(correlations, 0.0)
    assert not np.isnan(correlations).any()


def test_two_way_identification_identity_and_random_similarity():
    np.testing.assert_allclose(two_way_identification(np.eye(20)), 1.0)

    rng = np.random.default_rng(11)
    random_similarity = rng.normal(size=(400, 400))
    assert abs(two_way_identification(random_similarity) - 0.5) < 0.05


def test_bootstrap_ci_brackets_sample_mean():
    values = np.arange(1, 101, dtype=np.float32)

    low, high = bootstrap_ci(values, n_boot=1000, seed=0)

    assert low < float(values.mean()) < high
