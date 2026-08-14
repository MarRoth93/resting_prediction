import numpy as np
import pytest

from src.pipelines.vdvae_calibration import (
    apply_calibration,
    learn_calibration,
    load_calibration,
    make_folds,
    save_calibration,
)


def test_make_folds_partitions_rows_and_is_deterministic():
    folds = make_folds(n_rows=23, n_folds=4, seed=42)
    repeated = make_folds(n_rows=23, n_folds=4, seed=42)
    other_seed = make_folds(n_rows=23, n_folds=4, seed=43)

    assert folds.shape == (23,)
    assert np.issubdtype(folds.dtype, np.integer)
    np.testing.assert_array_equal(folds, repeated)
    assert not np.array_equal(folds, other_seed)
    np.testing.assert_array_equal(np.unique(folds), np.arange(4))
    counts = np.bincount(folds, minlength=4)
    assert int(counts.max() - counts.min()) <= 1


def test_make_folds_rejects_fewer_than_two_folds():
    with pytest.raises(ValueError, match="at least 2"):
        make_folds(n_rows=10, n_folds=1, seed=42)


def test_learn_and_apply_calibration_restore_target_moments():
    rng = np.random.default_rng(7)
    pred = rng.normal(
        loc=np.array([-3.0, 2.0, 10.0]),
        scale=np.array([0.5, 4.0, 2.0]),
        size=(2000, 3),
    )
    target = rng.normal(
        loc=np.array([8.0, -4.0, 1.5]),
        scale=np.array([3.0, 0.25, 5.0]),
        size=(2000, 3),
    )
    calibration = learn_calibration(
        oof_pred_mean=pred.mean(axis=0),
        oof_pred_std=pred.std(axis=0),
        target_mean=target.mean(axis=0),
        target_std=target.std(axis=0),
    )

    calibrated = apply_calibration(pred, calibration)

    np.testing.assert_allclose(calibrated.mean(axis=0), target.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(calibrated.std(axis=0), target.std(axis=0), rtol=1e-5)


def test_gain_cap_engages_for_near_zero_source_std():
    calibration = learn_calibration(
        oof_pred_mean=np.zeros(3),
        oof_pred_std=np.array([1.0, 1e-12, 2.0]),
        target_mean=np.zeros(3),
        target_std=np.array([1.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(calibration["gain"], np.array([1.0, 10.0, 1.0]))
    assert calibration["n_capped"] == 1
    assert calibration["median_gain"] == pytest.approx(1.0)


def test_save_load_round_trips_arrays_and_metadata(tmp_path):
    calibration = learn_calibration(
        oof_pred_mean=np.array([1.0, 2.0]),
        oof_pred_std=np.array([2.0, 4.0]),
        target_mean=np.array([-1.0, 3.0]),
        target_std=np.array([4.0, 2.0]),
    )
    metadata = {
        "n_folds": 3,
        "seed": 42,
        "alpha": 50000.0,
        "n_rows": 100,
        "gain_cap_multiple": 10.0,
        "n_capped": calibration["n_capped"],
    }
    path = tmp_path / "vdvae_calibration.npz"

    save_calibration(path, calibration, metadata)
    loaded, loaded_metadata = load_calibration(path)

    for key in ("offset_in", "gain", "offset_out"):
        np.testing.assert_array_equal(loaded[key], calibration[key])
    assert loaded["n_capped"] == calibration["n_capped"]
    assert loaded["median_gain"] == pytest.approx(calibration["median_gain"])
    assert loaded_metadata == metadata


def test_apply_calibration_returns_float32():
    calibration = learn_calibration(
        oof_pred_mean=np.array([0.0, 1.0]),
        oof_pred_std=np.array([1.0, 2.0]),
        target_mean=np.array([2.0, 3.0]),
        target_std=np.array([4.0, 5.0]),
    )

    result = apply_calibration(np.ones((4, 2), dtype=np.float64), calibration)

    assert result.dtype == np.float32
