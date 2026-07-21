import numpy as np
import pytest

from src.data.prepare_reliability_data import validate_trial_average_matches_existing
from src.data.prepare_reliability_data import (
    _load_selected_masked_volumes,
    _validated_trial_design,
)


def test_trial_design_accepts_matlab_row_vector():
    masterordering = np.asarray([[1, 1001, 2]])
    subjectim = np.zeros((2, 1001), dtype=np.int64)
    subjectim[0, 0] = 11
    subjectim[0, 1] = 12
    subjectim[0, 1000] = 13

    presentations, stimuli = _validated_trial_design(
        masterordering,
        subjectim,
        subject=1,
        n_trials=3,
    )

    np.testing.assert_array_equal(presentations, [1, 1001, 2])
    np.testing.assert_array_equal(stimuli, [10, 12, 11])


def test_selected_volume_loader_avoids_proxy_fancy_indexing():
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    class BasicSliceOnlyProxy:
        def __getitem__(self, key):
            assert isinstance(key[-1], slice)
            return values[key]

    class Image:
        shape = values.shape
        dataobj = BasicSliceOnlyProxy()

    mask = np.asarray([[True, False, True], [False, True, False]])
    indices = np.asarray([0, 2, 3])

    loaded = _load_selected_masked_volumes(Image(), mask, indices)

    np.testing.assert_array_equal(loaded, values[..., indices][mask].T)


def test_trial_average_validation_matches_frozen_rows():
    rng = np.random.RandomState(5)
    trials = rng.randn(36, 4).astype(np.float32)
    labels = np.repeat(np.arange(12), 3)
    averaged = np.stack([trials[labels == label].mean(axis=0) for label in range(12)])

    error = validate_trial_average_matches_existing(trials, labels, averaged)

    assert error == pytest.approx(0.0)


def test_trial_average_validation_fails_before_write_on_mismatch():
    rng = np.random.RandomState(6)
    trials = rng.randn(30, 3).astype(np.float32)
    labels = np.repeat(np.arange(10), 3)
    averaged = np.stack([trials[labels == label].mean(axis=0) for label in range(10)])
    averaged[2, 1] += 0.1

    with pytest.raises(ValueError, match="No files were written"):
        validate_trial_average_matches_existing(trials, labels, averaged)
