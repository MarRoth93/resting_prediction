import numpy as np
import pytest

from src.pipelines.multiexpert_evaluation import balanced_repeated_noise_ceiling


def test_three_repeat_noise_ceiling_is_balanced_and_order_invariant():
    rng = np.random.RandomState(14)
    stimuli, voxels, repeats = 40, 7, 3
    signal = rng.randn(stimuli, voxels).astype(np.float32)
    rows = []
    labels = []
    for stimulus in range(stimuli):
        for _ in range(repeats):
            rows.append(signal[stimulus] + 0.25 * rng.randn(voxels))
            labels.append(stimulus)
    rows = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels)

    first, n_repeats, n_stimuli = balanced_repeated_noise_ceiling(rows, labels)
    row_grid = np.arange(rows.shape[0]).reshape(stimuli, repeats)
    order = row_grid[:, [2, 0, 1]].reshape(-1)
    second, _, _ = balanced_repeated_noise_ceiling(rows[order], labels[order])

    assert n_repeats == 3
    assert n_stimuli == stimuli
    assert first.shape == (voxels,)
    assert np.all((first >= 0) & (first <= 1))
    np.testing.assert_allclose(second, first, atol=1e-6)
    assert float(np.median(first)) > 0.8


def test_noise_ceiling_uses_balanced_max_repeat_subset():
    rng = np.random.RandomState(2)
    labels = np.concatenate(
        [np.repeat(np.arange(12), 3), np.repeat(np.arange(12, 17), 2), np.arange(17, 20)]
    )
    rows = rng.randn(labels.size, 4).astype(np.float32)
    ceiling, repeats, retained = balanced_repeated_noise_ceiling(rows, labels)

    assert ceiling.shape == (4,)
    assert repeats == 3
    assert retained == 12
