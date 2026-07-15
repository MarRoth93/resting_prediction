import numpy as np

from src.pipelines.eval_split import fixed_eval_indices


def test_fixed_eval_indices_are_deterministic_and_unique():
    first = fixed_eval_indices(n_shared=1000, eval_size=200, seed=42)
    second = fixed_eval_indices(n_shared=1000, eval_size=200, seed=42)

    assert first.shape == (200,)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.unique(first))
    assert first.min() >= 0 and first.max() < 1000
