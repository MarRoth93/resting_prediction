import numpy as np

from src.data.prepare_task_data import _sorted_stimulus_ids


def test_stimulus_ids_are_sorted():
    sig = {12: [0], 3: [1], 8: [2]}
    out = _sorted_stimulus_ids(sig)
    np.testing.assert_array_equal(out, np.array([3, 8, 12], dtype=np.int64))
