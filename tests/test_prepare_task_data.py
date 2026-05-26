import numpy as np
import pytest

from src.data.prepare_task_data import _ordered_stimulus_ids


def test_ordered_stimulus_ids_sorted():
    sig = {12: [0], 3: [1], 8: [2]}
    out = _ordered_stimulus_ids(sig, stimulus_order="sorted")
    np.testing.assert_array_equal(out, np.array([3, 8, 12], dtype=np.int64))


def test_ordered_stimulus_ids_insertion():
    sig = {12: [0], 3: [1], 8: [2]}
    out = _ordered_stimulus_ids(sig, stimulus_order="insertion")
    np.testing.assert_array_equal(out, np.array([12, 3, 8], dtype=np.int64))


def test_ordered_stimulus_ids_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported stimulus_order"):
        _ordered_stimulus_ids({1: [0]}, stimulus_order="unknown")
