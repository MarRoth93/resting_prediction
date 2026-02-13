import logging

import numpy as np
import pytest

from src.pipelines.benchmark_reconstructions_vdvae_vd import _slice_eval_rows


def test_slice_eval_rows_all_valid():
    arr = np.arange(30, dtype=np.float32).reshape(10, 3)
    eval_indices = np.array([0, 2, 5, 9], dtype=np.int64)

    sliced, mask = _slice_eval_rows(arr, eval_indices, "VDVAE")

    assert mask.tolist() == [True, True, True, True]
    np.testing.assert_array_equal(sliced, arr[eval_indices])


def test_slice_eval_rows_drops_out_of_bounds(caplog):
    arr = np.arange(20, dtype=np.float32).reshape(10, 2)
    eval_indices = np.array([1, 4, 10, 12], dtype=np.int64)

    with caplog.at_level(logging.WARNING):
        sliced, mask = _slice_eval_rows(arr, eval_indices, "CLIP-text")

    assert mask.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(sliced, arr[[1, 4]])
    assert "CLIP-text test rows shorter than eval split" in caplog.text


def test_slice_eval_rows_raises_when_no_valid_rows():
    arr = np.arange(12, dtype=np.float32).reshape(6, 2)
    eval_indices = np.array([6, 7, 8], dtype=np.int64)

    with pytest.raises(ValueError, match="incompatible with eval split"):
        _slice_eval_rows(arr, eval_indices, "CLIP-vision")
