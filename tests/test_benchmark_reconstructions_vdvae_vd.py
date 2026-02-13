import logging

import numpy as np
import pytest

from src.pipelines.benchmark_reconstructions_vdvae_vd import _slice_eval_rows
from src.pipelines.benchmark_reconstructions_vdvae_vd import _align_train_rows


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


def test_align_train_rows_identity():
    train_matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
    train_stim_idx = np.arange(100, 110, dtype=np.int64)
    targets = np.arange(20, dtype=np.float32).reshape(10, 2)

    x_aligned, y_aligned, info = _align_train_rows(
        train_matrix=train_matrix,
        train_stim_idx=train_stim_idx,
        train_targets=targets,
        label="VDVAE",
    )

    np.testing.assert_array_equal(x_aligned, train_matrix)
    np.testing.assert_array_equal(y_aligned, targets)
    assert info["mode"] == "identity"
    assert info["rows_used"] == 10


def test_align_train_rows_stim_index_subset_and_reorder():
    train_matrix = np.arange(60, dtype=np.float32).reshape(12, 5)
    train_stim_idx = np.arange(200, 212, dtype=np.int64)
    targets = np.arange(20, dtype=np.float32).reshape(4, 5)
    target_stim_idx = np.array([210, 203, 209, 201], dtype=np.int64)

    x_aligned, y_aligned, info = _align_train_rows(
        train_matrix=train_matrix,
        train_stim_idx=train_stim_idx,
        train_targets=targets,
        label="CLIP-text",
        target_train_stim_idx=target_stim_idx,
    )

    np.testing.assert_array_equal(x_aligned, train_matrix[[10, 3, 9, 1]])
    np.testing.assert_array_equal(y_aligned, targets)
    assert info["mode"] == "stim_index"
    assert info["rows_used"] == 4


def test_align_train_rows_fallback_prefix_with_warning(caplog):
    train_matrix = np.arange(60, dtype=np.float32).reshape(12, 5)
    train_stim_idx = np.arange(12, dtype=np.int64)
    targets = np.arange(40, dtype=np.float32).reshape(8, 5)

    with caplog.at_level(logging.WARNING):
        x_aligned, y_aligned, info = _align_train_rows(
            train_matrix=train_matrix,
            train_stim_idx=train_stim_idx,
            train_targets=targets,
            label="CLIP-vision",
        )

    np.testing.assert_array_equal(x_aligned, train_matrix[:8])
    np.testing.assert_array_equal(y_aligned, targets)
    assert info["mode"] == "prefix"
    assert info["rows_used"] == 8
    assert "Falling back to first 8 rows" in caplog.text
