import numpy as np

from src.pipelines.benchmark_reconstructions_vdvae_vd import _align_train_rows


def test_align_train_rows_prefers_stim_index_when_counts_match():
    train_matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
    train_stim_idx = np.arange(100, 110, dtype=np.int64)
    targets = np.arange(20, dtype=np.float32).reshape(10, 2)
    target_stim_idx = np.array([109, 108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=np.int64)

    x_aligned, y_aligned, info = _align_train_rows(
        train_matrix=train_matrix,
        train_stim_idx=train_stim_idx,
        train_targets=targets,
        label="VDVAE",
        target_train_stim_idx=target_stim_idx,
    )

    np.testing.assert_array_equal(x_aligned, train_matrix[::-1])
    np.testing.assert_array_equal(y_aligned, targets)
    assert info["mode"] == "stim_index"
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
