"""
Tests for fixed evaluation split utilities.
"""

import numpy as np

from src.pipelines.eval_split import (
    load_or_create_fixed_eval_split,
    sample_shot_indices_from_fixed_eval,
)


def test_load_or_create_fixed_eval_split_roundtrip(tmp_path):
    model_dir = str(tmp_path / "model")
    eval_idx_1, meta_1, path_1 = load_or_create_fixed_eval_split(
        model_dir=model_dir,
        test_sub=7,
        n_shared=1000,
        eval_size=250,
        seed=42,
    )
    eval_idx_2, meta_2, path_2 = load_or_create_fixed_eval_split(
        model_dir=model_dir,
        test_sub=7,
        n_shared=1000,
        eval_size=250,
        seed=999,  # ignored after artifact exists
    )

    assert path_1 == path_2
    assert meta_1["source"] == "created"
    assert meta_2["source"] == "loaded"
    assert eval_idx_1.shape == (250,)
    assert np.array_equal(eval_idx_1, np.sort(eval_idx_1))
    assert np.array_equal(eval_idx_1, eval_idx_2)


def test_sample_shot_indices_from_fixed_eval_nonoverlap():
    n_shared = 100
    eval_indices = np.arange(20, dtype=np.int64)
    shots, actual = sample_shot_indices_from_fixed_eval(
        n_shared=n_shared,
        eval_indices=eval_indices,
        n_shots=30,
        seed=123,
    )

    assert actual == 30
    assert shots.shape == (30,)
    assert np.intersect1d(shots, eval_indices).size == 0
    assert shots.min() >= 20
    assert shots.max() < n_shared
