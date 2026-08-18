import numpy as np
import pytest

from src.pipelines.swap_control_reconstruction import (
    _parse_subjects,
    constant_rows,
    default_subjects,
    pairwise_summary,
    rowwise_pearson,
)


def test_constant_rows_flags_zero_variance():
    matrix = np.asarray([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    assert constant_rows(matrix).tolist() == [False, True, True]


def test_rowwise_pearson_perfect_and_inverse():
    a = np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]])
    b = np.asarray([[2.0, 4.0, 6.0], [-1.0, 0.0, 1.0]])
    values = rowwise_pearson(a, b)
    assert values == pytest.approx([1.0, -1.0])


def test_rowwise_pearson_rejects_constant_rows():
    a = np.ones((2, 3))
    b = np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]])
    with pytest.raises(ValueError, match="Constant row"):
        rowwise_pearson(a, b)


def test_pairwise_summary_identical_matrices():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(10, 8))
    summary = pairwise_summary({"a": shared, "b": shared.copy(), "c": shared.copy()})
    assert len(summary["pairs"]) == 3
    assert summary["same_image_mean"] == pytest.approx(1.0)
    assert abs(summary["mismatched_image_mean"]) < 0.9


def test_pairwise_summary_independent_matrices():
    rng = np.random.default_rng(1)
    summary = pairwise_summary(
        {"a": rng.normal(size=(50, 40)), "b": rng.normal(size=(50, 40))}
    )
    assert abs(summary["same_image_mean"]) < 0.3


def test_default_subjects_takes_sorted_prefix(tmp_path):
    for name in ("sub-0200", "sub-0100", "sub-0300", "not-a-subject"):
        (tmp_path / name).mkdir()
    assert default_subjects(tmp_path, 2) == ["subj07", "sub-0100", "sub-0200"]


def test_default_subjects_requires_enough_for_subjects(tmp_path):
    (tmp_path / "sub-0100").mkdir()
    with pytest.raises(ValueError, match="Need 3 FOR subjects"):
        default_subjects(tmp_path, 3)


def test_parse_subjects_rejects_invalid_and_duplicate(tmp_path):
    with pytest.raises(ValueError, match="Invalid subject labels"):
        _parse_subjects(tmp_path, ["subj07", "bogus"], 5)
    with pytest.raises(ValueError, match="unique"):
        _parse_subjects(tmp_path, ["sub-0100", "sub-0100"], 5)
    assert _parse_subjects(tmp_path, ["subj07,sub-0100"], 5) == ["subj07", "sub-0100"]
