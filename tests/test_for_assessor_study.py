import json

import numpy as np

from src.pipelines.for_assessor_study import (
    _bh_qvalues,
    _contrast,
    _read_scores,
    _stimulus_id,
)


def test_read_scores_normalizes_assessor_target_names(tmp_path):
    score_dir = tmp_path / "study" / "assessor_scores"
    score_dir.mkdir(parents=True)
    score_path = score_dir / "example_six.csv"
    score_path.write_text("image,Approach,Attention\nimage.png,1.25,2.5\n")
    score_path.with_suffix(".manifest.json").write_text(
        json.dumps({"status": "complete"})
    )
    config = {"_root": str(tmp_path), "study": {"output_root": "study"}}

    scores = _read_scores(config, "example", "six")

    assert scores == {"image.png": {"approach": 1.25, "attention": 2.5}}


def test_stimulus_id_supports_original_and_reconstruction_names():
    assert _stimulus_id("stim00123.png") == 123
    assert _stimulus_id("row00004_stim70123.png") == 70123


def test_subject_level_contrast_reports_requested_direction():
    result = _contrast(
        np.array([3.0, 4.0, 5.0]),
        np.array([1.0, 2.0, 3.0]),
        permutations=100,
        bootstrap_samples=100,
        rng=np.random.RandomState(2),
    )

    assert result["difference_first_minus_second"] == 2.0
    assert result["n_first"] == 3
    assert result["n_second"] == 3
    assert 0.0 < result["permutation_p"] <= 1.0


def test_bh_qvalues_are_monotonic_in_pvalue_order():
    pvalues = [0.04, 0.001, 0.02, 0.8]
    qvalues = _bh_qvalues(pvalues)
    ordered = sorted(zip(pvalues, qvalues))

    assert all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:]))
    assert all(0.0 <= value <= 1.0 for value in qvalues)
