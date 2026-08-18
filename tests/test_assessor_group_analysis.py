import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.pipelines import assessor_score_reconstructions
from src.pipelines.assessor_group_analysis import (
    DIMENSIONS,
    _bh_qvalues,
    _load_score_pickle,
    analyze_scores,
)


def _score_payload(
    subject: str,
    stimulus_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    ood: float,
) -> dict:
    ood_values = np.full(stimulus_ids.size, ood, dtype=np.float32)
    return {
        "subject": subject,
        "stimulus_ids": np.asarray(stimulus_ids, dtype=np.int64),
        "scores": {
            dimension: np.asarray(values, dtype=np.float32)
            for dimension, values in scores.items()
        },
        "va_extras": {
            dimension: {
                "interval_low": scores[dimension] - 0.25,
                "interval_high": scores[dimension] + 0.25,
                "ood_percentile": ood_values,
            }
            for dimension in ("valence", "arousal")
        },
        "bundle_shas": {"va": "va-sha", "six": "six-sha"},
        "image_dir_sha_sample": "sample-sha",
        "created_utc": "2026-08-14T00:00:00+00:00",
    }


def _write_pickle(path: Path, payload: dict) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def _synthetic_scores(tmp_path: Path) -> tuple[Path, Path]:
    scores_root = tmp_path / "scores"
    scores_root.mkdir()
    stimulus_ids = np.asarray([101, 202, 303], dtype=np.int64)
    originals = {
        dimension: np.asarray([1.0, 2.0, 3.0], dtype=np.float32) + index
        for index, dimension in enumerate(DIMENSIONS)
    }
    _write_pickle(
        scores_root / "originals.pkl",
        _score_payload("originals", stimulus_ids, originals, ood=5.0),
    )

    group_rows = ["subject,group"]
    for group, start, delta, ood in (
        ("healthy", 0, 0.1, 40.0),
        ("depressed", 25, 0.6, 60.0),
    ):
        for within_group in range(25):
            subject = f"sub-{start + within_group:04d}"
            subject_shift = (within_group - 12) * 0.01
            scores = {
                dimension: values + delta + subject_shift
                for dimension, values in originals.items()
            }
            _write_pickle(
                scores_root / f"{subject}.pkl",
                _score_payload(subject, stimulus_ids, scores, ood + subject_shift),
            )
            group_rows.append(f"{subject},{group}")

    reference_scores = {
        dimension: values + 0.3 for dimension, values in originals.items()
    }
    _write_pickle(
        scores_root / "subj07.pkl",
        _score_payload("subj07", stimulus_ids, reference_scores, ood=50.0),
    )
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text("\n".join(group_rows) + "\n")
    return scores_root, groups_csv


def test_synthetic_pickle_analysis_is_complete_and_deterministic(tmp_path):
    scores_root, groups_csv = _synthetic_scores(tmp_path)
    first_root = tmp_path / "analysis_first"
    second_root = tmp_path / "analysis_second"

    first = analyze_scores(
        scores_root=scores_root,
        groups_csv=groups_csv,
        output_root=first_root,
        permutations=31,
        seed=42,
    )
    second = analyze_scores(
        scores_root=scores_root,
        groups_csv=groups_csv,
        output_root=second_root,
        permutations=31,
        seed=42,
    )

    assert first["status"] == "exploratory"
    assert first["motion_corrected"] is False
    assert first["outside_frozen_endpoint_families"] is True
    assert first["owner_requested"] == "2026-08-14"
    assert first["dimensions"] == second["dimensions"]
    assert first["subject_level"] == second["subject_level"]
    assert first["dimensions"]["valence"]["mean_healthy_delta"] == pytest.approx(0.1)
    assert first["dimensions"]["valence"]["mean_depressed_delta"] == pytest.approx(0.6)
    assert first["dimensions"]["valence"][
        "difference_depressed_minus_healthy"
    ] == pytest.approx(0.5)
    assert first["dimensions"]["valence"]["mean_healthy_raw_score"] == pytest.approx(
        2.1
    )
    assert first["dimensions"]["valence"][
        "mean_depressed_raw_score"
    ] == pytest.approx(2.6)
    assert first["dimensions"]["dominance"]["dominance_exploratory"] is True

    for name in (
        "fig1_subject_mean_deltas.png",
        "fig2_per_image_differences.png",
        "fig3_content_dependence.png",
        "fig4_ood_percentiles.png",
        "summary.json",
        "per_image_differences.pkl",
    ):
        assert (first_root / name).is_file()
    saved_summary = json.loads((first_root / "summary.json").read_text())
    assert saved_summary["dimensions"] == first["dimensions"]
    with (first_root / "per_image_differences.pkl").open("rb") as handle:
        per_image = pickle.load(handle)
    assert per_image["status"] == "exploratory"
    assert per_image["motion_corrected"] is False
    np.testing.assert_allclose(per_image["differences"]["valence"], 0.5)


def test_score_pickle_nan_error_names_subject_and_dimension(tmp_path):
    path = tmp_path / "sub-0001.pkl"
    stimulus_ids = np.asarray([1, 2], dtype=np.int64)
    scores = {
        dimension: np.ones(2, dtype=np.float32) for dimension in DIMENSIONS
    }
    scores["attention"][1] = np.nan
    _write_pickle(path, _score_payload("sub-0001", stimulus_ids, scores, ood=50.0))

    with pytest.raises(ValueError, match=r"sub-0001/attention contains NaN/Inf"):
        _load_score_pickle(path, expected_subject="sub-0001", require_ood=True)


def test_scoring_source_has_no_for_group_reference():
    source = Path(assessor_score_reconstructions.__file__).read_text()
    assert "for_groups" not in source


def test_bh_qvalues_match_known_vector():
    assert _bh_qvalues([0.01, 0.04, 0.03, 0.002]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )
