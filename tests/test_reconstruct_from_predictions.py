from pathlib import Path

import numpy as np
import pytest

import src.pipelines.reconstruct_from_predictions as reconstruction


def test_parcel_average_responses_uses_decoder_column_order():
    parcel_columns = np.arange(1, 73, dtype=np.int64)
    target_parcel_ids = np.concatenate((parcel_columns, np.asarray([2, 7])))
    responses = np.stack(
        (
            np.arange(1, 75, dtype=np.float32),
            np.arange(101, 175, dtype=np.float32),
        )
    )

    averaged = reconstruction.parcel_average_responses(
        responses,
        target_parcel_ids,
        parcel_columns,
        subject="synthetic",
    )

    expected = responses[:, :72].copy()
    expected[:, 1] = responses[:, [1, 72]].mean(axis=1)
    expected[:, 6] = responses[:, [6, 73]].mean(axis=1)
    np.testing.assert_allclose(averaged, expected)


def test_decoder_column_mismatch_raises():
    parcel_columns = np.arange(1, 73, dtype=np.int64)
    mismatched = parcel_columns.copy()
    mismatched[-1] = 99
    with pytest.raises(ValueError, match="parcel set does not match decoder columns"):
        reconstruction.parcel_average_responses(
            np.ones((2, 72), dtype=np.float32),
            mismatched,
            parcel_columns,
            subject="sub-0001",
        )


def test_go_no_go_threshold_logic_is_exact():
    below = reconstruction.go_no_go_verdict(0.519, 0.500)
    boundary = reconstruction.go_no_go_verdict(0.520, 0.500)

    assert below["detectable_signal"] is False
    assert below["warning"] == (
        "WARNING: decoder carries no detectable signal at 72 parcels"
    )
    assert boundary["detectable_signal"] is True
    assert boundary["warning"] is None


def test_resumability_skips_existing_images_and_force_repeats(tmp_path):
    stimulus_ids = np.asarray([101, 202, 303], dtype=np.int64)
    for row, stimulus_id in enumerate(stimulus_ids.tolist()):
        (tmp_path / reconstruction._image_filename(row, stimulus_id)).write_bytes(b"png")

    assert reconstruction.missing_image_indices(tmp_path, stimulus_ids).size == 0
    (tmp_path / reconstruction._image_filename(1, stimulus_ids[1])).unlink()
    np.testing.assert_array_equal(
        reconstruction.missing_image_indices(tmp_path, stimulus_ids),
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        reconstruction.missing_image_indices(tmp_path, stimulus_ids, force=True),
        np.arange(3, dtype=np.int64),
    )


def test_module_has_no_clinical_label_file_reference():
    source = Path(reconstruction.__file__).read_text()
    forbidden_name = "for" + "_groups.csv"
    assert forbidden_name not in source
