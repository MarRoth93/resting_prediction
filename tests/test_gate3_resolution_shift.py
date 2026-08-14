import numpy as np
import pytest

from src.pipelines.gate3_resolution_shift import (
    _block_average,
    _coarse_block_mapping,
    _parse_folds,
    resolution_decision_fields,
    resolution_delta,
    transform_fingerprints,
)


def test_mask_vector_coordinates_and_blocks_round_trip_known_pattern():
    volume = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    mask = np.zeros(volume.shape, dtype=bool)
    selected_coordinates = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 2, 0],
            [2, 0, 0],
            [3, 1, 1],
        ],
        dtype=np.int64,
    )
    mask[tuple(selected_coordinates.T)] = True
    target_indices = np.array([0, 1, 2, 4, 5], dtype=np.int64)

    block_coordinates, inverse, counts = _coarse_block_mapping(
        mask,
        target_indices,
    )
    target_values = volume[mask][target_indices]
    coarse = _block_average(target_values[None, :], inverse, counts)

    assert np.array_equal(np.argwhere(mask), selected_coordinates)
    assert np.array_equal(
        block_coordinates,
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
    )
    assert np.array_equal(inverse, np.array([0, 0, 0, 1, 1]))
    assert np.array_equal(counts, np.array([3, 2]))
    assert np.array_equal(
        coarse,
        np.array(
            [[
                volume[tuple(selected_coordinates[[0, 1, 2]].T)].mean(),
                volume[tuple(selected_coordinates[[4, 5]].T)].mean(),
            ]],
            dtype=np.float32,
        ),
    )


def test_column_normalized_fingerprints_have_unit_column_norms():
    fingerprint = np.array(
        [[3.0, 0.0], [4.0, 5.0], [0.0, 12.0]],
        dtype=np.float64,
    )
    template = np.array(
        [[8.0, 0.0], [15.0, 7.0], [0.0, 24.0]],
        dtype=np.float64,
    )

    transformed_fingerprint, transformed_template = transform_fingerprints(
        fingerprint,
        template,
        variant="column_normalized",
        target_parcel_ids=np.array([1, 2, 3]),
    )

    assert np.array_equal(
        np.linalg.norm(transformed_fingerprint, axis=0),
        np.ones(2),
    )
    assert np.array_equal(
        np.linalg.norm(transformed_template, axis=0),
        np.ones(2),
    )


def test_volume_weights_apply_to_held_out_parcel_rows_only():
    fingerprint = np.arange(1, 9, dtype=np.float64).reshape(4, 2)
    template = fingerprint + 10.0
    parcel_ids = np.array([1, 1, 1, 3, 3], dtype=np.int64)
    expected_weights = np.array(
        [1.0 / np.sqrt(3.0), 1.0, 1.0 / np.sqrt(2.0), 1.0]
    )

    transformed_fingerprint, transformed_template = transform_fingerprints(
        fingerprint,
        template,
        variant="volume_weighted",
        target_parcel_ids=parcel_ids,
    )

    assert np.allclose(
        transformed_fingerprint,
        fingerprint * expected_weights[:, None],
    )
    assert np.allclose(
        transformed_template,
        template * expected_weights[:, None],
    )


def test_delta_selection_and_flag_logic_are_exact():
    assert resolution_delta(0.17, 0.18) == pytest.approx(-0.01)
    decisions = resolution_decision_fields(
        {
            "raw": [-0.02, -0.01],
            "column_normalized": [-0.005, -0.01],
            "volume_weighted": [-0.01, -0.01],
        }
    )
    assert decisions["mean_delta_per_variant"] == pytest.approx(
        {
            "raw": -0.015,
            "column_normalized": -0.0075,
            "volume_weighted": -0.01,
        }
    )
    assert decisions["selected_variant"] == "column_normalized"
    assert decisions["flag_first_order"] is False

    at_threshold = resolution_decision_fields(
        {
            "raw": [-0.02],
            "column_normalized": [-0.01],
            "volume_weighted": [-0.03],
        }
    )
    assert at_threshold["selected_variant"] == "column_normalized"
    assert at_threshold["flag_first_order"] is False

    below_threshold = resolution_decision_fields(
        {
            "raw": [-0.02],
            "column_normalized": [-0.011],
            "volume_weighted": [-0.03],
        }
    )
    assert below_threshold["selected_variant"] == "column_normalized"
    assert below_threshold["flag_first_order"] is True


def test_folds_reject_subject_seven():
    with pytest.raises(Exception, match="within 1-6"):
        _parse_folds("1,7")
