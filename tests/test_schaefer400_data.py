import csv

import h5py
import numpy as np
import pytest

from src.data.schaefer400 import (
    ParcelReducer,
    canonicalize_schaefer_volume_labels,
    load_for_schaefer400_subject,
)


def _write_for_subject(path, *, missing=(52,), timepoints=20):
    path.mkdir(parents=True)
    rng = np.random.RandomState(4)
    series = rng.randn(400, timepoints).astype(np.float32)
    for index in missing:
        series[index] = np.nan
    with h5py.File(path / "schaefer400_parcel_timeseries.mat", "w") as handle:
        handle.create_dataset("parcel_timeseries", data=series)
        handle.create_dataset("parcel_ids", data=np.arange(1, 401)[:, None])
        handle.create_dataset("TR_seconds", data=np.asarray([[2.0]]))
        handle.create_dataset(
            "atlas_file",
            data=np.asarray([ord(value) for value in "schaefer400_7net"], dtype=np.uint16),
        )
    with (path / "schaefer400_parcel_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["parcel_id", "voxel_count", "usable"],
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(400):
            writer.writerow(
                {
                    "parcel_id": index + 1,
                    "voxel_count": 0 if index in missing else 20,
                    "usable": 0 if index in missing else 1,
                }
            )


def test_canonicalize_cbig_volume_labels():
    source = np.asarray([0, 1000, 1001, 1200, 2000, 2001, 2200])
    np.testing.assert_array_equal(
        canonicalize_schaefer_volume_labels(source),
        np.asarray([0, 0, 1, 200, 0, 201, 400]),
    )
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_schaefer_volume_labels(np.asarray([3001]))


def test_parcel_reducer_returns_ordered_means():
    labels = np.arange(1, 401, dtype=np.int16).reshape(20, 20, 1)
    reducer = ParcelReducer(labels)
    values = np.empty((20, 20, 1, 3), dtype=np.float32)
    values[..., 0] = labels
    values[..., 1] = labels * 2
    values[..., 2] = -labels
    reduced = reducer.reduce_4d(values)
    assert reduced.shape == (3, 400)
    np.testing.assert_allclose(reduced[0], np.arange(1, 401))
    np.testing.assert_allclose(reduced[1], np.arange(1, 401) * 2)
    np.testing.assert_allclose(reduced[2], -np.arange(1, 401))


def test_for_loader_separates_missing_targets_from_ordered_seed_rows(tmp_path):
    subject_dir = tmp_path / "sub-0027"
    _write_for_subject(subject_dir, missing=(52, 392))
    subject = load_for_schaefer400_subject(subject_dir)

    assert subject.subject_id == 27
    assert subject.standardized_timeseries.shape == (20, 400)
    assert int(subject.available_parcels.sum()) == 398
    assert subject.rest_runs[0].shape == (20, 398)
    assert subject.seed_runs[0].shape == (20, 400)
    np.testing.assert_array_equal(subject.seed_runs[0][:, [52, 392]], 0)
    np.testing.assert_allclose(
        subject.standardized_timeseries[:, subject.available_parcels].mean(axis=0),
        0,
        atol=1e-6,
    )

    expanded = subject.expand_available(np.ones((3, 398), dtype=np.float32))
    assert expanded.shape == (3, 400)
    assert np.isnan(expanded[:, [52, 392]]).all()
    assert np.isfinite(expanded[:, subject.available_parcels]).all()
