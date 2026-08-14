import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from src.data import voxel_contract
from src.data.voxel_contract import (
    FOR_ATLAS_FILENAME,
    FOR_PARCEL_TIMESERIES_FILENAME,
    NSD_ATLAS_FILENAME,
    build_for_subject,
    build_nsd_contract,
    build_nsd_subject,
    decode_freesurfer_parcel_labels,
)


def _write_contract(output_root: Path, common_parcel_ids=range(1, 61)) -> None:
    output_root.mkdir(parents=True)
    (output_root / "contract.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "common_parcel_ids": list(common_parcel_ids),
            }
        )
    )


def _save_nifti(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values.astype(np.int16), np.eye(4)), str(path))


def test_decode_freesurfer_labels_ignores_aseg_and_background():
    encoded = np.asarray(
        [0, 17, 999, 1000, 1001, 1200, 1500, 2000, 2001, 2200, 2201]
    )
    expected = np.asarray([0, 0, 0, 0, 1, 200, 0, 0, 201, 400, 0])
    np.testing.assert_array_equal(decode_freesurfer_parcel_labels(encoded), expected)


def test_nsd_contract_uses_train_intersection_and_rejects_fewer_than_60(tmp_path):
    data_root = tmp_path / "processed"
    parcel_root = tmp_path / "processed_schaefer400"
    output_root = tmp_path / "contract"
    base_labels = np.arange(1, 62, dtype=np.int16).reshape(61, 1, 1)
    for subject in range(1, 7):
        subject_tag = f"subj{subject:02d}"
        subject_data = data_root / subject_tag
        subject_data.mkdir(parents=True)
        np.save(subject_data / "mask.npy", np.ones(base_labels.shape, dtype=bool))
        labels = base_labels.copy()
        if subject == 6:
            labels[-1, 0, 0] = 62
        _save_nifti(parcel_root / subject_tag / NSD_ATLAS_FILENAME, labels)

    manifest = build_nsd_contract(
        output_root=output_root,
        nsd_data_root=data_root,
        nsd_parcel_root=parcel_root,
    )
    assert manifest["contract_version"] == 1
    assert manifest["common_parcel_ids"] == list(range(1, 61))
    assert manifest["policies"] == voxel_contract.POLICIES
    assert set(manifest["nsd_atlas_sha256"]) == {
        f"subj{subject:02d}" for subject in range(1, 7)
    }
    assert len(manifest["nsd_parcel_voxel_counts"]["subj06"]) == 400
    assert json.loads((output_root / "contract.json").read_text()) == manifest
    assert not list(output_root.glob(".*.tmp"))

    failing_labels = base_labels.copy()
    failing_labels[-2:, 0, 0] = [62, 63]
    _save_nifti(parcel_root / "subj06" / NSD_ATLAS_FILENAME, failing_labels)
    with pytest.raises(ValueError, match="59 parcels; at least 60"):
        build_nsd_contract(
            output_root=output_root,
            nsd_data_root=data_root,
            nsd_parcel_root=parcel_root,
        )


def test_for_coordinate_validation_rejects_zero_based_ijk(tmp_path):
    mat_path = tmp_path / "sub-0001_time_by_voxel.mat"
    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset(
            "voxel_timeseries",
            data=np.zeros((1, 237), dtype=np.float32),
        )
        handle.create_dataset("voxel_ijk", data=np.asarray([[0], [1], [1]]))
        handle.create_dataset("bold_spatial_size", data=np.asarray([[2], [2], [2]]))
        handle.create_dataset("TR_seconds", data=np.asarray([[2.0]]))

    with pytest.raises(ValueError, match="expected 1-based indexing"):
        voxel_contract._load_for_voxel_mat(mat_path)


def _write_for_sources(voxel_root: Path, atlas_root: Path) -> tuple[np.ndarray, np.ndarray]:
    subject_label = "sub-0001"
    time = np.arange(237, dtype=np.float32)
    voxel_timeseries = np.stack(
        [
            np.full(237, 5.0, dtype=np.float32),
            100.0 + time,
            20.0 + np.sin(time / 9.0),
            30.0 + np.cos(time / 11.0),
            40.0 + np.sin(time / 7.0) + time / 20.0,
            50.0 + np.cos(time / 5.0) - time / 30.0,
            60.0 + time / 10.0,
        ]
    ).astype(np.float32)
    voxel_root.mkdir(parents=True)
    with h5py.File(voxel_root / f"{subject_label}_time_by_voxel.mat", "w") as handle:
        handle.create_dataset("voxel_timeseries", data=voxel_timeseries)
        handle.create_dataset(
            "voxel_ijk",
            data=np.asarray(
                [
                    np.arange(1, 8),
                    np.ones(7, dtype=np.int64),
                    np.ones(7, dtype=np.int64),
                ]
            ),
        )
        handle.create_dataset("bold_spatial_size", data=np.asarray([[7], [1], [1]]))
        handle.create_dataset("TR_seconds", data=np.asarray([[2.0]]))

    encoded_labels = np.asarray([1001, 1001, 1002, 1003, 1004, 1005, 17])
    subject_atlas_dir = atlas_root / subject_label
    _save_nifti(
        subject_atlas_dir / FOR_ATLAS_FILENAME,
        encoded_labels.reshape(7, 1, 1),
    )
    shipped = np.zeros((400, 237), dtype=np.float32)
    decoded = decode_freesurfer_parcel_labels(encoded_labels)
    for parcel_id in range(1, 6):
        shipped[parcel_id - 1] = voxel_timeseries[decoded == parcel_id].mean(axis=0)
    with h5py.File(subject_atlas_dir / FOR_PARCEL_TIMESERIES_FILENAME, "w") as handle:
        handle.create_dataset("parcel_timeseries", data=shipped)
    return voxel_timeseries, decoded


def test_for_preprocessing_drops_two_zscores_and_zero_fills(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    voxel_root = tmp_path / "voxel"
    atlas_root = tmp_path / "atlas"
    _write_contract(output_root)
    _, decoded = _write_for_sources(voxel_root, atlas_root)

    replacements = []
    real_replace = voxel_contract.os.replace

    def tracked_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(voxel_contract.os, "replace", tracked_replace)
    provenance = build_for_subject(
        "sub-0001",
        output_root=output_root,
        for_voxel_root=voxel_root,
        for_atlas_root=atlas_root,
    )

    subject_output = output_root / "for" / "sub-0001"
    targets = np.load(subject_output / "rest_targets.npy")
    seeds = np.load(subject_output / "rest_seeds.npy")
    available = np.load(subject_output / "seed_available.npy")
    target_parcels = np.load(subject_output / "target_parcel_ids.npy")
    assert targets.shape == (235, 6)
    assert targets.dtype == np.float32
    assert seeds.shape == (235, 400)
    assert seeds.dtype == np.float32
    assert available.shape == (400,)
    assert available.dtype == np.bool_
    np.testing.assert_array_equal(available[:6], [True, True, True, True, True, False])
    np.testing.assert_array_equal(seeds[:, ~available], 0)
    np.testing.assert_allclose(seeds[:, available].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(seeds[:, available].std(axis=0), 1, atol=1e-6)
    np.testing.assert_array_equal(target_parcels, decoded[:6])
    np.testing.assert_array_equal(targets[:, 0], 0)
    np.testing.assert_allclose(targets[:, 1:].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(targets[:, 1:].std(axis=0), 1, atol=1e-6)
    assert provenance["n_zero_variance_target_voxels"] == 1
    assert provenance["parcels_available"] == 5
    assert provenance["min_parcel_agreement_r"] > 0.999
    assert provenance["preprocessing"]["discard_initial_trs"] == 2
    loaded_provenance = json.loads((subject_output / "provenance.json").read_text())
    assert loaded_provenance == provenance
    assert any(
        source.name == ".provenance.json.tmp"
        and destination == subject_output / "provenance.json"
        for source, destination in replacements
    )
    assert not list(subject_output.glob(".*.tmp"))


def test_nsd_rest_alignment_rejects_filename_and_tr_mismatches(tmp_path):
    output_root = tmp_path / "output"
    data_root = tmp_path / "processed"
    parcel_root = tmp_path / "processed_schaefer400"
    _write_contract(output_root)
    voxel_dir = data_root / "subj01"
    parcel_dir = parcel_root / "subj01"
    voxel_dir.mkdir(parents=True)
    parcel_dir.mkdir(parents=True)
    mask = np.ones((60, 1, 1), dtype=bool)
    np.save(voxel_dir / "mask.npy", mask)
    _save_nifti(
        parcel_dir / NSD_ATLAS_FILENAME,
        np.arange(1, 61, dtype=np.int16).reshape(mask.shape),
    )
    np.save(voxel_dir / "rest_run1.npy", np.zeros((5, 60), dtype=np.float32))
    np.save(parcel_dir / "rest_run2.npy", np.zeros((5, 400), dtype=np.float32))

    with pytest.raises(ValueError, match="filename sets differ"):
        build_nsd_subject(
            1,
            output_root=output_root,
            nsd_data_root=data_root,
            nsd_parcel_root=parcel_root,
        )

    (parcel_dir / "rest_run2.npy").unlink()
    np.save(parcel_dir / "rest_run1.npy", np.zeros((4, 400), dtype=np.float32))
    with pytest.raises(ValueError, match="REST TR mismatch"):
        build_nsd_subject(
            1,
            output_root=output_root,
            nsd_data_root=data_root,
            nsd_parcel_root=parcel_root,
        )


def test_for_cli_continues_after_subject_failure_and_returns_nonzero(
    tmp_path,
    monkeypatch,
    caplog,
):
    output_root = tmp_path / "output"
    _write_contract(output_root)
    attempted = []

    def fake_build(subject_label, **kwargs):
        attempted.append(subject_label)
        if subject_label == "sub-0001":
            raise FileNotFoundError("missing atlas directory")
        return {}

    monkeypatch.setattr(voxel_contract, "build_for_subject", fake_build)
    result = voxel_contract.main(
        [
            "--dataset",
            "for",
            "--subjects",
            "sub-0001",
            "sub-0002",
            "--output-root",
            str(output_root),
        ]
    )

    assert attempted == ["sub-0001", "sub-0002"]
    assert result == 1
    assert "sub-0001: missing atlas directory" in caplog.text
