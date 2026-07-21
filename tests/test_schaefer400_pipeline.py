import csv
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from src.pipelines.predict_for_schaefer400 import predict_for_schaefer400
from src.pipelines.schaefer400_support import (
    fit_schaefer400_fusion,
    load_schaefer400_model,
    prepare_schaefer400_training,
    save_schaefer400_model,
)
from src.schaefer400_config import load_schaefer400_config


def _write_nsd_subject(root: Path, subject: int, rng: np.random.RandomState):
    destination = root / f"subj{subject:02d}"
    destination.mkdir(parents=True)
    train_ids = np.arange(12) + (subject - 1) * 4
    test_ids = np.arange(20, 28)
    np.save(destination / "train_stim_idx.npy", train_ids)
    np.save(destination / "test_stim_idx.npy", test_ids)
    np.save(destination / "train_fmri.npy", rng.randn(12, 400).astype(np.float32))
    np.save(destination / "test_fmri.npy", rng.randn(8, 400).astype(np.float32))
    for run in (1, 2):
        np.save(destination / f"rest_run{run}.npy", rng.randn(35, 400).astype(np.float32))
    np.save(destination / "parcel_voxel_counts.npy", np.full(400, 20, dtype=np.int32))
    (destination / "atlas.nii.gz").write_bytes(b"synthetic atlas contract")
    (destination / "task_data_summary.json").write_text("{}\n")
    (destination / "rest_run_manifest.json").write_text("{}\n")


def _write_for_subject(path: Path, rng: np.random.RandomState):
    path.mkdir(parents=True)
    series = rng.randn(400, 40).astype(np.float32)
    series[52] = np.nan
    with h5py.File(path / "schaefer400_parcel_timeseries.mat", "w") as handle:
        handle.create_dataset("parcel_timeseries", data=series)
        handle.create_dataset("parcel_ids", data=np.arange(1, 401)[:, None])
        handle.create_dataset("TR_seconds", data=np.asarray([[2.0]]))
    with (path / "schaefer400_parcel_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["parcel_id", "voxel_count", "usable"])
        for parcel in range(1, 401):
            usable = int(parcel != 53)
            writer.writerow([parcel, 20 * usable, usable])


def test_schaefer_training_artifact_and_for_prediction_roundtrip(tmp_path):
    rng = np.random.RandomState(12)
    data_root = tmp_path / "prepared"
    for subject in (1, 2):
        _write_nsd_subject(data_root, subject, rng)
    feature_path = tmp_path / "clip.npy"
    np.save(feature_path, rng.randn(40, 6).astype(np.float32))
    for_root = tmp_path / "for"
    _write_for_subject(for_root / "sub-0027", rng)

    config = deepcopy(load_schaefer400_config("config_schaefer400.yaml"))
    config["subjects"]["train"] = [1, 2]
    config["data_root"] = str(data_root)
    config["raw_data_root"] = str(tmp_path / "raw")
    config["for_data_root"] = str(for_root)
    config["output_root"] = str(tmp_path / "output")
    config["features"]["path"] = str(feature_path)
    config["atlas"]["nsd_filename"] = "atlas.nii.gz"
    config["experts"].update({"n_components": 3, "min_k": 2})
    config["experts"]["hybrid_cha"].update({"max_iters": 2, "tol": 1e-4})
    config["experts"]["connectivity_srm"].update({"max_iters": 2, "tol": 1e-4})
    config["fusion"]["backbone"].update(
        {"d_model": 12, "n_layers": 1, "n_heads": 3, "dropout": 0.0}
    )
    config["fusion"].update(
        {
            "method_projection_dim": 8,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "transformer_dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 1,
            "patience": 1,
            "val_fraction": 0.25,
            "device": "cpu",
        }
    )

    prepared = prepare_schaefer400_training(config)
    encoder = fit_schaefer400_fusion(prepared)
    model_dir = tmp_path / "model"
    manifest = save_schaefer400_model(prepared, encoder, output_dir=model_dir)
    assert manifest["representation"] == "schaefer400_parcel_means"
    assert manifest["n_parcels"] == 400
    loaded_manifest, contract, _, loaded_encoder = load_schaefer400_model(
        model_dir=model_dir,
        config=config,
    )
    assert loaded_manifest["expert_dims"] == {"hybrid_cha": 3, "connectivity_srm": 3}
    assert contract.training_subjects == (1, 2)
    assert loaded_encoder.network.num_regions == 400

    external_features = tmp_path / "external_clip.npy"
    np.save(external_features, rng.randn(5, 6).astype(np.float32))
    # The public entry point reloads YAML, so persist this small effective config.
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path / "prediction"
    result = predict_for_schaefer400(
        for_subject="sub-0027",
        features_path=external_features,
        config_path=str(config_path),
        model_dir=model_dir,
        output_dir=output,
    )
    prediction = np.load(output / "learned_fusion.npy")
    assert result["accuracy_computed"] is False
    assert prediction.shape == (5, 400)
    assert np.isnan(prediction[:, 52]).all()
    assert np.isfinite(np.delete(prediction, 52, axis=1)).all()
