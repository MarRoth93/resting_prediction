import json
from types import SimpleNamespace

import numpy as np
import pytest

import src.pipelines.for_inference as for_inference
from src.pipelines.multiexpert_artifacts import file_sha256


def _write_inputs(tmp_path):
    model_dir = tmp_path / "final" / "model"
    encoder_dir = model_dir / "encoder"
    encoder_dir.mkdir(parents=True)
    (model_dir / "builder.npz").write_bytes(b"synthetic builder")
    (encoder_dir / "metadata.json").write_text("{}\n")
    artifact_fingerprints = {
        "model/builder.npz": "synthetic-builder",
        "model/encoder/metadata.json": "synthetic-encoder",
    }
    (model_dir.parent / "manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "config-fingerprint",
                "effective_config_sha256": "effective-config-fingerprint",
            }
        )
    )
    (model_dir.parent / "final_result.json").write_text(
        json.dumps(
            {
                "artifact_fingerprints": artifact_fingerprints,
                "result_payload_fingerprint": "final-model-fingerprint",
            }
        )
    )

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    rng = np.random.RandomState(3)
    features = rng.normal(size=(5, 4)).astype(np.float32)
    np.save(selection_dir / "clip_features.npy", features)
    np.save(selection_dir / "nsd_stimulus_ids.npy", np.arange(10, 15, dtype=np.int64))
    (selection_dir / "selection_manifest.json").write_text(
        json.dumps(
            {
                "features_sha256": file_sha256(selection_dir / "clip_features.npy"),
                "prediction_rows": 5,
                "feature_width": 4,
            }
        )
    )

    contract_root = tmp_path / "contract"
    subject_dir = contract_root / "for" / "sub-0001"
    subject_dir.mkdir(parents=True)
    rest_seeds = rng.normal(size=(12, 400)).astype(np.float32)
    rest_seeds[:, -2:] = 0.0
    np.save(subject_dir / "rest_targets.npy", rng.normal(size=(12, 3)).astype(np.float32))
    np.save(subject_dir / "rest_seeds.npy", rest_seeds)
    seed_available = np.ones(400, dtype=bool)
    seed_available[-2:] = False
    np.save(subject_dir / "seed_available.npy", seed_available)
    np.save(subject_dir / "target_parcel_ids.npy", np.array([1, 2, 3], dtype=np.int16))
    (subject_dir / "provenance.json").write_text(
        json.dumps({"contract_version": 1, "n_target_voxels": 3})
    )
    return model_dir, selection_dir, contract_root, subject_dir


class _StubBuilder:
    ensemble_method = "average"

    def __init__(self):
        self.calls = 0

    def align_new_subject_zeroshot(self, *, rest_runs, external_seed_runs):
        self.calls += 1
        assert rest_runs[0].shape == (12, 3)
        assert external_seed_runs[0].shape == (12, 400)
        return (
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            np.eye(2, dtype=np.float32),
        )


class _StubEncoder:
    input_dim = 4

    def __init__(self):
        self.config = SimpleNamespace(device="cuda")

    def predict_voxels(self, features, P, R):
        assert P.shape == (3, 2)
        assert R.shape == (2, 2)
        return np.column_stack((features[:, 0], features[:, 1], features[:, 2]))


def _patch_model(monkeypatch):
    builder = _StubBuilder()
    encoder = _StubEncoder()
    monkeypatch.setattr(
        for_inference.SharedSpaceBuilder,
        "load",
        lambda _model_dir: builder,
    )
    monkeypatch.setattr(for_inference, "load_encoder", lambda _model_dir: encoder)
    return builder, encoder


def test_check_mode_names_the_missing_bundle_file(tmp_path):
    model_dir, selection_dir, contract_root, subject_dir = _write_inputs(tmp_path)
    (subject_dir / "rest_seeds.npy").unlink()

    with pytest.raises(FileNotFoundError, match=r"rest_seeds\.npy"):
        for_inference.main(
            [
                "--command",
                "check",
                "--subjects",
                "sub-0001",
                "--model-dir",
                str(model_dir),
                "--contract-root",
                str(contract_root),
                "--selection-dir",
                str(selection_dir),
            ]
        )


def test_synthetic_prediction_roundtrip_and_provenance(tmp_path, monkeypatch):
    model_dir, selection_dir, contract_root, subject_dir = _write_inputs(tmp_path)
    _, encoder = _patch_model(monkeypatch)
    output_root = tmp_path / "outputs"

    result = for_inference.predict_for_subjects(
        model_dir=model_dir,
        contract_root=contract_root,
        selection_dir=selection_dir,
        output_root=output_root,
        subjects=["sub-0001"],
        device="cpu",
    )

    subject_output = output_root / "sub-0001"
    assert result["subjects"]["sub-0001"]["status"] == "completed"
    assert encoder.config.device == "cpu"
    assert np.load(subject_output / "predicted_responses.npy").shape == (5, 3)
    assert np.load(subject_output / "alignment_P.npy").shape == (3, 2)
    assert np.load(subject_output / "alignment_R.npy").shape == (2, 2)
    assert np.load(subject_output / "connectivity.npy").shape == (400, 3)
    assert np.load(subject_output / "fingerprint.npy").shape == (400, 2)
    np.testing.assert_array_equal(
        np.load(subject_output / "target_parcel_ids.npy"),
        np.load(subject_dir / "target_parcel_ids.npy"),
    )

    provenance = json.loads((subject_output / "provenance.json").read_text())
    required = {
        "model_fingerprints",
        "model_manifest_sha256",
        "model_artifact_fingerprints",
        "model_result_sha256",
        "bundle_provenance_sha256",
        "selection_sha256s",
        "stimulus_ids_sha256",
        "seed_available",
        "n_target_voxels",
        "accuracy_computed",
        "accuracy_reason",
        "unseen_rationale",
        "timestamps",
    }
    assert required.issubset(provenance)
    assert provenance["seed_available"] == {"n_available": 398, "n_total": 400}
    assert provenance["accuracy_computed"] is False
    assert provenance["accuracy_reason"] == "FOR has no task fMRI"
    assert provenance["unseen_rationale"] == (
        "selection excludes all subject 1-6 train/test stimuli"
    )
    batch = json.loads((output_root / "batch_manifest.json").read_text())
    assert batch["schema_version"] == 1
    assert batch["subjects_completed"] == ["sub-0001"]
    assert batch["n_target_voxels"] == {"sub-0001": 3}
    assert batch["prediction_shape"] == {"sub-0001": [5, 3]}


def test_matching_provenance_skips_and_force_reruns(tmp_path, monkeypatch):
    model_dir, selection_dir, contract_root, _ = _write_inputs(tmp_path)
    builder, _ = _patch_model(monkeypatch)
    output_root = tmp_path / "outputs"
    kwargs = {
        "model_dir": model_dir,
        "contract_root": contract_root,
        "selection_dir": selection_dir,
        "output_root": output_root,
        "subjects": ["sub-0001"],
        "device": "cpu",
    }

    first = for_inference.predict_for_subjects(**kwargs)
    second = for_inference.predict_for_subjects(**kwargs)
    forced = for_inference.predict_for_subjects(**kwargs, force=True)

    assert first["subjects"]["sub-0001"]["status"] == "completed"
    assert second["subjects"]["sub-0001"]["status"] == "skipped"
    assert forced["subjects"]["sub-0001"]["status"] == "completed"
    assert builder.calls == 2
