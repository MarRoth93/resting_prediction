import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import src.pipelines.train_voxel_contract as train_voxel_contract
from src.pipelines.train_voxel_contract import (
    PARITY_REFERENCE,
    RIDGE_ALPHA_GRID,
    STAGE_A_MIN,
    _match_schaefer400_rest_runs,
    _parse_folds,
    _resume_or_reset_final,
    _resume_or_reset_fold,
    _training_shared_stimulus_intersection,
    decision_fields,
    fit_ridge_baseline,
    main,
    validate_config_parity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_drift_guard_names_drifted_key(tmp_path):
    frozen_path = REPO_ROOT / "config.yaml"
    candidate = yaml.safe_load(frozen_path.read_text())
    candidate["release"]["name"] = "voxel_contract_v1"
    candidate["voxel_contract"] = {
        "root": "data/processed_voxel_contract",
        "parcel_rest_root": "data/processed_schaefer400",
    }
    candidate["alignment"]["n_components"] += 1
    candidate_path = tmp_path / "drifted.yaml"
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    with pytest.raises(ValueError, match=r"alignment\.n_components"):
        validate_config_parity(candidate_path, frozen_path)


def test_folds_rejects_subject_outside_contract():
    with pytest.raises(Exception, match="within 1-6"):
        _parse_folds("1,7")


def test_final_rejects_nsd499_arm(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--arm", "nsd499", "--final"])
    assert exc_info.value.code == 2
    assert "--final requires --arm schaefer400" in capsys.readouterr().err


def test_final_rejects_explicit_folds(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--arm", "schaefer400", "--final", "--folds", "1"])
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--final" in error
    assert "--folds" in error
    assert "not allowed with argument" in error


def test_final_matching_artifact_refuses_rerun(tmp_path, monkeypatch):
    final_dir = tmp_path / "seed42"
    final_dir.mkdir()
    marker = final_dir / "existing"
    marker.write_text("complete\n")
    expected_identity = {
        "arm": "schaefer400",
        "seed": 42,
        "held_out": 7,
        "train_subjects": [1, 2, 3, 4, 5, 6],
        "subject7_usage": "evaluation_only_once_D08",
    }
    expected_fingerprints = {"config_sha256": "matching"}
    monkeypatch.setattr(
        train_voxel_contract,
        "_validate_completed_final",
        lambda *args, **kwargs: expected_identity,
    )

    with pytest.raises(FileExistsError, match="matching fingerprints"):
        _resume_or_reset_final(
            final_dir,
            expected_identity=expected_identity,
            expected_fingerprints=expected_fingerprints,
            force=False,
        )
    assert marker.is_file()


def test_run_alignment_mismatch_raises_with_names(tmp_path):
    voxel_dir = tmp_path / "processed" / "subj01"
    parcel_dir = tmp_path / "processed_schaefer400" / "subj01"
    voxel_dir.mkdir(parents=True)
    parcel_dir.mkdir(parents=True)
    np.save(voxel_dir / "rest_run1.npy", np.zeros((5, 8), dtype=np.float32))
    np.save(voxel_dir / "rest_run2.npy", np.zeros((6, 8), dtype=np.float32))
    np.save(parcel_dir / "rest_run1.npy", np.zeros((5, 400), dtype=np.float32))
    np.save(parcel_dir / "rest_run3.npy", np.zeros((6, 400), dtype=np.float32))

    with pytest.raises(ValueError) as exc_info:
        _match_schaefer400_rest_runs(
            1,
            voxel_subject_dir=voxel_dir,
            parcel_subject_dir=parcel_dir,
        )
    message = str(exc_info.value)
    assert "rest_run2.npy" in message
    assert "rest_run3.npy" in message

    (parcel_dir / "rest_run3.npy").unlink()
    np.save(parcel_dir / "rest_run2.npy", np.zeros((4, 400), dtype=np.float32))
    with pytest.raises(ValueError, match=r"REST TR mismatch for rest_run2\.npy"):
        _match_schaefer400_rest_runs(
            1,
            voxel_subject_dir=voxel_dir,
            parcel_subject_dir=parcel_dir,
        )


def test_fingerprint_mismatch_blocks_resume_without_force(tmp_path):
    fold_dir = tmp_path / "fold_sub01"
    fold_dir.mkdir()
    (fold_dir / "fold_result.json").write_text(
        json.dumps(
            {
                "arm": "schaefer400",
                "seed": 42,
                "held_out": 1,
                "train_subjects": [2, 3, 4, 5, 6],
                "input_fingerprints": {"config_sha256": "old"},
            }
        )
    )
    with pytest.raises(ValueError, match=r"config_sha256"):
        _resume_or_reset_fold(
            fold_dir,
            expected_identity={
                "arm": "schaefer400",
                "seed": 42,
                "held_out": 1,
                "train_subjects": [2, 3, 4, 5, 6],
            },
            expected_fingerprints={"config_sha256": "new"},
            force=False,
        )
    assert fold_dir.exists()


def test_ridge_alpha_selection_is_deterministic_under_fixed_seed():
    rng = np.random.RandomState(5)
    groups = np.repeat(np.arange(20), 3)
    X = rng.normal(size=(groups.size, 9)).astype(np.float32)
    weights = rng.normal(size=(9, 4)).astype(np.float32)
    Z = (X @ weights + 0.05 * rng.normal(size=(groups.size, 4))).astype(np.float32)

    first = fit_ridge_baseline(X, Z, groups, seed=42)
    second = fit_ridge_baseline(X, Z, groups, seed=42)

    assert first.alpha in RIDGE_ALPHA_GRID
    assert first.alpha == second.alpha
    assert first.validation_mse == pytest.approx(second.validation_mse, abs=0.0)
    np.testing.assert_array_equal(first.x_mean, second.x_mean)
    np.testing.assert_array_equal(first.x_std, second.x_std)
    np.testing.assert_allclose(first.model.coef_, second.model.coef_, rtol=0, atol=0)


def test_threshold_and_decision_logic_is_exact():
    assert decision_fields(
        "nsd499",
        42,
        PARITY_REFERENCE + 0.005,
        0.0,
    )["parity_ok"] is True
    assert decision_fields(
        "nsd499",
        42,
        PARITY_REFERENCE + 0.005001,
        0.0,
    )["parity_ok"] is False
    assert decision_fields(
        "schaefer400",
        42,
        STAGE_A_MIN,
        STAGE_A_MIN - 1e-6,
    )["stage_a_pass"] is True
    assert decision_fields(
        "schaefer400",
        42,
        STAGE_A_MIN,
        STAGE_A_MIN,
    )["stage_a_pass"] is False
    assert decision_fields("schaefer400", 43, 1.0, 0.0)["stage_a_pass"] is None


def test_held_out_isolation_assertion_rejects_intersection_input_leak():
    subject_one = SimpleNamespace(test_stim_idx=np.asarray([1, 2]))
    subject_two = SimpleNamespace(test_stim_idx=np.asarray([1, 2]))
    with pytest.raises(AssertionError, match="Held-out subject 2 leaked"):
        _training_shared_stimulus_intersection(
            {1: subject_one, 2: subject_two},
            train_subjects=[1],
            held_out=2,
        )
