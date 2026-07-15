import json

import numpy as np
import pytest

from src.data.prepare_rest_data import (
    build_rest_provenance,
    rest_preprocessing_hash,
    validate_rest_provenance,
)


def _config(enabled=True):
    return {
        "discard_initial_trs": 5,
        "detrend": True,
        "highpass_cutoff_hz": 0.01,
        "motion_censoring": {
            "enabled": True,
            "fd_threshold_mm": 0.5,
            "max_censored_fraction": 0.3,
            "strategy": "spike_regress_then_drop",
        },
        "nuisance_regression": {
            "enabled": enabled,
            "motion_model": "friston24",
            "standardize": True,
            "require_motion": True,
        },
        "zscore": True,
        "min_usable_trs": 100,
    }


def _write_manifest(tmp_path, config=None, mask=None, runs=None):
    config = config or _config()
    mask = np.ones((2, 2, 2), dtype=bool) if mask is None else mask
    runs = [np.zeros((4, int(mask.sum())), dtype=np.float32)] if runs is None else runs
    provenance = build_rest_provenance(
        sub=1,
        config=config,
        mask=mask,
        source_files=["run01.nii.gz"],
        rest_runs=runs,
    )
    subject_dir = tmp_path / "subj01"
    subject_dir.mkdir()
    with open(subject_dir / "rest_run_manifest.json", "w") as f:
        json.dump({"subject": 1, "rest_runs": ["run01.nii.gz"], "provenance": provenance}, f)
    return mask, runs, provenance


def test_rest_provenance_validates_matching_contract(tmp_path):
    mask, runs, expected = _write_manifest(tmp_path)

    actual = validate_rest_provenance(
        data_root=tmp_path,
        sub=1,
        expected_config=_config(),
        expected_mask=mask,
        reference_rest_runs=runs,
    )

    assert actual["provenance_hash"] == expected["provenance_hash"]


def test_rest_provenance_rejects_preprocessing_mismatch(tmp_path):
    mask, runs, _ = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="preprocessing provenance mismatch"):
        validate_rest_provenance(
            data_root=tmp_path,
            sub=1,
            expected_config=_config(enabled=False),
            expected_mask=mask,
            reference_rest_runs=runs,
        )


def test_rest_provenance_rejects_missing_block(tmp_path):
    subject_dir = tmp_path / "subj01"
    subject_dir.mkdir()
    with open(subject_dir / "rest_run_manifest.json", "w") as f:
        json.dump({"subject": 1, "rest_runs": ["run01.nii.gz"]}, f)

    with pytest.raises(ValueError, match="no provenance block"):
        validate_rest_provenance(
            data_root=tmp_path,
            sub=1,
            expected_config=_config(),
            expected_mask=np.ones((2, 2, 2), dtype=bool),
        )


def test_rest_hash_normalizes_omitted_defaults():
    explicit = _config(enabled=False)
    explicit["nuisance_regression"].update(
        {"motion_model": "friston24", "standardize": True, "require_motion": False}
    )
    minimal = {
        "motion_censoring": explicit["motion_censoring"],
        "nuisance_regression": {"enabled": False},
    }
    assert rest_preprocessing_hash(explicit) == rest_preprocessing_hash(minimal)
