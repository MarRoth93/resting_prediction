import copy

import numpy as np
import pytest

from src.pipelines.gate3a_cleanup_stress import (
    FLAG_FIRST_ORDER_MEANING,
    _match_minimal_rest_runs,
    _parse_folds,
    cleanup_decision_fields,
    cleanup_delta,
    derive_gate3a_schaefer400_config,
)


def _changed_paths(before, after, prefix=""):
    changed = set()
    if isinstance(before, dict) and isinstance(after, dict):
        for key in set(before) | set(after):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(path)
            else:
                changed.update(_changed_paths(before[key], after[key], path))
        return changed
    if before != after:
        changed.add(prefix)
    return changed


def test_derived_config_flips_only_motion_cleanup_fields():
    original = {
        "data_root": "data/original",
        "raw_data_root": "/raw",
        "rest_preprocessing": {
            "discard_initial_trs": 5,
            "detrend": True,
            "highpass_cutoff_hz": 0.01,
            "motion_censoring": {
                "enabled": True,
                "fd_threshold_mm": 0.5,
                "strategy": "spike_regress_then_drop",
            },
            "nuisance_regression": {
                "enabled": True,
                "motion_model": "friston24",
                "standardize": True,
                "require_motion": True,
            },
            "zscore": True,
            "min_usable_trs": 100,
        },
        "unrelated": {"keep": [1, 2, 3]},
    }
    untouched = copy.deepcopy(original)

    derived = derive_gate3a_schaefer400_config(original, "data/minclean")

    assert original == untouched
    assert _changed_paths(original, derived) == {
        "data_root",
        "rest_preprocessing.motion_censoring.enabled",
        "rest_preprocessing.nuisance_regression.enabled",
        "rest_preprocessing.nuisance_regression.require_motion",
    }
    assert derived["data_root"] == "data/minclean"
    assert derived["rest_preprocessing"]["motion_censoring"]["enabled"] is False
    assert derived["rest_preprocessing"]["nuisance_regression"]["enabled"] is False
    assert derived["rest_preprocessing"]["nuisance_regression"]["require_motion"] is False


def test_minimal_view_tr_mismatch_raises(tmp_path):
    voxel_dir = tmp_path / "voxel" / "subj01"
    parcel_dir = tmp_path / "parcel" / "subj01"
    voxel_dir.mkdir(parents=True)
    parcel_dir.mkdir(parents=True)
    np.save(voxel_dir / "rest_run1.npy", np.zeros((20, 8), dtype=np.float32))
    np.save(parcel_dir / "rest_run1.npy", np.zeros((19, 400), dtype=np.float32))

    with pytest.raises(ValueError, match=r"MINIMAL REST TR mismatch for rest_run1\.npy"):
        _match_minimal_rest_runs(
            1,
            voxel_subject_dir=voxel_dir,
            parcel_subject_dir=parcel_dir,
        )


def test_cleanup_delta_and_flag_logic_are_exact():
    assert cleanup_delta(0.17, 0.18) == pytest.approx(-0.01)
    at_threshold = cleanup_decision_fields([-0.01])
    assert at_threshold == {
        "mean_delta": -0.01,
        "flag_first_order": False,
        "flag_first_order_meaning": FLAG_FIRST_ORDER_MEANING,
    }
    below_threshold = cleanup_decision_fields([-0.02, -0.01])
    assert below_threshold["mean_delta"] == pytest.approx(-0.015)
    assert below_threshold["flag_first_order"] is True


def test_folds_reject_subject_seven():
    with pytest.raises(Exception, match="within 1-6"):
        _parse_folds("1,7")
