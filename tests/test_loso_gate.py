from copy import deepcopy

from src.multiexpert_config import canonical_config_hash, load_multiexpert_config
from src.pipelines.loso_multiexpert import (
    _frozen_baseline_config_fingerprint,
    compute_loso_gate,
)


def _results(*, delta=0.006, reliable_delta=-0.001, include_reliability=True):
    config = load_multiexpert_config()
    rows = []
    for subject in config["subjects"]["loso"]:
        for seed in config["evaluation"]["robustness_seeds"]:
            baseline = {"median_r": 0.10, "median_r_nc_ge_0_3": 0.12}
            candidate = {
                "median_r": 0.10 + delta,
                "median_r_nc_ge_0_3": 0.12 + reliable_delta,
            }
            if not include_reliability:
                baseline.pop("median_r_nc_ge_0_3")
                candidate.pop("median_r_nc_ge_0_3")
            rows.append(
                {
                    "heldout_subject": subject,
                    "seed": seed,
                    "modes": {
                        "zero_shot": {
                            "current_only": baseline,
                            "learned_fusion_dropout": candidate,
                        }
                    },
                }
            )
    return config, rows


def test_gate_passes_only_when_all_three_criteria_pass():
    config, results = _results()
    gate = compute_loso_gate(results, config)

    assert gate["passed"] is True
    assert gate["subject_wins"] == 6
    assert gate["mean_delta_median_r"] >= 0.005
    assert gate["mean_reliable_delta"] >= -0.002


def test_gate_fails_closed_without_reliability_or_complete_seeds():
    config, results = _results(include_reliability=False)
    gate = compute_loso_gate(results, config)
    assert gate["passed"] is False
    assert gate["criteria"]["reliability"] is False

    config, results = _results()
    incomplete = compute_loso_gate(results[:-1], config)
    assert incomplete["passed"] is False
    assert incomplete["complete"] is False


def test_gate_enforces_reliable_voxel_regression_limit():
    config, results = _results(reliable_delta=-0.0021)
    gate = compute_loso_gate(results, config)
    assert gate["passed"] is False
    assert gate["criteria"]["reliability"] is False


def test_gate_rejects_duplicate_or_unexpected_records():
    config, results = _results()
    import pytest

    with pytest.raises(ValueError, match="Duplicate"):
        compute_loso_gate(results + [results[0]], config)
    unexpected = dict(results[0])
    unexpected["seed"] = 999
    with pytest.raises(ValueError, match="Unexpected"):
        compute_loso_gate(results + [unexpected], config)


def test_gate_provenance_uses_fold_training_seed_registry_and_current_baseline():
    config, results = _results()
    for result in results:
        heldout = int(result["heldout_subject"])
        seed = int(result["seed"])
        effective = deepcopy(config)
        effective["random_seed"] = seed
        effective["subjects"]["seed_registry"] = [
            subject
            for subject in config["subjects"]["loso"]
            if int(subject) != heldout
        ]
        result["experiment_config_hash"] = canonical_config_hash(config)
        result["effective_config_hash"] = canonical_config_hash(effective)
        result["frozen_baseline_config_fingerprint"] = (
            _frozen_baseline_config_fingerprint("config.yaml", seed=seed)
        )
        result["input_data_fingerprint"] = f"inputs-{heldout}"
        result["fold_contract_hash"] = f"fold-{heldout}-{seed}"

    gate = compute_loso_gate(
        results,
        config,
        require_provenance=True,
        frozen_config_path="config.yaml",
    )
    assert gate["passed"] is True

    results[0]["frozen_baseline_config_fingerprint"] = "stale"
    import pytest

    with pytest.raises(ValueError, match="stale frozen baseline"):
        compute_loso_gate(
            results,
            config,
            require_provenance=True,
            frozen_config_path="config.yaml",
        )
