import json

import pytest

from src.multiexpert_config import load_multiexpert_config
from src.pipelines.loso_multiexpert import compute_loso_gate
from src.pipelines.predict_multiexpert import _require_gate_for_locked_subject


def _passing_gate(config):
    results = []
    for subject in config["subjects"]["loso"]:
        for seed in config["evaluation"]["robustness_seeds"]:
            results.append(
                {
                    "heldout_subject": subject,
                    "seed": seed,
                    "fold_contract_hash": f"sub{subject}-seed{seed}",
                    "modes": {
                        "zero_shot": {
                            "current_only": {
                                "median_r": 0.10,
                                "median_r_nc_ge_0_3": 0.12,
                            },
                            "learned_fusion_dropout": {
                                "median_r": 0.106,
                                "median_r_nc_ge_0_3": 0.119,
                            },
                        }
                    },
                }
            )
    gate = compute_loso_gate(results, config)
    assert gate["passed"] is True
    gate["artifact_verification_complete"] = True
    return gate


def test_subject7_requires_matching_complete_gate(tmp_path):
    config = load_multiexpert_config()
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(_passing_gate(config)))

    _require_gate_for_locked_subject(7, config, path)
    _require_gate_for_locked_subject(6, config, tmp_path / "missing.json")

    gate = json.loads(path.read_text())
    gate["criteria"]["reliability"] = False
    path.write_text(json.dumps(gate))
    with pytest.raises(PermissionError, match="criteria"):
        _require_gate_for_locked_subject(7, config, path)


def test_subject7_rejects_stale_config_gate(tmp_path):
    config = load_multiexpert_config()
    gate = _passing_gate(config)
    gate["config_hash"] = "stale"
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))

    with pytest.raises(PermissionError, match="different config"):
        _require_gate_for_locked_subject(7, config, path)
