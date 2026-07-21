import json

import pytest

from src.pipelines.loso_multiexpert import (
    _fold_artifact_fingerprints,
    _validate_completed_fold,
    collect_loso_results,
)
from src.pipelines.multiexpert_artifacts import json_fingerprint


def _completed_fold(tmp_path):
    fold = tmp_path / "seed42" / "fold_sub01"
    for name in (
        "model",
        "no_dropout_encoder",
        "current_baseline_encoder",
        "connectivity_srm_baseline_encoder",
        "predictions",
    ):
        directory = fold / name
        directory.mkdir(parents=True)
        (directory / "artifact.bin").write_bytes(name.encode())
    (fold / "no_dropout_effective_config.json").write_text("{}")
    input_manifest = {"schema_version": 1, "files": {}}
    input_fingerprint = json_fingerprint(input_manifest)
    input_manifest["fingerprint"] = input_fingerprint
    (fold / "input_manifest.json").write_text(json.dumps(input_manifest))
    base = {
        "experiment_config_hash": "experiment-v1",
        "effective_config_hash": "effective-v1",
        "frozen_baseline_config_fingerprint": "frozen-v1",
        "heldout_subject": 1,
        "seed": 42,
        "train_subjects": [2, 3, 4, 5, 6],
        "seed_manifest_fingerprint": "seeds-v1",
        "region_registry_fingerprint": "regions-v1",
        "expert_order": ["hybrid_cha", "connectivity_srm"],
        "input_data_fingerprint": input_fingerprint,
    }
    artifacts = _fold_artifact_fingerprints(fold)
    result = dict(base)
    result_payload_fingerprint = json_fingerprint(result)
    contract_hash = json_fingerprint(
        {
            "base": base,
            "artifacts": artifacts,
            "result_payload_fingerprint": result_payload_fingerprint,
        }
    )
    (fold / "fold_contract.json").write_text(
        json.dumps(
            {
                "base": base,
                "artifacts": artifacts,
                "result_payload_fingerprint": result_payload_fingerprint,
                "contract_hash": contract_hash,
            }
        )
    )
    result["fold_contract_hash"] = contract_hash
    result_path = fold / "result.json"
    result_path.write_text(json.dumps(result))
    return fold, result_path, base


def test_completed_fold_is_reused_only_when_artifacts_match(tmp_path):
    fold, result_path, base = _completed_fold(tmp_path)
    loaded = _validate_completed_fold(fold, result_path, base)
    assert loaded["heldout_subject"] == 1

    (fold / "model" / "artifact.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        _validate_completed_fold(fold, result_path, base)


def test_completed_fold_is_rejected_when_input_fingerprint_changes(tmp_path):
    fold, result_path, base = _completed_fold(tmp_path)
    changed = dict(base)
    changed["input_data_fingerprint"] = "new-inputs"

    with pytest.raises(ValueError, match="stale"):
        _validate_completed_fold(fold, result_path, changed)


def test_result_collection_rejects_another_experiment(tmp_path):
    _completed_fold(tmp_path)
    assert len(
        collect_loso_results(tmp_path, expected_config_hash="experiment-v1")
    ) == 1
    with pytest.raises(ValueError, match="another config"):
        collect_loso_results(tmp_path, expected_config_hash="experiment-v2")


def test_result_collection_rejects_contract_metadata_mismatch(tmp_path):
    _, result_path, _ = _completed_fold(tmp_path)
    result = json.loads(result_path.read_text())
    result["frozen_baseline_config_fingerprint"] = "different"
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="frozen_baseline_config_fingerprint"):
        collect_loso_results(tmp_path, expected_config_hash="experiment-v1")
