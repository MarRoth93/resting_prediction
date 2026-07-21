import pytest

from src.multiexpert_config import load_multiexpert_config
from src.pipelines.multiexpert_artifacts import (
    build_model_manifest,
    cached_input_file_manifest,
    directory_file_fingerprints,
    load_and_validate_model_manifest,
    save_model_manifest,
)


def test_directory_fingerprints_cover_nested_artifact_files(tmp_path):
    root = tmp_path / "experts"
    (root / "hybrid").mkdir(parents=True)
    (root / "hybrid" / "state.npz").write_bytes(b"first")

    first = directory_file_fingerprints(root)
    assert set(first) == {"hybrid/state.npz"}

    (root / "hybrid" / "state.npz").write_bytes(b"changed")
    assert directory_file_fingerprints(root) != first


def test_cached_input_manifest_tracks_content_changes(tmp_path):
    source = tmp_path / "input.npy"
    source.write_bytes(b"first")
    cache = tmp_path / "hash-cache.json"

    first = cached_input_file_manifest({"input": source}, cache_path=cache)
    second = cached_input_file_manifest({"input": source}, cache_path=cache)
    assert second == first

    source.write_bytes(b"changed")
    third = cached_input_file_manifest({"input": source}, cache_path=cache)
    assert third["fingerprint"] != first["fingerprint"]


def _manifest(config):
    return build_model_manifest(
        config=config,
        expert_dims={"hybrid_cha": 100, "connectivity_srm": 100},
        train_subjects=[1, 2],
        region_manifest={"training_subjects": [1, 2], "groups": ["a", "fallback"]},
        seed_manifest={"n_seeds": 12},
        input_dim=768,
        feature_slices={"clip": (0, 768)},
        model_variant="learned_fusion_dropout",
    )


def test_manifest_roundtrip_and_fingerprint_checks(tmp_path):
    config = load_multiexpert_config()
    save_model_manifest(tmp_path, _manifest(config))
    loaded = load_and_validate_model_manifest(
        tmp_path,
        config=config,
        region_manifest={"training_subjects": [1, 2], "groups": ["a", "fallback"]},
        seed_manifest={"n_seeds": 12},
        expected_train_subjects=[1, 2],
    )
    assert loaded["expert_order"] == ["hybrid_cha", "connectivity_srm"]

    with pytest.raises(ValueError, match="Region"):
        load_and_validate_model_manifest(
            tmp_path,
            config=config,
            region_manifest={
                "training_subjects": [1, 2],
                "groups": ["different", "fallback"],
            },
        )


def test_manifest_rejects_dimension_reordering():
    config = load_multiexpert_config()
    with pytest.raises(ValueError, match="order"):
        build_model_manifest(
            config=config,
            expert_dims={"connectivity_srm": 100, "hybrid_cha": 100},
            train_subjects=[1],
            region_manifest={},
            seed_manifest={},
            input_dim=768,
            feature_slices={"clip": (0, 768)},
            model_variant="learned_fusion_dropout",
        )


def test_manifest_rejects_wrong_subjects_and_variant(tmp_path):
    config = load_multiexpert_config()
    save_model_manifest(tmp_path, _manifest(config))

    with pytest.raises(ValueError, match="training subjects"):
        load_and_validate_model_manifest(
            tmp_path,
            config=config,
            expected_train_subjects=[1],
        )
    with pytest.raises(ValueError, match="variant"):
        load_and_validate_model_manifest(
            tmp_path,
            config=config,
            expected_model_variant="learned_fusion_no_dropout",
        )
