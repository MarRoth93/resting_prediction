from copy import deepcopy

import pytest

from src.multiexpert_config import (
    EXPECTED_EXPERT_ORDER,
    canonical_config_hash,
    load_multiexpert_config,
    resolve_data_roots,
)


def test_multiexpert_config_is_stage1_and_deterministic():
    config = load_multiexpert_config("config_multiexpert.yaml")

    assert config["release"]["status"] == "experimental"
    assert config["experts"]["order"] == EXPECTED_EXPERT_ORDER
    assert config["subjects"]["locked_test"] == [7]
    assert 7 not in config["subjects"]["seed_registry"]
    assert canonical_config_hash(config) == canonical_config_hash(deepcopy(config))


def test_multiexpert_config_rejects_expert_reordering(tmp_path):
    config = load_multiexpert_config("config_multiexpert.yaml")
    config["experts"]["order"] = list(reversed(EXPECTED_EXPERT_ORDER))
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="order"):
        load_multiexpert_config(path)


def test_effective_config_hash_tracks_resolved_data_roots(tmp_path):
    config = load_multiexpert_config("config_multiexpert.yaml")
    first = resolve_data_roots(
        config,
        data_root=tmp_path / "first",
        raw_data_root=tmp_path / "raw",
    )
    second = resolve_data_roots(
        config,
        data_root=tmp_path / "second",
        raw_data_root=tmp_path / "raw",
    )

    assert first["data_root"] == str((tmp_path / "first").resolve())
    assert canonical_config_hash(first) != canonical_config_hash(second)
