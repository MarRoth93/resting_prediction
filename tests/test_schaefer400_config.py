from copy import deepcopy

import pytest

from src.schaefer400_config import (
    load_schaefer400_config,
    resolve_schaefer400_roots,
)


def test_repository_schaefer400_config_is_strict_and_resolvable(tmp_path):
    config = load_schaefer400_config("config_schaefer400.yaml")
    assert config["atlas"]["n_parcels"] == 400
    assert config["atlas"]["n_networks"] == 7
    assert config["regions"] == {"grouping": "parcel", "n_groups": 400}
    assert config["parcel_seed_bank"]["missing_subject_policy"] == "zero_fill_and_mask"

    resolved = resolve_schaefer400_roots(
        config,
        data_root=tmp_path / "processed",
        raw_data_root=tmp_path / "raw",
        for_data_root=tmp_path / "for",
    )
    assert resolved["data_root"] == str((tmp_path / "processed").resolve())
    assert resolved["raw_data_root"] == str((tmp_path / "raw").resolve())
    assert resolved["for_data_root"] == str((tmp_path / "for").resolve())


def test_schaefer400_config_rejects_a_different_atlas(tmp_path):
    import yaml

    config = deepcopy(load_schaefer400_config("config_schaefer400.yaml"))
    config["atlas"]["n_networks"] = 17
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="7-network"):
        load_schaefer400_config(path)
