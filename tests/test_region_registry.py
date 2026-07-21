import json

import numpy as np
import pytest

from src.data.region_registry import (
    HCP_MMP_ATLAS_FILES,
    RegionRegistry,
    build_region_registry,
)


def _synthetic_region_inputs():
    shape = (2, 4, 1)
    masks = {
        1: np.ones(shape, dtype=bool),
        2: np.ones(shape, dtype=bool),
    }
    atlases = {
        1: {
            HCP_MMP_ATLAS_FILES[0]: np.array(
                [1, 1, 2, 0, 0, 0, 0, 0], dtype=np.int32
            ).reshape(shape),
            HCP_MMP_ATLAS_FILES[1]: np.array(
                [0, 0, 0, 3, 3, 4, 0, 0], dtype=np.int32
            ).reshape(shape),
        },
        2: {
            HCP_MMP_ATLAS_FILES[0]: np.array(
                [1, 1, 2, 2, 0, 0, 0, 0], dtype=np.int32
            ).reshape(shape),
            HCP_MMP_ATLAS_FILES[1]: np.array(
                [0, 0, 0, 0, 3, 3, 4, 4], dtype=np.int32
            ).reshape(shape),
        },
    }
    return atlases, masks


def test_registry_retains_only_regions_large_enough_in_every_training_subject():
    atlases, masks = _synthetic_region_inputs()
    registry = build_region_registry(
        [2, 1],
        atlases,
        masks,
        min_voxels_per_subject=2,
    )

    assert registry.training_subjects == (1, 2)
    assert [(region.atlas_file, region.label) for region in registry.regions] == [
        ("lh.HCP_MMP1.nii.gz", 1),
        ("rh.HCP_MMP1.nii.gz", 3),
    ]
    assert registry.group_names == (
        "lh.HCP_MMP1:label001",
        "rh.HCP_MMP1:label003",
        "fallback",
    )
    assert registry.fallback_index == 2

    groups = registry.group_indices(1, atlases[1], masks[1], require_registered=True)
    np.testing.assert_array_equal(groups, [0, 0, 2, 1, 1, 2, 2, 2])
    assert groups.shape == (int(masks[1].sum()),)


def test_registry_round_trip_and_training_fingerprints(tmp_path):
    atlases, masks = _synthetic_region_inputs()
    registry = build_region_registry(
        [1, 2],
        atlases,
        masks,
        min_voxels_per_subject=2,
    )
    path = registry.save(tmp_path / "regions.json")
    loaded = RegionRegistry.load(path)

    assert loaded.to_manifest() == registry.to_manifest()
    np.testing.assert_array_equal(
        loaded.group_indices(2, atlases[2], masks[2], require_registered=True),
        registry.group_indices(2, atlases[2], masks[2]),
    )

    changed = {name: atlas.copy() for name, atlas in atlases[1].items()}
    changed[HCP_MMP_ATLAS_FILES[0]].flat[-1] = 99
    with pytest.raises(ValueError, match="fingerprints"):
        loaded.group_indices(1, changed, masks[1], require_registered=True)


def test_registry_rejects_modified_manifest(tmp_path):
    atlases, masks = _synthetic_region_inputs()
    registry = build_region_registry(
        [1, 2],
        atlases,
        masks,
        min_voxels_per_subject=2,
    )
    path = registry.save(tmp_path / "regions.json")
    manifest = json.loads(path.read_text())
    manifest["min_voxels_per_subject"] = 3
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        RegionRegistry.load(path)


def test_unregistered_subject_maps_but_can_be_required():
    atlases, masks = _synthetic_region_inputs()
    registry = build_region_registry(
        [1, 2],
        atlases,
        masks,
        min_voxels_per_subject=2,
    )
    eval_atlases = {name: atlas.copy() for name, atlas in atlases[1].items()}
    groups = registry.group_indices(3, eval_atlases, masks[1])
    assert groups.shape == (8,)
    with pytest.raises(ValueError, match="not one of the registry training subjects"):
        registry.group_indices(3, eval_atlases, masks[1], require_registered=True)


def test_registry_rejects_overlapping_hemisphere_atlases():
    atlases, masks = _synthetic_region_inputs()
    atlases[1][HCP_MMP_ATLAS_FILES[1]].flat[0] = 3
    with pytest.raises(ValueError, match="overlap"):
        build_region_registry(
            [1, 2],
            atlases,
            masks,
            min_voxels_per_subject=2,
        )
