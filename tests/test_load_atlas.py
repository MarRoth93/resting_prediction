"""
Tests for atlas harmonization utilities.
"""

import json

import numpy as np

from src.data.load_atlas import harmonize_atlas_labels


def test_harmonize_label_remap_json_serializable():
    atlas_maps = {
        1: np.array([[0, 1, 2], [2, 3, 3]], dtype=np.int32),
        2: np.array([[1, 2, 2], [0, 3, 4]], dtype=np.int32),
    }
    masks = {
        1: np.ones((2, 3), dtype=np.int32),
        2: np.ones((2, 3), dtype=np.int32),
    }

    harmonized, common_labels, label_remap = harmonize_atlas_labels(
        atlas_maps=atlas_maps,
        masks=masks,
        min_k=1,
        min_voxels_per_parcel=1,
    )

    assert common_labels == [1, 2, 3]
    assert all(isinstance(lbl, int) for lbl in common_labels)
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in label_remap.items())
    assert json.loads(json.dumps(label_remap)) == {"1": 1, "2": 2, "3": 3}
    assert set(harmonized.keys()) == {1, 2}
