"""
Tests for atlas harmonization utilities.
"""

import json

import numpy as np
import pytest

from src.data.load_atlas import (
    atlas_utilization_summary,
    build_atlas_utilization_report,
    harmonize_atlas_labels,
)


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


def test_atlas_utilization_summary_reports_coverage_and_small_parcels():
    atlas_masked = np.array([1, 1, 0, 2, 2, 0, 3], dtype=np.int32)
    summary = atlas_utilization_summary(
        atlas_masked=atlas_masked,
        n_parcels=3,
        sub_id=7,
        min_voxels_per_parcel=2,
        min_labeled_fraction_warn=0.9,
    )

    assert summary["n_voxels_masked"] == 7
    assert summary["n_voxels_labeled"] == 5
    assert summary["labeled_fraction"] == pytest.approx(5 / 7)
    assert summary["n_parcels_present"] == 3
    assert summary["small_parcels"] == [3]
    assert any("only" in w for w in summary["warnings"])


def test_build_atlas_utilization_report_aggregates_subjects():
    atlas_by_subject = {
        1: np.array([1, 1, 2, 2, 0, 0], dtype=np.int32),
        2: np.array([1, 1, 2, 2, 1, 2], dtype=np.int32),
    }
    report = build_atlas_utilization_report(
        atlas_masked_by_subject=atlas_by_subject,
        n_parcels=2,
        min_voxels_per_parcel=2,
        min_labeled_fraction_warn=0.5,
    )

    assert report["n_subjects"] == 2
    assert report["n_parcels_expected"] == 2
    assert report["labeled_fraction_min"] == pytest.approx(4 / 6)
    assert report["parcels_present_min"] == 2
    assert report["subjects_with_warnings"] == []
    assert set(report["per_subject"]) == {"1", "2"}
