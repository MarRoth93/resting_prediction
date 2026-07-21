import json
from types import SimpleNamespace

import numpy as np

from src.alignment.external_seed_bank import SEED_SET, SeedDef
from src.pipelines.multiexpert_support import ExternalSeedSpec


def test_external_seed_spec_roundtrip_binds_registry_subjects(tmp_path):
    spec = ExternalSeedSpec(
        seed_set=SEED_SET,
        seed_defs=(
            SeedDef(
                seed_set=SEED_SET,
                atlas_file="lh.HCP_MMP1.nii.gz",
                label=1,
                name="left-one",
            ),
        ),
        coverage=({"subject": 2}, {"subject": 3}),
        rest_config={"zscore": True},
        min_voxels_per_seed=10,
        registry_subjects=(2, 3),
        missing_subject_policy="zero_fill_and_mask",
    )

    spec.save(tmp_path)
    loaded = ExternalSeedSpec.load(tmp_path)

    assert loaded == spec
    assert loaded.contract["registry_subjects"] == [2, 3]
    manifest = json.loads((tmp_path / "external_seed_info.json").read_text())
    assert manifest["registry_subjects"] == [2, 3]


def test_external_seed_spec_zero_fills_anatomically_missing_rows(
    monkeypatch,
    tmp_path,
):
    spec = ExternalSeedSpec(
        seed_set=SEED_SET,
        seed_defs=(
            SeedDef(SEED_SET, "atlas.nii.gz", 1, "present"),
            SeedDef(SEED_SET, "atlas.nii.gz", 2, "missing"),
        ),
        coverage=(),
        rest_config={"zscore": True},
        min_voxels_per_seed=1,
        registry_subjects=(1,),
        missing_subject_policy="zero_fill_and_mask",
    )
    subject = SimpleNamespace(
        sub=9,
        mask=np.ones((2, 2, 2), dtype=bool),
        rest_runs=[np.zeros((4, 3), dtype=np.float32)],
    )
    atlas = np.ones((2, 2, 2), dtype=np.int32)
    monkeypatch.setattr(
        "src.pipelines.multiexpert_support.load_atlas_array",
        lambda *args: atlas,
    )

    def fake_load(**kwargs):
        assert [seed.label for seed in kwargs["seed_defs"]] == [1]
        return [np.full((4, 1), 3.0, dtype=np.float32)]

    monkeypatch.setattr(
        "src.pipelines.multiexpert_support.load_or_prepare_external_seed_runs",
        fake_load,
    )
    availability = spec.availability_for_subject(subject, raw_data_root=str(tmp_path))
    runs = spec.runs_for_subject(
        subject,
        data_root=str(tmp_path),
        raw_data_root=str(tmp_path),
        allow_missing=True,
        availability=availability,
    )

    np.testing.assert_array_equal(availability, [True, False])
    np.testing.assert_array_equal(runs[0][:, 0], 3.0)
    np.testing.assert_array_equal(runs[0][:, 1], 0.0)
