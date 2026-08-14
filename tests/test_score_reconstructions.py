import csv

import h5py
import numpy as np
import pytest
from PIL import Image

from src.pipelines import score_reconstructions as scoring


def test_parse_stim_id_from_reconstruction_filename():
    assert scoring._parse_stim_id("row00012_stim345.png") == 345

    with pytest.raises(ValueError, match="Invalid reconstruction filename"):
        scoring._parse_stim_id("stim345.png")


def test_shuffled_identity_similarity_is_well_below_paired():
    similarities = np.eye(12, dtype=np.float32)

    paired = scoring.two_way_identification(similarities)
    shuffled = scoring.two_way_identification(
        np.roll(similarities, 1, axis=0)
    )

    assert paired == pytest.approx(1.0)
    assert shuffled < 0.2


def test_score_reconstructions_writes_csv_with_stubbed_embeddings(
    tmp_path,
    monkeypatch,
):
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, size=(3, 8, 8, 3), dtype=np.uint8)
    stimuli_hdf5 = tmp_path / "stimuli.hdf5"
    with h5py.File(stimuli_hdf5, "w") as stimuli_file:
        stimuli_file.create_dataset("imgBrick", data=images)

    condition_dir = tmp_path / "reconstructions" / "gt_fmri"
    condition_dir.mkdir(parents=True)
    for row, image in enumerate(images):
        Image.fromarray(image).save(
            condition_dir / f"row{row:05d}_stim{row}.png"
        )

    def stub_embeddings(
        image_list,
        clip_model,
        clip_preprocess,
        batch_size,
        device,
    ):
        del clip_model, clip_preprocess, batch_size, device
        embeddings = np.stack(
            [np.asarray(image, dtype=np.float32).reshape(-1) for image in image_list]
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    monkeypatch.setattr(scoring, "_clip_embeddings", stub_embeddings)
    output_csv = tmp_path / "scores.csv"
    rows = scoring.score_reconstructions(
        recon_root=tmp_path,
        subdir="reconstructions",
        conditions=["gt_fmri"],
        stimuli_hdf5=stimuli_hdf5,
        device="cpu",
        batch_size=2,
        output_csv=output_csv,
        clip_model=object(),
        clip_preprocess=object(),
    )

    assert len(rows) == 1
    assert rows[0]["pixcorr_mean"] == pytest.approx(1.0)
    assert rows[0]["clip_2way_id"] == pytest.approx(1.0)
    assert rows[0]["clip_2way_id_shuffled"] < rows[0]["clip_2way_id"]

    with open(output_csv, newline="") as csv_file:
        saved_rows = list(csv.DictReader(csv_file))
    assert len(saved_rows) == 1
    assert saved_rows[0]["condition"] == "gt_fmri"
    assert saved_rows[0]["subdir"] == "reconstructions"
    assert saved_rows[0]["n_images"] == "3"
