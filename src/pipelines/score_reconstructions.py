"""Score final reconstruction directories against their source stimuli."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.data.shared_paths import default_stimuli_hdf5
from src.pipelines.vdvae_calibration_battery import (
    _clip_embeddings,
    bootstrap_ci,
    pixcorr,
    two_way_identification,
)

logger = logging.getLogger(__name__)

_RECONSTRUCTION_FILENAME = re.compile(r"^row\d+_stim(?P<stim_id>\d+)\.png$")
_CSV_COLUMNS = [
    "condition",
    "subdir",
    "n_images",
    "pixcorr_mean",
    "pixcorr_lo",
    "pixcorr_hi",
    "pixcorr_shuffled_mean",
    "clip_2way_id",
    "clip_2way_id_shuffled",
]


def _parse_stim_id(path: str | Path) -> int:
    name = Path(path).name
    match = _RECONSTRUCTION_FILENAME.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Invalid reconstruction filename {name!r}; expected rowNNNNN_stimID.png."
        )
    return int(match.group("stim_id"))


def _resized_pixels(images: list[Image.Image]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                image.resize((64, 64), resample=Image.Resampling.BICUBIC)
            )
            for image in images
        ]
    )


def score_reconstructions(
    recon_root: Path,
    subdir: str,
    conditions: list[str],
    stimuli_hdf5: Path,
    device: str,
    batch_size: int,
    output_csv: Path,
    clip_model=None,
    clip_preprocess=None,
) -> list[dict[str, str | int | float]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if subdir not in {"reconstructions", "reconstructions_vdvae"}:
        raise ValueError(f"Unknown reconstruction subdir: {subdir!r}.")

    condition_paths = []
    for condition in conditions:
        condition_dir = recon_root / subdir / condition
        if not condition_dir.is_dir():
            logger.warning("Missing condition directory; skipping %s", condition_dir)
            continue
        decoded_paths = sorted(condition_dir.glob("row*_stim*.png"))
        if len(decoded_paths) < 2:
            raise ValueError(
                f"At least two reconstruction images are required in {condition_dir}; "
                f"found {len(decoded_paths)}."
            )
        condition_paths.append((condition, condition_dir, decoded_paths))

    if not condition_paths:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
        logger.info("Saved reconstruction scores to %s", output_csv)
        return []

    if condition_paths and (clip_model is None or clip_preprocess is None):
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L/14",
            pretrained="openai",
        )
        clip_model = clip_model.to(device).eval()

    rows = []
    with h5py.File(stimuli_hdf5, "r") as stimuli_file:
        stimulus_images = stimuli_file["imgBrick"]
        for condition, condition_dir, decoded_paths in condition_paths:
            stim_ids = [_parse_stim_id(path) for path in decoded_paths]
            decoded_images = []
            for path in decoded_paths:
                with Image.open(path) as image:
                    decoded_images.append(image.convert("RGB"))
            originals = [
                Image.fromarray(stimulus_images[stim_id]).convert("RGB")
                for stim_id in stim_ids
            ]

            decoded_pixels = _resized_pixels(decoded_images)
            original_pixels = _resized_pixels(originals)
            pixcorr_values = pixcorr(decoded_pixels, original_pixels)
            pixcorr_lo, pixcorr_hi = bootstrap_ci(pixcorr_values)
            pixcorr_shuffled = pixcorr(
                decoded_pixels,
                np.roll(original_pixels, 1, axis=0),
            )

            decoded_embeddings = _clip_embeddings(
                decoded_images,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=batch_size,
                device=device,
            )
            original_embeddings = _clip_embeddings(
                originals,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=batch_size,
                device=device,
            )
            similarities = decoded_embeddings @ original_embeddings.T
            row = {
                "condition": condition,
                "subdir": subdir,
                "n_images": int(len(decoded_paths)),
                "pixcorr_mean": float(np.mean(pixcorr_values)),
                "pixcorr_lo": pixcorr_lo,
                "pixcorr_hi": pixcorr_hi,
                "pixcorr_shuffled_mean": float(np.mean(pixcorr_shuffled)),
                "clip_2way_id": two_way_identification(similarities),
                "clip_2way_id_shuffled": two_way_identification(
                    np.roll(similarities, 1, axis=0)
                ),
            }
            rows.append(row)
            logger.info(
                "%s: n=%d PixCorr=%.6f CLIP 2-way ID=%.6f",
                condition_dir,
                len(decoded_paths),
                row["pixcorr_mean"],
                row["clip_2way_id"],
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved reconstruction scores to %s", output_csv)
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Score final reconstruction directories with PixCorr and CLIP identification."
    )
    parser.add_argument(
        "--recon-root",
        default="artifacts/reconstructions/subj07",
    )
    parser.add_argument(
        "--subdir",
        choices=("reconstructions", "reconstructions_vdvae"),
        default="reconstructions",
    )
    parser.add_argument(
        "--conditions",
        default="gt_fmri,zero_shot,few_shot",
    )
    parser.add_argument("--stimuli-hdf5", default=default_stimuli_hdf5())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV. Defaults to <recon-root>/reconstruction_scores.csv.",
    )
    args = parser.parse_args()

    conditions = [
        condition.strip()
        for condition in str(args.conditions).split(",")
        if condition.strip()
    ]
    if not conditions:
        parser.error("--conditions must name at least one condition.")
    recon_root = Path(args.recon_root)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else recon_root / "reconstruction_scores.csv"
    )
    score_reconstructions(
        recon_root=recon_root,
        subdir=str(args.subdir),
        conditions=conditions,
        stimuli_hdf5=Path(args.stimuli_hdf5),
        device=str(args.device),
        batch_size=int(args.batch_size),
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
