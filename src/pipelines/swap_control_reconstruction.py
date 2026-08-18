"""Swap-control: how much do rest-derived transforms change reconstructed images?

The shared prediction Z is image-only; each subject's rest data contributes
only the transform (P, R). "Subject A rendered with subject B's rest" is
therefore identical to subject B's own predicted responses, so the swap
control reduces to: remove the VDVAE prior-sampling randomness (fixed torch
seed, identical batch layout per subject) and measure how similar subjects'
reconstructions are at each stage:

  1. standardized decoder inputs (72 parcel means)   - deterministic
  2. calibrated VDVAE latents (91,168-d)             - deterministic
  3. seeded pixels (fixed prior draws)               - this module's decode
  4. unseeded pixels (existing recon artifacts)      - reference

The gap between stage-3 similarity and 1.0 is the causal contribution of the
rest-derived transform (plus per-subject input standardization) to the images.
Diagnosis-blind: no clinical labels are read. Illustrative QC under D-02.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.config import load_config
from src.pipelines.benchmark_reconstructions_vdvae_vd import (
    _decode_vdvae_latents,
    _load_vdvae_model,
)
from src.pipelines.reconstruct_from_predictions import (
    N_PARCELS,
    N_PREDICTION_ROWS,
    SUBJECT_RE,
    _load_decoder,
    _predict_from_saved_family,
    _selection_context,
    _standardize_with_decoder,
    _write_json_atomic,
)
from src.pipelines.vdvae_calibration import apply_calibration


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config_voxel_contract.yaml"
DEFAULT_RECON_ROOT = "artifacts/recon_from_predictions/seed42"
DEFAULT_OUTPUT_ROOT = "artifacts/swap_control/seed42"
DEFAULT_SELECTION_DIR = (
    "artifacts/schaefer400_multiexpert/prediction_inputs/"
    "random_unseen_500_seed42"
)
DEFAULT_RECON_FEATURE_DIR = "data/processed/reconstruction_features/subj07"
DEFAULT_PRIOR_SEED = 1234
DEFAULT_N_FOR_SUBJECTS = 5
SEEDED_DIR_NAME = "images_vdvae_seeded"
PIXEL_SIZE = 64


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per row between two equally shaped 2D arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got {a.shape} vs {b.shape}.")
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Constant row encountered in correlation input.")
    return (a_centered * b_centered).sum(axis=1) / denominator


def constant_rows(matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of rows with (near-)zero variance."""
    values = np.asarray(matrix, dtype=np.float64)
    return values.std(axis=1) < 1e-8


def pairwise_summary(matrices: dict[str, np.ndarray]) -> dict:
    """Same-image and mismatched-image cross-subject correlations for all pairs."""
    pairs = {}
    same_values = []
    mismatched_values = []
    for subject_a, subject_b in combinations(sorted(matrices), 2):
        same = rowwise_pearson(matrices[subject_a], matrices[subject_b])
        mismatched = rowwise_pearson(
            matrices[subject_a],
            np.roll(matrices[subject_b], 1, axis=0),
        )
        pairs[f"{subject_a}|{subject_b}"] = {
            "same_image_mean": float(same.mean()),
            "mismatched_image_mean": float(mismatched.mean()),
        }
        same_values.append(same.mean())
        mismatched_values.append(mismatched.mean())
    return {
        "pairs": pairs,
        "same_image_mean": float(np.mean(same_values)),
        "mismatched_image_mean": float(np.mean(mismatched_values)),
    }


def default_subjects(recon_root: Path, n_for_subjects: int) -> list[str]:
    for_labels = sorted(
        path.name
        for path in recon_root.glob("sub-*")
        if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
    )
    if len(for_labels) < n_for_subjects:
        raise ValueError(
            f"Need {n_for_subjects} FOR subjects under {recon_root}, "
            f"found {len(for_labels)}."
        )
    return ["subj07", *for_labels[:n_for_subjects]]


def _parse_subjects(recon_root: Path, requested: Sequence[str] | None, n_for: int) -> list[str]:
    if not requested:
        return default_subjects(recon_root, n_for)
    labels = []
    for value in requested:
        labels.extend(part.strip() for part in str(value).split(",") if part.strip())
    invalid = [
        label
        for label in labels
        if label != "subj07" and SUBJECT_RE.fullmatch(label) is None
    ]
    if invalid:
        raise ValueError(f"Invalid subject labels: {invalid}")
    if len(set(labels)) != len(labels):
        raise ValueError("Subject labels must be unique.")
    if len(labels) < 2:
        raise ValueError("Swap control needs at least 2 subjects.")
    return labels


def _subject_latents(
    subject: str,
    *,
    recon_root: Path,
    decoder: dict,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized parcel inputs and calibrated VDVAE latents (deterministic)."""
    parcel_path = recon_root / subject / "parcel_responses.npy"
    if not parcel_path.is_file():
        raise FileNotFoundError(f"{subject}: missing parcel responses: {parcel_path}")
    parcel_responses = np.asarray(np.load(parcel_path), dtype=np.float32)
    if parcel_responses.shape != (N_PREDICTION_ROWS, N_PARCELS):
        raise ValueError(
            f"{subject}: parcel responses have shape {parcel_responses.shape}, "
            f"expected {(N_PREDICTION_ROWS, N_PARCELS)}."
        )
    standardized = _standardize_with_decoder(parcel_responses, decoder, subject=subject)
    latents = _predict_from_saved_family(
        family="vdvae",
        x=standardized,
        decoder=decoder,
        subject=subject,
        chunk_size=chunk_size,
    )
    latents = apply_calibration(latents, decoder["calibration"])
    return standardized, latents.reshape(N_PREDICTION_ROWS, -1)


def _seeded_image_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / SEEDED_DIR_NAME


def _image_pixels(image_dir: Path, stimulus_ids: np.ndarray) -> np.ndarray:
    """Load per-row images (row%03d or row%05d naming) as flattened 64x64 pixels."""
    pixels = np.empty(
        (stimulus_ids.size, PIXEL_SIZE * PIXEL_SIZE * 3), dtype=np.float64
    )
    for row, stimulus_id in enumerate(np.asarray(stimulus_ids).tolist()):
        candidates = [
            image_dir / f"row{row:03d}_stim{int(stimulus_id)}.png",
            image_dir / f"row{row:05d}_stim{int(stimulus_id)}.png",
        ]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Missing image row {row} (stim {stimulus_id}) in {image_dir}"
            )
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(
                (PIXEL_SIZE, PIXEL_SIZE), resample=Image.Resampling.BICUBIC
            )
            pixels[row] = np.asarray(resized, dtype=np.float64).ravel()
    return pixels


def _seeded_decode_complete(image_dir: Path, stimulus_ids: np.ndarray) -> bool:
    return all(
        (image_dir / f"row{row:05d}_stim{int(stimulus_id)}.png").is_file()
        for row, stimulus_id in enumerate(np.asarray(stimulus_ids).tolist())
    )


def decode_seeded(
    *,
    subjects: Sequence[str],
    recon_root: Path,
    output_root: Path,
    recon_feature_dir: Path,
    recon_model_root: Path,
    config: dict,
    device: str,
    prior_seed: int,
    force: bool,
) -> dict[str, str]:
    import torch

    decoder = _load_decoder(recon_root / "decoder")
    selection = _selection_context(
        Path(config["_selection_dir"])
    )
    stimulus_ids = selection["stimulus_ids"]
    reconstruction_config = config["reconstruction"]
    ref_path = recon_feature_dir / "ref_latents.npz"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing VDVAE reference latents: {ref_path}")
    ref_latent = np.load(ref_path, allow_pickle=True)["ref_latent"]

    ema_vae = None
    results = {}
    for subject in subjects:
        image_dir = _seeded_image_dir(output_root, subject)
        if not force and _seeded_decode_complete(image_dir, stimulus_ids):
            logger.info("Skipping completed seeded decode for %s", subject)
            results[subject] = "skipped"
            continue
        _, latents = _subject_latents(
            subject,
            recon_root=recon_root,
            decoder=decoder,
            chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
        )
        if ema_vae is None:
            ema_vae = _load_vdvae_model(recon_model_root)
        # Identical seed + identical batch layout per subject: every subject
        # consumes the same prior-sample sequence, so remaining image
        # differences are attributable to the latents alone.
        torch.manual_seed(int(prior_seed))
        torch.cuda.manual_seed_all(int(prior_seed))
        _decode_vdvae_latents(
            ema_vae=ema_vae,
            pred_latents=latents,
            ref_latent=ref_latent,
            out_dir=image_dir,
            save_rows=np.arange(N_PREDICTION_ROWS, dtype=np.int64),
            save_stim=stimulus_ids,
            batch_size=int(reconstruction_config["vdvae_batch_size"]),
            device=device,
        )
        if not _seeded_decode_complete(image_dir, stimulus_ids):
            raise RuntimeError(f"{subject}: seeded decode left missing images.")
        results[subject] = "completed"
    return results


def analyze(
    *,
    subjects: Sequence[str],
    recon_root: Path,
    output_root: Path,
    config: dict,
    prior_seed: int,
) -> dict:
    decoder = _load_decoder(recon_root / "decoder")
    selection = _selection_context(Path(config["_selection_dir"]))
    stimulus_ids = selection["stimulus_ids"]
    chunk_size = int(config["reconstruction"]["vdvae_chunk_size"])

    inputs = {}
    latents = {}
    seeded_pixels = {}
    unseeded_pixels = {}
    for subject in subjects:
        standardized, latent = _subject_latents(
            subject,
            recon_root=recon_root,
            decoder=decoder,
            chunk_size=chunk_size,
        )
        inputs[subject] = standardized
        latents[subject] = latent
        seeded_pixels[subject] = _image_pixels(
            _seeded_image_dir(output_root, subject), stimulus_ids
        )
        unseeded_pixels[subject] = _image_pixels(
            recon_root / subject / "images_vdvae", stimulus_ids
        )

    # VDVAE pixel sampling occasionally produces NaN -> all-zero images for a
    # few stimuli (invalid-cast warning in vae_helpers). Exclude the union of
    # degenerate rows across subjects and both pixel arms so every level is
    # compared on the identical image set.
    excluded = np.zeros(N_PREDICTION_ROWS, dtype=bool)
    for pixel_arm in (seeded_pixels, unseeded_pixels):
        for matrix in pixel_arm.values():
            excluded |= constant_rows(matrix)
    keep = ~excluded
    if keep.sum() < 2:
        raise RuntimeError("Fewer than 2 non-degenerate images remain.")
    for level in (inputs, latents, seeded_pixels, unseeded_pixels):
        for subject in level:
            level[subject] = level[subject][keep]

    summary = {
        "subjects": sorted(subjects),
        "n_images": int(keep.sum()),
        "n_excluded_degenerate": int(excluded.sum()),
        "excluded_rows": np.flatnonzero(excluded).tolist(),
        "prior_seed": int(prior_seed),
        "levels": {
            "decoder_inputs_72parcel": pairwise_summary(inputs),
            "calibrated_vdvae_latents": pairwise_summary(latents),
            "pixels_seeded_priors": pairwise_summary(seeded_pixels),
            "pixels_unseeded_reference": pairwise_summary(unseeded_pixels),
        },
        "interpretation": (
            "same_image_mean is cross-subject similarity for the SAME stimulus; "
            "1 - pixels_seeded same_image_mean bounds the rest-transform "
            "contribution to images (sampling noise removed). "
            "mismatched_image_mean is the different-stimulus baseline."
        ),
    }
    _write_json_atomic(output_root / "swap_control_summary.json", summary)
    _write_json_atomic(
        output_root / "README.json",
        {
            "purpose": "swap-control QC only",
            "interpretation": "ILLUSTRATIVE under decision D-02",
            "clinical_labels_loaded": False,
        },
    )

    print(
        f"Swap control ({len(subjects)} subjects, {int(keep.sum())} images, "
        f"{int(excluded.sum())} degenerate rows excluded)"
    )
    print(f"{'level':<32}{'same-image r':>14}{'mismatched r':>14}")
    for level, values in summary["levels"].items():
        print(
            f"{level:<32}{values['same_image_mean']:>14.4f}"
            f"{values['mismatched_image_mean']:>14.4f}"
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, choices=("decode", "analyze", "all"))
    parser.add_argument(
        "--subjects",
        nargs="*",
        help=(
            "Subject labels; default is subj07 plus the first "
            f"{DEFAULT_N_FOR_SUBJECTS} FOR subjects (sorted)."
        ),
    )
    parser.add_argument("--recon-root", default=DEFAULT_RECON_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-dir", default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--recon-feature-dir", default=DEFAULT_RECON_FEATURE_DIR)
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument("--prior-seed", type=int, default=DEFAULT_PRIOR_SEED)
    parser.add_argument("--n-for-subjects", type=int, default=DEFAULT_N_FOR_SUBJECTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    recon_root = Path(args.recon_root).resolve()
    output_root = Path(args.output_root).resolve()
    config = load_config(CONFIG_PATH)
    config["_selection_dir"] = str(Path(args.selection_dir).resolve())
    subjects = _parse_subjects(recon_root, args.subjects, int(args.n_for_subjects))
    logger.info("Swap-control subjects: %s", subjects)

    if args.command in {"decode", "all"}:
        decode_seeded(
            subjects=subjects,
            recon_root=recon_root,
            output_root=output_root,
            recon_feature_dir=Path(args.recon_feature_dir).resolve(),
            recon_model_root=Path(args.recon_model_root).resolve(),
            config=config,
            device=str(args.device),
            prior_seed=int(args.prior_seed),
            force=bool(args.force),
        )
    if args.command in {"analyze", "all"}:
        analyze(
            subjects=subjects,
            recon_root=recon_root,
            output_root=output_root,
            config=config,
            prior_seed=int(args.prior_seed),
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
