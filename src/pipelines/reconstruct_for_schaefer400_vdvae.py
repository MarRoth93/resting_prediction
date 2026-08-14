"""Reconstruct FOR Schaefer-400 predictions with the retained VDVAE decoder.

The existing reconstruction benchmark learns a subject-specific voxel decoder.
FOR has no task-fMRI training split, so this path learns one canonical
Schaefer-400 decoder from the six NSD subjects' shared task images, applies it
to each FOR prediction batch, and reuses the existing VDVAE image decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.linalg import cho_factor, cho_solve

from src.pipelines.benchmark_reconstructions_vdvae_vd import (
    _latent_transformation,
    _load_vdvae_model,
)
from src.pipelines.multiexpert_artifacts import file_sha256


logger = logging.getLogger(__name__)
N_PARCELS = 400
ARTIFACT_VERSION = 1


def _array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("utf-8"))
    digest.update(json.dumps(values.shape).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def standardize_parcel_patterns(
    values: np.ndarray,
    *,
    available_parcels: np.ndarray | None = None,
) -> np.ndarray:
    """Standardize each available parcel across images; missing parcels become zero."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_PARCELS or values.shape[0] < 2:
        raise ValueError(f"Expected at least two rows by 400 parcels, got {values.shape}.")
    if available_parcels is None:
        available = np.ones(N_PARCELS, dtype=bool)
    else:
        available = np.asarray(available_parcels, dtype=bool)
        if available.shape != (N_PARCELS,):
            raise ValueError("available_parcels must have shape (400,).")
    if not np.any(available):
        raise ValueError("At least one parcel must be available.")
    selected = values[:, available]
    if not np.all(np.isfinite(selected)):
        raise ValueError("Available parcel predictions contain NaN/Inf.")
    mean = selected.mean(axis=0, keepdims=True)
    std = selected.std(axis=0, ddof=1, keepdims=True)
    std[std < 1e-8] = 1.0
    standardized = np.zeros(values.shape, dtype=np.float32)
    standardized[:, available] = (selected - mean) / std
    return standardized


def _load_target_stimulus_ids(feature_dir: Path) -> np.ndarray:
    vdvae_path = feature_dir / "vdvae_features.npz"
    if not vdvae_path.exists():
        raise FileNotFoundError(f"Missing VDVAE feature bundle: {vdvae_path}")
    with np.load(vdvae_path) as bundle:
        if "test_stim_idx" not in bundle.files:
            raise ValueError("VDVAE feature bundle has no test_stim_idx array.")
        stimulus_ids = np.asarray(bundle["test_stim_idx"], dtype=np.int64)
    if stimulus_ids.ndim != 1 or np.unique(stimulus_ids).size != stimulus_ids.size:
        raise ValueError("VDVAE test stimulus IDs must be a unique vector.")
    return stimulus_ids


def build_pooled_nsd_design(
    *,
    nsd_data_root: Path,
    reconstruction_feature_dir: Path,
    subjects: Iterable[int] = range(1, 7),
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Pool standardized NSD parcel responses aligned to existing VDVAE targets."""
    target_ids = _load_target_stimulus_ids(reconstruction_feature_dir)
    target_row = {int(stimulus): row for row, stimulus in enumerate(target_ids.tolist())}
    design_parts: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    rows_by_subject: dict[str, int] = {}
    for raw_subject in subjects:
        subject = int(raw_subject)
        subject_dir = nsd_data_root / f"subj{subject:02d}"
        responses = np.load(subject_dir / "test_fmri.npy", mmap_mode="r")
        stimulus_ids = np.load(subject_dir / "test_stim_idx.npy")
        if responses.shape != (stimulus_ids.size, N_PARCELS):
            raise ValueError(
                f"Subject {subject}: test response/ID shape mismatch: "
                f"{responses.shape} vs {stimulus_ids.shape}."
            )
        mapped = np.asarray([target_row.get(int(value), -1) for value in stimulus_ids], dtype=np.int64)
        keep = mapped >= 0
        if int(keep.sum()) < 2:
            raise ValueError(f"Subject {subject}: fewer than two VDVAE-aligned task images.")
        aligned = np.asarray(responses[keep], dtype=np.float32)
        design_parts.append(standardize_parcel_patterns(aligned))
        target_rows.append(mapped[keep])
        rows_by_subject[str(subject)] = int(keep.sum())
    return (
        np.concatenate(design_parts, axis=0),
        np.concatenate(target_rows, axis=0),
        rows_by_subject,
    )


def fit_chunked_ridge(
    *,
    design: np.ndarray,
    targets: np.ndarray,
    target_rows: np.ndarray,
    alpha: float,
    chunk_size: int,
    coefficient_path: Path,
    intercept_path: Path,
) -> None:
    """Fit one ridge system and stream its many VDVAE output dimensions."""
    design = np.asarray(design, dtype=np.float64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    if design.ndim != 2 or design.shape[1] != N_PARCELS:
        raise ValueError(f"Decoder design must be N x 400, got {design.shape}.")
    if target_rows.shape != (design.shape[0],):
        raise ValueError("target_rows must map every decoder design row.")
    if targets.ndim != 2 or np.any(target_rows < 0) or np.any(target_rows >= targets.shape[0]):
        raise ValueError("VDVAE targets or target row mapping are invalid.")
    if float(alpha) <= 0 or int(chunk_size) < 1:
        raise ValueError("Ridge alpha and chunk size must be positive.")

    x_mean = design.mean(axis=0)
    centered = design - x_mean
    gram = centered.T @ centered
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    factor = cho_factor(gram, lower=True, check_finite=False)

    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_coefficient = coefficient_path.with_name(f".{coefficient_path.name}.part")
    temporary_intercept = intercept_path.with_name(f".{intercept_path.name}.part")
    temporary_coefficient.unlink(missing_ok=True)
    temporary_intercept.unlink(missing_ok=True)
    coefficients = np.lib.format.open_memmap(
        temporary_coefficient,
        mode="w+",
        dtype=np.float32,
        shape=(N_PARCELS, int(targets.shape[1])),
    )
    intercept = np.lib.format.open_memmap(
        temporary_intercept,
        mode="w+",
        dtype=np.float32,
        shape=(int(targets.shape[1]),),
    )
    try:
        for start in range(0, int(targets.shape[1]), int(chunk_size)):
            stop = min(start + int(chunk_size), int(targets.shape[1]))
            y = np.asarray(targets[target_rows, start:stop], dtype=np.float64)
            y_mean = y.mean(axis=0)
            weights = cho_solve(
                factor,
                centered.T @ (y - y_mean),
                check_finite=False,
            )
            coefficients[:, start:stop] = weights.astype(np.float32)
            intercept[start:stop] = (y_mean - x_mean @ weights).astype(np.float32)
            logger.info("VDVAE ridge dimensions %d:%d/%d", start, stop, targets.shape[1])
        coefficients.flush()
        intercept.flush()
        del coefficients, intercept
        os.replace(temporary_coefficient, coefficient_path)
        os.replace(temporary_intercept, intercept_path)
    except Exception:
        temporary_coefficient.unlink(missing_ok=True)
        temporary_intercept.unlink(missing_ok=True)
        raise


def _decoder_paths(decoder_dir: Path) -> tuple[Path, Path, Path]:
    return (
        decoder_dir / "vdvae_coefficients.npy",
        decoder_dir / "vdvae_intercept.npy",
        decoder_dir / "decoder_manifest.json",
    )


def validate_decoder(decoder_dir: Path) -> dict:
    coefficient_path, intercept_path, manifest_path = _decoder_paths(decoder_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Decoder is not fitted: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("artifact_version", -1)) != ARTIFACT_VERSION:
        raise ValueError("Unsupported FOR VDVAE decoder artifact version.")
    coefficients = np.load(coefficient_path, mmap_mode="r")
    intercept = np.load(intercept_path, mmap_mode="r")
    expected_dim = int(manifest["vdvae_latent_dim"])
    if coefficients.shape != (N_PARCELS, expected_dim) or intercept.shape != (expected_dim,):
        raise ValueError("VDVAE decoder coefficient shapes do not match its manifest.")
    if manifest.get("coefficient_sha256") != file_sha256(coefficient_path):
        raise ValueError("VDVAE decoder coefficient checksum is invalid.")
    if manifest.get("intercept_sha256") != file_sha256(intercept_path):
        raise ValueError("VDVAE decoder intercept checksum is invalid.")
    return manifest


def fit_decoder(
    *,
    nsd_data_root: Path,
    reconstruction_feature_dir: Path,
    decoder_dir: Path,
    alpha: float,
    chunk_size: int,
) -> dict:
    coefficient_path, intercept_path, manifest_path = _decoder_paths(decoder_dir)
    if manifest_path.exists():
        logger.info("Using existing validated VDVAE decoder: %s", decoder_dir)
        manifest = validate_decoder(decoder_dir)
        if not np.isclose(float(manifest["ridge_alpha"]), float(alpha), rtol=0, atol=1e-12):
            raise RuntimeError(
                "Existing VDVAE decoder alpha does not match the validated selection: "
                f"stored={manifest['ridge_alpha']}, requested={alpha}."
            )
        return manifest
    stale = [path for path in (coefficient_path, intercept_path) if path.exists()]
    if stale:
        raise RuntimeError(
            "Partial decoder outputs exist without a manifest; move or remove them before retrying: "
            f"{[str(path) for path in stale]}"
        )

    design, target_rows, rows_by_subject = build_pooled_nsd_design(
        nsd_data_root=nsd_data_root,
        reconstruction_feature_dir=reconstruction_feature_dir,
    )
    vdvae_path = reconstruction_feature_dir / "vdvae_features.npz"
    with np.load(vdvae_path) as bundle:
        targets = np.asarray(bundle["test_latents"], dtype=np.float32)
        fit_chunked_ridge(
            design=design,
            targets=targets,
            target_rows=target_rows,
            alpha=alpha,
            chunk_size=chunk_size,
            coefficient_path=coefficient_path,
            intercept_path=intercept_path,
        )
        latent_dim = int(targets.shape[1])

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "representation": "subject_standardized_schaefer400_parcel_patterns",
        "training_subjects": [1, 2, 3, 4, 5, 6],
        "training_rows": int(design.shape[0]),
        "training_rows_by_subject": rows_by_subject,
        "ridge_alpha": float(alpha),
        "vdvae_latent_dim": latent_dim,
        "vdvae_feature_bundle": str(vdvae_path.resolve()),
        "vdvae_feature_bundle_sha256": file_sha256(vdvae_path),
        "coefficient_file": str(coefficient_path.resolve()),
        "coefficient_sha256": file_sha256(coefficient_path),
        "intercept_file": str(intercept_path.resolve()),
        "intercept_sha256": file_sha256(intercept_path),
        "missing_for_parcel_policy": "zero_after_available_parcel_standardization",
    }
    decoder_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _subject_labels(prediction_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        labels = [str(value) for value in requested]
    else:
        labels = sorted(path.name for path in prediction_root.glob("sub-*") if path.is_dir())
    if not labels:
        raise ValueError(f"No FOR subject directories found under {prediction_root}.")
    if requested is None and len(labels) != 50:
        raise ValueError(
            f"Expected the complete 50-subject FOR batch, found {len(labels)} subjects."
        )
    missing = [label for label in labels if not (prediction_root / label).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing requested FOR prediction directories: {missing}")
    return labels


def predict_vdvae_latents(
    *,
    subject_label: str,
    prediction_root: Path,
    output_root: Path,
    decoder_dir: Path,
    condition: str,
    stimulus_ids: np.ndarray,
    prediction_chunk_size: int,
) -> Path:
    decoder_manifest = validate_decoder(decoder_dir)
    coefficient_path, intercept_path, _ = _decoder_paths(decoder_dir)
    subject_prediction = prediction_root / subject_label
    activation_path = subject_prediction / f"{condition}.npy"
    available_path = subject_prediction / "available_parcels.npy"
    activations = np.load(activation_path, mmap_mode="r")
    available = np.load(available_path)
    if activations.shape != (stimulus_ids.size, N_PARCELS):
        raise ValueError(
            f"{subject_label}: expected {(stimulus_ids.size, N_PARCELS)} activations, "
            f"got {activations.shape}."
        )
    design = standardize_parcel_patterns(activations, available_parcels=available)
    coefficients = np.load(coefficient_path, mmap_mode="r")
    intercept = np.load(intercept_path, mmap_mode="r")

    destination = output_root / subject_label
    destination.mkdir(parents=True, exist_ok=True)
    latent_path = destination / "predicted_vdvae_latents.npy"
    provenance_path = destination / "latent_provenance.json"
    activation_sha256 = file_sha256(activation_path)
    available_sha256 = file_sha256(available_path)
    stimulus_ids_sha256 = _array_sha256(stimulus_ids)
    if latent_path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("condition") == condition
            and provenance.get("activation_sha256") == activation_sha256
            and provenance.get("available_parcels_sha256") == available_sha256
            and provenance.get("stimulus_ids_sha256") == stimulus_ids_sha256
            and provenance.get("decoder_coefficient_sha256")
            == decoder_manifest["coefficient_sha256"]
        ):
            existing = np.load(latent_path, mmap_mode="r")
            if (
                existing.shape
                == (stimulus_ids.size, int(decoder_manifest["vdvae_latent_dim"]))
                and provenance.get("latent_sha256") == file_sha256(latent_path)
            ):
                logger.info("Reusing predicted VDVAE latents for %s", subject_label)
                return latent_path
        raise RuntimeError(f"Stale VDVAE latent output exists for {subject_label}: {destination}")
    if latent_path.exists() != provenance_path.exists():
        raise RuntimeError(f"Partial VDVAE latent output exists for {subject_label}: {destination}")

    temporary = latent_path.with_name(f".{latent_path.name}.part")
    temporary.unlink(missing_ok=True)
    predicted = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(stimulus_ids.size, int(decoder_manifest["vdvae_latent_dim"])),
    )
    try:
        for start in range(0, predicted.shape[1], int(prediction_chunk_size)):
            stop = min(start + int(prediction_chunk_size), predicted.shape[1])
            predicted[:, start:stop] = (
                design @ np.asarray(coefficients[:, start:stop], dtype=np.float32)
                + np.asarray(intercept[start:stop], dtype=np.float32)
            )
        predicted.flush()
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"Predicted VDVAE latents contain NaN/Inf for {subject_label}.")
        del predicted
        os.replace(temporary, latent_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    provenance = {
        "artifact_version": ARTIFACT_VERSION,
        "subject": subject_label,
        "condition": condition,
        "prediction_rows": int(stimulus_ids.size),
        "activation_file": str(activation_path.resolve()),
        "activation_sha256": activation_sha256,
        "available_parcels_file": str(available_path.resolve()),
        "available_parcels_sha256": available_sha256,
        "available_parcels": int(np.asarray(available, dtype=bool).sum()),
        "stimulus_ids_sha256": stimulus_ids_sha256,
        "decoder_dir": str(decoder_dir.resolve()),
        "decoder_coefficient_sha256": decoder_manifest["coefficient_sha256"],
        "latent_file": str(latent_path.resolve()),
        "latent_sha256": file_sha256(latent_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return latent_path


def decode_vdvae_images(
    *,
    ema_vae,
    latent_path: Path,
    ref_latent,
    subject_label: str,
    output_root: Path,
    stimulus_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> int:
    import torch

    latents = np.load(latent_path, mmap_mode="r")
    if latents.shape[0] != stimulus_ids.size:
        raise ValueError(f"{subject_label}: latent/stimulus row mismatch.")
    image_dir = output_root / subject_label / "images_vdvae"
    image_dir.mkdir(parents=True, exist_ok=True)
    filenames = [
        f"row{row:05d}_stim{int(stimulus):05d}.png"
        for row, stimulus in enumerate(stimulus_ids)
    ]
    missing = np.asarray(
        [row for row, filename in enumerate(filenames) if not (image_dir / filename).exists()],
        dtype=np.int64,
    )
    if missing.size == 0:
        logger.info("All VDVAE images already exist for %s", subject_label)
        return 0
    hierarchical = _latent_transformation(latents, ref_latent)
    for start in range(0, missing.size, int(batch_size)):
        rows = missing[start : start + int(batch_size)]
        sample_latents = [
            torch.as_tensor(layer[rows], dtype=torch.float32, device=device)
            for layer in hierarchical
        ]
        with torch.no_grad():
            px_z = ema_vae.decoder.forward_manual_latents(len(rows), sample_latents, t=None)
            generated = ema_vae.decoder.out_net.sample(px_z)
        for row, array in zip(rows.tolist(), generated):
            destination = image_dir / filenames[int(row)]
            temporary = destination.with_name(f".{destination.name}.part")
            Image.fromarray(array).resize(
                (512, 512), resample=Image.Resampling.BICUBIC
            ).save(temporary, format="PNG")
            os.replace(temporary, destination)
        logger.info("%s: decoded %d/%d missing images", subject_label, min(start + len(rows), missing.size), missing.size)
    return int(missing.size)


def check_inputs(
    *,
    prediction_root: Path,
    selection_dir: Path,
    nsd_data_root: Path,
    reconstruction_feature_dir: Path,
    recon_model_root: Path,
    decoder_dir: Path,
    subjects: list[str] | None,
    condition: str,
) -> dict:
    stimulus_path = selection_dir / "nsd_stimulus_ids.npy"
    feature_path = selection_dir / "clip_features.npy"
    stimulus_ids = np.load(stimulus_path)
    if stimulus_ids.ndim != 1 or stimulus_ids.size != 500 or np.unique(stimulus_ids).size != 500:
        raise ValueError("Expected exactly 500 unique selected NSD stimulus IDs.")
    features = np.load(feature_path, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != stimulus_ids.size:
        raise ValueError("Selected CLIP features do not match the 500 stimulus IDs.")
    feature_sha256 = file_sha256(feature_path)
    expected_parcel_ids = np.arange(1, N_PARCELS + 1, dtype=np.int64)
    labels = _subject_labels(prediction_root, subjects)
    for subject in labels:
        root = prediction_root / subject
        activations = np.load(root / f"{condition}.npy", mmap_mode="r")
        available = np.load(root / "available_parcels.npy")
        parcel_ids = np.load(root / "parcel_ids.npy")
        provenance = json.loads((root / "provenance.json").read_text())
        if activations.shape != (stimulus_ids.size, N_PARCELS) or available.shape != (N_PARCELS,):
            raise ValueError(f"Invalid FOR prediction contract for {subject}.")
        if not np.array_equal(parcel_ids, expected_parcel_ids):
            raise ValueError(f"Unexpected Schaefer parcel order for {subject}.")
        if provenance.get("features_sha256") != feature_sha256:
            raise ValueError(f"Prediction feature provenance does not match for {subject}.")
        if not np.all(np.isfinite(activations[:, np.asarray(available, dtype=bool)])):
            raise ValueError(f"Available FOR predictions contain NaN/Inf for {subject}.")
    design, target_rows, rows_by_subject = build_pooled_nsd_design(
        nsd_data_root=nsd_data_root,
        reconstruction_feature_dir=reconstruction_feature_dir,
    )
    ref_path = reconstruction_feature_dir / "ref_latents.npz"
    checkpoint = recon_model_root / "vdvae" / "model" / "imagenet64-iter-1600000-model-ema.th"
    for path in (ref_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    decoder_status = "ready" if (decoder_dir / "decoder_manifest.json").exists() else "not_fitted"
    if decoder_status == "ready":
        validate_decoder(decoder_dir)
    return {
        "status": "ok",
        "for_subjects": len(labels),
        "prediction_rows": int(stimulus_ids.size),
        "prediction_feature_sha256": feature_sha256,
        "pooled_decoder_rows": int(design.shape[0]),
        "pooled_target_rows": int(target_rows.size),
        "training_rows_by_subject": rows_by_subject,
        "decoder_status": decoder_status,
        "manual_start_command": "./run_schaefer400_vdvae.sh all",
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "fit-decoder", "reconstruct", "all"])
    parser.add_argument(
        "--prediction-root",
        default="artifacts/schaefer400_multiexpert/for_predictions/random_unseen_500_seed42",
    )
    parser.add_argument(
        "--selection-dir",
        default="artifacts/schaefer400_multiexpert/prediction_inputs/random_unseen_500_seed42",
    )
    parser.add_argument("--nsd-data-root", default="data/processed_schaefer400")
    parser.add_argument(
        "--reconstruction-feature-dir",
        default="data/processed/reconstruction_features/subj07",
    )
    parser.add_argument(
        "--decoder-dir",
        default="artifacts/schaefer400_multiexpert/vdvae_decoder",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/schaefer400_multiexpert/reconstructions_vdvae/random_unseen_500_seed42",
    )
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument("--condition", default="learned_fusion")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--alpha", type=float, default=50000.0)
    parser.add_argument("--ridge-chunk-size", type=int, default=2048)
    parser.add_argument("--prediction-chunk-size", type=int, default=2048)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root).resolve()
    selection_dir = Path(args.selection_dir).resolve()
    nsd_data_root = Path(args.nsd_data_root).resolve()
    feature_dir = Path(args.reconstruction_feature_dir).resolve()
    decoder_dir = Path(args.decoder_dir).resolve()
    output_root = Path(args.output_root).resolve()
    recon_model_root = Path(args.recon_model_root).resolve()

    if args.command in {"reconstruct", "all"}:
        if args.device != "cuda":
            raise ValueError(
                "The retained VDVAE implementation is CUDA-only; use --device cuda."
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the retained VDVAE decoder.")

    if args.command == "check":
        report = check_inputs(
            prediction_root=prediction_root,
            selection_dir=selection_dir,
            nsd_data_root=nsd_data_root,
            reconstruction_feature_dir=feature_dir,
            recon_model_root=recon_model_root,
            decoder_dir=decoder_dir,
            subjects=args.subjects,
            condition=args.condition,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command in {"reconstruct", "all"}:
        check_inputs(
            prediction_root=prediction_root,
            selection_dir=selection_dir,
            nsd_data_root=nsd_data_root,
            reconstruction_feature_dir=feature_dir,
            recon_model_root=recon_model_root,
            decoder_dir=decoder_dir,
            subjects=args.subjects,
            condition=args.condition,
        )

    if args.command in {"fit-decoder", "all"}:
        fit_decoder(
            nsd_data_root=nsd_data_root,
            reconstruction_feature_dir=feature_dir,
            decoder_dir=decoder_dir,
            alpha=args.alpha,
            chunk_size=args.ridge_chunk_size,
        )
        if args.command == "fit-decoder":
            return

    if args.command in {"reconstruct", "all"}:
        validate_decoder(decoder_dir)
        labels = _subject_labels(prediction_root, args.subjects)
        stimulus_ids = np.load(selection_dir / "nsd_stimulus_ids.npy")
        ref_latent = np.load(feature_dir / "ref_latents.npz", allow_pickle=True)["ref_latent"]
        ema_vae = _load_vdvae_model(recon_model_root)
        for subject in labels:
            latent_path = predict_vdvae_latents(
                subject_label=subject,
                prediction_root=prediction_root,
                output_root=output_root,
                decoder_dir=decoder_dir,
                condition=args.condition,
                stimulus_ids=stimulus_ids,
                prediction_chunk_size=args.prediction_chunk_size,
            )
            decode_vdvae_images(
                ema_vae=ema_vae,
                latent_path=latent_path,
                ref_latent=ref_latent,
                subject_label=subject,
                output_root=output_root,
                stimulus_ids=stimulus_ids,
                batch_size=args.decode_batch_size,
                device=args.device,
            )


if __name__ == "__main__":
    main()
