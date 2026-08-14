"""Decode and score the VDVAE latent calibration battery."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import struct
import zipfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.prepare_reconstruction_features import (
    _VDVAE_LAYER_DIMS,
    _load_vdvae_model,
)
from src.data.shared_paths import default_stimuli_hdf5
from src.pipelines.diagnose_vdvae_latents import (
    _require_cuda,
    layer_bounds,
    per_row_correlation,
)

logger = logging.getLogger(__name__)


ARMS = [
    {
        "name": "true",
        "source": "true",
        "n_prefix_layers": len(_VDVAE_LAYER_DIMS),
    },
    {
        "name": "raw_pred",
        "source": "raw_pred",
        "n_prefix_layers": len(_VDVAE_LAYER_DIMS),
    },
    {
        "name": "renorm_pred",
        "source": "renorm_pred",
        "n_prefix_layers": len(_VDVAE_LAYER_DIMS),
    },
    {
        "name": "renorm_shuffled",
        "source": "renorm_shuffled",
        "n_prefix_layers": len(_VDVAE_LAYER_DIMS),
    },
    {"name": "prefix_0", "source": "renorm_pred", "n_prefix_layers": 0},
    {"name": "prefix_2", "source": "renorm_pred", "n_prefix_layers": 2},
    {"name": "prefix_6", "source": "renorm_pred", "n_prefix_layers": 6},
    {"name": "prefix_14", "source": "renorm_pred", "n_prefix_layers": 14},
]


def calibrate_affine(
    pred: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Match per-dimension moments using this batch's own moments.

    This is a transductive diagnostic calibration, not a production or
    out-of-fold calibration procedure.
    """
    pred = np.asarray(pred)
    calibrated = (
        (pred - pred.mean(axis=0)) / np.maximum(pred.std(axis=0), eps)
    ) * np.asarray(target_std) + np.asarray(target_mean)
    return calibrated.astype(np.float32)


def pixcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return per-image Pearson correlation between flattened RGB pixels."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 4 or b.ndim != 4 or a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError(
            f"Expected matching [N, H, W, 3] arrays, got {a.shape} and {b.shape}."
        )
    return per_row_correlation(
        a.reshape(len(a), -1).astype(np.float64),
        b.reshape(len(b), -1).astype(np.float64),
    )


def two_way_identification(sim: np.ndarray) -> float:
    """Return ordered-pair identification accuracy from a similarity matrix."""
    sim = np.asarray(sim)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Expected a square similarity matrix, got {sim.shape}.")
    if len(sim) < 2:
        raise ValueError("At least two images are required for two-way identification.")
    correct = np.diag(sim)[:, None] > sim
    off_diagonal = ~np.eye(len(sim), dtype=bool)
    return float(correct[off_diagonal].mean())


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return the percentile 95% bootstrap interval for the sample mean."""
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _memmap_npz_member(npz_path: Path, name: str) -> np.memmap:
    """Memory-map an uncompressed NPY member inside an NPZ archive."""
    member_name = f"{name}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"Cannot memory-map compressed member {member_name!r} in {npz_path}."
            )
        header_offset = info.header_offset

    with open(npz_path, "rb") as npz_file:
        npz_file.seek(header_offset)
        local_header = npz_file.read(30)
        if local_header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP member header for {member_name!r}.")
        filename_length, extra_length = struct.unpack("<HH", local_header[26:30])
        npz_file.seek(filename_length + extra_length, 1)
        version = np.lib.format.read_magic(npz_file)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                npz_file
            )
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                npz_file
            )
        else:
            raise ValueError(
                f"Unsupported NPY version {version} for {member_name!r}."
            )
        data_offset = npz_file.tell()

    if dtype.hasobject:
        raise ValueError(f"Cannot memory-map object array {member_name!r}.")
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def _load_eval_rows(
    feature_dir: Path,
    predicted_npy: Path,
    eval_indices_json: Path,
    n_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(eval_indices_json) as indices_file:
        eval_indices = np.asarray(
            json.load(indices_file)["eval_indices"], dtype=np.int64
        )

    requested_n_images = int(n_images)
    if requested_n_images <= 0:
        raise ValueError(f"n_images must be positive, got {requested_n_images}.")
    if requested_n_images > len(eval_indices):
        logger.info(
            "Clamping n_images from %d to %d available eval indices.",
            requested_n_images,
            len(eval_indices),
        )
    n_images = min(requested_n_images, len(eval_indices))
    rows = eval_indices[:n_images]

    feature_path = feature_dir / "vdvae_features.npz"
    test_stim_idx = _memmap_npz_member(feature_path, "test_stim_idx")
    test_latents = _memmap_npz_member(feature_path, "test_latents")
    stim_ids = np.asarray(test_stim_idx[rows], dtype=np.int64)
    true = np.asarray(test_latents[rows], dtype=np.float32)

    predicted = np.load(predicted_npy, mmap_mode="r")
    assert len(predicted) >= n_images, (
        f"Predicted row count mismatch: got {len(predicted)}, need at least {n_images}."
    )
    pred = np.asarray(predicted[:n_images], dtype=np.float32)
    logger.info(
        "Selected eval shapes: rows=%s stim_ids=%s true=%s pred=%s",
        rows.shape,
        stim_ids.shape,
        true.shape,
        pred.shape,
    )
    return rows, stim_ids, true, pred, eval_indices


def _compute_train_moments(
    train_latents: np.ndarray,
    column_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    target_mean = np.empty(train_latents.shape[1], dtype=np.float32)
    target_std = np.empty(train_latents.shape[1], dtype=np.float32)
    for start in range(0, train_latents.shape[1], column_chunk):
        end = min(start + column_chunk, train_latents.shape[1])
        chunk = np.asarray(train_latents[:, start:end], dtype=np.float64)
        target_mean[start:end] = chunk.mean(axis=0).astype(np.float32)
        target_std[start:end] = chunk.std(axis=0).astype(np.float32)
    return target_mean, target_std


def _parse_requested_arms(arms_arg: str) -> list[dict]:
    known_names = [arm["name"] for arm in ARMS]
    if arms_arg.strip() == "all":
        return list(ARMS)

    requested_names = {
        name.strip() for name in arms_arg.split(",") if name.strip()
    }
    unknown_names = sorted(requested_names.difference(known_names))
    if unknown_names:
        raise ValueError(
            f"Unknown arms: {', '.join(unknown_names)}. "
            f"Known arms: {', '.join(known_names)}."
        )
    if not requested_names:
        raise ValueError("No arms requested.")
    return [arm for arm in ARMS if arm["name"] in requested_names]


def _processed_latent_std(latents: np.ndarray, n_prefix_layers: int) -> float:
    if n_prefix_layers == 0:
        return 0.0
    end = layer_bounds(_VDVAE_LAYER_DIMS)[n_prefix_layers - 1][1]
    return float(np.std(latents[:, :end]))


def _construct_arms(
    feature_dir: Path,
    true: np.ndarray,
    raw_pred: np.ndarray,
    requested_arms: list[dict],
) -> list[dict]:
    train_latents = _memmap_npz_member(
        feature_dir / "vdvae_features.npz", "train_latents"
    )
    train_mean, train_std = _compute_train_moments(
        train_latents, column_chunk=4096
    )
    renorm_pred = calibrate_affine(raw_pred, train_mean, train_std)
    renorm_shuffled = np.roll(renorm_pred, 1, axis=0)
    latent_sources = {
        "true": true,
        "raw_pred": raw_pred,
        "renorm_pred": renorm_pred,
        "renorm_shuffled": renorm_shuffled,
    }

    arm_inputs = []
    for arm in requested_arms:
        latents = latent_sources[arm["source"]]
        latent_std = _processed_latent_std(
            latents, int(arm["n_prefix_layers"])
        )
        arm_input = {
            **arm,
            "latents": latents,
            "latent_std_after_processing": latent_std,
        }
        arm_inputs.append(arm_input)
        logger.info(
            "Arm %s latent std after processing: %.6f",
            arm["name"],
            latent_std,
        )
    return arm_inputs


def decode_prefix(
    ema_vae,
    latents_2d: np.ndarray,
    ref_latent,
    n_prefix_layers: int,
    out_dir: Path,
    save_rows: np.ndarray,
    save_stim: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> None:
    """Decode a flattened prefix of VDVAE latents and sample later priors."""
    bounds = layer_bounds(_VDVAE_LAYER_DIMS)
    if n_prefix_layers < 0 or n_prefix_layers > len(bounds):
        raise ValueError(
            f"n_prefix_layers must be in [0, {len(bounds)}], got {n_prefix_layers}."
        )

    latents_hier = []
    for layer_idx, (start, end) in enumerate(bounds[:n_prefix_layers]):
        layer = latents_2d[:, start:end]
        c, h, w = ref_latent[layer_idx]["z"].shape[1:]
        latents_hier.append(layer.reshape(len(latents_2d), c, h, w))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for start in range(0, len(latents_2d), batch_size):
        end = min(start + batch_size, len(latents_2d))
        sample_ids = range(start, end)
        n = len(sample_ids)
        sample_latents = [
            torch.tensor(layer[sample_ids]).float().to(device)
            for layer in latents_hier
        ]
        with torch.no_grad():
            px_z = ema_vae.decoder.forward_manual_latents(
                n, sample_latents, t=None
            )
            imgs = ema_vae.decoder.out_net.sample(px_z)

        for offset, arr in enumerate(imgs):
            idx = start + offset
            row = int(save_rows[idx])
            stim = int(save_stim[idx])
            image = Image.fromarray(arr).resize(
                (512, 512), resample=Image.Resampling.BICUBIC
            )
            image.save(out_dir / f"row{row:05d}_stim{stim}.png")


def _decoded_paths(
    arm_dir: Path,
    rows: np.ndarray,
    stim_ids: np.ndarray,
) -> list[Path]:
    return [
        arm_dir / f"row{int(row):05d}_stim{int(stim)}.png"
        for row, stim in zip(rows, stim_ids)
    ]


def _decode_requested_arms(
    arm_inputs: list[dict],
    feature_dir: Path,
    recon_model_root: Path,
    output_dir: Path,
    rows: np.ndarray,
    stim_ids: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
    skip_decode_if_exists: bool,
) -> dict[str, list[Path]]:
    paths_by_arm = {
        arm["name"]: _decoded_paths(output_dir / arm["name"], rows, stim_ids)
        for arm in arm_inputs
    }
    arms_to_decode = []
    for arm in arm_inputs:
        paths = paths_by_arm[arm["name"]]
        if skip_decode_if_exists and all(path.exists() for path in paths):
            logger.info("Skipping existing decodes for arm %s.", arm["name"])
        else:
            arms_to_decode.append(arm)

    if arms_to_decode:
        with np.load(
            feature_dir / "ref_latents.npz", allow_pickle=True
        ) as ref_npz:
            ref_latent = ref_npz["ref_latent"]
        ema_vae, _ = _load_vdvae_model(recon_model_root)
        for arm in arms_to_decode:
            logger.info(
                "Decoding arm %s with %d prefix layers.",
                arm["name"],
                arm["n_prefix_layers"],
            )
            decode_prefix(
                ema_vae=ema_vae,
                latents_2d=arm["latents"],
                ref_latent=ref_latent,
                n_prefix_layers=int(arm["n_prefix_layers"]),
                out_dir=output_dir / arm["name"],
                save_rows=rows,
                save_stim=stim_ids,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )

    for arm in arm_inputs:
        missing = [path for path in paths_by_arm[arm["name"]] if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Arm {arm['name']} has {len(missing)} missing decoded PNGs."
            )
        logger.info(
            "Arm %s has %d decoded PNGs.",
            arm["name"],
            len(paths_by_arm[arm["name"]]),
        )
    return paths_by_arm


def _clip_embeddings(
    images: list[Image.Image],
    clip_model,
    clip_preprocess,
    batch_size: int,
    device: str,
) -> np.ndarray:
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [clip_preprocess(image) for image in images[start : start + batch_size]]
            ).to(device)
            embeddings = clip_model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(
                1e-12
            )
            batches.append(embeddings.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def score_arm(
    decoded_paths: list[Path],
    original_pixels: np.ndarray,
    original_embeddings: np.ndarray,
    clip_model,
    clip_preprocess,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, float]:
    """Score one arm with PixCorr and CLIP ordered-pair identification."""
    decoded_images = []
    for path in decoded_paths:
        with Image.open(path) as image:
            decoded_images.append(image.convert("RGB"))

    decoded_pixels = np.stack(
        [
            np.asarray(
                image.resize((64, 64), resample=Image.Resampling.BICUBIC)
            )
            for image in decoded_images
        ]
    )
    pixcorr_values = pixcorr(decoded_pixels, original_pixels)
    pixcorr_low, pixcorr_high = bootstrap_ci(pixcorr_values, seed=seed)

    recon_embeddings = _clip_embeddings(
        decoded_images,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        batch_size=batch_size,
        device=device,
    )
    similarities = recon_embeddings @ original_embeddings.T
    return {
        "n_images": int(len(decoded_paths)),
        "pixcorr_mean": float(np.mean(pixcorr_values)),
        "pixcorr_lo": pixcorr_low,
        "pixcorr_hi": pixcorr_high,
        "clip_2way_id": two_way_identification(similarities),
    }


def _save_battery_panel(
    out_path: Path,
    original: Image.Image,
    arm_paths: list[tuple[str, Path]],
    row: int,
    stim: int,
) -> None:
    columns = [("Original", original.convert("RGB"))]
    for arm_name, image_path in arm_paths:
        with Image.open(image_path) as image:
            columns.append((arm_name, image.convert("RGB")))

    figure, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for axis, (name, image) in zip(axes, columns):
        axis.imshow(image)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    figure.suptitle(f"Row {row} | Stimulus {stim}", fontsize=11)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _panel_positions(rows: np.ndarray) -> list[int]:
    positions = []
    for target_row in (849, 973):
        matches = np.flatnonzero(rows == target_row)
        if matches.size:
            positions.append(int(matches[0]))
    if len(positions) != 2:
        return list(range(min(2, len(rows))))
    return positions


def run_battery(
    feature_dir: Path,
    recon_model_root: Path,
    predicted_npy: Path,
    eval_indices_json: Path,
    stimuli_hdf5: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    n_images: int,
    seed: int,
    requested_arms: list[dict],
    skip_decode_if_exists: bool,
) -> None:
    _require_cuda(device)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    rows, stim_ids, true, raw_pred, _ = _load_eval_rows(
        feature_dir=feature_dir,
        predicted_npy=predicted_npy,
        eval_indices_json=eval_indices_json,
        n_images=n_images,
    )
    arm_inputs = _construct_arms(
        feature_dir=feature_dir,
        true=true,
        raw_pred=raw_pred,
        requested_arms=requested_arms,
    )
    paths_by_arm = _decode_requested_arms(
        arm_inputs=arm_inputs,
        feature_dir=feature_dir,
        recon_model_root=recon_model_root,
        output_dir=output_dir,
        rows=rows,
        stim_ids=stim_ids,
        batch_size=batch_size,
        device=device,
        seed=seed,
        skip_decode_if_exists=skip_decode_if_exists,
    )

    originals = []
    with h5py.File(stimuli_hdf5, "r") as stimuli_file:
        images = stimuli_file["imgBrick"]
        for stim in stim_ids:
            originals.append(Image.fromarray(images[int(stim)]).convert("RGB"))
    original_pixels = np.stack(
        [
            np.asarray(
                image.resize((64, 64), resample=Image.Resampling.BICUBIC)
            )
            for image in originals
        ]
    )

    import open_clip

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L/14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    original_embeddings = _clip_embeddings(
        originals,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        batch_size=batch_size,
        device=device,
    )

    fieldnames = [
        "arm",
        "n_images",
        "pixcorr_mean",
        "pixcorr_lo",
        "pixcorr_hi",
        "clip_2way_id",
        "latent_std_after_processing",
    ]
    metric_rows = []
    for arm in arm_inputs:
        metrics = score_arm(
            decoded_paths=paths_by_arm[arm["name"]],
            original_pixels=original_pixels,
            original_embeddings=original_embeddings,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        metric_row = {
            "arm": arm["name"],
            **metrics,
            "latent_std_after_processing": arm[
                "latent_std_after_processing"
            ],
        }
        metric_rows.append(metric_row)
        logger.info(
            "Arm %s: PixCorr=%.6f, CLIP 2-way ID=%.6f",
            arm["name"],
            metrics["pixcorr_mean"],
            metrics["clip_2way_id"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "battery_metrics.csv"
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    panel_paths = []
    for position in _panel_positions(rows):
        row = int(rows[position])
        stim = int(stim_ids[position])
        panel_path = output_dir / "panels" / f"row{row:05d}_stim{stim}.png"
        _save_battery_panel(
            out_path=panel_path,
            original=originals[position],
            arm_paths=[
                (arm["name"], paths_by_arm[arm["name"]][position])
                for arm in arm_inputs
            ],
            row=row,
            stim=stim,
        )
        panel_paths.append(panel_path)

    summary = {
        "arms": [
            {
                "name": arm["name"],
                "source": arm["source"],
                "n_prefix_layers": int(arm["n_prefix_layers"]),
                "latent_std_after_processing": arm[
                    "latent_std_after_processing"
                ],
            }
            for arm in arm_inputs
        ],
        "seeds": {"decode": int(seed), "bootstrap": int(seed)},
        "files": {
            "vdvae_features": str(feature_dir / "vdvae_features.npz"),
            "ref_latents": str(feature_dir / "ref_latents.npz"),
            "predicted_npy": str(predicted_npy),
            "eval_indices_json": str(eval_indices_json),
            "stimuli_hdf5": str(stimuli_hdf5),
            "decoded_dirs": {
                arm["name"]: str(output_dir / arm["name"])
                for arm in arm_inputs
            },
            "metrics_csv": str(metrics_path),
            "panels": [str(path) for path in panel_paths],
        },
        "parameters": {
            "device": device,
            "batch_size": int(batch_size),
            "n_images_requested": int(n_images),
            "n_images_used": int(len(rows)),
            "train_moment_column_chunk": 4096,
            "clip_model": "ViT-L/14",
            "clip_pretrained": "openai",
            "pixcorr_image_size": 64,
            "decoded_image_size": 512,
            "skip_decode_if_exists": bool(skip_decode_if_exists),
        },
        "rows": rows.tolist(),
        "stim_ids": stim_ids.tolist(),
    }
    with open(output_dir / "battery_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    logger.info("Saved VDVAE calibration battery to %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Decode and score a VDVAE latent calibration battery."
    )
    parser.add_argument(
        "--feature-dir", default="data/processed/reconstruction_features/subj07"
    )
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument(
        "--predicted-npy",
        default="artifacts/reconstructions/subj07/predicted_features/gt_fmri_vdvae.npy",
    )
    parser.add_argument(
        "--eval-indices-json",
        default="artifacts/predictions/subj07/zeroshot_sub7_metrics.json",
    )
    parser.add_argument("--stimuli-hdf5", default=default_stimuli_hdf5())
    parser.add_argument(
        "--output-dir", default="artifacts/diagnostics/vdvae_battery"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--skip-decode-if-exists", action="store_true")
    args = parser.parse_args()

    try:
        requested_arms = _parse_requested_arms(str(args.arms))
    except ValueError as error:
        parser.error(str(error))
    run_battery(
        feature_dir=Path(args.feature_dir),
        recon_model_root=Path(args.recon_model_root),
        predicted_npy=Path(args.predicted_npy),
        eval_indices_json=Path(args.eval_indices_json),
        stimuli_hdf5=Path(args.stimuli_hdf5),
        output_dir=Path(args.output_dir),
        device=str(args.device),
        batch_size=int(args.batch_size),
        n_images=int(args.n_images),
        seed=int(args.seed),
        requested_arms=requested_arms,
        skip_decode_if_exists=bool(args.skip_decode_if_exists),
    )


if __name__ == "__main__":
    main()
