"""Diagnose VDVAE oracle decoding and stochastic-latent reliability."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.prepare_reconstruction_features import (
    _VDVAE_LAYER_DIMS,
    _load_vdvae_model,
    _StimuliDataset,
)
from src.data.shared_paths import default_stimuli_hdf5
from src.pipelines.benchmark_reconstructions_vdvae_vd import (
    _decode_vdvae_latents,
    _latent_transformation,
)

logger = logging.getLogger(__name__)


def layer_bounds(layer_dims: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous flattened-latent bounds for each layer."""
    ends = np.cumsum(np.asarray(layer_dims, dtype=np.int64))
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def per_dim_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation across images for each dimension."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"Expected matching 2D arrays, got {a.shape} and {b.shape}.")

    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    a_std = a_centered.std(axis=0)
    b_std = b_centered.std(axis=0)
    valid = (a_std >= 1e-10) & (b_std >= 1e-10)
    correlations = np.zeros(a.shape[1], dtype=np.float64)
    numerator = (a_centered * b_centered).sum(axis=0)
    denominator = a.shape[0] * a_std * b_std
    np.divide(numerator, denominator, out=correlations, where=valid)
    return correlations.astype(np.float32)


def per_row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation across dimensions for each image."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"Expected matching 2D arrays, got {a.shape} and {b.shape}.")

    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    a_std = a_centered.std(axis=1)
    b_std = b_centered.std(axis=1)
    valid = (a_std >= 1e-10) & (b_std >= 1e-10)
    correlations = np.zeros(a.shape[0], dtype=np.float64)
    numerator = (a_centered * b_centered).sum(axis=1)
    denominator = a.shape[1] * a_std * b_std
    np.divide(numerator, denominator, out=correlations, where=valid)
    return correlations.astype(np.float32)


def pairwise_reliability(repeats: list[np.ndarray]) -> dict:
    """Average per-dimension and per-row correlations over repeat pairs."""
    repeat_pairs = list(combinations(repeats, 2))
    if not repeat_pairs:
        raise ValueError("At least two repeats are required for pairwise reliability.")

    dim_correlations = np.stack(
        [per_dim_correlation(first, second) for first, second in repeat_pairs]
    ).mean(axis=0)
    row_correlations = np.stack(
        [per_row_correlation(first, second) for first, second in repeat_pairs]
    ).mean(axis=0)
    return {
        "mean_dim_reliability": float(np.mean(dim_correlations)),
        "median_dim_reliability": float(np.median(dim_correlations)),
        "mean_row_reliability": float(np.mean(row_correlations)),
        "n_pairs": len(repeat_pairs),
    }


def _parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",")]


def _require_cuda(device: str) -> None:
    if device != "cuda":
        raise RuntimeError(
            f"The VDVAE loader is CUDA-only; --device must be 'cuda', got {device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The VDVAE loader is CUDA-only, but torch.cuda.is_available() is False."
        )


def _save_oracle_panel(
    out_path: Path,
    original: Image.Image,
    oracle_path: Path,
    row: int,
    stim: int,
) -> None:
    columns = [
        ("Original", original.convert("RGB")),
        ("Oracle decode", Image.open(oracle_path).convert("RGB")),
    ]
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


def run_oracle(
    feature_dir: Path,
    recon_model_root: Path,
    output_dir: Path,
    device: str,
    rows_arg: str,
    batch_size: int,
    stimuli_hdf5: Path,
    scale: float = 1.0,
) -> None:
    _require_cuda(device)

    features = np.load(feature_dir / "vdvae_features.npz", mmap_mode="r")
    try:
        test_latents = features["test_latents"]
        test_stim_idx = features["test_stim_idx"]
        rows_list = _parse_int_list(rows_arg)
        for row in rows_list:
            if row < 0 or row >= len(test_latents):
                raise ValueError(
                    f"Requested row {row} is outside [0, {len(test_latents)})."
                )
        rows = np.asarray(rows_list, dtype=np.int64)
        stim_ids = np.asarray(test_stim_idx[rows], dtype=np.int64)
        true_latents = np.asarray(test_latents[rows], dtype=np.float32)
    finally:
        features.close()

    if float(scale) != 1.0:
        true_latents = true_latents * np.float32(scale)
        logger.info("Applied latent scale factor %.6f before decoding.", float(scale))

    with np.load(feature_dir / "ref_latents.npz", allow_pickle=True) as ref_npz:
        ref_latent = ref_npz["ref_latent"]

    ema_vae, _ = _load_vdvae_model(Path(recon_model_root))
    oracle_dir = output_dir / "oracle"
    _decode_vdvae_latents(
        ema_vae=ema_vae,
        pred_latents=true_latents,
        ref_latent=ref_latent,
        out_dir=oracle_dir,
        save_rows=rows,
        save_stim=stim_ids,
        batch_size=batch_size,
        device=device,
    )

    original_dir = output_dir / "original"
    panel_dir = output_dir / "panels"
    original_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(stimuli_hdf5, "r") as stimuli_file:
        images = stimuli_file["imgBrick"]
        for row, stim in zip(rows.tolist(), stim_ids.tolist()):
            original = Image.fromarray(images[stim]).resize(
                (512, 512), resample=Image.Resampling.BICUBIC
            )
            filename = f"row{row:05d}_stim{stim}.png"
            original.save(original_dir / filename)
            _save_oracle_panel(
                out_path=panel_dir / filename,
                original=original,
                oracle_path=oracle_dir / filename,
                row=row,
                stim=stim,
            )

    summary = {
        "rows": rows.tolist(),
        "stim_ids": stim_ids.tolist(),
        "n_decoded": int(len(rows)),
        "latent_scale": float(scale),
        "true_latent_stats": {
            "mean": float(np.mean(true_latents)),
            "std": float(np.std(true_latents)),
            "min": float(np.min(true_latents)),
            "max": float(np.max(true_latents)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "oracle_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    logger.info("Saved oracle diagnostics to %s", output_dir)


def run_reliability(
    feature_dir: Path,
    recon_model_root: Path,
    output_dir: Path,
    device: str,
    n_images: int,
    seeds_arg: str,
    batch_size: int,
    predicted_npy: Path,
    eval_indices_json: Path,
    stimuli_hdf5: Path,
) -> None:
    _require_cuda(device)

    with open(eval_indices_json) as indices_file:
        eval_indices = np.asarray(json.load(indices_file)["eval_indices"], dtype=np.int64)
    requested_n_images = int(n_images)
    if requested_n_images > len(eval_indices):
        logger.info(
            "Clamping n_images from %d to %d available eval indices.",
            requested_n_images,
            len(eval_indices),
        )
    n_images = min(requested_n_images, len(eval_indices))
    rows = eval_indices[:n_images]

    features = np.load(feature_dir / "vdvae_features.npz", mmap_mode="r")
    try:
        test_latents = features["test_latents"]
        test_stim_idx = features["test_stim_idx"]
        stim_ids = np.asarray(test_stim_idx[rows], dtype=np.int64)
        true = np.asarray(test_latents[rows], dtype=np.float32)
    finally:
        features.close()

    seeds = _parse_int_list(seeds_arg)
    ema_vae, preprocess_fn = _load_vdvae_model(Path(recon_model_root))
    dataset = _StimuliDataset(
        stimuli_hdf5=stimuli_hdf5,
        stim_idx=stim_ids,
        image_size=64,
        mode="vdvae",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    expected_width = int(_VDVAE_LAYER_DIMS.sum())
    repeats: list[np.ndarray] = []
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            seed_batches = []
            with torch.no_grad():
                for batch in loader:
                    data_input, _ = preprocess_fn(batch)
                    activations = ema_vae.encoder.forward(data_input)
                    _, stats = ema_vae.decoder.forward(activations, get_latents=True)
                    batch_latents = np.hstack(
                        [
                            stats[i]["z"]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(len(batch), -1)
                            for i in range(len(_VDVAE_LAYER_DIMS))
                        ]
                    )
                    seed_batches.append(batch_latents.astype(np.float32, copy=False))
            repeat = np.concatenate(seed_batches, axis=0).astype(np.float32, copy=False)
            if repeat.shape[1] != expected_width:
                raise ValueError(
                    f"VDVAE latent width mismatch: got {repeat.shape[1]}, "
                    f"expected {expected_width}."
                )
            repeats.append(repeat)
    finally:
        dataset.close()

    predicted = None
    if predicted_npy.exists():
        predicted_array = np.load(predicted_npy, mmap_mode="r")
        predicted = np.asarray(predicted_array[:n_images], dtype=np.float32)
        assert predicted.shape[0] == n_images, (
            f"Predicted row count mismatch: got {predicted.shape[0]}, expected {n_images}."
        )
    else:
        logger.warning(
            "Predicted VDVAE latents not found at %s; achieved columns will be empty.",
            predicted_npy,
        )

    fieldnames = [
        "layer",
        "start",
        "end",
        "n_dims",
        "mean_dim_reliability",
        "median_dim_reliability",
        "mean_row_reliability",
        "single_sample_ceiling",
        "achieved_mean_dim_r",
        "achieved_over_ceiling",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_layer_reliability.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for layer, (start, end) in enumerate(layer_bounds(_VDVAE_LAYER_DIMS)):
            layer_reliability = pairwise_reliability(
                [repeat[:, start:end] for repeat in repeats]
            )
            ceiling = float(
                np.sqrt(np.clip(layer_reliability["mean_dim_reliability"], 0.0, 1.0))
            )
            achieved = ""
            achieved_over_ceiling = ""
            if predicted is not None:
                achieved = float(
                    np.mean(per_dim_correlation(true[:, start:end], predicted[:, start:end]))
                )
                if ceiling > 1e-6:
                    achieved_over_ceiling = achieved / ceiling
            writer.writerow(
                {
                    "layer": layer,
                    "start": start,
                    "end": end,
                    "n_dims": end - start,
                    "mean_dim_reliability": layer_reliability[
                        "mean_dim_reliability"
                    ],
                    "median_dim_reliability": layer_reliability[
                        "median_dim_reliability"
                    ],
                    "mean_row_reliability": layer_reliability[
                        "mean_row_reliability"
                    ],
                    "single_sample_ceiling": ceiling,
                    "achieved_mean_dim_r": achieved,
                    "achieved_over_ceiling": achieved_over_ceiling,
                }
            )

    global_reliability = pairwise_reliability(repeats)
    global_ceiling = float(
        np.sqrt(np.clip(global_reliability["mean_dim_reliability"], 0.0, 1.0))
    )
    global_achieved = None
    global_achieved_over_ceiling = None
    if predicted is not None:
        global_achieved = float(np.mean(per_dim_correlation(true, predicted)))
        if global_ceiling > 1e-6:
            global_achieved_over_ceiling = global_achieved / global_ceiling

    summary = {
        "arguments": {
            "mode": "reliability",
            "feature_dir": str(feature_dir),
            "recon_model_root": str(recon_model_root),
            "output_dir": str(output_dir),
            "device": device,
            "batch_size": int(batch_size),
            "predicted_npy": str(predicted_npy),
            "eval_indices_json": str(eval_indices_json),
            "stimuli_hdf5": str(stimuli_hdf5),
        },
        "n_images": int(n_images),
        "seeds": seeds,
        "rows": rows.tolist(),
        "global": {
            **global_reliability,
            "single_sample_ceiling": global_ceiling,
            "achieved_mean_dim_r": global_achieved,
            "achieved_over_ceiling": global_achieved_over_ceiling,
        },
    }
    with open(output_dir / "reliability_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    logger.info("Saved reliability diagnostics to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Diagnose saved VDVAE latent features.")
    parser.add_argument("--mode", required=True, choices=("oracle", "reliability"))
    parser.add_argument(
        "--feature-dir", default="data/processed/reconstruction_features/subj07"
    )
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument("--output-dir", default="artifacts/diagnostics/vdvae_latents")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", default="849,899,650,534,973")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply the true latents by this factor before decoding (oracle mode only).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stimuli-hdf5", default=default_stimuli_hdf5())
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--predicted-npy",
        default="artifacts/reconstructions/subj07/predicted_features/gt_fmri_vdvae.npy",
    )
    parser.add_argument(
        "--eval-indices-json",
        default="artifacts/predictions/subj07/zeroshot_sub7_metrics.json",
    )
    args = parser.parse_args()

    if args.mode == "oracle":
        batch_size = 5 if args.batch_size is None else int(args.batch_size)
        run_oracle(
            feature_dir=Path(args.feature_dir),
            recon_model_root=Path(args.recon_model_root),
            output_dir=Path(args.output_dir),
            device=str(args.device),
            rows_arg=str(args.rows),
            batch_size=batch_size,
            stimuli_hdf5=Path(args.stimuli_hdf5),
            scale=float(args.scale),
        )
    else:
        batch_size = 8 if args.batch_size is None else int(args.batch_size)
        run_reliability(
            feature_dir=Path(args.feature_dir),
            recon_model_root=Path(args.recon_model_root),
            output_dir=Path(args.output_dir),
            device=str(args.device),
            n_images=int(args.n_images),
            seeds_arg=str(args.seeds),
            batch_size=batch_size,
            predicted_npy=Path(args.predicted_npy),
            eval_indices_json=Path(args.eval_indices_json),
            stimuli_hdf5=Path(args.stimuli_hdf5),
        )


if __name__ == "__main__":
    main()
