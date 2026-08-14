"""
Benchmark reconstructions from subject-level activation conditions using the
VDVAE + Versatile Diffusion workflow:
- ground-truth fMRI
- zero-shot predicted fMRI
- best few-shot predicted fMRI

Pipeline:
1) regress fMRI -> VDVAE latents / CLIP text / CLIP vision features
2) decode VDVAE latents to initial images
3) refine with Versatile Diffusion conditioned on predicted CLIP features
4) save side-by-side panels and summary metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge

from src.config import load_config
from src.data.shared_paths import default_stimuli_hdf5

from src.pipelines.reconstruction_utils import (
    _collect_fewshot_runs,
    _load_image_from_dir,
    _load_image_from_npy,
    _read_best_n_from_summary,
    _resolve_eval_indices_for_run,
    _rowwise_corr,
    _save_panel,
    _select_fewshot_run,
)
from src.pipelines.multiexpert_artifacts import file_sha256
from src.pipelines.predict_train_responses import _train_stim_idx_sha256
from src.pipelines.vdvae_calibration import (
    apply_calibration,
    learn_calibration,
    load_calibration,
    make_folds,
    save_calibration,
)

logger = logging.getLogger(__name__)

_VDVAE_LAYER_DIMS = np.array(
    [
        2**4,
        2**4,
        2**8,
        2**8,
        2**8,
        2**8,
        2**10,
        2**10,
        2**10,
        2**10,
        2**10,
        2**10,
        2**10,
        2**10,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**12,
        2**14,
    ],
    dtype=np.int64,
)


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _safe_std(x: np.ndarray, axis: int, ddof: int = 0) -> np.ndarray:
    s = x.std(axis=axis, ddof=ddof, keepdims=True)
    s[s < 1e-8] = 1.0
    return s


def _as_int_vector(arr: np.ndarray, name: str) -> np.ndarray:
    vec = np.asarray(arr, dtype=np.int64).ravel()
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={vec.shape}.")
    return vec


def _validate_sorted_unique_stim_idx(stim_idx: np.ndarray, name: str) -> np.ndarray:
    stim_idx = _as_int_vector(stim_idx, name=name)
    if stim_idx.size == 0:
        raise ValueError(f"{name} is empty.")
    unique = np.unique(stim_idx)
    if unique.size != stim_idx.size:
        raise ValueError(f"{name} contains duplicate stimulus IDs.")
    if not np.array_equal(unique, stim_idx):
        raise ValueError(f"{name} must be sorted ascending for deterministic row mapping.")
    return stim_idx


def _validate_unique_stim_idx(stim_idx: np.ndarray, name: str) -> np.ndarray:
    stim_idx = _as_int_vector(stim_idx, name=name)
    if stim_idx.size == 0:
        raise ValueError(f"{name} is empty.")
    unique = np.unique(stim_idx)
    if unique.size != stim_idx.size:
        raise ValueError(f"{name} contains duplicate stimulus IDs.")
    return stim_idx


def _align_rows_by_stim_idx(
    source_arr: np.ndarray,
    source_stim_idx: np.ndarray,
    query_stim_idx: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_stim_idx = _validate_unique_stim_idx(
        source_stim_idx,
        name=f"{label} source_stim_idx",
    )
    query_stim_idx = _validate_unique_stim_idx(query_stim_idx, name=f"{label} query_stim_idx")
    if int(source_arr.shape[0]) != int(source_stim_idx.shape[0]):
        raise ValueError(
            f"{label} row mismatch: array rows={int(source_arr.shape[0])}, "
            f"stim_idx rows={int(source_stim_idx.shape[0])}."
        )

    row_by_stim = {int(stim): idx for idx, stim in enumerate(source_stim_idx.tolist())}
    mapped_rows = np.array([row_by_stim.get(int(stim), -1) for stim in query_stim_idx], dtype=np.int64)
    valid_mask = mapped_rows >= 0
    n_valid = int(valid_mask.sum())
    n_total = int(query_stim_idx.shape[0])
    if n_valid != n_total:
        missing = query_stim_idx[~valid_mask]
        preview = ", ".join(str(int(v)) for v in missing[:10])
        suffix = "" if missing.size <= 10 else ", ..."
        raise ValueError(
            f"{label} missing {int(missing.size)}/{n_total} requested stimuli. "
            f"Examples: [{preview}{suffix}]"
        )
    return source_arr[mapped_rows], valid_mask


def _load_stim_idx_file(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")
    arr = np.load(path)
    return _validate_unique_stim_idx(arr, name=label)


def _align_train_rows(
    train_matrix: np.ndarray,
    train_stim_idx: np.ndarray,
    train_targets: np.ndarray,
    label: str,
    target_train_stim_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    n_target = int(train_targets.shape[0])
    if len(target_train_stim_idx) != n_target:
        raise ValueError(
            f"{label} train stim-index length mismatch: "
            f"indices={len(target_train_stim_idx)}, targets={n_target}."
        )
    row_by_stim = {int(stim): idx for idx, stim in enumerate(train_stim_idx.tolist())}
    mapped_rows = np.array(
        [row_by_stim.get(int(stim), -1) for stim in target_train_stim_idx],
        dtype=np.int64,
    )
    valid_mask = mapped_rows >= 0
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError(f"{label}: no target stimuli were found in subject train_stim_idx.")
    if n_valid != n_target:
        logger.warning(
            "%s train alignment by stimulus dropped %d rows (%d/%d kept).",
            label,
            n_target - n_valid,
            n_valid,
            n_target,
        )
    return (
        train_matrix[mapped_rows[valid_mask]],
        train_targets[valid_mask],
        {"mode": "stim_index", "rows_used": n_valid, "rows_requested": n_target},
    )


def _require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {label}: {path}. "
            "Generate local reconstruction features under data/processed/reconstruction_features "
            "and pass explicit paths if you use a non-default location."
        )


def _require_model_root(model_root: Path):
    vdvae_dir = model_root / "vdvae"
    vd_dir = model_root / "versatile_diffusion"
    if not vdvae_dir.exists() or not vd_dir.exists():
        raise FileNotFoundError(
            f"Model root is missing expected folders: {model_root}. "
            "Expected both 'vdvae/' and 'versatile_diffusion/'."
        )
    return model_root


def _load_clip_split_with_stim_idx(
    split_arr_path: Path,
    split_stim_idx_path: Path,
    split_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    _require_file(split_arr_path, split_label)
    arr = np.load(split_arr_path).astype(np.float32)
    stim_idx = _load_stim_idx_file(split_stim_idx_path, f"{split_label} stim_idx")
    if int(stim_idx.shape[0]) != int(arr.shape[0]):
        raise ValueError(
            f"{split_label} rows mismatch: array rows={int(arr.shape[0])}, "
            f"stim_idx rows={int(stim_idx.shape[0])}."
        )
    return arr, stim_idx


def _resolve_vdvae_stim_idx(
    vdvae_features: np.lib.npyio.NpzFile,
    split: str,
    expected_rows: int,
) -> tuple[np.ndarray, str]:
    key = f"{split}_stim_idx"
    if key not in vdvae_features.files:
        raise ValueError(
            f"VDVAE features are missing {split} stimulus indices (rows={expected_rows}). "
            f"Expected NPZ key {key!r}."
        )
    stim_idx = np.asarray(vdvae_features[key])
    if stim_idx.ndim != 1 or int(stim_idx.shape[0]) != int(expected_rows):
        raise ValueError(
            f"VDVAE {key} must have shape ({expected_rows},), got {stim_idx.shape}."
        )
    stim_idx = _validate_unique_stim_idx(
        np.asarray(stim_idx, dtype=np.int64),
        name=f"VDVAE {split} stim_idx ({key})",
    )
    return stim_idx, key


def _standardize_fmri(
    train_fmri: np.ndarray,
    cond_fmri: dict[str, np.ndarray],
    fmri_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    x_train = train_fmri.astype(np.float32) / float(fmri_scale)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = _safe_std(x_train, axis=0, ddof=1)
    x_train = (x_train - x_mean) / x_std

    x_cond: dict[str, np.ndarray] = {}
    for name, arr in cond_fmri.items():
        x = arr.astype(np.float32) / float(fmri_scale)
        x_cond[name] = (x - x_mean) / x_std
    return x_train, x_cond, x_mean, x_std


def _predict_train_responses_command(
    condition: str,
    test_sub: int,
    data_root: Path,
    predictions_dir: Path,
    n_shots: int,
    seed: int,
) -> str:
    args = [
        "PYTHONPATH=.",
        shlex.quote(sys.executable),
        "-m",
        "src.pipelines.predict_train_responses",
        "--mode",
        condition,
        "--test-sub",
        str(test_sub),
        "--data-root",
        shlex.quote(str(data_root)),
        "--predictions-dir",
        shlex.quote(str(predictions_dir)),
    ]
    if condition == "few_shot":
        args.extend(
            ["--n-shots", str(n_shots), "--seed", str(seed)]
        )
    return " ".join(args)


def _load_condition_train_source(
    condition: str,
    test_sub: int,
    train_stim_idx: np.ndarray,
    train_fmri: np.ndarray,
    data_root: Path,
    predictions_dir: Path,
    n_shots: int,
    seed: int,
    fmri_scale: float,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    prediction_path = predictions_dir / f"{condition}_sub{test_sub}_train_pred.npy"
    sidecar_path = predictions_dir / f"{condition}_sub{test_sub}_train_pred.json"
    command = _predict_train_responses_command(
        condition=condition,
        test_sub=test_sub,
        data_root=data_root,
        predictions_dir=predictions_dir,
        n_shots=n_shots,
        seed=seed,
    )
    if not prediction_path.exists() or not sidecar_path.exists():
        missing = prediction_path if not prediction_path.exists() else sidecar_path
        raise RuntimeError(
            f"Missing condition-matched training prediction: {missing}. "
            f"Run this command first: {command}"
        )

    predictions = np.load(prediction_path).astype(np.float32)
    expected_rows = int(len(train_stim_idx))
    expected_voxels = int(train_fmri.shape[1])
    if predictions.ndim != 2 or int(predictions.shape[0]) != expected_rows:
        raise RuntimeError(
            f"Training prediction row mismatch for {prediction_path}: got "
            f"{predictions.shape}, expected ({expected_rows}, {expected_voxels}). "
            f"Run this command first: {command}"
        )
    if int(predictions.shape[1]) != expected_voxels:
        raise RuntimeError(
            f"Training prediction voxel mismatch for {prediction_path}: got "
            f"{predictions.shape[1]}, expected {expected_voxels}. "
            f"Run this command first: {command}"
        )

    with open(sidecar_path) as sidecar_file:
        metadata = json.load(sidecar_file)
    expected_stim_sha256 = _train_stim_idx_sha256(train_stim_idx)
    if metadata.get("train_stim_idx_sha256") != expected_stim_sha256:
        raise RuntimeError(
            f"Training stimulus hash mismatch for {sidecar_path}. "
            f"Run this command first: {command}"
        )
    if int(metadata.get("n_rows", -1)) != expected_rows:
        raise RuntimeError(
            f"Training prediction sidecar row mismatch for {sidecar_path}: got "
            f"{metadata.get('n_rows')}, expected {expected_rows}. "
            f"Run this command first: {command}"
        )
    if metadata.get("mode") != condition:
        raise RuntimeError(
            f"Training prediction mode mismatch for {sidecar_path}: got "
            f"{metadata.get('mode')!r}, expected {condition!r}. "
            f"Run this command first: {command}"
        )
    if condition == "few_shot" and (
        int(metadata.get("n_shots", -1)) != int(n_shots)
        or int(metadata.get("seed", -1)) != int(seed)
    ):
        raise RuntimeError(
            f"Few-shot training prediction selection mismatch for {sidecar_path}: "
            f"got n_shots={metadata.get('n_shots')}, seed={metadata.get('seed')}; "
            f"expected n_shots={n_shots}, seed={seed}. "
            f"Run this command first: {command}"
        )

    source = predictions / float(fmri_scale)
    source = (source - x_mean) / x_std
    return source, str(prediction_path), file_sha256(prediction_path)


def _columnwise_corr_chunks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    correlations = []
    for start in range(0, y_true.shape[1], chunk_size):
        end = min(start + chunk_size, y_true.shape[1])
        true_chunk = y_true[:, start:end]
        pred_chunk = y_pred[:, start:end]
        true_centered = true_chunk - true_chunk.mean(axis=0, keepdims=True)
        pred_centered = pred_chunk - pred_chunk.mean(axis=0, keepdims=True)
        numerator = np.sum(true_centered * pred_centered, axis=0)
        denominator = np.sqrt(
            np.sum(true_centered**2, axis=0) * np.sum(pred_centered**2, axis=0)
        )
        correlations.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float32),
                where=denominator > 1e-8,
            )
        )
    return np.concatenate(correlations).astype(np.float32)


def _column_moments_chunks(
    values: np.ndarray,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.empty(values.shape[1], dtype=np.float32)
    std = np.empty(values.shape[1], dtype=np.float32)
    for start in range(0, values.shape[1], chunk_size):
        end = min(start + chunk_size, values.shape[1])
        chunk = np.asarray(values[:, start:end], dtype=np.float64)
        mean[start:end] = chunk.mean(axis=0).astype(np.float32)
        std[start:end] = chunk.std(axis=0).astype(np.float32)
    return mean, std


def _reconstruction_feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    true_flat = y_true.reshape(y_true.shape[0], -1)
    pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"Feature metric shape mismatch: true={true_flat.shape}, pred={pred_flat.shape}."
        )
    row_r = _rowwise_corr(true_flat, pred_flat)
    target_r = _columnwise_corr_chunks(true_flat, pred_flat)
    target_mean = true_flat.mean(axis=0, keepdims=True)
    residual_ss = np.sum((true_flat - pred_flat) ** 2, axis=0)
    total_ss = np.sum((true_flat - target_mean) ** 2, axis=0)
    valid_variance = total_ss > 1e-8
    target_r2 = np.zeros_like(residual_ss, dtype=np.float32)
    target_r2[valid_variance] = 1.0 - (
        residual_ss[valid_variance] / total_ss[valid_variance]
    )
    target_r2[~valid_variance & (residual_ss <= 1e-8)] = 1.0
    return {
        "r2_vs_true_eval": float(np.mean(target_r2)),
        "mean_target_r_vs_true_eval": float(np.mean(target_r)),
        "median_target_r_vs_true_eval": float(np.median(target_r)),
        "mean_row_r_vs_true_eval": float(np.mean(row_r)),
        "median_row_r_vs_true_eval": float(np.median(row_r)),
        "true_eval_mean": float(np.mean(true_flat)),
        "pred_eval_mean": float(np.mean(pred_flat)),
        "true_eval_std": float(np.std(true_flat)),
        "pred_eval_std": float(np.std(pred_flat)),
    }


def _predict_vdvae_latents(
    x_train: np.ndarray,
    x_cond: dict[str, np.ndarray],
    train_latents: np.ndarray,
    alpha: float,
    max_iter: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    n_train, latent_dim = train_latents.shape
    if x_train.shape[0] != n_train:
        raise ValueError(
            f"VDVAE train rows mismatch: train_fmri={x_train.shape[0]}, train_latents={n_train}"
        )

    preds = {
        name: np.zeros((x.shape[0], latent_dim), dtype=np.float32) for name, x in x_cond.items()
    }

    n_chunks = int(np.ceil(latent_dim / float(chunk_size)))
    for ci, start in enumerate(range(0, latent_dim, chunk_size), start=1):
        end = min(start + chunk_size, latent_dim)
        y_train = train_latents[:, start:end].astype(np.float32)
        reg = Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        reg.fit(x_train, y_train)

        for name, x in x_cond.items():
            preds[name][:, start:end] = reg.predict(x).astype(np.float32)

        if ci == 1 or ci % 10 == 0 or ci == n_chunks:
            logger.info("VDVAE regression chunk %d/%d (%d:%d)", ci, n_chunks, start, end)

    return preds


def _predict_clip_embeddings(
    x_train: np.ndarray,
    x_cond: dict[str, np.ndarray],
    train_clip: np.ndarray,
    alpha: float,
    max_iter: int,
    label: str,
) -> dict[str, np.ndarray]:
    n_train, n_tokens, n_dim = train_clip.shape
    if x_train.shape[0] != n_train:
        raise ValueError(
            f"{label} train rows mismatch: train_fmri={x_train.shape[0]}, train_clip={n_train}"
        )

    preds = {
        name: np.zeros((x.shape[0], n_tokens, n_dim), dtype=np.float32) for name, x in x_cond.items()
    }
    for token_idx in range(n_tokens):
        y_train = train_clip[:, token_idx, :].astype(np.float32)
        reg = Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        reg.fit(x_train, y_train)

        for name, x in x_cond.items():
            preds[name][:, token_idx, :] = reg.predict(x).astype(np.float32)

        if token_idx == 0 or (token_idx + 1) % 25 == 0 or token_idx + 1 == n_tokens:
            logger.info("%s regression token %d/%d", label, token_idx + 1, n_tokens)

    return preds


def _load_vdvae_model(recon_model_root: Path):
    vdvae_dir = recon_model_root / "vdvae"
    if str(vdvae_dir) not in sys.path:
        sys.path.insert(0, str(vdvae_dir))

    from model_utils import load_vaes, set_up_data

    model_dir = vdvae_dir / "model"
    hparams = {
        "image_size": 64,
        "image_channels": 3,
        "seed": 0,
        "port": 29500,
        "save_dir": "./saved_models/test",
        "data_root": "./",
        "desc": "test",
        "hparam_sets": "imagenet64",
        "restore_path": str(model_dir / "imagenet64-iter-1600000-model.th"),
        "restore_ema_path": str(model_dir / "imagenet64-iter-1600000-model-ema.th"),
        "restore_log_path": str(model_dir / "imagenet64-iter-1600000-log.jsonl"),
        "restore_optimizer_path": str(model_dir / "imagenet64-iter-1600000-opt.th"),
        "dataset": "imagenet64",
        "ema_rate": 0.999,
        "enc_blocks": "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
        "dec_blocks": "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
        "zdim": 16,
        "width": 512,
        "custom_width_str": "",
        "bottleneck_multiple": 0.25,
        "no_bias_above": 64,
        "scale_encblock": False,
        "test_eval": True,
        "warmup_iters": 100,
        "num_mixtures": 10,
        "grad_clip": 220.0,
        "skip_threshold": 380.0,
        "lr": 0.00015,
        "lr_prior": 0.00015,
        "wd": 0.01,
        "wd_prior": 0.0,
        "num_epochs": 10000,
        "n_batch": 4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.9,
        "temperature": 1.0,
        "iters_per_ckpt": 25000,
        "iters_per_print": 1000,
        "iters_per_save": 10000,
        "iters_per_images": 10000,
        "epochs_per_eval": 1,
        "epochs_per_probe": None,
        "epochs_per_eval_save": 1,
        "num_images_visualize": 8,
        "num_variables_visualize": 6,
        "num_temperatures_visualize": 3,
        "mpi_size": 1,
        "local_rank": 0,
        "rank": 0,
        "logdir": "./saved_models/test/log",
    }

    class _DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    hparams = _DotDict(hparams)
    hparams, _ = set_up_data(hparams)
    logger.info("Loading VDVAE checkpoint from %s", model_dir)
    return load_vaes(hparams)


def _latent_transformation(
    latents: np.ndarray,
    ref_latent,
    n_prefix_layers: int | None = None,
) -> list[np.ndarray]:
    if latents.ndim != 2:
        raise ValueError(f"Expected flattened VDVAE latents [N, D], got {latents.shape}")

    expected_dim = int(_VDVAE_LAYER_DIMS.sum())
    if latents.shape[1] != expected_dim:
        raise ValueError(
            f"VDVAE latent width mismatch: got {latents.shape[1]}, expected {expected_dim}."
        )
    if (
        n_prefix_layers is not None
        and not 1 <= n_prefix_layers <= len(_VDVAE_LAYER_DIMS)
    ):
        raise ValueError(
            f"n_prefix_layers must be in [1, {len(_VDVAE_LAYER_DIMS)}], "
            f"got {n_prefix_layers}."
        )

    transformed: list[np.ndarray] = []
    start = 0
    layer_dims = _VDVAE_LAYER_DIMS
    if n_prefix_layers is not None:
        layer_dims = layer_dims[:n_prefix_layers]
    for layer_idx, width in enumerate(layer_dims.tolist()):
        end = start + int(width)
        t_lat = latents[:, start:end]
        c, h, w = ref_latent[layer_idx]["z"].shape[1:]
        transformed.append(t_lat.reshape(len(latents), c, h, w))
        start = end
    return transformed


def _decode_vdvae_latents(
    ema_vae,
    pred_latents: np.ndarray,
    ref_latent,
    out_dir: Path,
    save_rows: np.ndarray,
    save_stim: np.ndarray,
    batch_size: int,
    device: str,
    n_prefix_layers: int | None = None,
):
    import torch

    latents_hier = _latent_transformation(
        pred_latents,
        ref_latent,
        n_prefix_layers=n_prefix_layers,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(pred_latents), batch_size):
        end = min(start + batch_size, len(pred_latents))
        sample_ids = range(start, end)
        sample_latents = [
            torch.tensor(layer[sample_ids]).float().to(device) for layer in latents_hier
        ]
        with torch.no_grad():
            px_z = ema_vae.decoder.forward_manual_latents(len(sample_ids), sample_latents, t=None)
            generated = ema_vae.decoder.out_net.sample(px_z)

        for offset, arr in enumerate(generated):
            idx = start + offset
            row = int(save_rows[idx])
            stim = int(save_stim[idx])
            img = Image.fromarray(arr).resize((512, 512), resample=Image.Resampling.BICUBIC)
            img.save(out_dir / f"row{row:05d}_stim{stim}.png")


def _load_versatile_components(
    recon_model_root: Path,
    vd_weights_path: Path,
    device: str,
    precision: str,
):
    import torch

    vd_root = recon_model_root / "versatile_diffusion"
    if str(vd_root) not in sys.path:
        sys.path.insert(0, str(vd_root))

    with _pushd(recon_model_root):
        from lib.cfg_helper import model_cfg_bank
        from lib.model_zoo import get_model
        from lib.model_zoo.ddim_vd import DDIMSampler_VD

        # `model_cfg_bank` resolves config paths relative to cwd.
        cfgm = model_cfg_bank()("vd_noema")
        net = get_model()(cfgm)
    state = torch.load(vd_weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)

    net.clip = net.clip.to(device)
    net.autokl = net.autokl.to(device)
    if precision == "fp16":
        net.autokl = net.autokl.half()

    sampler = DDIMSampler_VD(net)
    sampler.model.model.diffusion_model.device = device
    sampler.model.model.diffusion_model.to(device)
    if precision == "fp16":
        sampler.model.model.diffusion_model.half()

    with torch.no_grad():
        utx = net.clip_encode_text("")
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        uim = net.clip_encode_vision(dummy)
        if precision == "fp16":
            utx = utx.half()
            uim = uim.half()
    return net, sampler, utx, uim


def _decode_versatile(
    net,
    sampler,
    utx,
    uim,
    pred_cliptext: np.ndarray,
    pred_clipvision: np.ndarray,
    init_dir: Path,
    out_dir: Path,
    save_rows: np.ndarray,
    save_stim: np.ndarray,
    device: str,
    precision: str,
    strength: float,
    mixing: float,
    guidance_scale: float,
    ddim_steps: int,
    ddim_eta: float,
):
    import torch
    import torchvision.transforms as tvtrans

    out_dir.mkdir(parents=True, exist_ok=True)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(strength * ddim_steps)

    with torch.no_grad():
        for idx in range(len(save_rows)):
            row = int(save_rows[idx])
            stim = int(save_stim[idx])
            init_img = Image.open(init_dir / f"row{row:05d}_stim{stim}.png").convert("RGB")
            init_img = init_img.resize((512, 512), resample=Image.Resampling.BICUBIC)

            zin = tvtrans.ToTensor()(init_img).to(device)
            zin = (zin * 2.0) - 1.0
            zin = zin.unsqueeze(0)
            if precision == "fp16":
                zin = zin.half()

            init_latent = net.autokl_encode(zin)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc], device=device))

            cim = torch.from_numpy(pred_clipvision[idx : idx + 1]).to(device)
            ctx = torch.from_numpy(pred_cliptext[idx : idx + 1]).to(device)
            if precision == "fp16":
                cim = cim.half()
                ctx = ctx.half()
                z_enc = z_enc.half()

            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uim, cim],
                second_conditioning=[utx, ctx],
                t_start=t_enc,
                unconditional_guidance_scale=guidance_scale,
                xtype="image",
                first_ctype="vision",
                second_ctype="prompt",
                mixed_ratio=(1.0 - mixing),
            )

            z = z.to(device)
            if precision == "fp16":
                z = z.half()
            x = net.autokl_decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            out_img = tvtrans.ToPILImage()(x[0].float().cpu())
            out_img.save(out_dir / f"row{row:05d}_stim{stim}.png")

            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(save_rows):
                logger.info("Versatile decode %d/%d", idx + 1, len(save_rows))


def run_benchmark(
    test_sub: int,
    data_root: Path,
    predictions_dir: Path,
    fewshot_dir: Path,
    output_dir: Path,
    recon_model_root: Path,
    vdvae_feature_npz: Path,
    vdvae_ref_npz: Path,
    cliptext_train_npy: Path,
    cliptext_test_npy: Path,
    clipvision_train_npy: Path,
    clipvision_test_npy: Path,
    cliptext_train_stim_idx_npy: Path,
    cliptext_test_stim_idx_npy: Path,
    clipvision_train_stim_idx_npy: Path,
    clipvision_test_stim_idx_npy: Path,
    vd_weights_path: Path,
    test_images_npy: Path | None,
    test_images_dir: Path | None,
    fewshot_n_shots: int | None,
    fewshot_seed: int | None,
    fmri_scale: float,
    vdvae_alpha: float,
    cliptext_alpha: float,
    clipvision_alpha: float,
    ridge_max_iter: int,
    vdvae_chunk_size: int,
    vdvae_batch_size: int,
    device: str,
    precision: str,
    vd_strength: float,
    vd_mixing: float,
    vd_guidance_scale: float,
    vd_ddim_steps: int,
    vd_ddim_eta: float,
    n_panels: int,
    reuse_predicted_features: bool = False,
    vdvae_calibration: str = "oof",
    vdvae_calibration_folds: int = 3,
    vdvae_calibration_seed: int = 42,
    vdvae_prefix_layers: int = len(_VDVAE_LAYER_DIMS),
    test_images_hdf5: Path | None = None,
):
    if vdvae_calibration not in {"oof", "none", "per-condition"}:
        raise ValueError(f"Unknown VDVAE calibration mode: {vdvae_calibration!r}.")
    if vdvae_calibration in {"oof", "per-condition"} and vdvae_calibration_folds < 2:
        raise ValueError(
            f"vdvae_calibration_folds must be at least 2, got {vdvae_calibration_folds}."
        )
    if not 1 <= vdvae_prefix_layers <= len(_VDVAE_LAYER_DIMS):
        raise ValueError(
            f"vdvae_prefix_layers must be in [1, {len(_VDVAE_LAYER_DIMS)}], "
            f"got {vdvae_prefix_layers}."
        )

    recon_model_root = _require_model_root(recon_model_root)
    _require_file(vdvae_feature_npz, "VDVAE feature NPZ")
    _require_file(vdvae_ref_npz, "VDVAE reference latent NPZ")
    _require_file(vd_weights_path, "Versatile Diffusion checkpoint")
    _require_file(cliptext_train_npy, "CLIP-text train features")
    _require_file(cliptext_test_npy, "CLIP-text test features")
    _require_file(clipvision_train_npy, "CLIP-vision train features")
    _require_file(clipvision_test_npy, "CLIP-vision test features")

    subj_dir = data_root / f"subj{test_sub:02d}"
    train_fmri = np.load(subj_dir / "train_fmri.npy").astype(np.float32)
    train_stim_idx = _validate_unique_stim_idx(
        np.load(subj_dir / "train_stim_idx.npy").astype(np.int64),
        name=f"subj{test_sub:02d} train_stim_idx",
    )
    gt_test_fmri = np.load(subj_dir / "test_fmri.npy").astype(np.float32)
    test_stim_idx = _validate_unique_stim_idx(
        np.load(subj_dir / "test_stim_idx.npy").astype(np.int64),
        name=f"subj{test_sub:02d} test_stim_idx",
    )
    if train_stim_idx.shape[0] != train_fmri.shape[0]:
        raise ValueError(
            f"train_stim_idx rows mismatch: train_stim_idx={train_stim_idx.shape[0]}, "
            f"train_fmri={train_fmri.shape[0]}"
        )
    if test_stim_idx.shape[0] != gt_test_fmri.shape[0]:
        raise ValueError(
            f"test_stim_idx rows mismatch: test_stim_idx={test_stim_idx.shape[0]}, "
            f"test_fmri={gt_test_fmri.shape[0]}"
        )

    zero_pred_path = predictions_dir / f"zeroshot_sub{test_sub}_pred.npy"
    zero_test_fmri = np.load(zero_pred_path).astype(np.float32)
    if zero_test_fmri.shape != gt_test_fmri.shape:
        raise ValueError(
            f"Zero-shot shape mismatch: {zero_test_fmri.shape} vs {gt_test_fmri.shape}"
        )

    few_runs = _collect_fewshot_runs(test_sub, [predictions_dir, fewshot_dir])
    preferred_n = _read_best_n_from_summary(fewshot_dir / "fewshot_summary.csv")
    few_run = _select_fewshot_run(
        few_runs,
        preferred_n=preferred_n,
        force_n=fewshot_n_shots,
        force_seed=fewshot_seed,
    )
    few_pred = np.load(few_run.pred_path).astype(np.float32)
    eval_indices = _resolve_eval_indices_for_run(len(gt_test_fmri), few_run)
    if few_pred.shape[0] != len(eval_indices):
        raise ValueError(
            f"Few-shot rows mismatch: pred={few_pred.shape[0]}, expected={len(eval_indices)} "
            f"(N={few_run.n_shots}, seed={few_run.seed})"
        )
    if few_pred.shape[1] != gt_test_fmri.shape[1]:
        raise ValueError(
            f"Few-shot voxel mismatch: pred={few_pred.shape[1]}, gt={gt_test_fmri.shape[1]}"
        )

    gt_eval = gt_test_fmri[eval_indices]
    zero_eval = zero_test_fmri[eval_indices]
    few_eval = few_pred
    stim_eval = _validate_unique_stim_idx(
        np.asarray(test_stim_idx[eval_indices], dtype=np.int64),
        name=f"subj{test_sub:02d} eval_stim_idx",
    )

    cond_fmri = {
        "gt_fmri": gt_eval,
        "zero_shot": zero_eval,
        "few_shot": few_eval,
    }

    with np.load(vdvae_feature_npz) as vdvae_features:
        train_vdvae = vdvae_features["train_latents"].astype(np.float32)
        test_vdvae = vdvae_features["test_latents"].astype(np.float32)
        vdvae_train_stim_idx, vdvae_train_stim_key = _resolve_vdvae_stim_idx(
            vdvae_features,
            split="train",
            expected_rows=train_vdvae.shape[0],
        )
        vdvae_test_stim_idx, vdvae_test_stim_key = _resolve_vdvae_stim_idx(
            vdvae_features,
            split="test",
            expected_rows=test_vdvae.shape[0],
        )

    logger.info(
        "Using VDVAE train/test stimulus indices from keys '%s' and '%s'.",
        vdvae_train_stim_key,
        vdvae_test_stim_key,
    )
    test_vdvae_eval, vdvae_eval_mask = _align_rows_by_stim_idx(
        source_arr=test_vdvae,
        source_stim_idx=vdvae_test_stim_idx,
        query_stim_idx=stim_eval,
        label="VDVAE eval",
    )

    train_cliptext, cliptext_train_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=cliptext_train_npy,
        split_stim_idx_path=cliptext_train_stim_idx_npy,
        split_label="CLIP-text train",
    )
    test_cliptext, cliptext_test_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=cliptext_test_npy,
        split_stim_idx_path=cliptext_test_stim_idx_npy,
        split_label="CLIP-text test",
    )
    test_cliptext_eval, cliptext_eval_mask = _align_rows_by_stim_idx(
        source_arr=test_cliptext,
        source_stim_idx=cliptext_test_stim_idx,
        query_stim_idx=stim_eval,
        label="CLIP-text eval",
    )

    train_clipvision, clipvision_train_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=clipvision_train_npy,
        split_stim_idx_path=clipvision_train_stim_idx_npy,
        split_label="CLIP-vision train",
    )
    test_clipvision, clipvision_test_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=clipvision_test_npy,
        split_stim_idx_path=clipvision_test_stim_idx_npy,
        split_label="CLIP-vision test",
    )
    test_clipvision_eval, clipvision_eval_mask = _align_rows_by_stim_idx(
        source_arr=test_clipvision,
        source_stim_idx=clipvision_test_stim_idx,
        query_stim_idx=stim_eval,
        label="CLIP-vision eval",
    )

    pred_feature_dir = output_dir / "predicted_features"

    if not reuse_predicted_features or vdvae_calibration in {"oof", "per-condition"}:
        x_train_all, x_cond, x_mean, x_std = _standardize_fmri(
            train_fmri,
            cond_fmri,
            fmri_scale=fmri_scale,
        )
        x_train_vdvae, train_vdvae_aligned, vdvae_train_align = _align_train_rows(
            train_matrix=x_train_all,
            train_stim_idx=train_stim_idx,
            train_targets=train_vdvae,
            label="VDVAE",
            target_train_stim_idx=vdvae_train_stim_idx,
        )

    # --reuse-predicted-features: load cached predictions instead of re-running ridge
    if reuse_predicted_features:
        cond_names = list(cond_fmri.keys())
        expected_files = [
            pred_feature_dir / f"{name}_{mod}.npy"
            for name in cond_names
            for mod in ("vdvae", "cliptext", "clipvision")
        ]
        missing = [p for p in expected_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--reuse-predicted-features: missing cached files: {[str(p) for p in missing]}"
            )
        pred_vdvae = {name: np.load(pred_feature_dir / f"{name}_vdvae.npy") for name in cond_names}
        pred_cliptext = {name: np.load(pred_feature_dir / f"{name}_cliptext.npy") for name in cond_names}
        pred_clipvision = {name: np.load(pred_feature_dir / f"{name}_clipvision.npy") for name in cond_names}
        logger.info("Loaded cached predicted features from %s", pred_feature_dir)
        # Populate train-alignment info placeholders for summary
        if vdvae_calibration == "none":
            vdvae_train_align = {"rows_used": -1, "mode": "reused_cache"}
        cliptext_train_align = {"rows_used": -1, "mode": "reused_cache"}
        clipvision_train_align = {"rows_used": -1, "mode": "reused_cache"}
    else:
        x_train_cliptext, train_cliptext_aligned, cliptext_train_align = _align_train_rows(
            train_matrix=x_train_all,
            train_stim_idx=train_stim_idx,
            train_targets=train_cliptext,
            label="CLIP-text",
            target_train_stim_idx=cliptext_train_stim_idx,
        )
        x_train_clipvision, train_clipvision_aligned, clipvision_train_align = _align_train_rows(
            train_matrix=x_train_all,
            train_stim_idx=train_stim_idx,
            train_targets=train_clipvision,
            label="CLIP-vision",
            target_train_stim_idx=clipvision_train_stim_idx,
        )

        logger.info("Regressing VDVAE latents (alpha=%s)", vdvae_alpha)
        pred_vdvae = _predict_vdvae_latents(
            x_train=x_train_vdvae,
            x_cond=x_cond,
            train_latents=train_vdvae_aligned,
            alpha=vdvae_alpha,
            max_iter=ridge_max_iter,
            chunk_size=vdvae_chunk_size,
        )

        logger.info("Regressing CLIP-text features (alpha=%s)", cliptext_alpha)
        pred_cliptext = _predict_clip_embeddings(
            x_train=x_train_cliptext,
            x_cond=x_cond,
            train_clip=train_cliptext_aligned,
            alpha=cliptext_alpha,
            max_iter=ridge_max_iter,
            label="CLIP-text",
        )

        logger.info("Regressing CLIP-vision features (alpha=%s)", clipvision_alpha)
        pred_clipvision = _predict_clip_embeddings(
            x_train=x_train_clipvision,
            x_cond=x_cond,
            train_clip=train_clipvision_aligned,
            alpha=clipvision_alpha,
            max_iter=ridge_max_iter,
            label="CLIP-vision",
        )

    calibration_summary: dict = {"mode": "none"}
    pred_vdvae_cal = pred_vdvae
    if vdvae_calibration == "oof":
        n_rows = int(train_vdvae_aligned.shape[0])
        expected_calibration_metadata = {
            "n_folds": int(vdvae_calibration_folds),
            "seed": int(vdvae_calibration_seed),
            "alpha": float(vdvae_alpha),
            "n_rows": n_rows,
        }
        calibration_path = output_dir / "vdvae_calibration.npz"
        calibration = None
        cache_hit = False
        if calibration_path.exists():
            loaded_calibration, loaded_metadata = load_calibration(calibration_path)
            if all(
                loaded_metadata.get(key) == value
                for key, value in expected_calibration_metadata.items()
            ):
                calibration = loaded_calibration
                cache_hit = True
                logger.info("Loaded matching VDVAE calibration from %s", calibration_path)
            else:
                logger.info(
                    "VDVAE calibration cache metadata mismatch; recomputing %s",
                    calibration_path,
                )

        if calibration is None:
            folds = make_folds(
                n_rows=n_rows,
                n_folds=vdvae_calibration_folds,
                seed=vdvae_calibration_seed,
            )
            oof_pred = np.empty(train_vdvae_aligned.shape, dtype=np.float32)
            for fold_idx in range(vdvae_calibration_folds):
                train_mask = folds != fold_idx
                held_out_mask = folds == fold_idx
                logger.info(
                    "Regressing VDVAE OOF fold %d/%d (%d train, %d held out)",
                    fold_idx + 1,
                    vdvae_calibration_folds,
                    int(train_mask.sum()),
                    int(held_out_mask.sum()),
                )
                fold_pred = _predict_vdvae_latents(
                    x_train=x_train_vdvae[train_mask],
                    x_cond={"oof": x_train_vdvae[held_out_mask]},
                    train_latents=train_vdvae_aligned[train_mask],
                    alpha=vdvae_alpha,
                    max_iter=ridge_max_iter,
                    chunk_size=vdvae_chunk_size,
                )
                oof_pred[held_out_mask] = fold_pred["oof"]

            oof_mean, oof_std = _column_moments_chunks(oof_pred, chunk_size=4096)
            target_mean, target_std = _column_moments_chunks(
                train_vdvae_aligned,
                chunk_size=4096,
            )
            calibration = learn_calibration(
                oof_pred_mean=oof_mean,
                oof_pred_std=oof_std,
                target_mean=target_mean,
                target_std=target_std,
                gain_cap_multiple=10.0,
            )
            calibration_metadata = {
                **expected_calibration_metadata,
                "gain_cap_multiple": 10.0,
                "n_capped": int(calibration["n_capped"]),
            }
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            save_calibration(calibration_path, calibration, calibration_metadata)
            logger.info("Saved VDVAE calibration to %s", calibration_path)

        logger.info(
            "VDVAE calibration: n_capped=%d median_gain=%.6f",
            int(calibration["n_capped"]),
            float(calibration["median_gain"]),
        )
        # This measured-fMRI OOF calibration is shared by all conditions, including
        # zero/few-shot. The residual per-condition scale mismatch (~1.3x) is a known
        # limitation, second-order relative to the 6.3x raw mismatch.
        pred_vdvae_cal = {
            name: apply_calibration(pred, calibration) for name, pred in pred_vdvae.items()
        }
        calibration_summary = {
            "mode": "oof",
            "folds": int(vdvae_calibration_folds),
            "seed": int(vdvae_calibration_seed),
            "n_capped": int(calibration["n_capped"]),
            "median_gain": float(calibration["median_gain"]),
            "prefix_layers": int(vdvae_prefix_layers),
            "cache_hit": cache_hit,
        }
    elif vdvae_calibration == "per-condition":
        n_rows = int(train_vdvae_aligned.shape[0])
        folds = make_folds(
            n_rows=n_rows,
            n_folds=vdvae_calibration_folds,
            seed=vdvae_calibration_seed,
        )
        target_mean, target_std = _column_moments_chunks(
            train_vdvae_aligned,
            chunk_size=4096,
        )
        pred_vdvae_cal = {}
        condition_calibration_summary = {}

        for condition in cond_fmri:
            train_pred_sha256 = None
            if condition == "gt_fmri":
                condition_source = x_train_all
                source_description = "measured_train_fmri"
            else:
                condition_source, source_description, train_pred_sha256 = (
                    _load_condition_train_source(
                        condition=condition,
                        test_sub=test_sub,
                        train_stim_idx=train_stim_idx,
                        train_fmri=train_fmri,
                        data_root=data_root,
                        predictions_dir=predictions_dir,
                        n_shots=few_run.n_shots,
                        seed=few_run.seed,
                        fmri_scale=fmri_scale,
                        x_mean=x_mean,
                        x_std=x_std,
                    )
                )

            condition_source_aligned, condition_targets, _ = _align_train_rows(
                train_matrix=condition_source,
                train_stim_idx=train_stim_idx,
                train_targets=train_vdvae,
                label=f"VDVAE {condition} calibration",
                target_train_stim_idx=vdvae_train_stim_idx,
            )
            expected_calibration_metadata = {
                "n_folds": int(vdvae_calibration_folds),
                "seed": int(vdvae_calibration_seed),
                "alpha": float(vdvae_alpha),
                "n_rows": n_rows,
                "condition": condition,
                "fit_inputs": "measured",
            }
            if train_pred_sha256 is not None:
                expected_calibration_metadata["train_pred_sha256"] = train_pred_sha256

            calibration_path = output_dir / f"vdvae_calibration_{condition}.npz"
            calibration = None
            cache_hit = False
            if calibration_path.exists():
                loaded_calibration, loaded_metadata = load_calibration(calibration_path)
                if all(
                    loaded_metadata.get(key) == value
                    for key, value in expected_calibration_metadata.items()
                ):
                    calibration = loaded_calibration
                    cache_hit = True
                    logger.info(
                        "Loaded matching %s VDVAE calibration from %s",
                        condition,
                        calibration_path,
                    )
                else:
                    logger.info(
                        "%s VDVAE calibration cache metadata mismatch; recomputing %s",
                        condition,
                        calibration_path,
                    )

            if calibration is None:
                oof_pred = np.empty(condition_targets.shape, dtype=np.float32)
                for fold_idx in range(vdvae_calibration_folds):
                    train_mask = folds != fold_idx
                    held_out_mask = folds == fold_idx
                    logger.info(
                        "Regressing %s VDVAE OOF fold %d/%d (%d train, %d held out)",
                        condition,
                        fold_idx + 1,
                        vdvae_calibration_folds,
                        int(train_mask.sum()),
                        int(held_out_mask.sum()),
                    )
                    # Fit on MEASURED train fMRI (matching the deployed full-data
                    # ridge) and apply to the condition's held-out source rows.
                    # Fitting on condition inputs would learn a much more shrunk
                    # ridge and blow up the calibration gains.
                    fold_pred = _predict_vdvae_latents(
                        x_train=x_train_vdvae[train_mask],
                        x_cond={"oof": condition_source_aligned[held_out_mask]},
                        train_latents=train_vdvae_aligned[train_mask],
                        alpha=vdvae_alpha,
                        max_iter=ridge_max_iter,
                        chunk_size=vdvae_chunk_size,
                    )
                    oof_pred[held_out_mask] = fold_pred["oof"]

                oof_mean, oof_std = _column_moments_chunks(
                    oof_pred,
                    chunk_size=4096,
                )
                calibration = learn_calibration(
                    oof_pred_mean=oof_mean,
                    oof_pred_std=oof_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    gain_cap_multiple=10.0,
                )
                calibration_metadata = {
                    **expected_calibration_metadata,
                    "gain_cap_multiple": 10.0,
                    "n_capped": int(calibration["n_capped"]),
                }
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                save_calibration(calibration_path, calibration, calibration_metadata)
                logger.info(
                    "Saved %s VDVAE calibration to %s",
                    condition,
                    calibration_path,
                )

            logger.info(
                "%s VDVAE calibration: n_capped=%d median_gain=%.6f",
                condition,
                int(calibration["n_capped"]),
                float(calibration["median_gain"]),
            )
            pred_vdvae_cal[condition] = apply_calibration(
                pred_vdvae[condition],
                calibration,
            )
            condition_calibration_summary[condition] = {
                "n_capped": int(calibration["n_capped"]),
                "median_gain": float(calibration["median_gain"]),
                "cache_hit": cache_hit,
                "source": source_description,
            }

        calibration_summary = {
            "mode": "per-condition",
            "conditions": condition_calibration_summary,
        }

    pred_feature_dir.mkdir(parents=True, exist_ok=True)
    recon_vdvae_dir = output_dir / "reconstructions_vdvae"
    recon_final_dir = output_dir / "reconstructions"
    panels_dir = output_dir / "panels"
    pred_feature_dir.mkdir(parents=True, exist_ok=True)
    recon_vdvae_dir.mkdir(parents=True, exist_ok=True)
    recon_final_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "subject": test_sub,
        "few_shot_selected": {
            "n_shots": few_run.n_shots,
            "seed": few_run.seed,
            "median_r": few_run.median_r,
            "metrics_path": str(few_run.metrics_path),
            "pred_path": str(few_run.pred_path),
        },
        "regression": {
            "fmri_scale": fmri_scale,
            "vdvae_alpha": vdvae_alpha,
            "cliptext_alpha": cliptext_alpha,
            "clipvision_alpha": clipvision_alpha,
            "ridge_max_iter": ridge_max_iter,
            "vdvae_chunk_size": vdvae_chunk_size,
            "vdvae_eval_rows_used": int(vdvae_eval_mask.sum()),
            "cliptext_eval_rows_used": int(cliptext_eval_mask.sum()),
            "clipvision_eval_rows_used": int(clipvision_eval_mask.sum()),
            "vdvae_train_rows_used": int(vdvae_train_align["rows_used"]),
            "cliptext_train_rows_used": int(cliptext_train_align["rows_used"]),
            "clipvision_train_rows_used": int(clipvision_train_align["rows_used"]),
            "vdvae_train_alignment_mode": str(vdvae_train_align["mode"]),
            "cliptext_train_alignment_mode": str(cliptext_train_align["mode"]),
            "clipvision_train_alignment_mode": str(clipvision_train_align["mode"]),
            "vdvae_train_stim_key": vdvae_train_stim_key or "",
            "vdvae_test_stim_key": vdvae_test_stim_key or "",
        },
        "versatile_diffusion": {
            "weights": str(vd_weights_path),
            "strength": vd_strength,
            "mixing": vd_mixing,
            "guidance_scale": vd_guidance_scale,
            "ddim_steps": vd_ddim_steps,
            "ddim_eta": vd_ddim_eta,
        },
        "vdvae_calibration": calibration_summary,
        "conditions": {},
    }

    for name in cond_fmri:
        np.save(pred_feature_dir / f"{name}_vdvae.npy", pred_vdvae[name])
        if vdvae_calibration in {"oof", "per-condition"}:
            np.save(
                pred_feature_dir / f"{name}_vdvae_calibrated.npy",
                pred_vdvae_cal[name],
            )
        np.save(pred_feature_dir / f"{name}_cliptext.npy", pred_cliptext[name])
        np.save(pred_feature_dir / f"{name}_clipvision.npy", pred_clipvision[name])
        pred_vdvae_eval = pred_vdvae[name][vdvae_eval_mask]
        pred_cliptext_eval = pred_cliptext[name][cliptext_eval_mask]
        pred_clipvision_eval = pred_clipvision[name][clipvision_eval_mask]

        vdvae_metrics = _reconstruction_feature_metrics(test_vdvae_eval, pred_vdvae_eval)
        cliptext_metrics = _reconstruction_feature_metrics(test_cliptext_eval, pred_cliptext_eval)
        clipvision_metrics = _reconstruction_feature_metrics(
            test_clipvision_eval, pred_clipvision_eval
        )
        condition_summary = {
            "vdvae_latent_r2_vs_true_eval": vdvae_metrics["r2_vs_true_eval"],
            "cliptext_r2_vs_true_eval": cliptext_metrics["r2_vs_true_eval"],
            "clipvision_r2_vs_true_eval": clipvision_metrics["r2_vs_true_eval"],
            "vdvae": vdvae_metrics,
            "cliptext": cliptext_metrics,
            "clipvision": clipvision_metrics,
        }
        if vdvae_calibration in {"oof", "per-condition"}:
            pred_vdvae_cal_eval = pred_vdvae_cal[name][vdvae_eval_mask]
            condition_summary["vdvae_calibrated"] = _reconstruction_feature_metrics(
                test_vdvae_eval,
                pred_vdvae_cal_eval,
            )
        summary["conditions"][name] = condition_summary

    ref_latent = np.load(vdvae_ref_npz, allow_pickle=True)["ref_latent"]
    ema_vae = _load_vdvae_model(recon_model_root=recon_model_root)

    for name in cond_fmri:
        logger.info("Decoding VDVAE condition: %s", name)
        _decode_vdvae_latents(
            ema_vae=ema_vae,
            pred_latents=pred_vdvae_cal[name],
            ref_latent=ref_latent,
            out_dir=recon_vdvae_dir / name,
            save_rows=eval_indices,
            save_stim=stim_eval,
            batch_size=vdvae_batch_size,
            device=device,
            n_prefix_layers=vdvae_prefix_layers,
        )

    net, sampler, utx, uim = _load_versatile_components(
        recon_model_root=recon_model_root,
        vd_weights_path=vd_weights_path,
        device=device,
        precision=precision,
    )

    for name in cond_fmri:
        logger.info("Decoding Versatile Diffusion condition: %s", name)
        _decode_versatile(
            net=net,
            sampler=sampler,
            utx=utx,
            uim=uim,
            pred_cliptext=pred_cliptext[name],
            pred_clipvision=pred_clipvision[name],
            init_dir=recon_vdvae_dir / name,
            out_dir=recon_final_dir / name,
            save_rows=eval_indices,
            save_stim=stim_eval,
            device=device,
            precision=precision,
            strength=vd_strength,
            mixing=vd_mixing,
            guidance_scale=vd_guidance_scale,
            ddim_steps=vd_ddim_steps,
            ddim_eta=vd_ddim_eta,
        )

    zero_r = _rowwise_corr(gt_eval, zero_eval)
    few_r = _rowwise_corr(gt_eval, few_eval)
    delta = few_r - zero_r
    order = np.argsort(delta)[::-1]
    panel_rows = order[: min(n_panels, len(order))]

    manifest_rows: list[dict[str, str | int | float]] = []
    stimuli_h5 = None
    if (
        not (test_images_npy is not None and test_images_npy.exists())
        and not (test_images_dir is not None and test_images_dir.exists())
        and test_images_hdf5 is not None
        and test_images_hdf5.exists()
    ):
        stimuli_h5 = h5py.File(test_images_hdf5, "r")
    try:
        for rank, local_idx in enumerate(panel_rows, start=1):
            row = int(eval_indices[local_idx])
            stim = int(stim_eval[local_idx])
            fname = f"row{row:05d}_stim{stim}.png"

            gt_img_path = recon_final_dir / "gt_fmri" / fname
            zero_img_path = recon_final_dir / "zero_shot" / fname
            few_img_path = recon_final_dir / "few_shot" / fname

            original_img = None
            if test_images_npy is not None and test_images_npy.exists():
                try:
                    original_img = _load_image_from_npy(test_images_npy, row)
                except Exception:
                    original_img = None
            elif test_images_dir is not None and test_images_dir.exists():
                original_img = _load_image_from_dir(test_images_dir, row, stim)
            elif stimuli_h5 is not None:
                original_img = Image.fromarray(stimuli_h5["imgBrick"][stim]).convert("RGB")

            panel_name = f"rank{rank:02d}_row{row:05d}_stim{stim}.png"
            _save_panel(
                out_path=panels_dir / panel_name,
                original=original_img,
                gt_path=gt_img_path,
                zero_path=zero_img_path,
                few_path=few_img_path,
                title=f"row {row} | stim {stim} | act corr zero={zero_r[local_idx]:.3f}, few={few_r[local_idx]:.3f}",
            )

            manifest_rows.append(
                {
                    "rank": rank,
                    "test_row": row,
                    "stim_id": stim,
                    "zero_activation_pattern_r": float(zero_r[local_idx]),
                    "few_activation_pattern_r": float(few_r[local_idx]),
                    "delta_few_minus_zero": float(delta[local_idx]),
                    "panel_file": panel_name,
                }
            )
    finally:
        if stimuli_h5 is not None:
            stimuli_h5.close()

    with open(output_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "test_row",
                "stim_id",
                "zero_activation_pattern_r",
                "few_activation_pattern_r",
                "delta_few_minus_zero",
                "panel_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved VDVAE+VD benchmark outputs to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run VDVAE+Versatile-Diffusion reconstruction benchmark for GT/zero/few-shot conditions."
    )
    parser.add_argument("--test-sub", type=int, default=7)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--predictions-dir", default="artifacts/predictions/subj07")
    parser.add_argument("--fewshot-dir", default="artifacts/predictions/subj07")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Benchmark output directory. Defaults to artifacts/reconstructions/subjXX.",
    )
    parser.add_argument(
        "--recon-model-root",
        default="third_party",
        help="Model root containing vdvae/ and versatile_diffusion/.",
    )
    parser.add_argument(
        "--recon-feature-dir",
        default="",
        help="Directory with local reconstruction features. Defaults to data_root/reconstruction_features/subjXX.",
    )
    parser.add_argument(
        "--test-images-npy",
        default="",
        help="Optional .npy of test images in canonical test row order.",
    )
    parser.add_argument(
        "--test-images-dir",
        default="",
        help="Optional directory of test images named by row index or stim id.",
    )
    parser.add_argument(
        "--test-images-hdf5",
        default=default_stimuli_hdf5(),
        help="NSD stimulus HDF5 used for original images when npy/dir sources are unavailable.",
    )

    parser.add_argument("--fewshot-n-shots", type=int, default=None)
    parser.add_argument("--fewshot-seed", type=int, default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-panels", type=int, default=20)
    parser.add_argument(
        "--vdvae-calibration",
        choices=("oof", "none", "per-condition"),
        default="oof",
    )
    parser.add_argument("--vdvae-calibration-folds", type=int, default=3)
    parser.add_argument("--vdvae-calibration-seed", type=int, default=42)
    parser.add_argument(
        "--vdvae-prefix-layers",
        type=int,
        default=len(_VDVAE_LAYER_DIMS),
    )
    parser.add_argument(
        "--reuse-predicted-features",
        action="store_true",
        default=False,
        help="Skip ridge regression and load cached predicted features from a previous run.",
    )
    args = parser.parse_args()
    recon_cfg = load_config(args.config)["reconstruction"]

    subj_tag = f"subj{args.test_sub:02d}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts") / "reconstructions" / subj_tag
    )
    recon_model_root = Path(args.recon_model_root)
    feature_dir = (
        Path(args.recon_feature_dir)
        if args.recon_feature_dir
        else Path(args.data_root) / "reconstruction_features" / subj_tag
    )

    vdvae_feature_npz = feature_dir / "vdvae_features.npz"
    vdvae_ref_npz = feature_dir / "ref_latents.npz"
    cliptext_train_npy = feature_dir / "cliptext_train.npy"
    cliptext_test_npy = feature_dir / "cliptext_test.npy"
    clipvision_train_npy = feature_dir / "clipvision_train.npy"
    clipvision_test_npy = feature_dir / "clipvision_test.npy"
    cliptext_train_stim_idx_npy = feature_dir / "cliptext_train_stim_idx.npy"
    cliptext_test_stim_idx_npy = feature_dir / "cliptext_test_stim_idx.npy"
    clipvision_train_stim_idx_npy = feature_dir / "clipvision_train_stim_idx.npy"
    clipvision_test_stim_idx_npy = feature_dir / "clipvision_test_stim_idx.npy"
    vd_weights_path = (
        recon_model_root
        / "versatile_diffusion"
        / "pretrained"
        / "vd-four-flow-v1-0-fp16-deprecated.pth"
    )

    test_images_npy = Path(args.test_images_npy) if args.test_images_npy else None
    test_images_dir = Path(args.test_images_dir) if args.test_images_dir else None
    test_images_hdf5 = Path(args.test_images_hdf5) if args.test_images_hdf5 else None

    run_benchmark(
        test_sub=args.test_sub,
        data_root=Path(args.data_root),
        predictions_dir=Path(args.predictions_dir),
        fewshot_dir=Path(args.fewshot_dir),
        output_dir=output_dir,
        recon_model_root=recon_model_root,
        vdvae_feature_npz=vdvae_feature_npz,
        vdvae_ref_npz=vdvae_ref_npz,
        cliptext_train_npy=cliptext_train_npy,
        cliptext_test_npy=cliptext_test_npy,
        clipvision_train_npy=clipvision_train_npy,
        clipvision_test_npy=clipvision_test_npy,
        cliptext_train_stim_idx_npy=cliptext_train_stim_idx_npy,
        cliptext_test_stim_idx_npy=cliptext_test_stim_idx_npy,
        clipvision_train_stim_idx_npy=clipvision_train_stim_idx_npy,
        clipvision_test_stim_idx_npy=clipvision_test_stim_idx_npy,
        vd_weights_path=vd_weights_path,
        test_images_npy=test_images_npy,
        test_images_dir=test_images_dir,
        fewshot_n_shots=args.fewshot_n_shots,
        fewshot_seed=args.fewshot_seed,
        fmri_scale=float(recon_cfg["fmri_scale"]),
        vdvae_alpha=float(recon_cfg["vdvae_alpha"]),
        cliptext_alpha=float(recon_cfg["cliptext_alpha"]),
        clipvision_alpha=float(recon_cfg["clipvision_alpha"]),
        ridge_max_iter=int(recon_cfg["ridge_max_iter"]),
        vdvae_chunk_size=int(recon_cfg["vdvae_chunk_size"]),
        vdvae_batch_size=int(recon_cfg["vdvae_batch_size"]),
        device=args.device,
        precision=str(recon_cfg["precision"]),
        vd_strength=float(recon_cfg["vd_strength"]),
        vd_mixing=float(recon_cfg["vd_mixing"]),
        vd_guidance_scale=float(recon_cfg["vd_guidance_scale"]),
        vd_ddim_steps=int(recon_cfg["vd_ddim_steps"]),
        vd_ddim_eta=float(recon_cfg["vd_ddim_eta"]),
        n_panels=args.n_panels,
        reuse_predicted_features=args.reuse_predicted_features,
        vdvae_calibration=args.vdvae_calibration,
        vdvae_calibration_folds=args.vdvae_calibration_folds,
        vdvae_calibration_seed=args.vdvae_calibration_seed,
        vdvae_prefix_layers=args.vdvae_prefix_layers,
        test_images_hdf5=test_images_hdf5,
    )
