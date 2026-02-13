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
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

try:
    from .benchmark_reconstructions import (
        _collect_fewshot_runs,
        _compute_eval_indices,
        _load_image_from_dir,
        _load_image_from_npy,
        _read_best_n_from_summary,
        _rowwise_corr,
        _save_panel,
        _select_fewshot_run,
    )
except ImportError:
    from src.pipelines.benchmark_reconstructions import (
        _collect_fewshot_runs,
        _compute_eval_indices,
        _load_image_from_dir,
        _load_image_from_npy,
        _read_best_n_from_summary,
        _rowwise_corr,
        _save_panel,
        _select_fewshot_run,
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


def _slice_eval_rows(
    test_arr: np.ndarray,
    eval_indices: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = (eval_indices >= 0) & (eval_indices < test_arr.shape[0])
    n_valid = int(valid_mask.sum())
    n_total = int(len(eval_indices))
    if n_valid == 0:
        raise ValueError(
            f"{label} test rows are incompatible with eval split: "
            f"rows={test_arr.shape[0]}, eval_max={int(eval_indices.max(initial=-1))}."
        )
    if n_valid != n_total:
        logger.warning(
            "%s test rows shorter than eval split; using %d/%d rows for %s metrics "
            "(max eval idx=%d, rows=%d).",
            label,
            n_valid,
            n_total,
            label,
            int(eval_indices.max(initial=-1)),
            int(test_arr.shape[0]),
        )
    return test_arr[eval_indices[valid_mask]], valid_mask


def _find_split_stim_indices(
    features_npz: np.lib.npyio.NpzFile,
    split: str,
    expected_rows: int,
) -> tuple[np.ndarray | None, str | None]:
    split_l = split.lower()
    candidates: list[tuple[int, str, np.ndarray]] = []
    for key in features_npz.files:
        key_l = key.lower()
        if split_l not in key_l:
            continue
        if not any(tok in key_l for tok in ("stim", "img", "index", "idx", "nsd")):
            continue
        arr = np.asarray(features_npz[key])
        if arr.ndim != 1 or arr.shape[0] != expected_rows:
            continue
        try:
            arr64 = arr.astype(np.int64, copy=False)
        except (TypeError, ValueError):
            continue
        score = 0
        if "stim" in key_l:
            score += 4
        if "idx" in key_l or "index" in key_l:
            score += 3
        if "nsd" in key_l:
            score += 2
        if "img" in key_l:
            score += 1
        candidates.append((score, key, arr64))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (x[0], -len(x[1]), x[1]), reverse=True)
    _, best_key, best_arr = candidates[0]
    return best_arr, best_key


def _align_train_rows(
    train_matrix: np.ndarray,
    train_stim_idx: np.ndarray,
    train_targets: np.ndarray,
    label: str,
    target_train_stim_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    n_fmri = int(train_matrix.shape[0])
    n_target = int(train_targets.shape[0])
    if n_target == n_fmri:
        return (
            train_matrix,
            train_targets,
            {"mode": "identity", "rows_used": n_target, "rows_requested": n_target},
        )

    if target_train_stim_idx is not None:
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
            raise ValueError(
                f"{label} train rows mismatch and no target stimuli were found in "
                f"subject train_stim_idx (fmri_rows={n_fmri}, target_rows={n_target})."
            )
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

    n_common = min(n_fmri, n_target)
    logger.warning(
        "%s train rows mismatch (fmri=%d, target=%d). Falling back to first %d rows. "
        "Provide train stimulus indices in the feature NPZ for exact alignment.",
        label,
        n_fmri,
        n_target,
        n_common,
    )
    return (
        train_matrix[:n_common],
        train_targets[:n_common],
        {"mode": "prefix", "rows_used": n_common, "rows_requested": n_target},
    )


def _standardize_fmri(
    train_fmri: np.ndarray,
    cond_fmri: dict[str, np.ndarray],
    fmri_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_train = train_fmri.astype(np.float32) / float(fmri_scale)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = _safe_std(x_train, axis=0, ddof=1)
    x_train = (x_train - x_mean) / x_std

    x_cond: dict[str, np.ndarray] = {}
    for name, arr in cond_fmri.items():
        x = arr.astype(np.float32) / float(fmri_scale)
        x_cond[name] = (x - x_mean) / x_std
    return x_train, x_cond


def _renorm_to_train_distribution(pred: np.ndarray, train_chunk: np.ndarray) -> np.ndarray:
    train_mean = train_chunk.mean(axis=0, keepdims=True)
    train_std = _safe_std(train_chunk, axis=0, ddof=0)
    pred_mean = pred.mean(axis=0, keepdims=True)
    pred_std = _safe_std(pred, axis=0, ddof=0)
    return ((pred - pred_mean) / pred_std) * train_std + train_mean


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
            pred = reg.predict(x).astype(np.float32)
            preds[name][:, start:end] = _renorm_to_train_distribution(pred, y_train)

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
            pred = reg.predict(x).astype(np.float32)
            preds[name][:, token_idx, :] = _renorm_to_train_distribution(pred, y_train)

        if token_idx == 0 or (token_idx + 1) % 25 == 0 or token_idx + 1 == n_tokens:
            logger.info("%s regression token %d/%d", label, token_idx + 1, n_tokens)

    return preds


def _load_vdvae_model(brain_diffuser_root: Path):
    vdvae_dir = brain_diffuser_root / "vdvae"
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


def _latent_transformation(latents: np.ndarray, ref_latent) -> list[np.ndarray]:
    if latents.ndim != 2:
        raise ValueError(f"Expected flattened VDVAE latents [N, D], got {latents.shape}")

    expected_dim = int(_VDVAE_LAYER_DIMS.sum())
    if latents.shape[1] != expected_dim:
        raise ValueError(
            f"VDVAE latent width mismatch: got {latents.shape[1]}, expected {expected_dim}."
        )

    transformed: list[np.ndarray] = []
    start = 0
    for layer_idx, width in enumerate(_VDVAE_LAYER_DIMS.tolist()):
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
):
    import torch

    latents_hier = _latent_transformation(pred_latents, ref_latent)
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
    brain_diffuser_root: Path,
    vd_weights_path: Path,
    device: str,
    precision: str,
):
    import torch

    vd_root = brain_diffuser_root / "versatile_diffusion"
    if str(vd_root) not in sys.path:
        sys.path.insert(0, str(vd_root))

    with _pushd(brain_diffuser_root):
        from lib.cfg_helper import model_cfg_bank
        from lib.model_zoo import get_model
        from lib.model_zoo.ddim_vd import DDIMSampler_VD

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

            x = net.autokl_decode(z.to(device))
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            out_img = tvtrans.ToPILImage()(x[0].float().cpu())
            out_img.save(out_dir / f"row{row:05d}_stim{stim}.png")

            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(save_rows):
                logger.info("Versatile decode %d/%d", idx + 1, len(save_rows))


def run_benchmark(
    test_sub: int,
    data_root: Path,
    predictions_dir: Path,
    ablation_dir: Path,
    output_dir: Path,
    brain_diffuser_root: Path,
    vdvae_feature_npz: Path,
    vdvae_ref_npz: Path,
    cliptext_train_npy: Path,
    cliptext_test_npy: Path,
    clipvision_train_npy: Path,
    clipvision_test_npy: Path,
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
):
    subj_dir = data_root / f"subj{test_sub:02d}"
    train_fmri = np.load(subj_dir / "train_fmri.npy").astype(np.float32)
    train_stim_idx = np.load(subj_dir / "train_stim_idx.npy").astype(np.int64)
    gt_test_fmri = np.load(subj_dir / "test_fmri.npy").astype(np.float32)
    test_stim_idx = np.load(subj_dir / "test_stim_idx.npy").astype(np.int64)
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

    few_runs = _collect_fewshot_runs(test_sub, [predictions_dir, ablation_dir])
    preferred_n = _read_best_n_from_summary(ablation_dir / "fewshot_summary.csv")
    few_run = _select_fewshot_run(
        few_runs,
        preferred_n=preferred_n,
        force_n=fewshot_n_shots,
        force_seed=fewshot_seed,
    )
    few_pred = np.load(few_run.pred_path).astype(np.float32)
    eval_indices = _compute_eval_indices(len(gt_test_fmri), few_run.n_shots, few_run.seed)
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
    stim_eval = test_stim_idx[eval_indices]

    cond_fmri = {
        "gt_fmri": gt_eval,
        "zero_shot": zero_eval,
        "few_shot": few_eval,
    }

    vdvae_features = np.load(vdvae_feature_npz)
    train_vdvae = vdvae_features["train_latents"].astype(np.float32)
    test_vdvae = vdvae_features["test_latents"].astype(np.float32)
    vdvae_train_stim_idx, vdvae_train_stim_key = _find_split_stim_indices(
        vdvae_features,
        split="train",
        expected_rows=train_vdvae.shape[0],
    )
    if vdvae_train_stim_idx is not None and vdvae_train_stim_key is not None:
        logger.info("Using VDVAE train stimulus indices from key '%s' for row alignment.", vdvae_train_stim_key)
    test_vdvae_eval, vdvae_eval_mask = _slice_eval_rows(test_vdvae, eval_indices, "VDVAE")

    train_cliptext = np.load(cliptext_train_npy).astype(np.float32)
    test_cliptext = np.load(cliptext_test_npy).astype(np.float32)
    test_cliptext_eval, cliptext_eval_mask = _slice_eval_rows(test_cliptext, eval_indices, "CLIP-text")

    train_clipvision = np.load(clipvision_train_npy).astype(np.float32)
    test_clipvision = np.load(clipvision_test_npy).astype(np.float32)
    test_clipvision_eval, clipvision_eval_mask = _slice_eval_rows(
        test_clipvision, eval_indices, "CLIP-vision"
    )

    x_train_all, x_cond = _standardize_fmri(train_fmri, cond_fmri, fmri_scale=fmri_scale)
    cliptext_train_stim_idx = (
        vdvae_train_stim_idx if vdvae_train_stim_idx is not None and len(vdvae_train_stim_idx) == train_cliptext.shape[0] else None
    )
    clipvision_train_stim_idx = (
        vdvae_train_stim_idx if vdvae_train_stim_idx is not None and len(vdvae_train_stim_idx) == train_clipvision.shape[0] else None
    )
    x_train_vdvae, train_vdvae_aligned, vdvae_train_align = _align_train_rows(
        train_matrix=x_train_all,
        train_stim_idx=train_stim_idx,
        train_targets=train_vdvae,
        label="VDVAE",
        target_train_stim_idx=vdvae_train_stim_idx,
    )
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

    pred_feature_dir = output_dir / "predicted_features"
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
        },
        "versatile_diffusion": {
            "weights": str(vd_weights_path),
            "strength": vd_strength,
            "mixing": vd_mixing,
            "guidance_scale": vd_guidance_scale,
            "ddim_steps": vd_ddim_steps,
            "ddim_eta": vd_ddim_eta,
        },
        "conditions": {},
    }

    for name in cond_fmri:
        np.save(pred_feature_dir / f"{name}_vdvae.npy", pred_vdvae[name])
        np.save(pred_feature_dir / f"{name}_cliptext.npy", pred_cliptext[name])
        np.save(pred_feature_dir / f"{name}_clipvision.npy", pred_clipvision[name])
        pred_vdvae_eval = pred_vdvae[name][vdvae_eval_mask]
        pred_cliptext_eval = pred_cliptext[name][cliptext_eval_mask]
        pred_clipvision_eval = pred_clipvision[name][clipvision_eval_mask]

        summary["conditions"][name] = {
            "vdvae_latent_r2_vs_true_eval": float(
                r2_score(test_vdvae_eval, pred_vdvae_eval, multioutput="uniform_average")
            ),
            "cliptext_r2_vs_true_eval": float(
                r2_score(
                    test_cliptext_eval.reshape(len(test_cliptext_eval), -1),
                    pred_cliptext_eval.reshape(len(pred_cliptext_eval), -1),
                    multioutput="uniform_average",
                )
            ),
            "clipvision_r2_vs_true_eval": float(
                r2_score(
                    test_clipvision_eval.reshape(len(test_clipvision_eval), -1),
                    pred_clipvision_eval.reshape(len(pred_clipvision_eval), -1),
                    multioutput="uniform_average",
                )
            ),
        }

    ref_latent = np.load(vdvae_ref_npz, allow_pickle=True)["ref_latent"]
    ema_vae = _load_vdvae_model(brain_diffuser_root=brain_diffuser_root)

    for name in cond_fmri:
        logger.info("Decoding VDVAE condition: %s", name)
        _decode_vdvae_latents(
            ema_vae=ema_vae,
            pred_latents=pred_vdvae[name],
            ref_latent=ref_latent,
            out_dir=recon_vdvae_dir / name,
            save_rows=eval_indices,
            save_stim=stim_eval,
            batch_size=vdvae_batch_size,
            device=device,
        )

    net, sampler, utx, uim = _load_versatile_components(
        brain_diffuser_root=brain_diffuser_root,
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
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--ablation-dir", default="outputs/ablations/fewshot")
    parser.add_argument("--output-dir", default="outputs/reconstruction_benchmark_vdvae_vd/subj07")

    parser.add_argument("--brain-diffuser-root", default="/home/rothermm/brain-diffuser")
    parser.add_argument(
        "--vdvae-feature-npz",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_vdvae_features_31l.npz",
    )
    parser.add_argument(
        "--vdvae-ref-npz",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/ref_latents.npz",
    )
    parser.add_argument(
        "--cliptext-train-npy",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_cliptext_train.npy",
    )
    parser.add_argument(
        "--cliptext-test-npy",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_cliptext_test.npy",
    )
    parser.add_argument(
        "--clipvision-train-npy",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_clipvision_train.npy",
    )
    parser.add_argument(
        "--clipvision-test-npy",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_clipvision_test.npy",
    )
    parser.add_argument(
        "--vd-weights-path",
        default="/home/rothermm/brain-diffuser/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth",
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

    parser.add_argument("--fewshot-n-shots", type=int, default=None)
    parser.add_argument("--fewshot-seed", type=int, default=None)

    parser.add_argument("--fmri-scale", type=float, default=300.0)
    parser.add_argument("--vdvae-alpha", type=float, default=50000.0)
    parser.add_argument("--cliptext-alpha", type=float, default=100000.0)
    parser.add_argument("--clipvision-alpha", type=float, default=60000.0)
    parser.add_argument("--ridge-max-iter", type=int, default=50000)
    parser.add_argument("--vdvae-chunk-size", type=int, default=2048)
    parser.add_argument("--vdvae-batch-size", type=int, default=8)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--vd-strength", type=float, default=0.5)
    parser.add_argument("--vd-mixing", type=float, default=0.2)
    parser.add_argument("--vd-guidance-scale", type=float, default=20.0)
    parser.add_argument("--vd-ddim-steps", type=int, default=50)
    parser.add_argument("--vd-ddim-eta", type=float, default=0.0)

    parser.add_argument("--n-panels", type=int, default=20)
    args = parser.parse_args()

    test_images_npy = Path(args.test_images_npy) if args.test_images_npy else None
    test_images_dir = Path(args.test_images_dir) if args.test_images_dir else None

    run_benchmark(
        test_sub=args.test_sub,
        data_root=Path(args.data_root),
        predictions_dir=Path(args.predictions_dir),
        ablation_dir=Path(args.ablation_dir),
        output_dir=Path(args.output_dir),
        brain_diffuser_root=Path(args.brain_diffuser_root),
        vdvae_feature_npz=Path(args.vdvae_feature_npz),
        vdvae_ref_npz=Path(args.vdvae_ref_npz),
        cliptext_train_npy=Path(args.cliptext_train_npy),
        cliptext_test_npy=Path(args.cliptext_test_npy),
        clipvision_train_npy=Path(args.clipvision_train_npy),
        clipvision_test_npy=Path(args.clipvision_test_npy),
        vd_weights_path=Path(args.vd_weights_path),
        test_images_npy=test_images_npy,
        test_images_dir=test_images_dir,
        fewshot_n_shots=args.fewshot_n_shots,
        fewshot_seed=args.fewshot_seed,
        fmri_scale=args.fmri_scale,
        vdvae_alpha=args.vdvae_alpha,
        cliptext_alpha=args.cliptext_alpha,
        clipvision_alpha=args.clipvision_alpha,
        ridge_max_iter=args.ridge_max_iter,
        vdvae_chunk_size=args.vdvae_chunk_size,
        vdvae_batch_size=args.vdvae_batch_size,
        device=args.device,
        precision=args.precision,
        vd_strength=args.vd_strength,
        vd_mixing=args.vd_mixing,
        vd_guidance_scale=args.vd_guidance_scale,
        vd_ddim_steps=args.vd_ddim_steps,
        vd_ddim_eta=args.vd_ddim_eta,
        n_panels=args.n_panels,
    )
