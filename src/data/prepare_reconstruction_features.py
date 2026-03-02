"""
Prepare local reconstruction feature bundles for VDVAE+VD benchmarking.

Outputs under data_root/reconstruction_features/subjXX:
- vdvae_features.npz
  - train_latents: (N_train, 91168)
  - test_latents:  (N_test, 91168)
  - train_stim_idx: (N_train,)
  - test_stim_idx:  (N_test,)
- ref_latents.npz
  - ref_latent: compact layer-shape descriptors used by VDVAE decoder code
- cliptext_train.npy / cliptext_test.npy
- clipvision_train.npy / clipvision_test.npy
- cliptext_*_stim_idx.npy and clipvision_*_stim_idx.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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


def _validate_sorted_unique_stim_idx(stim_idx: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(stim_idx, dtype=np.int64).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{label} must be 1D, got {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{label} is empty.")
    unique = np.unique(arr)
    if unique.size != arr.size:
        raise ValueError(f"{label} contains duplicates.")
    if not np.array_equal(unique, arr):
        raise ValueError(f"{label} must be sorted ascending.")
    return arr


def _require_model_root(model_root: Path):
    vdvae_dir = model_root / "vdvae"
    vd_dir = model_root / "versatile_diffusion"
    if not vdvae_dir.exists() or not vd_dir.exists():
        raise FileNotFoundError(
            f"Model root missing required folders: {model_root}. "
            "Expected 'vdvae/' and 'versatile_diffusion/'."
        )


def _resolve_annots_path(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            Path("data/annots/COCO_73k_annots_curated.npy"),
            Path("nsddata/experiments/nsd/COCO_73k_annots_curated.npy"),
            Path("/home/rothermm/brain-diffuser/data/annots/COCO_73k_annots_curated.npy"),
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find COCO_73k_annots_curated.npy. "
        "Pass --annots-npy or place it under data/annots/."
    )


class _StimuliDataset(Dataset):
    def __init__(
        self,
        stimuli_hdf5: Path,
        stim_idx: np.ndarray,
        image_size: int,
        mode: str,
    ):
        self.stimuli_hdf5 = stimuli_hdf5
        self.stim_idx = np.asarray(stim_idx, dtype=np.int64)
        self.image_size = int(image_size)
        if mode not in {"vdvae", "clipvision"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

        self._h5 = h5py.File(stimuli_hdf5, "r")
        self._images = self._h5["imgBrick"]

    def __len__(self) -> int:
        return int(self.stim_idx.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        stim_id = int(self.stim_idx[int(idx)])
        img_arr = self._images[stim_id]
        img = Image.fromarray(img_arr)
        img = img.resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC)

        if self.mode == "vdvae":
            # VDVAE preprocess_fn expects float tensors in HWC layout.
            return torch.from_numpy(np.asarray(img, dtype=np.float32))

        # CLIP-vision path in Versatile Diffusion uses CHW tensor in [-1, 1].
        chw = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return (chw * 2.0) - 1.0

    def close(self):
        if hasattr(self, "_h5") and self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _load_vdvae_model(model_root: Path):
    vdvae_dir = model_root / "vdvae"
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
    hparams, preprocess_fn = set_up_data(hparams)
    logger.info("Loading VDVAE model from %s", model_dir)
    ema_vae = load_vaes(hparams)
    return ema_vae, preprocess_fn


def _load_vd_model(
    model_root: Path,
    weights_path: Path,
    device: str,
):
    vd_root = model_root / "versatile_diffusion"
    if str(vd_root) not in sys.path:
        sys.path.insert(0, str(vd_root))

    with _pushd(model_root):
        from lib.cfg_helper import model_cfg_bank
        from lib.model_zoo import get_model

    cfgm = model_cfg_bank()("vd_noema")
    net = get_model()(cfgm)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)

    net.clip = net.clip.to(device)
    net.eval()
    return net


def _save_compact_ref_latents(stats: list[dict], out_path: Path):
    compact = []
    for layer in stats:
        z = layer["z"]
        c, h, w = [int(v) for v in z.shape[1:]]
        compact.append({"z": np.zeros((1, c, h, w), dtype=np.float32)})
    np.savez(out_path, ref_latent=np.array(compact, dtype=object))


def _extract_vdvae_latents(
    ema_vae,
    preprocess_fn,
    loader: DataLoader,
    n_rows: int,
    out_npy: Path,
    ref_latent_out: Path | None,
) -> tuple[int, bool]:
    lat_mm = None
    write_pos = 0
    ref_saved = False
    expected_dim = int(_VDVAE_LAYER_DIMS.sum())

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            data_input, _ = preprocess_fn(batch)
            activations = ema_vae.encoder.forward(data_input)
            _, stats = ema_vae.decoder.forward(activations, get_latents=True)
            flat_parts = [
                stats[layer_idx]["z"].detach().cpu().numpy().reshape(len(data_input), -1)
                for layer_idx in range(len(_VDVAE_LAYER_DIMS))
            ]
            batch_lat = np.hstack(flat_parts).astype(np.float32, copy=False)

            if batch_lat.shape[1] != expected_dim:
                raise ValueError(
                    f"VDVAE latent width mismatch: got {batch_lat.shape[1]}, expected {expected_dim}."
                )
            if lat_mm is None:
                lat_mm = np.lib.format.open_memmap(
                    out_npy,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_rows, batch_lat.shape[1]),
                )

            start = write_pos
            end = start + int(batch_lat.shape[0])
            lat_mm[start:end] = batch_lat
            write_pos = end

            if ref_latent_out is not None and not ref_saved:
                _save_compact_ref_latents(stats, ref_latent_out)
                ref_saved = True

            if bidx == 0 or (bidx + 1) % 20 == 0:
                logger.info("VDVAE batches processed: %d", bidx + 1)

    if write_pos != n_rows:
        raise ValueError(f"VDVAE write mismatch: wrote {write_pos}, expected {n_rows}.")
    if lat_mm is not None:
        lat_mm.flush()
    return expected_dim, ref_saved


def _extract_clipvision(
    net,
    loader: DataLoader,
    n_rows: int,
    out_npy: Path,
    device: str,
) -> tuple[int, int]:
    clip_mm = None
    write_pos = 0
    n_tokens = -1
    n_dim = -1

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            batch = batch.to(device)
            clip = net.clip_encode_vision(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            if clip_mm is None:
                n_tokens, n_dim = int(clip.shape[1]), int(clip.shape[2])
                clip_mm = np.lib.format.open_memmap(
                    out_npy,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_rows, n_tokens, n_dim),
                )

            start = write_pos
            end = start + int(clip.shape[0])
            clip_mm[start:end] = clip
            write_pos = end

            if bidx == 0 or (bidx + 1) % 20 == 0:
                logger.info("CLIP-vision batches processed: %d", bidx + 1)

    if write_pos != n_rows:
        raise ValueError(f"CLIP-vision write mismatch: wrote {write_pos}, expected {n_rows}.")
    if clip_mm is not None:
        clip_mm.flush()
    return n_tokens, n_dim


def _iter_valid_captions(caption_row: Iterable[object]) -> list[str]:
    cleaned: list[str] = []
    for cap in caption_row:
        if cap is None:
            continue
        text = str(cap).strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        cleaned = [""]
    return cleaned


def _extract_cliptext(
    net,
    annots: np.ndarray,
    stim_idx: np.ndarray,
    out_npy: Path,
) -> tuple[int, int]:
    n_rows = int(stim_idx.shape[0])
    clip_mm = None
    n_tokens = -1
    n_dim = -1

    with torch.no_grad():
        for i, stim in enumerate(stim_idx.tolist()):
            prompts = _iter_valid_captions(annots[int(stim)])
            clip = (
                net.clip_encode_text(prompts)
                .mean(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if clip_mm is None:
                n_tokens, n_dim = int(clip.shape[0]), int(clip.shape[1])
                clip_mm = np.lib.format.open_memmap(
                    out_npy,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_rows, n_tokens, n_dim),
                )
            clip_mm[i] = clip

            if i == 0 or (i + 1) % 500 == 0:
                logger.info("CLIP-text rows processed: %d/%d", i + 1, n_rows)

    if clip_mm is not None:
        clip_mm.flush()
    return n_tokens, n_dim


def _all_outputs_exist(output_dir: Path) -> bool:
    required = [
        output_dir / "vdvae_features.npz",
        output_dir / "ref_latents.npz",
        output_dir / "cliptext_train.npy",
        output_dir / "cliptext_test.npy",
        output_dir / "clipvision_train.npy",
        output_dir / "clipvision_test.npy",
        output_dir / "cliptext_train_stim_idx.npy",
        output_dir / "cliptext_test_stim_idx.npy",
        output_dir / "clipvision_train_stim_idx.npy",
        output_dir / "clipvision_test_stim_idx.npy",
    ]
    return all(p.exists() for p in required)


def prepare_reconstruction_features(
    sub: int,
    data_root: Path,
    output_dir: Path,
    stimuli_hdf5: Path,
    recon_model_root: Path,
    vd_weights_path: Path,
    annots_npy: Path | None,
    vdvae_batch_size: int,
    clipvision_batch_size: int,
    device: str,
    skip_if_exists: bool,
):
    subj_dir = data_root / f"subj{sub:02d}"
    if not subj_dir.exists():
        raise FileNotFoundError(f"Missing subject directory: {subj_dir}")
    if not stimuli_hdf5.exists():
        raise FileNotFoundError(f"Missing stimuli file: {stimuli_hdf5}")
    if not vd_weights_path.exists():
        raise FileNotFoundError(f"Missing VD weights: {vd_weights_path}")

    _require_model_root(recon_model_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_if_exists and _all_outputs_exist(output_dir):
        logger.info("All reconstruction feature outputs already exist in %s; skipping.", output_dir)
        return

    train_stim_idx = _validate_sorted_unique_stim_idx(
        np.load(subj_dir / "train_stim_idx.npy"),
        label=f"subj{sub:02d} train_stim_idx",
    )
    test_stim_idx = _validate_sorted_unique_stim_idx(
        np.load(subj_dir / "test_stim_idx.npy"),
        label=f"subj{sub:02d} test_stim_idx",
    )

    annots_path = _resolve_annots_path(annots_npy)
    annots = np.load(annots_path, allow_pickle=True)
    if annots.ndim != 2 or annots.shape[0] < 73000:
        raise ValueError(f"Unexpected annotation shape: {annots.shape} from {annots_path}")
    max_stim = int(max(train_stim_idx.max(), test_stim_idx.max()))
    if max_stim >= int(annots.shape[0]):
        raise ValueError(
            f"Annotation table too short for max stim index {max_stim}: shape={annots.shape}"
        )

    logger.info("Subject %02d: train=%d, test=%d", sub, train_stim_idx.shape[0], test_stim_idx.shape[0])
    logger.info("Using annotation table: %s", annots_path)

    train_vdvae_ds = _StimuliDataset(stimuli_hdf5, train_stim_idx, image_size=64, mode="vdvae")
    test_vdvae_ds = _StimuliDataset(stimuli_hdf5, test_stim_idx, image_size=64, mode="vdvae")
    train_clipvis_ds = _StimuliDataset(stimuli_hdf5, train_stim_idx, image_size=512, mode="clipvision")
    test_clipvis_ds = _StimuliDataset(stimuli_hdf5, test_stim_idx, image_size=512, mode="clipvision")

    train_vdvae_loader = DataLoader(train_vdvae_ds, batch_size=vdvae_batch_size, shuffle=False, num_workers=0)
    test_vdvae_loader = DataLoader(test_vdvae_ds, batch_size=vdvae_batch_size, shuffle=False, num_workers=0)
    train_clipvis_loader = DataLoader(
        train_clipvis_ds, batch_size=clipvision_batch_size, shuffle=False, num_workers=0
    )
    test_clipvis_loader = DataLoader(
        test_clipvis_ds, batch_size=clipvision_batch_size, shuffle=False, num_workers=0
    )

    train_vdvae_npy = output_dir / "_tmp_train_vdvae_latents.npy"
    test_vdvae_npy = output_dir / "_tmp_test_vdvae_latents.npy"
    ref_latent_npz = output_dir / "ref_latents.npz"

    clipvision_train_npy = output_dir / "clipvision_train.npy"
    clipvision_test_npy = output_dir / "clipvision_test.npy"
    cliptext_train_npy = output_dir / "cliptext_train.npy"
    cliptext_test_npy = output_dir / "cliptext_test.npy"

    try:
        ema_vae, preprocess_fn = _load_vdvae_model(recon_model_root)
        latent_dim, ref_saved = _extract_vdvae_latents(
            ema_vae=ema_vae,
            preprocess_fn=preprocess_fn,
            loader=test_vdvae_loader,
            n_rows=int(test_stim_idx.shape[0]),
            out_npy=test_vdvae_npy,
            ref_latent_out=ref_latent_npz,
        )
        _extract_vdvae_latents(
            ema_vae=ema_vae,
            preprocess_fn=preprocess_fn,
            loader=train_vdvae_loader,
            n_rows=int(train_stim_idx.shape[0]),
            out_npy=train_vdvae_npy,
            ref_latent_out=None,
        )
        if not ref_saved:
            raise RuntimeError("Failed to save ref_latents.npz from test split.")
        logger.info("Extracted VDVAE latents: width=%d", latent_dim)

        net = _load_vd_model(
            model_root=recon_model_root,
            weights_path=vd_weights_path,
            device=device,
        )

        vis_tokens, vis_dim = _extract_clipvision(
            net=net,
            loader=test_clipvis_loader,
            n_rows=int(test_stim_idx.shape[0]),
            out_npy=clipvision_test_npy,
            device=device,
        )
        _extract_clipvision(
            net=net,
            loader=train_clipvis_loader,
            n_rows=int(train_stim_idx.shape[0]),
            out_npy=clipvision_train_npy,
            device=device,
        )
        logger.info("Extracted CLIP-vision embeddings: tokens=%d dim=%d", vis_tokens, vis_dim)

        text_tokens, text_dim = _extract_cliptext(
            net=net,
            annots=annots,
            stim_idx=test_stim_idx,
            out_npy=cliptext_test_npy,
        )
        _extract_cliptext(
            net=net,
            annots=annots,
            stim_idx=train_stim_idx,
            out_npy=cliptext_train_npy,
        )
        logger.info("Extracted CLIP-text embeddings: tokens=%d dim=%d", text_tokens, text_dim)

        np.save(output_dir / "cliptext_train_stim_idx.npy", train_stim_idx)
        np.save(output_dir / "cliptext_test_stim_idx.npy", test_stim_idx)
        np.save(output_dir / "clipvision_train_stim_idx.npy", train_stim_idx)
        np.save(output_dir / "clipvision_test_stim_idx.npy", test_stim_idx)

        train_latents = np.load(train_vdvae_npy, mmap_mode="r")
        test_latents = np.load(test_vdvae_npy, mmap_mode="r")
        np.savez(
            output_dir / "vdvae_features.npz",
            train_latents=train_latents,
            test_latents=test_latents,
            train_stim_idx=train_stim_idx,
            test_stim_idx=test_stim_idx,
        )
        del train_latents, test_latents

        summary = {
            "subject": int(sub),
            "train_rows": int(train_stim_idx.shape[0]),
            "test_rows": int(test_stim_idx.shape[0]),
            "vdvae_latent_dim": int(latent_dim),
            "cliptext_tokens": int(text_tokens),
            "cliptext_dim": int(text_dim),
            "clipvision_tokens": int(vis_tokens),
            "clipvision_dim": int(vis_dim),
            "stimuli_hdf5": str(stimuli_hdf5),
            "annots_npy": str(annots_path),
            "recon_model_root": str(recon_model_root),
            "vd_weights_path": str(vd_weights_path),
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved reconstruction feature bundle to %s", output_dir)

    finally:
        train_vdvae_ds.close()
        test_vdvae_ds.close()
        train_clipvis_ds.close()
        test_clipvis_ds.close()
        for tmp in (train_vdvae_npy, test_vdvae_npy):
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare local reconstruction feature bundle for one subject.")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stimuli-hdf5", default="nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    parser.add_argument("--recon-model-root", default="/home/rothermm/brain-diffuser")
    parser.add_argument(
        "--vd-weights-path",
        default="",
        help="Defaults to <recon-model-root>/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth",
    )
    parser.add_argument("--annots-npy", default="")
    parser.add_argument("--vdvae-batch-size", type=int, default=8)
    parser.add_argument("--clipvision-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-if-exists", action="store_true", default=False)
    args = parser.parse_args()

    subj_tag = f"subj{args.subject:02d}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.data_root) / "reconstruction_features" / subj_tag
    )
    recon_model_root = Path(args.recon_model_root)
    vd_weights_path = (
        Path(args.vd_weights_path)
        if args.vd_weights_path
        else recon_model_root
        / "versatile_diffusion"
        / "pretrained"
        / "vd-four-flow-v1-0-fp16-deprecated.pth"
    )
    annots_npy = Path(args.annots_npy) if args.annots_npy else None

    prepare_reconstruction_features(
        sub=int(args.subject),
        data_root=Path(args.data_root),
        output_dir=output_dir,
        stimuli_hdf5=Path(args.stimuli_hdf5),
        recon_model_root=recon_model_root,
        vd_weights_path=vd_weights_path,
        annots_npy=annots_npy,
        vdvae_batch_size=int(args.vdvae_batch_size),
        clipvision_batch_size=int(args.clipvision_batch_size),
        device=str(args.device),
        skip_if_exists=bool(args.skip_if_exists),
    )
