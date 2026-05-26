"""
Reconstruct only the ground-truth fMRI condition via VDVAE + Versatile Diffusion.

This is intended for controlled comparisons where the task-data definition itself
changes (for example, Brain-Diffuser-compatible 37-session prep vs the local
40-session prep), so zero-shot/few-shot predictions are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from src.pipelines.benchmark_reconstructions_vdvae_vd import _align_rows_by_stim_idx
from src.pipelines.benchmark_reconstructions_vdvae_vd import _align_train_rows
from src.pipelines.benchmark_reconstructions_vdvae_vd import _decode_vdvae_latents
from src.pipelines.benchmark_reconstructions_vdvae_vd import _decode_versatile
from src.pipelines.benchmark_reconstructions_vdvae_vd import _load_clip_split_with_stim_idx
from src.pipelines.benchmark_reconstructions_vdvae_vd import _load_vdvae_model
from src.pipelines.benchmark_reconstructions_vdvae_vd import _load_versatile_components
from src.pipelines.benchmark_reconstructions_vdvae_vd import _predict_clip_embeddings
from src.pipelines.benchmark_reconstructions_vdvae_vd import _predict_vdvae_latents
from src.pipelines.benchmark_reconstructions_vdvae_vd import _require_file
from src.pipelines.benchmark_reconstructions_vdvae_vd import _require_model_root
from src.pipelines.benchmark_reconstructions_vdvae_vd import _resolve_vdvae_stim_idx
from src.pipelines.benchmark_reconstructions_vdvae_vd import _standardize_fmri
from src.pipelines.benchmark_reconstructions_vdvae_vd import _validate_unique_stim_idx

logger = logging.getLogger(__name__)


def _resolve_path(value: str, default_path: Path) -> Path:
    return Path(value) if value else default_path


def _resolve_optional_path(value: str, default_path: Path) -> Path | None:
    if value:
        return Path(value)
    if default_path.exists():
        return default_path
    return None


def run_gt_reconstruction(
    test_sub: int,
    data_root: Path,
    output_dir: Path,
    recon_model_root: Path,
    vdvae_feature_npz: Path,
    vdvae_ref_npz: Path,
    cliptext_train_npy: Path,
    cliptext_test_npy: Path,
    clipvision_train_npy: Path,
    clipvision_test_npy: Path,
    cliptext_train_stim_idx_npy: Path | None,
    cliptext_test_stim_idx_npy: Path | None,
    clipvision_train_stim_idx_npy: Path | None,
    clipvision_test_stim_idx_npy: Path | None,
    vd_weights_path: Path,
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
):
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
    test_fmri = np.load(subj_dir / "test_fmri.npy").astype(np.float32)
    train_stim_idx = _validate_unique_stim_idx(
        np.load(subj_dir / "train_stim_idx.npy").astype(np.int64),
        name=f"subj{test_sub:02d} train_stim_idx",
    )
    test_stim_idx = _validate_unique_stim_idx(
        np.load(subj_dir / "test_stim_idx.npy").astype(np.int64),
        name=f"subj{test_sub:02d} test_stim_idx",
    )
    if train_stim_idx.shape[0] != train_fmri.shape[0]:
        raise ValueError(
            f"train_stim_idx rows mismatch: train_stim_idx={train_stim_idx.shape[0]}, "
            f"train_fmri={train_fmri.shape[0]}"
        )
    if test_stim_idx.shape[0] != test_fmri.shape[0]:
        raise ValueError(
            f"test_stim_idx rows mismatch: test_stim_idx={test_stim_idx.shape[0]}, "
            f"test_fmri={test_fmri.shape[0]}"
        )

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

    test_vdvae_eval, _ = _align_rows_by_stim_idx(
        source_arr=test_vdvae,
        source_stim_idx=vdvae_test_stim_idx,
        query_stim_idx=test_stim_idx,
        label="VDVAE GT eval",
        require_all=True,
    )

    train_cliptext, cliptext_train_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=cliptext_train_npy,
        split_stim_idx_path=cliptext_train_stim_idx_npy,
        fallback_stim_idx=vdvae_train_stim_idx,
        split_label="CLIP-text train",
    )
    test_cliptext, cliptext_test_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=cliptext_test_npy,
        split_stim_idx_path=cliptext_test_stim_idx_npy,
        fallback_stim_idx=vdvae_test_stim_idx,
        split_label="CLIP-text test",
    )
    test_cliptext_eval, _ = _align_rows_by_stim_idx(
        source_arr=test_cliptext,
        source_stim_idx=cliptext_test_stim_idx,
        query_stim_idx=test_stim_idx,
        label="CLIP-text GT eval",
        require_all=True,
    )

    train_clipvision, clipvision_train_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=clipvision_train_npy,
        split_stim_idx_path=clipvision_train_stim_idx_npy,
        fallback_stim_idx=vdvae_train_stim_idx,
        split_label="CLIP-vision train",
    )
    test_clipvision, clipvision_test_stim_idx = _load_clip_split_with_stim_idx(
        split_arr_path=clipvision_test_npy,
        split_stim_idx_path=clipvision_test_stim_idx_npy,
        fallback_stim_idx=vdvae_test_stim_idx,
        split_label="CLIP-vision test",
    )
    test_clipvision_eval, _ = _align_rows_by_stim_idx(
        source_arr=test_clipvision,
        source_stim_idx=clipvision_test_stim_idx,
        query_stim_idx=test_stim_idx,
        label="CLIP-vision GT eval",
        require_all=True,
    )

    x_train_all, x_cond = _standardize_fmri(
        train_fmri=train_fmri,
        cond_fmri={"gt_fmri": test_fmri},
        fmri_scale=fmri_scale,
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

    pred_vdvae = _predict_vdvae_latents(
        x_train=x_train_vdvae,
        x_cond=x_cond,
        train_latents=train_vdvae_aligned,
        alpha=vdvae_alpha,
        max_iter=ridge_max_iter,
        chunk_size=vdvae_chunk_size,
    )["gt_fmri"]
    pred_cliptext = _predict_clip_embeddings(
        x_train=x_train_cliptext,
        x_cond=x_cond,
        train_clip=train_cliptext_aligned,
        alpha=cliptext_alpha,
        max_iter=ridge_max_iter,
        label="CLIP-text",
    )["gt_fmri"]
    pred_clipvision = _predict_clip_embeddings(
        x_train=x_train_clipvision,
        x_cond=x_cond,
        train_clip=train_clipvision_aligned,
        alpha=clipvision_alpha,
        max_iter=ridge_max_iter,
        label="CLIP-vision",
    )["gt_fmri"]

    pred_feature_dir = output_dir / "predicted_features"
    recon_vdvae_dir = output_dir / "reconstructions_vdvae" / "gt_fmri"
    recon_final_dir = output_dir / "reconstructions" / "gt_fmri"
    pred_feature_dir.mkdir(parents=True, exist_ok=True)
    recon_vdvae_dir.mkdir(parents=True, exist_ok=True)
    recon_final_dir.mkdir(parents=True, exist_ok=True)

    np.save(pred_feature_dir / "gt_fmri_vdvae.npy", pred_vdvae)
    np.save(pred_feature_dir / "gt_fmri_cliptext.npy", pred_cliptext)
    np.save(pred_feature_dir / "gt_fmri_clipvision.npy", pred_clipvision)

    ref_latent = np.load(vdvae_ref_npz, allow_pickle=True)["ref_latent"]
    ema_vae = _load_vdvae_model(brain_diffuser_root=recon_model_root)
    _decode_vdvae_latents(
        ema_vae=ema_vae,
        pred_latents=pred_vdvae,
        ref_latent=ref_latent,
        out_dir=recon_vdvae_dir,
        save_rows=np.arange(len(test_stim_idx), dtype=np.int64),
        save_stim=test_stim_idx,
        batch_size=vdvae_batch_size,
        device=device,
    )

    net, sampler, utx, uim = _load_versatile_components(
        brain_diffuser_root=recon_model_root,
        vd_weights_path=vd_weights_path,
        device=device,
        precision=precision,
    )
    _decode_versatile(
        net=net,
        sampler=sampler,
        utx=utx,
        uim=uim,
        pred_cliptext=pred_cliptext,
        pred_clipvision=pred_clipvision,
        init_dir=recon_vdvae_dir,
        out_dir=recon_final_dir,
        save_rows=np.arange(len(test_stim_idx), dtype=np.int64),
        save_stim=test_stim_idx,
        device=device,
        precision=precision,
        strength=vd_strength,
        mixing=vd_mixing,
        guidance_scale=vd_guidance_scale,
        ddim_steps=vd_ddim_steps,
        ddim_eta=vd_ddim_eta,
    )

    summary = {
        "subject": int(test_sub),
        "train_rows": int(train_fmri.shape[0]),
        "test_rows": int(test_fmri.shape[0]),
        "task_data_summary_path": str(subj_dir / "task_data_summary.json"),
        "regression": {
            "fmri_scale": float(fmri_scale),
            "vdvae_alpha": float(vdvae_alpha),
            "cliptext_alpha": float(cliptext_alpha),
            "clipvision_alpha": float(clipvision_alpha),
            "ridge_max_iter": int(ridge_max_iter),
            "vdvae_chunk_size": int(vdvae_chunk_size),
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
            "strength": float(vd_strength),
            "mixing": float(vd_mixing),
            "guidance_scale": float(vd_guidance_scale),
            "ddim_steps": int(vd_ddim_steps),
            "ddim_eta": float(vd_ddim_eta),
            "precision": str(precision),
            "device": str(device),
        },
        "metrics": {
            "vdvae_latent_r2_vs_true": float(
                r2_score(test_vdvae_eval, pred_vdvae, multioutput="uniform_average")
            ),
            "cliptext_r2_vs_true": float(
                r2_score(
                    test_cliptext_eval.reshape(len(test_cliptext_eval), -1),
                    pred_cliptext.reshape(len(pred_cliptext), -1),
                    multioutput="uniform_average",
                )
            ),
            "clipvision_r2_vs_true": float(
                r2_score(
                    test_clipvision_eval.reshape(len(test_clipvision_eval), -1),
                    pred_clipvision.reshape(len(pred_clipvision), -1),
                    multioutput="uniform_average",
                )
            ),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved GT reconstruction summary to %s", output_dir / "summary.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Reconstruct ground-truth fMRI via VDVAE + Versatile Diffusion.")
    parser.add_argument("--test-sub", type=int, required=True)
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recon-model-root", default="/home/rothermm/brain-diffuser")
    parser.add_argument("--vdvae-feature-npz", default="")
    parser.add_argument("--vdvae-ref-npz", default="")
    parser.add_argument("--cliptext-train-npy", default="")
    parser.add_argument("--cliptext-test-npy", default="")
    parser.add_argument("--clipvision-train-npy", default="")
    parser.add_argument("--clipvision-test-npy", default="")
    parser.add_argument("--cliptext-train-stim-idx", default="")
    parser.add_argument("--cliptext-test-stim-idx", default="")
    parser.add_argument("--clipvision-train-stim-idx", default="")
    parser.add_argument("--clipvision-test-stim-idx", default="")
    parser.add_argument("--vd-weights-path", default="")
    parser.add_argument("--fmri-scale", type=float, default=300.0)
    parser.add_argument("--vdvae-alpha", type=float, default=50000.0)
    parser.add_argument("--cliptext-alpha", type=float, default=100000.0)
    parser.add_argument("--clipvision-alpha", type=float, default=60000.0)
    parser.add_argument("--ridge-max-iter", type=int, default=50000)
    parser.add_argument("--vdvae-chunk-size", type=int, default=2048)
    parser.add_argument("--vdvae-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--vd-strength", type=float, default=0.5)
    parser.add_argument("--vd-mixing", type=float, default=0.2)
    parser.add_argument("--vd-guidance-scale", type=float, default=20.0)
    parser.add_argument("--vd-ddim-steps", type=int, default=50)
    parser.add_argument("--vd-ddim-eta", type=float, default=0.0)
    args = parser.parse_args()

    subj_tag = f"subj{args.test_sub:02d}"
    data_root = Path(args.data_root)
    feature_dir = data_root / "reconstruction_features" / subj_tag
    recon_model_root = Path(args.recon_model_root)
    output_dir = Path(args.output_dir)

    run_gt_reconstruction(
        test_sub=int(args.test_sub),
        data_root=data_root,
        output_dir=output_dir,
        recon_model_root=recon_model_root,
        vdvae_feature_npz=_resolve_path(args.vdvae_feature_npz, feature_dir / "vdvae_features.npz"),
        vdvae_ref_npz=_resolve_path(args.vdvae_ref_npz, feature_dir / "ref_latents.npz"),
        cliptext_train_npy=_resolve_path(args.cliptext_train_npy, feature_dir / "cliptext_train.npy"),
        cliptext_test_npy=_resolve_path(args.cliptext_test_npy, feature_dir / "cliptext_test.npy"),
        clipvision_train_npy=_resolve_path(args.clipvision_train_npy, feature_dir / "clipvision_train.npy"),
        clipvision_test_npy=_resolve_path(args.clipvision_test_npy, feature_dir / "clipvision_test.npy"),
        cliptext_train_stim_idx_npy=_resolve_optional_path(
            args.cliptext_train_stim_idx,
            feature_dir / "cliptext_train_stim_idx.npy",
        ),
        cliptext_test_stim_idx_npy=_resolve_optional_path(
            args.cliptext_test_stim_idx,
            feature_dir / "cliptext_test_stim_idx.npy",
        ),
        clipvision_train_stim_idx_npy=_resolve_optional_path(
            args.clipvision_train_stim_idx,
            feature_dir / "clipvision_train_stim_idx.npy",
        ),
        clipvision_test_stim_idx_npy=_resolve_optional_path(
            args.clipvision_test_stim_idx,
            feature_dir / "clipvision_test_stim_idx.npy",
        ),
        vd_weights_path=_resolve_path(
            args.vd_weights_path,
            recon_model_root / "versatile_diffusion" / "pretrained" / "vd-four-flow-v1-0-fp16-deprecated.pth",
        ),
        fmri_scale=float(args.fmri_scale),
        vdvae_alpha=float(args.vdvae_alpha),
        cliptext_alpha=float(args.cliptext_alpha),
        clipvision_alpha=float(args.clipvision_alpha),
        ridge_max_iter=int(args.ridge_max_iter),
        vdvae_chunk_size=int(args.vdvae_chunk_size),
        vdvae_batch_size=int(args.vdvae_batch_size),
        device=str(args.device),
        precision=str(args.precision),
        vd_strength=float(args.vd_strength),
        vd_mixing=float(args.vd_mixing),
        vd_guidance_scale=float(args.vd_guidance_scale),
        vd_ddim_steps=int(args.vd_ddim_steps),
        vd_ddim_eta=float(args.vd_ddim_eta),
    )


if __name__ == "__main__":
    main()
