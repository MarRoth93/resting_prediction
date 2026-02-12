"""
Benchmark reconstructions from subject-level activation conditions:
- ground-truth fMRI
- zero-shot predicted fMRI
- best few-shot predicted fMRI

Uses the SDXL-VAE latent workflow:
1) fit ridge on train fMRI -> train image latents
2) predict test latents from each activation condition
3) decode predicted latents to images
4) save side-by-side panels and summary metrics
"""

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_FEWSHOT_METRICS_RE = re.compile(
    r"fewshot_sub(?P<sub>\d+)_N(?P<n_shots>\d+)_seed(?P<seed>\d+)_metrics\.json$"
)


@dataclass(frozen=True)
class FewShotRun:
    n_shots: int
    seed: int
    median_r: float
    metrics_path: Path
    pred_path: Path


def _read_best_n_from_summary(summary_csv_path: Path | None) -> int | None:
    if summary_csv_path is None or not summary_csv_path.exists():
        return None

    best_n = None
    best_score = -np.inf
    with open(summary_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                condition = int(float(row["condition"]))
                if condition <= 0:
                    continue
                score = float(row["median_r_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_n = condition
    return best_n


def _collect_fewshot_runs(test_sub: int, search_dirs: list[Path]) -> list[FewShotRun]:
    runs: list[FewShotRun] = []
    prefix = f"fewshot_sub{test_sub}_N"

    for d in search_dirs:
        if not d.exists():
            continue
        for fname in os.listdir(d):
            if not fname.startswith(prefix) or not fname.endswith("_metrics.json"):
                continue
            match = _FEWSHOT_METRICS_RE.fullmatch(fname)
            if match is None:
                continue
            if int(match.group("sub")) != test_sub:
                continue

            n_shots = int(match.group("n_shots"))
            seed = int(match.group("seed"))
            metrics_path = d / fname
            pred_path = d / fname.replace("_metrics.json", "_pred.npy")
            if not pred_path.exists():
                continue
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                median_r = float(metrics["median_r"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

            runs.append(
                FewShotRun(
                    n_shots=n_shots,
                    seed=seed,
                    median_r=median_r,
                    metrics_path=metrics_path,
                    pred_path=pred_path,
                )
            )
    return runs


def _select_fewshot_run(
    runs: list[FewShotRun],
    preferred_n: int | None = None,
    force_n: int | None = None,
    force_seed: int | None = None,
) -> FewShotRun:
    if not runs:
        raise FileNotFoundError("No few-shot metrics/prediction files found.")

    filtered = runs
    if force_n is not None:
        filtered = [r for r in filtered if r.n_shots == force_n]
        if not filtered:
            raise FileNotFoundError(f"No few-shot runs found for N={force_n}.")
    elif preferred_n is not None:
        preferred = [r for r in filtered if r.n_shots == preferred_n]
        if preferred:
            filtered = preferred

    if force_seed is not None:
        filtered = [r for r in filtered if r.seed == force_seed]
        if not filtered:
            raise FileNotFoundError(f"No few-shot runs found for seed={force_seed}.")

    return sorted(
        filtered,
        key=lambda r: (r.median_r, r.n_shots, -r.seed),
        reverse=True,
    )[0]


def _compute_eval_indices(n_shared: int, n_shots: int, seed: int) -> np.ndarray:
    min_eval = 50
    max_shots = n_shared - min_eval
    if max_shots < 1:
        raise ValueError(
            f"Not enough shared stimuli for few-shot split: n_shared={n_shared}, "
            f"need at least {min_eval + 1}."
        )
    actual_shots = min(n_shots, max_shots)
    rng = np.random.RandomState(seed)
    shot_indices = rng.choice(n_shared, size=actual_shots, replace=False)
    return np.setdiff1d(np.arange(n_shared), shot_indices)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    a_std = a_c.std(axis=1)
    b_std = b_c.std(axis=1)
    denom = a_std * b_std
    safe = denom > 1e-10
    out = np.zeros(a.shape[0], dtype=np.float32)
    out[safe] = (a_c[safe] * b_c[safe]).mean(axis=1) / denom[safe]
    return out


def _load_image_from_npy(images_npy: Path, row_idx: int) -> Image.Image:
    arr = np.load(images_npy, mmap_mode="r")
    if row_idx < 0 or row_idx >= len(arr):
        raise IndexError(f"Image row index out of range: {row_idx}")
    img = arr[row_idx]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def _load_image_from_dir(images_dir: Path, row_idx: int, stim_id: int) -> Image.Image | None:
    candidates = [
        images_dir / f"{row_idx}.png",
        images_dir / f"{row_idx:05d}.png",
        images_dir / f"{stim_id}.png",
        images_dir / f"{stim_id:05d}.png",
    ]
    for c in candidates:
        if c.exists():
            return Image.open(c).convert("RGB")
    return None


def _decode_sdxl_latents(
    pred_latents: np.ndarray,
    latent_shape: tuple[int, int, int],
    scaling_factor: float,
    vae_repo_id: str,
    device: str,
    precision: str,
    batch_size: int,
    save_rows: np.ndarray,
    save_stim: np.ndarray,
    out_dir: Path,
):
    from diffusers import AutoencoderKL
    import torch

    kwargs = {}
    if precision == "fp16" and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16

    vae = AutoencoderKL.from_pretrained(vae_repo_id, subfolder="vae", **kwargs)
    vae.to(device)
    vae.eval()
    dtype = next(vae.parameters()).dtype

    expected = int(np.prod(latent_shape))
    if pred_latents.ndim != 2 or pred_latents.shape[1] != expected:
        raise ValueError(
            f"Predicted latent shape {pred_latents.shape} does not match expected flattened width {expected}."
        )
    lat = pred_latents.reshape((-1, *latent_shape))

    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(lat), batch_size):
            end = min(start + batch_size, len(lat))
            batch = torch.from_numpy(lat[start:end]).to(device=device, dtype=dtype)
            if scaling_factor:
                batch = batch / scaling_factor
            decoded = vae.decode(batch).sample
            decoded = torch.clamp((decoded / 2.0) + 0.5, 0.0, 1.0)
            decoded = decoded.mul(255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
            for offset, arr in enumerate(decoded):
                idx = start + offset
                row = int(save_rows[idx])
                stim = int(save_stim[idx])
                Image.fromarray(arr).save(out_dir / f"row{row:05d}_stim{stim}.png")


def _save_panel(
    out_path: Path,
    original: Image.Image | None,
    gt_path: Path,
    zero_path: Path,
    few_path: Path,
    title: str,
):
    cols: list[tuple[str, Image.Image]] = []
    if original is not None:
        cols.append(("Original", original.convert("RGB")))
    cols.extend(
        [
            ("GT-fMRI recon", Image.open(gt_path).convert("RGB")),
            ("Zero-shot recon", Image.open(zero_path).convert("RGB")),
            ("Few-shot recon", Image.open(few_path).convert("RGB")),
        ]
    )

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, (name, image) in zip(axes, cols):
        ax.imshow(image)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    test_sub: int,
    data_root: Path,
    predictions_dir: Path,
    ablation_dir: Path,
    output_dir: Path,
    sdxl_feature_npz: Path,
    sdxl_ref_npz: Path,
    test_images_npy: Path | None,
    test_images_dir: Path | None,
    fewshot_n_shots: int | None,
    fewshot_seed: int | None,
    alpha_min: float,
    alpha_max: float,
    alpha_count: int,
    cv_folds: int,
    fmri_scale: float,
    decode_batch_size: int,
    device: str,
    precision: str,
    n_panels: int,
):
    subj_dir = data_root / f"subj{test_sub:02d}"
    train_fmri = np.load(subj_dir / "train_fmri.npy").astype(np.float32)
    gt_test_fmri = np.load(subj_dir / "test_fmri.npy").astype(np.float32)
    test_stim_idx = np.load(subj_dir / "test_stim_idx.npy").astype(np.int64)

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
            f"Few-shot voxel dimension mismatch: pred={few_pred.shape[1]}, "
            f"gt={gt_test_fmri.shape[1]}"
        )

    gt_eval = gt_test_fmri[eval_indices]
    zero_eval = zero_test_fmri[eval_indices]
    few_eval = few_pred
    stim_eval = test_stim_idx[eval_indices]

    features = np.load(sdxl_feature_npz)
    train_latents = features["train_latents"].astype(np.float32)
    test_latents = features["test_latents"].astype(np.float32)
    test_latents_eval = test_latents[eval_indices]

    fmri_scaler = StandardScaler()
    train_fmri_scaled = fmri_scaler.fit_transform(train_fmri / fmri_scale)

    latent_scaler = StandardScaler()
    train_latents_scaled = latent_scaler.fit_transform(train_latents)

    n_train = train_fmri_scaled.shape[0]
    cv = max(2, min(cv_folds, n_train))
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=alpha_count)
    reg = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True, scoring="r2")
    reg.fit(train_fmri_scaled, train_latents_scaled)
    best_alpha = float(reg.alpha_)
    logger.info("Selected ridge alpha: %.3e", best_alpha)

    cond_fmri = {
        "gt_fmri": gt_eval,
        "zero_shot": zero_eval,
        "few_shot": few_eval,
    }

    pred_latents_dir = output_dir / "predicted_latents"
    recon_dir = output_dir / "reconstructions"
    panels_dir = output_dir / "panels"
    pred_latents_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(sdxl_ref_npz)
    latent_shape = tuple(ref["latent_shape"].tolist())  # type: ignore[arg-type]
    scaling_factor = float(ref["scaling_factor"]) if "scaling_factor" in ref.files else 1.0
    vae_repo = str(ref["vae_repo_id"]) if "vae_repo_id" in ref.files else "stabilityai/stable-diffusion-xl-base-1.0"

    summary = {
        "subject": test_sub,
        "few_shot_selected": {
            "n_shots": few_run.n_shots,
            "seed": few_run.seed,
            "median_r": few_run.median_r,
            "metrics_path": str(few_run.metrics_path),
            "pred_path": str(few_run.pred_path),
        },
        "ridge": {
            "alpha": best_alpha,
            "alpha_grid": [float(a) for a in alphas],
            "cv_folds_used": cv,
        },
        "conditions": {},
    }

    for name, mat in cond_fmri.items():
        x_scaled = fmri_scaler.transform(mat / fmri_scale)
        pred_scaled = reg.predict(x_scaled)
        pred_latents = latent_scaler.inverse_transform(pred_scaled).astype(np.float32)
        np.save(pred_latents_dir / f"{name}.npy", pred_latents)

        cond_r2 = float(r2_score(test_latents_eval, pred_latents))
        summary["conditions"][name] = {"latent_r2_vs_true_eval": cond_r2}

        _decode_sdxl_latents(
            pred_latents=pred_latents,
            latent_shape=latent_shape,
            scaling_factor=scaling_factor,
            vae_repo_id=vae_repo,
            device=device,
            precision=precision,
            batch_size=decode_batch_size,
            save_rows=eval_indices,
            save_stim=stim_eval,
            out_dir=recon_dir / name,
        )
        logger.info("Decoded condition %s", name)

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

        gt_img_path = recon_dir / "gt_fmri" / fname
        zero_img_path = recon_dir / "zero_shot" / fname
        few_img_path = recon_dir / "few_shot" / fname

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

    logger.info("Saved benchmark outputs to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run reconstruction benchmark for subject predictions (GT vs zero-shot vs few-shot)."
    )
    parser.add_argument("--test-sub", type=int, default=7)
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--ablation-dir", default="outputs/ablations/fewshot")
    parser.add_argument("--output-dir", default="outputs/reconstruction_benchmark/subj07")

    parser.add_argument(
        "--sdxl-feature-npz",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/nsd_sdxl_vae_features.npz",
    )
    parser.add_argument(
        "--sdxl-ref-npz",
        default="/home/rothermm/brain-diffuser/data/extracted_features/subj07/sdxl_vae_ref_latents.npz",
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

    parser.add_argument("--alpha-min", type=float, default=1e4)
    parser.add_argument("--alpha-max", type=float, default=1e7)
    parser.add_argument("--alpha-count", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--fmri-scale", type=float, default=300.0)

    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
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
        sdxl_feature_npz=Path(args.sdxl_feature_npz),
        sdxl_ref_npz=Path(args.sdxl_ref_npz),
        test_images_npy=test_images_npy,
        test_images_dir=test_images_dir,
        fewshot_n_shots=args.fewshot_n_shots,
        fewshot_seed=args.fewshot_seed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_count=args.alpha_count,
        cv_folds=args.cv_folds,
        fmri_scale=args.fmri_scale,
        decode_batch_size=args.decode_batch_size,
        device=args.device,
        precision=args.precision,
        n_panels=args.n_panels,
    )
