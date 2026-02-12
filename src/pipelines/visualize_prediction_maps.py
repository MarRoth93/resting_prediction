"""
Render side-by-side brain activation maps:
ground truth vs zero-shot prediction vs best few-shot prediction.
"""

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import pattern_correlation

logger = logging.getLogger(__name__)

_FEWSHOT_METRICS_RE = re.compile(
    r"fewshot_sub(?P<sub>\d+)_N(?P<n_shots>\d+)_seed(?P<seed>\d+)_metrics\.json$"
)


@dataclass(frozen=True)
class FewShotRun:
    n_shots: int
    seed: int
    median_r: float
    metrics_path: str
    pred_path: str


def _existing_dirs(*dirs: str) -> list[str]:
    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _find_zeroshot_prediction(test_sub: int, search_dirs: list[str]) -> str:
    filename = f"zeroshot_sub{test_sub}_pred.npy"
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {search_dirs}"
    )


def _read_best_n_from_summary(summary_csv_path: str | None) -> int | None:
    if not summary_csv_path or not os.path.exists(summary_csv_path):
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


def _collect_fewshot_runs(test_sub: int, search_dirs: list[str]) -> list[FewShotRun]:
    runs_by_key: dict[tuple[int, int], FewShotRun] = {}
    prefix = f"fewshot_sub{test_sub}_N"

    for d in search_dirs:
        for fname in os.listdir(d):
            if not fname.startswith(prefix) or not fname.endswith("_metrics.json"):
                continue
            m = _FEWSHOT_METRICS_RE.fullmatch(fname)
            if m is None:
                continue

            sub = int(m.group("sub"))
            if sub != test_sub:
                continue

            n_shots = int(m.group("n_shots"))
            seed = int(m.group("seed"))
            metrics_path = os.path.join(d, fname)
            pred_path = metrics_path.replace("_metrics.json", "_pred.npy")
            if not os.path.exists(pred_path):
                continue

            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                median_r = float(metrics["median_r"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            run = FewShotRun(
                n_shots=n_shots,
                seed=seed,
                median_r=median_r,
                metrics_path=metrics_path,
                pred_path=pred_path,
            )
            key = (n_shots, seed)
            if key not in runs_by_key or run.median_r > runs_by_key[key].median_r:
                runs_by_key[key] = run

    return list(runs_by_key.values())


def _select_fewshot_run(
    runs: list[FewShotRun],
    preferred_n: int | None = None,
    force_n: int | None = None,
    force_seed: int | None = None,
) -> FewShotRun:
    if not runs:
        raise FileNotFoundError("No few-shot runs found.")

    filtered = runs
    if force_n is not None:
        filtered = [r for r in filtered if r.n_shots == force_n]
        if not filtered:
            raise FileNotFoundError(f"No few-shot runs found with N={force_n}.")
    elif preferred_n is not None:
        preferred = [r for r in filtered if r.n_shots == preferred_n]
        if preferred:
            filtered = preferred

    if force_seed is not None:
        filtered = [r for r in filtered if r.seed == force_seed]
        if not filtered:
            raise FileNotFoundError(
                f"No few-shot runs found with seed={force_seed} "
                f"(after applying N filter)."
            )

    filtered = sorted(
        filtered,
        key=lambda r: (r.median_r, r.n_shots, -r.seed),
        reverse=True,
    )
    return filtered[0]


def _compute_eval_indices(n_shared: int, n_shots: int, seed: int) -> np.ndarray:
    min_eval = 50
    max_shots = n_shared - min_eval
    if max_shots < 1:
        raise ValueError(
            f"Not enough shared stimuli: n_shared={n_shared}, need at least {min_eval + 1}."
        )
    actual_shots = min(n_shots, max_shots)
    rng = np.random.RandomState(seed)
    shot_indices = rng.choice(n_shared, size=actual_shots, replace=False)
    return np.setdiff1d(np.arange(n_shared), shot_indices)


def _vector_to_volume(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = values
    return vol


def _peak_coordinate(gt_vol: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    voxels = np.argwhere(mask)
    if voxels.size == 0:
        x, y, z = np.array(gt_vol.shape) // 2
        return int(x), int(y), int(z)

    abs_vals = np.abs(gt_vol[mask])
    if np.all(abs_vals <= 0):
        center = voxels[len(voxels) // 2]
        return int(center[0]), int(center[1]), int(center[2])

    peak = voxels[int(np.argmax(abs_vals))]
    return int(peak[0]), int(peak[1]), int(peak[2])


def _orth_slices(vol: np.ndarray, xyz: tuple[int, int, int]) -> list[np.ndarray]:
    x, y, z = xyz
    return [
        np.rot90(vol[x, :, :]),   # sagittal
        np.rot90(vol[:, y, :]),   # coronal
        np.rot90(vol[:, :, z]),   # axial
    ]


def _save_comparison_figure(
    gt_vec: np.ndarray,
    zero_vec: np.ndarray,
    few_vec: np.ndarray,
    mask: np.ndarray,
    stim_id: int,
    test_row: int,
    zero_r: float,
    few_r: float,
    few_run: FewShotRun,
    out_path: str,
):
    gt_vol = _vector_to_volume(gt_vec, mask)
    zero_vol = _vector_to_volume(zero_vec, mask)
    few_vol = _vector_to_volume(few_vec, mask)

    coord = _peak_coordinate(gt_vol, mask)
    gt_slices = _orth_slices(gt_vol, coord)
    zero_slices = _orth_slices(zero_vol, coord)
    few_slices = _orth_slices(few_vol, coord)

    stacked = np.concatenate([gt_vec, zero_vec, few_vec])
    vmax = float(np.percentile(np.abs(stacked), 99.0))
    if vmax < 1e-6:
        vmax = 1e-6

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    col_titles = ["Ground Truth", "Zero-shot", f"Few-shot (N={few_run.n_shots}, seed={few_run.seed})"]
    row_titles = ["Sagittal", "Coronal", "Axial"]
    cols = [gt_slices, zero_slices, few_slices]

    im = None
    for r in range(3):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(cols[c][r], cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)

    fig.suptitle(
        f"Stim {stim_id} (test row {test_row})  |  pattern r: zero={zero_r:.3f}, few={few_r:.3f}",
        fontsize=12,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label("Activation (beta)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _choose_example_indices(
    few_pattern_r: np.ndarray,
    n_examples: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    n = len(few_pattern_r)
    k = min(n_examples, n)
    if mode == "top":
        order = np.argsort(few_pattern_r)[::-1]
        return order[:k]
    if mode == "worst":
        order = np.argsort(few_pattern_r)
        return order[:k]

    rng = np.random.RandomState(seed)
    return rng.choice(n, size=k, replace=False)


def generate_images(
    test_sub: int,
    data_root: str,
    predictions_dir: str,
    ablation_dir: str,
    output_dir: str,
    n_examples: int,
    example_mode: str,
    example_seed: int,
    fewshot_n_shots: int | None,
    fewshot_seed: int | None,
):
    os.makedirs(output_dir, exist_ok=True)

    search_dirs = _existing_dirs(predictions_dir, ablation_dir)
    if not search_dirs:
        raise FileNotFoundError(
            f"No existing prediction directories found: {predictions_dir}, {ablation_dir}"
        )

    summary_csv = os.path.join(ablation_dir, "fewshot_summary.csv")
    preferred_n = _read_best_n_from_summary(summary_csv)
    if preferred_n is not None:
        logger.info(f"Best few-shot N from summary: {preferred_n}")
    else:
        logger.info("No fewshot_summary.csv found or parseable; selecting best single few-shot run.")

    fewshot_runs = _collect_fewshot_runs(test_sub, search_dirs)
    fewshot_run = _select_fewshot_run(
        fewshot_runs,
        preferred_n=preferred_n,
        force_n=fewshot_n_shots,
        force_seed=fewshot_seed,
    )
    logger.info(
        "Selected few-shot run: N=%d seed=%d median_r=%.4f (%s)",
        fewshot_run.n_shots,
        fewshot_run.seed,
        fewshot_run.median_r,
        fewshot_run.metrics_path,
    )

    zero_pred_path = _find_zeroshot_prediction(test_sub, search_dirs)
    logger.info(f"Using zero-shot prediction: {zero_pred_path}")

    subj_dir = os.path.join(data_root, f"subj{test_sub:02d}")
    test_fmri = np.load(os.path.join(subj_dir, "test_fmri.npy"), mmap_mode="r")
    test_stim_idx = np.load(os.path.join(subj_dir, "test_stim_idx.npy"))
    mask = np.load(os.path.join(subj_dir, "mask.npy")) > 0

    if int(mask.sum()) != int(test_fmri.shape[1]):
        raise ValueError(
            f"Mask voxel count ({int(mask.sum())}) does not match test_fmri V ({test_fmri.shape[1]})."
        )

    zero_pred = np.load(zero_pred_path, mmap_mode="r")
    few_pred = np.load(fewshot_run.pred_path, mmap_mode="r")

    if zero_pred.shape != test_fmri.shape:
        raise ValueError(
            f"Zero-shot shape mismatch: pred={zero_pred.shape}, truth={test_fmri.shape}"
        )

    eval_indices = _compute_eval_indices(
        n_shared=test_fmri.shape[0],
        n_shots=fewshot_run.n_shots,
        seed=fewshot_run.seed,
    )
    if few_pred.shape[0] != len(eval_indices):
        raise ValueError(
            "Few-shot shape mismatch: "
            f"pred rows={few_pred.shape[0]}, expected eval rows={len(eval_indices)} "
            f"for N={fewshot_run.n_shots}, seed={fewshot_run.seed}."
        )

    gt_eval = np.asarray(test_fmri[eval_indices], dtype=np.float32)
    zero_eval = np.asarray(zero_pred[eval_indices], dtype=np.float32)
    few_eval = np.asarray(few_pred, dtype=np.float32)
    stim_eval = np.asarray(test_stim_idx[eval_indices], dtype=np.int64)

    zero_pattern_r = pattern_correlation(gt_eval, zero_eval)
    few_pattern_r = pattern_correlation(gt_eval, few_eval)

    selected = _choose_example_indices(
        few_pattern_r=few_pattern_r,
        n_examples=n_examples,
        mode=example_mode,
        seed=example_seed,
    )

    manifest_rows: list[dict[str, float | int | str]] = []
    for rank, eval_row in enumerate(selected, start=1):
        global_row = int(eval_indices[eval_row])
        stim_id = int(stim_eval[eval_row])
        zero_r = float(zero_pattern_r[eval_row])
        few_r = float(few_pattern_r[eval_row])

        out_name = (
            f"sub{test_sub:02d}_rank{rank:02d}_stim{stim_id}"
            f"_fewN{fewshot_run.n_shots}_seed{fewshot_run.seed}.png"
        )
        out_path = os.path.join(output_dir, out_name)

        _save_comparison_figure(
            gt_vec=gt_eval[eval_row],
            zero_vec=zero_eval[eval_row],
            few_vec=few_eval[eval_row],
            mask=mask,
            stim_id=stim_id,
            test_row=global_row,
            zero_r=zero_r,
            few_r=few_r,
            few_run=fewshot_run,
            out_path=out_path,
        )

        manifest_rows.append(
            {
                "rank": rank,
                "stim_id": stim_id,
                "test_row": global_row,
                "zero_pattern_r": zero_r,
                "few_pattern_r": few_r,
                "figure": out_name,
            }
        )

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "stim_id",
                "test_row",
                "zero_pattern_r",
                "few_pattern_r",
                "figure",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("Saved %d figures to %s", len(manifest_rows), output_dir)
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Visualize GT vs zero-shot vs few-shot predicted activations."
    )
    parser.add_argument("--test-sub", type=int, default=7)
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--ablation-dir", default="outputs/ablations/fewshot")
    parser.add_argument("--output-dir", default="outputs/visualizations/prediction_maps")
    parser.add_argument("--n-examples", type=int, default=12)
    parser.add_argument(
        "--example-mode",
        choices=["top", "random", "worst"],
        default="top",
        help="How to choose examples from the few-shot eval set.",
    )
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument(
        "--fewshot-n-shots",
        type=int,
        default=None,
        help="Optional override. If set, pick best seed for this N unless --fewshot-seed is also set.",
    )
    parser.add_argument(
        "--fewshot-seed",
        type=int,
        default=None,
        help="Optional override. Requires files for the chosen seed to exist.",
    )
    args = parser.parse_args()

    generate_images(
        test_sub=args.test_sub,
        data_root=args.data_root,
        predictions_dir=args.predictions_dir,
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
        example_mode=args.example_mode,
        example_seed=args.example_seed,
        fewshot_n_shots=args.fewshot_n_shots,
        fewshot_seed=args.fewshot_seed,
    )
