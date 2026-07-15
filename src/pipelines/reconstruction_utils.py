"""Small shared helpers for the retained VDVAE + VD reconstruction path."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
    with open(summary_csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                condition = int(float(row["condition"]))
                score = float(row["median_r_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if condition > 0 and score > best_score:
                best_n = condition
                best_score = score
    return best_n


def _collect_fewshot_runs(test_sub: int, search_dirs: list[Path]) -> list[FewShotRun]:
    runs = []
    prefix = f"fewshot_sub{test_sub}_N"
    for directory in search_dirs:
        if not directory.exists():
            continue
        for filename in os.listdir(directory):
            if not filename.startswith(prefix):
                continue
            match = _FEWSHOT_METRICS_RE.fullmatch(filename)
            if match is None or int(match.group("sub")) != test_sub:
                continue
            metrics_path = directory / filename
            pred_path = directory / filename.replace("_metrics.json", "_pred.npy")
            if not pred_path.exists():
                continue
            try:
                with open(metrics_path) as handle:
                    median_r = float(json.load(handle)["median_r"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            runs.append(
                FewShotRun(
                    n_shots=int(match.group("n_shots")),
                    seed=int(match.group("seed")),
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
        filtered = [run for run in filtered if run.n_shots == force_n]
        if not filtered:
            raise FileNotFoundError(f"No few-shot runs found for N={force_n}.")
    elif preferred_n is not None:
        preferred = [run for run in filtered if run.n_shots == preferred_n]
        if preferred:
            filtered = preferred
    if force_seed is not None:
        filtered = [run for run in filtered if run.seed == force_seed]
        if not filtered:
            raise FileNotFoundError(f"No few-shot runs found for seed={force_seed}.")
    return max(filtered, key=lambda run: (run.median_r, run.n_shots, -run.seed))


def _compute_eval_indices(n_shared: int, n_shots: int, seed: int) -> np.ndarray:
    min_eval = 50
    max_shots = n_shared - min_eval
    if max_shots < 1:
        raise ValueError(f"Not enough shared stimuli: {n_shared}.")
    rng = np.random.RandomState(seed)
    shot_indices = rng.choice(n_shared, size=min(n_shots, max_shots), replace=False)
    return np.setdiff1d(np.arange(n_shared), shot_indices)


def _resolve_eval_indices_for_run(n_shared: int, run: FewShotRun) -> np.ndarray:
    try:
        with open(run.metrics_path) as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError):
        metrics = {}
    values = metrics.get("eval_indices")
    if isinstance(values, list) and values:
        indices = np.asarray(values, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= n_shared):
            raise ValueError(f"Few-shot eval indices are outside [0, {n_shared}).")
        if np.unique(indices).size != indices.size:
            raise ValueError("Few-shot eval indices contain duplicates.")
        return np.sort(indices)
    return _compute_eval_indices(n_shared, run.n_shots, run.seed)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denominator = a_centered.std(axis=1) * b_centered.std(axis=1)
    valid = denominator > 1e-10
    result = np.zeros(a.shape[0], dtype=np.float32)
    result[valid] = (
        (a_centered[valid] * b_centered[valid]).mean(axis=1) / denominator[valid]
    )
    return result


def _load_image_from_npy(images_npy: Path, row_idx: int) -> Image.Image:
    images = np.load(images_npy, mmap_mode="r")
    if row_idx < 0 or row_idx >= len(images):
        raise IndexError(f"Image row index out of range: {row_idx}")
    image = images[row_idx]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _load_image_from_dir(
    images_dir: Path,
    row_idx: int,
    stim_id: int,
) -> Image.Image | None:
    candidates = (
        images_dir / f"{row_idx}.png",
        images_dir / f"{row_idx:05d}.png",
        images_dir / f"{stim_id}.png",
        images_dir / f"{stim_id:05d}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    return None


def _save_panel(
    out_path: Path,
    original: Image.Image | None,
    gt_path: Path,
    zero_path: Path,
    few_path: Path,
    title: str,
) -> None:
    columns = []
    if original is not None:
        columns.append(("Original", original.convert("RGB")))
    columns.extend(
        [
            ("GT-fMRI recon", Image.open(gt_path).convert("RGB")),
            ("Zero-shot recon", Image.open(zero_path).convert("RGB")),
            ("Few-shot recon", Image.open(few_path).convert("RGB")),
        ]
    )
    figure, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for axis, (name, image) in zip(axes, columns):
        axis.imshow(image)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
