"""Pure helpers for out-of-fold VDVAE latent calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_folds(n_rows: int, n_folds: int, seed: int) -> np.ndarray:
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")

    permutation = np.random.default_rng(seed).permutation(n_rows)
    folds = np.empty(n_rows, dtype=np.int64)
    folds[permutation] = np.arange(n_rows, dtype=np.int64) % n_folds
    return folds


def learn_calibration(
    oof_pred_mean,
    oof_pred_std,
    target_mean,
    target_std,
    eps: float = 1e-6,
    gain_cap_multiple: float = 10.0,
) -> dict:
    offset_in = np.asarray(oof_pred_mean, dtype=np.float32)
    source_std = np.asarray(oof_pred_std, dtype=np.float32)
    offset_out = np.asarray(target_mean, dtype=np.float32)
    target_std = np.asarray(target_std, dtype=np.float32)

    uncapped_gain = target_std / np.maximum(source_std, eps)
    median_gain = float(np.median(uncapped_gain))
    gain_cap = gain_cap_multiple * median_gain
    n_capped = int(np.count_nonzero(uncapped_gain > gain_cap))
    gain = np.minimum(uncapped_gain, gain_cap).astype(np.float32)

    return {
        "offset_in": offset_in,
        "gain": gain,
        "offset_out": offset_out,
        "n_capped": n_capped,
        "median_gain": median_gain,
    }


def apply_calibration(pred: np.ndarray, calibration: dict) -> np.ndarray:
    calibrated = (
        (np.asarray(pred) - calibration["offset_in"]) * calibration["gain"]
        + calibration["offset_out"]
    )
    return calibrated.astype(np.float32)


def save_calibration(path: Path, calibration: dict, metadata: dict) -> None:
    np.savez(
        path,
        offset_in=np.asarray(calibration["offset_in"], dtype=np.float32),
        gain=np.asarray(calibration["gain"], dtype=np.float32),
        offset_out=np.asarray(calibration["offset_out"], dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_calibration(path: Path) -> tuple[dict, dict]:
    with np.load(path) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        gain = saved["gain"].astype(np.float32)
        calibration = {
            "offset_in": saved["offset_in"].astype(np.float32),
            "gain": gain,
            "offset_out": saved["offset_out"].astype(np.float32),
            "n_capped": int(metadata["n_capped"]),
            "median_gain": float(np.median(gain)),
        }
    return calibration, metadata
