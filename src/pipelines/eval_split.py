"""
Utilities for fixed held-out evaluation splits.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


def fixed_eval_split_path(model_dir: str, test_sub: int) -> str:
    """Path to persisted fixed eval split for one subject."""
    return os.path.join(model_dir, "eval_splits", f"sub{test_sub:02d}_fixed_eval_split.npz")


def _validate_eval_indices(eval_indices: np.ndarray, n_shared: int) -> np.ndarray:
    """Validate and canonicalize eval indices."""
    eval_indices = np.array(eval_indices, dtype=np.int64).ravel()
    if eval_indices.size == 0:
        raise ValueError("eval_indices is empty.")
    if np.any(eval_indices < 0) or np.any(eval_indices >= n_shared):
        raise ValueError(
            f"eval_indices out of bounds for n_shared={n_shared}: "
            f"min={int(eval_indices.min())}, max={int(eval_indices.max())}."
        )
    if np.unique(eval_indices).size != eval_indices.size:
        raise ValueError("eval_indices contains duplicates.")
    return np.sort(eval_indices)


def load_or_create_fixed_eval_split(
    model_dir: str,
    test_sub: int,
    n_shared: int,
    eval_size: int = 250,
    seed: int = 42,
) -> tuple[np.ndarray, dict, str]:
    """
    Load an existing fixed eval split from disk, or create and persist one.
    """
    if n_shared <= 1:
        raise ValueError(f"n_shared must be >1, got {n_shared}.")
    if eval_size < 1:
        raise ValueError(f"eval_size must be >=1, got {eval_size}.")
    if eval_size >= n_shared:
        raise ValueError(
            f"eval_size={eval_size} must be smaller than n_shared={n_shared}."
        )

    split_path = fixed_eval_split_path(model_dir, test_sub)
    split_dir = os.path.dirname(split_path)
    os.makedirs(split_dir, exist_ok=True)

    if os.path.exists(split_path):
        data = np.load(split_path, allow_pickle=True)
        saved_n_shared = int(data["n_shared"])
        if saved_n_shared != n_shared:
            raise ValueError(
                f"Fixed eval split n_shared mismatch for subject {test_sub}: "
                f"artifact has {saved_n_shared}, current data has {n_shared}. "
                f"Delete {split_path} and regenerate."
            )
        eval_indices = _validate_eval_indices(data["eval_indices"], n_shared)
        metadata = {
            "subject": int(test_sub),
            "mode": "fixed",
            "n_shared": int(n_shared),
            "eval_size": int(data["eval_size"]),
            "seed": int(data["seed"]),
            "source": "loaded",
            "split_path": split_path,
        }
        return eval_indices, metadata, split_path

    rng = np.random.RandomState(seed)
    eval_indices = np.sort(rng.choice(n_shared, size=eval_size, replace=False).astype(np.int64))
    metadata_json = json.dumps(
        {
            "subject": int(test_sub),
            "n_shared": int(n_shared),
            "eval_size": int(eval_size),
            "seed": int(seed),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    np.savez(
        split_path,
        eval_indices=eval_indices,
        n_shared=np.int64(n_shared),
        eval_size=np.int64(eval_size),
        seed=np.int64(seed),
        metadata=metadata_json,
    )
    metadata = {
        "subject": int(test_sub),
        "mode": "fixed",
        "n_shared": int(n_shared),
        "eval_size": int(eval_size),
        "seed": int(seed),
        "source": "created",
        "split_path": split_path,
    }
    return eval_indices, metadata, split_path


def sample_shot_indices_from_fixed_eval(
    n_shared: int,
    eval_indices: np.ndarray,
    n_shots: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """
    Sample few-shot indices from the complement of a fixed eval split.

    Returns:
        shot_indices: sorted row indices used as few-shot support.
        actual_shots: clamped number of sampled rows.
    """
    eval_indices = _validate_eval_indices(eval_indices, n_shared)
    pool = np.setdiff1d(np.arange(n_shared, dtype=np.int64), eval_indices, assume_unique=True)
    if pool.size == 0:
        raise ValueError("No rows available for few-shot support after fixed eval split.")
    if n_shots < 1:
        raise ValueError(f"n_shots must be >=1, got {n_shots}.")

    actual_shots = int(min(int(n_shots), int(pool.size)))
    rng = np.random.RandomState(seed)
    shot_indices = np.sort(rng.choice(pool, size=actual_shots, replace=False).astype(np.int64))
    return shot_indices, actual_shots

