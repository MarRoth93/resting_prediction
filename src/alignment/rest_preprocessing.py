"""Compute external seed-to-voxel connectivity from preprocessed REST runs."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_rest_connectivity(
    rest_runs: list[np.ndarray],
    seed_runs: list[np.ndarray],
    ensemble: str = "average",
) -> np.ndarray:
    """Return a shared-seed by subject-voxel connectivity matrix."""
    if not rest_runs:
        raise ValueError("rest_runs is empty; at least one REST run is required")
    if len(rest_runs) != len(seed_runs):
        raise ValueError(
            f"rest_runs and seed_runs must have the same length, got "
            f"{len(rest_runs)} and {len(seed_runs)}."
        )
    if not seed_runs:
        raise ValueError("seed_runs is empty")

    n_seeds = int(seed_runs[0].shape[1])
    n_voxels = int(rest_runs[0].shape[1])
    for i, (rest, seed) in enumerate(zip(rest_runs, seed_runs), start=1):
        if rest.ndim != 2 or seed.ndim != 2:
            raise ValueError(
                f"Run {i}: rest and seed arrays must both be 2D, got "
                f"{rest.shape} and {seed.shape}."
            )
        if int(rest.shape[0]) != int(seed.shape[0]):
            raise ValueError(
                f"Run {i}: rest/seed TR mismatch, rest={rest.shape[0]}, "
                f"seed={seed.shape[0]}."
            )
        if int(rest.shape[1]) != n_voxels:
            raise ValueError(
                f"Run {i}: inconsistent rest voxel count {rest.shape[1]} vs {n_voxels}."
            )
        if int(seed.shape[1]) != n_seeds:
            raise ValueError(
                f"Run {i}: inconsistent seed count {seed.shape[1]} vs {n_seeds}."
            )

    if ensemble == "concat":
        rest_concat = np.concatenate(rest_runs, axis=0)
        seed_concat = np.concatenate(seed_runs, axis=0)
        C = _correlate(seed_concat, rest_concat)
    elif ensemble == "average":
        Cs = [
            _correlate(seed, rest)
            for rest, seed in zip(rest_runs, seed_runs)
        ]
        C = np.mean(Cs, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble}")

    logger.info(f"External seed-bank connectivity: {C.shape}")
    return C.astype(np.float32)


def _correlate(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Column-wise Pearson correlation between A (T, M) and B (T, N).

    Returns (M, N) correlation matrix.
    """
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    A_std = A.std(axis=0)
    B_std = B.std(axis=0)
    A_std[A_std < 1e-10] = 1e-10
    B_std[B_std < 1e-10] = 1e-10

    A = A / A_std
    B = B / B_std

    return (A.T @ B / A.shape[0]).astype(np.float32)
