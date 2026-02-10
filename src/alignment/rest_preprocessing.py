"""
Compute resting-state connectivity matrices from preprocessed REST runs.

Two modes:
- parcellation: C is (R, V) — parcel-to-voxel correlation (required for zero-shot CHA)
- voxel_correlation: C is (V, V) — full voxel correlation (few-shot only)
"""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def compute_parcel_timeseries(
    rest_data: np.ndarray,
    atlas_masked: np.ndarray,
    n_parcels: int,
) -> np.ndarray:
    """
    Average voxel timeseries within each parcel.

    Args:
        rest_data: (T, V) preprocessed REST timeseries
        atlas_masked: (V,) integer parcel labels (1-based, 0=unassigned)
        n_parcels: number of parcels

    Returns:
        (T, R) parcel timeseries
    """
    T, V = rest_data.shape
    parcel_ts = np.zeros((T, n_parcels), dtype=np.float32)

    for p in range(1, n_parcels + 1):
        voxel_mask = atlas_masked == p
        if voxel_mask.sum() == 0:
            continue
        parcel_ts[:, p - 1] = rest_data[:, voxel_mask].mean(axis=1)

    return parcel_ts


def compute_rest_connectivity(
    rest_runs: list[np.ndarray],
    mode: str = "parcellation",
    atlas_masked: np.ndarray | None = None,
    n_parcels: int | None = None,
    ensemble: str = "average",
) -> np.ndarray:
    """
    Compute resting-state connectivity from multiple runs.

    Args:
        rest_runs: list of (T_run, V) preprocessed arrays
        mode: 'parcellation' or 'voxel_correlation'
        atlas_masked: (V,) integer labels, required for parcellation
        n_parcels: number of parcels, required for parcellation
        ensemble: 'average' (average per-run connectivity) or 'concat' (concat runs)

    Returns:
        C: connectivity matrix
           - parcellation: (R, V) parcel-to-voxel correlation
           - voxel_correlation: (V, V) voxel-voxel correlation

    Raises:
        ValueError: if mode='parcellation' and atlas is None
        ValueError: if mode='voxel_correlation' used without warning about zero-shot
    """
    if not rest_runs:
        raise ValueError("rest_runs is empty — need at least 1 REST run to compute connectivity")

    if mode == "parcellation":
        if atlas_masked is None or n_parcels is None:
            raise ValueError("parcellation mode requires atlas_masked and n_parcels")
        return _compute_parcellation_connectivity(
            rest_runs, atlas_masked, n_parcels, ensemble
        )
    elif mode == "voxel_correlation":
        logger.warning(
            "voxel_correlation mode: CANNOT be used for zero-shot CHA alignment "
            "(V differs across subjects). Use parcellation mode for zero-shot."
        )
        return _compute_voxel_connectivity(rest_runs, ensemble)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _compute_parcellation_connectivity(
    rest_runs: list[np.ndarray],
    atlas_masked: np.ndarray,
    n_parcels: int,
    ensemble: str,
) -> np.ndarray:
    """
    Compute parcel-to-voxel connectivity: C is (R, V).

    For each run: compute correlation between parcel timeseries (T, R)
    and voxel timeseries (T, V) → (R, V) correlation matrix.
    """
    V = rest_runs[0].shape[1]

    if ensemble == "concat":
        # Concatenate all runs
        rest_concat = np.concatenate(rest_runs, axis=0)  # (T_total, V)
        parcel_ts = compute_parcel_timeseries(rest_concat, atlas_masked, n_parcels)  # (T_total, R)
        C = _correlate(parcel_ts, rest_concat)  # (R, V)
    elif ensemble == "average":
        # Average per-run connectivity
        Cs = []
        for run in rest_runs:
            parcel_ts = compute_parcel_timeseries(run, atlas_masked, n_parcels)  # (T, R)
            C_run = _correlate(parcel_ts, run)  # (R, V)
            Cs.append(C_run)
        C = np.mean(Cs, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble}")

    logger.info(f"Parcellation connectivity: {C.shape}")
    return C.astype(np.float32)


def _compute_voxel_connectivity(
    rest_runs: list[np.ndarray],
    ensemble: str,
) -> np.ndarray:
    """
    Compute voxel-voxel connectivity: C is (V, V).

    Uses Ledoit-Wolf shrinkage for numerical stability.
    """
    if ensemble == "concat":
        rest_concat = np.concatenate(rest_runs, axis=0)
        lw = LedoitWolf()
        cov = lw.fit(rest_concat).covariance_
        # Convert covariance to correlation
        d = np.sqrt(np.diag(cov))
        d[d < 1e-10] = 1e-10
        C = cov / np.outer(d, d)
    elif ensemble == "average":
        Cs = []
        for run in rest_runs:
            lw = LedoitWolf()
            cov = lw.fit(run).covariance_
            d = np.sqrt(np.diag(cov))
            d[d < 1e-10] = 1e-10
            Cs.append(cov / np.outer(d, d))
        C = np.mean(Cs, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble}")

    logger.info(f"Voxel connectivity: {C.shape}, memory: {C.nbytes / 1e6:.0f} MB")
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
