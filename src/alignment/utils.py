"""
Alignment utility functions: Procrustes, SVD, column sign alignment.
"""

import logging

import numpy as np
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)


def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Orthogonal Procrustes: find R such that source @ R ≈ target.

    Both matrices are centered before computing alignment.

    Args:
        source: (N, k) matrix
        target: (N, k) matrix

    Returns:
        R: (k, k) orthogonal rotation matrix

    Raises:
        ValueError: if inputs contain NaN/Inf or near-constant columns
    """
    if np.any(~np.isfinite(source)) or np.any(~np.isfinite(target)):
        raise ValueError("NaN/Inf in Procrustes input")
    if np.any(source.std(axis=0) < 1e-10) or np.any(target.std(axis=0) < 1e-10):
        raise ValueError("Near-constant column in Procrustes input")

    # Center
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)

    M = source_c.T @ target_c  # (k, k)
    U, _, Vt = np.linalg.svd(M)
    return (U @ Vt).astype(np.float32)


def column_sign_fix(V: np.ndarray) -> np.ndarray:
    """
    Fix sign ambiguity of SVD columns.

    For each column, ensure the element with largest absolute value is positive.
    """
    max_abs_idx = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_idx, range(V.shape[1])])
    signs[signs == 0] = 1
    return V * signs


def compute_svd_basis(
    C: np.ndarray,
    n_components: int = 50,
    min_k: int = 10,
    use_randomized: bool = True,
) -> np.ndarray:
    """
    SVD of matrix C -> top-k right singular vectors as basis P.

    Args:
        C: (M, V) matrix (e.g., parcellation connectivity R×V, or V×V)
        n_components: desired number of components (ceiling)
        min_k: minimum acceptable k (fail-fast)
        use_randomized: use randomized SVD for large matrices

    Returns:
        P: (V, k_actual) right singular vectors, float32

    Raises:
        ValueError: if k_actual < min_k
    """
    M, V = C.shape
    max_rank = min(M, V) - 1
    k_actual = min(n_components, max_rank)

    if k_actual < min_k:
        raise ValueError(
            f"Effective k={k_actual} < min_k={min_k}. "
            f"Matrix shape=({M}, {V}), max_rank={max_rank}. "
            f"Use a higher-resolution atlas or voxel_correlation mode."
        )

    if k_actual < n_components:
        logger.warning(
            f"Clamping n_components from {n_components} to {k_actual} "
            f"(matrix rank limit: min({M},{V})-1={max_rank})"
        )

    if use_randomized and min(M, V) > 1000:
        logger.info(f"Using randomized SVD (matrix {M}x{V}, k={k_actual})")
        U, S, Vt = randomized_svd(
            C, n_components=k_actual, random_state=42, n_oversamples=10
        )
    else:
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        U = U[:, :k_actual]
        S = S[:k_actual]
        Vt = Vt[:k_actual]

    # P = right singular vectors (V × k)
    P = Vt.T.astype(np.float32)
    P = column_sign_fix(P)

    logger.info(f"SVD basis: ({V}, {k_actual}), explained variance ratio top-3: "
                f"{(S[:3]**2 / (S**2).sum())}")

    return P


def compute_global_k(
    connectivity_matrices: dict[int, np.ndarray],
    n_components: int = 50,
    min_k: int = 10,
) -> int:
    """
    Compute global k = min(effective k across all subjects).

    This ensures all subjects have the same number of components
    for Procrustes alignment.

    Args:
        connectivity_matrices: sub_id -> C matrix
        n_components: desired ceiling
        min_k: minimum acceptable

    Returns:
        k_global: int
    """
    k_values = []
    for sub_id, C in connectivity_matrices.items():
        max_rank = min(C.shape) - 1
        k = min(n_components, max_rank)
        k_values.append(k)
        logger.info(f"Subject {sub_id}: max_rank={max_rank}, effective_k={k}")

    k_global = min(k_values)
    if k_global < min_k:
        raise ValueError(
            f"Global k={k_global} < min_k={min_k}. "
            f"Per-subject k values: {dict(zip(connectivity_matrices.keys(), k_values))}"
        )

    logger.info(f"Global k = {k_global} (per-subject: {k_values})")
    return k_global
