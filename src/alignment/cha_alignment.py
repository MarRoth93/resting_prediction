"""
Connectivity-based Hyperalignment (CHA) for zero-shot subject alignment.

Uses resting-state connectivity fingerprints in parcel space to align
a new subject to the shared space without any task data.
"""

import logging

import numpy as np

from src.alignment.utils import procrustes_align

logger = logging.getLogger(__name__)


def align_via_connectivity_fingerprint(
    P_new: np.ndarray,
    C_new: np.ndarray,
    template_fingerprint: np.ndarray,
) -> np.ndarray:
    """
    Zero-shot alignment using connectivity fingerprints.

    For parcellation mode:
      C_new is (R, V_new) — parcel-to-voxel connectivity
      P_new is (V_new, k) — REST-derived basis
      F_new = C_new @ P_new → (R, k) — fingerprint in common parcel space

      template_fingerprint is (R, k) — calibrated from training subjects
      (already rotated to shared space in hybrid mode)

    Procrustes-align F_new to template → R_new (k, k)

    Args:
        P_new: (V_new, k) basis for new subject
        C_new: (R, V_new) parcellation connectivity
        template_fingerprint: (R, k) shared template fingerprint

    Returns:
        R_new: (k, k) rotation to shared space
    """
    # Compute fingerprint
    F_new = C_new @ P_new  # (R, k)

    logger.info(f"Zero-shot CHA: F_new {F_new.shape}, template {template_fingerprint.shape}")

    # Verify dimensions match
    if F_new.shape != template_fingerprint.shape:
        raise ValueError(
            f"Fingerprint shape mismatch: F_new={F_new.shape}, "
            f"template={template_fingerprint.shape}"
        )

    # Procrustes alignment
    R_new = procrustes_align(F_new, template_fingerprint)

    # Verify orthogonality
    ortho_error = np.linalg.norm(R_new @ R_new.T - np.eye(R_new.shape[0]))
    if ortho_error > 1e-4:
        logger.warning(f"Rotation orthogonality error: {ortho_error:.6f}")

    return R_new
