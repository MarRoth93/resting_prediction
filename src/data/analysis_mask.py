"""Frozen nsdgeneral prediction-mask handling."""

from __future__ import annotations

import numpy as np


def build_analysis_mask(sub: int, nsdgeneral_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the nsdgeneral mask and compact preprocessing provenance."""
    mask = np.asarray(nsdgeneral_mask, dtype=bool)
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError(f"Subject {sub}: nsdgeneral mask is empty.")
    return mask, {
        "subject": int(sub),
        "mode": "nsdgeneral",
        "nsdgeneral_voxels": n_voxels,
        "analysis_voxels": n_voxels,
    }
