"""Deterministic held-out evaluation split."""

from __future__ import annotations

import numpy as np


def fixed_eval_indices(n_shared: int, eval_size: int, seed: int) -> np.ndarray:
    """Select the config-defined evaluation rows without persisted mutable state."""
    if n_shared <= 1:
        raise ValueError(f"n_shared must be >1, got {n_shared}.")
    if not 1 <= eval_size < n_shared:
        raise ValueError(f"eval_size={eval_size} must be in [1, {n_shared - 1}].")
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_shared, size=eval_size, replace=False).astype(np.int64))
