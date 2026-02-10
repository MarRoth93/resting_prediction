"""
MNI Baseline: Ridge regression in common MNI space.

Simplest approach — pools training subjects in MNI space.
Requires spatial transforms (nilearn).
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class MNIBaselineModel:
    """
    Baseline: ridge regression in MNI space.

    Pools training subjects in a common MNI mask and trains
    a single encoding model. Ignores resting-state data entirely.
    """

    def __init__(self, alpha: float = 100.0):
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Train ridge from features to MNI-space voxels.

        Args:
            X: (N, F) stimulus features (concatenated across subjects)
            Y: (N, V_mni) MNI-space voxel responses (concatenated)
        """
        N, F = X.shape
        logger.info(f"MNI baseline: fitting X ({N}, {F}) → Y ({N}, {Y.shape[1]})")

        self.x_mean = X.mean(axis=0).astype(np.float32)
        self.x_std = X.std(axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-8] = 1e-8
        Xs = (X - self.x_mean) / self.x_std

        self.b = Y.mean(axis=0).astype(np.float32)
        Yc = Y - self.b

        self.W = np.linalg.solve(
            Xs.T @ Xs + self.alpha * np.eye(F, dtype=np.float32),
            Xs.T @ Yc,
        ).astype(np.float32)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict MNI-space activation from features."""
        Xs = (X_new - self.x_mean) / self.x_std
        return (Xs @ self.W + self.b).astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, W=self.W, b=self.b, x_mean=self.x_mean, x_std=self.x_std, alpha=self.alpha)

    @classmethod
    def load(cls, path: str) -> "MNIBaselineModel":
        data = np.load(path)
        model = cls(alpha=float(data["alpha"]))
        model.W = data["W"]
        model.b = data["b"]
        model.x_mean = data["x_mean"]
        model.x_std = data["x_std"]
        return model
