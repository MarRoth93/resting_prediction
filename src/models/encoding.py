"""
SharedSpaceEncoder: Ridge regression from stimulus features to shared-space responses.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class SharedSpaceEncoder:
    """
    Maps stimulus features to shared-space responses via ridge regression.

    Procedure:
    1. Standardize features: Xs = (X - mean) / std
    2. Center targets: Zc = Z - Z_mean
    3. Ridge: W = (Xs'Xs + αI)⁻¹ Xs'Zc
    4. Intercept: b = Z_mean
    """

    def __init__(self, alpha: float = 100.0):
        self.alpha = alpha
        self.W: np.ndarray | None = None  # (F, k)
        self.b: np.ndarray | None = None  # (k,)
        self.x_mean: np.ndarray | None = None  # (F,)
        self.x_std: np.ndarray | None = None  # (F,)

    def fit(self, X: np.ndarray, Z: np.ndarray):
        """
        Fit ridge: X (N, F) → Z (N, k) with explicit intercept.

        Args:
            X: (N, F) stimulus features
            Z: (N, k) aligned shared-space responses
        """
        N, F = X.shape
        _, k = Z.shape
        logger.info(f"Fitting encoder: X ({N}, {F}) → Z ({N}, {k}), alpha={self.alpha}")

        # Standardize features
        self.x_mean = X.mean(axis=0).astype(np.float32)
        self.x_std = X.std(axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-8] = 1e-8
        Xs = (X - self.x_mean) / self.x_std

        # Center targets
        self.b = Z.mean(axis=0).astype(np.float32)
        Zc = Z - self.b

        # Ridge: W = (Xs'Xs + αI)⁻¹ Xs'Zc
        XtX = Xs.T @ Xs  # (F, F)
        XtZ = Xs.T @ Zc  # (F, k)
        self.W = np.linalg.solve(
            XtX + self.alpha * np.eye(F, dtype=np.float32),
            XtZ,
        ).astype(np.float32)

        # Training R²
        Z_pred = Xs @ self.W + self.b
        ss_res = np.sum((Z - Z_pred) ** 2)
        ss_tot = np.sum((Z - Z.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        logger.info(f"Encoder fitted: W shape={self.W.shape}, train R²={r2:.4f}")

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict shared-space responses: (N, F) → (N, k)."""
        Xs = (X_new - self.x_mean) / self.x_std
        return (Xs @ self.W + self.b).astype(np.float32)

    def predict_voxels(
        self,
        X_new: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """
        Predict voxel-wise responses for a specific subject.

        Args:
            X_new: (N, F) stimulus features
            P: (V_sub, k) subject's REST basis
            R: (k, k) subject's rotation to shared space

        Returns:
            Y_hat: (N, V_sub) predicted voxel responses
        """
        Z_hat = self.predict(X_new)  # (N, k)
        return (Z_hat @ R.T @ P.T).astype(np.float32)  # (N, V_sub)

    def save(self, path: str):
        """Save encoder to .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            W=self.W,
            b=self.b,
            x_mean=self.x_mean,
            x_std=self.x_std,
            alpha=self.alpha,
        )
        logger.info(f"Saved encoder to {path}")

    @classmethod
    def load(cls, path: str) -> "SharedSpaceEncoder":
        """Load encoder from .npz file."""
        data = np.load(path)
        encoder = cls(alpha=float(data["alpha"]))
        encoder.W = data["W"]
        encoder.b = data["b"]
        encoder.x_mean = data["x_mean"]
        encoder.x_std = data["x_std"]
        return encoder


def fine_tune_encoder(
    encoder: SharedSpaceEncoder,
    X_new: np.ndarray,
    Z_new: np.ndarray,
    alpha: float | None = None,
    blend: float = 0.5,
) -> SharedSpaceEncoder:
    """
    Fine-tune encoder with new subject's data.

    Creates a blended encoder: W_new = blend * W_finetune + (1-blend) * W_orig

    Args:
        encoder: pre-trained encoder
        X_new: (N, F) new subject's stimulus features
        Z_new: (N, k) new subject's aligned responses
        alpha: ridge alpha for fine-tuning (default: same as original)
        blend: blend weight for new vs original weights

    Returns:
        New fine-tuned encoder
    """
    ft_encoder = SharedSpaceEncoder(alpha=alpha if alpha is not None else encoder.alpha)
    ft_encoder.x_mean = encoder.x_mean.copy()
    ft_encoder.x_std = encoder.x_std.copy()

    # Fit on new data
    Xs = (X_new - encoder.x_mean) / encoder.x_std
    b_new = Z_new.mean(axis=0).astype(np.float32)
    Zc = Z_new - b_new
    N, F = Xs.shape
    W_new = np.linalg.solve(
        Xs.T @ Xs + ft_encoder.alpha * np.eye(F, dtype=np.float32),
        Xs.T @ Zc,
    ).astype(np.float32)

    # Blend
    ft_encoder.W = (blend * W_new + (1 - blend) * encoder.W).astype(np.float32)
    ft_encoder.b = (blend * b_new + (1 - blend) * encoder.b).astype(np.float32)

    logger.info(f"Fine-tuned encoder: blend={blend}, alpha={ft_encoder.alpha}")
    return ft_encoder
