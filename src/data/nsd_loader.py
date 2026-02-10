"""
Unified data loading classes for NSD subjects and features.
"""

import logging
import os
from functools import cached_property
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class NSDSubjectData:
    """Lazy data loader for one NSD subject."""

    def __init__(self, sub: int, data_root: str = "processed_data"):
        self.sub = sub
        self.data_root = data_root
        self._dir = os.path.join(data_root, f"subj{sub:02d}")
        if not os.path.exists(self._dir):
            raise FileNotFoundError(f"Subject data directory not found: {self._dir}")

    @cached_property
    def train_fmri(self) -> np.ndarray:
        """(N_train, V_sub) averaged task betas, float32."""
        return np.load(os.path.join(self._dir, "train_fmri.npy"), mmap_mode="r")

    @cached_property
    def test_fmri(self) -> np.ndarray:
        """(N_test, V_sub) averaged task betas, float32."""
        return np.load(os.path.join(self._dir, "test_fmri.npy"), mmap_mode="r")

    @cached_property
    def train_stim_idx(self) -> np.ndarray:
        """(N_train,) NSD image indices for training stimuli, sorted."""
        return np.load(os.path.join(self._dir, "train_stim_idx.npy"))

    @cached_property
    def test_stim_idx(self) -> np.ndarray:
        """(N_test,) NSD image indices for test stimuli, sorted."""
        return np.load(os.path.join(self._dir, "test_stim_idx.npy"))

    @cached_property
    def test_fmri_trials(self) -> np.ndarray:
        """(N_test_trials, V_sub) trial-level test betas for noise ceiling."""
        path = os.path.join(self._dir, "test_fmri_trials.npy")
        if os.path.exists(path):
            return np.load(path, mmap_mode="r")
        return None

    @cached_property
    def test_trial_labels(self) -> np.ndarray:
        """(N_test_trials,) stimulus index per trial."""
        path = os.path.join(self._dir, "test_trial_labels.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    @cached_property
    def rest_runs(self) -> list[np.ndarray]:
        """List of (T_run, V_sub) preprocessed resting-state arrays."""
        runs = []
        i = 1
        while True:
            path = os.path.join(self._dir, f"rest_run{i}.npy")
            if not os.path.exists(path):
                break
            runs.append(np.load(path, mmap_mode="r"))
            i += 1
        if not runs:
            logger.warning(f"Subject {self.sub}: no REST runs found in {self._dir}")
        return runs

    @cached_property
    def mask(self) -> np.ndarray:
        """3D nsdgeneral mask (bool)."""
        return np.load(os.path.join(self._dir, "mask.npy"))

    @cached_property
    def num_voxels(self) -> int:
        """Number of masked voxels."""
        return int(self.mask.sum())

    def __repr__(self) -> str:
        return f"NSDSubjectData(sub={self.sub}, dir={self._dir})"


class NSDFeatures:
    """Feature loader for NSD stimuli."""

    def __init__(self, features_dir: str = "processed_data/features"):
        self.features_dir = features_dir

    def get_features(
        self,
        stim_indices: np.ndarray,
        feature_type: str = "clip",
    ) -> np.ndarray:
        """
        Get features for specific stimuli by NSD index.

        Args:
            stim_indices: (N,) array of NSD image indices (0-based)
            feature_type: 'clip', 'dinov2', 'clip_dinov2'

        Returns:
            (N, F) float32 feature array
        """
        if feature_type == "clip_dinov2":
            clip = self._load_features("clip")
            dinov2 = self._load_features("dinov2")
            combined = np.concatenate([clip, dinov2], axis=1)
            return combined[stim_indices].astype(np.float32)

        all_features = self._load_features(feature_type)
        return all_features[stim_indices].astype(np.float32)

    def _load_features(self, feature_type: str) -> np.ndarray:
        """Load full feature array from disk."""
        path = os.path.join(self.features_dir, f"{feature_type}_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Features not found: {path}. Run feature extraction first."
            )
        return np.load(path, mmap_mode="r")
