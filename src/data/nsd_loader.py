"""
Unified data loading classes for NSD subjects and features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBundle:
    """Concatenated feature streams plus stable column slices."""

    streams: dict[str, np.ndarray]
    slices: dict[str, tuple[int, int]]

    @property
    def array(self) -> np.ndarray:
        return np.concatenate(list(self.streams.values()), axis=1).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return int(sum(stream.shape[1] for stream in self.streams.values()))

    @classmethod
    def from_streams(cls, streams: dict[str, np.ndarray]) -> "FeatureBundle":
        if not streams:
            raise ValueError("FeatureBundle requires at least one stream.")

        n_rows = None
        start = 0
        slices: dict[str, tuple[int, int]] = {}
        clean_streams: dict[str, np.ndarray] = {}
        for name, values in streams.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Feature stream {name!r} must be 2D, got shape {arr.shape}.")
            if n_rows is None:
                n_rows = int(arr.shape[0])
            elif int(arr.shape[0]) != n_rows:
                raise ValueError(
                    f"Feature stream {name!r} has {arr.shape[0]} rows, expected {n_rows}."
                )
            end = start + int(arr.shape[1])
            slices[str(name)] = (start, end)
            clean_streams[str(name)] = arr
            start = end
        return cls(streams=clean_streams, slices=slices)


def resolve_feature_streams(feature_type: str, streams: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Resolve the single CLIP stream used by the frozen model."""
    if streams:
        resolved = [str(s) for s in streams]
        if resolved != ["clip"]:
            raise ValueError(f"Frozen model requires features.streams=['clip'], got {resolved}.")
        return resolved
    if feature_type != "clip":
        raise ValueError(f"Frozen model requires feature_type='clip', got {feature_type!r}.")
    return ["clip"]


class NSDSubjectData:
    """Lazy data loader for one NSD subject."""

    def __init__(self, sub: int, data_root: str = "data/processed"):
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

    def __init__(self, features_dir: str = "data/processed/features"):
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
            feature_type: must be 'clip'

        Returns:
            (N, F) float32 feature array
        """
        if feature_type != "clip":
            raise ValueError(f"Frozen model requires feature_type='clip', got {feature_type!r}.")

        all_features = self._load_features(feature_type)
        return all_features[stim_indices].astype(np.float32)

    def get_feature_streams(
        self,
        stim_indices: np.ndarray,
        feature_types: list[str] | tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """
        Get named feature streams for stimuli by NSD index.

        Args:
            stim_indices: (N,) array of NSD image indices
            feature_types: must be ["clip"]

        Returns:
            name -> (N, F_stream) float32 array
        """
        streams: dict[str, np.ndarray] = {}
        for feature_type in feature_types:
            name = str(feature_type)
            if name != "clip":
                raise ValueError(f"Frozen model requires feature stream 'clip', got {name!r}.")
            all_features = self._load_features(name)
            streams[name] = all_features[stim_indices].astype(np.float32)
        return streams

    def get_feature_bundle(
        self,
        stim_indices: np.ndarray,
        feature_types: list[str] | tuple[str, ...],
    ) -> FeatureBundle:
        """Return concatenated stream features with column slice metadata."""
        return FeatureBundle.from_streams(self.get_feature_streams(stim_indices, feature_types))

    def _load_features(self, feature_type: str) -> np.ndarray:
        """Load full feature array from disk."""
        path = os.path.join(self.features_dir, f"{feature_type}_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Features not found: {path}. Run feature extraction first."
            )
        return np.load(path, mmap_mode="r")
