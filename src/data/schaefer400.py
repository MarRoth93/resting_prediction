"""Canonical Schaefer-400 parcel data shared by NSD training and FOR inference."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from src.schaefer400_config import ATLAS_NAME, N_PARCELS


PARCEL_IDS = np.arange(1, N_PARCELS + 1, dtype=np.int16)


def canonicalize_schaefer_volume_labels(values: np.ndarray) -> np.ndarray:
    """Convert CBIG volume labels (1001/2001 series) to ordered IDs 1..400."""
    labels = np.asarray(values)
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, atol=1e-5, rtol=0):
        raise ValueError("Schaefer atlas contains non-integer labels.")
    labels = rounded.astype(np.int32)
    out = np.zeros(labels.shape, dtype=np.int16)
    direct = (labels >= 1) & (labels <= N_PARCELS)
    left = (labels >= 1001) & (labels <= 1200)
    right = (labels >= 2001) & (labels <= 2200)
    out[direct] = labels[direct]
    out[left] = labels[left] - 1000
    out[right] = labels[right] - 1800
    known_background = np.isin(labels, [0, 1000, 2000])
    unknown = ~(direct | left | right | known_background)
    if np.any(unknown):
        sample = np.unique(labels[unknown])[:10].tolist()
        raise ValueError(f"Atlas contains unsupported labels: {sample}")
    return out


def validate_schaefer400_atlas(
    atlas: np.ndarray,
    *,
    min_voxels_per_parcel: int = 1,
) -> np.ndarray:
    """Validate complete canonical coverage and return 400 parcel voxel counts."""
    labels = canonicalize_schaefer_volume_labels(atlas)
    counts = np.bincount(labels.ravel(), minlength=N_PARCELS + 1)[1:]
    if counts.shape != (N_PARCELS,):
        raise RuntimeError("Internal Schaefer parcel count has the wrong shape.")
    missing = PARCEL_IDS[counts < int(min_voxels_per_parcel)]
    if missing.size:
        raise ValueError(
            f"Atlas has {missing.size} parcels below {min_voxels_per_parcel} voxels: "
            f"{missing[:20].tolist()}"
        )
    return counts.astype(np.int32)


class ParcelReducer:
    """Memory-bounded mean aggregation from a functional volume to 400 parcels."""

    def __init__(self, labels: np.ndarray, *, min_voxels_per_parcel: int = 1):
        canonical = canonicalize_schaefer_volume_labels(labels)
        self.spatial_shape = tuple(int(value) for value in canonical.shape)
        self.counts = validate_schaefer400_atlas(
            canonical,
            min_voxels_per_parcel=min_voxels_per_parcel,
        )
        flat = canonical.ravel(order="C")
        columns = np.flatnonzero(flat > 0)
        rows = flat[columns].astype(np.int64) - 1
        weights = (1.0 / self.counts[rows]).astype(np.float32)
        self.membership = sparse.csr_matrix(
            (weights, (rows, columns)),
            shape=(N_PARCELS, flat.size),
            dtype=np.float32,
        )

    def reduce_4d(self, values: np.ndarray) -> np.ndarray:
        """Return time/trial by parcel means for an in-memory 4D array."""
        array = np.asarray(values)
        if array.ndim != 4 or tuple(array.shape[:3]) != self.spatial_shape:
            raise ValueError(
                f"Expected volume shape {self.spatial_shape} + time, got {array.shape}."
            )
        flat = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[3])
        result = (self.membership @ flat).T
        result = np.asarray(result, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("Parcellated data contain NaN/Inf.")
        return result

    def reduce_proxy(self, proxy: Any, *, chunk_size: int = 25) -> np.ndarray:
        """Reduce a nibabel 4D proxy without materializing the full NIfTI."""
        shape = tuple(int(value) for value in proxy.shape)
        if len(shape) != 4 or shape[:3] != self.spatial_shape:
            raise ValueError(
                f"Expected proxy shape {self.spatial_shape} + time, got {shape}."
            )
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive.")
        result = np.empty((shape[3], N_PARCELS), dtype=np.float32)
        for start in range(0, shape[3], int(chunk_size)):
            stop = min(start + int(chunk_size), shape[3])
            block = np.asarray(proxy[..., start:stop], dtype=np.float32)
            result[start:stop] = np.asarray(
                (self.membership @ block.reshape(-1, stop - start)).T,
                dtype=np.float32,
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("Parcellated data contain NaN/Inf.")
        return result


class SchaeferNSDSubjectData:
    """Lazy loader for one NSD subject represented by 400 ordered parcels."""

    def __init__(self, sub: int, data_root: str | Path):
        self.sub = int(sub)
        self.data_root = str(Path(data_root))
        self._dir = Path(data_root) / f"subj{self.sub:02d}"
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Schaefer subject directory not found: {self._dir}")

    def _array(self, name: str, *, mmap: bool = True) -> np.ndarray:
        path = self._dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing Schaefer subject array: {path}")
        return np.load(path, mmap_mode="r" if mmap else None)

    @cached_property
    def train_fmri(self) -> np.ndarray:
        return self._array("train_fmri.npy")

    @cached_property
    def test_fmri(self) -> np.ndarray:
        return self._array("test_fmri.npy")

    @cached_property
    def train_stim_idx(self) -> np.ndarray:
        return self._array("train_stim_idx.npy", mmap=False)

    @cached_property
    def test_stim_idx(self) -> np.ndarray:
        return self._array("test_stim_idx.npy", mmap=False)

    @cached_property
    def rest_runs(self) -> list[np.ndarray]:
        paths = sorted(
            self._dir.glob("rest_run*.npy"),
            key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
        )
        return [np.load(path, mmap_mode="r") for path in paths]

    @cached_property
    def parcel_voxel_counts(self) -> np.ndarray:
        return self._array("parcel_voxel_counts.npy", mmap=False)

    @property
    def num_voxels(self) -> int:
        return N_PARCELS

    @property
    def parcel_groups(self) -> np.ndarray:
        return np.arange(N_PARCELS, dtype=np.int64)

    def __repr__(self) -> str:
        return f"SchaeferNSDSubjectData(sub={self.sub}, dir={self._dir})"


def validate_schaefer_nsd_subject(subject: SchaeferNSDSubjectData) -> None:
    """Check the complete parcel/task/REST contract before fitting experts."""
    train = np.asarray(subject.train_fmri)
    test = np.asarray(subject.test_fmri)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"Subject {subject.sub}: task arrays must be 2D.")
    if train.shape[1] != N_PARCELS or test.shape[1] != N_PARCELS:
        raise ValueError(f"Subject {subject.sub}: task arrays must have 400 parcels.")
    if train.shape[0] != subject.train_stim_idx.size:
        raise ValueError(f"Subject {subject.sub}: training stimulus rows differ.")
    if test.shape[0] != subject.test_stim_idx.size:
        raise ValueError(f"Subject {subject.sub}: test stimulus rows differ.")
    if not subject.rest_runs:
        raise ValueError(f"Subject {subject.sub}: no Schaefer REST runs.")
    if any(run.ndim != 2 or run.shape[1] != N_PARCELS for run in subject.rest_runs):
        raise ValueError(f"Subject {subject.sub}: REST arrays must be T x 400.")
    if subject.parcel_voxel_counts.shape != (N_PARCELS,):
        raise ValueError(f"Subject {subject.sub}: parcel voxel counts are incomplete.")
    arrays = [train, test, *subject.rest_runs]
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"Subject {subject.sub}: prepared arrays contain NaN/Inf.")
    for ids, label in (
        (subject.train_stim_idx, "train"),
        (subject.test_stim_idx, "test"),
    ):
        if ids.ndim != 1 or np.unique(ids).size != ids.size or np.any(ids < 0):
            raise ValueError(f"Subject {subject.sub}: invalid {label} stimulus IDs.")


@dataclass(frozen=True)
class FORSchaeferSubjectData:
    """One FOR subject with missing parcels separated from the seed contract."""

    subject_label: str
    subject_id: int
    tr_seconds: float
    standardized_timeseries: np.ndarray
    available_parcels: np.ndarray
    voxel_counts: np.ndarray
    provenance: dict

    def __post_init__(self) -> None:
        series = np.asarray(self.standardized_timeseries, dtype=np.float32)
        available = np.asarray(self.available_parcels, dtype=bool)
        counts = np.asarray(self.voxel_counts, dtype=np.int32)
        if series.ndim != 2 or series.shape[1] != N_PARCELS or series.shape[0] < 2:
            raise ValueError(f"FOR time series must be T x 400, got {series.shape}.")
        if available.shape != (N_PARCELS,) or counts.shape != (N_PARCELS,):
            raise ValueError("FOR parcel availability/count arrays must have length 400.")
        if int(available.sum()) <= 0:
            raise ValueError("FOR subject has no usable Schaefer parcels.")
        if not np.all(np.isfinite(series)):
            raise ValueError("Standardized FOR time series contain NaN/Inf.")
        object.__setattr__(self, "standardized_timeseries", series)
        object.__setattr__(self, "available_parcels", available)
        object.__setattr__(self, "voxel_counts", counts)
        object.__setattr__(self, "subject_id", int(self.subject_id))
        object.__setattr__(self, "tr_seconds", float(self.tr_seconds))

    @property
    def sub(self) -> int:
        return self.subject_id

    @property
    def rest_runs(self) -> list[np.ndarray]:
        """Target columns omit missing parcels so their transform is not invented."""
        return [self.standardized_timeseries[:, self.available_parcels]]

    @property
    def seed_runs(self) -> list[np.ndarray]:
        """Seed rows keep canonical width; unavailable signals are explicit zeros."""
        return [self.standardized_timeseries]

    @property
    def parcel_groups(self) -> np.ndarray:
        return np.flatnonzero(self.available_parcels).astype(np.int64)

    def expand_available(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        expected = int(self.available_parcels.sum())
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"Expected prediction width {expected}, got {values.shape}.")
        expanded = np.full((values.shape[0], N_PARCELS), np.nan, dtype=np.float32)
        expanded[:, self.available_parcels] = values
        return expanded


def _hdf_scalar(handle: h5py.File, name: str) -> float:
    if name not in handle:
        raise ValueError(f"FOR MATLAB file is missing {name!r}.")
    values = np.asarray(handle[name]).squeeze()
    if values.size != 1:
        raise ValueError(f"FOR field {name!r} must be scalar, got {values.shape}.")
    return float(values)


def _decode_matlab_chars(values: np.ndarray) -> str:
    flat = np.asarray(values).astype(np.uint32).ravel(order="F")
    return "".join(chr(int(value)) for value in flat if int(value) != 0)


def _read_for_summary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(N_PARCELS, dtype=np.int32)
    usable = np.zeros(N_PARCELS, dtype=bool)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != N_PARCELS:
        raise ValueError(f"FOR summary must contain 400 parcels, got {len(rows)}: {path}")
    for row in rows:
        parcel_id = int(row["parcel_id"])
        if not 1 <= parcel_id <= N_PARCELS:
            raise ValueError(f"FOR summary has invalid parcel ID {parcel_id}.")
        counts[parcel_id - 1] = int(row["voxel_count"])
        usable[parcel_id - 1] = bool(int(row["usable"]))
    if np.unique([int(row["parcel_id"]) for row in rows]).size != N_PARCELS:
        raise ValueError("FOR summary has duplicate parcel IDs.")
    return counts, usable


def load_for_schaefer400_subject(
    subject_dir: str | Path,
    *,
    timeseries_file: str = "schaefer400_parcel_timeseries.mat",
    summary_file: str = "schaefer400_parcel_summary.tsv",
) -> FORSchaeferSubjectData:
    """Load and validate the MATLAB-v7.3 FOR export without task ground truth."""
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"FOR subject directory not found: {subject_dir}")
    mat_path = subject_dir / timeseries_file
    summary_path = subject_dir / summary_file
    if not mat_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"FOR subject {subject_dir.name} lacks {timeseries_file} or {summary_file}."
        )

    with h5py.File(mat_path, "r") as handle:
        if "parcel_timeseries" not in handle or "parcel_ids" not in handle:
            raise ValueError(f"FOR MATLAB file has no parcel arrays: {mat_path}")
        series = np.asarray(handle["parcel_timeseries"], dtype=np.float32)
        parcel_ids = np.asarray(handle["parcel_ids"]).astype(np.int64).ravel()
        tr_seconds = _hdf_scalar(handle, "TR_seconds")
        embedded_paths = {
            name: _decode_matlab_chars(np.asarray(handle[name]))
            for name in ("atlas_file", "bold_file", "mask_file")
            if name in handle
        }
    if series.shape[0] == N_PARCELS and series.shape[1] != N_PARCELS:
        series = series.T
    if series.ndim != 2 or series.shape[1] != N_PARCELS:
        raise ValueError(f"FOR parcel_timeseries must resolve to T x 400, got {series.shape}.")
    if not np.array_equal(parcel_ids, np.arange(1, N_PARCELS + 1)):
        raise ValueError("FOR parcel_ids are not the canonical ordered values 1..400.")

    counts, summary_usable = _read_for_summary(summary_path)
    finite_all = np.all(np.isfinite(series), axis=0)
    finite_any = np.any(np.isfinite(series), axis=0)
    partial = finite_any & ~finite_all
    if np.any(partial):
        raise ValueError(
            f"FOR subject {subject_dir.name} has partially missing parcel rows: "
            f"{PARCEL_IDS[partial].tolist()}"
        )
    available = finite_all & summary_usable & (counts > 0)
    if not np.array_equal(available, summary_usable):
        mismatch = PARCEL_IDS[available != summary_usable].tolist()
        raise ValueError(
            f"FOR summary and time-series availability disagree for parcels {mismatch}."
        )

    standardized = np.zeros_like(series, dtype=np.float32)
    usable_series = series[:, available]
    means = usable_series.mean(axis=0)
    stds = usable_series.std(axis=0)
    if np.any(stds < 1e-8):
        constant = PARCEL_IDS[available][stds < 1e-8].tolist()
        raise ValueError(f"FOR usable parcels are constant: {constant}")
    standardized[:, available] = (usable_series - means) / stds

    match = re.search(r"(\d+)$", subject_dir.name)
    if match is None:
        raise ValueError(f"Cannot derive numeric subject ID from {subject_dir.name!r}.")
    provenance = {
        "dataset": "FOR",
        "representation": "schaefer400_7network_order",
        "atlas_name": ATLAS_NAME,
        "subject_dir": str(subject_dir.resolve()),
        "timeseries_file": str(mat_path.resolve()),
        "summary_file": str(summary_path.resolve()),
        "timepoints": int(series.shape[0]),
        "tr_seconds": float(tr_seconds),
        "available_parcels": int(available.sum()),
        "missing_parcel_ids": PARCEL_IDS[~available].astype(int).tolist(),
        "embedded_source_paths": embedded_paths,
        "input_status": "precleaned_filtered_export",
        "common_final_step": "per_parcel_zscore",
    }
    return FORSchaeferSubjectData(
        subject_label=subject_dir.name,
        subject_id=int(match.group(1)),
        tr_seconds=tr_seconds,
        standardized_timeseries=standardized,
        available_parcels=available,
        voxel_counts=counts,
        provenance=provenance,
    )


def audit_for_schaefer400_dataset(root: str | Path) -> dict:
    """Validate every FOR subject and summarize the real transfer input contract."""
    root = Path(root)
    subject_dirs = sorted(path for path in root.glob("sub-*") if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No FOR sub-* directories found under {root}")
    subjects = [load_for_schaefer400_subject(path) for path in subject_dirs]
    summary = {
        "dataset_root": str(root.resolve()),
        "subjects": len(subjects),
        "timepoints": sorted({int(subject.standardized_timeseries.shape[0]) for subject in subjects}),
        "tr_seconds": sorted({float(subject.tr_seconds) for subject in subjects}),
        "available_parcels_min": min(int(subject.available_parcels.sum()) for subject in subjects),
        "available_parcels_max": max(int(subject.available_parcels.sum()) for subject in subjects),
        "subjects_with_missing_parcels": sum(
            int(not np.all(subject.available_parcels)) for subject in subjects
        ),
        "missing_parcels_total": sum(
            int((~subject.available_parcels).sum()) for subject in subjects
        ),
        "subject_rows": [
            {
                "subject": subject.subject_label,
                "timepoints": int(subject.standardized_timeseries.shape[0]),
                "tr_seconds": float(subject.tr_seconds),
                "available_parcels": int(subject.available_parcels.sum()),
                "missing_parcel_ids": PARCEL_IDS[~subject.available_parcels].astype(int).tolist(),
            }
            for subject in subjects
        ],
    }
    return summary


def save_for_audit(summary: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
