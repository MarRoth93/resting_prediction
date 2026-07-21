"""Create trial-level reliability arrays without rewriting processed task data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.prepare_task_data import discover_sessions, loadmat
from src.data.shared_paths import default_raw_data_root


logger = logging.getLogger(__name__)


def _validated_trial_design(
    masterordering: np.ndarray,
    subjectim: np.ndarray,
    *,
    subject: int,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize MATLAB row/column vectors and validate trial lookup bounds."""
    order = np.asarray(masterordering).ravel()
    subject_images = np.asarray(subjectim)
    if subject_images.ndim != 2:
        raise ValueError("NSD subjectim must be a 2D subject-by-presentation array.")
    if not 1 <= int(subject) <= int(subject_images.shape[0]):
        raise ValueError(f"Subject {subject} is outside the NSD experiment design.")
    if order.size < int(n_trials):
        raise ValueError(
            f"NSD masterordering has {order.size} trials, fewer than the "
            f"{int(n_trials)} discovered beta trials."
        )
    presentations = order[: int(n_trials)].astype(np.int64, copy=False)
    if not np.array_equal(order[: int(n_trials)], presentations):
        raise ValueError("NSD masterordering contains non-integer presentation IDs.")
    if np.any(presentations < 1) or np.any(presentations > subject_images.shape[1]):
        raise ValueError("NSD masterordering references an invalid subjectim column.")
    stimuli_raw = subject_images[int(subject) - 1, presentations - 1]
    stimuli = np.asarray(stimuli_raw, dtype=np.int64)
    if not np.array_equal(stimuli_raw, stimuli) or np.any(stimuli < 1):
        raise ValueError("NSD subjectim contains invalid stimulus IDs.")
    return presentations, stimuli - 1


def _load_selected_masked_volumes(
    image: nib.spatialimages.SpatialImage,
    mask: np.ndarray,
    local_indices: np.ndarray,
) -> np.ndarray:
    """Read selected NIfTI volumes using only proxy-supported basic slices."""
    mask = np.asarray(mask, dtype=bool)
    indices = np.asarray(local_indices, dtype=np.int64).ravel()
    if indices.size == 0:
        return np.empty((0, int(mask.sum())), dtype=np.float32)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Selected beta-volume indices must be strictly increasing.")
    if indices[0] < 0 or indices[-1] >= int(image.shape[-1]):
        raise ValueError("Selected beta-volume index is outside the NIfTI image.")
    if tuple(image.shape[:-1]) != tuple(mask.shape):
        raise ValueError("Beta image and processed mask shapes differ.")

    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    rows: list[np.ndarray] = []
    for group in groups:
        start = int(group[0])
        stop = int(group[-1]) + 1
        block = np.asarray(image.dataobj[..., start:stop], dtype=np.float32)
        if block.ndim == mask.ndim:
            block = block[..., None]
        rows.append(block[mask].T.astype(np.float32, copy=False))
    return np.concatenate(rows, axis=0)


def validate_trial_average_matches_existing(
    trial_fmri: np.ndarray,
    trial_labels: np.ndarray,
    existing_test_fmri: np.ndarray,
    *,
    atol: float = 1e-5,
) -> float:
    """Fail before writing if extracted trials do not reproduce frozen averages."""
    trial_fmri = np.asarray(trial_fmri, dtype=np.float32)
    trial_labels = np.asarray(trial_labels, dtype=np.int64)
    existing = np.asarray(existing_test_fmri, dtype=np.float32)
    if trial_fmri.ndim != 2 or existing.ndim != 2:
        raise ValueError("Trial-level and averaged fMRI must both be 2D.")
    if trial_fmri.shape[1] != existing.shape[1]:
        raise ValueError("Trial-level and averaged fMRI voxel counts differ.")
    if trial_labels.shape != (trial_fmri.shape[0],):
        raise ValueError("Trial labels do not match trial-level rows.")
    expected_labels = np.arange(existing.shape[0], dtype=np.int64)
    if not np.array_equal(np.unique(trial_labels), expected_labels):
        raise ValueError("Trial labels do not cover each frozen test row exactly.")
    reconstructed = np.stack(
        [trial_fmri[trial_labels == label].mean(axis=0) for label in expected_labels],
        axis=0,
    )
    max_error = float(np.max(np.abs(reconstructed - existing)))
    if not np.allclose(reconstructed, existing, atol=float(atol), rtol=0.0):
        raise ValueError(
            "Extracted trial averages do not reproduce existing test_fmri.npy; "
            f"maximum absolute error is {max_error:.3e}. No files were written."
        )
    return max_error


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare_reliability_data(
    subject: int,
    *,
    raw_data_root: str = default_raw_data_root(),
    processed_root: str = "data/processed",
) -> dict:
    """Extract only repeated test trials and preserve every existing core array."""
    subject = int(subject)
    subject_tag = f"subj{subject:02d}"
    subject_dir = Path(processed_root) / subject_tag
    required = {
        name: subject_dir / name
        for name in ("mask.npy", "test_stim_idx.npy", "test_fmri.npy")
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    mask = np.asarray(np.load(required["mask.npy"]), dtype=bool)
    existing_test_ids = np.asarray(np.load(required["test_stim_idx.npy"]), dtype=np.int64)
    existing_test_fmri = np.load(required["test_fmri.npy"], mmap_mode="r")
    if existing_test_fmri.shape != (existing_test_ids.size, int(mask.sum())):
        raise ValueError("Existing test fMRI, stimulus IDs, and mask are inconsistent.")

    raw_root = Path(raw_data_root)
    roi_path = raw_root / "nsddata" / "ppdata" / subject_tag / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
    raw_mask = np.asarray(nib.load(str(roi_path)).dataobj, dtype=np.float32) > 0
    if not np.array_equal(raw_mask, mask):
        raise ValueError("Existing processed mask differs from the raw nsdgeneral mask.")

    design_path = raw_root / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    design = loadmat(str(design_path))
    masterordering = np.asarray(design["masterordering"])
    subjectim = np.asarray(design["subjectim"])
    betas_dir = (
        raw_root
        / "nsddata_betas"
        / "ppdata"
        / subject_tag
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
    )
    sessions = discover_sessions(str(betas_dir))
    trials_per_session = 750
    n_trials = len(sessions) * trials_per_session
    presentation_ids, stimulus_ids = _validated_trial_design(
        masterordering,
        subjectim,
        subject=subject,
        n_trials=n_trials,
    )

    test_trials: dict[int, int] = {}
    raw_test_ids: set[int] = set()
    row_by_stimulus = {
        int(stimulus): int(row)
        for row, stimulus in enumerate(existing_test_ids)
    }
    for trial, (presentation, stimulus) in enumerate(
        zip(presentation_ids, stimulus_ids, strict=True)
    ):
        if int(presentation) > 1000:
            continue
        stimulus = int(stimulus)
        raw_test_ids.add(stimulus)
        if stimulus not in row_by_stimulus:
            raise ValueError(
                f"Raw test stimulus {stimulus} is absent from existing test_stim_idx.npy."
            )
        test_trials[trial] = row_by_stimulus[stimulus]
    if raw_test_ids != set(existing_test_ids.tolist()):
        raise ValueError("Raw and existing test stimulus sets differ; no files were written.")

    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for session_index, session in enumerate(sessions):
        global_start = session_index * trials_per_session
        selected_global = [
            trial
            for trial in range(global_start, global_start + trials_per_session)
            if trial in test_trials
        ]
        if not selected_global:
            continue
        local_indices = np.asarray(
            [trial - global_start for trial in selected_global], dtype=np.int64
        )
        beta_path = betas_dir / f"betas_session{int(session):02d}.nii.gz"
        image = nib.load(str(beta_path))
        rows.append(_load_selected_masked_volumes(image, mask, local_indices))
        labels.append(
            np.asarray([test_trials[trial] for trial in selected_global], dtype=np.int64)
        )
        logger.info(
            "Subject %d reliability: extracted %d test trials from session %d.",
            subject,
            len(selected_global),
            int(session),
        )
    trial_fmri = np.concatenate(rows, axis=0)
    trial_labels = np.concatenate(labels, axis=0)
    max_average_error = validate_trial_average_matches_existing(
        trial_fmri,
        trial_labels,
        existing_test_fmri,
    )

    _atomic_save(subject_dir / "test_fmri_trials.npy", trial_fmri)
    _atomic_save(subject_dir / "test_trial_labels.npy", trial_labels)
    counts = np.unique(trial_labels, return_counts=True)[1]
    summary = {
        "subject": subject,
        "raw_data_root": str(raw_root.resolve()),
        "processed_root": str(Path(processed_root).resolve()),
        "sessions": [int(value) for value in sessions],
        "trial_rows": int(trial_fmri.shape[0]),
        "test_stimuli": int(existing_test_ids.size),
        "repeats_per_stimulus": sorted({int(value) for value in counts}),
        "num_voxels": int(trial_fmri.shape[1]),
        "max_average_error": max_average_error,
        "core_arrays_rewritten": False,
    }
    _atomic_json(subject_dir / "reliability_data_summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--raw-data-root", default=default_raw_data_root())
    parser.add_argument("--processed-root", default="data/processed")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_reliability_data(
                args.sub,
                raw_data_root=args.raw_data_root,
                processed_root=args.processed_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
