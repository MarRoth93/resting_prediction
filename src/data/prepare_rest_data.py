"""
Prepare resting-state fMRI data for one subject.

Preprocessing pipeline per run:
1. Discard first N TRs (T1 equilibration)
2. Linear detrending
3. High-pass filter (Butterworth)
4. Motion censoring (drop high-FD TRs)
5. Nuisance regression (optional)
6. Z-score per voxel
"""

import json
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def read_tr_from_nifti(img: nib.Nifti1Image) -> float:
    """Read TR from NIfTI header. Returns TR in seconds."""
    zooms = img.header.get_zooms()
    if len(zooms) >= 4:
        tr = float(zooms[3])
        if tr > 0:
            return tr
    logger.warning("Could not read TR from NIfTI header, using default 1.333s")
    return 1.333


def compute_framewise_displacement(motion_params: np.ndarray) -> np.ndarray:
    """
    Compute framewise displacement from 6 motion parameters.

    Args:
        motion_params: (T, 6) — 3 translation (mm) + 3 rotation (rad)

    Returns:
        fd: (T,) framewise displacement in mm
    """
    # Convert rotations to mm (assuming 50mm head radius)
    params = motion_params.copy()
    params[:, 3:] *= 50.0  # rad to mm on 50mm sphere

    diff = np.diff(params, axis=0)
    fd = np.abs(diff).sum(axis=1)
    # First TR has no displacement
    return np.concatenate([[0.0], fd])


def highpass_filter(
    data: np.ndarray,
    cutoff_hz: float,
    tr: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply Butterworth high-pass filter along time axis (axis=0).

    Args:
        data: (T, V) time series
        cutoff_hz: cutoff frequency in Hz
        tr: repetition time in seconds
        order: filter order
    """
    fs = 1.0 / tr
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        logger.warning(f"Cutoff {cutoff_hz} Hz >= Nyquist {nyquist} Hz, skipping filter")
        return data
    b, a = signal.butter(order, cutoff_hz / nyquist, btype="high")
    return signal.filtfilt(b, a, data, axis=0).astype(data.dtype)


def preprocess_rest_run(
    run_data: np.ndarray,
    tr: float,
    discard_initial_trs: int = 5,
    detrend: bool = True,
    highpass_cutoff_hz: float | None = 0.01,
    motion_params: np.ndarray | None = None,
    fd_threshold_mm: float = 0.5,
    max_censored_fraction: float = 0.3,
    nuisance_regressors: np.ndarray | None = None,
    zscore: bool = True,
) -> np.ndarray | None:
    """
    Preprocess a single REST run.

    Args:
        run_data: (T, V) raw timeseries, float32
        tr: repetition time in seconds
        discard_initial_trs: number of initial TRs to discard
        detrend: apply linear detrending
        highpass_cutoff_hz: high-pass filter cutoff, None to skip
        motion_params: (T, 6) motion parameters for censoring
        fd_threshold_mm: FD threshold for censoring
        max_censored_fraction: max fraction of TRs to censor before excluding run
        nuisance_regressors: (T, R) nuisance regressors
        zscore: z-score per voxel

    Returns:
        Preprocessed (T_clean, V) array, or None if run is excluded
    """
    T_orig = run_data.shape[0]

    # 1. Discard initial TRs
    data = run_data[discard_initial_trs:].copy()
    if motion_params is not None:
        motion_params = motion_params[discard_initial_trs:]
    if nuisance_regressors is not None:
        nuisance_regressors = nuisance_regressors[discard_initial_trs:]

    T = data.shape[0]
    if T < 30:
        logger.warning(f"Run has only {T} TRs after discarding initial, excluding")
        return None

    # 2. Linear detrending
    if detrend:
        data = signal.detrend(data, axis=0, type="linear").astype(np.float32)

    # 3. High-pass filter
    if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0:
        data = highpass_filter(data, highpass_cutoff_hz, tr)

    # 4. Motion censoring
    keep_mask = np.ones(T, dtype=bool)
    if motion_params is not None:
        fd = compute_framewise_displacement(motion_params)
        keep_mask = fd <= fd_threshold_mm
        censored_frac = 1.0 - keep_mask.mean()
        logger.info(f"  Motion censoring: {censored_frac:.1%} TRs censored (FD > {fd_threshold_mm}mm)")
        if censored_frac > max_censored_fraction:
            logger.warning(f"  Excluding run: {censored_frac:.1%} > {max_censored_fraction:.1%} threshold")
            return None
        data = data[keep_mask]
        if nuisance_regressors is not None:
            nuisance_regressors = nuisance_regressors[keep_mask]

    # 5. Nuisance regression
    if nuisance_regressors is not None:
        X = nuisance_regressors
        # Add intercept
        X = np.column_stack([X, np.ones(X.shape[0])])
        betas = np.linalg.lstsq(X, data, rcond=None)[0]
        data = (data - X @ betas).astype(np.float32)

    # 6. Z-score per voxel
    if zscore:
        vox_mean = data.mean(axis=0)
        vox_std = data.std(axis=0)
        vox_std[vox_std < 1e-8] = 1e-8
        data = ((data - vox_mean) / vox_std).astype(np.float32)

    return data


def prepare_rest_data(
    sub: int,
    data_root: str = ".",
    output_root: str = "processed_data",
    config: dict | None = None,
) -> dict:
    """
    Prepare resting-state data for one subject.

    Returns dict with:
        rest_runs: list[np.ndarray] — each (T_run, V_sub) preprocessed
        mask: np.ndarray — 3D nsdgeneral mask
        num_voxels: int
        num_usable_trs: int — total TRs across all kept runs
    """
    if config is None:
        config = {
            "discard_initial_trs": 5,
            "detrend": True,
            "highpass_cutoff_hz": 0.01,
            "motion_censoring": {"enabled": True, "fd_threshold_mm": 0.5, "max_censored_fraction": 0.3},
            "nuisance_regression": False,
            "zscore": True,
            "min_usable_trs": 100,
        }

    roi_dir = os.path.join(data_root, f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/")
    ts_dir = os.path.join(data_root, f"nsddata_timeseries/ppdata/subj{sub:02d}/func1pt8mm/timeseries/")
    out_dir = os.path.join(output_root, f"subj{sub:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # Load manifest if available
    manifest_path = os.path.join(out_dir, "rest_run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        rest_files = manifest["rest_runs"]
        logger.info(f"Subject {sub}: loaded REST manifest with {len(rest_files)} runs")
    else:
        # Discover REST files by listing directory
        rest_files = sorted([
            f for f in os.listdir(ts_dir)
            if f.endswith(".nii.gz") and "rest" in f.lower()
        ]) if os.path.exists(ts_dir) else []
        if not rest_files:
            # Try all timeseries and log warning
            logger.warning(f"Subject {sub}: no REST files found with 'rest' pattern, listing all timeseries")
            if os.path.exists(ts_dir):
                rest_files = sorted([f for f in os.listdir(ts_dir) if f.endswith(".nii.gz")])

    if len(rest_files) < 2:
        raise ValueError(
            f"Subject {sub}: found {len(rest_files)} REST runs, need >= 2. "
            f"Checked: {ts_dir}"
        )

    # Load mask
    mask = nib.load(os.path.join(roi_dir, "nsdgeneral.nii.gz")).get_fdata() > 0
    num_voxels = int(mask.sum())

    # Process each run
    rest_runs = []
    for i, rest_file in enumerate(rest_files):
        logger.info(f"Subject {sub}: processing REST run {i+1}/{len(rest_files)}: {rest_file}")
        img = nib.load(os.path.join(ts_dir, rest_file))
        tr = read_tr_from_nifti(img)
        if abs(tr - 1.333) > 0.5:
            logger.warning(f"  Unexpected TR={tr:.3f}s (expected ~1.333s)")

        raw_data = img.get_fdata().astype(np.float32)
        # Apply mask: (X, Y, Z, T) -> (T, V)
        run_data = raw_data[mask].T
        del raw_data

        # Load motion params if available
        motion_params = None
        if config.get("motion_censoring", {}).get("enabled", False):
            motion_dir = os.path.join(
                data_root, f"nsddata_timeseries/ppdata/subj{sub:02d}/func1pt8mm/motion/"
            )
            motion_file = os.path.join(motion_dir, rest_file.replace(".nii.gz", "_motion.txt"))
            if os.path.exists(motion_file):
                motion_params = np.loadtxt(motion_file)
            else:
                logger.info(f"  No motion params found at {motion_file}, skipping censoring")

        processed = preprocess_rest_run(
            run_data,
            tr=tr,
            discard_initial_trs=config.get("discard_initial_trs", 5),
            detrend=config.get("detrend", True),
            highpass_cutoff_hz=config.get("highpass_cutoff_hz", 0.01),
            motion_params=motion_params,
            fd_threshold_mm=config.get("motion_censoring", {}).get("fd_threshold_mm", 0.5),
            max_censored_fraction=config.get("motion_censoring", {}).get("max_censored_fraction", 0.3),
            zscore=config.get("zscore", True),
        )

        if processed is not None:
            rest_runs.append(processed)
        else:
            logger.warning(f"  Run {rest_file} excluded")

    # Save with contiguous indices (no gaps) so loader can enumerate sequentially
    for run_idx, run_data in enumerate(rest_runs):
        np.save(os.path.join(out_dir, f"rest_run{run_idx + 1}.npy"), run_data)
        logger.info(f"  Saved rest_run{run_idx + 1}.npy: {run_data.shape}")

    # Check minimum usable TRs
    total_trs = sum(r.shape[0] for r in rest_runs)
    min_trs = config.get("min_usable_trs", 100)
    if total_trs < min_trs:
        raise ValueError(
            f"Subject {sub}: only {total_trs} usable TRs across {len(rest_runs)} runs "
            f"(need >= {min_trs})"
        )

    logger.info(f"Subject {sub}: {len(rest_runs)} REST runs, {total_trs} total TRs, {num_voxels} voxels")

    return {
        "rest_runs": rest_runs,
        "mask": mask,
        "num_voxels": num_voxels,
        "num_usable_trs": total_trs,
    }


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Prepare REST data for one subject")
    parser.add_argument("-sub", "--sub", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-root", default="processed_data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f).get("rest_preprocessing", {})

    prepare_rest_data(args.sub, args.data_root, args.output_root, cfg)
