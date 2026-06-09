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

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import signal

from src.data.load_atlas import build_analysis_mask
from src.data.shared_paths import default_raw_data_root

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


def _extract_session_run_key(filename: str) -> tuple[int | None, int | None]:
    """
    Extract session/run IDs from NSD-style filenames when present.

    Supports names such as timeseries_session40_run01.nii.gz and
    motion_40_run01.tsv. Returns None for fields that cannot be inferred.
    """
    base = os.path.basename(filename)
    sess = None
    run = None

    sess_match = re.search(r"session[_-]?(\d+)", base, flags=re.IGNORECASE)
    if sess_match:
        sess = int(sess_match.group(1))
    run_match = re.search(r"run[_-]?(\d+)", base, flags=re.IGNORECASE)
    if run_match:
        run = int(run_match.group(1))

    if sess is None:
        # NSD manual documents motion_BB_runCC.tsv, where BB is the session.
        m = re.search(r"motion[_-]?(\d+)[_-]run[_-]?(\d+)", base, flags=re.IGNORECASE)
        if m:
            sess = int(m.group(1))
            run = int(m.group(2))

    return sess, run


def find_motion_file(rest_file: str, motion_dir: str) -> str | None:
    """Find the NSD motion TSV matching a REST timeseries file."""
    if not os.path.exists(motion_dir):
        return None

    sess, run = _extract_session_run_key(rest_file)
    candidates: list[str] = []
    if sess is not None and run is not None:
        candidates.extend([
            f"motion_session{sess:02d}_run{run:02d}.tsv",
            f"motion_session{sess}_run{run}.tsv",
            f"motion_{sess:02d}_run{run:02d}.tsv",
            f"motion_{sess}_run{run}.tsv",
        ])
    if run is not None:
        candidates.extend([
            f"motion_run{run:02d}.tsv",
            f"motion_run{run}.tsv",
        ])

    for candidate in candidates:
        path = os.path.join(motion_dir, candidate)
        if os.path.exists(path):
            return path

    files = sorted(
        f for f in os.listdir(motion_dir)
        if f.lower().endswith((".tsv", ".txt")) and "motion" in f.lower()
    )
    if sess is not None and run is not None:
        matches = [
            f for f in files
            if re.search(fr"(session[_-]?{sess:02d}|session[_-]?{sess}\b|motion[_-]?{sess:02d}|motion[_-]?{sess}\b)", f, flags=re.IGNORECASE)
            and re.search(fr"run[_-]?0*{run}\b", f, flags=re.IGNORECASE)
        ]
        if len(matches) == 1:
            return os.path.join(motion_dir, matches[0])
        if len(matches) > 1:
            logger.warning("Multiple motion files match %s: %s", rest_file, matches)
            return None

    if run is not None:
        matches = [
            f for f in files
            if re.search(fr"run[_-]?0*{run}\b", f, flags=re.IGNORECASE)
        ]
        if len(matches) == 1:
            return os.path.join(motion_dir, matches[0])
        if len(matches) > 1:
            logger.warning(
                "Multiple run-only motion files match %s; need session in filename: %s",
                rest_file,
                matches,
            )
    return None


def load_motion_params(path: str, expected_trs: int | None = None) -> np.ndarray:
    """Load NSD 6-parameter motion file as (T, 6) float32."""
    try:
        arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    except ValueError:
        arr = np.genfromtxt(path, dtype=np.float32, skip_header=1)
        if arr.ndim == 1:
            arr = arr[None, :]

    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Motion file must have at least 6 columns, got {arr.shape}: {path}")
    arr = arr[:, :6].astype(np.float32, copy=False)
    if expected_trs is not None and arr.shape[0] != expected_trs:
        raise ValueError(
            f"Motion rows do not match timeseries TRs for {path}: "
            f"motion={arr.shape[0]}, timeseries={expected_trs}"
        )
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Motion file contains NaN/Inf values: {path}")
    return arr


def build_motion_confounds(
    motion_params: np.ndarray,
    model: str = "friston24",
    standardize: bool = True,
) -> np.ndarray:
    """
    Build motion nuisance regressors from 6 rigid-body parameters.

    Supported models:
    - motion6: original 6 parameters
    - motion12: original 6 + temporal derivatives
    - friston24: motion12 + squared original and derivative terms
    """
    model = str(model).strip().lower()
    motion = np.asarray(motion_params, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise ValueError(f"motion_params must be (T, 6), got {motion.shape}")

    deriv = np.vstack([np.zeros((1, 6), dtype=np.float32), np.diff(motion, axis=0)])
    if model in {"none", "", "false"}:
        confounds = np.zeros((motion.shape[0], 0), dtype=np.float32)
    elif model == "motion6":
        confounds = motion
    elif model == "motion12":
        confounds = np.column_stack([motion, deriv])
    elif model in {"friston24", "friston_24"}:
        confounds = np.column_stack([motion, deriv, motion ** 2, deriv ** 2])
    else:
        raise ValueError(
            f"Unknown motion nuisance model: {model}. "
            "Use motion6, motion12, or friston24."
        )

    if standardize and confounds.shape[1] > 0:
        mean = confounds.mean(axis=0, keepdims=True)
        std = confounds.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        confounds = (confounds - mean) / std
    return confounds.astype(np.float32)


def _enabled(config_value, default: bool = False) -> bool:
    """Read either a boolean config value or a dict with an enabled key."""
    if isinstance(config_value, dict):
        return bool(config_value.get("enabled", default))
    if config_value is None:
        return default
    return bool(config_value)


def build_spike_regressors(censor_mask: np.ndarray) -> np.ndarray:
    """Build one-hot nuisance columns for censored TRs."""
    mask = np.asarray(censor_mask, dtype=bool).ravel()
    n_spikes = int(mask.sum())
    if n_spikes == 0:
        return np.zeros((mask.shape[0], 0), dtype=np.float32)
    spikes = np.zeros((mask.shape[0], n_spikes), dtype=np.float32)
    spikes[np.flatnonzero(mask), np.arange(n_spikes)] = 1.0
    return spikes


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
    motion_censoring_enabled: bool = True,
    motion_censoring_strategy: str = "drop",
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
        motion_censoring_enabled: whether to drop high-FD TRs
        motion_censoring_strategy: drop high-FD TRs before regression ("drop") or
            regress censor spikes first and drop after ("spike_regress_then_drop")
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
    censor_mask = np.zeros(T, dtype=bool)
    strategy = str(motion_censoring_strategy or "drop").strip().lower()
    if motion_censoring_enabled and motion_params is not None:
        fd = compute_framewise_displacement(motion_params)
        keep_mask = fd <= fd_threshold_mm
        censor_mask = ~keep_mask
        censored_frac = 1.0 - keep_mask.mean()
        logger.info(f"  Motion censoring: {censored_frac:.1%} TRs censored (FD > {fd_threshold_mm}mm)")
        if censored_frac > max_censored_fraction:
            logger.warning(f"  Excluding run: {censored_frac:.1%} > {max_censored_fraction:.1%} threshold")
            return None
        if strategy in {"drop", "drop_before_regression"}:
            data = data[keep_mask]
            if nuisance_regressors is not None:
                nuisance_regressors = nuisance_regressors[keep_mask]
        elif strategy in {"spike_regress_then_drop", "spike", "scrub"}:
            spike_regressors = build_spike_regressors(censor_mask)
            if spike_regressors.shape[1] > 0:
                if nuisance_regressors is None:
                    nuisance_regressors = spike_regressors
                else:
                    nuisance_regressors = np.column_stack([nuisance_regressors, spike_regressors])
                logger.info(
                    "  Added %d censor spike regressors before dropping censored TRs",
                    spike_regressors.shape[1],
                )
        else:
            raise ValueError(
                f"Unknown motion_censoring strategy: {motion_censoring_strategy}. "
                "Use 'drop' or 'spike_regress_then_drop'."
            )

    # 5. Nuisance regression
    if nuisance_regressors is not None:
        X = nuisance_regressors
        # Add intercept
        X = np.column_stack([X, np.ones(X.shape[0])])
        betas = np.linalg.lstsq(X, data, rcond=None)[0]
        data = (data - X @ betas).astype(np.float32)

    if (
        motion_censoring_enabled
        and motion_params is not None
        and strategy in {"spike_regress_then_drop", "spike", "scrub"}
    ):
        data = data[keep_mask]

    # 6. Z-score per voxel
    if zscore:
        vox_mean = data.mean(axis=0)
        vox_std = data.std(axis=0)
        vox_std[vox_std < 1e-8] = 1e-8
        data = ((data - vox_mean) / vox_std).astype(np.float32)

    return data


def prepare_rest_data(
    sub: int,
    data_root: str = default_raw_data_root(),
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
            "nuisance_regression": {
                "enabled": False,
                "motion_model": "none",
                "standardize": True,
                "require_motion": False,
            },
            "zscore": True,
            "min_usable_trs": 100,
        }
    analysis_mask_config = config.get("analysis_mask", {}) or {}
    analysis_mask_mode = str(analysis_mask_config.get("mode", "nsdgeneral"))
    atlas_type = str(analysis_mask_config.get("atlas_type", "combined_rois"))
    common_label_subjects = analysis_mask_config.get("common_label_subjects")
    min_voxels_per_parcel = int(analysis_mask_config.get("min_voxels_per_parcel", 10))

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

    # Load mask. By default this is exactly nsdgeneral; optionally restrict to
    # nsdgeneral voxels that are labeled by the configured atlas.
    nsdgeneral_mask = nib.load(os.path.join(roi_dir, "nsdgeneral.nii.gz")).get_fdata() > 0
    mask, mask_summary = build_analysis_mask(
        sub=sub,
        nsdgeneral_mask=nsdgeneral_mask,
        mode=analysis_mask_mode,
        atlas_type=atlas_type,
        data_root=data_root,
        common_label_subjects=common_label_subjects,
        min_voxels_per_parcel=min_voxels_per_parcel,
    )
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

        # Load motion params if available. NSD documents these under:
        # nsddata_timeseries/ppdata/subjXX/func*/motion/motion_BB_runCC.tsv
        motion_params = None
        nuisance_raw_cfg = config.get("nuisance_regression", {}) or {}
        nuisance_enabled = _enabled(nuisance_raw_cfg, default=False)
        nuisance_cfg = nuisance_raw_cfg if isinstance(nuisance_raw_cfg, dict) else {}
        motion_censoring_enabled = config.get("motion_censoring", {}).get("enabled", False)
        needs_motion = motion_censoring_enabled or nuisance_enabled
        if needs_motion:
            motion_dir = os.path.join(
                data_root, f"nsddata_timeseries/ppdata/subj{sub:02d}/func1pt8mm/motion/"
            )
            motion_file = find_motion_file(rest_file, motion_dir)
            if motion_file is not None:
                motion_params = load_motion_params(motion_file, expected_trs=run_data.shape[0])
                logger.info(f"  Loaded motion params: {motion_file}")
            else:
                msg = f"  No motion params found for {rest_file} in {motion_dir}"
                if nuisance_enabled and bool(nuisance_cfg.get("require_motion", False)):
                    raise FileNotFoundError(msg)
                logger.info(f"{msg}; skipping motion-based steps")

        nuisance_regressors = None
        if nuisance_enabled:
            motion_model = str(nuisance_cfg.get("motion_model", "friston24"))
            if motion_params is not None and motion_model.lower() not in {"none", "", "false"}:
                nuisance_regressors = build_motion_confounds(
                    motion_params,
                    model=motion_model,
                    standardize=bool(nuisance_cfg.get("standardize", True)),
                )
                logger.info(
                    "  Built nuisance regressors: model=%s, shape=%s",
                    motion_model,
                    nuisance_regressors.shape,
                )

        processed = preprocess_rest_run(
            run_data,
            tr=tr,
            discard_initial_trs=config.get("discard_initial_trs", 5),
            detrend=config.get("detrend", True),
            highpass_cutoff_hz=config.get("highpass_cutoff_hz", 0.01),
            motion_params=motion_params,
            motion_censoring_enabled=motion_censoring_enabled,
            motion_censoring_strategy=str(
                config.get("motion_censoring", {}).get("strategy", "drop")
            ),
            fd_threshold_mm=config.get("motion_censoring", {}).get("fd_threshold_mm", 0.5),
            max_censored_fraction=config.get("motion_censoring", {}).get("max_censored_fraction", 0.3),
            nuisance_regressors=nuisance_regressors,
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
    with open(os.path.join(out_dir, "rest_analysis_mask_summary.json"), "w") as f:
        json.dump(mask_summary, f, indent=2)

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
    parser.add_argument("--data-root", default=default_raw_data_root())
    parser.add_argument("--output-root", default="processed_data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f) or {}
            cfg = full_cfg.get("rest_preprocessing", {}) or {}
            mask_cfg = full_cfg.get("analysis_mask", {}) or {}
            if mask_cfg:
                alignment_cfg = full_cfg.get("alignment", {}) or {}
                subject_cfg = full_cfg.get("subjects", {}) or {}
                common_label_subjects = None
                if bool(mask_cfg.get("use_common_labels", True)):
                    common_label_subjects = (
                        list(subject_cfg.get("train", [])) + list(subject_cfg.get("test", []))
                    )
                    if not common_label_subjects:
                        common_label_subjects = None
                cfg["analysis_mask"] = {
                    "mode": mask_cfg.get("mode", "nsdgeneral"),
                    "atlas_type": mask_cfg.get(
                        "atlas_type",
                        alignment_cfg.get("atlas_type", "combined_rois"),
                    ),
                    "common_label_subjects": common_label_subjects,
                    "min_voxels_per_parcel": int(
                        mask_cfg.get(
                            "min_voxels_per_parcel",
                            alignment_cfg.get("min_voxels_per_parcel", 10),
                        )
                    ),
                }

    prepare_rest_data(args.sub, args.data_root, args.output_root, cfg)
