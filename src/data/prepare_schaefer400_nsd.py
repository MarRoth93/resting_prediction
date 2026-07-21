"""Prepare parcel-level NSD task and REST arrays for Schaefer-400 training."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.prepare_rest_data import (
    _enabled,
    build_motion_confounds,
    find_motion_file,
    load_motion_params,
    preprocess_rest_run,
    read_tr_from_nifti,
    rest_preprocessing_hash,
)
from src.data.prepare_schaefer400_atlas import prepare_schaefer400_atlas
from src.data.prepare_task_data import discover_sessions, loadmat
from src.data.schaefer400 import (
    N_PARCELS,
    PARCEL_IDS,
    ParcelReducer,
    SchaeferNSDSubjectData,
    validate_schaefer_nsd_subject,
)
from src.pipelines.multiexpert_artifacts import file_sha256
from src.schaefer400_config import load_schaefer400_config, resolve_schaefer400_roots


logger = logging.getLogger(__name__)


def _subject_paths(subject: int, config: dict) -> dict[str, Path]:
    raw_root = Path(config["raw_data_root"])
    output_dir = Path(config["data_root"]) / f"subj{int(subject):02d}"
    return {
        "output": output_dir,
        "atlas": output_dir / str(config["atlas"]["nsd_filename"]),
        "design": raw_root / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat",
        "betas": raw_root
        / "nsddata_betas"
        / "ppdata"
        / f"subj{int(subject):02d}"
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR",
        "timeseries": raw_root
        / "nsddata_timeseries"
        / "ppdata"
        / f"subj{int(subject):02d}"
        / "func1pt8mm"
        / "timeseries",
        "motion": raw_root
        / "nsddata_timeseries"
        / "ppdata"
        / f"subj{int(subject):02d}"
        / "func1pt8mm"
        / "motion",
    }


def _load_reducer(subject: int, config: dict) -> tuple[ParcelReducer, np.ndarray]:
    paths = _subject_paths(subject, config)
    if not paths["atlas"].exists():
        raise FileNotFoundError(
            f"Missing registered Schaefer atlas: {paths['atlas']}. Run prepare-atlas first."
        )
    labels = np.asarray(nib.load(paths["atlas"]).dataobj)
    reducer = ParcelReducer(
        labels,
        min_voxels_per_parcel=int(config["atlas"]["min_voxels_per_parcel"]),
    )
    return reducer, labels


def _stimulus_trial_maps(subject: int, sessions: list[int], design_path: Path):
    design = loadmat(str(design_path))
    masterordering = np.asarray(design["masterordering"])
    subjectim = np.asarray(design["subjectim"])
    num_trials = len(sessions) * 750
    train: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    for trial in range(min(num_trials, masterordering.size)):
        nsd_id = int(subjectim[int(subject) - 1, masterordering[trial] - 1] - 1)
        destination = train if masterordering[trial] > 1000 else test
        destination.setdefault(nsd_id, []).append(trial)
    return train, test, num_trials


def prepare_schaefer400_task_data(
    subject: int,
    config: dict,
    *,
    force: bool = False,
    nifti_chunk_size: int = 25,
) -> dict:
    """Parcellate raw NSD task betas before averaging repeated stimuli."""
    subject = int(subject)
    paths = _subject_paths(subject, config)
    output = paths["output"]
    required = [
        output / "train_fmri.npy",
        output / "test_fmri.npy",
        output / "train_stim_idx.npy",
        output / "test_stim_idx.npy",
        output / "test_fmri_trials.npy",
        output / "test_trial_labels.npy",
        output / "parcel_ids.npy",
        output / "parcel_voxel_counts.npy",
        output / "task_data_summary.json",
    ]
    if all(path.exists() for path in required) and not force:
        summary = json.loads((output / "task_data_summary.json").read_text())
        if summary.get("atlas_name") != config["atlas"]["name"]:
            raise ValueError("Prepared task data use a different atlas.")
        return summary
    reducer, _ = _load_reducer(subject, config)
    if not paths["design"].exists():
        raise FileNotFoundError(f"Missing NSD design: {paths['design']}")
    sessions = discover_sessions(str(paths["betas"]))
    train_trials, test_trials, num_trials = _stimulus_trial_maps(
        subject,
        sessions,
        paths["design"],
    )
    train_stim_idx = np.asarray(sorted(train_trials), dtype=np.int64)
    test_stim_idx = np.asarray(sorted(test_trials), dtype=np.int64)
    trial_responses = np.empty((num_trials, N_PARCELS), dtype=np.float32)
    for position, session in enumerate(sessions):
        beta_path = paths["betas"] / f"betas_session{session:02d}.nii.gz"
        image = nib.load(beta_path)
        if tuple(image.shape[:3]) != reducer.spatial_shape or int(image.shape[3]) != 750:
            raise ValueError(f"Unexpected task beta shape {image.shape}: {beta_path}")
        start = position * 750
        trial_responses[start : start + 750] = reducer.reduce_proxy(
            image.dataobj,
            chunk_size=nifti_chunk_size,
        )
        logger.info("Subject %d: parcellated task session %d/%d", subject, position + 1, len(sessions))

    train_fmri = np.stack(
        [trial_responses[sorted(train_trials[int(stimulus)])].mean(axis=0) for stimulus in train_stim_idx]
    ).astype(np.float32)
    test_fmri = np.stack(
        [trial_responses[sorted(test_trials[int(stimulus)])].mean(axis=0) for stimulus in test_stim_idx]
    ).astype(np.float32)
    test_rows: list[np.ndarray] = []
    test_labels: list[int] = []
    for label, stimulus in enumerate(test_stim_idx):
        for trial in sorted(test_trials[int(stimulus)]):
            test_rows.append(trial_responses[trial])
            test_labels.append(label)

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "train_fmri.npy", train_fmri)
    np.save(output / "test_fmri.npy", test_fmri)
    np.save(output / "train_stim_idx.npy", train_stim_idx)
    np.save(output / "test_stim_idx.npy", test_stim_idx)
    np.save(output / "test_fmri_trials.npy", np.asarray(test_rows, dtype=np.float32))
    np.save(output / "test_trial_labels.npy", np.asarray(test_labels, dtype=np.int64))
    np.save(output / "parcel_ids.npy", PARCEL_IDS)
    np.save(output / "parcel_voxel_counts.npy", reducer.counts)
    summary = {
        "subject": subject,
        "atlas_name": config["atlas"]["name"],
        "representation": "trial_by_400_parcel_mean",
        "stimulus_order": "sorted_nsd_image_id",
        "sessions_used": [int(value) for value in sessions],
        "num_sessions": len(sessions),
        "train_rows": int(train_fmri.shape[0]),
        "test_rows": int(test_fmri.shape[0]),
        "test_trial_rows": len(test_rows),
        "n_parcels": N_PARCELS,
        "atlas_file": str(paths["atlas"].resolve()),
        "atlas_sha256": file_sha256(paths["atlas"]),
        "source_betas_dir": str(paths["betas"].resolve()),
    }
    (output / "task_data_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _discover_rest_files(timeseries_dir: Path) -> list[Path]:
    files = sorted(timeseries_dir.glob("*.nii.gz"))
    rest = [path for path in files if "rest" in path.name.lower()]
    selected = rest or files
    if len(selected) < 2:
        raise ValueError(f"Need at least two NSD REST runs under {timeseries_dir}.")
    return selected


def prepare_schaefer400_rest_data(
    subject: int,
    config: dict,
    *,
    force: bool = False,
    nifti_chunk_size: int = 25,
) -> dict:
    """Parcellate raw REST, then apply NSD cleanup and final parcel z-scoring."""
    subject = int(subject)
    paths = _subject_paths(subject, config)
    output = paths["output"]
    manifest_path = output / "rest_run_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        loaded = SchaeferNSDSubjectData(subject, config["data_root"])
        if not loaded.rest_runs:
            raise ValueError(f"REST manifest exists without run arrays: {manifest_path}")
        if manifest.get("preprocessing_hash") != rest_preprocessing_hash(
            config["rest_preprocessing"]
        ):
            raise ValueError("Prepared Schaefer REST uses a different preprocessing config.")
        return manifest

    reducer, _ = _load_reducer(subject, config)
    rest_cfg = config["rest_preprocessing"]
    source_files = _discover_rest_files(paths["timeseries"])
    processed_runs: list[np.ndarray] = []
    kept_files: list[str] = []
    run_trs: list[float] = []
    for index, source_path in enumerate(source_files, start=1):
        image = nib.load(source_path)
        if tuple(image.shape[:3]) != reducer.spatial_shape:
            raise ValueError(f"REST/atlas shape mismatch: {source_path}")
        tr = read_tr_from_nifti(image)
        parcel_series = reducer.reduce_proxy(image.dataobj, chunk_size=nifti_chunk_size)

        nuisance_raw = rest_cfg.get("nuisance_regression", {}) or {}
        nuisance_cfg = nuisance_raw if isinstance(nuisance_raw, dict) else {}
        nuisance_enabled = _enabled(nuisance_raw, default=False)
        motion_raw = rest_cfg.get("motion_censoring", {}) or {}
        motion_cfg = motion_raw if isinstance(motion_raw, dict) else {}
        motion_enabled = _enabled(motion_raw, default=False)
        motion_params = None
        if motion_enabled or nuisance_enabled:
            motion_path = find_motion_file(str(source_path), str(paths["motion"]))
            if motion_path is None:
                message = f"No motion file found for {source_path.name} under {paths['motion']}"
                if nuisance_enabled and bool(nuisance_cfg.get("require_motion", False)):
                    raise FileNotFoundError(message)
                logger.warning(message)
            else:
                motion_params = load_motion_params(
                    motion_path,
                    expected_trs=parcel_series.shape[0],
                )

        nuisance_regressors = None
        if nuisance_enabled and motion_params is not None:
            nuisance_regressors = build_motion_confounds(
                motion_params,
                model=str(nuisance_cfg.get("motion_model", "friston24")),
                standardize=bool(nuisance_cfg.get("standardize", True)),
            )
        processed = preprocess_rest_run(
            parcel_series,
            tr=tr,
            discard_initial_trs=int(rest_cfg.get("discard_initial_trs", 5)),
            detrend=bool(rest_cfg.get("detrend", True)),
            highpass_cutoff_hz=rest_cfg.get("highpass_cutoff_hz", 0.01),
            motion_params=motion_params,
            motion_censoring_enabled=motion_enabled,
            motion_censoring_strategy=str(motion_cfg.get("strategy", "drop")),
            fd_threshold_mm=float(motion_cfg.get("fd_threshold_mm", 0.5)),
            max_censored_fraction=float(motion_cfg.get("max_censored_fraction", 0.3)),
            nuisance_regressors=nuisance_regressors,
            zscore=bool(rest_cfg.get("zscore", True)),
        )
        if processed is not None:
            processed_runs.append(processed)
            kept_files.append(str(source_path.resolve()))
            run_trs.append(float(tr))
        logger.info("Subject %d: processed REST run %d/%d", subject, index, len(source_files))

    total_trs = sum(int(run.shape[0]) for run in processed_runs)
    if total_trs < int(rest_cfg.get("min_usable_trs", 100)):
        raise ValueError(
            f"Subject {subject}: only {total_trs} usable Schaefer REST TRs remain."
        )
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("rest_run*.npy"):
        stale.unlink()
    for index, run in enumerate(processed_runs, start=1):
        np.save(output / f"rest_run{index}.npy", run)
    np.save(output / "parcel_ids.npy", PARCEL_IDS)
    np.save(output / "parcel_voxel_counts.npy", reducer.counts)
    manifest = {
        "subject": subject,
        "atlas_name": config["atlas"]["name"],
        "representation": "time_by_400_parcels",
        "atlas_file": str(paths["atlas"].resolve()),
        "atlas_sha256": file_sha256(paths["atlas"]),
        "source_files": kept_files,
        "source_tr_seconds": run_trs,
        "run_shapes": [[int(value) for value in run.shape] for run in processed_runs],
        "total_usable_trs": total_trs,
        "preprocessing_hash": rest_preprocessing_hash(rest_cfg),
        "rest_preprocessing": rest_cfg,
        "operation_order": "parcel_mean_then_temporal_cleanup_then_parcel_zscore",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def prepare_schaefer400_nsd_subject(
    subject: int,
    config: dict,
    *,
    allow_download: bool = True,
    force: bool = False,
    prepare_task: bool = True,
    prepare_rest: bool = True,
) -> dict:
    """Prepare atlas plus selected data components and validate the final subject."""
    atlas = prepare_schaefer400_atlas(
        subject,
        config,
        allow_download=allow_download,
        force=force,
    )
    result: dict = {"atlas": atlas}
    if prepare_task:
        result["task"] = prepare_schaefer400_task_data(subject, config, force=force)
    if prepare_rest:
        result["rest"] = prepare_schaefer400_rest_data(subject, config, force=force)
    if prepare_task and prepare_rest:
        validate_schaefer_nsd_subject(
            SchaeferNSDSubjectData(subject, config["data_root"])
        )
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_schaefer400.yaml")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=["all", "task", "rest"], default="all")
    args = parser.parse_args()
    cfg = resolve_schaefer400_roots(
        load_schaefer400_config(args.config),
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
    )
    for subject_id in args.subjects or cfg["subjects"]["train"]:
        prepare_schaefer400_nsd_subject(
            subject_id,
            cfg,
            allow_download=not args.no_download,
            force=args.force,
            prepare_task=args.only in {"all", "task"},
            prepare_rest=args.only in {"all", "rest"},
        )
