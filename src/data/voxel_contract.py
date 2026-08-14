"""Build the experimental NSD/FOR parcel-seed to voxel-target data contract."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import nibabel as nib
import numpy as np

from src.pipelines.multiexpert_artifacts import file_sha256


logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
N_PARCELS = 400
N_FOR_TIMEPOINTS = 237
FOR_DISCARD_INITIAL_TRS = 2
TRAIN_NSD_SUBJECTS = tuple(range(1, 7))
NSD_ATLAS_FILENAME = "Schaefer2018_400Parcels_7Networks_order_func1pt8.nii.gz"
FOR_ATLAS_FILENAME = "schaefer400_bold_raw.nii.gz"
FOR_PARCEL_TIMESERIES_FILENAME = "schaefer400_parcel_timeseries.mat"

POLICIES = {
    "seed_missing": "zero_fill_and_mask",
    "for_discard_initial_trs": FOR_DISCARD_INITIAL_TRS,
    "for_zscore": True,
    "for_detrend": False,
    "for_highpass": False,
    "for_motion": "unavailable_by_design",
}


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _integer_labels(values: np.ndarray, *, source: Path) -> np.ndarray:
    labels = np.asarray(values)
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, atol=1e-5, rtol=0):
        raise ValueError(f"Atlas contains non-integer labels: {source}")
    return rounded.astype(np.int32)


def _load_nsd_masked_labels(
    subject: int,
    *,
    nsd_data_root: str | Path,
    nsd_parcel_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    subject_tag = f"subj{int(subject):02d}"
    mask_path = Path(nsd_data_root) / subject_tag / "mask.npy"
    atlas_path = Path(nsd_parcel_root) / subject_tag / NSD_ATLAS_FILENAME
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing NSD mask: {mask_path}")
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Missing NSD Schaefer atlas: {atlas_path}")

    mask = np.asarray(np.load(mask_path), dtype=bool)
    atlas = _integer_labels(
        np.asarray(nib.load(str(atlas_path)).dataobj),
        source=atlas_path,
    )
    if atlas.shape != mask.shape:
        raise ValueError(
            f"NSD mask/atlas shape mismatch for {subject_tag}: "
            f"{mask.shape} versus {atlas.shape}."
        )
    return mask, atlas[mask], atlas_path


def compute_common_parcel_ids(
    parcel_counts_by_subject: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return the sorted parcel intersection, requiring at least 60 parcels."""
    if not parcel_counts_by_subject:
        raise ValueError("No NSD parcel counts were provided.")
    present = []
    for subject_tag, raw_counts in parcel_counts_by_subject.items():
        counts = np.asarray(raw_counts)
        if counts.shape != (N_PARCELS,):
            raise ValueError(
                f"NSD parcel counts for {subject_tag} must have length {N_PARCELS}."
            )
        present.append(counts >= 1)
    common = np.flatnonzero(np.logical_and.reduce(present)).astype(np.int16) + 1
    if common.size < 60:
        raise ValueError(
            f"NSD common parcel intersection has {common.size} parcels; "
            "at least 60 are required."
        )
    return common


def build_nsd_contract(
    *,
    output_root: str | Path = "data/processed_voxel_contract",
    nsd_data_root: str | Path = "data/processed",
    nsd_parcel_root: str | Path = "data/processed_schaefer400",
) -> dict:
    """Build and atomically write the contract manifest from NSD subjects 1--6."""
    parcel_counts: dict[str, np.ndarray] = {}
    atlas_hashes: dict[str, str] = {}
    for subject in TRAIN_NSD_SUBJECTS:
        subject_tag = f"subj{subject:02d}"
        _, labels, atlas_path = _load_nsd_masked_labels(
            subject,
            nsd_data_root=nsd_data_root,
            nsd_parcel_root=nsd_parcel_root,
        )
        valid = labels[(labels >= 1) & (labels <= N_PARCELS)]
        parcel_counts[subject_tag] = np.bincount(
            valid,
            minlength=N_PARCELS + 1,
        )[1:].astype(np.int64)
        atlas_hashes[subject_tag] = file_sha256(atlas_path)

    common_parcel_ids = compute_common_parcel_ids(parcel_counts)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "common_parcel_ids": common_parcel_ids.astype(int).tolist(),
        "nsd_parcel_voxel_counts": {
            subject_tag: counts.astype(int).tolist()
            for subject_tag, counts in sorted(parcel_counts.items())
        },
        "nsd_atlas_sha256": dict(sorted(atlas_hashes.items())),
        "policies": POLICIES,
    }
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path / "contract.json", manifest)
    return manifest


def _load_contract(output_root: str | Path) -> tuple[dict, np.ndarray]:
    contract_path = Path(output_root) / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing voxel contract manifest: {contract_path}. Build the NSD dataset first."
        )
    contract = json.loads(contract_path.read_text())
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported voxel contract version: {contract.get('contract_version')!r}."
        )
    common = np.asarray(contract.get("common_parcel_ids", []), dtype=np.int64)
    if (
        common.ndim != 1
        or common.size == 0
        or np.any(common < 1)
        or np.any(common > N_PARCELS)
        or not np.array_equal(common, np.unique(common))
    ):
        raise ValueError("contract.json has invalid common_parcel_ids.")
    return contract, common


def _aligned_nsd_rest_runs(voxel_dir: Path, parcel_dir: Path) -> list[dict]:
    voxel_files = {path.name: path for path in voxel_dir.glob("rest_run*.npy")}
    parcel_files = {path.name: path for path in parcel_dir.glob("rest_run*.npy")}
    if set(voxel_files) != set(parcel_files):
        voxel_only = sorted(set(voxel_files) - set(parcel_files))
        parcel_only = sorted(set(parcel_files) - set(voxel_files))
        raise ValueError(
            "NSD voxel/parcel REST run filename sets differ: "
            f"voxel_only={voxel_only}, parcel_only={parcel_only}."
        )
    if not voxel_files:
        raise ValueError(f"No NSD REST run arrays found under {voxel_dir}.")

    run_table = []
    for filename in sorted(voxel_files):
        voxel_shape = np.load(voxel_files[filename], mmap_mode="r").shape
        parcel_shape = np.load(parcel_files[filename], mmap_mode="r").shape
        voxel_trs = int(voxel_shape[0])
        parcel_trs = int(parcel_shape[0])
        if voxel_trs != parcel_trs:
            raise ValueError(
                f"NSD REST TR mismatch for {filename}: "
                f"voxel={voxel_trs}, parcel={parcel_trs}."
            )
        run_table.append(
            {
                "filename": filename,
                "n_trs": voxel_trs,
                "voxel_path": str(voxel_files[filename].resolve()),
                "parcel_path": str(parcel_files[filename].resolve()),
            }
        )
    return run_table


def build_nsd_subject(
    sub: int,
    *,
    output_root: str | Path = "data/processed_voxel_contract",
    nsd_data_root: str | Path = "data/processed",
    nsd_parcel_root: str | Path = "data/processed_schaefer400",
    force: bool = False,
) -> dict:
    """Build one NSD voxel-target bundle that references existing REST arrays."""
    subject = int(sub)
    subject_tag = f"subj{subject:02d}"
    output_dir = Path(output_root) / "nsd" / subject_tag
    _, common_parcel_ids = _load_contract(output_root)
    required_outputs = [
        output_dir / "target_voxel_indices.npy",
        output_dir / "target_parcel_ids.npy",
        output_dir / "provenance.json",
    ]
    if all(path.is_file() for path in required_outputs) and not force:
        return json.loads((output_dir / "provenance.json").read_text())

    voxel_dir = Path(nsd_data_root) / subject_tag
    parcel_dir = Path(nsd_parcel_root) / subject_tag
    if not parcel_dir.is_dir():
        raise FileNotFoundError(
            f"Missing NSD parcel directory: {parcel_dir}. Run "
            f"`python -m src.data.prepare_schaefer400_nsd --subjects {subject} --only rest`."
        )
    mask, labels, atlas_path = _load_nsd_masked_labels(
        subject,
        nsd_data_root=nsd_data_root,
        nsd_parcel_root=nsd_parcel_root,
    )
    run_table = _aligned_nsd_rest_runs(voxel_dir, parcel_dir)
    target_mask = np.isin(labels, common_parcel_ids)
    target_voxel_indices = np.flatnonzero(target_mask).astype(np.int64)
    target_parcel_ids = labels[target_mask].astype(np.int16)
    valid_labels = labels[(labels >= 1) & (labels <= N_PARCELS)]
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "n_mask_voxels": int(mask.sum()),
        "n_target_voxels": int(target_voxel_indices.size),
        "n_dropped_label0": int(np.count_nonzero(labels == 0)),
        "n_parcels_present": int(np.unique(valid_labels).size),
        "rest_runs": run_table,
        "atlas_sha256": file_sha256(atlas_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(output_dir / "target_voxel_indices.npy", target_voxel_indices)
    _atomic_save(output_dir / "target_parcel_ids.npy", target_parcel_ids)
    _atomic_json(output_dir / "provenance.json", provenance)
    return provenance


def decode_freesurfer_parcel_labels(values: np.ndarray) -> np.ndarray:
    """Decode FOR FreeSurfer-coded cortical labels to Schaefer IDs 1..400."""
    labels = np.asarray(values)
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, atol=1e-5, rtol=0):
        raise ValueError("FOR Schaefer atlas contains non-integer labels.")
    labels = rounded.astype(np.int32)
    decoded = np.zeros(labels.shape, dtype=np.int16)
    left = (labels >= 1001) & (labels <= 1200)
    right = (labels >= 2001) & (labels <= 2200)
    decoded[left] = labels[left] - 1000
    decoded[right] = labels[right] - 1800
    return decoded


def _validate_for_coordinates(voxel_ijk: np.ndarray, grid: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(voxel_ijk)
    rounded = np.rint(coordinates)
    if not np.allclose(coordinates, rounded, atol=1e-5, rtol=0):
        raise ValueError("FOR voxel_ijk must contain integer 1-based coordinates.")
    coordinates = rounded.astype(np.int64)
    if coordinates.ndim != 2 or coordinates.shape[0] != 3:
        raise ValueError(f"FOR voxel_ijk must have shape (3, n_vox), got {coordinates.shape}.")
    grid = np.asarray(grid, dtype=np.int64).ravel()
    if grid.shape != (3,) or np.any(grid < 1):
        raise ValueError(f"FOR bold_spatial_size must contain three positive axes, got {grid}.")
    if coordinates.shape[1] == 0:
        raise ValueError("FOR voxel_ijk contains no voxels.")
    minimum = coordinates.min(axis=1)
    maximum = coordinates.max(axis=1)
    if np.any(minimum < 1) or np.any(maximum > grid):
        raise ValueError(
            "FOR voxel_ijk coordinates violate the expected 1-based indexing: "
            f"mins={minimum.tolist()}, maxes={maximum.tolist()}, grid={grid.tolist()}."
        )
    return coordinates


def _hdf_scalar(handle: h5py.File, name: str) -> float:
    if name not in handle:
        raise ValueError(f"FOR MATLAB file is missing {name!r}.")
    values = np.asarray(handle[name]).squeeze()
    if values.size != 1:
        raise ValueError(f"FOR field {name!r} must be scalar, got {values.shape}.")
    return float(values)


def _load_for_voxel_mat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FOR voxel time-series file: {path}")
    with h5py.File(path, "r") as handle:
        required = {"voxel_timeseries", "voxel_ijk", "bold_spatial_size", "TR_seconds"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"FOR MATLAB file is missing fields {missing}: {path}")
        timeseries = np.asarray(handle["voxel_timeseries"], dtype=np.float32)
        coordinates = np.asarray(handle["voxel_ijk"])
        grid = np.asarray(handle["bold_spatial_size"]).astype(np.int64).ravel()
        tr_seconds = _hdf_scalar(handle, "TR_seconds")

    if timeseries.ndim != 2 or timeseries.shape[1] != N_FOR_TIMEPOINTS:
        raise ValueError(
            f"FOR voxel_timeseries must have shape (n_vox, {N_FOR_TIMEPOINTS}), "
            f"got {timeseries.shape}."
        )
    coordinates = _validate_for_coordinates(coordinates, grid)
    if coordinates.shape[1] != timeseries.shape[0]:
        raise ValueError(
            "FOR voxel_ijk and voxel_timeseries disagree on the voxel count: "
            f"{coordinates.shape[1]} versus {timeseries.shape[0]}."
        )
    if not np.isclose(tr_seconds, 2.0, atol=1e-6, rtol=0):
        raise ValueError(f"FOR TR_seconds must be 2.0, got {tr_seconds}.")
    if not np.all(np.isfinite(timeseries)):
        raise ValueError(f"FOR voxel_timeseries contains NaN/Inf: {path}")
    return timeseries, coordinates, grid, tr_seconds


def _zscore_columns(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float32)
    means = values.mean(axis=0, dtype=np.float64)
    standard_deviations = values.std(axis=0, dtype=np.float64)
    guarded = standard_deviations < 1e-8
    denominators = standard_deviations.copy()
    denominators[guarded] = 1.0
    standardized = (values.astype(np.float64) - means) / denominators
    return standardized.astype(np.float32), int(guarded.sum())


def _parcel_means(
    time_by_voxel: np.ndarray,
    voxel_parcel_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    series = np.zeros((time_by_voxel.shape[0], N_PARCELS), dtype=np.float32)
    available = np.zeros(N_PARCELS, dtype=bool)
    for parcel_id in range(1, N_PARCELS + 1):
        selected = voxel_parcel_ids == parcel_id
        if np.any(selected):
            series[:, parcel_id - 1] = time_by_voxel[:, selected].mean(axis=1)
            available[parcel_id - 1] = True
    return series, available


def _parcel_agreement(
    raw_time_by_voxel: np.ndarray,
    voxel_parcel_ids: np.ndarray,
    available: np.ndarray,
    parcel_mat_path: Path,
) -> tuple[float, list[int]]:
    if not parcel_mat_path.is_file():
        raise FileNotFoundError(f"Missing FOR shipped parcel time series: {parcel_mat_path}")
    with h5py.File(parcel_mat_path, "r") as handle:
        if "parcel_timeseries" not in handle:
            raise ValueError(f"FOR parcel MATLAB file has no parcel_timeseries: {parcel_mat_path}")
        shipped = np.asarray(handle["parcel_timeseries"], dtype=np.float32)
    if shipped.shape != (N_PARCELS, N_FOR_TIMEPOINTS):
        raise ValueError(
            "FOR shipped parcel_timeseries must have shape "
            f"({N_PARCELS}, {N_FOR_TIMEPOINTS}), got {shipped.shape}."
        )

    agreement_parcels = (np.flatnonzero(available)[:5] + 1).astype(int).tolist()
    if len(agreement_parcels) < 5:
        raise ValueError("FOR subject has fewer than five parcels for agreement validation.")
    correlations = []
    for parcel_id in agreement_parcels:
        recomputed = raw_time_by_voxel[:, voxel_parcel_ids == parcel_id].mean(axis=1)
        expected = shipped[parcel_id - 1]
        recomputed_centered = recomputed.astype(np.float64) - float(recomputed.mean())
        expected_centered = expected.astype(np.float64) - float(expected.mean())
        denominator = float(
            np.linalg.norm(recomputed_centered) * np.linalg.norm(expected_centered)
        )
        correlation = (
            float(np.dot(recomputed_centered, expected_centered) / denominator)
            if denominator >= 1e-8
            else float("nan")
        )
        correlations.append(correlation)
    minimum = float(np.min(correlations))
    if not np.all(np.asarray(correlations) > 0.999):
        raise ValueError(
            "FOR raw voxel-to-parcel agreement must exceed 0.999 for all checked "
            f"parcels; correlations={correlations}."
        )
    return minimum, agreement_parcels


def build_for_subject(
    sub_label: str,
    *,
    output_root: str | Path = "data/processed_voxel_contract",
    for_voxel_root: str | Path = "/media/psycontrol/HDD/Datasets/FOR/voxel_timeseries",
    for_atlas_root: str | Path = "/media/psycontrol/HDD/Datasets/FOR/subjects_400_rest",
    force: bool = False,
) -> dict:
    """Build one FOR seed/voxel bundle from the raw-scale cleaned time series."""
    if re.fullmatch(r"sub-\d{4}", str(sub_label)) is None:
        raise ValueError(f"FOR subject label must match sub-XXXX, got {sub_label!r}.")
    subject_label = str(sub_label)
    output_dir = Path(output_root) / "for" / subject_label
    _, common_parcel_ids = _load_contract(output_root)
    required_outputs = [
        output_dir / "rest_targets.npy",
        output_dir / "rest_seeds.npy",
        output_dir / "seed_available.npy",
        output_dir / "target_parcel_ids.npy",
        output_dir / "provenance.json",
    ]
    if all(path.is_file() for path in required_outputs) and not force:
        return json.loads((output_dir / "provenance.json").read_text())

    source_mat = Path(for_voxel_root) / f"{subject_label}_time_by_voxel.mat"
    atlas_dir = Path(for_atlas_root) / subject_label
    if not atlas_dir.is_dir():
        raise FileNotFoundError(f"Missing FOR atlas directory for {subject_label}: {atlas_dir}")
    atlas_path = atlas_dir / FOR_ATLAS_FILENAME
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Missing FOR Schaefer atlas for {subject_label}: {atlas_path}")

    voxel_timeseries, voxel_ijk, grid, tr_seconds = _load_for_voxel_mat(source_mat)
    atlas = _integer_labels(
        np.asarray(nib.load(str(atlas_path)).dataobj),
        source=atlas_path,
    )
    if atlas.ndim != 3 or tuple(atlas.shape) != tuple(int(value) for value in grid):
        raise ValueError(
            f"FOR atlas/grid shape mismatch for {subject_label}: "
            f"{atlas.shape} versus {tuple(grid)}."
        )
    zero_based = voxel_ijk - 1
    encoded_labels = atlas[tuple(zero_based)]
    voxel_parcel_ids = decode_freesurfer_parcel_labels(encoded_labels)

    raw_time_by_voxel = voxel_timeseries.T
    min_agreement_r, agreement_parcels = _parcel_agreement(
        raw_time_by_voxel,
        voxel_parcel_ids,
        np.asarray(
            [np.any(voxel_parcel_ids == parcel_id) for parcel_id in range(1, N_PARCELS + 1)],
            dtype=bool,
        ),
        atlas_dir / FOR_PARCEL_TIMESERIES_FILENAME,
    )
    post_drop = raw_time_by_voxel[FOR_DISCARD_INITIAL_TRS:]
    raw_seeds, seed_available = _parcel_means(post_drop, voxel_parcel_ids)
    rest_seeds = np.zeros_like(raw_seeds, dtype=np.float32)
    standardized_seeds, n_zero_variance_seed_columns = _zscore_columns(
        raw_seeds[:, seed_available]
    )
    rest_seeds[:, seed_available] = standardized_seeds

    target_mask = np.isin(voxel_parcel_ids, common_parcel_ids)
    target_parcel_ids = voxel_parcel_ids[target_mask].astype(np.int16)
    rest_targets, n_zero_variance_target_voxels = _zscore_columns(
        post_drop[:, target_mask]
    )
    provenance = {
        "source_mat_sha256": file_sha256(source_mat),
        "atlas_sha256": file_sha256(atlas_path),
        "n_vox_wholebrain": int(voxel_timeseries.shape[0]),
        "n_target_voxels": int(target_parcel_ids.size),
        "parcels_available": int(seed_available.sum()),
        "min_parcel_agreement_r": min_agreement_r,
        "agreement_parcel_ids": agreement_parcels,
        "n_zero_variance_seed_columns": n_zero_variance_seed_columns,
        "n_zero_variance_target_voxels": n_zero_variance_target_voxels,
        "preprocessing": {
            "source": "ica_fix_cleaned_effectively_detrended_and_highpass_filtered",
            "discard_initial_trs": FOR_DISCARD_INITIAL_TRS,
            "zscore": True,
            "detrend": False,
            "highpass": False,
            "motion": "unavailable_by_design",
            "tr_seconds": tr_seconds,
        },
        "contract_version": CONTRACT_VERSION,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(output_dir / "rest_targets.npy", rest_targets.astype(np.float32))
    _atomic_save(output_dir / "rest_seeds.npy", rest_seeds.astype(np.float32))
    _atomic_save(output_dir / "seed_available.npy", seed_available)
    _atomic_save(output_dir / "target_parcel_ids.npy", target_parcel_ids)
    _atomic_json(output_dir / "provenance.json", provenance)
    return provenance


def _discover_for_subjects(voxel_root: str | Path) -> list[str]:
    suffix = "_time_by_voxel.mat"
    labels = [
        path.name[: -len(suffix)]
        for path in sorted(Path(voxel_root).glob(f"sub-*{suffix}"))
    ]
    labels = [label for label in labels if re.fullmatch(r"sub-\d{4}", label)]
    if not labels:
        raise FileNotFoundError(f"No FOR voxel time-series files found under {voxel_root}.")
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["nsd", "for"])
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--output-root", default="data/processed_voxel_contract")
    parser.add_argument("--nsd-data-root", default="data/processed")
    parser.add_argument("--nsd-parcel-root", default="data/processed_schaefer400")
    parser.add_argument(
        "--for-voxel-root",
        default="/media/psycontrol/HDD/Datasets/FOR/voxel_timeseries",
    )
    parser.add_argument(
        "--for-atlas-root",
        default="/media/psycontrol/HDD/Datasets/FOR/subjects_400_rest",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset == "nsd":
        try:
            subjects = (
                [int(value) for value in args.subjects]
                if args.subjects is not None
                else list(TRAIN_NSD_SUBJECTS)
            )
        except ValueError:
            parser.error("NSD --subjects values must be integers.")
        build_nsd_contract(
            output_root=args.output_root,
            nsd_data_root=args.nsd_data_root,
            nsd_parcel_root=args.nsd_parcel_root,
        )
        for subject in subjects:
            build_nsd_subject(
                subject,
                output_root=args.output_root,
                nsd_data_root=args.nsd_data_root,
                nsd_parcel_root=args.nsd_parcel_root,
                force=args.force,
            )
            logger.info("Built NSD voxel contract bundle for subj%02d", subject)
        return 0

    _load_contract(args.output_root)
    subjects = args.subjects or _discover_for_subjects(args.for_voxel_root)
    failures: list[tuple[str, str]] = []
    for subject_label in subjects:
        try:
            build_for_subject(
                subject_label,
                output_root=args.output_root,
                for_voxel_root=args.for_voxel_root,
                for_atlas_root=args.for_atlas_root,
                force=args.force,
            )
            logger.info("Built FOR voxel contract bundle for %s", subject_label)
        except Exception as error:
            logger.error("Failed FOR voxel contract bundle for %s: %s", subject_label, error)
            failures.append((subject_label, str(error)))
    if failures:
        logger.error(
            "FOR voxel contract failures: %s",
            "; ".join(f"{subject}: {message}" for subject, message in failures),
        )
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
