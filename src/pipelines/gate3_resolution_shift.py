"""Gate 3 sensitivity bound for coarse-resolution zero-shot alignment."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.alignment.utils import compute_svd_basis, procrustes_align
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.evaluation.metrics import voxelwise_correlation
from src.models.encoding_factory import load_encoder
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_artifacts import (
    directory_file_fingerprints,
    file_sha256,
)
from src.pipelines.train_shared_space import _get_feature_matrix_and_slices


logger = logging.getLogger(__name__)

LOSO_SUBJECTS = (1, 2, 3, 4, 5, 6)
FINGERPRINT_VARIANTS = (
    "raw",
    "column_normalized",
    "volume_weighted",
)


def _parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--folds must be comma-separated integers.") from exc
    if not folds:
        raise argparse.ArgumentTypeError("--folds must contain at least one subject id.")
    invalid = [fold for fold in folds if fold not in LOSO_SUBJECTS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"--folds values must be within 1-6; got {invalid}."
        )
    return folds


def _rest_run_paths(subject_dir: str | Path) -> list[Path]:
    pattern = re.compile(r"rest_run([0-9]+)\.npy$")
    indexed_paths = []
    for path in Path(subject_dir).glob("rest_run*.npy"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            indexed_paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed_paths)]


def _match_standard_rest_runs(
    subject_id: int,
    *,
    voxel_subject_dir: str | Path,
    parcel_subject_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Match standard cleaned voxel and Schaefer-400 REST runs."""
    voxel_dir = Path(voxel_subject_dir)
    parcel_dir = Path(parcel_subject_dir)
    if not parcel_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Schaefer-400 parcel REST for subject {subject_id}: {parcel_dir}. Run "
            f"`python -m src.data.prepare_schaefer400_nsd --subjects {subject_id} --only rest`."
        )
    voxel_files = {path.name: path for path in _rest_run_paths(voxel_dir)}
    parcel_files = {path.name: path for path in _rest_run_paths(parcel_dir)}
    if set(voxel_files) != set(parcel_files):
        voxel_only = sorted(set(voxel_files) - set(parcel_files))
        parcel_only = sorted(set(parcel_files) - set(voxel_files))
        raise ValueError(
            f"Subject {subject_id}: voxel/parcel REST filename mismatch: "
            f"voxel_only={voxel_only}, parcel_only={parcel_only}."
        )
    if not voxel_files:
        raise ValueError(f"Subject {subject_id}: no matched REST runs were found.")

    pairs = []
    for voxel_path in _rest_run_paths(voxel_dir):
        parcel_path = parcel_files[voxel_path.name]
        voxel_shape = np.load(voxel_path, mmap_mode="r").shape
        parcel_shape = np.load(parcel_path, mmap_mode="r").shape
        if len(voxel_shape) != 2 or len(parcel_shape) != 2:
            raise ValueError(
                f"Subject {subject_id}: REST arrays must be 2D for {voxel_path.name}: "
                f"voxel={voxel_shape}, parcel={parcel_shape}."
            )
        if int(voxel_shape[0]) != int(parcel_shape[0]):
            raise ValueError(
                f"Subject {subject_id}: REST TR mismatch for {voxel_path.name}: "
                f"voxel={int(voxel_shape[0])}, parcel={int(parcel_shape[0])}."
            )
        if int(parcel_shape[1]) != 400:
            raise ValueError(
                f"Subject {subject_id}: Schaefer REST {parcel_path.name} has "
                f"{int(parcel_shape[1])} seed rows, expected 400."
            )
        pairs.append((voxel_path, parcel_path))
    return pairs


def _coarse_block_mapping(
    mask: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map C-ordered masked-vector target voxels to occupied 2x2x2 blocks."""
    mask = np.asarray(mask, dtype=bool)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if mask.ndim != 3:
        raise ValueError(f"NSD mask must be 3D, got {mask.shape}.")
    if (
        target_indices.ndim != 1
        or target_indices.size == 0
        or np.any(target_indices < 0)
        or not np.array_equal(target_indices, np.unique(target_indices))
    ):
        raise ValueError("target_indices must be a non-empty sorted unique 1D array.")

    masked_coordinates = np.argwhere(mask)
    if int(target_indices[-1]) >= int(masked_coordinates.shape[0]):
        raise ValueError(
            f"Target voxel index {int(target_indices[-1])} exceeds the "
            f"C-ordered mask-vector size {int(masked_coordinates.shape[0])}."
        )
    target_blocks = masked_coordinates[target_indices] // 2
    block_coordinates, inverse = np.unique(
        target_blocks,
        axis=0,
        return_inverse=True,
    )
    counts = np.bincount(inverse, minlength=block_coordinates.shape[0]).astype(
        np.int64
    )
    if np.any(counts < 1):
        raise ValueError("Coarse mapping unexpectedly contains an empty block.")
    return (
        block_coordinates.astype(np.int64),
        inverse.astype(np.int64),
        counts,
    )


def _block_average(
    values: np.ndarray,
    block_inverse: np.ndarray,
    block_counts: np.ndarray,
) -> np.ndarray:
    """Average a row-by-target-voxel matrix within occupied coarse blocks."""
    values = np.asarray(values)
    block_inverse = np.asarray(block_inverse, dtype=np.int64)
    block_counts = np.asarray(block_counts, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"Block-average input must be 2D, got {values.shape}.")
    if block_inverse.shape != (values.shape[1],):
        raise ValueError(
            "Block mapping length does not match the target-voxel dimension: "
            f"{block_inverse.shape} vs {values.shape}."
        )
    if block_counts.ndim != 1 or block_counts.size == 0 or np.any(block_counts < 1):
        raise ValueError("block_counts must contain one positive count per block.")
    if int(block_inverse.min()) < 0 or int(block_inverse.max()) >= int(block_counts.size):
        raise ValueError("Block mapping contains an out-of-range block id.")

    sums = np.zeros((block_counts.size, values.shape[0]), dtype=np.float64)
    np.add.at(sums, block_inverse, values.T)
    return (sums / block_counts[:, None]).T.astype(np.float32)


def _parcel_row_weights(
    target_parcel_ids: np.ndarray,
    *,
    n_rows: int = 400,
) -> np.ndarray:
    """Return held-out parcel weights, leaving absent parcel rows at one."""
    parcel_ids = np.asarray(target_parcel_ids, dtype=np.int64)
    if parcel_ids.ndim != 1 or parcel_ids.size == 0:
        raise ValueError("target_parcel_ids must be a non-empty 1D array.")
    if np.any(parcel_ids < 1) or np.any(parcel_ids > n_rows):
        raise ValueError(f"target_parcel_ids must be within 1-{int(n_rows)}.")
    counts = np.bincount(parcel_ids, minlength=n_rows + 1)[1 : n_rows + 1]
    weights = np.ones(n_rows, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present].astype(np.float64))
    return weights


def _column_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot normalize a zero-norm fingerprint column.")
    return values / norms


def transform_fingerprints(
    fingerprint: np.ndarray,
    template: np.ndarray,
    *,
    variant: str,
    target_parcel_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one matching transform identically to subject and template spaces."""
    fingerprint = np.asarray(fingerprint, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if fingerprint.shape != template.shape:
        raise ValueError(
            f"Fingerprint shape mismatch: fingerprint={fingerprint.shape}, "
            f"template={template.shape}."
        )
    if fingerprint.ndim != 2:
        raise ValueError(f"Fingerprints must be 2D, got {fingerprint.shape}.")
    if variant == "raw":
        return fingerprint.copy(), template.copy()
    if variant == "column_normalized":
        return _column_normalize(fingerprint), _column_normalize(template)
    if variant == "volume_weighted":
        weights = _parcel_row_weights(
            target_parcel_ids,
            n_rows=int(fingerprint.shape[0]),
        )
        return fingerprint * weights[:, None], template * weights[:, None]
    raise ValueError(f"Unknown fingerprint variant: {variant!r}.")


def _align_fingerprint_variant(
    fingerprint: np.ndarray,
    template: np.ndarray,
    *,
    variant: str,
    target_parcel_ids: np.ndarray,
) -> tuple[np.ndarray, float]:
    transformed_fingerprint, transformed_template = transform_fingerprints(
        fingerprint,
        template,
        variant=variant,
        target_parcel_ids=target_parcel_ids,
    )
    rotation = procrustes_align(transformed_fingerprint, transformed_template)
    orthogonality_error = np.linalg.norm(
        rotation @ rotation.T - np.eye(rotation.shape[0])
    )
    if orthogonality_error > 1e-4:
        logger.warning(
            "Variant %s rotation orthogonality error: %.6f",
            variant,
            orthogonality_error,
        )
    template_norm = float(np.linalg.norm(transformed_template))
    if template_norm <= 0.0:
        raise ValueError(f"Variant {variant}: transformed template has zero norm.")
    residual = float(
        np.linalg.norm(
            transformed_fingerprint @ rotation - transformed_template
        )
        / template_norm
    )
    return rotation, residual


def _zero_variance_count(values: np.ndarray) -> int:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Zero-variance input must be 2D, got {values.shape}.")
    return int(np.count_nonzero(values.std(axis=0) < 1e-10))


def resolution_delta(arm_b_median_r: float, arm_a_median_r: float) -> float:
    return float(arm_b_median_r) - float(arm_a_median_r)


def resolution_decision_fields(
    deltas_by_variant: Mapping[str, Sequence[float]],
) -> dict:
    if set(deltas_by_variant) != set(FINGERPRINT_VARIANTS):
        raise ValueError(
            f"Deltas must contain exactly the variants {list(FINGERPRINT_VARIANTS)}."
        )
    mean_delta_per_variant = {}
    for variant in FINGERPRINT_VARIANTS:
        deltas = list(deltas_by_variant[variant])
        if not deltas:
            raise ValueError(f"Variant {variant} requires at least one fold delta.")
        mean_delta_per_variant[variant] = float(
            np.mean(np.asarray(deltas, dtype=np.float64))
        )
    selected_variant = max(
        FINGERPRINT_VARIANTS,
        key=mean_delta_per_variant.__getitem__,
    )
    best_mean_delta = mean_delta_per_variant[selected_variant]
    return {
        "mean_delta_per_variant": mean_delta_per_variant,
        "selected_variant": selected_variant,
        "flag_first_order": bool(best_mean_delta < -0.01),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "sha256": file_sha256(path),
    }


def _gate2_command(
    *,
    subject_id: int,
    seed: int,
    config_path: str | Path,
    loso_root: str | Path,
) -> str:
    return (
        "python -m src.pipelines.train_voxel_contract --arm schaefer400 "
        f"--seed {int(seed)} --folds {int(subject_id)} --config {config_path} "
        f"--output-root {loso_root}"
    )


def _require_fold_model(
    *,
    fold_dir: Path,
    subject_id: int,
    seed: int,
    config_path: str | Path,
    loso_root: str | Path,
) -> tuple[Path, Path]:
    model_dir = fold_dir / "model"
    encoder_dir = model_dir / "encoder"
    result_path = fold_dir / "fold_result.json"
    required = [
        model_dir / "builder.npz",
        encoder_dir / "metadata.json",
        encoder_dir / "model.pt",
        encoder_dir / "feature_standardizer.npz",
        encoder_dir / "target_standardizer.npz",
        result_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        command = _gate2_command(
            subject_id=subject_id,
            seed=seed,
            config_path=config_path,
            loso_root=loso_root,
        )
        raise FileNotFoundError(
            f"Missing Gate-2 fold model for subject {subject_id}: {missing}. "
            f"Run `{command}`."
        )
    return model_dir, result_path


def _load_contract_arrays(
    *,
    subject_id: int,
    contract_root: str | Path,
    source_voxel_count: int,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    subject_dir = (
        Path(contract_root) / "nsd" / f"subj{int(subject_id):02d}"
    )
    target_path = subject_dir / "target_voxel_indices.npy"
    parcel_path = subject_dir / "target_parcel_ids.npy"
    missing = [str(path) for path in (target_path, parcel_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing voxel contract bundle for subject {subject_id}: {missing}."
        )
    target_indices = np.asarray(np.load(target_path), dtype=np.int64)
    target_parcel_ids = np.asarray(np.load(parcel_path), dtype=np.int64)
    if (
        target_indices.ndim != 1
        or target_indices.size == 0
        or np.any(target_indices < 0)
        or not np.array_equal(target_indices, np.unique(target_indices))
    ):
        raise ValueError(
            f"Invalid target_voxel_indices for subject {subject_id}: {target_path}"
        )
    if int(target_indices[-1]) >= int(source_voxel_count):
        raise ValueError(
            f"Subject {subject_id}: target voxel index {int(target_indices[-1])} "
            f"exceeds source voxel count {int(source_voxel_count)}."
        )
    if target_parcel_ids.shape != target_indices.shape:
        raise ValueError(
            f"Subject {subject_id}: target parcel/index shape mismatch: "
            f"{target_parcel_ids.shape} vs {target_indices.shape}."
        )
    if np.any(target_parcel_ids < 1) or np.any(target_parcel_ids > 400):
        raise ValueError(
            f"Subject {subject_id}: target_parcel_ids must be within 1-400."
        )
    return target_indices, target_parcel_ids, target_path, parcel_path


def _evaluate_fold(
    *,
    subject_id: int,
    seed: int,
    config: dict,
    config_path: str | Path,
    loso_root: str | Path,
) -> dict:
    fold_dir = (
        Path(loso_root)
        / "schaefer400"
        / f"seed{int(seed)}"
        / f"fold_sub{int(subject_id):02d}"
    )
    model_dir, result_path = _require_fold_model(
        fold_dir=fold_dir,
        subject_id=subject_id,
        seed=seed,
        config_path=config_path,
        loso_root=loso_root,
    )
    reference_result = json.loads(result_path.read_text())
    expected_identity = {
        "arm": "schaefer400",
        "seed": int(seed),
        "held_out": int(subject_id),
    }
    actual_identity = {
        key: reference_result.get(key) for key in expected_identity
    }
    if actual_identity != expected_identity:
        raise ValueError(
            f"Gate-2 fold identity mismatch in {result_path}: "
            f"stored={actual_identity}, expected={expected_identity}."
        )

    source = NSDSubjectData(int(subject_id), str(config["data_root"]))
    test_shape = source.test_fmri.shape
    if len(test_shape) != 2 or int(test_shape[0]) != int(source.test_stim_idx.shape[0]):
        raise ValueError(
            f"Subject {subject_id}: test fMRI/stimulus shapes are incompatible: "
            f"{test_shape} and {source.test_stim_idx.shape}."
        )
    target_indices, target_parcel_ids, target_path, parcel_path = (
        _load_contract_arrays(
            subject_id=subject_id,
            contract_root=config["voxel_contract"]["root"],
            source_voxel_count=int(test_shape[1]),
        )
    )
    if int(reference_result.get("n_target_voxels", -1)) != int(target_indices.size):
        raise ValueError(
            f"Gate-2 fold target count mismatch in {result_path}: "
            f"stored={reference_result.get('n_target_voxels')}, "
            f"current={int(target_indices.size)}."
        )

    mask_path = Path(config["data_root"]) / f"subj{int(subject_id):02d}" / "mask.npy"
    mask = np.asarray(source.mask, dtype=bool)
    if int(mask.sum()) != int(test_shape[1]):
        raise ValueError(
            f"Subject {subject_id}: C-ordered mask-vector size {int(mask.sum())} "
            f"does not match fMRI voxel count {int(test_shape[1])}."
        )
    block_coordinates, block_inverse, block_counts = _coarse_block_mapping(
        mask,
        target_indices,
    )

    tag = f"subj{int(subject_id):02d}"
    pairs = _match_standard_rest_runs(
        int(subject_id),
        voxel_subject_dir=Path(config["data_root"]) / tag,
        parcel_subject_dir=Path(config["voxel_contract"]["parcel_rest_root"]) / tag,
    )
    native_rest_runs = []
    coarse_rest_runs = []
    external_seed_runs = []
    for voxel_path, seed_path in pairs:
        voxel_run = np.load(voxel_path, mmap_mode="r")
        if int(voxel_run.shape[1]) != int(test_shape[1]):
            raise ValueError(
                f"Subject {subject_id}: voxel REST {voxel_path.name} has "
                f"{int(voxel_run.shape[1])} voxels, expected {int(test_shape[1])}."
            )
        native_run = np.asarray(voxel_run[:, target_indices], dtype=np.float32)
        native_rest_runs.append(native_run)
        coarse_rest_runs.append(
            _block_average(native_run, block_inverse, block_counts)
        )
        external_seed_runs.append(
            np.asarray(np.load(seed_path, mmap_mode="r"), dtype=np.float32)
        )

    builder = SharedSpaceBuilder.load(str(model_dir))
    encoder = load_encoder(str(model_dir))
    if builder.k_global is None:
        raise ValueError(f"Gate-2 builder has no k_global: {model_dir}")
    if int(encoder.output_dim) != int(builder.k_global):
        raise ValueError(
            f"Gate-2 encoder/builder latent mismatch: "
            f"{int(encoder.output_dim)} vs {int(builder.k_global)}."
        )
    if builder.template_fingerprint is None:
        raise ValueError(f"Gate-2 builder has no fingerprint template: {model_dir}")

    features = NSDFeatures(Path(config["data_root"]) / "features")
    feature_type = str(config["features"]["type"])
    feature_streams = [
        str(value) for value in config["features"].get("streams", [])
    ] or None
    X_test, feature_slices = _get_feature_matrix_and_slices(
        features=features,
        stim_idx=source.test_stim_idx,
        feature_type=feature_type,
        streams=feature_streams,
    )
    expected_slices = encoder.feature_slices or None
    if feature_slices != expected_slices:
        raise ValueError(
            "Held-out CLIP feature slices differ from the frozen Gate-2 encoder."
        )
    eval_indices = fixed_eval_indices(
        n_shared=len(source.test_stim_idx),
        eval_size=int(config["evaluation"]["fixed_eval_size"]),
        seed=int(config["evaluation"]["eval_split_seed"]),
    )
    if int(reference_result.get("n_eval", -1)) != int(eval_indices.size):
        raise ValueError(
            f"Gate-2 fold evaluation size mismatch in {result_path}: "
            f"stored={reference_result.get('n_eval')}, current={int(eval_indices.size)}."
        )

    native_truth = np.asarray(
        source.test_fmri[:, target_indices],
        dtype=np.float32,
    )
    coarse_truth = _block_average(native_truth, block_inverse, block_counts)
    truth_eval = coarse_truth[eval_indices]

    P_native, R_native = builder.align_new_subject_zeroshot(
        rest_runs=native_rest_runs,
        external_seed_runs=external_seed_runs,
    )
    native_predictions = encoder.predict_voxels(X_test, P_native, R_native)
    if native_predictions.shape != native_truth.shape:
        raise ValueError(
            f"Subject {subject_id}: native prediction/truth shape mismatch: "
            f"{native_predictions.shape} vs {native_truth.shape}."
        )
    coarse_arm_a_predictions = _block_average(
        native_predictions,
        block_inverse,
        block_counts,
    )
    arm_a_median_r = float(
        np.median(
            voxelwise_correlation(
                truth_eval,
                coarse_arm_a_predictions[eval_indices],
            )
        )
    )

    coarse_connectivity = compute_rest_connectivity(
        coarse_rest_runs,
        seed_runs=external_seed_runs,
        ensemble=str(config["alignment"]["ensemble_method"]),
    )
    max_rank = min(coarse_connectivity.shape) - 1
    if int(builder.k_global) > int(max_rank):
        raise ValueError(
            f"Subject {subject_id}: Gate-2 k_global={int(builder.k_global)} exceeds "
            f"the coarse connectivity rank limit {int(max_rank)} for shape "
            f"{coarse_connectivity.shape}."
        )
    P_coarse = compute_svd_basis(
        coarse_connectivity,
        n_components=int(builder.k_global),
        min_k=int(config["alignment"]["min_k"]),
    )
    if int(P_coarse.shape[1]) != int(builder.k_global):
        raise ValueError(
            f"Subject {subject_id}: coarse basis has {int(P_coarse.shape[1])} "
            f"components, expected k_global={int(builder.k_global)}."
        )
    fingerprint = coarse_connectivity @ P_coarse

    variants = {}
    for variant in FINGERPRINT_VARIANTS:
        rotation, residual = _align_fingerprint_variant(
            fingerprint,
            builder.template_fingerprint,
            variant=variant,
            target_parcel_ids=target_parcel_ids,
        )
        predictions = encoder.predict_voxels(X_test, P_coarse, rotation)
        if predictions.shape != coarse_truth.shape:
            raise ValueError(
                f"Subject {subject_id}: {variant} prediction/truth shape mismatch: "
                f"{predictions.shape} vs {coarse_truth.shape}."
            )
        arm_b_median_r = float(
            np.median(
                voxelwise_correlation(
                    truth_eval,
                    predictions[eval_indices],
                )
            )
        )
        variants[variant] = {
            "arm_b_median_r": arm_b_median_r,
            "fingerprint_residual": residual,
            "delta": resolution_delta(arm_b_median_r, arm_a_median_r),
            "n_zero_variance_prediction_blocks": _zero_variance_count(
                predictions[eval_indices]
            ),
        }

    return {
        "held_out": int(subject_id),
        "arm_a_median_r": arm_a_median_r,
        "variants": variants,
        "n_coarse_voxels": int(block_coordinates.shape[0]),
        "n_eval": int(eval_indices.size),
        "n_zero_variance_truth_blocks": _zero_variance_count(truth_eval),
        "arm_a_n_zero_variance_prediction_blocks": _zero_variance_count(
            coarse_arm_a_predictions[eval_indices]
        ),
        "fingerprints": {
            "fold_model_files": directory_file_fingerprints(model_dir),
            "fold_result": _file_fingerprint(result_path),
            "mask": _file_fingerprint(mask_path),
            "target_voxel_indices": _file_fingerprint(target_path),
            "target_parcel_ids": _file_fingerprint(parcel_path),
        },
    }


def evaluate_gate3(
    *,
    seed: int,
    folds: Sequence[int],
    config_path: str | Path,
    loso_root: str | Path,
    output_root: str | Path,
    force: bool,
) -> dict:
    """Evaluate frozen Gate-2 fold models at native and coarse resolution."""
    config_path = Path(config_path)
    summary_path = Path(output_root) / f"seed{int(seed)}" / "gate3_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(
            f"Gate-3 summary already exists: {summary_path}. Use --force to overwrite it."
        )
    config = load_config(config_path)
    fold_results = [
        _evaluate_fold(
            subject_id=int(subject_id),
            seed=int(seed),
            config=config,
            config_path=config_path,
            loso_root=loso_root,
        )
        for subject_id in folds
    ]
    decisions = resolution_decision_fields(
        {
            variant: [
                float(result["variants"][variant]["delta"])
                for result in fold_results
            ]
            for variant in FINGERPRINT_VARIANTS
        }
    )
    summary = {
        "gate": "gate3_resolution_shift",
        "gate_type": "sensitivity_bound",
        "seed": int(seed),
        "folds": [int(subject_id) for subject_id in folds],
        "per_fold": {
            f"subj{int(result['held_out']):02d}": result
            for result in sorted(fold_results, key=lambda row: int(row["held_out"]))
        },
        **decisions,
        "config": _file_fingerprint(config_path),
    }
    _write_json(summary_path, summary)
    print(
        "SENSITIVITY_BOUND "
        f"seed={int(seed)} selected_variant={summary['selected_variant']} "
        f"mean_delta={summary['mean_delta_per_variant'][summary['selected_variant']]:.10f} "
        f"flag_first_order={summary['flag_first_order']}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        choices=("evaluate",),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=_parse_folds, default="1,2,3,4,5,6")
    parser.add_argument("--config", default="config_voxel_contract.yaml")
    parser.add_argument("--loso-root", default="artifacts/voxel_contract_loso")
    parser.add_argument("--output-root", default="artifacts/gate3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    evaluate_gate3(
        seed=args.seed,
        folds=args.folds,
        config_path=args.config,
        loso_root=args.loso_root,
        output_root=args.output_root,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
