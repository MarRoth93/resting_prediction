"""Reconstruct the pinned NSD image set from common-parcel predictions.

The decoder is fitted on measured NSD subject-7 responses in the 72-parcel
voxel-contract space. Subject-7 zero-shot predictions and existing FOR voxel
predictions are reduced through the same parcel operator before reconstruction.
All generated artifacts are illustrative quality-control outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDSubjectData
from src.data.shared_paths import default_stimuli_hdf5
from src.models.encoding_factory import load_encoder
from src.pipelines.benchmark_reconstructions_vdvae_vd import (
    _column_moments_chunks,
    _decode_vdvae_latents,
    _decode_versatile,
    _load_vdvae_model,
    _load_versatile_components,
    _predict_clip_embeddings,
    _predict_vdvae_latents,
    _standardize_fmri,
)
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_artifacts import file_sha256
from src.pipelines.vdvae_calibration import (
    apply_calibration,
    learn_calibration,
    load_calibration,
    make_folds,
    save_calibration,
)
from src.pipelines.vdvae_calibration_battery import (
    _clip_embeddings,
    bootstrap_ci,
    pixcorr,
    two_way_identification,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config_voxel_contract.yaml"
DEFAULT_FINAL_MODEL_DIR = "artifacts/voxel_contract_final/seed42/model"
DEFAULT_FOR_INFERENCE_ROOT = "artifacts/for_inference/seed42"
DEFAULT_SELECTION_DIR = (
    "artifacts/schaefer400_multiexpert/prediction_inputs/"
    "random_unseen_500_seed42"
)
DEFAULT_CONTRACT_ROOT = "data/processed_voxel_contract"
DEFAULT_RECON_FEATURE_DIR = "data/processed/reconstruction_features/subj07"
DEFAULT_OUTPUT_ROOT = "artifacts/recon_from_predictions/seed42"
ALPHA_GRID = (1e2, 1e3, 1e4, 1e5, 1e6)
N_PARCELS = 72
N_PREDICTION_ROWS = 500
N_ALPHA_FOLDS = 3
SUBJECT_RE = re.compile(r"sub-[0-9]{4}")
DECODER_FAMILIES = ("vdvae", "cliptext", "clipvision")
SCORE_COLUMNS = (
    "subject",
    "image_stage",
    "n_images",
    "pixcorr_mean",
    "pixcorr_lo",
    "pixcorr_hi",
    "pixcorr_shuffled_mean",
    "clip_2way_id",
    "clip_2way_id_shuffled",
)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def _write_standardization_atomic(
    path: Path,
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    fmri_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        x_mean=np.asarray(x_mean, dtype=np.float32),
        x_std=np.asarray(x_std, dtype=np.float32),
        fmri_scale=np.asarray(float(fmri_scale), dtype=np.float64),
    )
    os.replace(temporary, path)


def _require_finite(subject: str, name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{subject}: {name} contains NaN/Inf.")


def _validate_parcel_columns(parcel_columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(parcel_columns, dtype=np.int64).ravel()
    if columns.shape != (N_PARCELS,):
        raise ValueError(
            f"Decoder parcel columns must have shape ({N_PARCELS},), got {columns.shape}."
        )
    if not np.array_equal(columns, np.unique(columns)):
        raise ValueError("Decoder parcel columns must be sorted and unique.")
    return columns


def _validate_subject_parcels(
    target_parcel_ids: np.ndarray,
    parcel_columns: np.ndarray,
    subject: str,
) -> np.ndarray:
    raw_ids = np.asarray(target_parcel_ids)
    if raw_ids.ndim != 1 or raw_ids.size == 0:
        raise ValueError(f"{subject}: target_parcel_ids must be a non-empty vector.")
    if not np.all(np.isfinite(raw_ids)):
        raise ValueError(f"{subject}: target_parcel_ids contains NaN/Inf.")
    ids = raw_ids.astype(np.int64)
    columns = _validate_parcel_columns(parcel_columns)
    observed = np.unique(ids)
    if not np.array_equal(observed, columns):
        missing = columns[~np.isin(columns, observed)].tolist()
        extra = observed[~np.isin(observed, columns)].tolist()
        raise ValueError(
            f"{subject}: parcel set does not match decoder columns; "
            f"missing={missing}, extra={extra}."
        )
    return ids


def parcel_average_responses(
    responses: np.ndarray,
    target_parcel_ids: np.ndarray,
    parcel_columns: np.ndarray,
    *,
    subject: str,
) -> np.ndarray:
    """Average voxel responses within parcels in decoder-column order."""
    values = np.asarray(responses, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{subject}: responses must be 2D, got {values.shape}.")
    ids = _validate_subject_parcels(target_parcel_ids, parcel_columns, subject)
    if values.shape[1] != ids.size:
        raise ValueError(
            f"{subject}: response width {values.shape[1]} does not match "
            f"target_parcel_ids length {ids.size}."
        )
    _require_finite(subject, "responses", values)
    columns = _validate_parcel_columns(parcel_columns)
    averaged = np.empty((values.shape[0], columns.size), dtype=np.float32)
    for column, parcel_id in enumerate(columns.tolist()):
        averaged[:, column] = values[:, ids == parcel_id].mean(axis=1)
    _require_finite(subject, "parcel-averaged responses", averaged)
    return averaged


def go_no_go_verdict(
    paired_id: float,
    shuffled_id: float,
    *,
    threshold: float = 0.02,
) -> dict[str, object]:
    difference = float(paired_id) - float(shuffled_id)
    detectable = bool(difference >= float(threshold))
    return {
        "paired_id": float(paired_id),
        "shuffled_id": float(shuffled_id),
        "paired_minus_shuffled": difference,
        "threshold": float(threshold),
        "detectable_signal": detectable,
        "warning": (
            None
            if detectable
            else "WARNING: decoder carries no detectable signal at 72 parcels"
        ),
    }


def _image_filename(row: int, stimulus_id: int) -> str:
    return f"row{int(row):03d}_stim{int(stimulus_id)}.png"


def missing_image_indices(
    image_dir: Path,
    stimulus_ids: np.ndarray,
    *,
    force: bool = False,
) -> np.ndarray:
    stimulus_ids = np.asarray(stimulus_ids, dtype=np.int64).ravel()
    if force:
        return np.arange(stimulus_ids.size, dtype=np.int64)
    return np.asarray(
        [
            row
            for row, stimulus_id in enumerate(stimulus_ids.tolist())
            if not (image_dir / _image_filename(row, stimulus_id)).is_file()
        ],
        dtype=np.int64,
    )


def _normalize_imported_decoder_names(
    image_dir: Path,
    stimulus_ids: np.ndarray,
) -> None:
    for row, stimulus_id in enumerate(np.asarray(stimulus_ids).tolist()):
        imported = image_dir / f"row{row:05d}_stim{int(stimulus_id)}.png"
        expected = image_dir / _image_filename(row, int(stimulus_id))
        if imported.is_file():
            os.replace(imported, expected)


def _load_parcel_columns(contract_root: Path) -> np.ndarray:
    contract = _read_json(contract_root / "contract.json")
    if "common_parcel_ids" not in contract:
        raise ValueError(
            f"Voxel contract has no common_parcel_ids: {contract_root / 'contract.json'}"
        )
    columns = np.sort(np.asarray(contract["common_parcel_ids"], dtype=np.int64))
    return _validate_parcel_columns(columns)


def _load_decoder_columns(decoder_dir: Path) -> np.ndarray:
    path = decoder_dir / "parcel_columns.json"
    try:
        columns = np.asarray(json.loads(path.read_text()), dtype=np.int64)
    except FileNotFoundError:
        raise FileNotFoundError(f"Decoder is missing parcel columns: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid decoder parcel columns: {path}") from exc
    return _validate_parcel_columns(columns)


def _selection_context(selection_dir: Path) -> dict:
    manifest_path = selection_dir / "selection_manifest.json"
    features_path = selection_dir / "clip_features.npy"
    stimulus_ids_path = selection_dir / "nsd_stimulus_ids.npy"
    image_rows_path = selection_dir / "image_rows.tsv"
    for path in (manifest_path, features_path, stimulus_ids_path, image_rows_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing selection file: {path}")

    manifest = _read_json(manifest_path)
    actual_hashes = {
        "features_sha256": file_sha256(features_path),
        "stimulus_ids_sha256": file_sha256(stimulus_ids_path),
        "image_rows_sha256": file_sha256(image_rows_path),
    }
    for key, actual in actual_hashes.items():
        if manifest.get(key) != actual:
            raise ValueError(f"Selection {key} does not match {manifest_path}.")

    clip_features = np.asarray(np.load(features_path), dtype=np.float32)
    stimulus_ids = np.asarray(np.load(stimulus_ids_path), dtype=np.int64)
    if clip_features.ndim != 2 or clip_features.shape[0] != N_PREDICTION_ROWS:
        raise ValueError(
            f"Expected clip_features.npy to have {N_PREDICTION_ROWS} rows, "
            f"got {clip_features.shape}."
        )
    if (
        stimulus_ids.shape != (N_PREDICTION_ROWS,)
        or np.unique(stimulus_ids).size != N_PREDICTION_ROWS
    ):
        raise ValueError(
            f"Expected exactly {N_PREDICTION_ROWS} unique NSD stimulus IDs."
        )
    if int(manifest.get("prediction_rows", -1)) != N_PREDICTION_ROWS:
        raise ValueError("Selection manifest prediction_rows is not 500.")
    if int(manifest.get("feature_width", -1)) != int(clip_features.shape[1]):
        raise ValueError("Selection manifest feature_width does not match CLIP features.")
    _require_finite("selection", "clip_features", clip_features)
    return {
        "clip_features": clip_features,
        "stimulus_ids": stimulus_ids,
        "hashes": {
            "selection_manifest_sha256": file_sha256(manifest_path),
            **actual_hashes,
        },
    }


def _subject_labels(
    for_inference_root: Path,
    requested: Sequence[str] | None,
) -> list[str]:
    if requested:
        labels = []
        for value in requested:
            labels.extend(part.strip() for part in str(value).split(",") if part.strip())
    else:
        for_labels = sorted(
            path.name
            for path in for_inference_root.glob("sub-*")
            if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
        )
        if len(for_labels) != 50:
            raise ValueError(
                f"Expected all 50 FOR subjects under {for_inference_root}, "
                f"found {len(for_labels)}."
            )
        labels = ["subj07", *for_labels]
    invalid = [
        label
        for label in labels
        if label != "subj07" and SUBJECT_RE.fullmatch(label) is None
    ]
    if invalid:
        raise ValueError(f"Invalid subject labels: {invalid}")
    if not labels:
        raise ValueError("At least one subject is required.")
    if len(set(labels)) != len(labels):
        raise ValueError("Subject labels must be unique.")
    return labels


def _rest_run_paths(subject_dir: Path) -> list[Path]:
    pattern = re.compile(r"rest_run([0-9]+)\.npy$")
    indexed = []
    for path in subject_dir.glob("rest_run*.npy"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed)]


def _match_rest_runs(
    *,
    voxel_subject_dir: Path,
    parcel_subject_dir: Path,
) -> list[tuple[Path, Path]]:
    voxel_files = {path.name: path for path in _rest_run_paths(voxel_subject_dir)}
    parcel_files = {path.name: path for path in _rest_run_paths(parcel_subject_dir)}
    if set(voxel_files) != set(parcel_files):
        voxel_only = sorted(set(voxel_files) - set(parcel_files))
        parcel_only = sorted(set(parcel_files) - set(voxel_files))
        raise ValueError(
            "subj07: voxel/parcel REST filename mismatch: "
            f"voxel_only={voxel_only}, parcel_only={parcel_only}."
        )
    if not voxel_files:
        raise ValueError("subj07: no matched REST runs were found.")
    pairs = []
    for voxel_path in _rest_run_paths(voxel_subject_dir):
        parcel_path = parcel_files[voxel_path.name]
        voxel_shape = np.load(voxel_path, mmap_mode="r").shape
        parcel_shape = np.load(parcel_path, mmap_mode="r").shape
        if len(voxel_shape) != 2 or len(parcel_shape) != 2:
            raise ValueError(
                f"subj07: REST arrays must be 2D for {voxel_path.name}: "
                f"voxel={voxel_shape}, parcel={parcel_shape}."
            )
        if int(voxel_shape[0]) != int(parcel_shape[0]):
            raise ValueError(
                f"subj07: REST TR mismatch for {voxel_path.name}: "
                f"voxel={voxel_shape[0]}, parcel={parcel_shape[0]}."
            )
        if int(parcel_shape[1]) != 400:
            raise ValueError(
                f"subj07: parcel REST width must be 400 for {parcel_path}, "
                f"got {parcel_shape[1]}."
            )
        pairs.append((voxel_path, parcel_path))
    return pairs


def _load_subject7_contract(
    contract_root: Path,
    source_voxel_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    subject_dir = contract_root / "nsd" / "subj07"
    target_indices_path = subject_dir / "target_voxel_indices.npy"
    target_parcels_path = subject_dir / "target_parcel_ids.npy"
    for path in (target_indices_path, target_parcels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing subject-7 voxel-contract file: {path}")
    target_indices = np.asarray(np.load(target_indices_path), dtype=np.int64)
    target_parcels = np.asarray(np.load(target_parcels_path), dtype=np.int64)
    if (
        target_indices.ndim != 1
        or target_indices.size == 0
        or not np.array_equal(target_indices, np.unique(target_indices))
        or np.any(target_indices < 0)
        or int(target_indices[-1]) >= int(source_voxel_count)
    ):
        raise ValueError(f"subj07: invalid target voxel indices: {target_indices_path}")
    if target_parcels.shape != target_indices.shape:
        raise ValueError(
            f"subj07: target parcel/index shape mismatch: "
            f"{target_parcels.shape} vs {target_indices.shape}."
        )
    return target_indices, target_parcels


def _aligned_training_arrays(
    x_train: np.ndarray,
    subject_stimulus_ids: np.ndarray,
    targets: np.ndarray,
    target_stimulus_ids: np.ndarray,
    *,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | str]]:
    subject_ids = np.asarray(subject_stimulus_ids, dtype=np.int64).ravel()
    target_ids = np.asarray(target_stimulus_ids, dtype=np.int64).ravel()
    if subject_ids.size != x_train.shape[0] or np.unique(subject_ids).size != subject_ids.size:
        raise ValueError("subj07: train stimulus IDs are invalid or misaligned.")
    if target_ids.size != targets.shape[0] or np.unique(target_ids).size != target_ids.size:
        raise ValueError(f"subj07: {family} target stimulus IDs are invalid or misaligned.")
    subject_row = {int(stimulus): row for row, stimulus in enumerate(subject_ids.tolist())}
    source_rows = np.asarray(
        [subject_row.get(int(stimulus), -1) for stimulus in target_ids],
        dtype=np.int64,
    )
    keep = source_rows >= 0
    if not np.any(keep):
        raise ValueError(f"subj07: no {family} targets align to measured training rows.")
    target_rows = np.flatnonzero(keep)
    source_rows = source_rows[keep]
    aligned_ids = target_ids[keep]
    aligned_x = x_train if np.array_equal(source_rows, np.arange(x_train.shape[0])) else x_train[source_rows]
    aligned_targets = (
        targets
        if np.array_equal(target_rows, np.arange(targets.shape[0]))
        else targets[target_rows]
    )
    _require_finite("subj07", f"{family} training targets", aligned_targets)
    return (
        aligned_x,
        aligned_targets,
        aligned_ids,
        {
            "mode": "stim_index",
            "rows_requested": int(target_ids.size),
            "rows_used": int(aligned_ids.size),
        },
    )


def _stimulus_grouped_folds(
    stimulus_ids: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(stimulus_ids, dtype=np.int64).ravel()
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    group_folds = make_folds(len(unique_ids), n_folds=n_folds, seed=seed)
    return group_folds[inverse]


def _squared_error_sum(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    row_chunk: int = 16,
) -> tuple[float, int]:
    if expected.shape != predicted.shape:
        raise ValueError(f"Prediction shape mismatch: {predicted.shape} vs {expected.shape}.")
    total = 0.0
    count = 0
    for start in range(0, expected.shape[0], row_chunk):
        stop = min(start + row_chunk, expected.shape[0])
        difference = np.asarray(predicted[start:stop], dtype=np.float64)
        difference -= np.asarray(expected[start:stop], dtype=np.float64)
        np.square(difference, out=difference)
        total += float(difference.sum())
        count += int(difference.size)
    return total, count


def _family_prediction(
    *,
    family: str,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    reconstruction_config: dict,
) -> np.ndarray:
    if family == "vdvae":
        return _predict_vdvae_latents(
            x_train=x_train,
            x_cond={"validation": x_validation},
            train_latents=targets,
            alpha=float(alpha),
            max_iter=int(reconstruction_config["ridge_max_iter"]),
            chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
        )["validation"]
    return _predict_clip_embeddings(
        x_train=x_train,
        x_cond={"validation": x_validation},
        train_clip=targets,
        alpha=float(alpha),
        max_iter=int(reconstruction_config["ridge_max_iter"]),
        label="CLIP-text" if family == "cliptext" else "CLIP-vision",
    )["validation"]


def _select_family_alpha(
    *,
    family: str,
    x_train: np.ndarray,
    targets: np.ndarray,
    stimulus_ids: np.ndarray,
    seed: int,
    reconstruction_config: dict,
) -> tuple[float, dict]:
    folds = _stimulus_grouped_folds(
        stimulus_ids,
        n_folds=N_ALPHA_FOLDS,
        seed=seed,
    )
    results = []
    for alpha in ALPHA_GRID:
        total_error = 0.0
        total_values = 0
        fold_mse = []
        for fold_index in range(N_ALPHA_FOLDS):
            training = folds != fold_index
            validation = folds == fold_index
            prediction = _family_prediction(
                family=family,
                x_train=x_train[training],
                x_validation=x_train[validation],
                targets=targets[training],
                alpha=alpha,
                reconstruction_config=reconstruction_config,
            )
            _require_finite("subj07", f"{family} alpha-selection predictions", prediction)
            error, count = _squared_error_sum(targets[validation], prediction)
            fold_mse.append(error / count)
            total_error += error
            total_values += count
            del prediction
        results.append(
            {
                "alpha": float(alpha),
                "validation_mse": float(total_error / total_values),
                "fold_mse": [float(value) for value in fold_mse],
            }
        )
        logger.info(
            "%s alpha=%s validation MSE=%.10f",
            family,
            alpha,
            results[-1]["validation_mse"],
        )
    best = min(results, key=lambda row: row["validation_mse"])
    return float(best["alpha"]), {
        "folding": "stimulus_grouped",
        "n_folds": N_ALPHA_FOLDS,
        "seed": int(seed),
        "grid": [float(value) for value in ALPHA_GRID],
        "results": results,
        "chosen_alpha": float(best["alpha"]),
    }


def _fit_family_coefficients(
    *,
    family: str,
    x_train: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    max_iter: int,
    chunk_size: int,
    decoder_dir: Path,
) -> tuple[Path, Path]:
    flattened = targets.reshape(targets.shape[0], -1)
    coefficient_path = decoder_dir / f"{family}_coefficients.npy"
    intercept_path = decoder_dir / f"{family}_intercept.npy"
    temporary_coefficient = coefficient_path.with_name(f".{coefficient_path.name}.part")
    temporary_intercept = intercept_path.with_name(f".{intercept_path.name}.part")
    temporary_coefficient.unlink(missing_ok=True)
    temporary_intercept.unlink(missing_ok=True)
    coefficients = np.lib.format.open_memmap(
        temporary_coefficient,
        mode="w+",
        dtype=np.float32,
        shape=(x_train.shape[1], flattened.shape[1]),
    )
    intercept = np.lib.format.open_memmap(
        temporary_intercept,
        mode="w+",
        dtype=np.float32,
        shape=(flattened.shape[1],),
    )
    try:
        for start in range(0, flattened.shape[1], chunk_size):
            stop = min(start + chunk_size, flattened.shape[1])
            regressor = Ridge(
                alpha=float(alpha),
                max_iter=int(max_iter),
                fit_intercept=True,
            )
            regressor.fit(x_train, np.asarray(flattened[:, start:stop], dtype=np.float32))
            coefficients[:, start:stop] = np.asarray(regressor.coef_.T, dtype=np.float32)
            intercept[start:stop] = np.asarray(regressor.intercept_, dtype=np.float32)
            logger.info(
                "%s final ridge dimensions %d:%d/%d",
                family,
                start,
                stop,
                flattened.shape[1],
            )
        coefficients.flush()
        intercept.flush()
        del coefficients, intercept
        os.replace(temporary_coefficient, coefficient_path)
        os.replace(temporary_intercept, intercept_path)
    except Exception:
        temporary_coefficient.unlink(missing_ok=True)
        temporary_intercept.unlink(missing_ok=True)
        raise
    return coefficient_path, intercept_path


def _fit_vdvae_calibration(
    *,
    x_train: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    seed: int,
    reconstruction_config: dict,
    decoder_dir: Path,
) -> tuple[dict, dict]:
    folds = make_folds(
        n_rows=int(x_train.shape[0]),
        n_folds=N_ALPHA_FOLDS,
        seed=seed,
    )
    temporary = decoder_dir / ".vdvae_oof_predictions.npy.part"
    temporary.unlink(missing_ok=True)
    oof_predictions = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=targets.shape,
    )
    try:
        for fold_index in range(N_ALPHA_FOLDS):
            training = folds != fold_index
            validation = folds == fold_index
            prediction = _predict_vdvae_latents(
                x_train=x_train[training],
                x_cond={"oof": x_train[validation]},
                train_latents=targets[training],
                alpha=float(alpha),
                max_iter=int(reconstruction_config["ridge_max_iter"]),
                chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
            )["oof"]
            oof_predictions[validation] = prediction
            del prediction
        oof_predictions.flush()
        _require_finite("subj07", "VDVAE OOF predictions", oof_predictions)
        oof_mean, oof_std = _column_moments_chunks(oof_predictions, chunk_size=4096)
        target_mean, target_std = _column_moments_chunks(targets, chunk_size=4096)
        calibration = learn_calibration(
            oof_pred_mean=oof_mean,
            oof_pred_std=oof_std,
            target_mean=target_mean,
            target_std=target_std,
            gain_cap_multiple=10.0,
        )
    finally:
        del oof_predictions
        temporary.unlink(missing_ok=True)

    metadata = {
        "n_folds": N_ALPHA_FOLDS,
        "seed": int(seed),
        "alpha": float(alpha),
        "n_rows": int(x_train.shape[0]),
        "gain_cap_multiple": 10.0,
        "n_capped": int(calibration["n_capped"]),
    }
    calibration_path = decoder_dir / "vdvae_calibration.npz"
    temporary_calibration = decoder_dir / ".vdvae_calibration.part.npz"
    temporary_calibration.unlink(missing_ok=True)
    save_calibration(temporary_calibration, calibration, metadata)
    os.replace(temporary_calibration, calibration_path)
    return calibration, metadata


def _predict_from_saved_family(
    *,
    family: str,
    x: np.ndarray,
    decoder: dict,
    subject: str,
    chunk_size: int,
) -> np.ndarray:
    coefficients = decoder[family]["coefficients"]
    intercept = decoder[family]["intercept"]
    output_shape = tuple(decoder[family]["output_shape"])
    flattened_width = int(np.prod(output_shape))
    prediction = np.empty((x.shape[0], flattened_width), dtype=np.float32)
    for start in range(0, flattened_width, chunk_size):
        stop = min(start + chunk_size, flattened_width)
        prediction[:, start:stop] = (
            x @ np.asarray(coefficients[:, start:stop], dtype=np.float32)
            + np.asarray(intercept[start:stop], dtype=np.float32)
        )
    prediction = prediction.reshape((x.shape[0], *output_shape))
    _require_finite(subject, f"predicted {family}", prediction)
    return prediction


def _load_decoder(
    decoder_dir: Path,
    *,
    manifest: dict | None = None,
    require_validation: bool = True,
) -> dict:
    manifest_path = decoder_dir / "decoder_manifest.json"
    if manifest is None:
        manifest = _read_json(manifest_path)
    columns = _load_decoder_columns(decoder_dir)
    standardization_path = decoder_dir / "standardization.npz"
    try:
        with np.load(standardization_path) as standardization:
            x_mean = np.asarray(standardization["x_mean"], dtype=np.float32)
            x_std = np.asarray(standardization["x_std"], dtype=np.float32)
            fmri_scale = float(standardization["fmri_scale"].item())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Decoder is missing standardization stats: {standardization_path}"
        ) from None
    if x_mean.shape != (1, N_PARCELS) or x_std.shape != (1, N_PARCELS):
        raise ValueError("Decoder standardization arrays have invalid shapes.")
    _require_finite("decoder", "standardization mean", x_mean)
    _require_finite("decoder", "standardization std", x_std)
    if np.any(x_std <= 0) or not np.isfinite(fmri_scale) or fmri_scale <= 0:
        raise ValueError("Decoder standardization values are invalid.")

    decoder = {
        "manifest": manifest,
        "parcel_columns": columns,
        "x_mean": x_mean,
        "x_std": x_std,
        "fmri_scale": fmri_scale,
    }
    family_metadata = manifest.get("families")
    if not isinstance(family_metadata, dict):
        raise ValueError(f"Decoder manifest has no family metadata: {manifest_path}")
    for family in DECODER_FAMILIES:
        metadata = family_metadata.get(family)
        if not isinstance(metadata, dict):
            raise ValueError(f"Decoder manifest has no {family} metadata.")
        output_shape = tuple(int(value) for value in metadata["output_shape"])
        flattened_width = int(np.prod(output_shape))
        coefficient_path = decoder_dir / f"{family}_coefficients.npy"
        intercept_path = decoder_dir / f"{family}_intercept.npy"
        coefficients = np.load(coefficient_path, mmap_mode="r")
        intercept = np.load(intercept_path, mmap_mode="r")
        if coefficients.shape != (N_PARCELS, flattened_width):
            raise ValueError(f"Decoder {family} coefficients have shape {coefficients.shape}.")
        if intercept.shape != (flattened_width,):
            raise ValueError(f"Decoder {family} intercept has shape {intercept.shape}.")
        _require_finite("decoder", f"{family} coefficients", coefficients)
        _require_finite("decoder", f"{family} intercept", intercept)
        decoder[family] = {
            "coefficients": coefficients,
            "intercept": intercept,
            "output_shape": output_shape,
            "alpha": float(metadata["alpha"]),
        }
    calibration, calibration_metadata = load_calibration(
        decoder_dir / "vdvae_calibration.npz"
    )
    for name in ("offset_in", "gain", "offset_out"):
        values = np.asarray(calibration[name])
        if values.shape != decoder["vdvae"]["output_shape"]:
            raise ValueError(
                f"Decoder calibration {name} has shape {values.shape}, expected "
                f"{decoder['vdvae']['output_shape']}."
            )
        _require_finite("decoder", f"calibration {name}", values)
    decoder["calibration"] = calibration
    decoder["calibration_metadata"] = calibration_metadata
    if require_validation and not (decoder_dir / "validation.json").is_file():
        raise FileNotFoundError(
            f"Decoder has no completed go/no-go validation: {decoder_dir / 'validation.json'}"
        )
    return decoder


def _standardize_with_decoder(
    parcel_responses: np.ndarray,
    decoder: dict,
    *,
    subject: str,
) -> np.ndarray:
    values = np.asarray(parcel_responses, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != N_PARCELS:
        raise ValueError(
            f"{subject}: expected parcel responses shaped (N, {N_PARCELS}), "
            f"got {values.shape}."
        )
    _require_finite(subject, "parcel responses", values)
    # Predicted parcel responses are structurally over-dispersed for subjects
    # with few target voxels (the encoder spreads fixed latent energy over V_t
    # voxels, so per-voxel amplitude scales ~1/sqrt(V_t); FOR V_t ~1,900 vs
    # subj07 12,672 gave ~4x hotter inputs and saturated the decoder to
    # constant images). Fix: standardize each subject's parcel matrix by ITS
    # OWN column moments (500 unsupervised rows, label-free, deterministic),
    # which places every subject's inputs in the decoder's standardized-train
    # distribution (the ridge was fit on unit-variance columns). This unified
    # path also serves the fit-time go/no-go validation on measured inputs,
    # where own-moment and train-stats standardization nearly coincide.
    column_mean = values.mean(axis=0, keepdims=True)
    column_std = values.std(axis=0, keepdims=True)
    if np.any(column_std < 1e-8):
        raise ValueError(f"{subject}: zero-variance parcel column in predicted responses.")
    standardized = np.asarray((values - column_mean) / column_std, dtype=np.float32)
    _require_finite(subject, "standardized parcel responses", standardized)
    return standardized


def _load_original_images(stimulus_ids: np.ndarray) -> list[Image.Image]:
    stimuli_path = Path(default_stimuli_hdf5())
    if not stimuli_path.is_file():
        raise FileNotFoundError(f"Missing NSD stimuli HDF5: {stimuli_path}")
    with h5py.File(stimuli_path, "r") as stimuli_file:
        images = stimuli_file["imgBrick"]
        return [
            Image.fromarray(images[int(stimulus_id)]).convert("RGB")
            for stimulus_id in np.asarray(stimulus_ids).tolist()
        ]


def _load_clip_model(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L/14",
        pretrained="openai",
    )
    return model.to(device).eval(), preprocess


def _print_validation_verdict(validation: dict) -> None:
    line = (
        "VERDICT "
        f"paired ID={float(validation['paired_id']):.6f} "
        f"shuffled ID={float(validation['shuffled_id']):.6f}"
    )
    if validation.get("warning"):
        line += f" {validation['warning']}"
    print(line)


def _run_decoder_validation(
    *,
    decoder_dir: Path,
    decoder: dict,
    measured_test_parcels: np.ndarray,
    test_stimulus_ids: np.ndarray,
    evaluation_config: dict,
    reconstruction_config: dict,
    recon_feature_dir: Path,
    recon_model_root: Path,
    device: str,
) -> dict:
    eval_indices = fixed_eval_indices(
        n_shared=int(measured_test_parcels.shape[0]),
        eval_size=int(evaluation_config["fixed_eval_size"]),
        seed=int(evaluation_config["eval_split_seed"]),
    )
    stimulus_ids = np.asarray(test_stimulus_ids[eval_indices], dtype=np.int64)
    x_eval = _standardize_with_decoder(
        measured_test_parcels[eval_indices],
        decoder,
        subject="subj07",
    )
    predicted = _predict_from_saved_family(
        family="vdvae",
        x=x_eval,
        decoder=decoder,
        subject="subj07",
        chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
    )
    predicted = apply_calibration(predicted, decoder["calibration"])
    _require_finite("subj07", "calibrated validation VDVAE predictions", predicted)

    ref_path = recon_feature_dir / "ref_latents.npz"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing VDVAE reference latents: {ref_path}")
    ref_latent = np.load(ref_path, allow_pickle=True)["ref_latent"]
    validation_image_dir = decoder_dir / "validation_images_vdvae"
    ema_vae = _load_vdvae_model(recon_model_root)
    _decode_vdvae_latents(
        ema_vae=ema_vae,
        pred_latents=predicted,
        ref_latent=ref_latent,
        out_dir=validation_image_dir,
        save_rows=eval_indices,
        save_stim=stimulus_ids,
        batch_size=int(reconstruction_config["vdvae_batch_size"]),
        device=device,
    )
    del ema_vae

    decoded_images = []
    for row, stimulus_id in zip(eval_indices.tolist(), stimulus_ids.tolist()):
        path = validation_image_dir / f"row{row:05d}_stim{stimulus_id}.png"
        with Image.open(path) as image:
            decoded_images.append(image.convert("RGB"))
    originals = _load_original_images(stimulus_ids)
    clip_model, clip_preprocess = _load_clip_model(device)
    decoded_embeddings = _clip_embeddings(
        decoded_images,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        batch_size=int(reconstruction_config["vdvae_batch_size"]),
        device=device,
    )
    original_embeddings = _clip_embeddings(
        originals,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        batch_size=int(reconstruction_config["vdvae_batch_size"]),
        device=device,
    )
    _require_finite("subj07", "validation decoded CLIP embeddings", decoded_embeddings)
    _require_finite("subj07", "validation original CLIP embeddings", original_embeddings)
    similarities = decoded_embeddings @ original_embeddings.T
    _require_finite("subj07", "validation CLIP similarities", similarities)
    validation = go_no_go_verdict(
        two_way_identification(similarities),
        two_way_identification(np.roll(similarities, 1, axis=0)),
    )
    validation.update(
        {
            "subject": "subj07",
            "input": "measured_test_fmri_parcel_averaged",
            "image_stage": "vdvae",
            "n_images": int(eval_indices.size),
            "eval_indices": eval_indices.tolist(),
            "stimulus_ids": stimulus_ids.tolist(),
        }
    )
    _write_json_atomic(decoder_dir / "validation.json", validation)
    _print_validation_verdict(validation)
    return validation


def fit_decoder(
    *,
    output_root: Path,
    contract_root: Path,
    recon_feature_dir: Path,
    recon_model_root: Path,
    config: dict,
    device: str,
    force: bool,
) -> dict:
    decoder_dir = output_root / "decoder"
    manifest_path = decoder_dir / "decoder_manifest.json"
    if manifest_path.is_file() and not force:
        decoder = _load_decoder(decoder_dir)
        validation = _read_json(decoder_dir / "validation.json")
        _print_validation_verdict(validation)
        return decoder["manifest"]
    if decoder_dir.exists() and any(decoder_dir.iterdir()) and not force:
        raise RuntimeError(
            f"Partial decoder output exists without a complete manifest: {decoder_dir}. "
            "Use --force to re-run it."
        )
    if force:
        manifest_path.unlink(missing_ok=True)
        (decoder_dir / "validation.json").unlink(missing_ok=True)

    reconstruction_config = config["reconstruction"]
    evaluation_config = config["evaluation"]
    seed = int(config["random_seed"])
    source = NSDSubjectData(7, str(config["data_root"]))
    target_indices, target_parcel_ids = _load_subject7_contract(
        contract_root,
        source_voxel_count=int(source.train_fmri.shape[1]),
    )
    parcel_columns = _load_parcel_columns(contract_root)
    train_voxels = np.asarray(source.train_fmri[:, target_indices], dtype=np.float32)
    test_voxels = np.asarray(source.test_fmri[:, target_indices], dtype=np.float32)
    train_parcels = parcel_average_responses(
        train_voxels,
        target_parcel_ids,
        parcel_columns,
        subject="subj07",
    )
    test_parcels = parcel_average_responses(
        test_voxels,
        target_parcel_ids,
        parcel_columns,
        subject="subj07",
    )
    del train_voxels, test_voxels
    x_train, _, x_mean, x_std = _standardize_fmri(
        train_parcels,
        {},
        fmri_scale=float(reconstruction_config["fmri_scale"]),
    )
    decoder_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(decoder_dir / "parcel_columns.json", parcel_columns.tolist())
    _write_standardization_atomic(
        decoder_dir / "standardization.npz",
        x_mean=x_mean,
        x_std=x_std,
        fmri_scale=float(reconstruction_config["fmri_scale"]),
    )

    family_manifest = {}
    calibration_metadata = None
    for family in DECODER_FAMILIES:
        if family == "vdvae":
            feature_path = recon_feature_dir / "vdvae_features.npz"
            if not feature_path.is_file():
                raise FileNotFoundError(f"Missing VDVAE feature bundle: {feature_path}")
            with np.load(feature_path) as feature_bundle:
                targets = np.asarray(feature_bundle["train_latents"], dtype=np.float32)
                target_stimulus_ids = np.asarray(
                    feature_bundle["train_stim_idx"], dtype=np.int64
                )
        else:
            feature_path = recon_feature_dir / f"{family}_train.npy"
            stimulus_path = recon_feature_dir / f"{family}_train_stim_idx.npy"
            for path in (feature_path, stimulus_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing reconstruction feature file: {path}")
            targets = np.load(feature_path, mmap_mode="r")
            target_stimulus_ids = np.asarray(np.load(stimulus_path), dtype=np.int64)

        aligned_x, aligned_targets, aligned_ids, alignment = _aligned_training_arrays(
            x_train,
            source.train_stim_idx,
            targets,
            target_stimulus_ids,
            family=family,
        )
        chosen_alpha, selection = _select_family_alpha(
            family=family,
            x_train=aligned_x,
            targets=aligned_targets,
            stimulus_ids=aligned_ids,
            seed=seed,
            reconstruction_config=reconstruction_config,
        )
        coefficient_path, intercept_path = _fit_family_coefficients(
            family=family,
            x_train=aligned_x,
            targets=aligned_targets,
            alpha=chosen_alpha,
            max_iter=int(reconstruction_config["ridge_max_iter"]),
            chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
            decoder_dir=decoder_dir,
        )
        if family == "vdvae":
            _, calibration_metadata = _fit_vdvae_calibration(
                x_train=aligned_x,
                targets=aligned_targets,
                alpha=chosen_alpha,
                seed=seed,
                reconstruction_config=reconstruction_config,
                decoder_dir=decoder_dir,
            )
        family_manifest[family] = {
            "alpha": float(chosen_alpha),
            "alpha_selection": selection,
            "alignment": alignment,
            "output_shape": [int(value) for value in aligned_targets.shape[1:]],
            "coefficient_sha256": file_sha256(coefficient_path),
            "intercept_sha256": file_sha256(intercept_path),
            "target_file": str(feature_path.resolve()),
            "target_file_sha256": file_sha256(feature_path),
        }
        del targets, aligned_targets

    provisional_manifest = {
        "artifact_version": 1,
        "subject": "subj07",
        "representation": "72_parcel_means_in_sorted_common_parcel_order",
        "training_rows": int(x_train.shape[0]),
        "input_parcels": N_PARCELS,
        "families": family_manifest,
        "calibration": calibration_metadata,
        "config": {
            "evaluation": evaluation_config,
            "reconstruction": reconstruction_config,
        },
    }
    decoder = _load_decoder(
        decoder_dir,
        manifest=provisional_manifest,
        require_validation=False,
    )
    _run_decoder_validation(
        decoder_dir=decoder_dir,
        decoder=decoder,
        measured_test_parcels=test_parcels,
        test_stimulus_ids=source.test_stim_idx,
        evaluation_config=evaluation_config,
        reconstruction_config=reconstruction_config,
        recon_feature_dir=recon_feature_dir,
        recon_model_root=recon_model_root,
        device=device,
    )
    _write_json_atomic(manifest_path, provisional_manifest)
    return provisional_manifest


def _completed_parcel_output(
    subject_dir: Path,
    *,
    subject: str,
    parcel_columns: np.ndarray,
) -> bool:
    response_path = subject_dir / "parcel_responses.npy"
    provenance_path = subject_dir / "parcel_provenance.json"
    if not response_path.is_file() or not provenance_path.is_file():
        return False
    responses = np.load(response_path, mmap_mode="r")
    if responses.shape != (N_PREDICTION_ROWS, N_PARCELS):
        return False
    _require_finite(subject, "parcel responses", responses)
    provenance = _read_json(provenance_path)
    return provenance.get("parcel_columns") == parcel_columns.tolist()


def predict_parcels(
    *,
    subjects: Sequence[str],
    output_root: Path,
    final_model_dir: Path,
    for_inference_root: Path,
    selection_dir: Path,
    contract_root: Path,
    config: dict,
    device: str,
    force: bool,
) -> dict[str, str]:
    decoder_columns = _load_decoder_columns(output_root / "decoder")
    contract_columns = _load_parcel_columns(contract_root)
    if not np.array_equal(decoder_columns, contract_columns):
        raise ValueError("Current voxel-contract parcels do not match decoder columns.")
    selection = _selection_context(selection_dir)
    results = {}

    builder = None
    encoder = None
    for subject in subjects:
        subject_output = output_root / subject
        if not force and _completed_parcel_output(
            subject_output,
            subject=subject,
            parcel_columns=decoder_columns,
        ):
            logger.info("Skipping completed parcel responses for %s", subject)
            results[subject] = "skipped"
            continue

        if subject == "subj07":
            source = NSDSubjectData(7, str(config["data_root"]))
            target_indices, target_parcel_ids = _load_subject7_contract(
                contract_root,
                source_voxel_count=int(source.train_fmri.shape[1]),
            )
            pairs = _match_rest_runs(
                voxel_subject_dir=Path(config["data_root"]) / "subj07",
                parcel_subject_dir=Path(config["voxel_contract"]["parcel_rest_root"])
                / "subj07",
            )
            rest_runs = []
            seed_runs = []
            for voxel_path, parcel_path in pairs:
                voxel_run = np.load(voxel_path, mmap_mode="r")
                if int(voxel_run.shape[1]) != int(source.train_fmri.shape[1]):
                    raise ValueError(
                        f"subj07: REST voxel width mismatch for {voxel_path}: "
                        f"{voxel_run.shape[1]} vs {source.train_fmri.shape[1]}."
                    )
                sliced = np.asarray(voxel_run[:, target_indices], dtype=np.float32)
                seeds = np.asarray(np.load(parcel_path, mmap_mode="r"), dtype=np.float32)
                _require_finite(subject, f"REST {voxel_path.name}", sliced)
                _require_finite(subject, f"parcel REST {parcel_path.name}", seeds)
                rest_runs.append(sliced)
                seed_runs.append(seeds)
            if builder is None:
                builder = SharedSpaceBuilder.load(str(final_model_dir))
                encoder = load_encoder(str(final_model_dir))
                encoder.config.device = str(device)
            if int(selection["clip_features"].shape[1]) != int(encoder.input_dim):
                raise ValueError(
                    f"Selection feature width {selection['clip_features'].shape[1]} "
                    f"does not match encoder input_dim {encoder.input_dim}."
                )
            P, R = builder.align_new_subject_zeroshot(
                rest_runs=rest_runs,
                external_seed_runs=seed_runs,
            )
            _require_finite(subject, "alignment P", P)
            _require_finite(subject, "alignment R", R)
            predicted = np.asarray(
                encoder.predict_voxels(selection["clip_features"], P, R),
                dtype=np.float32,
            )
            expected_shape = (N_PREDICTION_ROWS, target_indices.size)
            if predicted.shape != expected_shape:
                raise ValueError(
                    f"subj07: predicted responses have shape {predicted.shape}, "
                    f"expected {expected_shape}."
                )
            _require_finite(subject, "predicted responses", predicted)
            parcels = parcel_average_responses(
                predicted,
                target_parcel_ids,
                decoder_columns,
                subject=subject,
            )
            provenance = {
                "subject": subject,
                "source": "final_model_zero_shot_prediction",
                "subject7_use": "third_use_illustrative_only",
                "final_model_dir": str(final_model_dir),
                "selection_sha256s": selection["hashes"],
                "parcel_columns": decoder_columns.tolist(),
            }
        else:
            source_dir = for_inference_root / subject
            response_path = source_dir / "predicted_responses.npy"
            parcel_path = source_dir / "target_parcel_ids.npy"
            source_provenance_path = source_dir / "provenance.json"
            for path in (response_path, parcel_path, source_provenance_path):
                if not path.is_file():
                    raise FileNotFoundError(f"{subject}: missing FOR inference file: {path}")
            source_provenance = _read_json(source_provenance_path)
            source_selection = source_provenance.get("selection_sha256s", {})
            selection_matches = (
                source_selection.get("selection_manifest_sha256")
                == selection["hashes"]["selection_manifest_sha256"]
                and source_selection.get("clip_features_sha256")
                == selection["hashes"]["features_sha256"]
                and source_provenance.get("stimulus_ids_sha256")
                == selection["hashes"]["stimulus_ids_sha256"]
            )
            if not selection_matches:
                raise ValueError(
                    f"{subject}: FOR inference selection hashes do not match the "
                    "requested 500-image selection."
                )
            predicted = np.asarray(np.load(response_path), dtype=np.float32)
            target_parcel_ids = np.asarray(np.load(parcel_path))
            if predicted.shape[0] != N_PREDICTION_ROWS:
                raise ValueError(
                    f"{subject}: expected {N_PREDICTION_ROWS} predicted rows, "
                    f"got {predicted.shape}."
                )
            _require_finite(subject, "predicted responses", predicted)
            parcels = parcel_average_responses(
                predicted,
                target_parcel_ids,
                decoder_columns,
                subject=subject,
            )
            provenance = {
                "subject": subject,
                "source": "existing_for_inference_predictions",
                "predicted_responses_file": str(response_path.resolve()),
                "predicted_responses_sha256": file_sha256(response_path),
                "target_parcel_ids_file": str(parcel_path.resolve()),
                "target_parcel_ids_sha256": file_sha256(parcel_path),
                "source_provenance_sha256": file_sha256(source_provenance_path),
                "selection_sha256s": selection["hashes"],
                "parcel_columns": decoder_columns.tolist(),
            }

        _write_npy_atomic(subject_output / "parcel_responses.npy", parcels)
        _write_json_atomic(subject_output / "parcel_provenance.json", provenance)
        results[subject] = "completed"
    return results


def _decoder_ready_for_reconstruction(decoder_dir: Path) -> dict:
    decoder = _load_decoder(decoder_dir)
    validation = _read_json(decoder_dir / "validation.json")
    if "paired_id" not in validation or "shuffled_id" not in validation:
        raise ValueError(
            f"Decoder validation is incomplete: {decoder_dir / 'validation.json'}"
        )
    return decoder


def _prepare_vd_init_dir(
    subject_dir: Path,
    stimulus_ids: np.ndarray,
    rows: np.ndarray,
) -> Path:
    init_dir = subject_dir / ".vd_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    vdvae_dir = subject_dir / "images_vdvae"
    for row in rows.tolist():
        stimulus_id = int(stimulus_ids[row])
        source = vdvae_dir / _image_filename(row, stimulus_id)
        destination = init_dir / f"row{row:05d}_stim{stimulus_id}.png"
        destination.unlink(missing_ok=True)
        destination.symlink_to(source.resolve())
    return init_dir


def _clean_vd_init_dir(init_dir: Path, stimulus_ids: np.ndarray) -> None:
    for row, stimulus_id in enumerate(np.asarray(stimulus_ids).tolist()):
        (init_dir / f"row{row:05d}_stim{int(stimulus_id)}.png").unlink(
            missing_ok=True
        )
    try:
        init_dir.rmdir()
    except OSError:
        pass


def reconstruct_subjects(
    *,
    subjects: Sequence[str],
    output_root: Path,
    selection_dir: Path,
    recon_feature_dir: Path,
    recon_model_root: Path,
    config: dict,
    device: str,
    skip_vd: bool,
    force: bool,
) -> dict[str, str]:
    decoder = _decoder_ready_for_reconstruction(output_root / "decoder")
    selection = _selection_context(selection_dir)
    stimulus_ids = selection["stimulus_ids"]
    reconstruction_config = config["reconstruction"]
    ref_path = recon_feature_dir / "ref_latents.npz"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing VDVAE reference latents: {ref_path}")
    ref_latent = np.load(ref_path, allow_pickle=True)["ref_latent"]

    ema_vae = None
    versatile_components = None
    results = {}
    for subject in subjects:
        subject_dir = output_root / subject
        parcel_path = subject_dir / "parcel_responses.npy"
        if not parcel_path.is_file():
            raise FileNotFoundError(f"{subject}: missing parcel responses: {parcel_path}")
        parcel_responses = np.asarray(np.load(parcel_path), dtype=np.float32)
        if parcel_responses.shape != (N_PREDICTION_ROWS, N_PARCELS):
            raise ValueError(
                f"{subject}: parcel responses have shape {parcel_responses.shape}, "
                f"expected {(N_PREDICTION_ROWS, N_PARCELS)}."
            )
        _require_finite(subject, "parcel responses", parcel_responses)

        vdvae_dir = subject_dir / "images_vdvae"
        final_dir = subject_dir / "images_final"
        _normalize_imported_decoder_names(vdvae_dir, stimulus_ids)
        _normalize_imported_decoder_names(final_dir, stimulus_ids)
        final_missing = missing_image_indices(final_dir, stimulus_ids, force=force)
        if not skip_vd and final_missing.size == 0:
            logger.info("Skipping completed reconstruction subject %s", subject)
            results[subject] = "skipped"
            continue
        vdvae_missing = missing_image_indices(vdvae_dir, stimulus_ids, force=force)
        if skip_vd and vdvae_missing.size == 0:
            logger.info("Skipping completed VDVAE subject %s", subject)
            results[subject] = "skipped"
            continue

        standardized = _standardize_with_decoder(
            parcel_responses,
            decoder,
            subject=subject,
        )
        predicted_vdvae = _predict_from_saved_family(
            family="vdvae",
            x=standardized,
            decoder=decoder,
            subject=subject,
            chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
        )
        predicted_vdvae = apply_calibration(
            predicted_vdvae,
            decoder["calibration"],
        )
        _require_finite(subject, "calibrated predicted vdvae", predicted_vdvae)
        if vdvae_missing.size:
            if ema_vae is None:
                ema_vae = _load_vdvae_model(recon_model_root)
            _decode_vdvae_latents(
                ema_vae=ema_vae,
                pred_latents=predicted_vdvae[vdvae_missing],
                ref_latent=ref_latent,
                out_dir=vdvae_dir,
                save_rows=vdvae_missing,
                save_stim=stimulus_ids[vdvae_missing],
                batch_size=int(reconstruction_config["vdvae_batch_size"]),
                device=device,
            )
            _normalize_imported_decoder_names(vdvae_dir, stimulus_ids)
        remaining_vdvae = missing_image_indices(vdvae_dir, stimulus_ids)
        if remaining_vdvae.size:
            raise RuntimeError(
                f"{subject}: {remaining_vdvae.size} VDVAE images are still missing."
            )
        if skip_vd:
            results[subject] = "completed"
            continue

        final_missing = missing_image_indices(final_dir, stimulus_ids, force=force)
        if final_missing.size:
            predicted_cliptext = _predict_from_saved_family(
                family="cliptext",
                x=standardized,
                decoder=decoder,
                subject=subject,
                chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
            )
            predicted_clipvision = _predict_from_saved_family(
                family="clipvision",
                x=standardized,
                decoder=decoder,
                subject=subject,
                chunk_size=int(reconstruction_config["vdvae_chunk_size"]),
            )
            if versatile_components is None:
                weights_path = (
                    recon_model_root
                    / "versatile_diffusion"
                    / "pretrained"
                    / "vd-four-flow-v1-0-fp16-deprecated.pth"
                )
                versatile_components = _load_versatile_components(
                    recon_model_root=recon_model_root,
                    vd_weights_path=weights_path,
                    device=device,
                    precision=str(reconstruction_config["precision"]),
                )
            init_dir = _prepare_vd_init_dir(subject_dir, stimulus_ids, final_missing)
            try:
                net, sampler, utx, uim = versatile_components
                _decode_versatile(
                    net=net,
                    sampler=sampler,
                    utx=utx,
                    uim=uim,
                    pred_cliptext=predicted_cliptext[final_missing],
                    pred_clipvision=predicted_clipvision[final_missing],
                    init_dir=init_dir,
                    out_dir=final_dir,
                    save_rows=final_missing,
                    save_stim=stimulus_ids[final_missing],
                    device=device,
                    precision=str(reconstruction_config["precision"]),
                    strength=float(reconstruction_config["vd_strength"]),
                    mixing=float(reconstruction_config["vd_mixing"]),
                    guidance_scale=float(reconstruction_config["vd_guidance_scale"]),
                    ddim_steps=int(reconstruction_config["vd_ddim_steps"]),
                    ddim_eta=float(reconstruction_config["vd_ddim_eta"]),
                )
            finally:
                _clean_vd_init_dir(init_dir, stimulus_ids)
            _normalize_imported_decoder_names(final_dir, stimulus_ids)
        remaining_final = missing_image_indices(final_dir, stimulus_ids)
        if remaining_final.size:
            raise RuntimeError(
                f"{subject}: {remaining_final.size} final images are still missing."
            )
        results[subject] = "completed"
    return results


def _resized_pixels(images: list[Image.Image]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(image.resize((64, 64), resample=Image.Resampling.BICUBIC))
            for image in images
        ]
    )


def _write_qc_readme(output_root: Path) -> None:
    _write_json_atomic(
        output_root / "README.json",
        {
            "purpose": "QC only",
            "interpretation": "ILLUSTRATIVE reconstructions under decision D-02",
            "group_statistics": "not performed by this module",
            "clinical_labels_loaded": False,
            "comparison_target": "500 known NSD original images",
        },
    )


def score_subjects(
    *,
    subjects: Sequence[str],
    output_root: Path,
    selection_dir: Path,
    config: dict,
    device: str,
    skip_vd: bool,
) -> list[dict[str, object]]:
    selection = _selection_context(selection_dir)
    stimulus_ids = selection["stimulus_ids"]
    originals = _load_original_images(stimulus_ids)
    original_pixels = _resized_pixels(originals)
    clip_model, clip_preprocess = _load_clip_model(device)
    batch_size = int(config["reconstruction"]["vdvae_batch_size"])
    original_embeddings = _clip_embeddings(
        originals,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        batch_size=batch_size,
        device=device,
    )
    image_stage = "vdvae" if skip_vd else "final"
    directory_name = "images_vdvae" if skip_vd else "images_final"
    scores_path = output_root / "scores.csv"
    _write_qc_readme(output_root)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not scores_path.exists() or scores_path.stat().st_size == 0
    rows = []
    with open(scores_path, "a", newline="") as scores_file:
        writer = csv.DictWriter(scores_file, fieldnames=SCORE_COLUMNS)
        if write_header:
            writer.writeheader()
        for subject in subjects:
            image_dir = output_root / subject / directory_name
            missing = missing_image_indices(image_dir, stimulus_ids)
            if missing.size:
                raise RuntimeError(
                    f"{subject}: cannot score; {missing.size} {image_stage} images are missing."
                )
            decoded_images = []
            for row, stimulus_id in enumerate(stimulus_ids.tolist()):
                path = image_dir / _image_filename(row, stimulus_id)
                with Image.open(path) as image:
                    decoded_images.append(image.convert("RGB"))
            decoded_pixels = _resized_pixels(decoded_images)
            pixcorr_values = pixcorr(decoded_pixels, original_pixels)
            _require_finite(subject, "PixCorr values", pixcorr_values)
            pixcorr_lo, pixcorr_hi = bootstrap_ci(pixcorr_values)
            shuffled_pixcorr = pixcorr(
                decoded_pixels,
                np.roll(original_pixels, 1, axis=0),
            )
            _require_finite(subject, "shuffled PixCorr values", shuffled_pixcorr)
            decoded_embeddings = _clip_embeddings(
                decoded_images,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=batch_size,
                device=device,
            )
            _require_finite(subject, "decoded CLIP embeddings", decoded_embeddings)
            similarities = decoded_embeddings @ original_embeddings.T
            _require_finite(subject, "CLIP similarities", similarities)
            row = {
                "subject": subject,
                "image_stage": image_stage,
                "n_images": N_PREDICTION_ROWS,
                "pixcorr_mean": float(np.mean(pixcorr_values)),
                "pixcorr_lo": float(pixcorr_lo),
                "pixcorr_hi": float(pixcorr_hi),
                "pixcorr_shuffled_mean": float(np.mean(shuffled_pixcorr)),
                "clip_2way_id": two_way_identification(similarities),
                "clip_2way_id_shuffled": two_way_identification(
                    np.roll(similarities, 1, axis=0)
                ),
            }
            writer.writerow(row)
            scores_file.flush()
            rows.append(row)
            logger.info(
                "%s: PixCorr=%.6f CLIP 2-way ID=%.6f",
                subject,
                row["pixcorr_mean"],
                row["clip_2way_id"],
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        choices=("fit-decoder", "predict-parcels", "reconstruct", "score", "all"),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Subject labels as comma- and/or space-separated values; default is subj07 plus all 50 FOR subjects.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--final-model-dir", default=DEFAULT_FINAL_MODEL_DIR)
    parser.add_argument("--for-inference-root", default=DEFAULT_FOR_INFERENCE_ROOT)
    parser.add_argument("--selection-dir", default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--contract-root", default=DEFAULT_CONTRACT_ROOT)
    parser.add_argument("--recon-feature-dir", default=DEFAULT_RECON_FEATURE_DIR)
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-vd", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    recon_model_root = Path(args.recon_model_root).resolve()
    output_root = Path(args.output_root).resolve()
    final_model_dir = Path(args.final_model_dir).resolve()
    for_inference_root = Path(args.for_inference_root).resolve()
    selection_dir = Path(args.selection_dir).resolve()
    contract_root = Path(args.contract_root).resolve()
    recon_feature_dir = Path(args.recon_feature_dir).resolve()
    config = load_config(CONFIG_PATH)
    config["data_root"] = str((REPO_ROOT / config["data_root"]).resolve())
    config["voxel_contract"]["parcel_rest_root"] = str(
        (REPO_ROOT / config["voxel_contract"]["parcel_rest_root"]).resolve()
    )
    subjects = (
        []
        if args.command == "fit-decoder"
        else _subject_labels(for_inference_root, args.subjects)
    )

    if args.command in {"fit-decoder", "all"}:
        fit_decoder(
            output_root=output_root,
            contract_root=contract_root,
            recon_feature_dir=recon_feature_dir,
            recon_model_root=recon_model_root,
            config=config,
            device=str(args.device),
            force=bool(args.force),
        )
    if args.command in {"predict-parcels", "all"}:
        predict_parcels(
            subjects=subjects,
            output_root=output_root,
            final_model_dir=final_model_dir,
            for_inference_root=for_inference_root,
            selection_dir=selection_dir,
            contract_root=contract_root,
            config=config,
            device=str(args.device),
            force=bool(args.force),
        )
    if args.command in {"reconstruct", "all"}:
        reconstruct_subjects(
            subjects=subjects,
            output_root=output_root,
            selection_dir=selection_dir,
            recon_feature_dir=recon_feature_dir,
            recon_model_root=recon_model_root,
            config=config,
            device=str(args.device),
            skip_vd=bool(args.skip_vd),
            force=bool(args.force),
        )
    if args.command in {"score", "all"}:
        score_subjects(
            subjects=subjects,
            output_root=output_root,
            selection_dir=selection_dir,
            config=config,
            device=str(args.device),
            skip_vd=bool(args.skip_vd),
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
