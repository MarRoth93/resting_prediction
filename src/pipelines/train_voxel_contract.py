"""Gate-2 LOSO training and evaluation on the NSD voxel contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from sklearn.linear_model import Ridge

from src.alignment.external_seed_bank import (
    load_or_prepare_external_seed_runs,
    load_rest_manifest,
    seed_bank_cache_id,
    seed_defs_from_jsonable,
)
from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.data.prepare_rest_data import (
    array_sha256,
    rest_preprocessing_hash,
    validate_rest_provenance,
)
from src.evaluation.metrics import voxelwise_correlation
from src.models.encoding_factory import build_encoder, load_encoder
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_artifacts import file_sha256, json_fingerprint
from src.pipelines.train_shared_space import (
    _build_shared_stimulus_intersection,
    _get_feature_matrix_and_slices,
    set_seeds,
)


logger = logging.getLogger(__name__)

GATE_FROZEN_MEAN = 0.1781831592
STAGE_A_MIN = 0.1757758409
PARITY_REFERENCE = 0.18128718932469687
RIDGE_ALPHA_GRID = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

LOSO_SUBJECTS = (1, 2, 3, 4, 5, 6)
ARMS = ("schaefer400", "nsd499")
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class ContractSubject:
    subject_id: int
    source: NSDSubjectData
    target_indices: np.ndarray
    target_indices_path: Path
    train_fmri: np.ndarray
    test_fmri: np.ndarray
    full_rest_runs: list[np.ndarray]
    rest_runs: list[np.ndarray]
    rest_names: list[str]

    @property
    def train_stim_idx(self) -> np.ndarray:
        return self.source.train_stim_idx

    @property
    def test_stim_idx(self) -> np.ndarray:
        return self.source.test_stim_idx


@dataclass
class RidgeBaseline:
    model: Ridge
    alpha: float
    x_mean: np.ndarray
    x_std: np.ndarray
    validation_mse: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_yaml(path: str | Path) -> dict:
    loaded = yaml.safe_load(Path(path).read_text())
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return loaded


def _allowed_config_difference(path: str) -> bool:
    return (
        path == "release.name"
        or path in {"data_root", "raw_data_root"}
        or path == "voxel_contract"
        or path.startswith("voxel_contract.")
    )


def _first_config_drift(candidate: Any, frozen: Any, path: str = "") -> str | None:
    if _allowed_config_difference(path):
        return None
    if isinstance(candidate, dict) and isinstance(frozen, dict):
        for key in sorted(set(candidate) | set(frozen)):
            child = f"{path}.{key}" if path else str(key)
            if _allowed_config_difference(child):
                continue
            if key not in candidate or key not in frozen:
                return child
            drift = _first_config_drift(candidate[key], frozen[key], child)
            if drift is not None:
                return drift
        return None
    if candidate != frozen:
        return path
    return None


def validate_config_parity(
    config_path: str | Path,
    frozen_config_path: str | Path = FROZEN_CONFIG_PATH,
) -> None:
    """Reject any drift outside the explicitly allowed voxel-contract fields."""
    candidate = _read_yaml(config_path)
    frozen = _read_yaml(frozen_config_path)
    drift = _first_config_drift(candidate, frozen)
    if drift is not None:
        raise ValueError(
            f"Config drift at {drift}: {config_path} must match {frozen_config_path}."
        )
    voxel_contract = candidate.get("voxel_contract")
    if not isinstance(voxel_contract, dict):
        raise ValueError("Config drift at voxel_contract: required block is missing.")
    for key in ("root", "parcel_rest_root"):
        if not voxel_contract.get(key):
            raise ValueError(f"Config drift at voxel_contract.{key}: required value is missing.")


def _rest_run_paths(subject_dir: Path) -> list[Path]:
    pattern = re.compile(r"rest_run([0-9]+)\.npy$")
    paths = []
    for path in subject_dir.glob("rest_run*.npy"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def _contract_build_command() -> str:
    subjects = " ".join(str(subject) for subject in LOSO_SUBJECTS)
    return f"python -m src.data.voxel_contract --dataset nsd --subjects {subjects}"


def _load_contract_subject(
    subject_id: int,
    *,
    data_root: str | Path,
    contract_root: str | Path,
) -> ContractSubject:
    tag = f"subj{subject_id:02d}"
    target_path = Path(contract_root) / "nsd" / tag / "target_voxel_indices.npy"
    if not target_path.is_file():
        raise FileNotFoundError(
            f"Missing voxel contract bundle for {tag}: {target_path}. Run "
            f"`{_contract_build_command()}`."
        )

    source = NSDSubjectData(subject_id, str(data_root))
    target_indices = np.asarray(np.load(target_path), dtype=np.int64)
    if (
        target_indices.ndim != 1
        or target_indices.size == 0
        or np.any(target_indices < 0)
        or not np.array_equal(target_indices, np.unique(target_indices))
    ):
        raise ValueError(f"Invalid target_voxel_indices for {tag}: {target_path}")

    train_shape = source.train_fmri.shape
    test_shape = source.test_fmri.shape
    if len(train_shape) != 2 or len(test_shape) != 2 or train_shape[1] != test_shape[1]:
        raise ValueError(
            f"Subject {subject_id}: train/test fMRI shapes are incompatible: "
            f"{train_shape} and {test_shape}."
        )
    if int(target_indices[-1]) >= int(train_shape[1]):
        raise ValueError(
            f"Subject {subject_id}: target voxel index {int(target_indices[-1])} "
            f"exceeds source voxel count {int(train_shape[1])}."
        )
    if int(train_shape[0]) != int(source.train_stim_idx.shape[0]):
        raise ValueError(f"Subject {subject_id}: train fMRI/stimulus row mismatch.")
    if int(test_shape[0]) != int(source.test_stim_idx.shape[0]):
        raise ValueError(f"Subject {subject_id}: test fMRI/stimulus row mismatch.")

    voxel_dir = Path(data_root) / tag
    rest_paths = _rest_run_paths(voxel_dir)
    if not rest_paths:
        raise FileNotFoundError(f"Subject {subject_id}: no voxel REST runs under {voxel_dir}.")
    full_rest_runs: list[np.ndarray] = []
    sliced_rest_runs: list[np.ndarray] = []
    for path in rest_paths:
        full = np.load(path, mmap_mode="r")
        if full.ndim != 2 or int(full.shape[1]) != int(train_shape[1]):
            raise ValueError(
                f"Subject {subject_id}: REST array {path.name} has shape {full.shape}, "
                f"expected (*, {int(train_shape[1])})."
            )
        full_rest_runs.append(full)
        sliced_rest_runs.append(
            np.asarray(full[:, target_indices], dtype=np.float32)
        )

    return ContractSubject(
        subject_id=subject_id,
        source=source,
        target_indices=target_indices,
        target_indices_path=target_path,
        train_fmri=np.asarray(source.train_fmri[:, target_indices], dtype=np.float32),
        test_fmri=np.asarray(source.test_fmri[:, target_indices], dtype=np.float32),
        full_rest_runs=full_rest_runs,
        rest_runs=sliced_rest_runs,
        rest_names=[path.name for path in rest_paths],
    )


def _match_schaefer400_rest_runs(
    subject_id: int,
    *,
    voxel_subject_dir: str | Path,
    parcel_subject_dir: str | Path,
) -> list[tuple[Path, Path]]:
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
        pairs.append((voxel_path, parcel_path))
    return pairs


def _assert_training_only_scope(
    held_out: int,
    scoped_subjects: Sequence[int],
    context: str,
) -> None:
    assert held_out not in set(int(subject) for subject in scoped_subjects), (
        f"Held-out subject {held_out} leaked into {context}."
    )


def _training_shared_stimulus_intersection(
    subjects: dict[int, ContractSubject],
    train_subjects: list[int],
    held_out: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    _assert_training_only_scope(held_out, train_subjects, "shared-stimulus intersection")
    _assert_training_only_scope(held_out, list(subjects), "shared-stimulus intersection input")
    assert set(subjects) == set(train_subjects), (
        "Shared-stimulus intersection input must contain exactly the training subjects."
    )
    return _build_shared_stimulus_intersection(subjects, train_subjects)


def _standardize_training_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float32)
    x_mean = values.mean(axis=0).astype(np.float32)
    x_std = values.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-8] = 1e-8
    standardized = ((values - x_mean) / x_std).astype(np.float32)
    return standardized, x_mean, x_std


def fit_ridge_baseline(
    X: np.ndarray,
    Z: np.ndarray,
    sample_groups: np.ndarray,
    *,
    seed: int,
) -> RidgeBaseline:
    """Select ridge alpha on a deterministic stimulus-grouped split and refit."""
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    groups = np.asarray(sample_groups, dtype=np.int64).ravel()
    if X.ndim != 2 or Z.ndim != 2 or X.shape[0] != Z.shape[0]:
        raise ValueError(f"Ridge X/Z shapes are incompatible: {X.shape} and {Z.shape}.")
    if groups.shape[0] != X.shape[0]:
        raise ValueError(f"Ridge sample_groups has {groups.shape[0]} rows, expected {X.shape[0]}.")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Ridge stimulus-grouped validation requires at least two groups.")

    X_standardized, x_mean, x_std = _standardize_training_features(X)
    rng = np.random.RandomState(int(seed))
    shuffled_groups = unique_groups[rng.permutation(unique_groups.size)]
    n_val_groups = int(round(unique_groups.size * 0.10))
    n_val_groups = max(1, min(n_val_groups, unique_groups.size - 1))
    val_groups = shuffled_groups[:n_val_groups]
    val_mask = np.isin(groups, val_groups)
    train_rows = np.flatnonzero(~val_mask)
    val_rows = np.flatnonzero(val_mask)

    best_alpha = None
    best_mse = np.inf
    for alpha in RIDGE_ALPHA_GRID:
        candidate = Ridge(alpha=float(alpha))
        candidate.fit(X_standardized[train_rows], Z[train_rows])
        prediction = candidate.predict(X_standardized[val_rows])
        mse = float(np.mean((Z[val_rows] - prediction) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    assert best_alpha is not None

    model = Ridge(alpha=best_alpha)
    model.fit(X_standardized, Z)
    return RidgeBaseline(
        model=model,
        alpha=best_alpha,
        x_mean=x_mean,
        x_std=x_std,
        validation_mse=best_mse,
    )


def _require_existing_nsd499_cache(
    subject: ContractSubject,
    *,
    data_root: str | Path,
    raw_data_root: str | Path,
    seed_defs: list,
    rest_cfg: dict,
    seed_set: str,
) -> str:
    """Verify the loader will take its read-only cache path before calling it."""
    provenance = validate_rest_provenance(
        data_root=data_root,
        sub=subject.subject_id,
        expected_config=rest_cfg,
        expected_mask=subject.source.mask,
        reference_rest_runs=subject.full_rest_runs,
    )
    rest_files = load_rest_manifest(data_root, raw_data_root, subject.subject_id)
    cache_id = seed_bank_cache_id(seed_set, seed_defs, rest_cfg)
    cache_dir = (
        Path(data_root)
        / f"subj{subject.subject_id:02d}"
        / "external_seed_banks"
        / f"{seed_set}_{cache_id}"
    )
    manifest_path = cache_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(
            f"A valid existing NSD seed-bank cache is required by this read-only driver: "
            f"{manifest_path}"
        ) from exc
    expected_manifest = {
        "reference_rest_provenance_hash": provenance.get("provenance_hash"),
        "prediction_mask_sha256": array_sha256(subject.source.mask),
        "rest_preprocessing_hash": rest_preprocessing_hash(rest_cfg),
        "cache_id": cache_id,
        "n_seeds": len(seed_defs),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Existing NSD seed-bank cache is stale at {manifest_path}: {key} mismatch."
            )
    if len(rest_files) != len(subject.full_rest_runs):
        raise ValueError(
            f"Subject {subject.subject_id}: seed manifest/voxel REST run count mismatch."
        )
    for run_index, reference in enumerate(subject.full_rest_runs, start=1):
        path = cache_dir / f"external_seed_run{run_index}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"A valid existing NSD seed-bank cache is required: missing {path}."
            )
        shape = np.load(path, mmap_mode="r").shape
        expected_shape = (int(reference.shape[0]), len(seed_defs))
        if shape != expected_shape:
            raise ValueError(
                f"Existing NSD seed-bank cache has stale shape for {path}: "
                f"{shape}, expected {expected_shape}."
            )
    return cache_id


def _load_seed_runs(
    arm: str,
    *,
    subjects: dict[int, ContractSubject],
    train_subjects: list[int],
    held_out: int,
    config: dict,
) -> tuple[dict[int, list[np.ndarray]], dict]:
    all_fold_subjects = list(train_subjects) + [held_out]
    data_root = str(config["data_root"])
    if arm == "schaefer400":
        parcel_root = Path(config["voxel_contract"]["parcel_rest_root"])
        seed_runs: dict[int, list[np.ndarray]] = {}
        source_files: dict[str, list[dict[str, str]]] = {}
        for subject_id in all_fold_subjects:
            pairs = _match_schaefer400_rest_runs(
                subject_id,
                voxel_subject_dir=Path(data_root) / f"subj{subject_id:02d}",
                parcel_subject_dir=parcel_root / f"subj{subject_id:02d}",
            )
            names = [voxel_path.name for voxel_path, _ in pairs]
            if names != subjects[subject_id].rest_names:
                raise ValueError(
                    f"Subject {subject_id}: matched REST names changed during contract loading: "
                    f"loaded={subjects[subject_id].rest_names}, matched={names}."
                )
            seed_runs[subject_id] = [
                np.asarray(np.load(parcel_path, mmap_mode="r"), dtype=np.float32)
                for _, parcel_path in pairs
            ]
            source_files[f"subj{subject_id:02d}"] = [
                {"name": parcel_path.name, "sha256": file_sha256(parcel_path)}
                for _, parcel_path in pairs
            ]
        source = {
            "identifier": "schaefer400",
            "parcel_rest_root": str(parcel_root.resolve()),
            "files": source_files,
        }
        source["fingerprint"] = json_fingerprint(source)
        return seed_runs, source

    # Parity arm: replicate the FROZEN seed registry exactly. The frozen release
    # built its 499 seed defs over all six train subjects; recomputing over the
    # fold's five training subjects yields different defs, a different cache id,
    # and breaks parity with 0.18129. Registry scoping is therefore inherited
    # from the reference protocol (a property of the frozen release, not new
    # leakage introduced by this driver). The registry is fixed a priori, so no
    # per-fold training-only scope assertion applies here.
    external_cfg = config["alignment"]["external_seed_bank"]
    seed_set = str(external_cfg["seed_set"])
    frozen_seed_info_path = Path("artifacts/model/external_seed_info.json")
    if not frozen_seed_info_path.is_file():
        raise FileNotFoundError(
            f"Frozen seed registry required for the nsd499 parity arm: {frozen_seed_info_path}"
        )
    with open(frozen_seed_info_path) as handle:
        frozen_seed_info = json.load(handle)
    if str(frozen_seed_info.get("seed_set")) != seed_set:
        raise ValueError(
            f"Frozen seed registry seed_set {frozen_seed_info.get('seed_set')!r} "
            f"does not match config seed_set {seed_set!r}."
        )
    seed_defs = seed_defs_from_jsonable(frozen_seed_info["seed_defs"])
    if len(seed_defs) != int(frozen_seed_info["n_seeds"]):
        raise ValueError(
            f"Frozen seed registry is inconsistent: {len(seed_defs)} defs vs "
            f"n_seeds={frozen_seed_info['n_seeds']}."
        )

    cache_ids = {}
    seed_runs = {}
    for subject_id in all_fold_subjects:
        subject = subjects[subject_id]
        cache_ids[f"subj{subject_id:02d}"] = _require_existing_nsd499_cache(
            subject,
            data_root=data_root,
            raw_data_root=str(config["raw_data_root"]),
            seed_defs=seed_defs,
            rest_cfg=config["rest_preprocessing"],
            seed_set=seed_set,
        )
        seed_runs[subject_id] = load_or_prepare_external_seed_runs(
            sub=subject_id,
            data_root=data_root,
            raw_data_root=str(config["raw_data_root"]),
            pred_mask=subject.source.mask,
            seed_defs=seed_defs,
            rest_cfg=config["rest_preprocessing"],
            seed_set=seed_set,
            reference_rest_runs=subject.full_rest_runs,
            force_recompute=False,
        )
    source = {
        "identifier": seed_set,
        "registry": "frozen_release_external_seed_info",
        "registry_file_sha256": file_sha256(Path("artifacts/model/external_seed_info.json")),
        "n_seeds": len(seed_defs),
        "cache_ids": cache_ids,
        "seed_definitions": [
            {
                "seed_set": seed.seed_set,
                "atlas_file": seed.atlas_file,
                "label": int(seed.label),
                "name": seed.name,
            }
            for seed in seed_defs
        ],
    }
    source["fingerprint"] = json_fingerprint(source)
    return seed_runs, source


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _effective_rank(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    denominator = float(np.sum(values**2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(values) ** 2 / denominator)


def _connectivity_diagnostic(connectivity: np.ndarray) -> dict:
    singular_values = np.linalg.svd(connectivity, compute_uv=False)
    return {
        "top_150_singular_values": singular_values[:150].astype(float).tolist(),
        "effective_rank": _effective_rank(singular_values),
    }


def _input_fingerprints(
    *,
    config_path: str | Path,
    effective_config: dict,
    subjects: dict[int, ContractSubject],
    seed_source: dict,
    train_subjects: list[int],
    held_out: int,
) -> dict:
    return {
        "config_sha256": file_sha256(config_path),
        "effective_config_sha256": json_fingerprint(effective_config),
        "target_voxel_indices_sha256": {
            f"subj{subject_id:02d}": file_sha256(subject.target_indices_path)
            for subject_id, subject in sorted(subjects.items())
        },
        "seed_source_identifier": seed_source,
        "training_scope": {
            "train_subjects": list(train_subjects),
            "held_out": int(held_out),
        },
    }


def _first_value_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                return child
            mismatch = _first_value_difference(expected[key], actual[key], child)
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return path
    return None


def _fold_artifact_fingerprints(fold_dir: Path) -> dict[str, str]:
    model_dir = fold_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Completed fold is missing artifact directory: {model_dir}")
    files = [path for path in sorted(model_dir.rglob("*")) if path.is_file()]
    if not files:
        raise ValueError(f"Completed fold has no model artifacts: {model_dir}")
    return {
        str(path.relative_to(fold_dir)): file_sha256(path)
        for path in files
    }


def _validate_completed_fold(
    fold_dir: Path,
    *,
    expected_identity: dict,
    expected_fingerprints: dict,
) -> dict:
    result_path = fold_dir / "fold_result.json"
    try:
        result = json.loads(result_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Existing fold has no valid fold_result.json: {fold_dir}") from exc

    mismatch = _first_value_difference(
        expected_fingerprints,
        result.get("input_fingerprints"),
        "input_fingerprints",
    )
    if mismatch is not None:
        raise ValueError(f"Fold fingerprint mismatch at {mismatch}: {result_path}")
    mismatch = _first_value_difference(expected_identity, {
        key: result.get(key) for key in expected_identity
    })
    if mismatch is not None:
        raise ValueError(f"Fold fingerprint mismatch at {mismatch}: {result_path}")

    stored_payload_fingerprint = result.get("result_payload_fingerprint")
    payload = dict(result)
    payload.pop("result_payload_fingerprint", None)
    if stored_payload_fingerprint != json_fingerprint(payload):
        raise ValueError(f"Existing fold result payload was modified: {result_path}")
    actual_artifacts = _fold_artifact_fingerprints(fold_dir)
    if result.get("artifact_fingerprints") != actual_artifacts:
        raise ValueError(f"Existing fold artifacts were modified or are incomplete: {fold_dir}")

    builder = SharedSpaceBuilder.load(str(fold_dir / "model"))
    encoder = load_encoder(str(fold_dir / "model"))
    if sorted(builder.subject_bases) != expected_identity["train_subjects"]:
        raise ValueError(f"Existing fold builder training subjects are stale: {fold_dir}")
    if int(encoder.output_dim) != int(builder.k_global):
        raise ValueError(f"Existing fold encoder latent dimension is stale: {fold_dir}")
    return result


def _resume_or_reset_fold(
    fold_dir: Path,
    *,
    expected_identity: dict,
    expected_fingerprints: dict,
    force: bool,
) -> dict | None:
    if not fold_dir.exists():
        return None
    if not any(fold_dir.iterdir()):
        return None
    try:
        result = _validate_completed_fold(
            fold_dir,
            expected_identity=expected_identity,
            expected_fingerprints=expected_fingerprints,
        )
    except (FileNotFoundError, ValueError) as exc:
        if not force:
            raise
        logger.warning("Discarding stale fold under --force: %s (%s)", fold_dir, exc)
        shutil.rmtree(fold_dir)
        return None
    logger.info("Skipping completed fold with matching fingerprints: %s", fold_dir)
    return result


def _validate_completed_final(
    final_dir: Path,
    *,
    expected_identity: dict,
    expected_fingerprints: dict,
) -> dict:
    result_path = final_dir / "final_result.json"
    try:
        result = json.loads(result_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Existing final artifact has no valid final_result.json: {final_dir}"
        ) from exc

    mismatch = _first_value_difference(
        expected_fingerprints,
        result.get("input_fingerprints"),
        "input_fingerprints",
    )
    if mismatch is not None:
        raise ValueError(f"Final artifact fingerprint mismatch at {mismatch}: {result_path}")
    mismatch = _first_value_difference(expected_identity, {
        key: result.get(key) for key in expected_identity
    })
    if mismatch is not None:
        raise ValueError(f"Final artifact fingerprint mismatch at {mismatch}: {result_path}")
    if result.get("config_hash") != expected_fingerprints.get("effective_config_sha256"):
        raise ValueError(f"Final artifact config hash mismatch: {result_path}")

    manifest_path = final_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Existing final artifact has no valid manifest.json: {final_dir}"
        ) from exc
    mismatch = _first_value_difference(
        expected_fingerprints,
        manifest,
        "manifest",
    )
    if mismatch is not None:
        raise ValueError(f"Final artifact manifest mismatch at {mismatch}: {manifest_path}")

    stored_payload_fingerprint = result.get("result_payload_fingerprint")
    payload = dict(result)
    payload.pop("result_payload_fingerprint", None)
    if stored_payload_fingerprint != json_fingerprint(payload):
        raise ValueError(f"Existing final result payload was modified: {result_path}")
    actual_artifacts = _fold_artifact_fingerprints(final_dir)
    if result.get("artifact_fingerprints") != actual_artifacts:
        raise ValueError(
            f"Existing final artifacts were modified or are incomplete: {final_dir}"
        )

    builder = SharedSpaceBuilder.load(str(final_dir / "model"))
    encoder = load_encoder(str(final_dir / "model"))
    if sorted(builder.subject_bases) != expected_identity["train_subjects"]:
        raise ValueError(f"Existing final builder training subjects are stale: {final_dir}")
    if int(encoder.output_dim) != int(builder.k_global):
        raise ValueError(f"Existing final encoder latent dimension is stale: {final_dir}")
    return result


def _resume_or_reset_final(
    final_dir: Path,
    *,
    expected_identity: dict,
    expected_fingerprints: dict,
    force: bool,
) -> None:
    if not final_dir.exists():
        return
    if not any(final_dir.iterdir()):
        return
    try:
        _validate_completed_final(
            final_dir,
            expected_identity=expected_identity,
            expected_fingerprints=expected_fingerprints,
        )
    except (FileNotFoundError, ValueError) as exc:
        if not force:
            raise
        logger.warning(
            "Discarding stale final artifact under --force: %s (%s)", final_dir, exc
        )
        shutil.rmtree(final_dir)
        return
    raise FileExistsError(
        f"Final artifact already exists with matching fingerprints; refusing to re-run: {final_dir}"
    )


def _save_ridge_baseline(model_dir: Path, ridge: RidgeBaseline) -> None:
    np.savez(
        model_dir / "ridge_baseline.npz",
        alpha=np.asarray(ridge.alpha),
        validation_mse=np.asarray(ridge.validation_mse),
        x_mean=ridge.x_mean,
        x_std=ridge.x_std,
        coef=np.asarray(ridge.model.coef_),
        intercept=np.asarray(ridge.model.intercept_),
    )


def _save_fold_result(fold_dir: Path, result: dict) -> dict:
    payload = dict(result)
    payload["artifact_fingerprints"] = _fold_artifact_fingerprints(fold_dir)
    payload["result_payload_fingerprint"] = json_fingerprint(payload)
    result_path = fold_dir / "fold_result.json"
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return payload


def _save_final_result(final_dir: Path, result: dict) -> dict:
    manifest_path = final_dir / "manifest.json"
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_temporary.write_text(
        json.dumps(result["input_fingerprints"], indent=2, sort_keys=True) + "\n"
    )
    manifest_temporary.replace(manifest_path)

    payload = dict(result)
    payload["artifact_fingerprints"] = _fold_artifact_fingerprints(final_dir)
    payload["result_payload_fingerprint"] = json_fingerprint(payload)
    result_path = final_dir / "final_result.json"
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return payload


def _run_fold(
    *,
    arm: str,
    seed: int,
    held_out: int,
    config: dict,
    config_path: str | Path,
    output_root: str | Path,
    force: bool,
    train_subjects: list[int] | None = None,
    final_mode: bool = False,
) -> dict:
    started_at = _utc_now()
    if train_subjects is None:
        train_subjects = [subject for subject in LOSO_SUBJECTS if subject != held_out]
    else:
        train_subjects = list(train_subjects)
    _assert_training_only_scope(held_out, train_subjects, "fold training set")
    all_subjects = list(train_subjects) + [held_out]
    contract_root = str(config["voxel_contract"]["root"])
    subjects = {
        subject_id: _load_contract_subject(
            subject_id,
            data_root=str(config["data_root"]),
            contract_root=contract_root,
        )
        for subject_id in all_subjects
    }
    external_seed_runs, seed_source = _load_seed_runs(
        arm,
        subjects=subjects,
        train_subjects=train_subjects,
        held_out=held_out,
        config=config,
    )
    fingerprints = _input_fingerprints(
        config_path=config_path,
        effective_config=config,
        subjects=subjects,
        seed_source=seed_source,
        train_subjects=train_subjects,
        held_out=held_out,
    )
    identity = {
        "arm": arm,
        "seed": int(seed),
        "held_out": int(held_out),
        "train_subjects": list(train_subjects),
    }
    if final_mode:
        identity["subject7_usage"] = "evaluation_only_once_D08"
        fold_dir = Path(output_root) / f"seed{seed}"
        _resume_or_reset_final(
            fold_dir,
            expected_identity=identity,
            expected_fingerprints=fingerprints,
            force=force,
        )
    else:
        fold_dir = (
            Path(output_root)
            / arm
            / f"seed{seed}"
            / f"fold_sub{held_out:02d}"
        )
        resumed = _resume_or_reset_fold(
            fold_dir,
            expected_identity=identity,
            expected_fingerprints=fingerprints,
            force=force,
        )
        if resumed is not None:
            return resumed

    scoped_subjects = {subject: subjects[subject] for subject in train_subjects}
    shared_stimulus, shared_rows, _ = _training_shared_stimulus_intersection(
        scoped_subjects,
        train_subjects,
        held_out,
    )
    rest_runs = {subject: subjects[subject].rest_runs for subject in train_subjects}
    task_responses_shared = {
        subject: np.asarray(
            subjects[subject].test_fmri[shared_rows[subject]],
            dtype=np.float32,
        )
        for subject in train_subjects
    }

    alignment = config["alignment"]
    builder = SharedSpaceBuilder(
        n_components=int(alignment["n_components"]),
        min_k=int(alignment["min_k"]),
        ensemble_method=str(alignment["ensemble_method"]),
        max_iters=int(alignment["max_iters"]),
        tol=float(alignment["tol"]),
    )
    builder.fit(
        rest_runs=rest_runs,
        task_responses_shared=task_responses_shared,
        external_seed_runs={
            subject: external_seed_runs[subject] for subject in train_subjects
        },
    )

    _assert_training_only_scope(
        held_out,
        train_subjects,
        "feature standardizer and ridge alpha selection",
    )
    features = NSDFeatures(Path(config["data_root"]) / "features")
    feature_type = str(config["features"]["type"])
    feature_streams = [str(value) for value in config["features"].get("streams", [])] or None
    X_all: list[np.ndarray] = []
    Z_all: list[np.ndarray] = []
    sample_groups_all: list[np.ndarray] = []
    feature_slices = None
    for subject_id in train_subjects:
        subject = subjects[subject_id]
        X, slices = _get_feature_matrix_and_slices(
            features=features,
            stim_idx=subject.train_stim_idx,
            feature_type=feature_type,
            streams=feature_streams,
        )
        if feature_slices is None:
            feature_slices = slices
        elif slices != feature_slices:
            raise ValueError(f"Subject {subject_id}: CLIP feature slices differ within fold.")
        P = builder.subject_bases[subject_id]
        R = builder.subject_rotations[subject_id]
        Z = subject.train_fmri @ P @ R
        X_all.append(np.asarray(X, dtype=np.float32))
        Z_all.append(np.asarray(Z, dtype=np.float32))
        sample_groups_all.append(np.asarray(subject.train_stim_idx, dtype=np.int64))
    X_pooled = np.concatenate(X_all, axis=0)
    Z_pooled = np.concatenate(Z_all, axis=0)
    sample_groups = np.concatenate(sample_groups_all, axis=0)

    encoder = build_encoder(
        config=config,
        input_dim=int(X_pooled.shape[1]),
        output_dim=int(Z_pooled.shape[1]),
        feature_slices=feature_slices,
    )
    encoder.fit(X_pooled, Z_pooled, sample_groups=sample_groups)
    ridge = fit_ridge_baseline(X_pooled, Z_pooled, sample_groups, seed=seed)

    held = subjects[held_out]
    P_held, R_held = builder.align_new_subject_zeroshot(
        rest_runs=held.rest_runs,
        external_seed_runs=external_seed_runs[held_out],
    )
    eval_indices = fixed_eval_indices(
        n_shared=len(held.test_stim_idx),
        eval_size=int(config["evaluation"]["fixed_eval_size"]),
        seed=int(config["evaluation"]["eval_split_seed"]),
    )
    X_test, test_slices = _get_feature_matrix_and_slices(
        features=features,
        stim_idx=held.test_stim_idx,
        feature_type=feature_type,
        streams=feature_streams,
    )
    if test_slices != feature_slices:
        raise ValueError("Held-out CLIP feature slices differ from pooled training features.")
    encoder_voxels = encoder.predict_voxels(X_test, P_held, R_held)
    X_test_standardized = (
        (np.asarray(X_test, dtype=np.float32) - ridge.x_mean) / ridge.x_std
    ).astype(np.float32)
    ridge_shared = ridge.model.predict(X_test_standardized)
    ridge_voxels = (ridge_shared @ R_held.T @ P_held.T).astype(np.float32)
    truth = held.test_fmri
    encoder_median = float(
        np.median(
            voxelwise_correlation(
                truth[eval_indices],
                encoder_voxels[eval_indices],
            )
        )
    )
    ridge_median = float(
        np.median(
            voxelwise_correlation(
                truth[eval_indices],
                ridge_voxels[eval_indices],
            )
        )
    )

    diagnostics = {
        "training_subjects": {
            f"subj{subject_id:02d}": _connectivity_diagnostic(
                builder.subject_connectivity[subject_id]
            )
            for subject_id in train_subjects
        }
    }
    held_connectivity = compute_rest_connectivity(
        held.rest_runs,
        seed_runs=external_seed_runs[held_out],
        ensemble=builder.ensemble_method,
    )
    held_diagnostic = _connectivity_diagnostic(held_connectivity)
    held_fingerprint = held_connectivity @ P_held @ R_held
    template_norm = float(np.linalg.norm(builder.template_fingerprint))
    if template_norm <= 0.0:
        raise ValueError("Shared-space fingerprint template has zero norm.")
    held_diagnostic["fingerprint_residual"] = float(
        np.linalg.norm(held_fingerprint - builder.template_fingerprint) / template_norm
    )
    diagnostics["held_out"] = held_diagnostic

    fold_dir.mkdir(parents=True, exist_ok=True)
    model_dir = fold_dir / "model"
    builder.save(str(model_dir))
    encoder.save(str(model_dir / "encoder"))
    _save_ridge_baseline(model_dir, ridge)
    result = {
        **identity,
        "encoder_median_r": encoder_median,
        "ridge_median_r": ridge_median,
        "ridge_alpha": ridge.alpha,
        "n_target_voxels": int(held.target_indices.size),
        "n_eval": int(eval_indices.size),
        "eval_indices_sha256": _array_sha256(eval_indices),
        "shared_stimulus_sha256": _array_sha256(shared_stimulus),
        "input_fingerprints": fingerprints,
        "diagnostics": diagnostics,
        "timestamps": {
            "started_at": started_at,
            "finished_at": _utc_now(),
        },
    }
    if final_mode:
        result["config_hash"] = fingerprints["effective_config_sha256"]
        return _save_final_result(fold_dir, result)
    return _save_fold_result(fold_dir, result)


def decision_fields(
    arm: str,
    seed: int,
    seed_mean_encoder: float,
    seed_mean_ridge: float,
) -> dict:
    parity_ok = None
    if arm == "nsd499":
        parity_delta = abs(seed_mean_encoder - PARITY_REFERENCE)
        parity_ok = bool(
            parity_delta < 0.005
            or np.isclose(parity_delta, 0.005, rtol=0.0, atol=1e-12)
        )
    stage_a_pass = None
    if arm == "schaefer400" and int(seed) == 42:
        stage_a_pass = bool(
            seed_mean_encoder >= STAGE_A_MIN
            and seed_mean_encoder > seed_mean_ridge
        )
    return {
        "parity_ok": parity_ok,
        "stage_a_pass": stage_a_pass,
        "gate_threshold": GATE_FROZEN_MEAN,
    }


def _write_summary(
    *,
    output_root: str | Path,
    arm: str,
    seed: int,
    fold_results: list[dict],
) -> dict:
    encoder_values = [float(result["encoder_median_r"]) for result in fold_results]
    ridge_values = [float(result["ridge_median_r"]) for result in fold_results]
    seed_mean_encoder = float(np.mean(encoder_values))
    seed_mean_ridge = float(np.mean(ridge_values))
    summary = {
        "arm": arm,
        "seed": int(seed),
        "per_fold_medians": {
            f"subj{int(result['held_out']):02d}": {
                "encoder_median_r": float(result["encoder_median_r"]),
                "ridge_median_r": float(result["ridge_median_r"]),
            }
            for result in sorted(fold_results, key=lambda value: int(value["held_out"]))
        },
        "seed_mean_encoder": seed_mean_encoder,
        "seed_mean_ridge": seed_mean_ridge,
        **decision_fields(arm, seed, seed_mean_encoder, seed_mean_ridge),
        "timestamp": _utc_now(),
    }
    summary_dir = Path(output_root) / arm / f"seed{seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    print(
        "VERDICT "
        f"arm={arm} seed={seed} encoder_mean={seed_mean_encoder:.10f} "
        f"ridge_mean={seed_mean_ridge:.10f} gate_threshold={GATE_FROZEN_MEAN:.10f} "
        f"parity_ok={summary['parity_ok']} stage_a_pass={summary['stage_a_pass']}"
    )
    return summary


def run_loso(
    *,
    arm: str,
    seed: int = 42,
    folds: list[int] | None = None,
    config_path: str | Path = "config_voxel_contract.yaml",
    output_root: str | Path = "artifacts/voxel_contract_loso",
    force: bool = False,
    device: str = "cuda",
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
    selected_folds = list(LOSO_SUBJECTS if folds is None else folds)
    invalid = [fold for fold in selected_folds if fold not in LOSO_SUBJECTS]
    if invalid:
        raise ValueError(f"Requested folds must be within 1-6; got {invalid}.")

    validate_config_parity(config_path)
    config = copy.deepcopy(load_config(config_path))
    config["random_seed"] = int(seed)
    config["encoding"]["seed"] = int(seed)
    config["encoding"]["device"] = str(device)
    config["features"]["device"] = str(device)
    set_seeds(seed)

    fold_results = []
    for held_out in selected_folds:
        fold_results.append(
            _run_fold(
                arm=arm,
                seed=seed,
                held_out=held_out,
                config=config,
                config_path=config_path,
                output_root=output_root,
                force=force,
            )
        )
    return _write_summary(
        output_root=output_root,
        arm=arm,
        seed=seed,
        fold_results=fold_results,
    )


def run_final(
    *,
    arm: str,
    seed: int = 42,
    config_path: str | Path = "config_voxel_contract.yaml",
    output_root: str | Path = "artifacts/voxel_contract_final",
    force: bool = False,
    device: str = "cuda",
) -> dict:
    if arm != "schaefer400":
        raise ValueError("--final requires --arm schaefer400.")

    validate_config_parity(config_path)
    config = copy.deepcopy(load_config(config_path))
    config["random_seed"] = int(seed)
    config["encoding"]["seed"] = int(seed)
    config["encoding"]["device"] = str(device)
    config["features"]["device"] = str(device)
    set_seeds(seed)

    return _run_fold(
        arm=arm,
        seed=seed,
        held_out=7,
        config=config,
        config_path=config_path,
        output_root=output_root,
        force=force,
        train_subjects=list(LOSO_SUBJECTS),
        final_mode=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--folds", type=_parse_folds, default="1,2,3,4,5,6")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--config", default="config_voxel_contract.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.final:
        if args.arm != "schaefer400":
            parser.error("--final requires --arm schaefer400.")
        logger.info("Subject 7 evaluation is one-time and report-only (D-08).")
        run_final(
            arm=args.arm,
            seed=args.seed,
            config_path=args.config,
            output_root=(
                args.output_root
                if args.output_root is not None
                else "artifacts/voxel_contract_final"
            ),
            force=args.force,
            device=args.device,
        )
    else:
        run_loso(
            arm=args.arm,
            seed=args.seed,
            folds=args.folds,
            config_path=args.config,
            output_root=(
                args.output_root
                if args.output_root is not None
                else "artifacts/voxel_contract_loso"
            ),
            force=args.force,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
