"""Gate 3a sensitivity bound for MINIMAL-cleanup zero-shot alignment."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.data.prepare_rest_data import prepare_rest_data
from src.data.prepare_schaefer400_nsd import prepare_schaefer400_nsd_subject
from src.evaluation.metrics import voxelwise_correlation
from src.models.encoding_factory import load_encoder
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_artifacts import (
    directory_file_fingerprints,
    file_sha256,
)
from src.pipelines.train_shared_space import _get_feature_matrix_and_slices
from src.schaefer400_config import (
    load_schaefer400_config,
    resolve_schaefer400_roots,
)


logger = logging.getLogger(__name__)

LOSO_SUBJECTS = (1, 2, 3, 4, 5, 6)
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHAEFER400_CONFIG_PATH = REPO_ROOT / "config_schaefer400.yaml"
FLAG_FIRST_ORDER_MEANING = (
    "cleanup shift is comparable to the seed-bank cost and must be reviewed "
    "before FOR inference"
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


def _read_yaml(path: str | Path) -> dict:
    loaded = yaml.safe_load(Path(path).read_text())
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return loaded


def derive_minimal_rest_preprocessing(rest_preprocessing: dict) -> dict:
    """Copy a REST profile and disable only motion cleanup requirements."""
    minimal = copy.deepcopy(rest_preprocessing)
    motion = minimal.get("motion_censoring")
    nuisance = minimal.get("nuisance_regression")
    if not isinstance(motion, dict) or not isinstance(nuisance, dict):
        raise TypeError(
            "rest_preprocessing motion_censoring and nuisance_regression must be mappings."
        )
    motion["enabled"] = False
    nuisance["enabled"] = False
    nuisance["require_motion"] = False
    return minimal


def derive_gate3a_schaefer400_config(
    schaefer400_config: dict,
    minclean_parcel_root: str | Path,
) -> dict:
    """Return the Schaefer-400 config copy used for Gate 3a REST preparation."""
    derived = copy.deepcopy(schaefer400_config)
    rest_preprocessing = derived.get("rest_preprocessing")
    if not isinstance(rest_preprocessing, dict):
        raise TypeError("Schaefer-400 config requires a rest_preprocessing mapping.")
    derived["rest_preprocessing"] = derive_minimal_rest_preprocessing(
        rest_preprocessing
    )
    derived["data_root"] = str(minclean_parcel_root)
    return derived


def _rest_run_paths(subject_dir: str | Path) -> list[Path]:
    pattern = re.compile(r"rest_run([0-9]+)\.npy$")
    indexed_paths = []
    for path in Path(subject_dir).glob("rest_run*.npy"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            indexed_paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed_paths)]


def _match_minimal_rest_runs(
    subject_id: int,
    *,
    voxel_subject_dir: str | Path,
    parcel_subject_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Match the two MINIMAL REST views by filename and verify their TR counts."""
    voxel_dir = Path(voxel_subject_dir)
    parcel_dir = Path(parcel_subject_dir)
    voxel_files = {path.name: path for path in _rest_run_paths(voxel_dir)}
    parcel_files = {path.name: path for path in _rest_run_paths(parcel_dir)}
    if set(voxel_files) != set(parcel_files):
        voxel_only = sorted(set(voxel_files) - set(parcel_files))
        parcel_only = sorted(set(parcel_files) - set(voxel_files))
        raise ValueError(
            f"Subject {subject_id}: MINIMAL voxel/parcel REST filename mismatch: "
            f"voxel_only={voxel_only}, parcel_only={parcel_only}."
        )
    if not voxel_files:
        raise ValueError(f"Subject {subject_id}: no matched MINIMAL REST runs were found.")

    pairs = []
    for voxel_path in _rest_run_paths(voxel_dir):
        parcel_path = parcel_files[voxel_path.name]
        voxel_shape = np.load(voxel_path, mmap_mode="r").shape
        parcel_shape = np.load(parcel_path, mmap_mode="r").shape
        if len(voxel_shape) != 2 or len(parcel_shape) != 2:
            raise ValueError(
                f"Subject {subject_id}: MINIMAL REST arrays must be 2D for "
                f"{voxel_path.name}: voxel={voxel_shape}, parcel={parcel_shape}."
            )
        if int(voxel_shape[0]) != int(parcel_shape[0]):
            raise ValueError(
                f"Subject {subject_id}: MINIMAL REST TR mismatch for "
                f"{voxel_path.name}: voxel={int(voxel_shape[0])}, "
                f"parcel={int(parcel_shape[0])}."
            )
        pairs.append((voxel_path, parcel_path))
    return pairs


def cleanup_delta(median_r_minclean: float, median_r_reference: float) -> float:
    return float(median_r_minclean) - float(median_r_reference)


def cleanup_decision_fields(deltas: Sequence[float]) -> dict:
    if not deltas:
        raise ValueError("At least one fold delta is required.")
    mean_delta = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    return {
        "mean_delta": mean_delta,
        "flag_first_order": bool(mean_delta < -0.01),
        "flag_first_order_meaning": FLAG_FIRST_ORDER_MEANING,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "sha256": file_sha256(path),
    }


def _minimal_rest_fingerprints(
    pairs: Sequence[tuple[Path, Path]],
) -> dict[str, dict[str, dict]]:
    return {
        "voxel": {
            voxel_path.name: _file_fingerprint(voxel_path)
            for voxel_path, _ in pairs
        },
        "parcel": {
            parcel_path.name: _file_fingerprint(parcel_path)
            for _, parcel_path in pairs
        },
    }


def _subject_outputs_exist(voxel_dir: Path, parcel_dir: Path) -> bool:
    return bool(
        (voxel_dir / "rest_run_manifest.json").is_file()
        and (parcel_dir / "rest_run_manifest.json").is_file()
        and _rest_run_paths(voxel_dir)
        and _rest_run_paths(parcel_dir)
    )


def prepare_gate3a(
    *,
    folds: Sequence[int],
    config_path: str | Path,
    output_root: str | Path,
    minclean_voxel_root: str | Path,
    minclean_parcel_root: str | Path,
    force: bool,
) -> dict:
    """Prepare matched MINIMAL-cleanup voxel and Schaefer-400 REST views."""
    config_path = Path(config_path)
    output_root = Path(output_root)
    minclean_voxel_root = Path(minclean_voxel_root)
    minclean_parcel_root = Path(minclean_parcel_root)
    config = load_config(config_path)
    minimal_profile = derive_minimal_rest_preprocessing(
        config["rest_preprocessing"]
    )

    schaefer400_config = _read_yaml(SCHAEFER400_CONFIG_PATH)
    derived_schaefer400 = derive_gate3a_schaefer400_config(
        schaefer400_config,
        minclean_parcel_root,
    )
    if derived_schaefer400["rest_preprocessing"] != minimal_profile:
        raise ValueError(
            "Voxel-contract and Schaefer-400 REST profiles differ outside the "
            "Gate 3a motion-cleanup switches."
        )
    derived_config_path = output_root / "config_gate3a_schaefer400.yaml"
    _write_yaml(derived_config_path, derived_schaefer400)
    effective_schaefer400 = resolve_schaefer400_roots(
        load_schaefer400_config(derived_config_path),
        data_root=minclean_parcel_root,
        raw_data_root=config["raw_data_root"],
    )

    subjects = {}
    for subject_id in folds:
        tag = f"subj{int(subject_id):02d}"
        voxel_dir = minclean_voxel_root / tag
        parcel_dir = minclean_parcel_root / tag
        skipped = _subject_outputs_exist(voxel_dir, parcel_dir) and not force
        if skipped:
            logger.info("Subject %d: skipping existing MINIMAL REST outputs", subject_id)
        else:
            prepare_rest_data(
                int(subject_id),
                data_root=str(config["raw_data_root"]),
                output_root=str(minclean_voxel_root),
                config=minimal_profile,
            )
            prepare_schaefer400_nsd_subject(
                int(subject_id),
                effective_schaefer400,
                force=force,
                prepare_task=False,
                prepare_rest=True,
            )
        pairs = _match_minimal_rest_runs(
            int(subject_id),
            voxel_subject_dir=voxel_dir,
            parcel_subject_dir=parcel_dir,
        )
        subjects[tag] = {
            "status": "skipped_existing" if skipped else "prepared",
            "minimal_rest_files": _minimal_rest_fingerprints(pairs),
        }

    manifest = {
        "gate": "gate3a_cleanup_stress",
        "minimal_rest_preprocessing": minimal_profile,
        "config": _file_fingerprint(config_path),
        "derived_schaefer400_config": _file_fingerprint(derived_config_path),
        "subjects": subjects,
    }
    _write_json(output_root / "prepare_manifest.json", manifest)
    return manifest


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
    result_path = fold_dir / "fold_result.json"
    required = [
        model_dir / "builder.npz",
        model_dir / "encoder" / "metadata.json",
        model_dir / "ridge_baseline.npz",
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


def _load_target_indices(
    *,
    subject_id: int,
    contract_root: str | Path,
    source_voxel_count: int,
) -> tuple[np.ndarray, Path]:
    target_path = (
        Path(contract_root)
        / "nsd"
        / f"subj{int(subject_id):02d}"
        / "target_voxel_indices.npy"
    )
    if not target_path.is_file():
        raise FileNotFoundError(
            f"Missing voxel contract target indices for subject {subject_id}: {target_path}."
        )
    target_indices = np.asarray(np.load(target_path), dtype=np.int64)
    if (
        target_indices.ndim != 1
        or target_indices.size == 0
        or np.any(target_indices < 0)
        or not np.array_equal(target_indices, np.unique(target_indices))
    ):
        raise ValueError(f"Invalid target_voxel_indices for subject {subject_id}: {target_path}")
    if int(target_indices[-1]) >= int(source_voxel_count):
        raise ValueError(
            f"Subject {subject_id}: target voxel index {int(target_indices[-1])} "
            f"exceeds source voxel count {int(source_voxel_count)}."
        )
    return target_indices, target_path


def _evaluate_fold(
    *,
    subject_id: int,
    seed: int,
    config: dict,
    config_path: str | Path,
    loso_root: str | Path,
    minclean_voxel_root: str | Path,
    minclean_parcel_root: str | Path,
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
    if "encoder_median_r" not in reference_result:
        raise ValueError(f"Gate-2 fold result has no encoder_median_r: {result_path}")

    source = NSDSubjectData(int(subject_id), str(config["data_root"]))
    test_shape = source.test_fmri.shape
    if len(test_shape) != 2 or int(test_shape[0]) != int(source.test_stim_idx.shape[0]):
        raise ValueError(
            f"Subject {subject_id}: test fMRI/stimulus shapes are incompatible: "
            f"{test_shape} and {source.test_stim_idx.shape}."
        )
    target_indices, target_path = _load_target_indices(
        subject_id=subject_id,
        contract_root=config["voxel_contract"]["root"],
        source_voxel_count=int(test_shape[1]),
    )

    tag = f"subj{int(subject_id):02d}"
    pairs = _match_minimal_rest_runs(
        int(subject_id),
        voxel_subject_dir=Path(minclean_voxel_root) / tag,
        parcel_subject_dir=Path(minclean_parcel_root) / tag,
    )
    rest_runs = []
    external_seed_runs = []
    for voxel_path, parcel_path in pairs:
        voxel_run = np.load(voxel_path, mmap_mode="r")
        if int(voxel_run.shape[1]) != int(test_shape[1]):
            raise ValueError(
                f"Subject {subject_id}: MINIMAL voxel REST {voxel_path.name} has "
                f"{int(voxel_run.shape[1])} voxels, expected {int(test_shape[1])}."
            )
        rest_runs.append(
            np.asarray(voxel_run[:, target_indices], dtype=np.float32)
        )
        external_seed_runs.append(
            np.asarray(np.load(parcel_path, mmap_mode="r"), dtype=np.float32)
        )

    builder = SharedSpaceBuilder.load(str(model_dir))
    encoder = load_encoder(str(model_dir))
    P, R = builder.align_new_subject_zeroshot(
        rest_runs=rest_runs,
        external_seed_runs=external_seed_runs,
    )
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
    predicted_voxels = encoder.predict_voxels(X_test, P, R)
    truth = np.asarray(source.test_fmri[:, target_indices], dtype=np.float32)
    if predicted_voxels.shape != truth.shape:
        raise ValueError(
            f"Subject {subject_id}: prediction/truth shape mismatch: "
            f"{predicted_voxels.shape} vs {truth.shape}."
        )
    eval_indices = fixed_eval_indices(
        n_shared=len(source.test_stim_idx),
        eval_size=int(config["evaluation"]["fixed_eval_size"]),
        seed=int(config["evaluation"]["eval_split_seed"]),
    )
    median_r_minclean = float(
        np.median(
            voxelwise_correlation(
                truth[eval_indices],
                predicted_voxels[eval_indices],
            )
        )
    )
    median_r_reference = float(reference_result["encoder_median_r"])
    return {
        "held_out": int(subject_id),
        "median_r_minclean": median_r_minclean,
        "median_r_reference": median_r_reference,
        "delta": cleanup_delta(median_r_minclean, median_r_reference),
        "n_target_voxels": int(target_indices.size),
        "n_eval": int(eval_indices.size),
        "fingerprints": {
            "fold_model_files": directory_file_fingerprints(model_dir),
            "minimal_rest_files": _minimal_rest_fingerprints(pairs),
            "target_voxel_indices": _file_fingerprint(target_path),
        },
    }


def evaluate_gate3a(
    *,
    seed: int,
    folds: Sequence[int],
    config_path: str | Path,
    loso_root: str | Path,
    output_root: str | Path,
    minclean_voxel_root: str | Path,
    minclean_parcel_root: str | Path,
) -> dict:
    """Evaluate frozen Gate-2 fold models with MINIMAL-cleanup REST alignment."""
    config_path = Path(config_path)
    config = load_config(config_path)
    fold_results = [
        _evaluate_fold(
            subject_id=int(subject_id),
            seed=int(seed),
            config=config,
            config_path=config_path,
            loso_root=loso_root,
            minclean_voxel_root=minclean_voxel_root,
            minclean_parcel_root=minclean_parcel_root,
        )
        for subject_id in folds
    ]
    decisions = cleanup_decision_fields(
        [float(result["delta"]) for result in fold_results]
    )
    summary = {
        "gate": "gate3a_cleanup_stress",
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
    summary_path = Path(output_root) / f"seed{int(seed)}" / "gate3a_summary.json"
    _write_json(summary_path, summary)
    print(
        "SENSITIVITY_BOUND "
        f"seed={int(seed)} mean_delta={summary['mean_delta']:.10f} "
        f"flag_first_order={summary['flag_first_order']}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        choices=("prepare", "evaluate", "all"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=_parse_folds, default="1,2,3,4,5,6")
    parser.add_argument("--config", default="config_voxel_contract.yaml")
    parser.add_argument("--loso-root", default="artifacts/voxel_contract_loso")
    parser.add_argument("--output-root", default="artifacts/gate3a")
    parser.add_argument("--minclean-voxel-root", default="data/processed_gate3a")
    parser.add_argument(
        "--minclean-parcel-root",
        default="data/processed_gate3a_schaefer400",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command in {"prepare", "all"}:
        prepare_gate3a(
            folds=args.folds,
            config_path=args.config,
            output_root=args.output_root,
            minclean_voxel_root=args.minclean_voxel_root,
            minclean_parcel_root=args.minclean_parcel_root,
            force=args.force,
        )
    if args.command in {"evaluate", "all"}:
        evaluate_gate3a(
            seed=args.seed,
            folds=args.folds,
            config_path=args.config,
            loso_root=args.loso_root,
            output_root=args.output_root,
            minclean_voxel_root=args.minclean_voxel_root,
            minclean_parcel_root=args.minclean_parcel_root,
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
