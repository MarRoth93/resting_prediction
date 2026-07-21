"""Leave-one-subject-out evaluation and the subject-7 unlock gate."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from src.alignment.external_seed_bank import SEED_ATLAS_FILES, roi_dir
from src.config import load_config
from src.data.nsd_loader import NSDSubjectData
from src.data.region_registry import (
    build_hcp_mmp_region_registry,
    load_subject_region_groups,
)
from src.models.encoding_factory import build_encoder
from src.models.nonlinear_encoding import StaticTransformerConfig, StaticTransformerEncoder
from src.models.multiexpert_training import (
    FittedMultiExpertEncoder,
    FusionOptimizationConfig,
)
from src.multiexpert_config import (
    canonical_config_hash,
    load_multiexpert_config,
    resolve_data_roots,
)
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_evaluation import (
    compute_subject_noise_ceiling,
    evaluate_voxel_prediction,
    threshold_key,
    validate_repeated_trial_contract,
)
from src.pipelines.multiexpert_artifacts import (
    cached_input_file_manifest,
    file_sha256,
    json_fingerprint,
)
from src.pipelines.multiexpert_support import (
    ExternalSeedSpec,
    alignment_expert_fingerprints,
    align_new_subject,
    build_external_seed_spec,
    fusion_architecture_config,
    validate_subject_data,
)
from src.pipelines.predict_multiexpert import load_multiexpert_model, select_fewshot_rows
from src.pipelines.train_multiexpert import (
    PreparedMultiExpertTraining,
    fit_prepared_fusion,
    prepare_multiexpert_training,
    save_multiexpert_model,
)


logger = logging.getLogger(__name__)


METHODS = (
    "current_only",
    "connectivity_srm_only",
    "equal_average",
    "learned_fusion_no_dropout",
    "learned_fusion_dropout",
)

RESULT_BASE_CONTRACT_KEYS = (
    "experiment_config_hash",
    "effective_config_hash",
    "frozen_baseline_config_fingerprint",
    "heldout_subject",
    "train_subjects",
    "seed",
    "seed_manifest_fingerprint",
    "region_registry_fingerprint",
    "expert_order",
    "input_data_fingerprint",
)


def _validate_fusion_encoder_config(
    encoder: FittedMultiExpertEncoder,
    config: dict,
) -> None:
    expected_architecture = asdict(fusion_architecture_config(config))
    if asdict(encoder.network.config) != expected_architecture:
        raise ValueError("Resumed fusion encoder architecture/config does not match this fold.")
    expected_optimization = asdict(
        FusionOptimizationConfig.from_pipeline_config(config)
    )
    if asdict(encoder.optimization) != expected_optimization:
        raise ValueError("Resumed fusion optimization config does not match this fold.")


def _validate_static_encoder_config(
    encoder: StaticTransformerEncoder,
    *,
    seed: int,
    frozen_config_path: str,
    expected_output_dim: int,
    expected_feature_slices: dict[str, tuple[int, int]],
) -> None:
    frozen = load_config(frozen_config_path)
    expected = StaticTransformerConfig.from_config(
        frozen["encoding"],
        seed=int(seed),
    )
    if encoder.config != expected:
        raise ValueError("Resumed single-expert encoder config does not match this fold.")
    if encoder.output_dim != int(expected_output_dim):
        raise ValueError("Resumed single-expert encoder latent dimension is stale.")
    if encoder.feature_slices != expected_feature_slices:
        raise ValueError("Resumed single-expert feature slices are stale.")


def _fold_base_contract(
    *,
    base_config: dict,
    effective_config: dict,
    heldout_subject: int,
    train_subjects: list[int],
    seed_spec: ExternalSeedSpec,
    region_fingerprint: str,
    frozen_config_path: str,
    input_data_fingerprint: str,
) -> dict:
    return {
        "schema_version": 1,
        "experiment_config_hash": canonical_config_hash(base_config),
        "effective_config_hash": canonical_config_hash(effective_config),
        "frozen_baseline_config_fingerprint": _frozen_baseline_config_fingerprint(
            frozen_config_path,
            seed=int(effective_config["random_seed"]),
        ),
        "heldout_subject": int(heldout_subject),
        "train_subjects": [int(value) for value in train_subjects],
        "seed": int(effective_config["random_seed"]),
        "seed_manifest_fingerprint": seed_spec.fingerprint,
        "region_registry_fingerprint": str(region_fingerprint),
        "expert_order": list(effective_config["experts"]["order"]),
        "input_data_fingerprint": str(input_data_fingerprint),
    }


def _frozen_baseline_config_fingerprint(
    frozen_config_path: str,
    *,
    seed: int,
) -> str:
    frozen = copy.deepcopy(load_config(frozen_config_path))
    frozen["random_seed"] = int(seed)
    return json_fingerprint(frozen)


def _fold_artifact_fingerprints(fold_dir: Path) -> dict[str, str]:
    roots = (
        fold_dir / "model",
        fold_dir / "no_dropout_encoder",
        fold_dir / "current_baseline_encoder",
        fold_dir / "connectivity_srm_baseline_encoder",
        fold_dir / "predictions",
    )
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Completed fold is missing artifact directory: {root}")
        files.extend(path for path in root.rglob("*") if path.is_file())
    for name in ("no_dropout_effective_config.json", "input_manifest.json"):
        path = fold_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
    return {
        str(path.relative_to(fold_dir)): file_sha256(path)
        for path in sorted(files)
    }


def _validate_completed_fold(
    fold_dir: Path,
    result_path: Path,
    expected_base_contract: dict,
) -> dict:
    contract_path = fold_dir / "fold_contract.json"
    if not contract_path.exists():
        raise ValueError(
            f"Existing LOSO result has no fold contract and cannot be reused: {result_path}"
        )
    contract = json.loads(contract_path.read_text())
    if contract.get("base") != expected_base_contract:
        raise ValueError(f"Existing LOSO fold contract is stale: {contract_path}")
    input_manifest_path = fold_dir / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    stored_input_fingerprint = input_manifest.pop("fingerprint", None)
    if (
        stored_input_fingerprint != json_fingerprint(input_manifest)
        or stored_input_fingerprint
        != expected_base_contract.get("input_data_fingerprint")
    ):
        raise ValueError(
            f"Existing LOSO input manifest is stale or invalid: {input_manifest_path}"
        )
    actual_artifacts = _fold_artifact_fingerprints(fold_dir)
    if contract.get("artifacts") != actual_artifacts:
        raise ValueError(f"Existing LOSO artifacts were modified or are incomplete: {fold_dir}")
    result = json.loads(result_path.read_text())
    for key in RESULT_BASE_CONTRACT_KEYS:
        if result.get(key) != expected_base_contract.get(key):
            raise ValueError(
                f"Existing LOSO result field {key!r} does not match its fold contract: "
                f"{result_path}"
            )
    result_payload = dict(result)
    result_payload.pop("fold_contract_hash", None)
    actual_result_fingerprint = json_fingerprint(result_payload)
    if contract.get("result_payload_fingerprint") != actual_result_fingerprint:
        raise ValueError(f"Existing LOSO result payload was modified: {result_path}")
    expected_hash = json_fingerprint(
        {
            "base": expected_base_contract,
            "artifacts": actual_artifacts,
            "result_payload_fingerprint": actual_result_fingerprint,
        }
    )
    if contract.get("contract_hash") != expected_hash:
        raise ValueError(f"Existing LOSO contract checksum is invalid: {contract_path}")
    if result.get("fold_contract_hash") != expected_hash:
        raise ValueError(f"Existing LOSO result does not match its artifacts: {result_path}")
    if (
        int(result.get("heldout_subject", -1)) != expected_base_contract["heldout_subject"]
        or int(result.get("seed", -1)) != expected_base_contract["seed"]
        or result.get("train_subjects") != expected_base_contract["train_subjects"]
    ):
        raise ValueError(f"Existing LOSO result identity is stale: {result_path}")
    return result


def _validate_result_contract_metadata(result_path: Path) -> dict:
    """Validate result-to-contract binding without rehashing large model files."""
    contract_path = result_path.parent / "fold_contract.json"
    if not contract_path.exists():
        raise ValueError(f"LOSO result has no fold contract: {result_path}")
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    base = contract.get("base", {})
    for key in RESULT_BASE_CONTRACT_KEYS:
        if result.get(key) != base.get(key):
            raise ValueError(
                f"LOSO result field {key!r} does not match its fold contract: "
                f"{result_path}"
            )
    payload = dict(result)
    payload.pop("fold_contract_hash", None)
    payload_fingerprint = json_fingerprint(payload)
    if contract.get("result_payload_fingerprint") != payload_fingerprint:
        raise ValueError(f"LOSO result payload was modified: {result_path}")
    expected_hash = json_fingerprint(
        {
            "base": contract.get("base"),
            "artifacts": contract.get("artifacts"),
            "result_payload_fingerprint": payload_fingerprint,
        }
    )
    if (
        contract.get("contract_hash") != expected_hash
        or result.get("fold_contract_hash") != expected_hash
    ):
        raise ValueError(f"LOSO result contract checksum is invalid: {result_path}")
    return result


def _require_trial_reliability_data(data_root: str, subjects: Iterable[int]) -> None:
    missing: list[str] = []
    for subject in subjects:
        subject_dir = Path(data_root) / f"subj{int(subject):02d}"
        for name in ("test_fmri_trials.npy", "test_trial_labels.npy"):
            path = subject_dir / name
            if not path.exists():
                missing.append(str(path))
    if missing:
        preview = "\n  ".join(missing)
        raise FileNotFoundError(
            "The Stage-1 gate requires trial-level reliability for every LOSO subject. "
            "Prepare it before training LOSO folds. Missing:\n  " + preview
        )
    for subject in subjects:
        subject_dir = Path(data_root) / f"subj{int(subject):02d}"
        trial_fmri = np.load(subject_dir / "test_fmri_trials.npy", mmap_mode="r")
        labels = np.load(subject_dir / "test_trial_labels.npy", mmap_mode="r")
        averaged = np.load(subject_dir / "test_fmri.npy", mmap_mode="r")
        validate_repeated_trial_contract(
            trial_fmri,
            labels,
            expected_voxels=int(averaged.shape[1]),
        )


def _prepare_fold_input_manifest(
    config: dict,
    *,
    subjects: Iterable[int],
    heldout_subject: int,
    seed_spec: ExternalSeedSpec,
    data_root: str,
    raw_data_root: str,
    loso_root: str | Path,
    ensure_seed_caches: bool,
) -> dict:
    """Hash every processed/raw artifact that can affect one LOSO fold."""
    files: dict[str, Path] = {
        "features/clip_features.npy": Path(data_root) / "features" / "clip_features.npy",
    }
    for subject_id in sorted({int(value) for value in subjects}):
        subject = NSDSubjectData(subject_id, data_root)
        validate_subject_data(subject)
        tag = f"subj{subject_id:02d}"
        subject_dir = Path(data_root) / tag
        for name in (
            "mask.npy",
            "train_fmri.npy",
            "train_stim_idx.npy",
            "test_fmri.npy",
            "test_stim_idx.npy",
            "test_fmri_trials.npy",
            "test_trial_labels.npy",
            "rest_run_manifest.json",
        ):
            files[f"processed/{tag}/{name}"] = subject_dir / name
        for run_index in range(1, len(subject.rest_runs) + 1):
            name = f"rest_run{run_index}.npy"
            files[f"processed/{tag}/{name}"] = subject_dir / name

        availability = seed_spec.availability_for_subject(
            subject,
            raw_data_root=raw_data_root,
        )
        allow_missing = subject_id == int(heldout_subject)
        if ensure_seed_caches:
            seed_spec.runs_for_subject(
                subject,
                data_root=data_root,
                raw_data_root=raw_data_root,
                force_recompute=bool(
                    config["external_seed_bank"]["force_recompute"]
                ),
                allow_missing=allow_missing,
                availability=availability,
            )
        elif not allow_missing and not np.all(availability):
            raise ValueError(
                f"Training subject {subject_id} is missing a fold seed definition."
            )
        cache_dir = seed_spec.cache_dir_for_subject(
            subject,
            data_root=data_root,
            availability=availability,
        )
        files[f"seed_cache/{tag}/manifest.json"] = cache_dir / "manifest.json"
        for run_index in range(1, len(subject.rest_runs) + 1):
            name = f"external_seed_run{run_index}.npy"
            files[f"seed_cache/{tag}/{name}"] = cache_dir / name

        raw_roi_dir = roi_dir(raw_data_root, subject_id)
        for atlas_file in SEED_ATLAS_FILES:
            files[f"raw_roi/{tag}/{atlas_file}"] = raw_roi_dir / atlas_file

    return cached_input_file_manifest(
        files,
        cache_path=Path(loso_root) / "input_hash_cache.json",
    )


def _train_single_expert_baseline(
    prepared: PreparedMultiExpertTraining,
    *,
    expert_name: str,
    frozen_config_path: str,
) -> object:
    """Train a direct single-expert encoder with the frozen current architecture."""
    if expert_name not in prepared.experts:
        raise ValueError(f"Unknown single-expert baseline: {expert_name!r}.")
    config = copy.deepcopy(load_config(frozen_config_path))
    config["random_seed"] = int(prepared.config["random_seed"])
    config["encoding"]["seed"] = int(prepared.config["random_seed"])
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for subject in prepared.train_subjects:
        view = prepared.views[subject]
        x_rows.append(
            np.asarray(
                prepared.feature_matrix[view.stimulus_ids],
                dtype=np.float32,
            )
        )
        z_rows.append(view.latent_targets[expert_name])
        groups.append(np.asarray(view.stimulus_ids, dtype=np.int64))
    X = np.concatenate(x_rows, axis=0)
    Z = np.concatenate(z_rows, axis=0)
    sample_groups = np.concatenate(groups, axis=0)
    encoder = build_encoder(
        config,
        input_dim=int(X.shape[1]),
        output_dim=int(Z.shape[1]),
        feature_slices=prepared.feature_slices,
    )
    encoder.fit(X, Z, sample_groups=sample_groups)
    return encoder


def _fold_predictions(
    *,
    prepared: PreparedMultiExpertTraining,
    heldout_subject: int,
    dropout_encoder: FittedMultiExpertEncoder,
    no_dropout_encoder: FittedMultiExpertEncoder,
    current_encoder,
    csrm_encoder,
    data_root: str,
    raw_data_root: str,
    fold_dir: Path,
) -> dict:
    config = prepared.config
    subject = NSDSubjectData(heldout_subject, data_root)
    validate_subject_data(subject)
    seed_availability = prepared.seed_spec.availability_for_subject(
        subject,
        raw_data_root=raw_data_root,
    )
    seed_runs = prepared.seed_spec.runs_for_subject(
        subject,
        data_root=data_root,
        raw_data_root=raw_data_root,
        allow_missing=True,
        availability=seed_availability,
    )
    voxel_groups = load_subject_region_groups(
        prepared.region_registry,
        heldout_subject,
        data_root=data_root,
        raw_data_root=raw_data_root,
        require_registered=False,
    )
    eval_indices = fixed_eval_indices(
        n_shared=int(subject.test_stim_idx.size),
        eval_size=int(config["evaluation"]["fixed_eval_size"]),
        seed=int(config["evaluation"]["eval_split_seed"]),
    )
    eval_stimuli = subject.test_stim_idx[eval_indices]
    eval_features = np.asarray(
        prepared.feature_matrix[eval_stimuli], dtype=np.float32
    )
    truth = np.asarray(subject.test_fmri[eval_indices], dtype=np.float32)
    n_shots = int(config["evaluation"]["fewshot_n"])
    shot_rows, template_rows = select_fewshot_rows(
        subject_stimulus_ids=subject.test_stim_idx,
        shared_stimulus_ids=prepared.shared_stimulus_ids,
        eval_indices=eval_indices,
        n_shots=n_shots,
        seed=int(config["random_seed"]),
    )
    reliability = list(config["evaluation"]["reliability_thresholds"])
    noise_ceiling = compute_subject_noise_ceiling(subject)
    result = {
        "heldout_subject": int(heldout_subject),
        "seed": int(config["random_seed"]),
        "train_subjects": list(prepared.train_subjects),
        "eval_indices": eval_indices.tolist(),
        "fewshot_rows": shot_rows.tolist(),
        "fewshot_template_rows": template_rows.tolist(),
        "heldout_seed_availability": {
            "total": int(seed_availability.size),
            "available": int(seed_availability.sum()),
            "missing": [
                seed.name
                for seed, keep in zip(
                    prepared.seed_spec.seed_defs,
                    seed_availability,
                    strict=True,
                )
                if not keep
            ],
            "policy": prepared.seed_spec.missing_subject_policy,
        },
        "modes": {},
    }

    for mode in ("zero_shot", "few_shot"):
        task_fmri = (
            None
            if mode == "zero_shot"
            else np.asarray(subject.test_fmri[shot_rows], dtype=np.float32)
        )
        transforms = align_new_subject(
            prepared.experts,
            subject=subject,
            external_seed_runs=seed_runs,
            seed_manifest_fingerprint=prepared.seed_spec.fingerprint,
            task_fmri_shared=task_fmri,
            shot_indices=None if mode == "zero_shot" else template_rows,
        )
        dropout = dropout_encoder.predict_subject(
            eval_features,
            transforms=transforms,
            voxel_groups=voxel_groups,
        )
        no_dropout = no_dropout_encoder.predict_subject(
            eval_features,
            transforms=transforms,
            voxel_groups=voxel_groups,
        )
        csrm_latent = csrm_encoder.predict(eval_features)
        csrm_only = transforms["connectivity_srm"].reconstruct(csrm_latent)
        current_latent = current_encoder.predict(eval_features)
        current_only = transforms["hybrid_cha"].reconstruct(current_latent)
        equal_average = 0.5 * (current_only + csrm_only)
        predictions = {
            "current_only": current_only,
            "connectivity_srm_only": csrm_only,
            "equal_average": equal_average,
            "learned_fusion_no_dropout": no_dropout["fused"],
            "learned_fusion_dropout": dropout["fused"],
        }
        mode_dir = fold_dir / "predictions" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_metrics: dict[str, dict] = {}
        for method in METHODS:
            np.save(mode_dir / f"{method}.npy", predictions[method])
            metrics, correlations = evaluate_voxel_prediction(
                subject,
                truth,
                predictions[method],
                reliability_thresholds=reliability,
                precomputed_noise_ceiling=noise_ceiling,
            )
            np.save(mode_dir / f"{method}_voxel_corrs.npy", correlations)
            mode_metrics[method] = metrics
        np.save(mode_dir / "learned_fusion_dropout_weights.npy", dropout["regional_weights"])
        np.save(
            mode_dir / "learned_fusion_no_dropout_weights.npy",
            no_dropout["regional_weights"],
        )
        result["modes"][mode] = mode_metrics
    return result


def run_loso_fold(
    *,
    base_config: dict,
    heldout_subject: int,
    seed: int,
    data_root: str,
    raw_data_root: str,
    loso_root: str | Path,
    frozen_config_path: str = "config.yaml",
    seed_spec: ExternalSeedSpec | None = None,
) -> dict:
    config = copy.deepcopy(base_config)
    config["random_seed"] = int(seed)
    train_subjects = [
        int(subject)
        for subject in config["subjects"]["loso"]
        if int(subject) != int(heldout_subject)
    ]
    # The held-out subject must not decide which connectivity seeds exist.
    config["subjects"]["seed_registry"] = list(train_subjects)
    fold_dir = Path(loso_root) / f"seed{int(seed)}" / f"fold_sub{int(heldout_subject):02d}"
    result_path = fold_dir / "result.json"
    if seed_spec is None:
        seed_spec = build_external_seed_spec(
            config,
            data_root=data_root,
            raw_data_root=raw_data_root,
            registry_subjects=train_subjects,
        )
    elif list(seed_spec.registry_subjects) != train_subjects:
        raise ValueError(
            "LOSO seed registry must contain exactly the five training subjects."
        )
    input_manifest = _prepare_fold_input_manifest(
        config,
        subjects=config["subjects"]["loso"],
        heldout_subject=int(heldout_subject),
        seed_spec=seed_spec,
        data_root=data_root,
        raw_data_root=raw_data_root,
        loso_root=loso_root,
        ensure_seed_caches=not result_path.exists(),
    )
    expected_registry = build_hcp_mmp_region_registry(
        train_subjects,
        data_root=data_root,
        raw_data_root=raw_data_root,
        min_voxels_per_subject=int(
            config["regions"]["min_voxels_every_training_subject"]
        ),
    )
    base_contract = _fold_base_contract(
        base_config=base_config,
        effective_config=config,
        heldout_subject=heldout_subject,
        train_subjects=train_subjects,
        seed_spec=seed_spec,
        region_fingerprint=expected_registry.fingerprint,
        frozen_config_path=frozen_config_path,
        input_data_fingerprint=input_manifest["fingerprint"],
    )
    if result_path.exists():
        logger.info("Validating completed LOSO fold before reuse: %s", result_path)
        return _validate_completed_fold(fold_dir, result_path, base_contract)

    prepared = prepare_multiexpert_training(
        config,
        train_subjects=train_subjects,
        data_root=data_root,
        raw_data_root=raw_data_root,
        seed_spec=seed_spec,
    )
    if prepared.region_registry.fingerprint != expected_registry.fingerprint:
        raise RuntimeError("Prepared fold region registry changed during construction.")
    dropout_model_dir = fold_dir / "model"
    if (dropout_model_dir / "manifest.json").exists():
        logger.info("Resuming from completed dropout model: %s", dropout_model_dir)
        loaded_manifest, _, _, loaded_experts, dropout_encoder = load_multiexpert_model(
            model_dir=dropout_model_dir,
            config=config,
            expected_train_subjects=train_subjects,
        )
        if loaded_manifest.get("input_data_fingerprint") != input_manifest["fingerprint"]:
            raise ValueError("Resumed dropout model was trained from different input data.")
        if alignment_expert_fingerprints(loaded_experts) != alignment_expert_fingerprints(
            prepared.experts
        ):
            raise ValueError(
                "Saved dropout encoder and freshly prepared alignment transforms differ."
            )
    else:
        dropout_encoder = fit_prepared_fusion(prepared)
        save_multiexpert_model(
            prepared,
            dropout_encoder,
            output_dir=dropout_model_dir,
            model_variant="learned_fusion_dropout",
            input_data_fingerprint=input_manifest["fingerprint"],
        )
    _validate_fusion_encoder_config(dropout_encoder, config)

    no_dropout_config = copy.deepcopy(config)
    no_dropout_config["fusion"]["method_dropout"] = 0.0
    no_dropout_prepared = replace(prepared, config=no_dropout_config)
    no_dropout_dir = fold_dir / "no_dropout_encoder"
    # Ancillary encoders have no standalone alignment manifest. If a fold did
    # not reach result.json, retrain and overwrite their known files rather
    # than risk sealing a copied/stale same-shape checkpoint into a new fold.
    no_dropout_encoder = fit_prepared_fusion(no_dropout_prepared)
    no_dropout_encoder.save(no_dropout_dir)
    _validate_fusion_encoder_config(no_dropout_encoder, no_dropout_config)
    (fold_dir / "no_dropout_effective_config.json").write_text(
        json.dumps(no_dropout_config, indent=2, sort_keys=True) + "\n"
    )

    current_dir = fold_dir / "current_baseline_encoder"
    current_encoder = _train_single_expert_baseline(
        prepared,
        expert_name="hybrid_cha",
        frozen_config_path=frozen_config_path,
    )
    current_encoder.save(str(current_dir))
    _validate_static_encoder_config(
        current_encoder,
        seed=int(seed),
        frozen_config_path=frozen_config_path,
        expected_output_dim=prepared.expert_dims["hybrid_cha"],
        expected_feature_slices=prepared.feature_slices,
    )
    csrm_dir = fold_dir / "connectivity_srm_baseline_encoder"
    csrm_encoder = _train_single_expert_baseline(
        prepared,
        expert_name="connectivity_srm",
        frozen_config_path=frozen_config_path,
    )
    csrm_encoder.save(str(csrm_dir))
    _validate_static_encoder_config(
        csrm_encoder,
        seed=int(seed),
        frozen_config_path=frozen_config_path,
        expected_output_dim=prepared.expert_dims["connectivity_srm"],
        expected_feature_slices=prepared.feature_slices,
    )
    result = _fold_predictions(
        prepared=prepared,
        heldout_subject=int(heldout_subject),
        dropout_encoder=dropout_encoder,
        no_dropout_encoder=no_dropout_encoder,
        current_encoder=current_encoder,
        csrm_encoder=csrm_encoder,
        data_root=data_root,
        raw_data_root=raw_data_root,
        fold_dir=fold_dir,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n"
    )
    for key in RESULT_BASE_CONTRACT_KEYS:
        result[key] = base_contract[key]
    artifact_fingerprints = _fold_artifact_fingerprints(fold_dir)
    result_payload_fingerprint = json_fingerprint(result)
    contract_hash = json_fingerprint(
        {
            "base": base_contract,
            "artifacts": artifact_fingerprints,
            "result_payload_fingerprint": result_payload_fingerprint,
        }
    )
    fold_contract = {
        "base": base_contract,
        "artifacts": artifact_fingerprints,
        "result_payload_fingerprint": result_payload_fingerprint,
        "contract_hash": contract_hash,
    }
    (fold_dir / "fold_contract.json").write_text(
        json.dumps(fold_contract, indent=2, sort_keys=True) + "\n"
    )
    result["fold_contract_hash"] = contract_hash
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def collect_loso_results(
    loso_root: str | Path,
    *,
    expected_config_hash: str | None = None,
) -> list[dict]:
    results = []
    for path in sorted(Path(loso_root).glob("seed*/fold_sub*/result.json")):
        result = _validate_result_contract_metadata(path)
        if (
            expected_config_hash is not None
            and result.get("experiment_config_hash") != expected_config_hash
        ):
            raise ValueError(f"Stale LOSO result belongs to another config: {path}")
        results.append(result)
    return results


def compute_loso_gate(
    results: list[dict],
    config: dict,
    *,
    require_provenance: bool = False,
    frozen_config_path: str = "config.yaml",
) -> dict:
    subjects = sorted(int(value) for value in config["subjects"]["loso"])
    seeds = sorted(int(value) for value in config["evaluation"]["robustness_seeds"])
    gate_cfg = config["evaluation"]["gate"]
    mode = str(gate_cfg["mode"])
    expected = {(subject, seed) for subject in subjects for seed in seeds}
    keyed: dict[tuple[int, int], dict] = {}
    experiment_hash = canonical_config_hash(config)
    for result in results:
        key = (int(result["heldout_subject"]), int(result["seed"]))
        if key in keyed:
            raise ValueError(f"Duplicate LOSO result for subject/seed {key}.")
        if key not in expected:
            raise ValueError(f"Unexpected LOSO result for subject/seed {key}.")
        if require_provenance:
            effective = copy.deepcopy(config)
            effective["random_seed"] = key[1]
            effective["subjects"]["seed_registry"] = [
                int(subject)
                for subject in config["subjects"]["loso"]
                if int(subject) != key[0]
            ]
            if result.get("experiment_config_hash") != experiment_hash:
                raise ValueError(f"LOSO result {key} has a stale experiment config hash.")
            if result.get("effective_config_hash") != canonical_config_hash(effective):
                raise ValueError(f"LOSO result {key} has a stale seeded config hash.")
            expected_frozen = _frozen_baseline_config_fingerprint(
                frozen_config_path,
                seed=key[1],
            )
            if result.get("frozen_baseline_config_fingerprint") != expected_frozen:
                raise ValueError(
                    f"LOSO result {key} uses a stale frozen baseline config."
                )
            if not str(result.get("input_data_fingerprint", "")):
                raise ValueError(f"LOSO result {key} has no input-data fingerprint.")
            if not str(result.get("fold_contract_hash", "")):
                raise ValueError(f"LOSO result {key} has no verified fold contract.")
        keyed[key] = result
    missing = sorted(expected - set(keyed))
    reliable_key = f"median_r_nc_ge_{threshold_key(gate_cfg['reliability_threshold'])}"
    deltas: list[float] = []
    reliable_deltas: list[float] = []
    subject_deltas: dict[str, float] = {}
    reliable_missing: list[list[int]] = []
    for subject in subjects:
        per_subject = []
        for seed in seeds:
            record = keyed.get((subject, seed))
            if record is None:
                continue
            methods = record["modes"][mode]
            candidate = methods["learned_fusion_dropout"]
            baseline = methods["current_only"]
            delta = float(candidate["median_r"] - baseline["median_r"])
            deltas.append(delta)
            per_subject.append(delta)
            if reliable_key in candidate and reliable_key in baseline:
                reliable_deltas.append(
                    float(candidate[reliable_key] - baseline[reliable_key])
                )
            else:
                reliable_missing.append([subject, seed])
        if per_subject:
            subject_deltas[str(subject)] = float(np.mean(per_subject))

    mean_delta = float(np.mean(deltas)) if deltas else None
    reliable_mean_delta = (
        float(np.mean(reliable_deltas)) if reliable_deltas else None
    )
    subject_wins = int(sum(delta > 0 for delta in subject_deltas.values()))
    complete = not missing
    reliability_complete = not reliable_missing and len(reliable_deltas) == len(expected)
    performance_ok = complete and mean_delta is not None and mean_delta >= float(
        gate_cfg["minimum_mean_delta_median_r"]
    )
    wins_ok = complete and subject_wins >= int(gate_cfg["minimum_subject_wins"])
    reliability_ok = reliability_complete and reliable_mean_delta is not None and reliable_mean_delta >= -float(
        gate_cfg["maximum_reliable_voxel_regression"]
    )
    passed = bool(performance_ok and wins_ok and reliability_ok)
    return {
        "passed": passed,
        "config_hash": canonical_config_hash(config),
        "mode": mode,
        "candidate": "learned_fusion_dropout",
        "baseline": "current_only",
        "subjects": subjects,
        "seeds": seeds,
        "complete": complete,
        "missing_subject_seed_pairs": [list(pair) for pair in missing],
        "mean_delta_median_r": mean_delta,
        "minimum_mean_delta_median_r": float(
            gate_cfg["minimum_mean_delta_median_r"]
        ),
        "subject_mean_deltas": subject_deltas,
        "subject_wins": subject_wins,
        "minimum_subject_wins": int(gate_cfg["minimum_subject_wins"]),
        "reliable_metric": reliable_key,
        "reliability_complete": reliability_complete,
        "missing_reliability_pairs": reliable_missing,
        "mean_reliable_delta": reliable_mean_delta,
        "minimum_allowed_reliable_delta": -float(
            gate_cfg["maximum_reliable_voxel_regression"]
        ),
        "criteria": {
            "performance": bool(performance_ok),
            "subject_wins": bool(wins_ok),
            "reliability": bool(reliability_ok),
        },
        "verified_fold_contracts": [
            {
                "subject": subject,
                "seed": seed,
                "contract_hash": keyed[(subject, seed)].get("fold_contract_hash"),
            }
            for subject, seed in sorted(keyed)
        ],
    }


def write_loso_summary(
    loso_root: str | Path,
    config: dict,
    *,
    verify_artifacts: bool = False,
    frozen_config_path: str = "config.yaml",
) -> dict:
    root = Path(loso_root)
    results = collect_loso_results(
        root,
        expected_config_hash=canonical_config_hash(config),
    )
    gate = compute_loso_gate(
        results,
        config,
        require_provenance=True,
        frozen_config_path=frozen_config_path,
    )
    if verify_artifacts:
        fold_contexts: dict[int, tuple[list[int], ExternalSeedSpec, str, str]] = {}
        for path in sorted(root.glob("seed*/fold_sub*/result.json")):
            result = json.loads(path.read_text())
            heldout = int(result["heldout_subject"])
            seed = int(result["seed"])
            if heldout not in fold_contexts:
                train_subjects = [
                    int(subject)
                    for subject in config["subjects"]["loso"]
                    if int(subject) != heldout
                ]
                context_config = copy.deepcopy(config)
                context_config["subjects"]["seed_registry"] = list(train_subjects)
                seed_spec = build_external_seed_spec(
                    context_config,
                    data_root=str(context_config["data_root"]),
                    raw_data_root=str(context_config["raw_data_root"]),
                    registry_subjects=train_subjects,
                )
                registry = build_hcp_mmp_region_registry(
                    train_subjects,
                    data_root=str(context_config["data_root"]),
                    raw_data_root=str(context_config["raw_data_root"]),
                    min_voxels_per_subject=int(
                        context_config["regions"][
                            "min_voxels_every_training_subject"
                        ]
                    ),
                )
                input_manifest = _prepare_fold_input_manifest(
                    context_config,
                    subjects=context_config["subjects"]["loso"],
                    heldout_subject=heldout,
                    seed_spec=seed_spec,
                    data_root=str(context_config["data_root"]),
                    raw_data_root=str(context_config["raw_data_root"]),
                    loso_root=root,
                    ensure_seed_caches=False,
                )
                fold_contexts[heldout] = (
                    train_subjects,
                    seed_spec,
                    registry.fingerprint,
                    input_manifest["fingerprint"],
                )
            train_subjects, seed_spec, region_fingerprint, input_fingerprint = (
                fold_contexts[heldout]
            )
            effective = copy.deepcopy(config)
            effective["random_seed"] = seed
            effective["subjects"]["seed_registry"] = list(train_subjects)
            expected_base = _fold_base_contract(
                base_config=config,
                effective_config=effective,
                heldout_subject=heldout,
                train_subjects=train_subjects,
                seed_spec=seed_spec,
                region_fingerprint=region_fingerprint,
                frozen_config_path=frozen_config_path,
                input_data_fingerprint=input_fingerprint,
            )
            _validate_completed_fold(path.parent, path, expected_base)
    gate["artifact_verification_complete"] = bool(verify_artifacts)
    if not verify_artifacts:
        gate["passed"] = False
    summary = {
        "n_completed_folds": len(results),
        "methods": list(METHODS),
        "gate": gate,
        "results": [
            {
                "subject": result["heldout_subject"],
                "seed": result["seed"],
                "metrics": result["modes"],
            }
            for result in results
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (root / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return summary


def run_loso(
    *,
    config_path: str = "config_multiexpert.yaml",
    data_root: str | None = None,
    raw_data_root: str | None = None,
    output_dir: str | None = None,
    subjects: list[int] | None = None,
    seeds: list[int] | None = None,
    frozen_config_path: str = "config.yaml",
) -> dict:
    config = resolve_data_roots(
        load_multiexpert_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    data_root = str(config["data_root"])
    raw_data_root = str(config["raw_data_root"])
    loso_root = str(output_dir or (Path(config["output_root"]) / "loso"))
    configured_subjects = [int(value) for value in config["subjects"]["loso"]]
    configured_seeds = [int(value) for value in config["evaluation"]["robustness_seeds"]]
    selected_subjects = configured_subjects if subjects is None else [int(v) for v in subjects]
    selected_seeds = configured_seeds if seeds is None else [int(v) for v in seeds]
    if not set(selected_subjects).issubset(configured_subjects):
        raise ValueError("Requested folds are outside configured LOSO subjects.")
    if not set(selected_seeds).issubset(configured_seeds):
        raise ValueError("Requested seeds are outside configured robustness seeds.")
    _require_trial_reliability_data(data_root, configured_subjects)
    for seed in selected_seeds:
        for heldout in selected_subjects:
            run_loso_fold(
                base_config=config,
                heldout_subject=heldout,
                seed=seed,
                data_root=data_root,
                raw_data_root=raw_data_root,
                loso_root=loso_root,
                frozen_config_path=frozen_config_path,
            )
            write_loso_summary(
                loso_root,
                config,
                verify_artifacts=False,
                frozen_config_path=frozen_config_path,
            )
    return write_loso_summary(
        loso_root,
        config,
        verify_artifacts=True,
        frozen_config_path=frozen_config_path,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_multiexpert.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--fold", type=int, action="append", dest="subjects")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--frozen-config", default="config.yaml")
    args = parser.parse_args()
    run_loso(
        config_path=args.config,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        output_dir=args.output_dir,
        subjects=args.subjects,
        seeds=args.seeds,
        frozen_config_path=args.frozen_config,
    )
