"""Zero-shot and few-shot inference for the Stage-1 multi-expert model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.data.nsd_loader import NSDSubjectData
from src.data.region_registry import RegionRegistry, load_subject_region_groups
from src.models.multiexpert_training import FittedMultiExpertEncoder
from src.models.multiexpert_training import FusionOptimizationConfig
from src.multiexpert_config import (
    canonical_config_hash,
    load_multiexpert_config,
    resolve_data_roots,
)
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.multiexpert_artifacts import (
    directory_file_fingerprints,
    file_sha256,
    load_and_validate_model_manifest,
)
from src.pipelines.multiexpert_evaluation import (
    compute_subject_noise_ceiling,
    evaluate_voxel_prediction,
)
from src.pipelines.multiexpert_support import (
    ExternalSeedSpec,
    alignment_expert_fingerprints,
    align_new_subject,
    fusion_architecture_config,
    load_alignment_experts,
    validate_subject_data,
)


logger = logging.getLogger(__name__)


def _require_gate_for_locked_subject(
    subject: int,
    config: dict,
    gate_path: str | Path | None,
) -> None:
    if int(subject) not in {int(value) for value in config["subjects"]["locked_test"]}:
        return
    path = Path(gate_path or (Path(config["output_root"]) / "loso" / "gate.json"))
    if not path.exists():
        raise PermissionError(
            f"Subject {subject} is locked until the LOSO gate passes; missing {path}."
        )
    gate = json.loads(path.read_text())
    if gate.get("passed") is not True:
        raise PermissionError(
            f"Subject {subject} remains locked because the LOSO gate did not pass."
        )
    if gate.get("config_hash") != canonical_config_hash(config):
        raise PermissionError("Subject-unlock gate was produced by a different config.")
    if gate.get("complete") is not True or not all(
        gate.get("criteria", {}).get(name) is True
        for name in ("performance", "subject_wins", "reliability")
    ):
        raise PermissionError("Subject-unlock gate is incomplete or has failed criteria.")
    if gate.get("artifact_verification_complete") is not True:
        raise PermissionError("Subject-unlock gate has not verified all fold artifacts.")
    expected_subjects = sorted(int(value) for value in config["subjects"]["loso"])
    if sorted(int(value) for value in gate.get("subjects", [])) != expected_subjects:
        raise PermissionError("Subject-unlock gate does not cover every LOSO subject.")
    expected_seeds = sorted(int(value) for value in config["evaluation"]["robustness_seeds"])
    if sorted(int(value) for value in gate.get("seeds", [])) != expected_seeds:
        raise PermissionError("Subject-unlock gate does not cover all configured robustness seeds.")
    expected_pairs = {
        (subject_id, seed)
        for subject_id in expected_subjects
        for seed in expected_seeds
    }
    verified_pairs = {
        (int(row["subject"]), int(row["seed"]))
        for row in gate.get("verified_fold_contracts", [])
        if str(row.get("contract_hash", ""))
    }
    if verified_pairs != expected_pairs:
        raise PermissionError("Subject-unlock gate lacks verified contracts for all folds.")


def select_fewshot_rows(
    *,
    subject_stimulus_ids: np.ndarray,
    shared_stimulus_ids: np.ndarray,
    eval_indices: np.ndarray,
    n_shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subject_stimulus_ids = np.asarray(subject_stimulus_ids, dtype=np.int64)
    shared_stimulus_ids = np.asarray(shared_stimulus_ids, dtype=np.int64)
    row_by_stimulus = {
        int(stimulus): int(row)
        for row, stimulus in enumerate(subject_stimulus_ids)
    }
    candidate_rows: list[int] = []
    candidate_template_rows: list[int] = []
    eval_set = set(np.asarray(eval_indices, dtype=np.int64).tolist())
    for template_row, stimulus in enumerate(shared_stimulus_ids):
        subject_row = row_by_stimulus.get(int(stimulus))
        if subject_row is not None and subject_row not in eval_set:
            candidate_rows.append(subject_row)
            candidate_template_rows.append(template_row)
    if len(candidate_rows) < int(n_shots):
        raise ValueError(
            f"Only {len(candidate_rows)} shared support stimuli remain after excluding "
            f"evaluation rows; requested {n_shots}."
        )
    rng = np.random.RandomState(int(seed))
    choice = np.sort(rng.choice(len(candidate_rows), size=int(n_shots), replace=False))
    return (
        np.asarray(candidate_rows, dtype=np.int64)[choice],
        np.asarray(candidate_template_rows, dtype=np.int64)[choice],
    )


def load_multiexpert_model(
    *,
    model_dir: str | Path,
    config: dict,
    expected_train_subjects: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict, ExternalSeedSpec, RegionRegistry, dict, FittedMultiExpertEncoder]:
    model_dir = Path(model_dir)
    seed_spec = ExternalSeedSpec.load(model_dir)
    expected_registry_subjects = tuple(
        int(value) for value in config["subjects"]["seed_registry"]
    )
    if tuple(seed_spec.registry_subjects) != expected_registry_subjects:
        raise ValueError(
            "External seed registry subjects do not match the effective config."
        )
    registry = RegionRegistry.load(model_dir / "region_registry.json")
    manifest = load_and_validate_model_manifest(
        model_dir,
        config=config,
        region_manifest=registry.to_manifest(),
        seed_manifest=seed_spec.contract,
        expected_train_subjects=(
            config["subjects"]["train"]
            if expected_train_subjects is None
            else expected_train_subjects
        ),
        expected_model_variant="learned_fusion_dropout",
    )
    actual_expert_files = directory_file_fingerprints(model_dir / "experts")
    if manifest.get("expert_artifact_fingerprints") != actual_expert_files:
        raise ValueError("Alignment expert files do not match the checksummed manifest.")
    shared_stimulus_path = model_dir / "shared_stim_idx.npy"
    if not shared_stimulus_path.exists() or manifest.get(
        "shared_stimulus_sha256"
    ) != file_sha256(shared_stimulus_path):
        raise ValueError("Shared-stimulus IDs do not match the checksummed manifest.")
    shared_stimulus_ids = np.load(shared_stimulus_path, mmap_mode="r")
    if (
        shared_stimulus_ids.ndim != 1
        or int(shared_stimulus_ids.size)
        != int(manifest.get("shared_stimulus_count", -1))
        or np.unique(shared_stimulus_ids).size != shared_stimulus_ids.size
        or np.any(shared_stimulus_ids < 0)
    ):
        raise ValueError("Saved shared-stimulus IDs are invalid or incomplete.")
    experts = load_alignment_experts(
        model_dir,
        expert_order=list(manifest["expert_order"]),
        expert_dims=manifest["expert_dims"],
        seed_manifest_fingerprint=seed_spec.fingerprint,
    )
    stored_expert_fingerprints = manifest.get("expert_state_fingerprints")
    actual_expert_fingerprints = alignment_expert_fingerprints(experts)
    if stored_expert_fingerprints != actual_expert_fingerprints:
        raise ValueError("Alignment expert state does not match the model manifest.")
    encoder = FittedMultiExpertEncoder.load(
        model_dir / "encoder",
        expected_expert_order=manifest["expert_order"],
        expected_expert_dims=manifest["expert_dims"],
    )
    encoder_dir = model_dir / "encoder"
    actual_encoder_files = {
        str(path.relative_to(encoder_dir)): file_sha256(path)
        for path in sorted(encoder_dir.rglob("*"))
        if path.is_file()
    }
    if manifest.get("encoder_artifact_fingerprints") != actual_encoder_files:
        raise ValueError("Encoder files do not match the checksummed model manifest.")
    if int(manifest.get("training_seed", -1)) != int(config["random_seed"]):
        raise ValueError("Model training seed does not match the effective config.")
    if encoder.network.config != fusion_architecture_config(config):
        raise ValueError("Encoder architecture does not match the effective config.")
    if encoder.optimization != FusionOptimizationConfig.from_pipeline_config(config):
        raise ValueError("Encoder optimization settings do not match the effective config.")
    if encoder.network.num_regions != registry.n_groups:
        raise ValueError("Encoder region count does not match the saved HCP-MMP registry.")
    if int(manifest.get("n_regions_including_fallback", -1)) != registry.n_groups:
        raise ValueError("Model manifest region count does not match the registry.")
    if int(manifest["input_dim"]) != encoder.network.input_dim:
        raise ValueError("Model manifest and encoder input dimensions differ.")
    encoder_slices = {
        name: [int(bounds[0]), int(bounds[1])]
        for name, bounds in encoder.network.feature_slices.items()
    }
    if manifest["feature_slices"] != encoder_slices:
        raise ValueError("Model manifest and encoder feature slices differ.")
    return manifest, seed_spec, registry, experts, encoder


def predict_multiexpert(
    *,
    test_subject: int,
    mode: str,
    config_path: str = "config_multiexpert.yaml",
    model_dir: str | None = None,
    data_root: str | None = None,
    raw_data_root: str | None = None,
    output_dir: str | None = None,
    n_shots: int | None = None,
    seed: int = 42,
    gate_path: str | None = None,
) -> dict:
    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError("mode must be 'zero_shot' or 'few_shot'.")
    config = resolve_data_roots(
        load_multiexpert_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    _require_gate_for_locked_subject(test_subject, config, gate_path)
    data_root = str(config["data_root"])
    raw_data_root = str(config["raw_data_root"])
    model_dir = str(model_dir or (Path(config["output_root"]) / "model"))
    output_dir = str(output_dir or (Path(config["output_root"]) / "predictions"))
    n_shots = int(n_shots or config["evaluation"]["fewshot_n"])

    manifest, seed_spec, registry, experts, encoder = load_multiexpert_model(
        model_dir=model_dir,
        config=config,
    )
    subject = NSDSubjectData(int(test_subject), data_root)
    validate_subject_data(subject)
    feature_path = Path(data_root) / "features" / "clip_features.npy"
    features = np.load(feature_path, mmap_mode="r")
    if int(features.shape[1]) != int(manifest["input_dim"]):
        raise ValueError("CLIP feature width does not match the trained model.")
    seed_availability = seed_spec.availability_for_subject(
        subject,
        raw_data_root=raw_data_root,
    )
    seed_runs = seed_spec.runs_for_subject(
        subject,
        data_root=data_root,
        raw_data_root=raw_data_root,
        allow_missing=True,
        availability=seed_availability,
    )
    eval_indices = fixed_eval_indices(
        n_shared=int(subject.test_stim_idx.size),
        eval_size=int(config["evaluation"]["fixed_eval_size"]),
        seed=int(config["evaluation"]["eval_split_seed"]),
    )
    shot_rows = None
    shot_template_rows = None
    if mode == "few_shot":
        shared_stimulus_ids = np.load(Path(model_dir) / "shared_stim_idx.npy")
        shot_rows, shot_template_rows = select_fewshot_rows(
            subject_stimulus_ids=subject.test_stim_idx,
            shared_stimulus_ids=shared_stimulus_ids,
            eval_indices=eval_indices,
            n_shots=n_shots,
            seed=seed,
        )
        task_fmri = np.asarray(subject.test_fmri[shot_rows], dtype=np.float32)
    else:
        task_fmri = None
    transforms = align_new_subject(
        experts,
        subject=subject,
        external_seed_runs=seed_runs,
        seed_manifest_fingerprint=seed_spec.fingerprint,
        task_fmri_shared=task_fmri,
        shot_indices=shot_template_rows,
    )
    voxel_groups = load_subject_region_groups(
        registry,
        subject.sub,
        data_root=data_root,
        raw_data_root=raw_data_root,
        require_registered=subject.sub in registry.training_subjects,
    )
    eval_stimulus_ids = subject.test_stim_idx[eval_indices]
    eval_features = np.asarray(features[eval_stimulus_ids], dtype=np.float32)
    truth = np.asarray(subject.test_fmri[eval_indices], dtype=np.float32)
    fused = encoder.predict_subject(
        eval_features,
        transforms=transforms,
        voxel_groups=voxel_groups,
    )
    equal = encoder.predict_subject(
        eval_features,
        transforms=transforms,
        voxel_groups=voxel_groups,
        equal_weights=True,
    )
    expert_only = {
        name: encoder.predict_subject(
            eval_features,
            transforms=transforms,
            voxel_groups=voxel_groups,
            active_experts=[name],
        )["fused"]
        for name in manifest["expert_order"]
    }
    predictions = {
        "learned_fusion": fused["fused"],
        "equal_fusion_weights": equal["fused"],
        **{f"expert_{name}": value for name, value in expert_only.items()},
    }
    reliability = list(config["evaluation"]["reliability_thresholds"])
    metrics: dict[str, dict] = {}
    voxel_corrs: dict[str, np.ndarray] = {}
    noise_ceiling = compute_subject_noise_ceiling(subject)
    for name, prediction in predictions.items():
        metrics[name], voxel_corrs[name] = evaluate_voxel_prediction(
            subject,
            truth,
            prediction,
            reliability_thresholds=reliability,
            precomputed_noise_ceiling=noise_ceiling,
        )

    tag = "zero_shot" if mode == "zero_shot" else f"fewshot_N{n_shots}_seed{seed}"
    destination = Path(output_dir) / f"subj{subject.sub:02d}" / tag
    destination.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        np.save(destination / f"{name}.npy", prediction)
        np.save(destination / f"{name}_voxel_corrs.npy", voxel_corrs[name])
    np.save(destination / "regional_weights.npy", fused["regional_weights"])
    np.save(destination / "eval_indices.npy", eval_indices)
    for name, transform in transforms.items():
        transform.save(destination / f"transform_{name}.npz")
    provenance = {
        "mode": mode,
        "subject": subject.sub,
        "model_dir": str(Path(model_dir).resolve()),
        "expert_order": list(manifest["expert_order"]),
        "eval_indices": eval_indices.tolist(),
        "eval_stimulus_ids": eval_stimulus_ids.tolist(),
        "n_shots": int(n_shots) if mode == "few_shot" else 0,
        "shot_rows": [] if shot_rows is None else shot_rows.tolist(),
        "shot_template_rows": (
            [] if shot_template_rows is None else shot_template_rows.tolist()
        ),
        "fewshot_seed": int(seed) if mode == "few_shot" else None,
        "seed_availability": {
            "total": int(seed_availability.size),
            "available": int(seed_availability.sum()),
            "missing": [
                seed_def.name
                for seed_def, keep in zip(
                    seed_spec.seed_defs,
                    seed_availability,
                    strict=True,
                )
                if not keep
            ],
            "policy": seed_spec.missing_subject_policy,
        },
        "metrics": metrics,
    }
    (destination / "metrics.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Saved %s subject-%d predictions to %s", mode, subject.sub, destination)
    return {
        "predictions": predictions,
        "regional_weights": fused["regional_weights"],
        "metrics": metrics,
        "destination": str(destination),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], required=True)
    parser.add_argument("--test-sub", type=int, required=True)
    parser.add_argument("--config", default="config_multiexpert.yaml")
    parser.add_argument("--model-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-shots", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-path")
    args = parser.parse_args()
    predict_multiexpert(
        test_subject=args.test_sub,
        mode=args.mode,
        config_path=args.config,
        model_dir=args.model_dir,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        output_dir=args.output_dir,
        n_shots=args.n_shots,
        seed=args.seed,
        gate_path=args.gate_path,
    )
