"""Train the experimental Stage-1 alignment-expert fusion model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.alignment.experts import AlignmentExpert
from src.data.multiexpert_batching import StimulusDisjointSplit, stimulus_disjoint_split
from src.data.nsd_loader import NSDSubjectData
from src.data.region_registry import (
    RegionRegistry,
    build_hcp_mmp_region_registry,
    load_subject_region_groups,
)
from src.models.multiexpert_training import (
    FittedMultiExpertEncoder,
    FusionOptimizationConfig,
    SubjectFusionData,
)
from src.multiexpert_config import load_multiexpert_config, resolve_data_roots
from src.pipelines.multiexpert_artifacts import (
    build_model_manifest,
    directory_file_fingerprints,
    file_sha256,
    save_model_manifest,
)
from src.pipelines.multiexpert_support import (
    ExternalSeedSpec,
    alignment_expert_fingerprints,
    build_alignment_experts,
    build_external_seed_spec,
    build_fusion_network,
    fit_alignment_experts,
    project_responses_chunked,
    set_global_seed,
    shared_task_intersection,
    training_transforms,
    validate_subject_data,
)


logger = logging.getLogger(__name__)


@dataclass
class PreparedMultiExpertTraining:
    config: dict
    train_subjects: tuple[int, ...]
    subjects: dict[int, NSDSubjectData]
    feature_matrix: np.ndarray
    feature_slices: dict[str, tuple[int, int]]
    seed_spec: ExternalSeedSpec
    experts: dict[str, AlignmentExpert]
    region_registry: RegionRegistry
    shared_stimulus_ids: np.ndarray
    split: StimulusDisjointSplit
    views: dict[int, SubjectFusionData]

    @property
    def expert_dims(self) -> dict[str, int]:
        return {
            name: int(expert.k_global)
            for name, expert in self.experts.items()
        }


def prepare_multiexpert_training(
    config: dict,
    *,
    train_subjects: list[int] | tuple[int, ...],
    data_root: str,
    raw_data_root: str,
    seed_spec: ExternalSeedSpec | None = None,
) -> PreparedMultiExpertTraining:
    """Fit alignment experts and materialize compact latent targets."""
    train_subjects = tuple(sorted(int(subject) for subject in train_subjects))
    if not train_subjects:
        raise ValueError("No training subjects supplied.")
    set_global_seed(int(config["random_seed"]))
    subjects = {
        subject: NSDSubjectData(subject, data_root)
        for subject in train_subjects
    }
    for subject in train_subjects:
        validate_subject_data(subjects[subject])

    feature_path = Path(data_root) / "features" / "clip_features.npy"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing CLIP feature matrix: {feature_path}")
    feature_matrix = np.load(feature_path, mmap_mode="r")
    if feature_matrix.ndim != 2:
        raise ValueError(f"CLIP feature matrix must be 2D, got {feature_matrix.shape}.")
    max_stimulus = max(
        int(np.max(subjects[subject].train_stim_idx))
        for subject in train_subjects
    )
    if max_stimulus >= int(feature_matrix.shape[0]):
        raise ValueError(
            f"Training stimulus {max_stimulus} is outside CLIP rows {feature_matrix.shape[0]}."
        )
    feature_slices = {"clip": (0, int(feature_matrix.shape[1]))}

    if seed_spec is None:
        seed_spec = build_external_seed_spec(
            config,
            data_root=data_root,
            raw_data_root=raw_data_root,
        )
    expected_seed_registry = tuple(
        int(value) for value in config["subjects"]["seed_registry"]
    )
    if tuple(seed_spec.registry_subjects) != expected_seed_registry:
        raise ValueError(
            "External seed specification was built from different registry subjects."
        )
    if set(expected_seed_registry) != set(train_subjects):
        raise ValueError(
            "The external seed registry must contain exactly the model's training subjects."
        )
    seed_cfg = config["external_seed_bank"]
    external_seed_runs = {
        subject: seed_spec.runs_for_subject(
            subjects[subject],
            data_root=data_root,
            raw_data_root=raw_data_root,
            force_recompute=bool(seed_cfg["force_recompute"]),
        )
        for subject in train_subjects
    }

    shared_stimulus_ids, shared_rows = shared_task_intersection(subjects)
    task_responses_shared = {
        subject: np.asarray(
            subjects[subject].test_fmri[shared_rows[subject]],
            dtype=np.float32,
        )
        for subject in train_subjects
    }
    experts = build_alignment_experts(
        config,
        seed_manifest_fingerprint=seed_spec.fingerprint,
    )
    fit_alignment_experts(
        experts,
        subjects=subjects,
        external_seed_runs=external_seed_runs,
        task_responses_shared=task_responses_shared,
    )
    del task_responses_shared

    regions_cfg = config["regions"]
    registry = build_hcp_mmp_region_registry(
        train_subjects,
        data_root=data_root,
        raw_data_root=raw_data_root,
        min_voxels_per_subject=int(
            regions_cfg["min_voxels_every_training_subject"]
        ),
    )
    split = stimulus_disjoint_split(
        {
            subject: subjects[subject].train_stim_idx
            for subject in train_subjects
        },
        val_fraction=float(config["fusion"]["val_fraction"]),
        seed=int(config["random_seed"]),
    )

    views: dict[int, SubjectFusionData] = {}
    for subject in train_subjects:
        transforms = training_transforms(experts, subject)
        latent_targets = {
            name: project_responses_chunked(
                transform,
                subjects[subject].train_fmri,
            )
            for name, transform in transforms.items()
        }
        subject_split = split.by_subject[subject]
        views[subject] = SubjectFusionData(
            subject_id=subject,
            stimulus_ids=subjects[subject].train_stim_idx,
            responses=subjects[subject].train_fmri,
            latent_targets=latent_targets,
            transforms=transforms,
            voxel_groups=load_subject_region_groups(
                registry,
                subject,
                data_root=data_root,
                raw_data_root=raw_data_root,
                require_registered=True,
            ),
            train_indices=subject_split.train_indices,
            val_indices=subject_split.val_indices,
        )

    return PreparedMultiExpertTraining(
        config=config,
        train_subjects=train_subjects,
        subjects=subjects,
        feature_matrix=feature_matrix,
        feature_slices=feature_slices,
        seed_spec=seed_spec,
        experts=experts,
        region_registry=registry,
        shared_stimulus_ids=shared_stimulus_ids,
        split=split,
        views=views,
    )


def fit_prepared_fusion(
    prepared: PreparedMultiExpertTraining,
) -> FittedMultiExpertEncoder:
    network = build_fusion_network(
        prepared.config,
        input_dim=int(prepared.feature_matrix.shape[1]),
        expert_dims=prepared.expert_dims,
        num_regions=prepared.region_registry.n_groups,
        feature_slices=prepared.feature_slices,
    )
    encoder = FittedMultiExpertEncoder(
        network,
        FusionOptimizationConfig.from_pipeline_config(prepared.config),
    )
    return encoder.fit(prepared.feature_matrix, prepared.views)


def save_multiexpert_model(
    prepared: PreparedMultiExpertTraining,
    encoder: FittedMultiExpertEncoder,
    *,
    output_dir: str | Path,
    model_variant: str,
    input_data_fingerprint: str | None = None,
) -> dict:
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing multi-expert model: {output_dir}"
        )
    (output_dir / "experts").mkdir(parents=True, exist_ok=True)
    for name, expert in prepared.experts.items():
        expert.save(output_dir / "experts" / name)
    encoder.save(output_dir / "encoder")
    prepared.region_registry.save(output_dir / "region_registry.json")
    prepared.seed_spec.save(output_dir)
    np.save(output_dir / "shared_stim_idx.npy", prepared.shared_stimulus_ids)
    (output_dir / "effective_config.json").write_text(
        json.dumps(prepared.config, indent=2, sort_keys=True) + "\n"
    )
    split_manifest = {
        "split_type": "global_stimulus_disjoint",
        "train_stimuli": prepared.split.train_stimuli.tolist(),
        "validation_stimuli": prepared.split.val_stimuli.tolist(),
        "rows": {
            str(subject): {
                "train": int(prepared.views[subject].train_indices.size),
                "validation": int(prepared.views[subject].val_indices.size),
            }
            for subject in prepared.train_subjects
        },
    }
    (output_dir / "training_split.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest = build_model_manifest(
        config=prepared.config,
        expert_dims=prepared.expert_dims,
        train_subjects=list(prepared.train_subjects),
        region_manifest=prepared.region_registry.to_manifest(),
        seed_manifest=prepared.seed_spec.contract,
        input_dim=int(prepared.feature_matrix.shape[1]),
        feature_slices=prepared.feature_slices,
        model_variant=model_variant,
    )
    manifest["training_seed"] = int(prepared.config["random_seed"])
    if input_data_fingerprint is not None:
        manifest["input_data_fingerprint"] = str(input_data_fingerprint)
    manifest["expert_state_fingerprints"] = alignment_expert_fingerprints(
        prepared.experts
    )
    manifest["expert_artifact_fingerprints"] = directory_file_fingerprints(
        output_dir / "experts"
    )
    manifest["n_regions_including_fallback"] = prepared.region_registry.n_groups
    manifest["encoder_artifact_fingerprints"] = {
        str(path.relative_to(output_dir / "encoder")): file_sha256(path)
        for path in sorted((output_dir / "encoder").rglob("*"))
        if path.is_file()
    }
    manifest["shared_stimulus_count"] = int(prepared.shared_stimulus_ids.size)
    manifest["shared_stimulus_sha256"] = file_sha256(
        output_dir / "shared_stim_idx.npy"
    )
    manifest["best_validation"] = encoder.validation_summary
    save_model_manifest(output_dir, manifest)
    return manifest


def train_multiexpert(
    *,
    config_path: str = "config_multiexpert.yaml",
    data_root: str | None = None,
    raw_data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    config = resolve_data_roots(
        load_multiexpert_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    data_root = str(config["data_root"])
    raw_data_root = str(config["raw_data_root"])
    output_dir = str(output_dir or (Path(config["output_root"]) / "model"))
    prepared = prepare_multiexpert_training(
        config,
        train_subjects=config["subjects"]["train"],
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    encoder = fit_prepared_fusion(prepared)
    return save_multiexpert_model(
        prepared,
        encoder,
        output_dir=output_dir,
        model_variant="learned_fusion_dropout",
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
    args = parser.parse_args()
    train_multiexpert(
        config_path=args.config,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        output_dir=args.output_dir,
    )
