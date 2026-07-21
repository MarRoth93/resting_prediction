"""Training and artifact contracts for the dedicated Schaefer-400 path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.alignment.connectivity_srm import ConnectivitySRMExpert
from src.alignment.experts import AlignmentExpert, HybridCHAExpert
from src.data.multiexpert_batching import StimulusDisjointSplit, stimulus_disjoint_split
from src.data.schaefer400 import (
    N_PARCELS,
    PARCEL_IDS,
    SchaeferNSDSubjectData,
    validate_schaefer_nsd_subject,
)
from src.models.multiexpert_training import (
    FittedMultiExpertEncoder,
    FusionOptimizationConfig,
    SubjectFusionData,
)
from src.pipelines.multiexpert_artifacts import (
    build_model_manifest,
    directory_file_fingerprints,
    file_sha256,
    json_fingerprint,
    load_and_validate_model_manifest,
    save_model_manifest,
)
from src.pipelines.multiexpert_support import (
    alignment_expert_fingerprints,
    build_fusion_network,
    fit_alignment_experts,
    fusion_architecture_config,
    load_alignment_experts,
    project_responses_chunked,
    set_global_seed,
    shared_task_intersection,
    training_transforms,
)


CONTRACT_VERSION = 1


@dataclass(frozen=True)
class Schaefer400ModelContract:
    """One authoritative definition of seed, target, and fusion parcel order."""

    training_subjects: tuple[int, ...]
    rest_preprocessing: dict
    atlas_name: str = "Schaefer2018_400Parcels_7Networks_order"
    seed_set: str = "schaefer400_ordered_parcels"
    missing_subject_policy: str = "zero_fill_and_mask"

    @property
    def seed_manifest(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "seed_set": self.seed_set,
            "atlas_name": self.atlas_name,
            "n_seeds": N_PARCELS,
            "parcel_ids": PARCEL_IDS.astype(int).tolist(),
            "training_subjects": [int(value) for value in self.training_subjects],
            "missing_subject_policy": self.missing_subject_policy,
            "rest_preprocessing": self.rest_preprocessing,
        }

    @property
    def region_manifest(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "atlas_name": self.atlas_name,
            "representation": "schaefer400_parcel_means",
            "grouping": "one_fusion_group_per_parcel",
            "n_groups": N_PARCELS,
            "parcel_ids": PARCEL_IDS.astype(int).tolist(),
            "training_subjects": [int(value) for value in self.training_subjects],
        }

    @property
    def seed_fingerprint(self) -> str:
        return json_fingerprint(self.seed_manifest)

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir) / "schaefer400_contract.json"
        payload = {
            "seed_manifest": self.seed_manifest,
            "region_manifest": self.region_manifest,
        }
        payload["fingerprint"] = json_fingerprint(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> "Schaefer400ModelContract":
        path = Path(model_dir) / "schaefer400_contract.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Schaefer model contract: {path}")
        payload = json.loads(path.read_text())
        stored_fingerprint = payload.pop("fingerprint", None)
        if stored_fingerprint != json_fingerprint(payload):
            raise ValueError("Schaefer model contract checksum is invalid.")
        seed = payload.get("seed_manifest", {})
        region = payload.get("region_manifest", {})
        contract = cls(
            training_subjects=tuple(int(value) for value in seed["training_subjects"]),
            rest_preprocessing=dict(seed["rest_preprocessing"]),
            atlas_name=str(seed["atlas_name"]),
            seed_set=str(seed["seed_set"]),
            missing_subject_policy=str(seed["missing_subject_policy"]),
        )
        if contract.seed_manifest != seed or contract.region_manifest != region:
            raise ValueError("Schaefer model contract is internally inconsistent.")
        return contract


@dataclass
class PreparedSchaefer400Training:
    config: dict
    subjects: dict[int, SchaeferNSDSubjectData]
    feature_matrix: np.ndarray
    feature_slices: dict[str, tuple[int, int]]
    contract: Schaefer400ModelContract
    experts: dict[str, AlignmentExpert]
    shared_stimulus_ids: np.ndarray
    split: StimulusDisjointSplit
    views: dict[int, SubjectFusionData]

    @property
    def train_subjects(self) -> tuple[int, ...]:
        return tuple(sorted(self.subjects))

    @property
    def expert_dims(self) -> dict[str, int]:
        return {name: int(expert.k_global) for name, expert in self.experts.items()}


def build_schaefer400_experts(
    config: dict,
    *,
    seed_manifest_fingerprint: str,
) -> dict[str, AlignmentExpert]:
    expert_cfg = config["experts"]
    common = {
        "n_components": int(expert_cfg["n_components"]),
        "min_k": int(expert_cfg["min_k"]),
        "ensemble_method": str(config["parcel_seed_bank"]["ensemble_method"]),
        "seed_manifest_fingerprint": seed_manifest_fingerprint,
    }
    experts: dict[str, AlignmentExpert] = {
        "hybrid_cha": HybridCHAExpert(
            **common,
            max_iters=int(expert_cfg["hybrid_cha"]["max_iters"]),
            tol=float(expert_cfg["hybrid_cha"]["tol"]),
        ),
        "connectivity_srm": ConnectivitySRMExpert(
            **common,
            max_iters=int(expert_cfg["connectivity_srm"]["max_iters"]),
            tol=float(expert_cfg["connectivity_srm"]["tol"]),
        ),
    }
    if list(experts) != list(expert_cfg["order"]):
        raise RuntimeError("Schaefer expert order differs from the config contract.")
    return experts


def prepare_schaefer400_training(config: dict) -> PreparedSchaefer400Training:
    """Fit both alignment experts and build compact parcel-level training views."""
    train_subjects = tuple(sorted(int(value) for value in config["subjects"]["train"]))
    set_global_seed(int(config["random_seed"]))
    subjects = {
        subject: SchaeferNSDSubjectData(subject, config["data_root"])
        for subject in train_subjects
    }
    for subject in subjects.values():
        validate_schaefer_nsd_subject(subject)

    feature_path = Path(config["features"]["path"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing CLIP feature matrix: {feature_path}")
    feature_matrix = np.load(feature_path, mmap_mode="r")
    if feature_matrix.ndim != 2:
        raise ValueError(f"CLIP features must be 2D, got {feature_matrix.shape}.")
    max_stimulus = max(int(subject.train_stim_idx.max()) for subject in subjects.values())
    if max_stimulus >= int(feature_matrix.shape[0]):
        raise ValueError("A training stimulus lies outside the CLIP feature matrix.")
    feature_slices = {"clip": (0, int(feature_matrix.shape[1]))}

    contract = Schaefer400ModelContract(
        training_subjects=train_subjects,
        rest_preprocessing=dict(config["rest_preprocessing"]),
        atlas_name=str(config["atlas"]["name"]),
        seed_set=str(config["parcel_seed_bank"]["seed_set"]),
        missing_subject_policy=str(
            config["parcel_seed_bank"]["missing_subject_policy"]
        ),
    )
    shared_stimulus_ids, shared_rows = shared_task_intersection(subjects)
    shared_task = {
        subject: np.asarray(
            subjects[subject].test_fmri[shared_rows[subject]],
            dtype=np.float32,
        )
        for subject in train_subjects
    }
    experts = build_schaefer400_experts(
        config,
        seed_manifest_fingerprint=contract.seed_fingerprint,
    )
    fit_alignment_experts(
        experts,
        subjects=subjects,
        external_seed_runs={
            subject: subjects[subject].rest_runs for subject in train_subjects
        },
        task_responses_shared=shared_task,
    )

    split = stimulus_disjoint_split(
        {subject: subjects[subject].train_stim_idx for subject in train_subjects},
        val_fraction=float(config["fusion"]["val_fraction"]),
        seed=int(config["random_seed"]),
    )
    views: dict[int, SubjectFusionData] = {}
    for subject in train_subjects:
        transforms = training_transforms(experts, subject)
        responses = subjects[subject].train_fmri
        subject_split = split.by_subject[subject]
        views[subject] = SubjectFusionData(
            subject_id=subject,
            stimulus_ids=subjects[subject].train_stim_idx,
            responses=responses,
            latent_targets={
                name: project_responses_chunked(transform, responses)
                for name, transform in transforms.items()
            },
            transforms=transforms,
            voxel_groups=subjects[subject].parcel_groups,
            train_indices=subject_split.train_indices,
            val_indices=subject_split.val_indices,
        )
    return PreparedSchaefer400Training(
        config=config,
        subjects=subjects,
        feature_matrix=feature_matrix,
        feature_slices=feature_slices,
        contract=contract,
        experts=experts,
        shared_stimulus_ids=shared_stimulus_ids,
        split=split,
        views=views,
    )


def fit_schaefer400_fusion(
    prepared: PreparedSchaefer400Training,
) -> FittedMultiExpertEncoder:
    network = build_fusion_network(
        prepared.config,
        input_dim=int(prepared.feature_matrix.shape[1]),
        expert_dims=prepared.expert_dims,
        num_regions=N_PARCELS,
        feature_slices=prepared.feature_slices,
    )
    encoder = FittedMultiExpertEncoder(
        network,
        FusionOptimizationConfig.from_pipeline_config(prepared.config),
    )
    return encoder.fit(prepared.feature_matrix, prepared.views)


def save_schaefer400_model(
    prepared: PreparedSchaefer400Training,
    encoder: FittedMultiExpertEncoder,
    *,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite a trained model: {output_dir}")
    (output_dir / "experts").mkdir(parents=True, exist_ok=True)
    for name, expert in prepared.experts.items():
        expert.save(output_dir / "experts" / name)
    encoder.save(output_dir / "encoder")
    prepared.contract.save(output_dir)
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
    input_summaries = {
        "feature_matrix": {
            "path": str(Path(prepared.config["features"]["path"]).resolve()),
            "size_bytes": int(Path(prepared.config["features"]["path"]).stat().st_size),
            "sha256": file_sha256(prepared.config["features"]["path"]),
        },
        "subjects": {},
    }
    for subject in prepared.train_subjects:
        subject_dir = Path(prepared.config["data_root"]) / f"subj{subject:02d}"
        named_files = {
            "atlas": subject_dir / prepared.config["atlas"]["nsd_filename"],
            "task_summary": subject_dir / "task_data_summary.json",
            "rest_manifest": subject_dir / "rest_run_manifest.json",
            "train_fmri": subject_dir / "train_fmri.npy",
            "test_fmri": subject_dir / "test_fmri.npy",
            "train_stim_idx": subject_dir / "train_stim_idx.npy",
            "test_stim_idx": subject_dir / "test_stim_idx.npy",
            "parcel_voxel_counts": subject_dir / "parcel_voxel_counts.npy",
        }
        named_files.update(
            {
                f"rest_run_{index}": path
                for index, path in enumerate(
                    sorted(
                        subject_dir.glob("rest_run*.npy"),
                        key=lambda value: int(value.stem.removeprefix("rest_run")),
                    ),
                    start=1,
                )
            }
        )
        input_summaries["subjects"][str(subject)] = {
            name: {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
            for name, path in named_files.items()
        }
    (output_dir / "training_inputs.json").write_text(
        json.dumps(input_summaries, indent=2, sort_keys=True) + "\n"
    )
    manifest = build_model_manifest(
        config=prepared.config,
        expert_dims=prepared.expert_dims,
        train_subjects=list(prepared.train_subjects),
        region_manifest=prepared.contract.region_manifest,
        seed_manifest=prepared.contract.seed_manifest,
        input_dim=int(prepared.feature_matrix.shape[1]),
        feature_slices=prepared.feature_slices,
        model_variant="learned_fusion_dropout",
    )
    manifest.update(
        {
            "representation": "schaefer400_parcel_means",
            "n_parcels": N_PARCELS,
            "n_regions_including_fallback": N_PARCELS,
            "training_seed": int(prepared.config["random_seed"]),
            "shared_stimulus_count": int(prepared.shared_stimulus_ids.size),
            "shared_stimulus_sha256": file_sha256(output_dir / "shared_stim_idx.npy"),
            "best_validation": encoder.validation_summary,
            "expert_state_fingerprints": alignment_expert_fingerprints(prepared.experts),
            "expert_artifact_fingerprints": directory_file_fingerprints(
                output_dir / "experts"
            ),
            "encoder_artifact_fingerprints": directory_file_fingerprints(
                output_dir / "encoder"
            ),
            "contract_sha256": file_sha256(output_dir / "schaefer400_contract.json"),
            "training_inputs_sha256": file_sha256(output_dir / "training_inputs.json"),
        }
    )
    save_model_manifest(output_dir, manifest)
    return manifest


def load_schaefer400_model(
    *,
    model_dir: str | Path,
    config: dict,
) -> tuple[dict, Schaefer400ModelContract, dict[str, AlignmentExpert], FittedMultiExpertEncoder]:
    """Load only an artifact whose atlas, seeds, experts, and config all agree."""
    model_dir = Path(model_dir)
    contract = Schaefer400ModelContract.load(model_dir)
    if contract.training_subjects != tuple(int(v) for v in config["subjects"]["train"]):
        raise ValueError("Model contract training subjects differ from the config.")
    if contract.rest_preprocessing != config["rest_preprocessing"]:
        raise ValueError("Model contract REST preprocessing differs from the config.")
    manifest = load_and_validate_model_manifest(
        model_dir,
        config=config,
        region_manifest=contract.region_manifest,
        seed_manifest=contract.seed_manifest,
        expected_train_subjects=config["subjects"]["train"],
        expected_model_variant="learned_fusion_dropout",
    )
    if manifest.get("representation") != "schaefer400_parcel_means":
        raise ValueError("Artifact is not a Schaefer parcel model.")
    if manifest.get("contract_sha256") != file_sha256(
        model_dir / "schaefer400_contract.json"
    ):
        raise ValueError("Saved Schaefer contract does not match the manifest.")
    if manifest.get("training_inputs_sha256") != file_sha256(
        model_dir / "training_inputs.json"
    ):
        raise ValueError("Saved training-input manifest is invalid.")
    if manifest.get("shared_stimulus_sha256") != file_sha256(
        model_dir / "shared_stim_idx.npy"
    ):
        raise ValueError("Saved shared-stimulus IDs are invalid.")
    shared_ids = np.load(model_dir / "shared_stim_idx.npy", mmap_mode="r")
    if shared_ids.ndim != 1 or shared_ids.size != int(manifest["shared_stimulus_count"]):
        raise ValueError("Saved shared-stimulus IDs have the wrong shape.")
    if manifest.get("expert_artifact_fingerprints") != directory_file_fingerprints(
        model_dir / "experts"
    ):
        raise ValueError("Alignment expert files do not match the manifest.")
    experts = load_alignment_experts(
        model_dir,
        expert_order=list(manifest["expert_order"]),
        expert_dims=manifest["expert_dims"],
        seed_manifest_fingerprint=contract.seed_fingerprint,
    )
    if manifest.get("expert_state_fingerprints") != alignment_expert_fingerprints(experts):
        raise ValueError("Alignment expert state does not match the manifest.")
    encoder = FittedMultiExpertEncoder.load(
        model_dir / "encoder",
        expected_expert_order=manifest["expert_order"],
        expected_expert_dims=manifest["expert_dims"],
    )
    if manifest.get("encoder_artifact_fingerprints") != directory_file_fingerprints(
        model_dir / "encoder"
    ):
        raise ValueError("Encoder files do not match the manifest.")
    if encoder.network.num_regions != N_PARCELS:
        raise ValueError("Schaefer encoder must contain exactly 400 fusion groups.")
    if encoder.network.config != fusion_architecture_config(config):
        raise ValueError("Encoder architecture differs from the effective config.")
    if encoder.optimization != FusionOptimizationConfig.from_pipeline_config(config):
        raise ValueError("Encoder optimization differs from the effective config.")
    return manifest, contract, experts, encoder
