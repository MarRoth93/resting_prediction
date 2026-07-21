"""Shared orchestration helpers for multi-expert train, LOSO, and prediction."""

from __future__ import annotations

import json
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src.alignment.connectivity_srm import ConnectivitySRMExpert
from src.alignment.experts import (
    AlignmentExpert,
    HybridCHAExpert,
    SubjectTransform,
    load_alignment_expert,
)
from src.alignment.external_seed_bank import (
    SEED_SET,
    SeedDef,
    build_common_seed_defs,
    load_atlas_array,
    load_or_prepare_external_seed_runs,
    save_external_seed_info,
    seed_bank_cache_id,
    seed_defs_from_jsonable,
    seed_defs_to_jsonable,
)
from src.data.nsd_loader import NSDSubjectData
from src.models.multiexpert_encoding import (
    MultiExpertFusionConfig,
    MultiExpertFusionNetwork,
)
from src.pipelines.multiexpert_artifacts import json_fingerprint


logger = logging.getLogger(__name__)


def _array_fingerprint(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def alignment_expert_fingerprints(
    experts: Mapping[str, AlignmentExpert],
) -> dict[str, str]:
    """Fingerprint transforms and templates that define each expert's coordinates."""
    out: dict[str, str] = {}
    for name, expert in experts.items():
        if isinstance(expert, HybridCHAExpert):
            subject_ids = sorted(expert.builder.subject_bases)
            state = {
                "name": name,
                "k": expert.k_global,
                "seed_manifest_fingerprint": expert.seed_manifest_fingerprint,
                "transforms": {
                    str(subject): expert.training_transform(subject).fingerprint
                    for subject in subject_ids
                },
                "task_template": _array_fingerprint(expert.builder.template_Z),
                "connectivity_template": _array_fingerprint(
                    expert.builder.template_fingerprint
                ),
            }
        elif isinstance(expert, ConnectivitySRMExpert):
            subject_ids = sorted(expert.subject_weights)
            state = {
                "name": name,
                "k": expert.k_global,
                "seed_manifest_fingerprint": expert.seed_manifest_fingerprint,
                "transforms": {
                    str(subject): expert.training_transform(subject).fingerprint
                    for subject in subject_ids
                },
                "shared_response": _array_fingerprint(expert.shared_response),
                "task_template": _array_fingerprint(expert.task_template),
            }
        else:
            raise TypeError(f"Unsupported Stage-1 expert type: {type(expert).__name__}.")
        out[name] = json_fingerprint(state)
    return out


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def validate_subject_data(subject: NSDSubjectData) -> None:
    """Validate every row and voxel contract used by the new pipeline."""
    train_shape = tuple(int(value) for value in subject.train_fmri.shape)
    test_shape = tuple(int(value) for value in subject.test_fmri.shape)
    if train_shape[0] != int(subject.train_stim_idx.shape[0]):
        raise ValueError(f"Subject {subject.sub}: training row mismatch.")
    if test_shape[0] != int(subject.test_stim_idx.shape[0]):
        raise ValueError(f"Subject {subject.sub}: test row mismatch.")
    if train_shape[1] != test_shape[1] or train_shape[1] != subject.num_voxels:
        raise ValueError(
            f"Subject {subject.sub}: task arrays and mask have different voxel counts."
        )
    if not subject.rest_runs:
        raise ValueError(f"Subject {subject.sub}: no processed REST runs.")
    rest_voxels = {int(run.shape[1]) for run in subject.rest_runs}
    if rest_voxels != {train_shape[1]}:
        raise ValueError(
            f"Subject {subject.sub}: REST voxel counts {sorted(rest_voxels)} do not "
            f"match task voxels {train_shape[1]}."
        )
    if np.any(subject.train_stim_idx < 0) or np.any(subject.test_stim_idx < 0):
        raise ValueError(f"Subject {subject.sub}: negative stimulus IDs.")


def shared_task_intersection(
    subjects: Mapping[int, NSDSubjectData],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return canonical shared test stimuli and aligned row indices."""
    if not subjects:
        raise ValueError("No subjects supplied for the shared task intersection.")
    sets: list[set[int]] = []
    clean: dict[int, np.ndarray] = {}
    for subject in sorted(subjects):
        ids = np.asarray(subjects[subject].test_stim_idx, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError(f"Subject {subject}: test stimulus IDs are empty or not 1D.")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"Subject {subject}: duplicate test stimulus IDs.")
        clean[subject] = ids
        sets.append(set(ids.tolist()))
    shared = np.asarray(sorted(set.intersection(*sets)), dtype=np.int64)
    if shared.size == 0:
        raise ValueError("Training subjects have no shared task stimuli.")
    rows: dict[int, np.ndarray] = {}
    for subject, ids in clean.items():
        order = np.argsort(ids)
        sorted_ids = ids[order]
        positions = np.searchsorted(sorted_ids, shared)
        valid = (positions < sorted_ids.size) & (sorted_ids[positions] == shared)
        if not np.all(valid):
            raise RuntimeError(f"Subject {subject}: internal shared-stimulus mapping failed.")
        rows[subject] = order[positions].astype(np.int64, copy=False)
    return shared, rows


@dataclass(frozen=True)
class ExternalSeedSpec:
    seed_set: str
    seed_defs: tuple[SeedDef, ...]
    coverage: tuple[dict, ...]
    rest_config: dict
    min_voxels_per_seed: int
    registry_subjects: tuple[int, ...]
    missing_subject_policy: str

    @property
    def contract(self) -> dict:
        return {
            "seed_set": self.seed_set,
            "n_seeds": len(self.seed_defs),
            "min_voxels_per_seed": int(self.min_voxels_per_seed),
            "registry_subjects": [int(value) for value in self.registry_subjects],
            "missing_subject_policy": self.missing_subject_policy,
            "cache_id": seed_bank_cache_id(
                self.seed_set, list(self.seed_defs), self.rest_config
            ),
            "seed_defs": seed_defs_to_jsonable(list(self.seed_defs)),
            "rest_preprocessing": self.rest_config,
        }

    @property
    def fingerprint(self) -> str:
        return json_fingerprint(self.contract)

    def save(self, model_dir: str | Path) -> None:
        save_external_seed_info(
            output_dir=model_dir,
            seed_set=self.seed_set,
            seed_defs=list(self.seed_defs),
            coverage_rows=list(self.coverage),
            rest_cfg=self.rest_config,
            min_voxels_per_seed=self.min_voxels_per_seed,
        )
        path = Path(model_dir) / "external_seed_info.json"
        info = json.loads(path.read_text())
        info["registry_subjects"] = [int(value) for value in self.registry_subjects]
        info["missing_subject_policy"] = self.missing_subject_policy
        path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, model_dir: str | Path) -> "ExternalSeedSpec":
        path = Path(model_dir) / "external_seed_info.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing external seed manifest: {path}")
        info = json.loads(path.read_text())
        seed_set = str(info["seed_set"])
        if seed_set != SEED_SET:
            raise ValueError(f"Unsupported external seed set: {seed_set!r}.")
        spec = cls(
            seed_set=seed_set,
            seed_defs=tuple(seed_defs_from_jsonable(info["seed_defs"])),
            coverage=tuple(dict(row) for row in info.get("coverage", [])),
            rest_config=dict(info.get("rest_preprocessing", {}) or {}),
            min_voxels_per_seed=int(info["min_voxels_per_seed"]),
            registry_subjects=tuple(int(value) for value in info["registry_subjects"]),
            missing_subject_policy=str(info["missing_subject_policy"]),
        )
        if (
            not spec.registry_subjects
            or len(set(spec.registry_subjects)) != len(spec.registry_subjects)
            or spec.missing_subject_policy != "zero_fill_and_mask"
        ):
            raise ValueError("External seed manifest has an invalid subject/policy contract.")
        stored_contract = {
            key: info[key]
            for key in (
                "seed_set",
                "n_seeds",
                "min_voxels_per_seed",
                "registry_subjects",
                "missing_subject_policy",
                "cache_id",
                "seed_defs",
                "rest_preprocessing",
            )
        }
        if spec.contract != stored_contract:
            raise ValueError("External seed manifest is internally inconsistent.")
        return spec

    def runs_for_subject(
        self,
        subject: NSDSubjectData,
        *,
        data_root: str,
        raw_data_root: str,
        force_recompute: bool = False,
        allow_missing: bool = False,
        availability: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        if availability is None:
            availability = self.availability_for_subject(
                subject,
                raw_data_root=raw_data_root,
            )
        availability = np.asarray(availability, dtype=bool)
        if availability.shape != (len(self.seed_defs),):
            raise ValueError("External seed availability has the wrong shape.")
        if not np.all(availability) and not allow_missing:
            missing = int((~availability).sum())
            raise ValueError(
                f"Subject {subject.sub} is missing {missing} external seeds."
            )
        if not np.any(availability):
            raise ValueError(f"Subject {subject.sub} has no usable external seeds.")
        available_defs = [
            seed for seed, keep in zip(self.seed_defs, availability, strict=True) if keep
        ]
        available_runs = load_or_prepare_external_seed_runs(
            sub=subject.sub,
            data_root=data_root,
            raw_data_root=raw_data_root,
            pred_mask=subject.mask,
            seed_defs=available_defs,
            rest_cfg=self.rest_config,
            seed_set=self.seed_set,
            reference_rest_runs=subject.rest_runs,
            force_recompute=bool(force_recompute),
        )
        if np.all(availability):
            return available_runs
        if self.missing_subject_policy != "zero_fill_and_mask":
            raise ValueError(
                f"Unsupported missing-seed policy: {self.missing_subject_policy!r}."
            )
        logger.warning(
            "Subject %d: zero-filling %d/%d anatomically absent external seeds.",
            subject.sub,
            int((~availability).sum()),
            int(availability.size),
        )
        expanded: list[np.ndarray] = []
        for run in available_runs:
            full = np.zeros((run.shape[0], availability.size), dtype=np.float32)
            full[:, availability] = run
            expanded.append(full)
        return expanded

    def availability_for_subject(
        self,
        subject: NSDSubjectData,
        *,
        raw_data_root: str,
    ) -> np.ndarray:
        """Return anatomical presence for every ordered seed definition."""
        atlas_cache: dict[str, np.ndarray] = {}
        available = np.zeros(len(self.seed_defs), dtype=bool)
        for index, seed in enumerate(self.seed_defs):
            if seed.atlas_file not in atlas_cache:
                atlas = load_atlas_array(
                    raw_data_root,
                    int(subject.sub),
                    seed.atlas_file,
                )
                if atlas.shape != np.asarray(subject.mask).shape:
                    raise ValueError(
                        f"Subject {subject.sub}: seed atlas and prediction mask shapes differ."
                    )
                atlas_cache[seed.atlas_file] = atlas
            available[index] = bool(
                np.any(atlas_cache[seed.atlas_file] == int(seed.label))
            )
        return available

    def cache_dir_for_subject(
        self,
        subject: NSDSubjectData,
        *,
        data_root: str,
        availability: np.ndarray,
    ) -> Path:
        """Locate the exact present-seed cache consumed for this subject."""
        availability = np.asarray(availability, dtype=bool)
        if availability.shape != (len(self.seed_defs),) or not np.any(availability):
            raise ValueError("Cannot locate a seed cache for invalid availability.")
        available_defs = [
            seed for seed, keep in zip(self.seed_defs, availability, strict=True) if keep
        ]
        cache_id = seed_bank_cache_id(
            self.seed_set,
            available_defs,
            self.rest_config,
        )
        return (
            Path(data_root)
            / f"subj{int(subject.sub):02d}"
            / "external_seed_banks"
            / f"{self.seed_set}_{cache_id}"
        )


def build_external_seed_spec(
    config: dict,
    *,
    data_root: str,
    raw_data_root: str,
    registry_subjects: Iterable[int] | None = None,
) -> ExternalSeedSpec:
    if registry_subjects is None:
        registry_subjects = config["subjects"]["seed_registry"]
    registry_subjects = [int(value) for value in registry_subjects]
    if not registry_subjects or len(set(registry_subjects)) != len(registry_subjects):
        raise ValueError("External seed registry subjects must be non-empty and unique.")
    masks = {
        subject: np.load(
            Path(data_root) / f"subj{subject:02d}" / "mask.npy",
            mmap_mode="r",
        )
        for subject in registry_subjects
    }
    seed_cfg = config["external_seed_bank"]
    seed_defs, coverage = build_common_seed_defs(
        seed_set=str(seed_cfg["seed_set"]),
        raw_data_root=raw_data_root,
        subjects=registry_subjects,
        pred_masks=masks,
        min_voxels_per_seed=int(seed_cfg["min_voxels_per_seed"]),
    )
    return ExternalSeedSpec(
        seed_set=str(seed_cfg["seed_set"]),
        seed_defs=tuple(seed_defs),
        coverage=tuple(coverage),
        rest_config=dict(config["rest_preprocessing"]),
        min_voxels_per_seed=int(seed_cfg["min_voxels_per_seed"]),
        registry_subjects=tuple(registry_subjects),
        missing_subject_policy=str(seed_cfg["missing_subject_policy"]),
    )


def build_alignment_experts(
    config: dict,
    *,
    seed_manifest_fingerprint: str,
) -> dict[str, AlignmentExpert]:
    expert_cfg = config["experts"]
    common = {
        "n_components": int(expert_cfg["n_components"]),
        "min_k": int(expert_cfg["min_k"]),
        "ensemble_method": str(config["external_seed_bank"]["ensemble_method"]),
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
        raise RuntimeError("Internal expert construction order differs from the config.")
    return experts


def fit_alignment_experts(
    experts: Mapping[str, AlignmentExpert],
    *,
    subjects: Mapping[int, NSDSubjectData],
    external_seed_runs: Mapping[int, list[np.ndarray]],
    task_responses_shared: Mapping[int, np.ndarray],
) -> None:
    """Fit both experts while computing the large connectivity matrices once."""
    hybrid = experts["hybrid_cha"]
    csrm = experts["connectivity_srm"]
    if not isinstance(hybrid, HybridCHAExpert) or not isinstance(csrm, ConnectivitySRMExpert):
        raise TypeError("Stage 1 requires HybridCHAExpert followed by ConnectivitySRMExpert.")
    rest_runs = {subject: subjects[subject].rest_runs for subject in sorted(subjects)}
    hybrid.fit(
        rest_runs=rest_runs,
        task_responses_shared=dict(task_responses_shared),
        external_seed_runs=dict(external_seed_runs),
    )
    csrm.fit_connectivity(
        hybrid.builder.subject_connectivity,
        task_responses_shared=dict(task_responses_shared),
    )
    if hybrid.k_global != csrm.k_global:
        raise RuntimeError(
            f"Expert latent dimensions differ: hybrid={hybrid.k_global}, cSRM={csrm.k_global}."
        )


def training_transforms(
    experts: Mapping[str, AlignmentExpert],
    subject: int,
) -> dict[str, SubjectTransform]:
    return {
        name: expert.training_transform(subject)
        for name, expert in experts.items()
    }


def load_alignment_experts(
    model_dir: str | Path,
    *,
    expert_order: list[str],
    expert_dims: Mapping[str, int],
    seed_manifest_fingerprint: str,
) -> dict[str, AlignmentExpert]:
    experts = {}
    for name in expert_order:
        experts[name] = load_alignment_expert(
            name,
            Path(model_dir) / "experts" / name,
            expected_n_components=int(expert_dims[name]),
            expected_seed_manifest_fingerprint=seed_manifest_fingerprint,
        )
    return experts


def align_new_subject(
    experts: Mapping[str, AlignmentExpert],
    *,
    subject: NSDSubjectData,
    external_seed_runs: list[np.ndarray],
    seed_manifest_fingerprint: str,
    task_fmri_shared: np.ndarray | None = None,
    shot_indices: np.ndarray | None = None,
) -> dict[str, SubjectTransform]:
    transforms: dict[str, SubjectTransform] = {}
    for name, expert in experts.items():
        if task_fmri_shared is None:
            transform = expert.align_new_subject_zeroshot(
                subject.rest_runs,
                external_seed_runs,
                subject_id=subject.sub,
                seed_manifest_fingerprint=seed_manifest_fingerprint,
            )
        else:
            transform = expert.align_new_subject_fewshot(
                subject.rest_runs,
                task_fmri_shared,
                external_seed_runs,
                shot_indices=shot_indices,
                subject_id=subject.sub,
                seed_manifest_fingerprint=seed_manifest_fingerprint,
            )
        transforms[name] = transform
    return transforms


def project_responses_chunked(
    transform: SubjectTransform,
    responses: np.ndarray,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    n_rows = int(responses.shape[0])
    projected = np.empty((n_rows, transform.n_components), dtype=np.float32)
    for start in range(0, n_rows, int(chunk_size)):
        stop = min(start + int(chunk_size), n_rows)
        projected[start:stop] = transform.project(
            np.asarray(responses[start:stop], dtype=np.float32)
        )
    return projected


def build_fusion_network(
    config: dict,
    *,
    input_dim: int,
    expert_dims: Mapping[str, int],
    num_regions: int,
    feature_slices: Mapping[str, tuple[int, int]],
) -> MultiExpertFusionNetwork:
    # Network construction itself consumes RNG. Seed here so ablations start
    # from the same weights regardless of what trained immediately before them.
    set_global_seed(int(config["random_seed"]))
    architecture = fusion_architecture_config(config)
    return MultiExpertFusionNetwork(
        input_dim=int(input_dim),
        expert_dims=expert_dims,
        num_regions=int(num_regions),
        feature_slices=feature_slices,
        config=architecture,
    )


def fusion_architecture_config(config: dict) -> MultiExpertFusionConfig:
    """Translate the pipeline config into the persisted neural architecture."""
    fusion = config["fusion"]
    backbone = fusion["backbone"]
    return MultiExpertFusionConfig(
        backbone_dim=int(backbone["d_model"]),
        backbone_layers=int(backbone["n_layers"]),
        backbone_heads=int(backbone["n_heads"]),
        backbone_ff_multiplier=float(backbone["ff_multiplier"]),
        backbone_dropout=float(backbone["dropout"]),
        fusion_dim=int(fusion["method_projection_dim"]),
        fusion_layers=int(fusion["transformer_layers"]),
        fusion_heads=int(fusion["transformer_heads"]),
        fusion_ff_multiplier=float(fusion["transformer_ff_multiplier"]),
        fusion_dropout=float(fusion["transformer_dropout"]),
        method_dropout=float(fusion["method_dropout"]),
        seed=int(config["random_seed"]),
    )
