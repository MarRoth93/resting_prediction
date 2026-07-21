"""Strict configuration loading for the experimental multi-expert pipeline."""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import yaml


EXPECTED_EXPERT_ORDER = ["hybrid_cha", "connectivity_srm"]


def canonical_config_hash(config: dict) -> str:
    """Hash the effective config rather than filesystem formatting."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_data_roots(
    config: dict,
    *,
    data_root: str | Path | None = None,
    raw_data_root: str | Path | None = None,
) -> dict:
    """Return an effective config whose data provenance uses absolute paths."""
    effective = copy.deepcopy(config)
    effective["data_root"] = str(
        Path(data_root or effective["data_root"]).expanduser().resolve()
    )
    effective["raw_data_root"] = str(
        Path(raw_data_root or effective["raw_data_root"]).expanduser().resolve()
    )
    return effective


def load_multiexpert_config(path: str | Path = "config_multiexpert.yaml") -> dict:
    """Load and validate only the Stage-1 experimental schema."""
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")

    release = config.get("release", {}) or {}
    if release.get("status") != "experimental" or int(release.get("stage", -1)) != 1:
        raise ValueError(f"Expected an experimental Stage-1 config in {path}.")
    if config.get("analysis_mask", {}).get("mode") != "nsdgeneral":
        raise ValueError("Stage 1 supports only the subject-native nsdgeneral mask.")
    if config.get("features", {}).get("type") != "clip":
        raise ValueError("Stage 1 supports only CLIP stimulus features.")
    if list(config.get("features", {}).get("streams", [])) != ["clip"]:
        raise ValueError("Stage 1 requires features.streams=['clip'].")

    subjects = config.get("subjects", {}) or {}
    train = _positive_unique_ints(subjects.get("train"), "subjects.train")
    loso = _positive_unique_ints(subjects.get("loso"), "subjects.loso")
    locked = _positive_unique_ints(subjects.get("locked_test"), "subjects.locked_test")
    seed_registry = _positive_unique_ints(
        subjects.get("seed_registry"), "subjects.seed_registry"
    )
    if loso != train:
        raise ValueError("Stage-1 LOSO subjects must exactly match subjects.train.")
    if set(train) & set(locked):
        raise ValueError("Locked test subjects must not appear in training subjects.")
    if not set(train).issubset(seed_registry):
        raise ValueError("subjects.seed_registry must cover every training subject.")
    if set(locked) & set(seed_registry):
        raise ValueError("Locked test subjects must not shape the external seed registry.")

    experts = config.get("experts", {}) or {}
    order = list(experts.get("order", []))
    if order != EXPECTED_EXPERT_ORDER:
        raise ValueError(
            f"experts.order must be {EXPECTED_EXPERT_ORDER}; got {order}. "
            "Expert order is part of the artifact contract."
        )
    n_components = int(experts.get("n_components", 0))
    min_k = int(experts.get("min_k", 0))
    if n_components < min_k or min_k < 1:
        raise ValueError("experts.n_components must be >= experts.min_k >= 1.")
    for name in EXPECTED_EXPERT_ORDER:
        cfg = experts.get(name, {}) or {}
        if int(cfg.get("max_iters", 0)) < 1 or float(cfg.get("tol", 0.0)) <= 0:
            raise ValueError(f"experts.{name} requires positive max_iters and tol.")

    seed_bank = config.get("external_seed_bank", {}) or {}
    if seed_bank.get("missing_subject_policy") != "zero_fill_and_mask":
        raise ValueError(
            "Stage 1 requires external_seed_bank.missing_subject_policy="
            "'zero_fill_and_mask'."
        )

    regions = config.get("regions", {}) or {}
    if list(regions.get("atlas_files", [])) != [
        "lh.HCP_MMP1.nii.gz",
        "rh.HCP_MMP1.nii.gz",
    ]:
        raise ValueError("Stage 1 requires the left and right HCP-MMP atlases in fixed order.")
    if int(regions.get("min_voxels_every_training_subject", 0)) < 1:
        raise ValueError("regions.min_voxels_every_training_subject must be positive.")

    fusion = config.get("fusion", {}) or {}
    backbone = fusion.get("backbone", {}) or {}
    if int(backbone.get("d_model", 0)) != 384:
        raise ValueError("Stage-1 backbone width is fixed at 384.")
    if int(backbone.get("n_layers", 0)) != 8 or int(backbone.get("n_heads", 0)) != 8:
        raise ValueError("Stage-1 backbone is fixed at 8 layers and 8 heads.")
    if int(fusion.get("method_projection_dim", 0)) != 128:
        raise ValueError("Stage-1 method projection width is fixed at 128.")
    if int(fusion.get("transformer_layers", 0)) != 2:
        raise ValueError("Stage-1 fusion transformer is fixed at 2 layers.")
    if int(fusion.get("transformer_heads", 0)) != 4:
        raise ValueError("Stage-1 fusion transformer is fixed at 4 heads.")
    method_dropout = float(fusion.get("method_dropout", -1.0))
    if not 0.0 <= method_dropout < 1.0:
        raise ValueError("fusion.method_dropout must be in [0, 1).")
    for key in ("batch_size", "max_epochs", "patience"):
        if int(fusion.get(key, 0)) < 1:
            raise ValueError(f"fusion.{key} must be positive.")
    val_fraction = float(fusion.get("val_fraction", 0.0))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("fusion.val_fraction must be between 0 and 1.")

    evaluation = config.get("evaluation", {}) or {}
    _positive_unique_ints(evaluation.get("robustness_seeds"), "evaluation.robustness_seeds")
    if int(evaluation.get("fewshot_n", 0)) < 1:
        raise ValueError("evaluation.fewshot_n must be positive.")
    if int(evaluation.get("fixed_eval_size", 0)) < 1:
        raise ValueError("evaluation.fixed_eval_size must be positive.")
    gate = evaluation.get("gate", {}) or {}
    if str(gate.get("mode")) != "zero_shot":
        raise ValueError("Stage-1 gate is fixed to zero_shot LOSO evaluation.")
    if float(gate.get("minimum_mean_delta_median_r", -1.0)) < 0:
        raise ValueError("Gate mean-delta threshold must be non-negative.")
    if not 1 <= int(gate.get("minimum_subject_wins", 0)) <= len(loso):
        raise ValueError("Gate subject-win threshold is outside the LOSO subject count.")
    if float(gate.get("maximum_reliable_voxel_regression", -1.0)) < 0:
        raise ValueError("Gate reliable-voxel regression limit must be non-negative.")

    return config


def _positive_unique_ints(values, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list.")
    clean = [int(value) for value in values]
    if any(value < 1 for value in clean) or len(set(clean)) != len(clean):
        raise ValueError(f"{field} must contain unique positive subject IDs.")
    return clean
