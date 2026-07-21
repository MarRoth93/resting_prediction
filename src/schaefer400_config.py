"""Strict configuration for the separate Schaefer-400/FOR path."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.multiexpert_config import EXPECTED_EXPERT_ORDER


ATLAS_NAME = "Schaefer2018_400Parcels_7Networks_order"
N_PARCELS = 400


def _positive_unique_ints(values, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list.")
    clean = [int(value) for value in values]
    if any(value < 1 for value in clean) or len(set(clean)) != len(clean):
        raise ValueError(f"{field} must contain unique positive subject IDs.")
    return clean


def load_schaefer400_config(
    path: str | Path = "config_schaefer400.yaml",
) -> dict:
    """Load the dedicated parcel-level config without widening Stage-1 config."""
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")

    release = config.get("release", {}) or {}
    if (
        release.get("status") != "experimental"
        or int(release.get("stage", -1)) != 1
        or release.get("representation") != "schaefer400_parcels"
    ):
        raise ValueError(f"Expected the experimental Schaefer-400 Stage-1 config in {path}.")

    subjects = _positive_unique_ints(
        (config.get("subjects", {}) or {}).get("train"),
        "subjects.train",
    )
    if len(subjects) < 2:
        raise ValueError("Schaefer-400 training requires at least two NSD subjects.")

    atlas = config.get("atlas", {}) or {}
    if (
        atlas.get("name") != ATLAS_NAME
        or int(atlas.get("n_parcels", 0)) != N_PARCELS
        or int(atlas.get("n_networks", 0)) != 7
    ):
        raise ValueError("The FOR-compatible path is fixed to Schaefer-400, 7-network order.")
    if list(atlas.get("cortical_layers", [])) != [1, 2, 3]:
        raise ValueError("atlas.cortical_layers must be [1, 2, 3].")
    if int(atlas.get("min_voxels_per_parcel", 0)) < 1:
        raise ValueError("atlas.min_voxels_per_parcel must be positive.")

    features = config.get("features", {}) or {}
    if features.get("type") != "clip" or list(features.get("streams", [])) != ["clip"]:
        raise ValueError("The Schaefer-400 path requires the single CLIP feature stream.")
    if not str(features.get("path", "")):
        raise ValueError("features.path is required.")

    seed_bank = config.get("parcel_seed_bank", {}) or {}
    if seed_bank.get("seed_set") != "schaefer400_ordered_parcels":
        raise ValueError("parcel_seed_bank.seed_set has an incompatible parcel order.")
    if seed_bank.get("missing_subject_policy") != "zero_fill_and_mask":
        raise ValueError("FOR missing parcels must use zero_fill_and_mask for seed rows.")
    if seed_bank.get("ensemble_method") not in {"concat", "average"}:
        raise ValueError("parcel_seed_bank.ensemble_method must be concat or average.")

    experts = config.get("experts", {}) or {}
    if list(experts.get("order", [])) != EXPECTED_EXPERT_ORDER:
        raise ValueError(f"experts.order must be {EXPECTED_EXPERT_ORDER}.")
    n_components = int(experts.get("n_components", 0))
    min_k = int(experts.get("min_k", 0))
    if n_components < min_k or min_k < 1 or n_components >= N_PARCELS:
        raise ValueError("experts must satisfy 1 <= min_k <= n_components < 400.")
    for name in EXPECTED_EXPERT_ORDER:
        method = experts.get(name, {}) or {}
        if int(method.get("max_iters", 0)) < 1 or float(method.get("tol", 0)) <= 0:
            raise ValueError(f"experts.{name} requires positive max_iters and tol.")

    regions = config.get("regions", {}) or {}
    if regions.get("grouping") != "parcel" or int(regions.get("n_groups", 0)) != N_PARCELS:
        raise ValueError("Schaefer fusion must use one canonical group per parcel.")

    fusion = config.get("fusion", {}) or {}
    backbone = fusion.get("backbone", {}) or {}
    if int(backbone.get("d_model", 0)) % int(backbone.get("n_heads", 1)):
        raise ValueError("fusion backbone width must be divisible by its head count.")
    if int(fusion.get("method_projection_dim", 0)) % int(fusion.get("transformer_heads", 1)):
        raise ValueError("fusion projection width must be divisible by its head count.")
    method_dropout = float(fusion.get("method_dropout", -1.0))
    if not 0.0 <= method_dropout < 1.0:
        raise ValueError("fusion.method_dropout must be in [0, 1).")
    for key in ("batch_size", "max_epochs", "patience"):
        if int(fusion.get(key, 0)) < 1:
            raise ValueError(f"fusion.{key} must be positive.")
    if not 0.0 < float(fusion.get("val_fraction", 0.0)) < 1.0:
        raise ValueError("fusion.val_fraction must be between 0 and 1.")

    for field in ("data_root", "raw_data_root", "for_data_root", "output_root"):
        if not str(config.get(field, "")):
            raise ValueError(f"{field} is required.")
    return config


def resolve_schaefer400_roots(
    config: dict,
    *,
    data_root: str | Path | None = None,
    raw_data_root: str | Path | None = None,
    for_data_root: str | Path | None = None,
) -> dict:
    """Resolve every data-bearing path used in provenance and artifact checks."""
    effective = copy.deepcopy(config)
    overrides = {
        "data_root": data_root,
        "raw_data_root": raw_data_root,
        "for_data_root": for_data_root,
    }
    for field, override in overrides.items():
        effective[field] = str(Path(override or effective[field]).expanduser().resolve())
    effective["features"]["path"] = str(
        Path(effective["features"]["path"]).expanduser().resolve()
    )
    return effective
