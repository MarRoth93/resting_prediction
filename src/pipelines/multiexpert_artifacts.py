"""Versioned artifact contracts for the experimental multi-expert path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.multiexpert_config import EXPECTED_EXPERT_ORDER, canonical_config_hash


ARTIFACT_VERSION = 1
MODEL_VARIANTS = {
    "learned_fusion_dropout",
    "learned_fusion_no_dropout",
}


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_file_fingerprints(root: str | Path) -> dict[str, str]:
    """Hash every regular file below a required artifact directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing artifact directory: {root}")
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise ValueError(f"Artifact directory contains no files: {root}")
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in files
    }


def cached_input_file_manifest(
    files: Mapping[str, str | Path],
    *,
    cache_path: str | Path,
) -> dict:
    """Content-hash named inputs, reusing hashes only when file stats match."""
    cache_path = Path(cache_path)
    try:
        cache = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {"schema_version": 1, "entries": {}}
    if cache.get("schema_version") != 1 or not isinstance(cache.get("entries"), dict):
        cache = {"schema_version": 1, "entries": {}}

    entries = cache["entries"]
    manifest_files: dict[str, dict] = {}
    for logical_name, raw_path in sorted(files.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing LOSO input file: {path}")
        stat = path.stat()
        cache_key = str(path)
        cached = entries.get(cache_key, {})
        if (
            int(cached.get("size", -1)) == int(stat.st_size)
            and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and str(cached.get("sha256", ""))
        ):
            sha256 = str(cached["sha256"])
        else:
            sha256 = file_sha256(path)
        entries[cache_key] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256,
        }
        manifest_files[str(logical_name)] = {
            "path": str(path),
            "size": int(stat.st_size),
            "sha256": sha256,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, cache_path)
    payload = {"schema_version": 1, "files": manifest_files}
    payload["fingerprint"] = json_fingerprint(payload)
    return payload


def build_model_manifest(
    *,
    config: dict,
    expert_dims: dict[str, int],
    train_subjects: list[int],
    region_manifest: dict,
    seed_manifest: dict,
    input_dim: int,
    feature_slices: dict[str, tuple[int, int]],
    model_variant: str,
) -> dict:
    order = list(config["experts"]["order"])
    if order != EXPECTED_EXPERT_ORDER:
        raise ValueError(f"Unexpected expert order: {order}.")
    if list(expert_dims) != order:
        raise ValueError(
            f"expert_dims insertion order must match experts.order: {list(expert_dims)} vs {order}."
        )
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported multi-expert model variant: {model_variant!r}.")
    if int(input_dim) < 1:
        raise ValueError("Model input_dim must be positive.")
    return {
        "artifact_version": ARTIFACT_VERSION,
        "release_name": str(config["release"]["name"]),
        "stage": 1,
        "model_variant": str(model_variant),
        "config_hash": canonical_config_hash(config),
        "expert_order": order,
        "expert_dims": {name: int(expert_dims[name]) for name in order},
        "train_subjects": [int(subject) for subject in train_subjects],
        "input_dim": int(input_dim),
        "feature_slices": {
            str(name): [int(bounds[0]), int(bounds[1])]
            for name, bounds in feature_slices.items()
        },
        "region_fingerprint": json_fingerprint(region_manifest),
        "seed_manifest_fingerprint": json_fingerprint(seed_manifest),
    }


def save_model_manifest(model_dir: str | Path, manifest: dict) -> Path:
    path = Path(model_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_and_validate_model_manifest(
    model_dir: str | Path,
    *,
    config: dict,
    region_manifest: dict | None = None,
    seed_manifest: dict | None = None,
    expected_train_subjects: list[int] | tuple[int, ...] | None = None,
    expected_model_variant: str | None = "learned_fusion_dropout",
    expected_input_dim: int | None = None,
    expected_feature_slices: dict[str, tuple[int, int]] | None = None,
) -> dict:
    path = Path(model_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-expert manifest: {path}")
    manifest = json.loads(path.read_text())
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "stage": 1,
        "config_hash": canonical_config_hash(config),
        "expert_order": list(config["experts"]["order"]),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Incompatible multi-expert artifact field {field}: "
                f"stored={manifest.get(field)!r}, expected={value!r}."
            )
    dims = manifest.get("expert_dims", {})
    if set(dims) != set(manifest["expert_order"]) or any(
        int(value) < 1 for value in dims.values()
    ):
        raise ValueError("Artifact expert dimensions are missing, reordered, or invalid.")
    if region_manifest is not None:
        expected_region = json_fingerprint(region_manifest)
        if manifest.get("region_fingerprint") != expected_region:
            raise ValueError("Region registry does not match the trained artifact.")
        registered_subjects = [int(value) for value in region_manifest["training_subjects"]]
        if [int(value) for value in manifest.get("train_subjects", [])] != registered_subjects:
            raise ValueError("Artifact training subjects do not match the region registry.")
    if seed_manifest is not None:
        expected_seed = json_fingerprint(seed_manifest)
        if manifest.get("seed_manifest_fingerprint") != expected_seed:
            raise ValueError("External seed manifest does not match the trained artifact.")
    if expected_train_subjects is not None and [
        int(value) for value in manifest.get("train_subjects", [])
    ] != [int(value) for value in expected_train_subjects]:
        raise ValueError("Artifact training subjects do not match the requested model.")
    if (
        expected_model_variant is not None
        and manifest.get("model_variant") != expected_model_variant
    ):
        raise ValueError("Artifact model variant does not match the requested model.")
    if expected_input_dim is not None and int(manifest.get("input_dim", -1)) != int(
        expected_input_dim
    ):
        raise ValueError("Artifact input dimension does not match the feature matrix.")
    if expected_feature_slices is not None:
        expected_slices = {
            str(name): [int(bounds[0]), int(bounds[1])]
            for name, bounds in expected_feature_slices.items()
        }
        if manifest.get("feature_slices") != expected_slices:
            raise ValueError("Artifact feature slices do not match the requested model.")
    return manifest
