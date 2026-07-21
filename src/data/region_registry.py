"""Deterministic HCP-MMP region groups for multi-expert voxel fusion.

The registry is fitted from training subjects only.  A cortical parcel is kept
as its own fusion group when it contains enough voxels inside every training
subject's prediction mask.  Voxels belonging to sparse parcels, unlabeled
voxels, and any other atlas labels are assigned to one final fallback group.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REGISTRY_SCHEMA_VERSION = 1
HCP_MMP_ATLAS_FILES = (
    "lh.HCP_MMP1.nii.gz",
    "rh.HCP_MMP1.nii.gz",
)
FALLBACK_REGION_NAME = "fallback"


def _canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray, dtype: np.dtype | type) -> str:
    """Hash array values together with their shape using a canonical dtype."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    return _array_sha256(np.asarray(mask, dtype=np.uint8), np.uint8)


def _atlas_sha256(atlas: np.ndarray) -> str:
    # Explicit little-endian storage keeps the digest independent of host byte order.
    return _array_sha256(atlas, np.dtype("<i4"))


def _group_map_sha256(groups: np.ndarray) -> str:
    return _array_sha256(groups, np.dtype("<i4"))


@dataclass(frozen=True)
class RegionDefinition:
    """One retained, hemisphere-qualified HCP-MMP parcel."""

    atlas_file: str
    label: int
    name: str

    @classmethod
    def create(cls, atlas_file: str, label: int) -> "RegionDefinition":
        stem = str(atlas_file).removesuffix(".nii.gz")
        return cls(
            atlas_file=str(atlas_file),
            label=int(label),
            name=f"{stem}:label{int(label):03d}",
        )


def _validate_subject_arrays(
    subject: int,
    atlas_files: Sequence[str],
    atlases: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 3:
        raise ValueError(
            f"Subject {subject}: prediction mask must be 3D, got {mask_array.shape}."
        )
    if not np.any(mask_array):
        raise ValueError(f"Subject {subject}: prediction mask is empty.")

    expected = set(atlas_files)
    available = set(atlases)
    if available != expected:
        missing = sorted(expected - available)
        extra = sorted(available - expected)
        raise ValueError(
            f"Subject {subject}: atlas set mismatch; missing={missing}, extra={extra}."
        )

    clean_atlases: dict[str, np.ndarray] = {}
    occupied = np.zeros(mask_array.shape, dtype=bool)
    for atlas_file in atlas_files:
        atlas = np.asarray(atlases[atlas_file], dtype=np.int32)
        if atlas.shape != mask_array.shape:
            raise ValueError(
                f"Subject {subject}: atlas {atlas_file} shape {atlas.shape} does not "
                f"match prediction mask {mask_array.shape}."
            )
        labeled = (atlas > 0) & mask_array
        overlap = occupied & labeled
        if np.any(overlap):
            raise ValueError(
                f"Subject {subject}: HCP-MMP hemisphere atlases overlap inside the "
                f"prediction mask at {int(overlap.sum())} voxels."
            )
        occupied |= labeled
        clean_atlases[str(atlas_file)] = atlas
    return clean_atlases, mask_array


def _map_regions(
    regions: Sequence[RegionDefinition],
    atlases: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    """Map masked voxels to retained region indices and the final fallback."""
    mask_array = np.asarray(mask, dtype=bool)
    fallback_index = len(regions)
    groups = np.full(int(mask_array.sum()), fallback_index, dtype=np.int32)
    already_assigned = np.zeros(groups.shape, dtype=bool)
    masked_by_atlas = {
        atlas_file: np.asarray(atlas, dtype=np.int32)[mask_array]
        for atlas_file, atlas in atlases.items()
    }
    for group_index, region in enumerate(regions):
        matches = masked_by_atlas[region.atlas_file] == int(region.label)
        if np.any(matches & already_assigned):
            raise ValueError(f"Retained HCP-MMP regions overlap at {region.name}.")
        groups[matches] = int(group_index)
        already_assigned |= matches
    return groups


@dataclass(frozen=True)
class RegionRegistry:
    """Training-fitted region schema shared by every fusion subject."""

    training_subjects: tuple[int, ...]
    atlas_files: tuple[str, ...]
    min_voxels_per_subject: int
    regions: tuple[RegionDefinition, ...]
    subject_fingerprints: dict[str, dict]

    @property
    def fallback_index(self) -> int:
        return len(self.regions)

    @property
    def n_groups(self) -> int:
        return len(self.regions) + 1

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.regions) + (FALLBACK_REGION_NAME,)

    def _payload(self) -> dict:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "training_subjects": [int(subject) for subject in self.training_subjects],
            "atlas_files": list(self.atlas_files),
            "min_voxels_per_subject": int(self.min_voxels_per_subject),
            "regions": [asdict(region) for region in self.regions],
            "fallback": {
                "index": int(self.fallback_index),
                "name": FALLBACK_REGION_NAME,
            },
            "subject_fingerprints": self.subject_fingerprints,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_json_sha256(self._payload())

    def to_manifest(self) -> dict:
        manifest = self._payload()
        manifest["registry_fingerprint"] = self.fingerprint
        return manifest

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_manifest(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RegionRegistry":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        expected_keys = {
            "schema_version",
            "training_subjects",
            "atlas_files",
            "min_voxels_per_subject",
            "regions",
            "fallback",
            "subject_fingerprints",
            "registry_fingerprint",
        }
        if set(manifest) != expected_keys:
            raise ValueError(
                f"Region registry fields are incompatible: expected {sorted(expected_keys)}, "
                f"got {sorted(manifest)}."
            )
        if int(manifest["schema_version"]) != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported region registry schema {manifest['schema_version']}; "
                f"expected {REGISTRY_SCHEMA_VERSION}."
            )

        regions = tuple(
            RegionDefinition(
                atlas_file=str(row["atlas_file"]),
                label=int(row["label"]),
                name=str(row["name"]),
            )
            for row in manifest["regions"]
        )
        registry = cls(
            training_subjects=tuple(int(value) for value in manifest["training_subjects"]),
            atlas_files=tuple(str(value) for value in manifest["atlas_files"]),
            min_voxels_per_subject=int(manifest["min_voxels_per_subject"]),
            regions=regions,
            subject_fingerprints={
                str(subject): dict(values)
                for subject, values in manifest["subject_fingerprints"].items()
            },
        )
        registry._validate_definition(manifest["fallback"])
        if str(manifest["registry_fingerprint"]) != registry.fingerprint:
            raise ValueError("Region registry fingerprint mismatch; manifest is stale or modified.")
        return registry

    def _validate_definition(self, fallback: Mapping[str, object]) -> None:
        if self.atlas_files != HCP_MMP_ATLAS_FILES:
            raise ValueError(
                f"Region registry atlas order must be {HCP_MMP_ATLAS_FILES}, "
                f"got {self.atlas_files}."
            )
        if not self.training_subjects:
            raise ValueError("Region registry requires at least one training subject.")
        if tuple(sorted(set(self.training_subjects))) != self.training_subjects:
            raise ValueError("Region registry training subjects must be unique and sorted.")
        if self.min_voxels_per_subject < 1:
            raise ValueError("min_voxels_per_subject must be positive.")

        atlas_order = {name: index for index, name in enumerate(self.atlas_files)}
        region_keys = [(region.atlas_file, int(region.label)) for region in self.regions]
        if len(region_keys) != len(set(region_keys)):
            raise ValueError("Region registry contains duplicate atlas/label pairs.")
        if any(atlas_file not in atlas_order or label <= 0 for atlas_file, label in region_keys):
            raise ValueError("Region registry contains an invalid atlas file or non-positive label.")
        expected_order = sorted(region_keys, key=lambda key: (atlas_order[key[0]], key[1]))
        if region_keys != expected_order:
            raise ValueError("Region registry entries are not in deterministic atlas/label order.")
        for region in self.regions:
            if region != RegionDefinition.create(region.atlas_file, region.label):
                raise ValueError(f"Region registry has a non-canonical name: {region}.")

        expected_fallback = {"index": self.fallback_index, "name": FALLBACK_REGION_NAME}
        if dict(fallback) != expected_fallback:
            raise ValueError(
                f"Region registry fallback is incompatible: expected {expected_fallback}."
            )
        expected_subject_keys = {str(subject) for subject in self.training_subjects}
        if set(self.subject_fingerprints) != expected_subject_keys:
            raise ValueError(
                "Region registry subject fingerprints do not match the training subjects."
            )

    def group_indices(
        self,
        subject: int,
        atlases: Mapping[str, np.ndarray],
        mask: np.ndarray,
        *,
        require_registered: bool = False,
    ) -> np.ndarray:
        """Return one group index per fMRI column, validating saved data when known."""
        subject = int(subject)
        clean_atlases, clean_mask = _validate_subject_arrays(
            subject,
            self.atlas_files,
            atlases,
            mask,
        )
        groups = _map_regions(self.regions, clean_atlases, clean_mask)
        key = str(subject)
        if key not in self.subject_fingerprints:
            if require_registered:
                raise ValueError(
                    f"Subject {subject} is not one of the registry training subjects "
                    f"{self.training_subjects}."
                )
            return groups

        actual = _subject_fingerprint(clean_atlases, clean_mask, groups)
        expected = self.subject_fingerprints[key]
        if actual != expected:
            differing = sorted(
                field for field in set(actual) | set(expected) if actual.get(field) != expected.get(field)
            )
            raise ValueError(
                f"Subject {subject} region inputs do not match the saved registry "
                f"fingerprints (different: {differing})."
            )
        return groups


def _subject_fingerprint(
    atlases: Mapping[str, np.ndarray],
    mask: np.ndarray,
    groups: np.ndarray,
) -> dict:
    return {
        "num_voxels": int(np.asarray(mask, dtype=bool).sum()),
        "mask_sha256": _mask_sha256(mask),
        "atlas_sha256": {
            atlas_file: _atlas_sha256(atlas)
            for atlas_file, atlas in sorted(atlases.items())
        },
        "group_map_sha256": _group_map_sha256(groups),
    }


def build_region_registry(
    training_subjects: Sequence[int],
    atlases_by_subject: Mapping[int, Mapping[str, np.ndarray]],
    masks_by_subject: Mapping[int, np.ndarray],
    *,
    min_voxels_per_subject: int = 20,
    atlas_files: Sequence[str] = HCP_MMP_ATLAS_FILES,
) -> RegionRegistry:
    """Fit retained HCP-MMP parcels using only the supplied training subjects."""
    subjects = tuple(sorted(int(subject) for subject in training_subjects))
    if not subjects:
        raise ValueError("At least one training subject is required.")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Training subjects must be unique.")
    atlas_files_tuple = tuple(str(name) for name in atlas_files)
    if atlas_files_tuple != HCP_MMP_ATLAS_FILES:
        raise ValueError(
            f"HCP-MMP region fusion requires atlas_files={HCP_MMP_ATLAS_FILES}."
        )
    if int(min_voxels_per_subject) < 1:
        raise ValueError("min_voxels_per_subject must be positive.")

    clean_inputs: dict[int, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    valid_by_subject: list[set[tuple[str, int]]] = []
    for subject in subjects:
        if subject not in atlases_by_subject or subject not in masks_by_subject:
            raise ValueError(f"Missing HCP-MMP atlas or mask arrays for subject {subject}.")
        atlases, mask = _validate_subject_arrays(
            subject,
            atlas_files_tuple,
            atlases_by_subject[subject],
            masks_by_subject[subject],
        )
        clean_inputs[subject] = (atlases, mask)

        valid: set[tuple[str, int]] = set()
        for atlas_file in atlas_files_tuple:
            masked_labels = atlases[atlas_file][mask]
            labels, counts = np.unique(masked_labels[masked_labels > 0], return_counts=True)
            valid.update(
                (atlas_file, int(label))
                for label, count in zip(labels, counts)
                if int(count) >= int(min_voxels_per_subject)
            )
        valid_by_subject.append(valid)

    retained = set.intersection(*valid_by_subject)
    atlas_order = {name: index for index, name in enumerate(atlas_files_tuple)}
    retained_order = sorted(retained, key=lambda key: (atlas_order[key[0]], key[1]))
    regions = tuple(
        RegionDefinition.create(atlas_file, label)
        for atlas_file, label in retained_order
    )
    if not regions:
        raise ValueError(
            "No HCP-MMP parcels meet the minimum voxel count in every training subject."
        )

    subject_fingerprints: dict[str, dict] = {}
    for subject in subjects:
        atlases, mask = clean_inputs[subject]
        groups = _map_regions(regions, atlases, mask)
        subject_fingerprints[str(subject)] = _subject_fingerprint(atlases, mask, groups)

    registry = RegionRegistry(
        training_subjects=subjects,
        atlas_files=atlas_files_tuple,
        min_voxels_per_subject=int(min_voxels_per_subject),
        regions=regions,
        subject_fingerprints=subject_fingerprints,
    )
    registry._validate_definition(
        {"index": registry.fallback_index, "name": FALLBACK_REGION_NAME}
    )
    return registry


def load_hcp_mmp_subject_arrays(
    subject: int,
    *,
    data_root: str | Path,
    raw_data_root: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load hemisphere atlases and the processed prediction mask for one subject."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Loading HCP-MMP NIfTI atlases requires nibabel.") from exc

    subject = int(subject)
    subject_tag = f"subj{subject:02d}"
    mask_path = Path(data_root) / subject_tag / "mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Processed prediction mask not found: {mask_path}")
    mask = np.asarray(np.load(mask_path), dtype=bool)

    roi_root = (
        Path(raw_data_root)
        / "nsddata"
        / "ppdata"
        / subject_tag
        / "func1pt8mm"
        / "roi"
    )
    atlases: dict[str, np.ndarray] = {}
    for atlas_file in HCP_MMP_ATLAS_FILES:
        atlas_path = roi_root / atlas_file
        if not atlas_path.exists():
            raise FileNotFoundError(
                f"Required HCP-MMP atlas not found: {atlas_path}. "
                "Run `python -m src.data.download_nsddata --only-rois`."
            )
        atlases[atlas_file] = np.asarray(
            nib.load(str(atlas_path)).dataobj,
            dtype=np.int32,
        )
    clean_atlases, clean_mask = _validate_subject_arrays(
        subject,
        HCP_MMP_ATLAS_FILES,
        atlases,
        mask,
    )
    return clean_atlases, clean_mask


def build_hcp_mmp_region_registry(
    training_subjects: Sequence[int],
    *,
    data_root: str | Path,
    raw_data_root: str | Path,
    min_voxels_per_subject: int = 20,
) -> RegionRegistry:
    """Load local NSD arrays and fit the training-only HCP-MMP registry."""
    subjects = tuple(sorted(int(subject) for subject in training_subjects))
    atlases_by_subject: dict[int, dict[str, np.ndarray]] = {}
    masks_by_subject: dict[int, np.ndarray] = {}
    for subject in subjects:
        atlases, mask = load_hcp_mmp_subject_arrays(
            subject,
            data_root=data_root,
            raw_data_root=raw_data_root,
        )
        atlases_by_subject[subject] = atlases
        masks_by_subject[subject] = mask
    return build_region_registry(
        subjects,
        atlases_by_subject,
        masks_by_subject,
        min_voxels_per_subject=min_voxels_per_subject,
    )


def load_subject_region_groups(
    registry: RegionRegistry,
    subject: int,
    *,
    data_root: str | Path,
    raw_data_root: str | Path,
    require_registered: bool = False,
) -> np.ndarray:
    """Load one subject's atlas inputs and map fMRI columns to fusion groups."""
    atlases, mask = load_hcp_mmp_subject_arrays(
        subject,
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    return registry.group_indices(
        subject,
        atlases,
        mask,
        require_registered=require_registered,
    )
