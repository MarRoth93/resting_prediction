"""Register the official Schaefer-400/7-network atlas into NSD func1pt8 space."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_annot

from src.data.schaefer400 import N_PARCELS, validate_schaefer400_atlas
from src.schaefer400_config import (
    ATLAS_NAME,
    load_schaefer400_config,
    resolve_schaefer400_roots,
)


logger = logging.getLogger(__name__)

CBIG_COMMIT = "cb2e5bd8f5587485669f14e723c691ba83d0ae26"
CBIG_ANNOTATION_ROOT = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
    f"{CBIG_COMMIT}/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label"
)
NSD_PUBLIC_ROOT = "https://natural-scenes-dataset.s3.amazonaws.com"
ANNOTATION_SHA256 = {
    "lh": "351848da486c2b178ef825640c076fcbcf054095bed0c91d5854d2c593e5c1ee",
    "rh": "595444aecc3b59a8a6838b2e5c7f2052e88e25b1c5f833c5a6bdb984e3fa978f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    allow_download: bool = True,
) -> Path:
    if destination.exists():
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        raise ValueError(f"Cached file checksum is invalid: {destination}")
    if not allow_download:
        raise FileNotFoundError(
            f"Missing registration asset {destination}. Re-run without --no-download."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    logger.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, temporary)
        if expected_sha256 is not None and _sha256(temporary) != expected_sha256:
            raise ValueError(f"Downloaded file checksum is invalid: {url}")
        os.replace(temporary, destination)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        temporary.unlink(missing_ok=True)
        raise
    return destination


def registration_asset_paths(
    *,
    subject: int,
    data_root: str | Path,
    layers: Iterable[int] = (1, 2, 3),
) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    """Return stable cache locations for annotations and NSD transforms."""
    root = Path(data_root) / "_registration_assets"
    annotations = {
        hemi: root
        / "annotations"
        / f"{hemi}.Schaefer2018_400Parcels_7Networks_order.annot"
        for hemi in ("lh", "rh")
    }
    transforms: dict[tuple[str, str], Path] = {}
    for hemi in ("lh", "rh"):
        transforms[(hemi, "fsaverage-to-white")] = (
            root
            / f"subj{int(subject):02d}"
            / "transforms"
            / f"{hemi}.fsaverage-to-white.mgz"
        )
        for layer in layers:
            name = f"func1pt8-to-layerB{int(layer)}"
            transforms[(hemi, name)] = (
                root
                / f"subj{int(subject):02d}"
                / "transforms"
                / f"{hemi}.{name}.mgz"
            )
    return annotations, transforms


def ensure_registration_assets(
    *,
    subject: int,
    data_root: str | Path,
    layers: Iterable[int] = (1, 2, 3),
    allow_download: bool = True,
) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    """Fetch only the small files required for surface-to-functional mapping."""
    layers = tuple(int(layer) for layer in layers)
    annotations, transforms = registration_asset_paths(
        subject=subject,
        data_root=data_root,
        layers=layers,
    )
    for hemi, path in annotations.items():
        _download(
            f"{CBIG_ANNOTATION_ROOT}/{path.name}",
            path,
            expected_sha256=ANNOTATION_SHA256[hemi],
            allow_download=allow_download,
        )
    for (hemi, name), path in transforms.items():
        _download(
            f"{NSD_PUBLIC_ROOT}/nsddata/ppdata/subj{int(subject):02d}/"
            f"transforms/{hemi}.{name}.mgz",
            path,
            allow_download=allow_download,
        )
        # Loading here catches truncated S3 responses before the expensive map.
        nib.load(path)
    return annotations, transforms


def _native_parcel_labels(
    annotation_path: Path,
    fsaverage_to_white_path: Path,
    *,
    parcel_offset: int,
) -> np.ndarray:
    fs_labels, _, names = read_annot(str(annotation_path), orig_ids=False)
    if (
        fs_labels.ndim != 1
        or int(fs_labels.min()) != 0
        or int(fs_labels.max()) != 200
        or len(names) != 201
    ):
        raise ValueError(f"Unexpected Schaefer annotation contract: {annotation_path}")
    mapping = np.asarray(nib.load(fsaverage_to_white_path).dataobj).reshape(-1)
    if not np.all(np.isfinite(mapping)) or not np.allclose(mapping, np.rint(mapping)):
        raise ValueError(f"Non-integer fsaverage mapping: {fsaverage_to_white_path}")
    mapping = np.rint(mapping).astype(np.int64) - 1
    if np.any(mapping < 0) or np.any(mapping >= fs_labels.size):
        raise ValueError(f"Out-of-range fsaverage mapping: {fsaverage_to_white_path}")
    native = fs_labels[mapping].astype(np.int16)
    native[native > 0] += int(parcel_offset)
    return native


def rasterize_surface_labels_wta(
    surfaces: Iterable[tuple[np.ndarray, np.ndarray]],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Map discrete native-surface labels with NSD's surface-WTA weighting."""
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 3 or any(value < 1 for value in target_shape):
        raise ValueError(f"Invalid target_shape: {target_shape}")
    label_parts: list[np.ndarray] = []
    coordinate_parts: list[np.ndarray] = []
    for raw_labels, raw_coordinates in surfaces:
        labels = np.asarray(raw_labels, dtype=np.int16).reshape(-1)
        coordinates = np.asarray(raw_coordinates, dtype=np.float32).reshape(-1, 3)
        if labels.shape[0] != coordinates.shape[0]:
            raise ValueError("A surface has different label and coordinate counts.")
        if np.any(labels < 0) or np.any(labels > N_PARCELS):
            raise ValueError("Surface labels must use canonical IDs 0..400.")
        finite = np.all(np.isfinite(coordinates), axis=1)
        label_parts.append(labels[finite])
        coordinate_parts.append(coordinates[finite])
    if not label_parts:
        raise ValueError("At least one labeled surface is required.")

    labels = np.concatenate(label_parts)
    coordinates = np.concatenate(coordinate_parts)
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    coordinates = coordinates[order]
    bounds = np.searchsorted(labels, np.arange(1, N_PARCELS + 2), side="left")

    n_voxels = int(np.prod(target_shape))
    best_weight = np.zeros(n_voxels, dtype=np.float32)
    output = np.zeros(n_voxels, dtype=np.int16)
    for parcel_id in range(1, N_PARCELS + 1):
        xyz = coordinates[bounds[parcel_id - 1] : bounds[parcel_id]]
        if xyz.size == 0:
            raise ValueError(f"Surface mapping contains no vertices for parcel {parcel_id}.")
        voxel_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for upper_x in (False, True):
            x = np.ceil(xyz[:, 0]) if upper_x else np.floor(xyz[:, 0])
            dx = x - xyz[:, 0] if upper_x else xyz[:, 0] - x
            for upper_y in (False, True):
                y = np.ceil(xyz[:, 1]) if upper_y else np.floor(xyz[:, 1])
                dy = y - xyz[:, 1] if upper_y else xyz[:, 1] - y
                for upper_z in (False, True):
                    z = np.ceil(xyz[:, 2]) if upper_z else np.floor(xyz[:, 2])
                    dz = z - xyz[:, 2] if upper_z else xyz[:, 2] - z
                    xi = x.astype(np.int64) - 1
                    yi = y.astype(np.int64) - 1
                    zi = z.astype(np.int64) - 1
                    valid = (
                        (xi >= 0)
                        & (xi < target_shape[0])
                        & (yi >= 0)
                        & (yi < target_shape[1])
                        & (zi >= 0)
                        & (zi < target_shape[2])
                    )
                    voxel_parts.append(
                        np.ravel_multi_index(
                            (xi[valid], yi[valid], zi[valid]),
                            target_shape,
                            order="F",
                        )
                    )
                    # This mirrors the public NSD nsd_mapdata surfacewta code.
                    weight_parts.append(
                        ((1 - dx[valid]) + (1 - dy[valid]) + (1 - dz[valid])).astype(
                            np.float32
                        )
                    )
        voxels = np.concatenate(voxel_parts)
        weights = np.concatenate(weight_parts)
        unique_voxels, inverse = np.unique(voxels, return_inverse=True)
        totals = np.bincount(inverse, weights=weights).astype(np.float32)
        wins = totals > best_weight[unique_voxels]
        winning_voxels = unique_voxels[wins]
        best_weight[winning_voxels] = totals[wins]
        output[winning_voxels] = parcel_id
    return output.reshape(target_shape, order="F")


def prepare_schaefer400_atlas(
    subject: int,
    config: dict,
    *,
    allow_download: bool = True,
    force: bool = False,
) -> dict:
    """Create one complete subject-native NSD Schaefer atlas and provenance."""
    subject = int(subject)
    data_root = Path(config["data_root"])
    raw_data_root = Path(config["raw_data_root"])
    atlas_cfg = config["atlas"]
    destination_dir = data_root / f"subj{subject:02d}"
    destination = destination_dir / str(atlas_cfg["nsd_filename"])
    summary_path = destination_dir / "schaefer400_atlas_summary.json"
    if destination.exists() and not force:
        image = nib.load(destination)
        counts = validate_schaefer400_atlas(
            np.asarray(image.dataobj),
            min_voxels_per_parcel=int(atlas_cfg["min_voxels_per_parcel"]),
        )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Atlas exists without its provenance summary: {summary_path}"
            )
        summary = json.loads(summary_path.read_text())
        if summary.get("parcel_voxel_counts") != counts.astype(int).tolist():
            raise ValueError(f"Atlas and provenance counts differ: {destination}")
        return summary

    reference_path = (
        raw_data_root
        / "nsddata"
        / "ppdata"
        / f"subj{subject:02d}"
        / "func1pt8mm"
        / "roi"
        / "nsdgeneral.nii.gz"
    )
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing NSD func1pt8 reference: {reference_path}")
    reference = nib.load(reference_path)
    layers = tuple(int(value) for value in atlas_cfg["cortical_layers"])
    annotations, transforms = ensure_registration_assets(
        subject=subject,
        data_root=data_root,
        layers=layers,
        allow_download=allow_download,
    )

    surfaces: list[tuple[np.ndarray, np.ndarray]] = []
    for hemi, offset in (("lh", 0), ("rh", 200)):
        native_labels = _native_parcel_labels(
            annotations[hemi],
            transforms[(hemi, "fsaverage-to-white")],
            parcel_offset=offset,
        )
        for layer in layers:
            transform_path = transforms[(hemi, f"func1pt8-to-layerB{layer}")]
            coordinates = np.asarray(nib.load(transform_path).dataobj).reshape(-1, 3)
            surfaces.append((native_labels, coordinates))

    labels = rasterize_surface_labels_wta(surfaces, tuple(reference.shape))
    counts = validate_schaefer400_atlas(
        labels,
        min_voxels_per_parcel=int(atlas_cfg["min_voxels_per_parcel"]),
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(labels, reference.affine, header), destination)
    summary = {
        "subject": subject,
        "atlas_name": ATLAS_NAME,
        "representation": "canonical_parcel_ids_1_to_400",
        "networks": 7,
        "target_space": "NSD func1pt8",
        "target_shape": [int(value) for value in reference.shape],
        "target_zooms_mm": [float(value) for value in reference.header.get_zooms()[:3]],
        "reference_file": str(reference_path.resolve()),
        "atlas_file": str(destination.resolve()),
        "registration_method": "fsaverage labels to native surface, three-layer surface WTA",
        "registration_source": "official CBIG annotations and official NSD transforms",
        "cbig_commit": CBIG_COMMIT,
        "cortical_layers": list(layers),
        "labeled_voxels": int((labels > 0).sum()),
        "parcel_voxel_count_min": int(counts.min()),
        "parcel_voxel_count_median": float(np.median(counts)),
        "parcel_voxel_count_max": int(counts.max()),
        "parcel_voxel_counts": counts.astype(int).tolist(),
        "annotation_sha256": {
            hemi: _sha256(path) for hemi, path in annotations.items()
        },
        "transform_files": {
            f"{hemi}.{name}": str(path.resolve())
            for (hemi, name), path in transforms.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Subject %d: wrote %s (%d voxels, all 400 parcels)",
        subject,
        destination,
        int((labels > 0).sum()),
    )
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_schaefer400.yaml")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = resolve_schaefer400_roots(
        load_schaefer400_config(args.config),
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
    )
    subjects = args.subjects or cfg["subjects"]["train"]
    for subject_id in subjects:
        prepare_schaefer400_atlas(
            subject_id,
            cfg,
            allow_download=not args.no_download,
            force=args.force,
        )
