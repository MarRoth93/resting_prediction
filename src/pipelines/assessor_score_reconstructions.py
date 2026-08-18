"""Score reconstruction images with the frozen assessor bundles."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import logging
import os
import pickle
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.data.shared_paths import default_stimuli_hdf5
from src.pipelines.for_assessor_study import (
    _assessor_paths,
    _path,
    load_study_config,
    validate_frozen_assessor,
)
from src.pipelines.multiexpert_artifacts import file_sha256, json_fingerprint


logger = logging.getLogger(__name__)

DIMENSION_MODEL = {
    "valence": "va",
    "arousal": "va",
    "approach": "six",
    "attention": "six",
    "control": "six",
    "dominance": "six",
}
_IMAGE_NAME = re.compile(
    r"^(?:row(?P<row>\d+)_)?stim(?P<stimulus>\d+)\.png$"
)


def _load_assessor_module(source_path: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_assessor_reconstruction_scoring", source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import frozen assessor: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    os.replace(temporary, path)


def _image_dir_sha_sample(paths: list[Path]) -> str:
    sample_size = min(5, len(paths))
    indices = np.linspace(0, len(paths) - 1, sample_size, dtype=int)
    sample = [
        {"name": paths[int(index)].name, "sha256": file_sha256(paths[int(index)])}
        for index in indices
    ]
    return json_fingerprint(sample)


def _parse_reconstruction_images(
    subject: str,
    image_dir: Path,
    expected_stimulus_ids: np.ndarray,
) -> tuple[list[Path], np.ndarray] | None:
    parsed = []
    for path in image_dir.glob("row*_stim*.png") if image_dir.is_dir() else []:
        match = _IMAGE_NAME.fullmatch(path.name)
        if match is not None and match.group("row") is not None:
            parsed.append(
                (int(match.group("row")), int(match.group("stimulus")), path)
            )
    parsed.sort(key=lambda item: item[0])
    rows = [item[0] for item in parsed]
    stimulus_ids = np.asarray([item[1] for item in parsed], dtype=np.int64)
    expected_ids = np.asarray(expected_stimulus_ids, dtype=np.int64)
    complete = (
        len(parsed) == expected_ids.size
        and rows == list(range(expected_ids.size))
        and np.array_equal(np.sort(stimulus_ids), np.sort(expected_ids))
        and np.unique(stimulus_ids).size == stimulus_ids.size
    )
    if not complete:
        logger.warning(
            "Skipping %s: incomplete reconstruction images in %s (%d/%d).",
            subject,
            image_dir,
            len(parsed),
            expected_ids.size,
        )
        return None
    return [item[2] for item in parsed], stimulus_ids


def _predict_rows(
    assessor,
    paths: list[Path],
    *,
    batch_size: int,
    interval: float | None,
    ood: bool,
) -> dict[str, dict[str, float]]:
    value_columns = []
    for target in assessor.targets:
        value_columns.append(target)
        if interval is not None:
            value_columns.extend([f"{target}_low", f"{target}_high"])
    if ood:
        value_columns.append("ood_percentile")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["image", *value_columns])
    writer.writeheader()
    for start in range(0, len(paths), int(batch_size)):
        chunk = paths[start : start + int(batch_size)]
        results = assessor.predict_paths(
            chunk,
            batch_size=batch_size,
            interval=interval,
            ood=ood,
        )
        for path in chunk:
            writer.writerow({"image": str(path), **results[str(path)]})

    buffer.seek(0)
    parsed = {}
    for row in csv.DictReader(buffer):
        parsed[row["image"]] = {
            key.lower(): float(value) for key, value in row.items() if key != "image"
        }
    if len(parsed) != len(paths):
        raise RuntimeError("Incomplete or duplicate frozen-assessor scores.")
    return parsed


def _score_payload(
    *,
    subject: str,
    paths: list[Path],
    stimulus_ids: np.ndarray,
    assessors: dict,
    frozen: dict,
    interval: float,
    batch_size: int,
) -> dict:
    rows_by_model = {
        "va": _predict_rows(
            assessors["va"],
            paths,
            batch_size=batch_size,
            interval=interval,
            ood=True,
        ),
        "six": _predict_rows(
            assessors["six"],
            paths,
            batch_size=batch_size,
            interval=None,
            ood=False,
        ),
    }
    scores = {}
    for dimension, model_name in DIMENSION_MODEL.items():
        values = np.asarray(
            [rows_by_model[model_name][str(path)][dimension] for path in paths],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{subject}: {dimension} contains NaN/Inf.")
        scores[dimension] = values

    va_extras = {}
    for dimension in ("valence", "arousal"):
        rows = rows_by_model["va"]
        va_extras[dimension] = {
            "interval_low": np.asarray(
                [rows[str(path)][f"{dimension}_low"] for path in paths],
                dtype=np.float32,
            ),
            "interval_high": np.asarray(
                [rows[str(path)][f"{dimension}_high"] for path in paths],
                dtype=np.float32,
            ),
            "ood_percentile": np.asarray(
                [rows[str(path)]["ood_percentile"] for path in paths],
                dtype=np.float32,
            ),
        }

    return {
        "subject": subject,
        "stimulus_ids": np.asarray(stimulus_ids, dtype=np.int64),
        "scores": scores,
        "va_extras": va_extras,
        "bundle_shas": {
            name: frozen["bundles"][name]["sha256"] for name in ("va", "six")
        },
        "image_dir_sha_sample": _image_dir_sha_sample(paths),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _discover_subjects(recon_root: Path) -> list[str]:
    if not recon_root.is_dir():
        raise FileNotFoundError(f"Missing reconstruction root: {recon_root}")
    return sorted(
        path.name
        for path in recon_root.iterdir()
        if path.is_dir() and (path.name == "subj07" or path.name.startswith("sub-"))
    )


def _extract_original_images(
    stimulus_ids: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    stimuli_path = Path(default_stimuli_hdf5())
    if not stimuli_path.is_file():
        raise FileNotFoundError(f"Missing NSD stimuli HDF5: {stimuli_path}")
    paths = []
    with h5py.File(stimuli_path, "r") as handle:
        images = handle["imgBrick"]
        for stimulus_id in np.asarray(stimulus_ids, dtype=np.int64).tolist():
            destination = output_dir / f"stim{int(stimulus_id):05d}.png"
            Image.fromarray(
                np.asarray(images[int(stimulus_id)], dtype=np.uint8)
            ).save(destination, format="PNG")
            paths.append(destination)
    return paths


def score_reconstructions(
    *,
    recon_root: Path,
    subjects: list[str] | None,
    output_root: Path,
    config: dict,
    device: str,
    batch_size: int,
    include_originals: bool,
    force: bool,
) -> dict:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    frozen = validate_frozen_assessor(config)
    selected_subjects = _discover_subjects(recon_root) if subjects is None else subjects
    selected_subjects = list(dict.fromkeys(str(subject) for subject in selected_subjects))
    selection_dir = _path(config, config["inputs"]["prediction_selection_dir"])
    stimulus_ids_path = selection_dir / "nsd_stimulus_ids.npy"
    expected_stimulus_ids = np.asarray(
        np.load(stimulus_ids_path), dtype=np.int64
    ).reshape(-1)

    jobs = []
    skipped_existing = []
    skipped_incomplete = []
    for subject in selected_subjects:
        output_path = output_root / f"{subject}.pkl"
        if output_path.exists() and not force:
            logger.info("Skipping existing score pickle: %s", output_path)
            skipped_existing.append(subject)
            continue
        parsed = _parse_reconstruction_images(
            subject,
            recon_root / subject / "images_vdvae",
            expected_stimulus_ids,
        )
        if parsed is None:
            skipped_incomplete.append(subject)
            continue
        paths, stimulus_ids = parsed
        jobs.append((subject, paths, stimulus_ids, output_path))

    originals_path = output_root / "originals.pkl"
    score_originals = include_originals and (force or not originals_path.exists())
    if include_originals and not score_originals:
        logger.info("Skipping existing score pickle: %s", originals_path)
        skipped_existing.append("originals")

    if not jobs and not score_originals:
        return {
            "scored": [],
            "skipped_existing": skipped_existing,
            "skipped_incomplete": skipped_incomplete,
            "output_root": str(output_root),
        }

    assessor_paths = _assessor_paths(config)
    module = _load_assessor_module(assessor_paths["source"])
    assessors = {
        name: module.Assessor(
            bundle_path=assessor_paths["bundles"][name], device=device
        )
        for name in ("va", "six")
    }
    interval = float(config["assessor"]["interval"])
    scored = []
    for subject, paths, stimulus_ids, output_path in jobs:
        payload = _score_payload(
            subject=subject,
            paths=paths,
            stimulus_ids=stimulus_ids,
            assessors=assessors,
            frozen=frozen,
            interval=interval,
            batch_size=batch_size,
        )
        _write_pickle_atomic(output_path, payload)
        scored.append(subject)
        logger.info("Saved frozen-assessor scores to %s", output_path)

    if score_originals:
        with tempfile.TemporaryDirectory(prefix="assessor-originals-") as temporary:
            original_paths = _extract_original_images(
                expected_stimulus_ids, Path(temporary)
            )
            payload = _score_payload(
                subject="originals",
                paths=original_paths,
                stimulus_ids=expected_stimulus_ids,
                assessors=assessors,
                frozen=frozen,
                interval=interval,
                batch_size=batch_size,
            )
        _write_pickle_atomic(originals_path, payload)
        scored.append("originals")
        logger.info("Saved frozen-assessor scores to %s", originals_path)

    return {
        "scored": scored,
        "skipped_existing": skipped_existing,
        "skipped_incomplete": skipped_incomplete,
        "output_root": str(output_root),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recon-root", default="artifacts/recon_from_predictions/seed42"
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--output-root",
        default="",
        help="Defaults to <recon-root>/assessor_scores.",
    )
    parser.add_argument("--config", default="config_for_assessor_study.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--include-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    recon_root = Path(args.recon_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else recon_root / "assessor_scores"
    )
    result = score_reconstructions(
        recon_root=recon_root,
        subjects=args.subjects,
        output_root=output_root,
        config=load_study_config(args.config),
        device=str(args.device),
        batch_size=int(args.batch_size),
        include_originals=bool(args.include_originals),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
