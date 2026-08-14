"""Frozen-assessor scoring and subject-level comparisons for the FOR study."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import yaml
from PIL import Image

from src.pipelines.multiexpert_artifacts import file_sha256, json_fingerprint


DIMENSION_MODEL = {
    "valence": "va",
    "arousal": "va",
    "approach": "six",
    "attention": "six",
    "control": "six",
    "dominance": "six",
}
STIMULUS_RE = re.compile(r"stim(?P<stimulus>\d+)\.png$")
ARTIFACT_VERSION = 1


def load_study_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Study configuration must be a mapping.")
    config["_config_path"] = str(path)
    config["_root"] = str(path.parent)
    return config


def _path(config: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    return path.resolve()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _assessor_paths(config: dict) -> dict:
    assessor = config["assessor"]
    root = _path(config, assessor["project_root"])
    bundles = {
        name: root / entry["path"]
        for name, entry in assessor["bundles"].items()
    }
    return {"root": root, "source": root / "assessor.py", "bundles": bundles}


def validate_frozen_assessor(config: dict) -> dict:
    paths = _assessor_paths(config)
    assessor = config["assessor"]
    if file_sha256(paths["source"]) != assessor["source_sha256"]:
        raise ValueError("Frozen assessor.py checksum does not match the study contract.")
    for name, path in paths["bundles"].items():
        expected = assessor["bundles"][name]["sha256"]
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen assessor bundle checksum is invalid: {name}")
    return {
        "source": str(paths["source"]),
        "source_sha256": assessor["source_sha256"],
        "bundles": {
            name: {"path": str(path), "sha256": assessor["bundles"][name]["sha256"]}
            for name, path in paths["bundles"].items()
        },
    }


def load_groups(config: dict) -> dict[str, str]:
    path = _path(config, config["inputs"]["group_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FOR group file: {path}. Copy for_groups.example.csv and fill all subjects."
        )
    allowed = set(config["study"]["groups"])
    groups: dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"subject", "group"}.issubset(reader.fieldnames):
            raise ValueError("FOR group CSV must contain subject and group columns.")
        for row in reader:
            subject = str(row["subject"]).strip()
            group = str(row["group"]).strip().lower()
            if not subject or group not in allowed or subject in groups:
                raise ValueError(f"Invalid or duplicate FOR group row: {row}")
            groups[subject] = group
    prediction_root = _path(config, config["inputs"]["for_prediction_root"])
    expected = sorted(path.name for path in prediction_root.glob("sub-*") if path.is_dir())
    if sorted(groups) != expected:
        missing = sorted(set(expected) - set(groups))
        extra = sorted(set(groups) - set(expected))
        raise ValueError(f"FOR group membership mismatch; missing={missing}, extra={extra}")
    if set(groups.values()) != allowed:
        raise ValueError(f"Both configured groups are required: {sorted(allowed)}")
    return groups


def check_study(config: dict) -> dict:
    assessor = validate_frozen_assessor(config)
    groups = load_groups(config)
    inputs = config["inputs"]
    required = [
        _path(config, inputs["stimuli_hdf5"]),
        _path(config, inputs["prediction_selection_dir"]) / "nsd_stimulus_ids.npy",
        _path(config, inputs["for_prediction_root"]),
        _path(config, config["decoder_validation"]["output_dir"]).parent,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    counts = {group: list(groups.values()).count(group) for group in config["study"]["groups"]}
    return {
        "status": "ok",
        "for_subjects": len(groups),
        "group_counts": counts,
        "frozen_assessor": assessor,
        "decoder_validation_subjects": config["decoder_validation"]["subjects"],
        "primary_dimensions": config["study"]["primary_dimensions"],
        "exploratory_dimensions": config["study"]["exploratory_dimensions"],
    }


def _validation_stimulus_ids(config: dict) -> np.ndarray:
    manifest_path = (
        _path(config, config["decoder_validation"]["output_dir"]) / "split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    return np.asarray(manifest["split"]["final_stimulus_ids"], dtype=np.int64)


def _original_sets(config: dict) -> dict[str, tuple[np.ndarray, Path]]:
    study_root = _path(config, config["study"]["output_root"])
    selection_dir = _path(config, config["inputs"]["prediction_selection_dir"])
    return {
        "for_original": (
            np.asarray(np.load(selection_dir / "nsd_stimulus_ids.npy"), dtype=np.int64),
            study_root / "originals" / "for_selected",
        ),
        "nsd_original": (
            _validation_stimulus_ids(config),
            study_root / "originals" / "nsd_validation",
        ),
    }


def extract_originals(config: dict, set_names: Iterable[str]) -> dict:
    available = _original_sets(config)
    selected = [str(value) for value in set_names]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown original image sets: {unknown}")
    stimuli_path = _path(config, config["inputs"]["stimuli_hdf5"])
    reports = {}
    with h5py.File(stimuli_path, "r") as handle:
        images = handle["imgBrick"]
        for name in selected:
            stimulus_ids, output_dir = available[name]
            output_dir.mkdir(parents=True, exist_ok=True)
            for stimulus in stimulus_ids.tolist():
                destination = output_dir / f"stim{int(stimulus):05d}.png"
                if destination.exists():
                    continue
                temporary = destination.with_name(f".{destination.name}.part")
                Image.fromarray(np.asarray(images[int(stimulus)], dtype=np.uint8)).save(
                    temporary, format="PNG"
                )
                os.replace(temporary, destination)
            found = list(output_dir.glob("stim*.png"))
            if len(found) != stimulus_ids.size:
                raise RuntimeError(f"Incomplete original image extraction for {name}.")
            reports[name] = {"images": int(stimulus_ids.size), "directory": str(output_dir)}
    _write_json_atomic(
        _path(config, config["study"]["output_root"]) / "originals" / "manifest.json",
        {"artifact_version": ARTIFACT_VERSION, "sets": reports},
    )
    return reports


def _expected_validation_reconstructions(config: dict) -> int:
    root = _path(config, config["decoder_validation"]["output_dir"]) / "final"
    total = 0
    for subject in config["decoder_validation"]["subjects"]:
        metrics = json.loads((root / f"subj{int(subject):02d}" / "metrics.json").read_text())
        total += int(metrics["evaluation_rows"])
    return total


def image_sets(config: dict, set_names: Iterable[str]) -> dict[str, list[Path]]:
    original = _original_sets(config)
    for_recon_root = _path(config, config["inputs"]["for_reconstruction_root"])
    validation_root = _path(config, config["decoder_validation"]["output_dir"]) / "final"
    selected = [str(value) for value in set_names]
    factories = {
        "for_original": lambda: (
            sorted(original["for_original"][1].glob("stim*.png")),
            int(original["for_original"][0].size),
        ),
        "nsd_original": lambda: (
            sorted(original["nsd_original"][1].glob("stim*.png")),
            int(original["nsd_original"][0].size),
        ),
        "for_reconstruction": lambda: (
            sorted(for_recon_root.glob("sub-*/images_vdvae/*.png")),
            50 * int(original["for_original"][0].size),
        ),
        "nsd_reconstruction": lambda: (
            sorted(validation_root.glob("subj*/images_vdvae/*.png")),
            _expected_validation_reconstructions(config),
        ),
    }
    unknown = sorted(set(selected) - set(factories))
    if unknown:
        raise ValueError(f"Unknown assessor image sets: {unknown}")
    result = {}
    for name in selected:
        paths, expected = factories[name]()
        if len(paths) != expected:
            raise RuntimeError(f"Image set {name} is incomplete: {len(paths)}/{expected}")
        result[name] = paths
    return result


def _load_assessor_module(source_path: Path):
    spec = importlib.util.spec_from_file_location("frozen_assessor_for_study", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import frozen assessor: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_contract(paths: list[Path]) -> dict:
    entries = []
    for path in paths:
        stat = path.stat()
        entries.append((str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)))
    return {"count": len(entries), "fingerprint": json_fingerprint(entries)}


def _score_image_set(
    *,
    assessor,
    model_name: str,
    bundle_sha256: str,
    paths: list[Path],
    output_path: Path,
    interval: float | None,
    ood: bool,
    batch_size: int,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    contract = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model_name,
        "bundle_sha256": bundle_sha256,
        "interval": None if interval is None else float(interval),
        "ood": bool(ood),
        "images": _image_contract(paths),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise RuntimeError(f"Stale assessor score output: {output_path}")
        if manifest.get("status") == "complete":
            return
    elif output_path.exists():
        raise RuntimeError(f"Assessor CSV exists without provenance: {output_path}")
    else:
        _write_json_atomic(manifest_path, {"status": "in_progress", "contract": contract})

    completed: set[str] = set()
    if output_path.exists():
        with open(output_path, newline="") as handle:
            completed = {row["image"] for row in csv.DictReader(handle)}
    value_columns = []
    for target in assessor.targets:
        value_columns.append(target)
        if interval is not None:
            value_columns.extend([f"{target}_low", f"{target}_high"])
    if ood:
        value_columns.append("ood_percentile")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *value_columns])
        if mode == "w":
            writer.writeheader()
        pending = [path for path in paths if str(path) not in completed]
        for start in range(0, len(pending), int(batch_size)):
            chunk = pending[start : start + int(batch_size)]
            results = assessor.predict_paths(
                chunk,
                batch_size=batch_size,
                interval=interval,
                ood=ood,
            )
            for path in chunk:
                values = results[str(path)]
                writer.writerow({"image": str(path), **values})
            handle.flush()
            os.fsync(handle.fileno())
    with open(output_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(paths) or len({row["image"] for row in rows}) != len(paths):
        raise RuntimeError(f"Incomplete or duplicate assessor scores: {output_path}")
    _write_json_atomic(manifest_path, {"status": "complete", "contract": contract})


def score_images(config: dict, set_names: Iterable[str]) -> dict:
    frozen = validate_frozen_assessor(config)
    selected = [str(value) for value in set_names]
    all_sets = image_sets(config, selected)
    assessor_paths = _assessor_paths(config)
    module = _load_assessor_module(assessor_paths["source"])
    study_root = _path(config, config["study"]["output_root"])
    reports = {}
    for model_name in ("va", "six"):
        model = module.Assessor(
            bundle_path=assessor_paths["bundles"][model_name],
            device=config["assessor"]["device"],
        )
        for set_name in selected:
            output_path = study_root / "assessor_scores" / f"{set_name}_{model_name}.csv"
            _score_image_set(
                assessor=model,
                model_name=model_name,
                bundle_sha256=frozen["bundles"][model_name]["sha256"],
                paths=all_sets[set_name],
                output_path=output_path,
                interval=(
                    float(config["assessor"]["interval"])
                    if model_name == "va"
                    else None
                ),
                ood=model_name == "va",
                batch_size=int(config["assessor"]["batch_size"]),
            )
            reports[f"{set_name}_{model_name}"] = str(output_path)
        del model
    return reports


def _read_scores(config: dict, set_name: str, model: str) -> dict[str, dict[str, float]]:
    path = (
        _path(config, config["study"]["output_root"])
        / "assessor_scores"
        / f"{set_name}_{model}.csv"
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Assessor scores are incomplete: {path}")
    result = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["image"]] = {
                key.lower(): float(value) for key, value in row.items() if key != "image"
            }
    return result


def _stimulus_id(path: str | Path) -> int:
    match = STIMULUS_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse stimulus ID from {path}")
    return int(match.group("stimulus"))


def _subject_id(path: str | Path, prefix: str) -> str:
    for part in Path(path).parts:
        if part.startswith(prefix):
            return part
    raise ValueError(f"Cannot parse subject from {path}")


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(x.std()) < 1e-10 or float(y.std()) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def validate_assessor_preservation(config: dict) -> dict:
    originals = {
        model: _read_scores(config, "nsd_original", model) for model in ("va", "six")
    }
    reconstructions = {
        model: _read_scores(config, "nsd_reconstruction", model)
        for model in ("va", "six")
    }
    original_by_stimulus = {
        model: {_stimulus_id(path): row for path, row in values.items()}
        for model, values in originals.items()
    }
    report = {
        "status": "complete",
        "interpretation": (
            "Held-out NSD uses measured ROI responses and is an external calibration "
            "reference, not a matched healthy control for FOR."
        ),
        "dimensions": {},
    }
    for dimension, model in DIMENSION_MODEL.items():
        original_values = []
        reconstructed_values = []
        ood_values = []
        for path, row in reconstructions[model].items():
            stimulus = _stimulus_id(path)
            original_values.append(original_by_stimulus[model][stimulus][dimension])
            reconstructed_values.append(row[dimension])
            if model == "va":
                ood_values.append(row["ood_percentile"])
        original_array = np.asarray(original_values)
        reconstructed_array = np.asarray(reconstructed_values)
        entry = {
            "images": int(original_array.size),
            "original_reconstruction_r": _correlation(original_array, reconstructed_array),
            "mean_delta": float(np.mean(reconstructed_array - original_array)),
            "mae": float(np.mean(np.abs(reconstructed_array - original_array))),
        }
        if ood_values:
            entry["median_reconstruction_ood_percentile"] = float(np.median(ood_values))
            entry["share_reconstruction_ood_gt_99"] = float(np.mean(np.asarray(ood_values) > 99))
        report["dimensions"][dimension] = entry
    output_path = (
        _path(config, config["study"]["output_root"])
        / "validation"
        / "assessor_preservation.json"
    )
    _write_json_atomic(output_path, report)
    return report


def _approval_inputs(config: dict) -> dict:
    validation_root = _path(config, config["decoder_validation"]["output_dir"])
    study_root = _path(config, config["study"]["output_root"])
    paths = {
        "decoder_split": validation_root / "split_manifest.json",
        "selected_alpha": validation_root / "selected_alpha.json",
        "decoder_validation": validation_root / "validation_summary.json",
        "assessor_preservation": study_root / "validation" / "assessor_preservation.json",
        "nsd_original_va_scores": study_root
        / "assessor_scores"
        / "nsd_original_va.csv",
        "nsd_original_six_scores": study_root
        / "assessor_scores"
        / "nsd_original_six.csv",
        "nsd_reconstruction_va_scores": study_root
        / "assessor_scores"
        / "nsd_reconstruction_va.csv",
        "nsd_reconstruction_six_scores": study_root
        / "assessor_scores"
        / "nsd_reconstruction_six.csv",
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
    }


def approve_validation(config: dict) -> dict:
    approval = {
        "artifact_version": ARTIFACT_VERSION,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "meaning": "User explicitly approved the frozen NSD decoder and assessor validation.",
        "inputs": _approval_inputs(config),
        "frozen_assessor": validate_frozen_assessor(config),
    }
    path = _path(config, config["study"]["output_root"]) / "validation" / "APPROVED.json"
    _write_json_atomic(path, approval)
    return approval


def check_approval(config: dict) -> dict:
    path = _path(config, config["study"]["output_root"]) / "validation" / "APPROVED.json"
    if not path.exists():
        raise RuntimeError("NSD validation has not been explicitly approved.")
    approval = json.loads(path.read_text())
    if approval.get("inputs") != _approval_inputs(config):
        raise RuntimeError("NSD validation changed after approval; review and approve it again.")
    if approval.get("frozen_assessor") != validate_frozen_assessor(config):
        raise RuntimeError("Frozen assessor changed after approval.")
    return {"status": "approved", "approval_file": str(path)}


def _subject_summaries(
    *,
    config: dict,
    reconstruction_set: str,
    original_set: str,
    cohort: str,
    groups: dict[str, str] | None,
) -> tuple[list[dict], list[dict]]:
    original_scores = {
        model: _read_scores(config, original_set, model) for model in ("va", "six")
    }
    reconstruction_scores = {
        model: _read_scores(config, reconstruction_set, model) for model in ("va", "six")
    }
    original_by_stimulus = {
        model: {_stimulus_id(path): row for path, row in values.items()}
        for model, values in original_scores.items()
    }
    image_rows = []
    prefix = "sub-" if cohort == "for" else "subj"
    va_paths = sorted(reconstruction_scores["va"])
    for path in va_paths:
        subject = _subject_id(path, prefix)
        stimulus = _stimulus_id(path)
        group = groups[subject] if groups is not None else "nsd_reference"
        row = {
            "cohort": cohort,
            "subject": subject,
            "group": group,
            "stimulus_id": stimulus,
            "image": path,
            "reconstruction_ood_percentile": reconstruction_scores["va"][path][
                "ood_percentile"
            ],
            "original_ood_percentile": original_by_stimulus["va"][stimulus][
                "ood_percentile"
            ],
        }
        for dimension, model in DIMENSION_MODEL.items():
            reconstruction = reconstruction_scores[model][path][dimension]
            original = original_by_stimulus[model][stimulus][dimension]
            row[f"{dimension}_reconstruction"] = reconstruction
            row[f"{dimension}_original"] = original
            row[f"{dimension}_delta"] = reconstruction - original
        image_rows.append(row)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in image_rows:
        for dimension in DIMENSION_MODEL:
            grouped[(row["subject"], dimension)].append(row)
    summaries = []
    for (subject, dimension), rows in sorted(grouped.items()):
        reconstructed = np.asarray([row[f"{dimension}_reconstruction"] for row in rows])
        original = np.asarray([row[f"{dimension}_original"] for row in rows])
        ood = np.asarray([row["reconstruction_ood_percentile"] for row in rows])
        summaries.append(
            {
                "cohort": rows[0]["cohort"],
                "subject": subject,
                "group": rows[0]["group"],
                "dimension": dimension,
                "n_images": len(rows),
                "mean_reconstruction": float(reconstructed.mean()),
                "mean_original": float(original.mean()),
                "mean_delta": float((reconstructed - original).mean()),
                "mae": float(np.abs(reconstructed - original).mean()),
                "original_reconstruction_r": _correlation(original, reconstructed),
                "median_ood_percentile": float(np.median(ood)),
                "share_ood_gt_99": float(np.mean(ood > 99)),
            }
        )
    return image_rows, summaries


def _contrast(
    first: np.ndarray,
    second: np.ndarray,
    *,
    permutations: int,
    bootstrap_samples: int,
    rng: np.random.RandomState,
) -> dict:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    observed = float(first.mean() - second.mean())
    combined = np.concatenate([first, second])
    exceed = 0
    for _ in range(int(permutations)):
        shuffled = combined[rng.permutation(combined.size)]
        difference = shuffled[: first.size].mean() - shuffled[first.size :].mean()
        exceed += int(abs(difference) >= abs(observed))
    boot = np.empty(int(bootstrap_samples), dtype=float)
    for index in range(int(bootstrap_samples)):
        a = first[rng.randint(0, first.size, size=first.size)]
        b = second[rng.randint(0, second.size, size=second.size)]
        boot[index] = a.mean() - b.mean()
    return {
        "n_first": int(first.size),
        "n_second": int(second.size),
        "mean_first": float(first.mean()),
        "mean_second": float(second.mean()),
        "difference_first_minus_second": observed,
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "permutation_p": float((exceed + 1) / (int(permutations) + 1)),
    }


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_study(config: dict) -> dict:
    check_approval(config)
    groups = load_groups(config)
    for_images, for_summaries = _subject_summaries(
        config=config,
        reconstruction_set="for_reconstruction",
        original_set="for_original",
        cohort="for",
        groups=groups,
    )
    nsd_images, nsd_summaries = _subject_summaries(
        config=config,
        reconstruction_set="nsd_reconstruction",
        original_set="nsd_original",
        cohort="nsd_oof_measured",
        groups=None,
    )
    study_root = _path(config, config["study"]["output_root"])
    analysis_root = study_root / "analysis"
    _write_csv(analysis_root / "image_level_scores.csv", for_images + nsd_images)
    all_summaries = for_summaries + nsd_summaries
    _write_csv(analysis_root / "subject_level_scores.csv", all_summaries)

    grouped_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in all_summaries:
        grouped_values[(row["group"], row["dimension"])].append(float(row["mean_delta"]))
    contrasts = [
        ("depressed_minus_healthy", "depressed", "healthy"),
        ("healthy_minus_nsd_reference", "healthy", "nsd_reference"),
        ("depressed_minus_nsd_reference", "depressed", "nsd_reference"),
    ]
    rng = np.random.RandomState(int(config["study"]["seed"]))
    comparison_rows = []
    dimensions = [
        *config["study"]["primary_dimensions"],
        *config["study"]["exploratory_dimensions"],
    ]
    for contrast_name, first, second in contrasts:
        contrast_rows = []
        for dimension in dimensions:
            result = _contrast(
                np.asarray(grouped_values[(first, dimension)]),
                np.asarray(grouped_values[(second, dimension)]),
                permutations=int(config["study"]["permutations"]),
                bootstrap_samples=int(config["study"]["bootstrap_samples"]),
                rng=rng,
            )
            contrast_rows.append(
                {
                    "contrast": contrast_name,
                    "first_group": first,
                    "second_group": second,
                    "dimension": dimension,
                    "status": "primary" if dimension in config["study"]["primary_dimensions"] else "exploratory",
                    **result,
                    "fdr_q_within_contrast_primary": "",
                }
            )
        primary = [row for row in contrast_rows if row["status"] == "primary"]
        for row, qvalue in zip(
            primary,
            _bh_qvalues([float(row["permutation_p"]) for row in primary]),
            strict=True,
        ):
            row["fdr_q_within_contrast_primary"] = qvalue
        comparison_rows.extend(contrast_rows)
    _write_csv(analysis_root / "group_comparisons.csv", comparison_rows)

    quality_rows = []
    valence_rows = [row for row in all_summaries if row["dimension"] == "valence"]
    for contrast_name, first, second in contrasts:
        for metric in ("median_ood_percentile", "share_ood_gt_99"):
            first_values = np.asarray(
                [float(row[metric]) for row in valence_rows if row["group"] == first]
            )
            second_values = np.asarray(
                [float(row[metric]) for row in valence_rows if row["group"] == second]
            )
            quality_rows.append(
                {
                    "contrast": contrast_name,
                    "first_group": first,
                    "second_group": second,
                    "quality_metric": metric,
                    **_contrast(
                        first_values,
                        second_values,
                        permutations=int(config["study"]["permutations"]),
                        bootstrap_samples=int(config["study"]["bootstrap_samples"]),
                        rng=rng,
                    ),
                }
            )
    _write_csv(analysis_root / "technical_quality_comparisons.csv", quality_rows)
    report = {
        "status": "complete",
        "for_subjects": len(groups),
        "nsd_reference_subjects": len(config["decoder_validation"]["subjects"]),
        "primary_contrast": "depressed_minus_healthy",
        "secondary_external_reference_contrasts": [
            "healthy_minus_nsd_reference",
            "depressed_minus_nsd_reference",
        ],
        "important_caveat": (
            "FOR uses model-predicted responses; NSD reference uses held-out measured task "
            "responses and is not a matched clinical control group."
        ),
        "outputs": {
            "image_level": str(analysis_root / "image_level_scores.csv"),
            "subject_level": str(analysis_root / "subject_level_scores.csv"),
            "comparisons": str(analysis_root / "group_comparisons.csv"),
            "technical_quality": str(
                analysis_root / "technical_quality_comparisons.csv"
            ),
        },
    }
    _write_json_atomic(analysis_root / "analysis_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "check",
            "extract-originals",
            "score",
            "validate-assessors",
            "approve-validation",
            "check-approval",
            "analyze",
        ],
    )
    parser.add_argument("--config", default="config_for_assessor_study.yaml")
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=["for_original", "nsd_original", "for_reconstruction", "nsd_reconstruction"],
    )
    args = parser.parse_args()
    config = load_study_config(args.config)

    if args.command == "check":
        result = check_study(config)
    elif args.command == "extract-originals":
        if not args.sets:
            parser.error("extract-originals requires --sets")
        result = extract_originals(config, args.sets)
    elif args.command == "score":
        if not args.sets:
            parser.error("score requires --sets")
        result = score_images(config, args.sets)
    elif args.command == "validate-assessors":
        result = validate_assessor_preservation(config)
    elif args.command == "approve-validation":
        result = approve_validation(config)
    elif args.command == "check-approval":
        result = check_approval(config)
    else:
        result = analyze_study(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
