"""Zero-shot FOR inference through the frozen voxel contract."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.models.encoding_factory import load_encoder
from src.pipelines.multiexpert_artifacts import file_sha256


logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
N_PARCELS = 400
DEFAULT_MODEL_DIR = "artifacts/voxel_contract_final/seed42/model"
DEFAULT_CONTRACT_ROOT = "data/processed_voxel_contract"
DEFAULT_SELECTION_DIR = (
    "artifacts/schaefer400_multiexpert/prediction_inputs/"
    "random_unseen_500_seed42"
)
DEFAULT_OUTPUT_ROOT = "artifacts/for_inference/seed42"
SUBJECT_RE = re.compile(r"sub-[0-9]{4}")
BUNDLE_FILES = (
    "rest_targets.npy",
    "rest_seeds.npy",
    "seed_available.npy",
    "target_parcel_ids.npy",
    "provenance.json",
)
OUTPUT_FILES = (
    "predicted_responses.npy",
    "alignment_P.npy",
    "alignment_R.npy",
    "connectivity.npy",
    "fingerprint.npy",
    "target_parcel_ids.npy",
    "provenance.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def _subject_labels(contract_root: Path, subjects: Sequence[str] | None) -> list[str]:
    for_root = contract_root / "for"
    if subjects:
        labels = [str(subject) for subject in subjects]
    else:
        labels = sorted(
            path.name
            for path in for_root.glob("sub-*")
            if path.is_dir() and SUBJECT_RE.fullmatch(path.name)
        )
    if not labels:
        raise FileNotFoundError(f"No FOR contract subjects found under {for_root}")
    invalid = [label for label in labels if SUBJECT_RE.fullmatch(label) is None]
    if invalid:
        raise ValueError(f"Invalid FOR subject labels: {invalid}")
    if len(set(labels)) != len(labels):
        raise ValueError("FOR subject labels must be unique.")
    return sorted(labels)


def _model_context(model_dir: Path) -> dict:
    final_dir = model_dir.parent
    manifest_path = final_dir / "manifest.json"
    result_path = final_dir / "final_result.json"
    required_paths = (
        manifest_path,
        result_path,
        model_dir / "builder.npz",
        model_dir / "encoder" / "metadata.json",
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing final-model artifact: {path}")

    manifest = _read_json(manifest_path)
    final_result = _read_json(result_path)
    artifact_fingerprints = final_result.get("artifact_fingerprints")
    if not isinstance(artifact_fingerprints, dict) or not artifact_fingerprints:
        raise ValueError(
            f"Missing artifact_fingerprints in final-model result: {result_path}"
        )
    for relative_path in sorted(artifact_fingerprints):
        artifact_path = final_dir / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing final-model artifact: {artifact_path}")

    result_sha256 = file_sha256(result_path)
    return {
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "artifact_fingerprints": artifact_fingerprints,
        "result_sha256": result_sha256,
        "model_fingerprint": final_result.get(
            "result_payload_fingerprint", result_sha256
        ),
    }


def _selection_context(selection_dir: Path) -> dict:
    manifest_path = selection_dir / "selection_manifest.json"
    features_path = selection_dir / "clip_features.npy"
    stimulus_ids_path = selection_dir / "nsd_stimulus_ids.npy"
    for path in (manifest_path, features_path, stimulus_ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing selection file: {path}")

    manifest = _read_json(manifest_path)
    feature_sha256 = file_sha256(features_path)
    if manifest.get("features_sha256") != feature_sha256:
        raise ValueError(
            f"clip_features.npy sha256 does not match {manifest_path}"
        )
    features = np.load(features_path, mmap_mode="r")
    stimulus_ids = np.load(stimulus_ids_path, mmap_mode="r")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D CLIP features: {features_path}")
    if stimulus_ids.ndim != 1 or int(stimulus_ids.shape[0]) != int(features.shape[0]):
        raise ValueError(
            "nsd_stimulus_ids.npy rows do not match clip_features.npy rows."
        )
    if int(manifest.get("prediction_rows", -1)) != int(features.shape[0]):
        raise ValueError("Selection manifest prediction_rows does not match CLIP features.")
    if int(manifest.get("feature_width", -1)) != int(features.shape[1]):
        raise ValueError("Selection manifest feature_width does not match CLIP features.")

    return {
        "features_path": features_path,
        "stimulus_ids_path": stimulus_ids_path,
        "prediction_rows": int(features.shape[0]),
        "feature_width": int(features.shape[1]),
        "selection_sha256s": {
            "selection_manifest_sha256": file_sha256(manifest_path),
            "clip_features_sha256": feature_sha256,
        },
        "stimulus_ids_sha256": file_sha256(stimulus_ids_path),
    }


def _bundle_context(for_root: Path, subject: str) -> dict:
    subject_dir = for_root / subject
    paths = {name: subject_dir / name for name in BUNDLE_FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing FOR bundle file for {subject}: {path}")

    provenance = _read_json(paths["provenance.json"])
    if provenance.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"FOR contract_version mismatch for {subject}: "
            f"{provenance.get('contract_version')!r} != {CONTRACT_VERSION}"
        )

    rest_targets = np.load(paths["rest_targets.npy"], mmap_mode="r")
    rest_seeds = np.load(paths["rest_seeds.npy"], mmap_mode="r")
    seed_available = np.load(paths["seed_available.npy"], mmap_mode="r")
    target_parcel_ids = np.load(paths["target_parcel_ids.npy"], mmap_mode="r")
    if rest_targets.ndim != 2 or rest_seeds.ndim != 2:
        raise ValueError(f"FOR REST arrays must be 2D for {subject}.")
    if int(rest_targets.shape[0]) != int(rest_seeds.shape[0]):
        raise ValueError(f"FOR REST row mismatch for {subject}.")
    if int(rest_seeds.shape[1]) != N_PARCELS:
        raise ValueError(
            f"FOR seed width must be {N_PARCELS} for {subject}, got {rest_seeds.shape}."
        )
    if seed_available.shape != (N_PARCELS,) or seed_available.dtype != np.bool_:
        raise ValueError(f"seed_available.npy must be bool ({N_PARCELS},) for {subject}.")
    if target_parcel_ids.ndim != 1 or int(target_parcel_ids.size) != int(
        rest_targets.shape[1]
    ):
        raise ValueError(f"FOR target parcel IDs do not match target voxels for {subject}.")
    if int(provenance.get("n_target_voxels", -1)) != int(rest_targets.shape[1]):
        raise ValueError(f"FOR provenance target count mismatch for {subject}.")

    return {
        "subject_dir": subject_dir,
        "paths": paths,
        "bundle_provenance_sha256": file_sha256(paths["provenance.json"]),
        "n_available_seeds": int(np.asarray(seed_available, dtype=bool).sum()),
        "n_target_voxels": int(rest_targets.shape[1]),
    }


def _validate_inputs(
    *,
    model_dir: str | Path,
    contract_root: str | Path,
    selection_dir: str | Path,
    subjects: Sequence[str] | None,
) -> dict:
    model_path = Path(model_dir).expanduser().resolve()
    contract_path = Path(contract_root).expanduser().resolve()
    selection_path = Path(selection_dir).expanduser().resolve()
    labels = _subject_labels(contract_path, subjects)
    return {
        "model_dir": model_path,
        "model": _model_context(model_path),
        "selection": _selection_context(selection_path),
        "subjects": labels,
        "bundles": {
            subject: _bundle_context(contract_path / "for", subject)
            for subject in labels
        },
    }


def check_inputs(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    contract_root: str | Path = DEFAULT_CONTRACT_ROOT,
    selection_dir: str | Path = DEFAULT_SELECTION_DIR,
    subjects: Sequence[str] | None = None,
) -> dict:
    """Read and validate the frozen model, selection, and requested bundles."""
    context = _validate_inputs(
        model_dir=model_dir,
        contract_root=contract_root,
        selection_dir=selection_dir,
        subjects=subjects,
    )
    return {
        "status": "ok",
        "for_subjects": len(context["subjects"]),
        "prediction_rows": context["selection"]["prediction_rows"],
        "feature_width": context["selection"]["feature_width"],
        "n_target_voxels": {
            subject: context["bundles"][subject]["n_target_voxels"]
            for subject in context["subjects"]
        },
        "n_available_seeds": {
            subject: context["bundles"][subject]["n_available_seeds"]
            for subject in context["subjects"]
        },
    }


def _provenance_identity(context: dict, subject: str) -> dict:
    bundle = context["bundles"][subject]
    selection = context["selection"]
    model = context["model"]
    return {
        "subject": subject,
        "model_fingerprints": model["manifest"],
        "model_manifest_sha256": model["manifest_sha256"],
        "model_artifact_fingerprints": model["artifact_fingerprints"],
        "model_result_sha256": model["result_sha256"],
        "bundle_provenance_sha256": bundle["bundle_provenance_sha256"],
        "selection_sha256s": selection["selection_sha256s"],
        "stimulus_ids_sha256": selection["stimulus_ids_sha256"],
        "seed_available": {
            "n_available": bundle["n_available_seeds"],
            "n_total": N_PARCELS,
        },
        "n_target_voxels": bundle["n_target_voxels"],
        "accuracy_computed": False,
        "accuracy_reason": "FOR has no task fMRI",
        "unseen_rationale": "selection excludes all subject 1-6 train/test stimuli",
    }


def _matching_completed_output(output_dir: Path, expected: dict) -> bool:
    if not all((output_dir / name).is_file() for name in OUTPUT_FILES):
        return False
    try:
        existing = _read_json(output_dir / "provenance.json")
    except (FileNotFoundError, ValueError):
        return False
    return all(existing.get(key) == value for key, value in expected.items())


def _require_finite(subject: str, name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{subject}: {name} contains NaN/Inf.")


def _predict_subject(
    *,
    context: dict,
    subject: str,
    output_root: Path,
    builder: SharedSpaceBuilder,
    encoder,
    clip_features: np.ndarray,
    force: bool,
) -> dict:
    output_dir = output_root / subject
    expected_provenance = _provenance_identity(context, subject)
    if not force and _matching_completed_output(output_dir, expected_provenance):
        logger.info("Skipping completed FOR subject with matching provenance: %s", subject)
        return {
            "status": "skipped",
            "n_target_voxels": context["bundles"][subject]["n_target_voxels"],
            "prediction_shape": [
                context["selection"]["prediction_rows"],
                context["bundles"][subject]["n_target_voxels"],
            ],
        }

    started_at = _utc_now()
    paths = context["bundles"][subject]["paths"]
    rest_targets = np.asarray(np.load(paths["rest_targets.npy"]), dtype=np.float32)
    rest_seeds = np.asarray(np.load(paths["rest_seeds.npy"]), dtype=np.float32)
    target_parcel_ids = np.asarray(np.load(paths["target_parcel_ids.npy"]))

    P, R = builder.align_new_subject_zeroshot(
        rest_runs=[rest_targets],
        external_seed_runs=[rest_seeds],
    )
    P = np.asarray(P, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    _require_finite(subject, "alignment_P", P)
    _require_finite(subject, "alignment_R", R)
    n_target_voxels = int(rest_targets.shape[1])
    if P.ndim != 2 or int(P.shape[0]) != n_target_voxels:
        raise ValueError(f"{subject}: alignment_P has invalid shape {P.shape}.")
    if R.shape != (int(P.shape[1]), int(P.shape[1])):
        raise ValueError(f"{subject}: alignment_R has invalid shape {R.shape}.")

    predicted = np.asarray(
        encoder.predict_voxels(clip_features, P, R),
        dtype=np.float32,
    )
    expected_shape = (int(clip_features.shape[0]), n_target_voxels)
    if predicted.shape != expected_shape:
        raise ValueError(
            f"{subject}: predicted responses have shape {predicted.shape}, "
            f"expected {expected_shape}."
        )
    _require_finite(subject, "predicted_responses", predicted)

    connectivity = compute_rest_connectivity(
        [rest_targets],
        seed_runs=[rest_seeds],
        ensemble=builder.ensemble_method,
    )
    connectivity = np.asarray(connectivity, dtype=np.float32)
    fingerprint = np.asarray(connectivity @ P @ R, dtype=np.float32)
    _require_finite(subject, "connectivity", connectivity)
    _require_finite(subject, "fingerprint", fingerprint)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_npy_atomic(output_dir / "predicted_responses.npy", predicted)
    _write_npy_atomic(output_dir / "alignment_P.npy", P)
    _write_npy_atomic(output_dir / "alignment_R.npy", R)
    _write_npy_atomic(output_dir / "connectivity.npy", connectivity)
    _write_npy_atomic(output_dir / "fingerprint.npy", fingerprint)
    _write_npy_atomic(output_dir / "target_parcel_ids.npy", target_parcel_ids)
    provenance = {
        **expected_provenance,
        "timestamps": {
            "started_at": started_at,
            "finished_at": _utc_now(),
        },
    }
    _write_json_atomic(output_dir / "provenance.json", provenance)
    logger.info("Saved FOR voxel predictions for %s to %s", subject, output_dir)
    return {
        "status": "completed",
        "n_target_voxels": n_target_voxels,
        "prediction_shape": list(predicted.shape),
    }


def predict_for_subjects(
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    contract_root: str | Path = DEFAULT_CONTRACT_ROOT,
    selection_dir: str | Path = DEFAULT_SELECTION_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    subjects: Sequence[str] | None = None,
    device: str = "cuda",
    force: bool = False,
) -> dict:
    """Predict the pinned selection for every requested FOR contract subject."""
    context = _validate_inputs(
        model_dir=model_dir,
        contract_root=contract_root,
        selection_dir=selection_dir,
        subjects=subjects,
    )
    builder = SharedSpaceBuilder.load(str(context["model_dir"]))
    encoder = load_encoder(str(context["model_dir"]))
    encoder.config.device = str(device)
    clip_features = np.asarray(
        np.load(context["selection"]["features_path"]),
        dtype=np.float32,
    )
    if int(clip_features.shape[1]) != int(encoder.input_dim):
        raise ValueError(
            f"CLIP feature width {clip_features.shape[1]} does not match "
            f"encoder input_dim {encoder.input_dim}."
        )

    destination = Path(output_root).expanduser().resolve()
    results = {
        subject: _predict_subject(
            context=context,
            subject=subject,
            output_root=destination,
            builder=builder,
            encoder=encoder,
            clip_features=clip_features,
            force=force,
        )
        for subject in context["subjects"]
    }
    destination.mkdir(parents=True, exist_ok=True)
    batch_manifest = {
        "schema_version": 1,
        "subjects_completed": list(context["subjects"]),
        "n_target_voxels": {
            subject: results[subject]["n_target_voxels"]
            for subject in context["subjects"]
        },
        "prediction_shape": {
            subject: results[subject]["prediction_shape"]
            for subject in context["subjects"]
        },
        "model_fingerprint": context["model"]["model_fingerprint"],
        "selection_sha256s": {
            **context["selection"]["selection_sha256s"],
            "stimulus_ids_sha256": context["selection"]["stimulus_ids_sha256"],
        },
    }
    _write_json_atomic(destination / "batch_manifest.json", batch_manifest)
    return {
        "status": "ok",
        "output_root": str(destination),
        "subjects": results,
        "batch_manifest": batch_manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, choices=("check", "predict", "all"))
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--contract-root", default=DEFAULT_CONTRACT_ROOT)
    parser.add_argument("--selection-dir", default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    common = {
        "model_dir": args.model_dir,
        "contract_root": args.contract_root,
        "selection_dir": args.selection_dir,
        "subjects": args.subjects,
    }
    if args.command in {"check", "all"}:
        print(json.dumps(check_inputs(**common), indent=2, sort_keys=True))
    if args.command in {"predict", "all"}:
        result = predict_for_subjects(
            **common,
            output_root=args.output_root,
            device=args.device,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
