"""Read-only readiness checks for the Stage-1 multi-expert workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.nsd_loader import NSDSubjectData
from src.data.region_registry import (
    build_hcp_mmp_region_registry,
    load_subject_region_groups,
)
from src.multiexpert_config import load_multiexpert_config, resolve_data_roots
from src.pipelines.multiexpert_support import (
    build_external_seed_spec,
    validate_subject_data,
)
from src.pipelines.multiexpert_evaluation import validate_repeated_trial_contract
from src.pipelines.predict_multiexpert import (
    _require_gate_for_locked_subject,
    load_multiexpert_model,
)


def check_multiexpert(
    *,
    config_path: str = "config_multiexpert.yaml",
    data_root: str | None = None,
    raw_data_root: str | None = None,
    model_dir: str | None = None,
    gate_path: str | None = None,
    require: str = "train",
) -> dict:
    if require not in {"train", "loso", "predict"}:
        raise ValueError("require must be train, loso, or predict.")
    config = resolve_data_roots(
        load_multiexpert_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    data_root = str(config["data_root"])
    raw_data_root = str(config["raw_data_root"])
    model_dir = str(model_dir or (Path(config["output_root"]) / "model"))
    all_subjects = sorted(
        set(config["subjects"]["train"] + config["subjects"]["locked_test"])
    )
    subjects = {subject: NSDSubjectData(subject, data_root) for subject in all_subjects}
    for subject in all_subjects:
        validate_subject_data(subjects[subject])
    feature_path = Path(data_root) / "features" / "clip_features.npy"
    features = np.load(feature_path, mmap_mode="r")
    max_stimulus = max(
        int(np.max(np.concatenate([
            subjects[subject].train_stim_idx,
            subjects[subject].test_stim_idx,
        ])))
        for subject in all_subjects
    )
    if max_stimulus >= int(features.shape[0]):
        raise ValueError("Processed stimulus IDs exceed the CLIP feature matrix.")

    registry = build_hcp_mmp_region_registry(
        config["subjects"]["train"],
        data_root=data_root,
        raw_data_root=raw_data_root,
        min_voxels_per_subject=int(
            config["regions"]["min_voxels_every_training_subject"]
        ),
    )
    subject_voxels = {}
    for subject in all_subjects:
        groups = load_subject_region_groups(
            registry,
            subject,
            data_root=data_root,
            raw_data_root=raw_data_root,
            require_registered=subject in registry.training_subjects,
        )
        if groups.shape[0] != subjects[subject].num_voxels:
            raise ValueError(f"Subject {subject}: region-map voxel count mismatch.")
        subject_voxels[str(subject)] = int(groups.size)
    seed_spec = build_external_seed_spec(
        config,
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    if len(seed_spec.seed_defs) < int(config["experts"]["n_components"]):
        raise ValueError("External seed bank is smaller than the requested latent space.")

    reliability_missing = []
    reliability_errors = []
    for subject in config["subjects"]["loso"]:
        subject_dir = Path(data_root) / f"subj{int(subject):02d}"
        for name in ("test_fmri_trials.npy", "test_trial_labels.npy"):
            path = subject_dir / name
            if not path.exists():
                reliability_missing.append(str(path))
        if not any(str(subject_dir) in path for path in reliability_missing):
            try:
                validate_repeated_trial_contract(
                    np.load(subject_dir / "test_fmri_trials.npy", mmap_mode="r"),
                    np.load(subject_dir / "test_trial_labels.npy", mmap_mode="r"),
                    expected_voxels=subjects[int(subject)].num_voxels,
                )
            except ValueError as exc:
                reliability_errors.append(f"Subject {int(subject)}: {exc}")
    model_path = Path(model_dir)
    model_present = (model_path / "manifest.json").exists()
    model_manifest = None
    if model_present:
        model_manifest, _, _, _, _ = load_multiexpert_model(
            model_dir=model_path,
            config=config,
        )
    gate_ready = False
    gate_error = None
    if not reliability_missing and not reliability_errors:
        try:
            for subject in config["subjects"]["locked_test"]:
                _require_gate_for_locked_subject(subject, config, gate_path)
            gate_ready = True
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            gate_error = str(exc)

    report = {
        "training_ready": True,
        "loso_ready": not reliability_missing and not reliability_errors,
        "prediction_ready": bool(model_present and gate_ready),
        "config": str(Path(config_path).resolve()),
        "feature_shape": [int(value) for value in features.shape],
        "subject_voxels": subject_voxels,
        "regions_retained": len(registry.regions),
        "regions_including_fallback": registry.n_groups,
        "region_registry_fingerprint": registry.fingerprint,
        "external_seed_count": len(seed_spec.seed_defs),
        "external_seed_fingerprint": seed_spec.fingerprint,
        "missing_loso_reliability_files": reliability_missing,
        "loso_reliability_errors": reliability_errors,
        "model_present": model_present,
        "model_manifest": model_manifest,
        "subject7_gate_passed": gate_ready,
        "subject7_gate_error": gate_error,
    }
    if require == "loso" and not report["loso_ready"]:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    if require == "predict" and not report["prediction_ready"]:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_multiexpert.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--gate-path")
    parser.add_argument("--require", choices=["train", "loso", "predict"], default="train")
    args = parser.parse_args()
    try:
        report = check_multiexpert(
            config_path=args.config,
            data_root=args.data_root,
            raw_data_root=args.raw_data_root,
            model_dir=args.model_dir,
            gate_path=args.gate_path,
            require=args.require,
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2, sort_keys=True))
