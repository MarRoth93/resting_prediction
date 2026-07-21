"""REST-only Schaefer-400 transfer from a trained NSD model to one FOR subject."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.data.schaefer400 import (
    N_PARCELS,
    PARCEL_IDS,
    load_for_schaefer400_subject,
)
from src.pipelines.multiexpert_artifacts import file_sha256
from src.pipelines.multiexpert_support import align_new_subject
from src.pipelines.schaefer400_support import load_schaefer400_model
from src.schaefer400_config import load_schaefer400_config, resolve_schaefer400_roots


logger = logging.getLogger(__name__)


def _resolve_for_subject(subject: str | Path, for_data_root: str | Path) -> Path:
    candidate = Path(subject).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    under_root = Path(for_data_root) / str(subject)
    if under_root.is_dir():
        return under_root.resolve()
    raise FileNotFoundError(
        f"FOR subject must be a directory or a name under {for_data_root}: {subject}"
    )


def predict_for_schaefer400(
    *,
    for_subject: str | Path,
    features_path: str | Path,
    config_path: str = "config_schaefer400.yaml",
    model_dir: str | Path | None = None,
    for_data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Predict parcel responses for supplied CLIP rows; deliberately no accuracy."""
    config = resolve_schaefer400_roots(
        load_schaefer400_config(config_path),
        for_data_root=for_data_root,
    )
    model_dir = Path(model_dir or (Path(config["output_root"]) / "model"))
    manifest, contract, experts, encoder = load_schaefer400_model(
        model_dir=model_dir,
        config=config,
    )
    subject_dir = _resolve_for_subject(for_subject, config["for_data_root"])
    inference_cfg = config["for_inference"]
    subject = load_for_schaefer400_subject(
        subject_dir,
        timeseries_file=str(inference_cfg["timeseries_file"]),
        summary_file=str(inference_cfg["summary_file"]),
    )
    if int(subject.available_parcels.sum()) <= int(max(manifest["expert_dims"].values())):
        raise ValueError(
            "FOR subject has too few usable parcels for the trained latent dimension."
        )

    features_path = Path(features_path).expanduser().resolve()
    if not features_path.exists():
        raise FileNotFoundError(f"Missing FOR stimulus features: {features_path}")
    features = np.load(features_path, mmap_mode="r")
    if features.ndim != 2 or int(features.shape[1]) != int(manifest["input_dim"]):
        raise ValueError(
            f"FOR features must be N x {manifest['input_dim']}, got {features.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("FOR stimulus features contain NaN/Inf.")

    transforms = align_new_subject(
        experts,
        subject=subject,
        external_seed_runs=subject.seed_runs,
        seed_manifest_fingerprint=contract.seed_fingerprint,
    )
    prediction = encoder.predict_subject(
        np.asarray(features, dtype=np.float32),
        transforms=transforms,
        voxel_groups=subject.parcel_groups,
    )
    equal_prediction = encoder.predict_subject(
        np.asarray(features, dtype=np.float32),
        transforms=transforms,
        voxel_groups=subject.parcel_groups,
        equal_weights=True,
    )

    destination = Path(
        output_dir
        or (Path(config["output_root"]) / "for_predictions" / subject.subject_label)
    )
    destination.mkdir(parents=True, exist_ok=True)
    fused_full = subject.expand_available(prediction["fused"])
    equal_full = subject.expand_available(equal_prediction["fused"])
    np.save(destination / "learned_fusion.npy", fused_full)
    np.save(destination / "equal_fusion_weights.npy", equal_full)
    for name, values in prediction["per_expert"].items():
        np.save(destination / f"expert_{name}.npy", subject.expand_available(values))
    np.save(destination / "regional_weights.npy", prediction["regional_weights"])
    np.save(destination / "available_parcels.npy", subject.available_parcels)
    np.save(destination / "parcel_ids.npy", PARCEL_IDS)
    for name, transform in transforms.items():
        transform.save(destination / f"transform_{name}.npz")

    provenance = {
        "mode": "zero_shot_rest_only",
        "accuracy_computed": False,
        "accuracy_reason": "FOR export contains no matching task-fMRI ground truth",
        "subject": subject.subject_label,
        "model_dir": str(model_dir.resolve()),
        "model_manifest_sha256": file_sha256(model_dir / "manifest.json"),
        "features_file": str(features_path),
        "features_sha256": file_sha256(features_path),
        "prediction_rows": int(features.shape[0]),
        "input_feature_width": int(features.shape[1]),
        "atlas_name": config["atlas"]["name"],
        "n_parcels": N_PARCELS,
        "available_parcels": int(subject.available_parcels.sum()),
        "missing_parcel_ids": PARCEL_IDS[~subject.available_parcels].astype(int).tolist(),
        "missing_output_value": "nan",
        "expert_order": list(manifest["expert_order"]),
        "expert_dims": manifest["expert_dims"],
        "regional_weight_shape": [
            int(value) for value in prediction["regional_weights"].shape
        ],
        "for_input": subject.provenance,
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Saved FOR zero-shot predictions to %s", destination)
    return {
        "destination": str(destination),
        "prediction_shape": tuple(fused_full.shape),
        "available_parcels": int(subject.available_parcels.sum()),
        "accuracy_computed": False,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--for-subject", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", default="config_schaefer400.yaml")
    parser.add_argument("--model-dir")
    parser.add_argument("--for-data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    predict_for_schaefer400(
        for_subject=args.for_subject,
        features_path=args.features,
        config_path=args.config,
        model_dir=args.model_dir,
        for_data_root=args.for_data_root,
        output_dir=args.output_dir,
    )
