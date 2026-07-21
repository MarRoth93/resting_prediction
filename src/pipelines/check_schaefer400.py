"""Preflight the separate Schaefer-400 NSD training and FOR transfer path."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.prepare_task_data import discover_sessions
from src.data.schaefer400 import (
    SchaeferNSDSubjectData,
    audit_for_schaefer400_dataset,
    validate_schaefer_nsd_subject,
)
from src.pipelines.schaefer400_support import load_schaefer400_model
from src.schaefer400_config import load_schaefer400_config, resolve_schaefer400_roots


def _raw_subject_check(subject: int, config: dict) -> dict:
    raw = Path(config["raw_data_root"])
    beta_dir = (
        raw
        / "nsddata_betas"
        / "ppdata"
        / f"subj{subject:02d}"
        / "func1pt8mm"
        / "betas_fithrf_GLMdenoise_RR"
    )
    timeseries_dir = (
        raw
        / "nsddata_timeseries"
        / "ppdata"
        / f"subj{subject:02d}"
        / "func1pt8mm"
        / "timeseries"
    )
    reference = (
        raw
        / "nsddata"
        / "ppdata"
        / f"subj{subject:02d}"
        / "func1pt8mm"
        / "roi"
        / "nsdgeneral.nii.gz"
    )
    if not reference.exists():
        raise FileNotFoundError(f"Missing NSD reference: {reference}")
    sessions = discover_sessions(str(beta_dir))
    rest_files = sorted(timeseries_dir.glob("*.nii.gz"))
    if len(rest_files) < 2:
        raise ValueError(f"Subject {subject} has fewer than two raw REST runs.")
    return {
        "subject": subject,
        "beta_sessions": len(sessions),
        "raw_rest_files": len(rest_files),
        "reference": str(reference.resolve()),
    }


def check_schaefer400(
    *,
    config_path: str = "config_schaefer400.yaml",
    data_root: str | None = None,
    raw_data_root: str | None = None,
    for_data_root: str | None = None,
    model_dir: str | None = None,
    require: str = "raw",
) -> dict:
    if require not in {"for", "raw", "train", "predict"}:
        raise ValueError("require must be for, raw, train, or predict.")
    config = resolve_schaefer400_roots(
        load_schaefer400_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
        for_data_root=for_data_root,
    )
    result = {
        "status": "ok",
        "require": require,
        "config": str(Path(config_path).resolve()),
        "atlas_name": config["atlas"]["name"],
        "n_parcels": int(config["atlas"]["n_parcels"]),
        "for": audit_for_schaefer400_dataset(config["for_data_root"]),
    }
    if require == "for":
        return result
    design = (
        Path(config["raw_data_root"])
        / "nsddata"
        / "experiments"
        / "nsd"
        / "nsd_expdesign.mat"
    )
    if not design.exists():
        raise FileNotFoundError(f"Missing NSD experiment design: {design}")
    result["raw_nsd"] = [
        _raw_subject_check(int(subject), config)
        for subject in config["subjects"]["train"]
    ]
    if require == "raw":
        return result
    features = Path(config["features"]["path"])
    if not features.exists():
        raise FileNotFoundError(f"Missing CLIP features: {features}")
    prepared_rows = []
    for subject in config["subjects"]["train"]:
        loaded = SchaeferNSDSubjectData(int(subject), config["data_root"])
        validate_schaefer_nsd_subject(loaded)
        prepared_rows.append(
            {
                "subject": int(subject),
                "train_rows": int(loaded.train_fmri.shape[0]),
                "test_rows": int(loaded.test_fmri.shape[0]),
                "rest_runs": len(loaded.rest_runs),
                "rest_trs": sum(int(run.shape[0]) for run in loaded.rest_runs),
                "parcels": int(loaded.train_fmri.shape[1]),
            }
        )
    result["prepared_nsd"] = prepared_rows
    result["features"] = {
        "path": str(features.resolve()),
        "size_bytes": int(features.stat().st_size),
    }
    if require == "train":
        return result
    destination = Path(model_dir or (Path(config["output_root"]) / "model"))
    manifest, contract, _, encoder = load_schaefer400_model(
        model_dir=destination,
        config=config,
    )
    result["model"] = {
        "path": str(destination.resolve()),
        "train_subjects": list(contract.training_subjects),
        "expert_dims": manifest["expert_dims"],
        "fusion_groups": int(encoder.network.num_regions),
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_schaefer400.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--for-data-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--require", choices=["for", "raw", "train", "predict"], default="raw")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    report = check_schaefer400(
        config_path=args.config,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        for_data_root=args.for_data_root,
        model_dir=args.model_dir,
        require=args.require,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)
