"""Predict subject-level voxel responses for training stimuli."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.data.nsd_loader import NSDFeatures, NSDSubjectData
from src.data.shared_paths import default_raw_data_root
from src.models.encoding_factory import load_encoder
from src.pipelines.eval_split import fixed_eval_indices
from src.pipelines.predict_subject import (
    _load_external_seed_runs,
    _resolve_fewshot_support_candidates,
    _sample_support_shots,
    _validate_feature_indices,
    _validate_subject_row_contract,
)

logger = logging.getLogger(__name__)


def _train_stim_idx_sha256(train_stim_idx: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(train_stim_idx, dtype=np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def predict_train_responses(
    mode: str,
    test_sub: int = 7,
    config_path: str = "config.yaml",
    model_dir: str = "artifacts/model",
    data_root: str = "data/processed",
    raw_data_root: str = default_raw_data_root(),
    predictions_dir: str = "artifacts/predictions/subj07",
    n_shots: int = 100,
    seed: int = 42,
) -> np.ndarray:
    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError(f"Unknown prediction mode: {mode!r}.")

    config = load_config(config_path)
    feature_type = str(config["features"]["type"])

    builder = SharedSpaceBuilder.load(model_dir)
    encoder = load_encoder(model_dir)

    subject = NSDSubjectData(test_sub, data_root)
    _validate_subject_row_contract(subject)
    features = NSDFeatures(os.path.join(data_root, "features"))
    _validate_feature_indices(subject, features, feature_type)

    external_seed_runs = _load_external_seed_runs(
        test_subj=subject,
        test_sub=test_sub,
        model_dir=model_dir,
        data_root=data_root,
        raw_data_root=raw_data_root,
    )

    actual_shots = 0
    if mode == "zero_shot":
        P, R = builder.align_new_subject_zeroshot(
            rest_runs=subject.rest_runs,
            external_seed_runs=external_seed_runs,
        )
    else:
        evaluation_cfg = config["evaluation"]
        support_candidates, support_template_candidates, support_strategy = (
            _resolve_fewshot_support_candidates(
                test_stim_idx=subject.test_stim_idx,
                model_dir=model_dir,
            )
        )
        logger.info(
            "Few-shot support pool (%s): %d candidate rows before eval exclusion.",
            support_strategy,
            int(support_candidates.size),
        )
        eval_indices = fixed_eval_indices(
            n_shared=int(len(subject.test_stim_idx)),
            eval_size=int(evaluation_cfg["fixed_eval_size"]),
            seed=int(evaluation_cfg["eval_split_seed"]),
        )
        support_mask = ~np.isin(support_candidates, eval_indices, assume_unique=False)
        support_pool_rows = support_candidates[support_mask]
        support_pool_template_rows = support_template_candidates[support_mask]
        if support_pool_rows.size == 0:
            raise ValueError(
                "No few-shot rows remain in the shared-support pool after applying fixed eval split."
            )
        shot_indices, template_shot_indices, actual_shots = _sample_support_shots(
            support_rows=support_pool_rows,
            template_rows=support_pool_template_rows,
            n_shots=n_shots,
            seed=seed,
        )
        shared_fmri = np.array(subject.test_fmri, dtype=np.float32)[shot_indices]
        P, R = builder.align_new_subject_fewshot(
            rest_runs=subject.rest_runs,
            task_fmri_shared=shared_fmri,
            shot_indices=template_shot_indices,
            external_seed_runs=external_seed_runs,
        )

    X_train = features.get_features(subject.train_stim_idx, feature_type)
    predictions = np.asarray(
        encoder.predict_voxels(X_train, P, R),
        dtype=np.float32,
    )
    expected_rows = int(len(subject.train_stim_idx))
    if predictions.ndim != 2 or int(predictions.shape[0]) != expected_rows:
        raise ValueError(
            f"Training prediction shape mismatch: got {predictions.shape}, "
            f"expected ({expected_rows}, V)."
        )

    output_dir = Path(predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{mode}_sub{test_sub}_train_pred.npy"
    sidecar_path = output_dir / f"{mode}_sub{test_sub}_train_pred.json"
    np.save(prediction_path, predictions)
    metadata = {
        "mode": mode,
        "n_shots": int(actual_shots),
        "seed": int(seed),
        "n_rows": int(predictions.shape[0]),
        "n_voxels": int(predictions.shape[1]),
        "train_stim_idx_sha256": _train_stim_idx_sha256(subject.train_stim_idx),
    }
    with open(sidecar_path, "w") as sidecar_file:
        json.dump(metadata, sidecar_file, indent=2)
    logger.info("Saved %s training responses to %s", mode, prediction_path)
    return predictions


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Predict subject voxel responses for training stimuli."
    )
    parser.add_argument(
        "--mode",
        choices=("zero_shot", "few_shot"),
        required=True,
    )
    parser.add_argument("--test-sub", type=int, default=7)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--raw-data-root", default=default_raw_data_root())
    parser.add_argument(
        "--predictions-dir",
        default="artifacts/predictions/subj07",
    )
    parser.add_argument("--n-shots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predict_train_responses(
        mode=str(args.mode),
        test_sub=int(args.test_sub),
        config_path=str(args.config),
        model_dir=str(args.model_dir),
        data_root=str(args.data_root),
        raw_data_root=str(args.raw_data_root),
        predictions_dir=str(args.predictions_dir),
        n_shots=int(args.n_shots),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
